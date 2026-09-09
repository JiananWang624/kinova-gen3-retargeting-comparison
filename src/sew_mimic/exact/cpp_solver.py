"""Production stateful Exact-SEW solver backed by the compiled event core.

The extension owns the fixed-slot geometry and event refinement.  Python owns
the task-frame conversion, joint representation, and MuJoCo-derived pinch-site
acceptance check; that final physical check is deliberately authoritative.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..angles import wrap_to_pi
from ..common import ExactSewTarget, SolverDiagnostics, SolverResult, SolverStatus, joint_limit_margin
from ..kinematics import Gen3Kinematics
from ..sew import Gen3StereoSewGeometry, StereoSew, StereoSewSingularityError
from .acceptance import ORIENTATION_ACCEPTANCE_RAD, POSITION_ACCEPTANCE_M, SEW_ACCEPTANCE_RAD
from .residuals import robot_exact_sew_residuals


_ROTATION_TOL = 1e-10


@dataclass(frozen=True)
class NativeStereoSewTarget:
    position: np.ndarray
    rotation_07: np.ndarray
    psi: float


def to_native_stereo_sew_target(target: ExactSewTarget, robot: Gen3Kinematics,
                                geometry: Gen3StereoSewGeometry) -> NativeStereoSewTarget:
    """Convert the aligned pinch target to the native PoE terminal frame."""
    if not isinstance(target, ExactSewTarget):
        raise ValueError("target must be an ExactSewTarget")
    position = np.asarray(target.position, dtype=float)
    rotation = np.asarray(target.rotation, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("target position must be finite with shape (3,)")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("target rotation must be finite with shape (3, 3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_TOL, rtol=0.0) or not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_TOL, rtol=0.0):
        raise ValueError("target rotation must be a proper rotation")
    return NativeStereoSewTarget(position.copy(), rotation @ robot.R_robot_align.T @ geometry.R_7T.T,
                                 wrap_to_pi(float(target.psi)))


def _represent(q: np.ndarray, robot: Gen3Kinematics) -> tuple[np.ndarray, bool, float]:
    """Choose a deterministic in-limit 2*pi representative for each joint."""
    answer = np.asarray(q, dtype=float).copy()
    valid = True
    for index, (lower, upper) in enumerate(robot.joint_limits):
        angle = wrap_to_pi(answer[index])
        if not math.isfinite(lower) and not math.isfinite(upper):
            answer[index] = angle
            continue
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("Gen3 joint limits must be both finite or both unlimited")
        integers = range(math.ceil((lower - angle - 1e-12) / (2.0 * math.pi)),
                         math.floor((upper - angle + 1e-12) / (2.0 * math.pi)) + 1)
        equivalents = [angle + 2.0 * math.pi * integer for integer in integers]
        if equivalents:
            answer[index] = min(equivalents, key=lambda value: (abs(value - angle), value))
        else:
            answer[index] = angle
            valid = False
    margin = joint_limit_margin(answer, robot)
    return answer, valid and margin >= -1e-12, margin


@dataclass
class _State:
    previous_q: np.ndarray | None = None
    previous_branch_slot: int | None = None
    previous_wrist_angle: float | None = None
    previous_previous_wrist_angle: float | None = None
    previous_branch_id: str | None = None
    valid: bool = False

    def reset(self) -> None:
        self.previous_q = self.previous_branch_slot = self.previous_wrist_angle = None
        self.previous_previous_wrist_angle = self.previous_branch_id = None
        self.valid = False


_MAX_EVENT_LIMIT = 1_000_000


@dataclass(frozen=True)
class ExactSewConfig:
    """Controls actually consumed by the compiled trajectory solver."""

    radii_rad: tuple[float, ...] = (0.002, 0.004, 0.01, 0.04)
    local_partitions_min: int = 8
    global_partitions: int = 64
    maximum_event_evaluations: int = 50_000
    maximum_wrapped_joint_step_rad: float = 0.5

    def __post_init__(self) -> None:
        radii = tuple(float(value) for value in self.radii_rad)
        object.__setattr__(self, "radii_rad", radii)
        if not radii or tuple(sorted(radii)) != radii or any(not math.isfinite(x) or x <= 0 for x in radii):
            raise ValueError("radii_rad must contain ascending positive finite values")
        if (
            self.local_partitions_min < 1
            or self.global_partitions < 1
            or self.maximum_event_evaluations < 1
        ):
            raise ValueError("event limits must be positive")
        if any(
            value > _MAX_EVENT_LIMIT
            for value in (
                self.local_partitions_min,
                self.global_partitions,
                self.maximum_event_evaluations,
            )
        ):
            raise ValueError("event limits exceed the supported hard maximum")
        if (
            self.local_partitions_min >= self.maximum_event_evaluations
            or self.global_partitions >= self.maximum_event_evaluations
        ):
            raise ValueError("partitions require partitions + 1 <= maximum_event_evaluations")
        if not math.isfinite(self.maximum_wrapped_joint_step_rad) or self.maximum_wrapped_joint_step_rad <= 0:
            raise ValueError("maximum_wrapped_joint_step_rad must be positive")


class ExactSewSolver:
    """Stateful local continuation with deterministic compiled global recovery."""

    def __init__(
        self,
        robot: Gen3Kinematics,
        geometry: Gen3StereoSewGeometry,
        stereo: StereoSew,
        *,
        config: ExactSewConfig = ExactSewConfig(),
    ) -> None:
        if not isinstance(config, ExactSewConfig):
            raise ValueError("config must be ExactSewConfig")
        self.robot, self.geometry, self.stereo, self.config = robot, geometry, stereo, config
        self.state = _State()

    def reset(self) -> None:
        self.state.reset()

    @staticmethod
    def _failure(status: SolverStatus, started: float, metadata: dict[str, Any], message: str) -> SolverResult:
        return SolverResult("exact_sew", status, None, SolverDiagnostics(
            solve_time_ms=1000.0 * (time.perf_counter() - started), metadata=metadata), message)

    def _update(self, result: SolverResult) -> None:
        if result.status is not SolverStatus.SUCCESS_EXACT or result.q is None:
            return
        metadata = result.diagnostics.metadata
        angle, slot = metadata.get("wrist_search_angle"), metadata.get("search_branch")
        if not (isinstance(angle, (int, float)) and math.isfinite(angle) and isinstance(slot, int)):
            return
        self.state.previous_previous_wrist_angle = self.state.previous_wrist_angle
        self.state.previous_wrist_angle = float(angle)
        self.state.previous_branch_slot = slot
        self.state.previous_q = result.q.copy()
        self.state.previous_branch_id = result.diagnostics.branch_id
        self.state.valid = True

    def _fallback(self, target: ExactSewTarget, started: float, metadata: dict[str, Any]) -> SolverResult:
        """C++ global event recovery; Python only applies authoritative FK."""
        from . import _exact_sew_core as core

        native = to_native_stereo_sew_target(target, self.robot, self.geometry)
        wrist = native.position - native.rotation_07 @ self.geometry.P[:, 7]
        p17 = wrist - self.geometry.P[:, 0]
        plane_normal = self.stereo.inverse(self.geometry.P[:, 0], wrist, native.psi).plane_normal
        physical_candidates: list[tuple[np.ndarray, float, int, float, Any, float, bool, int]] = []
        roots = list(core.event_aware_roots(
            self.geometry.H, self.geometry.P, p17, plane_normal, native.rotation_07,
            -math.pi, 0.0, self.config.global_partitions, self.config.maximum_event_evaluations,
        ))
        rejected_joint = rejected_authoritative = 0
        root_ordinals: dict[int, int] = {}
        for root in roots:
            q, valid, margin = _represent(np.asarray(root["q"], dtype=float), self.robot)
            residual = robot_exact_sew_residuals(q, target, self.robot, self.geometry, self.stereo)
            if (residual.position_error_m < POSITION_ACCEPTANCE_M
                    and residual.orientation_error_rad < ORIENTATION_ACCEPTANCE_RAD
                    and residual.sew_error_rad is not None and residual.sew_error_rad < SEW_ACCEPTANCE_RAD):
                slot = int(root["slot"])
                ordinal = root_ordinals.get(slot, 0)
                root_ordinals[slot] = ordinal + 1
                physical_candidates.append((
                    q, float(root["angle"]), slot, float(root["residual"]),
                    residual, margin, valid, ordinal,
                ))
                if not valid:
                    rejected_joint += 1
            else:
                rejected_authoritative += 1
        accepted = [candidate for candidate in physical_candidates if candidate[6]]
        if not accepted:
            status = SolverStatus.NO_VALID_BRANCH if not roots else (
                SolverStatus.JOINT_LIMIT
                if rejected_joint == len(physical_candidates) and physical_candidates
                else SolverStatus.NUMERICAL_FAILURE
            )
            return self._failure(status, started, {**metadata, "fallback_used": True,
                "cpp_root_count": len(roots), "cpp_joint_rejected_count": rejected_joint,
                "cpp_authoritative_rejected_count": rejected_authoritative,
                "cpp_physical_candidate_count": len(physical_candidates)}, "no selectable exact branch")
        accepted.sort(key=lambda item: (item[1], item[2], tuple(item[0].tolist())))
        if self.state.previous_q is None:
            chosen_index, chosen = min(enumerate(accepted), key=lambda item: (-item[1][5],
                (item[1][4].position_error_m / POSITION_ACCEPTANCE_M) ** 2 + (item[1][4].orientation_error_rad / ORIENTATION_ACCEPTANCE_RAD) ** 2 + (item[1][4].sew_error_rad / SEW_ACCEPTANCE_RAD) ** 2, item[0]))
        else:
            chosen_index, chosen = min(enumerate(accepted), key=lambda item: (float(np.sum(np.arctan2(np.sin(item[1][0] - self.state.previous_q), np.cos(item[1][0] - self.state.previous_q)) ** 2)), -item[1][5],
                (item[1][4].position_error_m / POSITION_ACCEPTANCE_M) ** 2 + (item[1][4].orientation_error_rad / ORIENTATION_ACCEPTANCE_RAD) ** 2 + (item[1][4].sew_error_rad / SEW_ACCEPTANCE_RAD) ** 2, item[0]))
        q, angle, slot, root_residual, residual, margin, _valid, root_ordinal = chosen
        merged = {**metadata, "fallback_used": True, "cpp_global_candidate_count": len(accepted),
                  "cpp_root_count": len(roots), "cpp_joint_rejected_count": rejected_joint,
                  "cpp_authoritative_rejected_count": rejected_authoritative, "candidate_index": chosen_index,
                  "cpp_physical_candidate_count": len(physical_candidates),
                  "wrist_search_angle": angle, "search_branch": slot, "cpp_root_residual": root_residual}
        result = SolverResult("exact_sew", SolverStatus.SUCCESS_EXACT, q, SolverDiagnostics(
            position_error_m=residual.position_error_m, orientation_error_rad=residual.orientation_error_rad,
            sew_error_rad=residual.sew_error_rad, joint_limit_margin_rad=margin,
            solve_time_ms=1000.0 * (time.perf_counter() - started),
            branch_id=f"r2r2r2r2r:slot={slot}:wrist_root={root_ordinal}", metadata=merged))
        self._update(result)
        return result

    def solve(self, target: ExactSewTarget) -> SolverResult:
        started = time.perf_counter()
        predicted = self.state.previous_wrist_angle
        if predicted is not None and self.state.previous_previous_wrist_angle is not None:
            predicted += predicted - self.state.previous_previous_wrist_angle
        metadata: dict[str, Any] = {
            "fast_path_used": False, "fast_path_success": False,
            "fallback_used": False, "local_branch_slot": self.state.previous_branch_slot,
            "predicted_wrist_angle": predicted, "recovered_wrist_angle": None,
        }
        if not self.state.valid or predicted is None or self.state.previous_branch_slot is None:
            return self._fallback_or_failure(target, started, metadata)
        try:
            from . import _exact_sew_core as core

            native = to_native_stereo_sew_target(target, self.robot, self.geometry)
            wrist = native.position - native.rotation_07 @ self.geometry.P[:, 7]
            p17 = wrist - self.geometry.P[:, 0]
            plane_normal = self.stereo.inverse(self.geometry.P[:, 0], wrist, native.psi).plane_normal
            metadata["fast_path_used"] = True
            roots: list[dict[str, Any]] = []
            for radius in self.config.radii_rad:
                left, right = max(-math.pi, predicted - radius), min(0.0, predicted + radius)
                if right <= left:
                    continue
                # A narrow q45 island may be entirely between fixed alignment
                # samples.  Reuse the same C++ three-level event certification
                # as global recovery, restricted to this predicted window.
                roots = [root for root in core.event_aware_roots(
                    self.geometry.H, self.geometry.P, p17, plane_normal, native.rotation_07,
                    left, right, max(self.config.local_partitions_min, math.ceil(self.config.global_partitions * (right - left) / math.pi)), self.config.maximum_event_evaluations,
                ) if int(root["slot"]) == self.state.previous_branch_slot]
                if len(roots) == 1:
                    break
                roots = []
            if len(roots) != 1:
                metadata["cpp_local_root_count"] = len(roots)
                return self._fallback_or_failure(target, started, metadata)
            root = roots[0]
            q, valid, margin = _represent(np.asarray(root["q"], dtype=float), self.robot)
            residual = robot_exact_sew_residuals(q, target, self.robot, self.geometry, self.stereo)
            step = np.arctan2(np.sin(q - self.state.previous_q), np.cos(q - self.state.previous_q))
            if (not valid or residual.position_error_m >= POSITION_ACCEPTANCE_M
                    or residual.orientation_error_rad >= ORIENTATION_ACCEPTANCE_RAD
                    or residual.sew_error_rad is None or residual.sew_error_rad >= SEW_ACCEPTANCE_RAD
                    or float(np.linalg.norm(step)) > self.config.maximum_wrapped_joint_step_rad):
                metadata["cpp_authoritative_rejected"] = True
                return self._fallback_or_failure(target, started, metadata)
            angle = float(root["angle"])
            branch_id = f"cpp_continuation:{self.state.previous_branch_id or 'slot=' + str(self.state.previous_branch_slot)}"
            metadata.update(fast_path_success=True, wrist_search_angle=angle,
                            recovered_wrist_angle=angle, search_branch=self.state.previous_branch_slot,
                            cpp_root_residual=float(root["residual"]), cpp_local_root_count=1)
            result = SolverResult("exact_sew", SolverStatus.SUCCESS_EXACT, q, SolverDiagnostics(
                position_error_m=residual.position_error_m, orientation_error_rad=residual.orientation_error_rad,
                sew_error_rad=residual.sew_error_rad, joint_limit_margin_rad=margin,
                solve_time_ms=1000.0 * (time.perf_counter() - started), branch_id=branch_id,
                metadata=metadata))
            self._update(result)
            return result
        except StereoSewSingularityError as error:
            return self._failure(SolverStatus.SEW_SINGULAR, started, metadata, str(error))
        except (ValueError, TypeError) as error:
            return self._failure(SolverStatus.INVALID_INPUT, started, metadata, str(error))
        except Exception as error:
            metadata["cpp_exception_type"] = type(error).__name__
            return self._fallback_or_failure(target, started, metadata)

    def _fallback_or_failure(
        self, target: ExactSewTarget, started: float, metadata: dict[str, Any]
    ) -> SolverResult:
        try:
            return self._fallback(target, started, metadata)
        except StereoSewSingularityError as error:
            return self._failure(SolverStatus.SEW_SINGULAR, started, metadata, str(error))
        except (ValueError, TypeError) as error:
            return self._failure(SolverStatus.INVALID_INPUT, started, metadata, str(error))
        except Exception as error:
            metadata["cpp_fallback_exception_type"] = type(error).__name__
            return self._failure(SolverStatus.NUMERICAL_FAILURE, started, metadata, str(error))
