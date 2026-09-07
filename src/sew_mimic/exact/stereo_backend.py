"""Production Gen3 R-2R-2R-2R Stereo-SEW candidate enumeration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
import time
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..angles import wrap_to_pi
from ..common import ExactSewTarget, joint_limit_margin
from ..geometry import rot, sp1, sp2, sp3
from ..kinematics import Gen3Kinematics
from ..sew import Gen3StereoSewGeometry, StereoSew
from .residuals import robot_exact_sew_residuals
from .acceptance import (
    ORIENTATION_ACCEPTANCE_RAD,
    POSITION_ACCEPTANCE_M,
    SEW_ACCEPTANCE_RAD,
)
from .root_search import (
    EventAwareSearchConfig,
    RootSearchConfig,
    discover_feasible_intervals,
    search_fixed_slot_roots,
    solve_alignment_roots,
    sp2_feasibility_margin,
    sp3_feasibility_margin,
)


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
_ROTATION_TOL = 1e-10
_SUBPROBLEM_VECTOR_RELATIVE_TOL = 1e-10
_SP1_ALIGNMENT_RELATIVE_TOL = 1e-8


@dataclass(frozen=True)
class R2R2R2RSearchConfig:
    mode: str = "event_aware"
    reference_fixed_grid: RootSearchConfig = RootSearchConfig()
    event_aware: EventAwareSearchConfig = EventAwareSearchConfig()

    def __post_init__(self) -> None:
        if self.mode not in ("event_aware", "reference_fixed_grid"):
            raise ValueError("invalid search mode")


@dataclass(frozen=True)
class ExactSewCandidate:
    q: Vector
    wrist_search_angle: float
    search_branch: int
    position_error_m: float
    orientation_error_rad: float
    sew_error_rad: float
    joint_limit_valid: bool
    joint_limit_margin_rad: float
    exact: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("candidate q must be finite shape (7,)")
        object.__setattr__(self, "q", _readonly(q))
        if not isinstance(self.search_branch, int) or self.search_branch < 0:
            raise ValueError("search_branch must be a nonnegative integer")
        if not isinstance(self.joint_limit_valid, bool) or not isinstance(self.exact, bool):
            raise ValueError("joint_limit_valid and exact must be booleans")
        for name in ("wrist_search_angle", "position_error_m", "orientation_error_rad", "sew_error_rad", "joint_limit_margin_rad"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ExactSewCandidateSet:
    candidates: tuple[ExactSewCandidate, ...]
    search_root_count: int
    exact_geometric_count: int
    joint_limit_valid_count: int
    rejection_counts: MappingProxyType
    elapsed_ms: float

    def __post_init__(self) -> None:
        if any(not isinstance(candidate, ExactSewCandidate) for candidate in self.candidates):
            raise ValueError("candidates must contain ExactSewCandidate values")
        if min(self.search_root_count, self.exact_geometric_count, self.joint_limit_valid_count) < 0:
            raise ValueError("candidate counts must be nonnegative")
        if not math.isfinite(float(self.elapsed_ms)) or self.elapsed_ms < 0.0:
            raise ValueError("elapsed_ms must be finite and nonnegative")
        object.__setattr__(self, "rejection_counts", MappingProxyType(dict(self.rejection_counts)))


def _readonly(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copy into immutable backing storage so callers cannot re-enable writes."""
    return np.frombuffer(np.asarray(array, dtype=np.float64).tobytes(), dtype=np.float64).reshape(array.shape)


def _vector3(value: object, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite with shape (3,)")
    return vector.copy()


def _rotation3(value: object, name: str) -> Matrix:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be finite with shape (3, 3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_TOL, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_TOL, rtol=0.0):
        raise ValueError(f"{name} must have determinant +1")
    return rotation.copy()


@dataclass(frozen=True)
class NativeStereoSewTarget:
    """Exact-SEW target in the Phase-3 native PoE terminal convention.

    ``rotation_07`` maps the fixed terminal frame 7 into native Gen3 base 0;
    it deliberately excludes both the native pinch-site tool rotation and the
    established aligned-hand rotation.
    """

    position: Vector
    rotation_07: Matrix
    psi: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _readonly(_vector3(self.position, "position")))
        object.__setattr__(self, "rotation_07", _readonly(_rotation3(self.rotation_07, "rotation_07")))
        psi = float(self.psi)
        if not np.isfinite(psi):
            raise ValueError("psi must be finite")
        object.__setattr__(self, "psi", wrap_to_pi(psi))


def to_native_stereo_sew_target(
    target: ExactSewTarget,
    robot: Gen3Kinematics,
    geometry: Gen3StereoSewGeometry,
) -> NativeStereoSewTarget:
    """Convert aligned pinch target orientation to the Phase-3 native R_07.

    ``R_07 = R_target R_robot_align^T R_7T^T``.  Position and the already
    wrapped target psi retain their physical/task-space definitions unchanged.
    """
    if not isinstance(target, ExactSewTarget):
        raise ValueError("target must be an ExactSewTarget")
    rotation_07 = target.rotation @ robot.R_robot_align.T @ geometry.R_7T.T
    return NativeStereoSewTarget(target.position, rotation_07, target.psi)


def _valid_vector(left: Vector, right: Vector, tolerance: float = _SUBPROBLEM_VECTOR_RELATIVE_TOL) -> bool:
    scale = max(1.0, float(np.linalg.norm(left)), float(np.linalg.norm(right)))
    return float(np.linalg.norm(left - right)) <= tolerance * scale


def _represent(q: Vector, robot: Gen3Kinematics) -> tuple[Vector, bool, float]:
    answer = q.copy()
    valid = True
    for index, (lower, upper) in enumerate(robot.joint_limits):
        angle = wrap_to_pi(answer[index])
        if not math.isfinite(lower) and not math.isfinite(upper):
            answer[index] = angle
            continue
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("Gen3 joint limits must be both finite or both unlimited")
        lower_integer = math.ceil((lower - angle - 1e-12) / (2.0 * math.pi))
        upper_integer = math.floor((upper - angle + 1e-12) / (2.0 * math.pi))
        equivalents = [
            angle + 2.0 * math.pi * integer
            for integer in range(lower_integer, upper_integer + 1)
        ]
        if equivalents:
            answer[index] = min(equivalents, key=lambda value: (abs(value - angle), value))
        else:
            answer[index] = angle
            valid = False
    margin = joint_limit_margin(answer, robot)
    return answer, valid and margin >= -1e-12, margin


def enumerate_exact_sew_candidates(
    target: ExactSewTarget,
    robot: Gen3Kinematics,
    geometry: Gen3StereoSewGeometry,
    stereo: StereoSew,
    *,
    search_config: R2R2R2RSearchConfig = R2R2R2RSearchConfig(),
) -> ExactSewCandidateSet:
    """Enumerate all strict R-2R-2R-2R candidates; never select one."""
    if not isinstance(target, ExactSewTarget):
        raise ValueError("target must be ExactSewTarget")
    native = to_native_stereo_sew_target(target, robot, geometry)
    h, p = geometry.H, geometry.P
    counts = {
        "rejected_sp3": 0,
        "rejected_sp2_q23": 0,
        "rejected_sp2_q45": 0,
        "rejected_sp1": 0,
        "rejected_final": 0,
    }
    event_counts = {
        "event_margin_evaluations": 0,
        "event_alignment_evaluations": 0,
        "event_refinements": 0,
        "event_intervals": 0,
        "event_budget_exhausted": 0,
    }

    def record_event(report: Any) -> None:
        event_counts["event_margin_evaluations"] += int(report.margin_evaluations)
        event_counts["event_alignment_evaluations"] += int(report.alignment_evaluations)
        event_counts["event_refinements"] += int(report.maximum_refinements)
        event_counts["event_intervals"] += len(report.intervals)
        event_counts["event_budget_exhausted"] += int(report.budget_exhausted)

    started = time.perf_counter()

    # These are target-specific invariants.  Event discovery evaluates the
    # same wrist-circle angle many times, so keep both the invariant geometry
    # and exact-float-keyed intermediate results local to this enumeration.
    wrist = native.position - native.rotation_07 @ p[:, 7]
    shoulder = p[:, 0]
    p17 = wrist - shoulder
    p17_length = float(np.linalg.norm(p17))
    p3_length = float(np.linalg.norm(p[:, 3]))
    p5_length = float(np.linalg.norm(p[:, 5]))
    plane_normal = (
        stereo.inverse(shoulder, wrist, native.psi).plane_normal
        if p17_length > 1e-14
        else None
    )
    # A bounded cache preserves exact floating-point inputs while preventing a
    # pathological event tree from retaining one array per callback forever.
    _CACHE_MAXSIZE = 4096

    @lru_cache(maxsize=_CACHE_MAXSIZE)
    def pwe_trace(angle: float) -> tuple[Vector, Vector]:
        assert plane_normal is not None
        pwe = rot(plane_normal, angle) @ (-p17 / p17_length) * p5_length
        return p17, pwe

    @lru_cache(maxsize=_CACHE_MAXSIZE)
    def q1_geometry_trace(angle: float, branch: int) -> tuple[float, Matrix, Vector]:
        trace_p17, pwe = pwe_trace(angle)
        values = list(sp3(p[:, 1], trace_p17 + pwe, h[:, 0], p3_length).angles)
        q1 = values[min(branch, len(values) - 1)]
        r10 = rot(h[:, 0], -q1)
        return q1, r10, r10 @ trace_p17 + r10 @ pwe - p[:, 1]

    @lru_cache(maxsize=_CACHE_MAXSIZE)
    def q23_pairs_trace(angle: float, branch: int) -> tuple[tuple[float, float], ...]:
        _, _, vector23 = q1_geometry_trace(angle, branch)
        return tuple(sp2(p[:, 3], vector23, h[:, 2], -h[:, 1]))

    @lru_cache(maxsize=_CACHE_MAXSIZE)
    def q45_trace(angle: float, branch_i: int, branch_j: int) -> tuple[tuple[float, float], ...]:
        _, r10, _ = q1_geometry_trace(angle, branch_i)
        pairs23 = q23_pairs_trace(angle, branch_i)
        q3, q2 = pairs23[min(branch_j, len(pairs23) - 1)]
        vector45 = (
            rot(h[:, 2], -q3) @ rot(h[:, 1], -q2) @ (r10 @ p17 - p[:, 1])
            - p[:, 3]
        )
        return tuple(sp2(p[:, 5], vector45, h[:, 4], -h[:, 3]))

    def partial(angle: float, *, record: bool = False) -> tuple[Vector, list[Vector | None]]:
        values = np.full(8, np.nan)
        parts: list[Vector | None] = [None] * 8
        if p17_length <= 1e-14:
            return values, parts
        _, pwe = pwe_trace(angle)
        result = sp3(p[:, 1], p17 + pwe, h[:, 0], p3_length)
        q1_branches = list(zip(result.angles, result.is_exact, result.residuals))
        for i, (q1, exact, residual) in enumerate(q1_branches[:2]):
            if not exact or not math.isfinite(residual):
                if record:
                    counts["rejected_sp3"] += 4
                continue
            r10 = rot(h[:, 0], -q1)
            vector23 = r10 @ p17 + r10 @ pwe - p[:, 1]
            try:
                pairs23 = sp2(p[:, 3], vector23, h[:, 2], -h[:, 1])
            except ValueError:
                if record:
                    counts["rejected_sp2_q23"] += 4
                continue
            pairs23 = list(pairs23)
            for j, (q3, q2) in enumerate(pairs23[:2]):
                if not _valid_vector(
                    rot(h[:, 2], q3) @ p[:, 3],
                    rot(-h[:, 1], q2) @ vector23,
                ):
                    if record:
                        counts["rejected_sp2_q23"] += 2
                    continue
                r21 = rot(h[:, 1], -q2)
                r32 = rot(h[:, 2], -q3)
                vector45 = r32 @ r21 @ (r10 @ p17 - p[:, 1]) - p[:, 3]
                try:
                    pairs45 = sp2(p[:, 5], vector45, h[:, 4], -h[:, 3])
                except ValueError:
                    if record:
                        counts["rejected_sp2_q45"] += 2
                    continue
                pairs45 = list(pairs45)
                for k, (q5, q4) in enumerate(pairs45[:2]):
                    if not _valid_vector(
                        rot(h[:, 4], q5) @ p[:, 5],
                        rot(-h[:, 3], q4) @ vector45,
                    ):
                        if record:
                            counts["rejected_sp2_q45"] += 1
                        continue
                    r05 = (
                        r10.T
                        @ r21.T
                        @ r32.T
                        @ rot(h[:, 3], q4)
                        @ rot(h[:, 4], q5)
                    )
                    slot = 4 * i + 2 * j + k
                    values[slot] = (
                        h[:, 5] @ r05.T @ native.rotation_07 @ h[:, 6]
                        - h[:, 5] @ h[:, 6]
                    )
                    parts[slot] = np.array([q1, q2, q3, q4, q5])
        return values, parts

    def trace_alignment_slot(angle: float, slot: int) -> float:
        """Evaluate one diagnostic continuation slot without materializing all eight."""
        if p17_length <= 1e-14:
            return math.nan
        i, remainder = divmod(slot, 4)
        j, k = divmod(remainder, 2)
        try:
            _, r10, _ = q1_geometry_trace(angle, i)
            pairs23 = q23_pairs_trace(angle, i)
            q3, q2 = pairs23[min(j, len(pairs23) - 1)]
            pairs45 = q45_trace(angle, i, j)
        except (IndexError, ValueError):
            return math.nan
        q5, q4 = pairs45[min(k, len(pairs45) - 1)]
        r05 = (
            r10.T
            @ rot(h[:, 1], q2)
            @ rot(h[:, 2], q3)
            @ rot(h[:, 3], q4)
            @ rot(h[:, 4], q5)
        )
        return float(
            h[:, 5] @ r05.T @ native.rotation_07 @ h[:, 6]
            - h[:, 5] @ h[:, 6]
        )

    if search_config.mode == "reference_fixed_grid":
        roots = search_fixed_slot_roots(
            lambda angle: partial(angle)[0],
            search_config.reference_fixed_grid,
        ).roots
    else:
        event_roots = []
        base_event = search_config.event_aware

        # Lexical event tree: each child only searches an already certified
        # parent interval, so narrow downstream discriminants are never hidden
        # behind a broad SP3-only coarse grid.
        parent = discover_feasible_intervals(
            lambda angle: np.array(
                [
                    sp3_feasibility_margin(
                        p[:, 1],
                        sum(pwe_trace(angle)),
                        h[:, 0],
                        p3_length,
                    )
                ]
            ),
            base_event,
        )
        record_event(parent)
        for i in range(2):
            for parent_interval in parent.intervals:
                parent_config = replace(
                    base_event,
                    minimum=parent_interval.left,
                    maximum=parent_interval.right,
                )

                def q23_margin(angle: float, branch=i) -> Vector:
                    _, _, vector = q1_geometry_trace(angle, branch)
                    value = sp2_feasibility_margin(
                        p[:, 3], vector, h[:, 2], -h[:, 1]
                    )
                    return np.array([value])

                children23 = discover_feasible_intervals(q23_margin, parent_config)
                record_event(children23)
                for j in range(2):
                    for interval23 in children23.intervals:
                        config23 = replace(
                            base_event,
                            minimum=interval23.left,
                            maximum=interval23.right,
                        )

                        def q45_margin(angle: float, branch_i=i, branch_j=j) -> Vector:
                            _, r10, _ = q1_geometry_trace(angle, branch_i)
                            pairs = q23_pairs_trace(angle, branch_i)
                            q3, q2 = pairs[min(branch_j, len(pairs) - 1)]
                            vector45 = (
                                rot(h[:, 2], -q3)
                                @ rot(h[:, 1], -q2)
                                @ (r10 @ p17 - p[:, 1])
                                - p[:, 3]
                            )
                            value = sp2_feasibility_margin(
                                p[:, 5], vector45, h[:, 4], -h[:, 3]
                            )
                            return np.array([value])

                        children45 = discover_feasible_intervals(q45_margin, config23)
                        record_event(children45)
                        for k in range(2):
                            for interval45 in children45.intervals:
                                leaf_config = replace(
                                    base_event,
                                    minimum=interval45.left,
                                    maximum=interval45.right,
                                )

                                def align(
                                    angle: float, slot=4 * i + 2 * j + k
                                ) -> Vector:
                                    return np.array([trace_alignment_slot(angle, slot)])

                                leaf = solve_alignment_roots((interval45,), align, leaf_config)
                                record_event(leaf)
                                event_roots.extend(
                                    type(root)(
                                        root.angle,
                                        4 * i + 2 * j + k,
                                        root.residual,
                                    )
                                    for root in leaf.roots
                                )
        roots = tuple(
            sorted(event_roots, key=lambda root: (root.angle, root.slot))
        )
    candidates: list[ExactSewCandidate] = []
    for root in roots:
        q12345 = partial(root.angle, record=True)[1][root.slot]
        if q12345 is None: continue
        r05 = np.eye(3)
        for axis, qangle in zip(h[:, :5].T, q12345):
            r05 = r05 @ rot(axis, qangle)
        try:
            q6 = sp1(
                h[:, 6], r05.T @ native.rotation_07 @ h[:, 6], h[:, 5]
            )
            q7 = sp1(
                h[:, 5], native.rotation_07.T @ r05 @ h[:, 5], -h[:, 6]
            )
        except ValueError:
            counts["rejected_sp1"] += 1
            continue
        q6_exact = _valid_vector(
            rot(h[:, 5], q6) @ h[:, 6],
            r05.T @ native.rotation_07 @ h[:, 6],
            _SP1_ALIGNMENT_RELATIVE_TOL,
        )
        q7_exact = _valid_vector(
            rot(-h[:, 6], q7) @ h[:, 5],
            native.rotation_07.T @ r05 @ h[:, 5],
            _SP1_ALIGNMENT_RELATIVE_TOL,
        )
        if not q6_exact or not q7_exact:
            counts["rejected_sp1"] += 1
            continue
        q, valid, margin_value = _represent(
            np.concatenate((q12345, [q6, q7])), robot
        )
        try:
            residual = robot_exact_sew_residuals(
                q, target, robot, geometry, stereo
            )
        except ValueError:
            counts["rejected_final"] += 1
            continue
        exact = (
            residual.position_error_m < POSITION_ACCEPTANCE_M
            and residual.orientation_error_rad < ORIENTATION_ACCEPTANCE_RAD
            and residual.sew_error_rad is not None
            and residual.sew_error_rad < SEW_ACCEPTANCE_RAD
        )
        if not exact:
            counts["rejected_final"] += 1
            continue
        assert residual.sew_error_rad is not None
        candidates.append(
            ExactSewCandidate(
                q,
                root.angle,
                root.slot,
                residual.position_error_m,
                residual.orientation_error_rad,
                residual.sew_error_rad,
                valid,
                margin_value,
                True,
                {"root_residual": root.residual},
            )
        )
    candidates.sort(
        key=lambda item: (
            item.wrist_search_angle,
            item.search_branch,
            tuple(item.q.tolist()),
        )
    )
    return ExactSewCandidateSet(
        tuple(candidates),
        len(roots),
        len(candidates),
        sum(candidate.joint_limit_valid for candidate in candidates),
        MappingProxyType({**counts, **event_counts}),
        1000 * (time.perf_counter() - started),
    )
