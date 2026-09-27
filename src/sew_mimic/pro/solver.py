"""Stateful seven-variable Pro pose plus engineering Stereo-SEW solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from ..angles import angular_difference
from ..common import HumanArmTarget, SolverDiagnostics, SolverResult, SolverStatus
from ..config import CONFIG
from ..sew.stereo import StereoSew, StereoSewReference, StereoSewSingularityError
from .model import ProKinematics

try:
    from ._pro_7d_core import ProCore
except ImportError as error:
    raise ImportError("Pro C++ solver is missing; run 'python -m pip install -e .' to build it") from error


@dataclass(frozen=True)
class ProTarget:
    position: np.ndarray
    rotation: np.ndarray
    psi: float


class ProSewGeometry:
    """S=J1, E=J4, W=tool_link; W is not a physical wrist center."""

    def __init__(self, robot: ProKinematics):
        self.robot = robot
        settings = CONFIG["tiago_pro"]["sew"]
        if settings["selected"] != ["J1", "J4", "arm_right_tool_link"]:
            raise ValueError("Pro engineering SEW must be J1/J4/arm_right_tool_link")
        self.stereo = StereoSew(StereoSewReference(
            np.asarray(settings["e_t"], dtype=float),
            np.asarray(settings["e_r"], dtype=float),
        ))

    def psi(self, q: np.ndarray) -> float:
        return self.stereo.forward(*self.robot.sew_points(q))

    def target(self, human: HumanArmTarget) -> ProTarget:
        psi = self.stereo.forward(human.shoulder, human.elbow, human.wrist)
        return ProTarget(human.task_point.copy(), human.hand_rotation.copy(), psi)


class ProSewSolver:
    method = "tiago_pro_7d"

    def __init__(self, robot: ProKinematics, geometry: ProSewGeometry):
        self.robot = robot
        self.geometry = geometry
        settings = CONFIG["tiago_pro"]["solver"]
        if settings["backend"] != "pro_7d_cpp":
            raise ValueError("unsupported Pro production backend")
        self.settings = settings
        source = robot.geometry_at_zero()
        self.core = ProCore(source["axes"], source["anchors"], source["tool_position"],
                            source["grasp_rotation"], geometry.stereo.reference.e_t,
                            geometry.stereo.reference.e_r, robot.joint_limits)
        self.previous_q: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_q = None

    def _seeds(self) -> list[np.ndarray]:
        low, high = self.robot.joint_limits.T
        midpoint = (low + high) / 2
        seeds = [np.zeros(7), midpoint]
        # Fixed low-discrepancy recovery points, independent of frame order.
        primes = (2, 3, 5, 7, 11, 13, 17)
        for index in range(1, int(self.settings["recovery_seeds"]) + 1):
            coords = []
            for prime in primes:
                n, fraction, scale = index, 0.0, 1.0 / prime
                while n:
                    n, digit = divmod(n, prime)
                    fraction += digit * scale
                    scale /= prime
                coords.append(fraction)
            seeds.append(low + np.asarray(coords) * (high - low))
        return seeds

    def _postvalidate(self, q: np.ndarray, target: ProTarget) -> tuple[float, float, float, float] | None:
        if np.any(q < self.robot.joint_limits[:, 0] - 1e-12) or np.any(q > self.robot.joint_limits[:, 1] + 1e-12):
            return None
        position, rotation = self.robot.task_poses(q)
        position_error = float(np.linalg.norm(position - target.position))
        orientation_error = float(Rotation.from_matrix(target.rotation @ rotation.T).magnitude())
        try:
            sew_error = abs(angular_difference(self.geometry.psi(q), target.psi))
        except StereoSewSingularityError:
            return None
        limit_margin = float(np.min(np.minimum(q - self.robot.joint_limits[:, 0],
                                               self.robot.joint_limits[:, 1] - q)))
        if (position_error >= self.settings["position_tolerance_m"] or
            orientation_error >= self.settings["orientation_tolerance_rad"] or
            sew_error >= self.settings["sew_tolerance_rad"]):
            return None
        return position_error, orientation_error, sew_error, limit_margin

    def solve(self, target: ProTarget) -> SolverResult:
        started = perf_counter()
        if (target.position.shape != (3,) or target.rotation.shape != (3, 3) or
            not np.all(np.isfinite(target.position)) or not np.all(np.isfinite(target.rotation)) or
            not math.isfinite(float(target.psi))):
            return SolverResult(self.method, SolverStatus.INVALID_INPUT, None,
                                SolverDiagnostics(solve_time_ms=(perf_counter() - started) * 1000),
                                "invalid Pro task target")
        if self.previous_q is None:
            seeds = [np.zeros(7)]
        else:
            predicted = np.asarray(self.core.predict(target.position, target.rotation,
                                                     float(target.psi), self.previous_q), dtype=float)
            seeds = [predicted, self.previous_q.copy()]
        recoveries = self._seeds()
        attempts = 0
        iterations = 0
        best_cost = math.inf
        best_reason = "no valid strict in-limit solution found by bounded search"
        tried: list[np.ndarray] = []
        for seed in seeds + recoveries:
            if any(np.allclose(seed, past, atol=1e-12, rtol=0) for past in tried):
                continue
            tried.append(seed)
            max_iterations = int(self.settings["local_max_iterations"] if attempts == 0 else
                                 self.settings["recovery_max_iterations"])
            raw = self.core.solve(target.position, target.rotation, float(target.psi), seed, max_iterations,
                                  float(self.settings["position_tolerance_m"]),
                                  float(self.settings["orientation_tolerance_rad"]),
                                  float(self.settings["sew_tolerance_rad"]))
            attempts += 1
            iterations += int(raw["iterations"])
            best_cost = min(best_cost, float(raw["cost"]))
            q = np.asarray(raw["q"], dtype=float)
            if raw["success"]:
                checked = self._postvalidate(q, target)
                if checked is not None:
                    self.previous_q = q.copy()
                    pos, orientation, sew, margin = checked
                    return SolverResult(self.method, SolverStatus.SUCCESS_EXACT, q,
                                        SolverDiagnostics(pos, orientation, sew, margin,
                                                          (perf_counter() - started) * 1000,
                                                          branch_id=f"seed_{attempts-1}",
                                                          metadata={"iterations": iterations, "attempts": attempts}),
                                        "strict MuJoCo-validated Pro 7D solution")
                best_reason = "C++ candidate failed authoritative MuJoCo post-validation"
        return SolverResult(self.method, SolverStatus.NUMERICAL_FAILURE, None,
                            SolverDiagnostics(solve_time_ms=(perf_counter() - started) * 1000,
                                              metadata={"iterations": iterations, "attempts": attempts,
                                                        "best_cost": best_cost}),
                            best_reason)
