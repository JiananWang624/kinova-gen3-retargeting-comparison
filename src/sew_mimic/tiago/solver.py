"""Stateful TIAGo pose plus Stereo-SEW solvers, without oracle fallback."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..angles import angular_difference
from ..common import HumanArmTarget, SolverDiagnostics, SolverResult, SolverStatus
from ..config import CONFIG
from ..sew.stereo import StereoSew, StereoSewReference, StereoSewSingularityError
from .model import TiagoKinematics
from .wrist import decompose_wrist, validate_spherical_wrist, wrist_center


@dataclass(frozen=True)
class TiagoSewTarget:
    position: np.ndarray
    rotation: np.ndarray
    psi: float


class SolverBackend(Protocol):
    def solve(self, target: TiagoSewTarget) -> SolverResult: ...
    def reset(self) -> None: ...


class TiagoSewGeometry:
    def __init__(self, robot: TiagoKinematics):
        self.robot = robot
        settings = CONFIG["tiago"]["sew"]
        self.selection = tuple(settings["selected"])
        self.stereo = StereoSew(StereoSewReference(np.array(settings["e_t"]), np.array(settings["e_r"])))

    def psi(self, q: np.ndarray) -> float:
        points = self.robot.sew_points(q, self.selection)
        return self.stereo.forward(points.shoulder, points.elbow, points.wrist)

    def target(self, human: HumanArmTarget) -> TiagoSewTarget:
        return TiagoSewTarget(human.task_point.copy(), human.hand_rotation.copy(),
                              self.stereo.forward(human.shoulder, human.elbow, human.wrist))


class _TiagoSolverBase:
    method = "tiago_sew"

    def __init__(self, robot: TiagoKinematics, geometry: TiagoSewGeometry):
        self.robot, self.geometry = robot, geometry
        self.settings = CONFIG["tiago"]["solver"]
        self.previous_q: np.ndarray | None = None
        self.home = np.array(CONFIG["tiago"]["home_q_rad"], dtype=float)

    def reset(self) -> None:
        self.previous_q = None

    def _seeds(self, dimension: int) -> list[np.ndarray]:
        seeds = []
        if self.previous_q is not None:
            seeds.append(self.previous_q[:dimension].copy())
        seeds.append(self.home[:dimension].copy())
        rng = np.random.default_rng(20260927)
        lower, upper = self.robot.joint_limits[:dimension].T
        for _ in range(int(self.settings["recovery_seeds"])):
            seeds.append(rng.uniform(lower, upper))
        return seeds

    def _valid_result(self, q: np.ndarray, target: TiagoSewTarget, started: float, branch: str) -> SolverResult | None:
        position, rotation = self.robot.tcp_pose(q)
        position_error = float(np.linalg.norm(position - target.position))
        rotation_error = float(Rotation.from_matrix(rotation.T @ target.rotation).magnitude())
        try:
            sew_error = abs(angular_difference(self.geometry.psi(q), target.psi))
        except StereoSewSingularityError:
            return None
        if not (position_error < self.settings["position_tolerance_m"] and
                rotation_error < self.settings["orientation_tolerance_rad"] and
                sew_error < self.settings["sew_tolerance_rad"]):
            return None
        if np.any((q < self.robot.joint_limits[:, 0] - 1e-12) |
                  (q > self.robot.joint_limits[:, 1] + 1e-12)):
            return None
        self.previous_q = q.copy()
        margin = float(np.min(np.minimum(q - self.robot.joint_limits[:, 0], self.robot.joint_limits[:, 1] - q)))
        return SolverResult(self.method, SolverStatus.SUCCESS_EXACT, q,
                            SolverDiagnostics(position_error_m=position_error,
                                              orientation_error_rad=rotation_error,
                                              sew_error_rad=sew_error,
                                              joint_limit_margin_rad=margin,
                                              solve_time_ms=(time.perf_counter() - started) * 1000,
                                              branch_id=branch), None)

    def _failure(self, started: float, reason: str) -> SolverResult:
        return SolverResult(self.method, SolverStatus.NUMERICAL_FAILURE, None,
                            SolverDiagnostics(solve_time_ms=(time.perf_counter() - started) * 1000), reason)


class TiagoSewSolver(_TiagoSolverBase):
    """Four-dimensional arm solve, exact wrist branches, then one boundary correction."""

    def __init__(self, robot: TiagoKinematics, geometry: TiagoSewGeometry):
        super().__init__(robot, geometry)
        if geometry.selection[2] != "wrist_center":
            raise ValueError("TIAGo semi-analytic solve requires SEW W at the spherical wrist center")
        gate = validate_spherical_wrist(robot, samples=1000)
        if not gate.passed:
            raise ValueError(f"TIAGo semi-analytic wrist hard gate failed: {gate.reason}; {gate}")
        q = self.home
        position, rotation = robot.tcp_pose(q)
        self.offset_in_tool = rotation.T @ (position - wrist_center(robot, q))

    def solve(self, target: TiagoSewTarget) -> SolverResult:
        started = time.perf_counter()
        target_center = target.position - target.rotation @ self.offset_in_tool
        lower, upper = self.robot.joint_limits[:4].T
        wrist_lower, wrist_upper = self.robot.joint_limits[4:].T
        previous = self.previous_q
        boundary_seed: tuple[float, int, int, np.ndarray] | None = None

        def residual(prefix: np.ndarray) -> np.ndarray:
            q = np.concatenate((prefix, np.zeros(3)))
            points = self.robot.sew_points(q, self.geometry.selection)
            try:
                psi = self.geometry.stereo.forward(points.shoulder, points.elbow, points.wrist)
                psi_error = angular_difference(psi, target.psi)
            except StereoSewSingularityError:
                psi_error = math.pi
            return np.r_[points.wrist - target_center, 0.2 * psi_error]

        def pose_residual(q: np.ndarray) -> np.ndarray:
            position, rotation = self.robot.tcp_pose(q)
            orientation_error = Rotation.from_matrix(rotation.T @ target.rotation).as_rotvec()
            try:
                psi_error = angular_difference(self.geometry.psi(q), target.psi)
            except StereoSewSingularityError:
                psi_error = math.pi
            return np.r_[position - target.position, 0.2 * orientation_error,
                         0.2 * psi_error]

        for number, seed in enumerate(self._seeds(4)):
            fit = least_squares(residual, seed, bounds=(lower, upper),
                                max_nfev=int(self.settings["local_max_nfev"] if number == 0 and previous is not None
                                             else self.settings["global_max_nfev"]),
                                xtol=1e-11, ftol=1e-11, gtol=1e-11)
            if float(np.linalg.norm(fit.fun)) > 0.005:
                continue
            analytic_branches = decompose_wrist(self.robot, fit.x, target.rotation,
                                                 enforce_limits=False)
            branches = [(index, branch) for index, branch in enumerate(analytic_branches)
                        if np.all(branch >= wrist_lower - 1e-12)
                        and np.all(branch <= wrist_upper + 1e-12)]
            if previous is not None:
                branches.sort(key=lambda item: float(np.linalg.norm((item[1] - previous[4:] + np.pi) % (2 * np.pi) - np.pi)))
            for branch_number, branch in branches:
                result = self._valid_result(np.r_[fit.x, branch], target, started,
                                            f"seed-{number}-wrist-{branch_number}")
                if result is not None:
                    return result
            for branch_number, branch in enumerate(analytic_branches):
                if np.any((branch < wrist_lower - 1e-12) | (branch > wrist_upper + 1e-12)):
                    projected = np.clip(branch, wrist_lower, wrist_upper)
                    candidate = np.r_[fit.x, projected]
                    score = float(np.linalg.norm(pose_residual(candidate)))
                    if boundary_seed is None or score < boundary_seed[0]:
                        boundary_seed = (score, number, branch_number, candidate)
        if boundary_seed is not None:
            _, seed_number, branch_number, candidate = boundary_seed
            refined = least_squares(pose_residual, candidate,
                                    bounds=self.robot.joint_limits.T,
                                    max_nfev=int(self.settings["local_max_nfev"]),
                                    xtol=1e-11, ftol=1e-11, gtol=1e-11)
            result = self._valid_result(refined.x, target, started,
                                        f"seed-{seed_number}-wrist-{branch_number}-boundary")
            if result is not None:
                result.diagnostics.metadata["solver_stage"] = "analytic_wrist_boundary_refinement"
                return result
        return self._failure(started, "no exact or boundary-refined wrist branch met strict TCP/SEW tolerances")


class TiagoNumericalSolver(_TiagoSolverBase):
    """Independent seven-variable TIAGo validation and fallback candidate."""

    def solve(self, target: TiagoSewTarget) -> SolverResult:
        started = time.perf_counter()
        lower, upper = self.robot.joint_limits.T
        previous = self.previous_q

        def residual(q: np.ndarray) -> np.ndarray:
            position, rotation = self.robot.tcp_pose(q)
            angle = Rotation.from_matrix(rotation.T @ target.rotation).as_rotvec()
            try:
                psi_error = angular_difference(self.geometry.psi(q), target.psi)
            except StereoSewSingularityError:
                psi_error = math.pi
            return np.r_[position - target.position, 0.2 * angle, 0.2 * psi_error]

        for number, seed in enumerate(self._seeds(7)):
            fit = least_squares(residual, seed, bounds=(lower, upper),
                                max_nfev=int(self.settings["local_max_nfev"] if number == 0 and previous is not None
                                             else self.settings["global_max_nfev"]),
                                xtol=1e-11, ftol=1e-11, gtol=1e-11)
            result = self._valid_result(fit.x, target, started, f"seed-{number}")
            if result is not None:
                return result
        return self._failure(started, "seven-variable TIAGo search did not meet strict TCP/SEW tolerances")


def solve_tiago_sew(target: TiagoSewTarget, robot: TiagoKinematics | None = None,
                    geometry: TiagoSewGeometry | None = None) -> SolverResult:
    robot = robot or TiagoKinematics()
    geometry = geometry or TiagoSewGeometry(robot)
    backend = CONFIG["tiago"]["solver"]["backend"]
    if backend == "tiago_semi_analytic":
        return TiagoSewSolver(robot, geometry).solve(target)
    if backend == "tiago_numerical":
        return TiagoNumericalSolver(robot, geometry).solve(target)
    raise ValueError(f"unsupported TIAGo production backend {backend}")
