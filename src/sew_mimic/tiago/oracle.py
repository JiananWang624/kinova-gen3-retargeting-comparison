"""Validation-only MuJoCo/SciPy oracle; never used by production solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..angles import angular_difference
from ..config import CONFIG, project_path
from ..sew.stereo import StereoSew, StereoSewReference, StereoSewSingularityError
from .solver import TiagoSewTarget


@dataclass(frozen=True)
class OracleResult:
    classification: str
    q: np.ndarray | None
    best_residual_norm: float
    reason: str


class TiagoNumericalOracle:
    """Own MuJoCo model/data and direct FK, independent of production model code."""

    def __init__(self):
        settings = CONFIG["tiago"]
        self.model = mujoco.MjModel.from_xml_path(str(project_path(settings["model_path"])))
        self.data = mujoco.MjData(self.model)
        self.arm_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                 for name in settings["joint_names"]])
        self.qpos_ids = self.model.jnt_qposadr[self.arm_ids]
        self.limits = np.array(self.model.jnt_range[self.arm_ids])
        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, settings["tcp_site"])
        self.fixed = [(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)], value)
                      for name, value in settings["fixed_joints"].items()]
        sew = settings["sew"]
        self.selection = tuple(sew["selected"])
        self.stereo = StereoSew(StereoSewReference(np.array(sew["e_t"]), np.array(sew["e_r"])))
        self.home = np.array(settings["home_q_rad"])
        self.tolerances = settings["solver"]
        self._forward(self.home)
        anchors = self.data.xanchor[self.arm_ids]
        tcp = self.data.site_xpos[self.tcp_id]
        self.shoulder = anchors[0].copy()
        self.max_radius = float(np.sum(np.linalg.norm(np.diff(anchors, axis=0), axis=1))
                                + np.linalg.norm(tcp - anchors[-1]))

    def _forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        self.data.qpos[self.qpos_ids] = q
        for index, value in self.fixed:
            self.data.qpos[index] = value
        mujoco.mj_forward(self.model, self.data)
        position = self.data.site_xpos[self.tcp_id].copy()
        rotation = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        anchors = self.data.xanchor[self.arm_ids]
        axes = self.data.xaxis[self.arm_ids]
        def selected_point(name: str) -> np.ndarray:
            if name.startswith("J") and len(name) == 2 and name[1:].isdigit():
                return anchors[int(name[1:]) - 1]
            if name == "wrist_center":
                return anchors[5]
            a, b = {"J12": (0, 1), "J34": (2, 3), "J45": (3, 4)}[name]
            coefficients = np.linalg.lstsq(np.column_stack((axes[a], -axes[b])),
                                            anchors[b] - anchors[a], rcond=None)[0]
            return (anchors[a] + coefficients[0] * axes[a] +
                    anchors[b] + coefficients[1] * axes[b]) / 2
        psi = self.stereo.forward(*(selected_point(name) for name in self.selection))
        return position, rotation, psi

    def _errors(self, q: np.ndarray, target: TiagoSewTarget) -> tuple[float, float, float]:
        p, r, psi = self._forward(q)
        return float(np.linalg.norm(p - target.position)), float(Rotation.from_matrix(r.T @ target.rotation).magnitude()), abs(angular_difference(psi, target.psi))

    def _search(self, target: TiagoSewTarget, lower: np.ndarray, upper: np.ndarray,
                *, seeds: int, seed: int) -> tuple[np.ndarray | None, float]:
        rng = np.random.default_rng(seed)
        points = [np.clip(self.home, lower + 1e-9, upper - 1e-9)]
        points.extend(rng.uniform(lower, upper, (seeds - 1, 7)))
        best = float("inf")

        def residual(q: np.ndarray) -> np.ndarray:
            try:
                p, r, psi = self._forward(q)
                return np.r_[p - target.position,
                             0.2 * Rotation.from_matrix(r.T @ target.rotation).as_rotvec(),
                             0.2 * angular_difference(psi, target.psi)]
            except StereoSewSingularityError:
                return np.ones(7) * 10

        for point in points:
            fit = least_squares(residual, point, bounds=(lower, upper),
                                max_nfev=250, xtol=1e-11, ftol=1e-11, gtol=1e-11)
            norm = float(np.linalg.norm(fit.fun))
            best = min(best, norm)
            try:
                errors = self._errors(fit.x, target)
            except StereoSewSingularityError:
                continue
            if (errors[0] < self.tolerances["position_tolerance_m"] and
                errors[1] < self.tolerances["orientation_tolerance_rad"] and
                errors[2] < self.tolerances["sew_tolerance_rad"]):
                return fit.x.copy(), best
        return None, best

    def solve(self, target: TiagoSewTarget, *, seeds: int = 32, seed: int = 20260929) -> OracleResult:
        if np.linalg.norm(target.position - self.shoulder) > self.max_radius + 1e-10:
            return OracleResult("workspace_unreachable", None, float("inf"),
                                "TCP target exceeds triangle-inequality arm reach bound")
        found, best = self._search(target, self.limits[:, 0], self.limits[:, 1], seeds=seeds, seed=seed)
        if found is not None:
            return OracleResult("reachable", found, best, "strict MuJoCo pose and SEW solution found")
        relaxed, relaxed_best = self._search(target, np.full(7, -2 * math.pi),
                                              np.full(7, 2 * math.pi), seeds=seeds, seed=seed + 1)
        if relaxed is not None:
            return OracleResult("joint_limit_failure", relaxed, min(best, relaxed_best),
                                "strict solution found with relaxed limits after bounded multistart found none")
        return OracleResult("oracle_inconclusive", None, min(best, relaxed_best),
                            "multistart SciPy search found no strict solution; reachability unproven")
