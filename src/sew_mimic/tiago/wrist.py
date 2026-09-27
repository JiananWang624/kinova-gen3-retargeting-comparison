"""Strict model-derived spherical-wrist gate and orientation decomposition."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import CONFIG
from .model import TiagoKinematics


GEOMETRY_TOL_M = 1e-10
GEOMETRY_TOL_RAD = 1e-10


@dataclass(frozen=True)
class WristGate:
    passed: bool
    maximum_axis_separation_m: float
    parallel_error_rad: float
    transverse_error_rad: float
    maximum_center_motion_m: float
    maximum_tool_offset_error_m: float
    maximum_reconstruction_error_rad: float
    tested_samples: int
    reason: str


def _line_distance(point: np.ndarray, anchor: np.ndarray, axis: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(point - anchor, axis)))


def _angle_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), abs(float(a @ b))))


def wrist_center(robot: TiagoKinematics, q: np.ndarray) -> np.ndarray:
    _, anchors = robot.axes_and_anchors(q)
    return anchors[5]


def decompose_wrist(robot: TiagoKinematics, q1_q4: np.ndarray, target_rotation: np.ndarray) -> list[np.ndarray]:
    """Enumerate the two regular ZXZ branches for actual J5/J6/J7 axes."""
    q_prefix = np.asarray(q1_q4, dtype=float)
    q_zero = np.concatenate((q_prefix, np.zeros(3)))
    zero_rotation = robot.tcp_pose(q_zero)[1]
    axes, _ = robot.axes_and_anchors(q_zero)
    a = axes[4]
    b = axes[5] - np.dot(axes[5], a) * a
    b /= np.linalg.norm(b)
    basis = np.column_stack((b, np.cross(a, b), a))
    delta = target_rotation @ zero_rotation.T
    local = basis.T @ delta @ basis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        alpha, beta, gamma = Rotation.from_matrix(local).as_euler("ZXZ")
    sign7 = 1.0 if float(axes[6] @ a) > 0 else -1.0
    raw = ((alpha, beta, sign7 * gamma), (alpha + np.pi, -beta, sign7 * (gamma + np.pi)))
    result: list[np.ndarray] = []
    lower, upper = robot.joint_limits[4:].T
    for branch in raw:
        wrapped = (np.asarray(branch) + np.pi) % (2 * np.pi) - np.pi
        if np.all(wrapped >= lower - 1e-12) and np.all(wrapped <= upper + 1e-12):
            result.append(wrapped)
    return result


def validate_spherical_wrist(robot: TiagoKinematics, *, samples: int = 1000, seed: int = 20260927) -> WristGate:
    """Reject a semi-analytic backend unless real-model geometry is exact."""
    if samples < 1:
        raise ValueError("samples must be positive")
    home = np.asarray(CONFIG["tiago"]["home_q_rad"], dtype=float)
    axes, anchors = robot.axes_and_anchors(home)
    center = anchors[5]
    axis_error = max(_line_distance(center, anchors[i], axes[i]) for i in (4, 5, 6))
    parallel_error = _angle_error(axes[4], axes[6])
    transverse_error = abs(float(axes[4] @ axes[5]))
    rng = np.random.default_rng(seed)
    max_center = 0.0
    max_tool = 0.0
    max_rotation = 0.0
    home_position, home_rotation = robot.tcp_pose(home)
    offset_in_tool = home_rotation.T @ (home_position - center)
    if axis_error > GEOMETRY_TOL_M or parallel_error > GEOMETRY_TOL_RAD or transverse_error > GEOMETRY_TOL_RAD:
        return WristGate(False, axis_error, parallel_error, transverse_error, 0.0, 0.0, 0.0, 0, "joint axes do not form an exact spherical ZXZ wrist")
    for _ in range(samples):
        q = rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1])
        center_actual = wrist_center(robot, q)
        center_zero = wrist_center(robot, np.concatenate((q[:4], np.zeros(3))))
        max_center = max(max_center, float(np.linalg.norm(center_actual - center_zero)))
        position, rotation = robot.tcp_pose(q)
        max_tool = max(max_tool, float(np.linalg.norm(rotation.T @ (position - center_actual) - offset_in_tool)))
        branches = decompose_wrist(robot, q[:4], rotation)
        if not branches:
            return WristGate(False, axis_error, parallel_error, transverse_error, max_center, max_tool, max_rotation, samples, "analytic wrist decomposition omitted an in-limit branch")
        branch_errors = []
        for branch in branches:
            reconstructed = robot.tcp_pose(np.concatenate((q[:4], branch)))[1]
            branch_errors.append(Rotation.from_matrix(reconstructed.T @ rotation).magnitude())
        max_rotation = max(max_rotation, min(branch_errors))
        if max_center > GEOMETRY_TOL_M or max_tool > GEOMETRY_TOL_M or max_rotation > GEOMETRY_TOL_RAD:
            return WristGate(False, axis_error, parallel_error, transverse_error, max_center, max_tool, max_rotation, samples, "wrist center, tool offset, or orientation reconstruction exceeds tolerance")
    return WristGate(True, axis_error, parallel_error, transverse_error, max_center, max_tool, max_rotation, samples, "validated exact spherical wrist")
