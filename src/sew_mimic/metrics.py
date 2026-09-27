"""Retargeting error metrics."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike

from .tiago.model import TiagoKinematics


class RetargetDiagnostics(TypedDict):
    upper_arm_error_deg: float
    lower_arm_error_deg: float
    wrist_rotation_error_deg: float
    joint_limit_valid: bool


def _vector_angle_deg(first: ArrayLike, second: ArrayLike) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))


def _rotation_angle_deg(rotation: ArrayLike) -> float:
    matrix = np.asarray(rotation, dtype=float)
    sine = 0.5 * np.linalg.norm(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    cosine = 0.5 * (np.trace(matrix) - 1.0)
    return float(np.degrees(np.arctan2(sine, cosine)))


def compute_retarget_diagnostics(
    q: ArrayLike,
    upper_arm_direction: ArrayLike,
    lower_arm_direction: ArrayLike,
    hand_orientation: ArrayLike,
    robot: TiagoKinematics,
) -> RetargetDiagnostics:
    """Measure the three Algorithm 1 alignment residuals and joint validity."""
    configuration = np.asarray(q, dtype=float)
    robot_upper_arm = robot.R_0_i(configuration, 3) @ robot.arm_proxy_axis(3)
    robot_lower_arm = robot.R_0_i(configuration, 5) @ robot.arm_proxy_axis(5)
    hand = np.asarray(hand_orientation, dtype=float)
    limits = robot.joint_limits
    limit_valid = bool(
        np.all(configuration >= limits[:, 0] - 1e-12)
        and np.all(configuration <= limits[:, 1] + 1e-12)
    )

    return {
        "upper_arm_error_deg": _vector_angle_deg(upper_arm_direction, robot_upper_arm),
        "lower_arm_error_deg": _vector_angle_deg(lower_arm_direction, robot_lower_arm),
        "wrist_rotation_error_deg": _rotation_angle_deg(
            robot.aligned_ee_rotation(configuration).T @ hand
        ),
        "joint_limit_valid": limit_valid,
    }
