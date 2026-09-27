"""Closed-form SEW-Mimic retargeting steps."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import sp1, sp2
from .human_input import compute_lower_arm_direction, compute_upper_arm_direction
from .metrics import RetargetDiagnostics, compute_retarget_diagnostics
from .tiago.model import TiagoKinematics


Vector = NDArray[np.float64]

_LIMIT_TOL = 1e-12


def _configuration(q: ArrayLike, dof: int) -> Vector:
    configuration = np.asarray(q, dtype=float)
    if configuration.shape != (dof,):
        raise ValueError(f"q0 must have shape ({dof},), got {configuration.shape}")
    if not np.all(np.isfinite(configuration)):
        raise ValueError("q0 must contain only finite values")
    return configuration


def _unit_vector(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _rotation_matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must be orthogonal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must have determinant +1")
    return matrix


def _bound_angle(angle: float, current: float, limits: NDArray[np.float64]) -> float:
    """Implement Algorithm 3's BoundJoints for one periodic revolute joint."""
    lower, upper = map(float, limits)
    period = 2.0 * np.pi

    if np.isneginf(lower) and np.isposinf(upper):
        return float(angle + period * np.round((current - angle) / period))

    minimum_turn = int(np.ceil((lower - angle - _LIMIT_TOL) / period))
    maximum_turn = int(np.floor((upper - angle + _LIMIT_TOL) / period))
    if minimum_turn > maximum_turn:
        raise ValueError("no q7 solution satisfies the joint limits")
    turns = np.arange(minimum_turn, maximum_turn + 1)
    candidates = angle + period * turns
    return float(candidates[np.argmin(np.abs(candidates - current))])


def align_axis(i: int, q0: ArrayLike, v: ArrayLike, robot: TiagoKinematics) -> Vector:
    """Implement Algorithm 2 ``AlignAxis`` using 1-based joint index ``i``.

    The returned pair contains the absolute angles ``(q[i-2], q[i-1])`` in
    the paper's 1-based notation.
    """
    if not isinstance(i, (int, np.integer)) or not 3 <= int(i) <= robot.dof:
        raise ValueError(f"i must be an integer in [3, {robot.dof}]")
    joint_index = int(i)
    configuration = _configuration(q0, robot.dof)
    target = _unit_vector(v, "v")
    frame_index = joint_index - 2
    first_q_index = joint_index - 3
    second_q_index = joint_index - 2

    # Algorithm 2, line 1: put target v into the current (i-2) frame.
    rotation_0_to_frame = robot.R_0_i(configuration, frame_index)
    target_in_frame = rotation_0_to_frame.T @ target

    # Algorithm 2, line 2: express the axis to align in frame (i-2). For the
    # Upper/lower limbs use signed pointing proxies; axes[] remains
    # the native kinematic rotation-axis convention. Wrist i=7 stays native.
    axis_to_align_local = (
        robot.arm_proxy_axis(joint_index)
        if joint_index in (3, 5)
        else robot.axes[joint_index - 1]
    )
    axis_to_align_in_frame = (
        rotation_0_to_frame.T
        @ robot.R_0_i(configuration, joint_index)
        @ axis_to_align_local
    )

    # Algorithm 2, line 3: predecessor axes are always native rotation axes.
    predecessor_axis_in_frame = (
        rotation_0_to_frame.T
        @ robot.R_0_i(configuration, joint_index - 1)
        @ robot.axes[joint_index - 2]
    )

    # Algorithm 2, lines 4-5: solve the two-axis alignment with SP2.
    angle_deltas = sp2(
        target_in_frame,
        axis_to_align_in_frame,
        -robot.axes[joint_index - 3],
        predecessor_axis_in_frame,
    )
    current_pair = configuration[[first_q_index, second_q_index]]
    candidates = current_pair + angle_deltas

    # Algorithm 2, line 6: discard candidates violating either joint limit.
    limits = robot.joint_limits[[first_q_index, second_q_index]]
    within_limits = np.all(
        (candidates >= limits[:, 0] - _LIMIT_TOL)
        & (candidates <= limits[:, 1] + _LIMIT_TOL),
        axis=1,
    )
    bounded_candidates = candidates[within_limits]
    if bounded_candidates.size == 0:
        raise ValueError(
            f"no AlignAxis solution for joints {joint_index - 2} and "
            f"{joint_index - 1} satisfies the joint limits"
        )

    # Algorithm 2, line 7: choose argmin |q0[i-2]-a| + |q0[i-1]-b|.
    distances = np.sum(np.abs(bounded_candidates - current_pair), axis=1)
    return bounded_candidates[int(np.argmin(distances))].copy()


def align_wrist(
    q0: ArrayLike,
    H: ArrayLike,
    robot: TiagoKinematics,
) -> Vector:
    """Implement Algorithm 3 for the configured parallel wrist.

    ``H`` uses the paper's right-handed hand convention. The returned vector
    contains the absolute joint angles ``(q5, q6, q7)``.
    """
    configuration = _configuration(q0, robot.dof)
    hand_orientation = _rotation_matrix(H, "H")

    # Algorithm 3, line 1: fixed EE orientation in the seventh joint frame.
    rotation_local_7_to_tool = robot.ee_rotation_in_7

    # Algorithm 3, line 2: desired orientation of the seventh joint frame.
    desired_rotation_0_to_7 = (
        hand_orientation
        @ robot.R_robot_align.T
        @ rotation_local_7_to_tool.T
    )

    # Algorithm 3, line 3: align h7 to the desired tool pointing direction
    # using AlignAxis/SP2. The paper writes R_des[:, 1] under its h7=+X
    # frame convention; R_des @ h7 is the coordinate-invariant equivalent.
    desired_axis_7 = desired_rotation_0_to_7 @ robot.axes[6]
    q5_q6 = align_axis(7, configuration, desired_axis_7, robot)
    wrist_configuration = configuration.copy()
    wrist_configuration[4:6] = q5_q6

    # Algorithm 3, line 4: put h6 and its desired direction in frame 7.
    desired_axis_6_in_7 = (
        desired_rotation_0_to_7.T
        @ robot.R_0_i(wrist_configuration, 6)
        @ robot.axes[5]
    )
    rotation_local_6_to_7 = robot.fixed_parent_to_child[6, :3, :3]
    axis_6_in_7 = rotation_local_6_to_7.T @ robot.axes[5]

    # Algorithm 3, line 5: solve q7 with SP1, then enforce its joint limits.
    q7 = sp1(axis_6_in_7, desired_axis_6_in_7, -robot.axes[6])
    q7 = _bound_angle(q7, configuration[6], robot.joint_limits[6])

    return np.array([q5_q6[0], q5_q6[1], q7])


def sew_mimic(
    q0: ArrayLike,
    shoulder: ArrayLike,
    elbow: ArrayLike,
    wrist: ArrayLike,
    H: ArrayLike,
    robot: TiagoKinematics,
) -> tuple[Vector, RetargetDiagnostics]:
    """Implement Algorithm 1 on the explicitly supplied seven-joint arm."""
    # Algorithm 1, line 1: initialize the output configuration from q0.
    q = _configuration(q0, robot.dof).copy()

    # Algorithm 1, line 2: compute the human upper/lower-arm directions.
    upper_arm = compute_upper_arm_direction(shoulder, elbow)
    lower_arm = compute_lower_arm_direction(elbow, wrist)

    # Algorithm 1, line 3: align h3 by solving q1 and q2.
    q[0:2] = align_axis(3, q, upper_arm, robot)

    # Algorithm 1, line 4: align h5 using the updated q1 and q2.
    q[2:4] = align_axis(5, q, lower_arm, robot)

    # Algorithm 1, line 5: align the wrist using the updated q1 through q4.
    q[4:7] = align_wrist(q, H, robot)

    diagnostics = compute_retarget_diagnostics(q, upper_arm, lower_arm, H, robot)
    return q, diagnostics
