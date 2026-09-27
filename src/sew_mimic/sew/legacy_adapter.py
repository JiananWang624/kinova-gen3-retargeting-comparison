"""Result-contract adapter for the unchanged legacy SEW-Mimic solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..common import SolverDiagnostics, SolverResult, SolverStatus
from ..retarget import sew_mimic


METHOD_NAME = "legacy_sew_mimic"
LEGACY_CONSTRAINT_SET = "legacy_sew_direction_orientation"

# This tolerance classifies the existing angular post-validation diagnostics;
# it does not change the legacy solver or its candidate selection.
_LEGACY_EXACT_ANGLE_TOL_DEG = 1e-8


def _failure_status(error: ValueError) -> SolverStatus:
    message = str(error).lower()
    if "joint limits" in message:
        return SolverStatus.JOINT_LIMIT

    if message.startswith("q0 must"):
        return SolverStatus.INVALID_INPUT
    invalid_hand_markers = (
        "h must have shape",
        "h must contain only finite",
        "h must be orthogonal",
        "h must have determinant +1",
    )
    if any(marker in message for marker in invalid_hand_markers):
        return SolverStatus.INVALID_INPUT
    if message.startswith(("upper arm ", "lower arm ")) and (
        "must have shape" in message
        or "must contain only finite" in message
        or "endpoints must be distinct" in message
    ):
        return SolverStatus.INVALID_INPUT
    return SolverStatus.LEGACY_FAILURE


def solve_legacy_sew_mimic(
    q0: ArrayLike,
    shoulder: ArrayLike,
    elbow: ArrayLike,
    wrist: ArrayLike,
    hand_rotation: ArrayLike,
    robot=None,
) -> SolverResult:
    """Run Method 0 through the shared result contract.

    ``SUCCESS_EXACT`` is relative only to Method 0's legacy constraints:
    upper-arm direction, lower-arm direction, and aligned hand orientation.
    Method 0 does not constrain or validate human-hand/pinch-site position.
    """
    try:
        q, legacy = sew_mimic(q0, shoulder, elbow, wrist, hand_rotation, robot)
    except ValueError as error:
        return SolverResult(
            method=METHOD_NAME,
            status=_failure_status(error),
            q=None,
            diagnostics=SolverDiagnostics(
                metadata={
                    "constraint_set": LEGACY_CONSTRAINT_SET,
                    "legacy_exception_type": type(error).__name__,
                }
            ),
            message=str(error),
        )

    angular_errors_deg = np.array(
        [
            legacy["upper_arm_error_deg"],
            legacy["lower_arm_error_deg"],
            legacy["wrist_rotation_error_deg"],
        ],
        dtype=float,
    )
    exact = bool(
        legacy["joint_limit_valid"]
        and np.all(angular_errors_deg <= _LEGACY_EXACT_ANGLE_TOL_DEG)
    )
    status = SolverStatus.SUCCESS_EXACT if exact else SolverStatus.SUCCESS_APPROX

    return SolverResult(
        method=METHOD_NAME,
        status=status,
        q=q,
        diagnostics=SolverDiagnostics(
            orientation_error_rad=float(
                np.deg2rad(legacy["wrist_rotation_error_deg"])
            ),
            metadata={
                "constraint_set": LEGACY_CONSTRAINT_SET,
                "exact_angle_tolerance_deg": _LEGACY_EXACT_ANGLE_TOL_DEG,
                **legacy,
            },
        ),
        message=None if exact else "legacy constraints did not meet exact tolerance",
    )
