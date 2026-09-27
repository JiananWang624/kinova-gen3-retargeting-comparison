"""Typed result structures shared by retargeting methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .status import SolverStatus


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

_ROTATION_MATRIX_TOL = 1e-10


def _vector3(value: object, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def _rotation3(value: object, name: str) -> Matrix:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(
        matrix.T @ matrix,
        np.eye(3),
        atol=_ROTATION_MATRIX_TOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(
        np.linalg.det(matrix),
        1.0,
        atol=_ROTATION_MATRIX_TOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must have determinant +1")
    return matrix.copy()


def _optional_finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite when present")
    return number


@dataclass
class HumanArmTarget:
    """One human arm target expressed entirely in one documented frame.

    ``hand_rotation`` maps the canonical hand frame into the containing frame.
    ``task_point`` is the physical point that an end-effector position solver
    or evaluator should track.
    """

    shoulder: Vector
    elbow: Vector
    wrist: Vector
    hand_rotation: Matrix
    task_point: Vector

    def __post_init__(self) -> None:
        self.shoulder = _vector3(self.shoulder, "shoulder")
        self.elbow = _vector3(self.elbow, "elbow")
        self.wrist = _vector3(self.wrist, "wrist")
        self.hand_rotation = _rotation3(self.hand_rotation, "hand_rotation")
        self.task_point = _vector3(self.task_point, "task_point")


@dataclass
class SolverDiagnostics:
    """Common diagnostics, with method-specific values kept in ``metadata``."""

    position_error_m: float | None = None
    orientation_error_rad: float | None = None
    sew_error_rad: float | None = None
    joint_limit_margin_rad: float | None = None
    solve_time_ms: float | None = None
    branch_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "position_error_m",
            "orientation_error_rad",
            "sew_error_rad",
            "joint_limit_margin_rad",
            "solve_time_ms",
        ):
            setattr(self, name, _optional_finite(getattr(self, name), name))
        if self.branch_id is not None and not isinstance(self.branch_id, str):
            raise ValueError("branch_id must be a string when present")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary while preserving method metadata values."""
        return asdict(self)


@dataclass
class SolverResult:
    """A joint solution and its explicit solver outcome."""

    method: str
    status: SolverStatus
    q: Vector | None
    diagnostics: SolverDiagnostics = field(default_factory=SolverDiagnostics)
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a nonempty string")
        if not isinstance(self.status, SolverStatus):
            raise ValueError("status must be a SolverStatus")
        if not isinstance(self.diagnostics, SolverDiagnostics):
            raise ValueError("diagnostics must be SolverDiagnostics")
        if self.message is not None and not isinstance(self.message, str):
            raise ValueError("message must be a string when present")
        if self.q is not None:
            configuration = np.asarray(self.q, dtype=float)
            if configuration.shape != (7,):
                raise ValueError(f"q must have shape (7,), got {configuration.shape}")
            if not np.all(np.isfinite(configuration)):
                raise ValueError("q must contain only finite values")
            self.q = configuration.copy()
        success = self.status in (
            SolverStatus.SUCCESS_EXACT,
            SolverStatus.SUCCESS_APPROX,
        )
        if success and self.q is None:
            raise ValueError("successful solver results must contain q")
        if not success and self.q is not None:
            raise ValueError("failed solver results must not contain q")

    def to_dict(self) -> dict[str, Any]:
        """Normalize the shared fields while preserving metadata values."""
        return {
            "method": self.method,
            "status": self.status.value,
            "q": None if self.q is None else self.q.tolist(),
            "diagnostics": self.diagnostics.to_dict(),
            "message": self.message,
        }
