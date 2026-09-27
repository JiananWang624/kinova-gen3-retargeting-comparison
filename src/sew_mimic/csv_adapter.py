"""Adapter from the human trajectory CSV schema to SEW-Mimic inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .config import CONFIG
from .human_input import wrist_euler_to_rotation


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

POSITION_COLUMNS = (
    "Shoulder_X",
    "Shoulder_Y",
    "Shoulder_Z",
    "Elbow_X",
    "Elbow_Y",
    "Elbow_Z",
    "Wrist_X",
    "Wrist_Y",
    "Wrist_Z",
)
EULER_COLUMNS = ("Wrist_Rx", "Wrist_Ry", "Wrist_Rz")
REQUIRED_COLUMNS = POSITION_COLUMNS + EULER_COLUMNS

_HUMAN_CSV_CONFIG = CONFIG["human_csv"]
R_BODY_FROM_CSV = np.asarray(
    _HUMAN_CSV_CONFIG["rotation_body_from_csv"], dtype=float
)
R_INPUT_ALIGN = np.asarray(_HUMAN_CSV_CONFIG["rotation_input_align"], dtype=float)
WRIST_EULER_ORDER = str(_HUMAN_CSV_CONFIG["wrist_euler_order"])
WRIST_EULER_DEGREES = bool(_HUMAN_CSV_CONFIG["wrist_euler_degrees"])
WRIST_EULER_CONVENTION = str(_HUMAN_CSV_CONFIG["wrist_euler_convention"])


@dataclass(frozen=True)
class HumanTrajectory:
    shoulders: NDArray[np.float64]
    elbows: NDArray[np.float64]
    wrists: NDArray[np.float64]
    hand_orientations: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.shoulders)


@dataclass(frozen=True)
class HumanCSVAdapter:
    """Apply every CSV-to-robot coordinate conversion in one place.

    Position scale, frame rotations, and wrist Euler convention come from the
    project configuration unless explicitly overridden.
    """

    rotation_body_from_csv: ArrayLike = field(
        default_factory=lambda: R_BODY_FROM_CSV.copy()
    )
    position_scale: float = float(_HUMAN_CSV_CONFIG["position_scale_to_m"])
    rotation_input_align: ArrayLike = field(
        default_factory=lambda: R_INPUT_ALIGN.copy()
    )
    wrist_euler_order: str = WRIST_EULER_ORDER
    wrist_euler_degrees: bool = WRIST_EULER_DEGREES
    wrist_euler_convention: str = WRIST_EULER_CONVENTION

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_body_from_csv, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation_body_from_csv must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
            raise ValueError("rotation_body_from_csv must be orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("rotation_body_from_csv must have determinant +1")
        if not np.isfinite(self.position_scale) or self.position_scale <= 0.0:
            raise ValueError("position_scale must be positive and finite")
        input_alignment = np.asarray(self.rotation_input_align, dtype=float)
        if input_alignment.shape != (3, 3) or not np.all(np.isfinite(input_alignment)):
            raise ValueError("rotation_input_align must be a finite 3x3 matrix")
        if not np.allclose(
            input_alignment.T @ input_alignment, np.eye(3), atol=1e-10, rtol=0.0
        ):
            raise ValueError("rotation_input_align must be orthogonal")
        if not np.isclose(np.linalg.det(input_alignment), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("rotation_input_align must have determinant +1")
        object.__setattr__(self, "rotation_body_from_csv", rotation.copy())
        object.__setattr__(self, "rotation_input_align", input_alignment.copy())

    def position_to_world(self, point: ArrayLike) -> Vector:
        """Apply ``p_world = R_body_from_csv @ (scale * p_csv)``."""
        position = np.asarray(point, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("CSV position must be a finite length-3 vector")
        return self.rotation_body_from_csv @ (self.position_scale * position)

    def adapt_frame(
        self,
        shoulder: ArrayLike,
        elbow: ArrayLike,
        wrist: ArrayLike,
        wrist_euler: ArrayLike,
    ) -> tuple[Vector, Vector, Vector, Matrix]:
        """Convert one CSV frame into robot-body-frame ``(s, e, w, H)``."""
        hand_orientation_csv = wrist_euler_to_rotation(
            wrist_euler,
            order=self.wrist_euler_order,
            degrees=self.wrist_euler_degrees,
            convention=self.wrist_euler_convention,
        )
        converted_points = tuple(
            self.position_to_world(point) for point in (shoulder, elbow, wrist)
        )

        # R_wrist_csv maps Motive device axes into CSV world axes. The left
        # product changes world basis; the right product changes the local
        # device basis into canonical hand X(pointing), Y(palm), Z(thumb).
        hand_orientation_world = (
            self.rotation_body_from_csv
            @ hand_orientation_csv
            @ self.rotation_input_align
        )
        return (*converted_points, hand_orientation_world)


def load_human_trajectory_csv(
    path: str | Path,
    adapter: HumanCSVAdapter | None = None,
) -> HumanTrajectory:
    """Load and adapt every row of a human trajectory CSV."""
    table = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    values = table.loc[:, REQUIRED_COLUMNS].to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("CSV must contain at least one frame")
    if not np.all(np.isfinite(values)):
        bad_rows = np.flatnonzero(~np.isfinite(values).all(axis=1))
        raise ValueError(f"CSV contains non-finite required values in rows {bad_rows.tolist()}")

    converter = adapter if adapter is not None else HumanCSVAdapter()
    frame_count = len(table)
    shoulders = np.empty((frame_count, 3))
    elbows = np.empty((frame_count, 3))
    wrists = np.empty((frame_count, 3))
    hand_orientations = np.empty((frame_count, 3, 3))
    for frame, row in enumerate(values):
        shoulder, elbow, wrist, hand = converter.adapt_frame(
            row[0:3], row[3:6], row[6:9], row[9:12]
        )
        shoulders[frame] = shoulder
        elbows[frame] = elbow
        wrists[frame] = wrist
        hand_orientations[frame] = hand

    return HumanTrajectory(shoulders, elbows, wrists, hand_orientations)
