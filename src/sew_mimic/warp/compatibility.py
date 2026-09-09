"""Fixed-link measurements for the tested Gen3 S1/E45/W67 definition."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from ..common.evaluation import gen3_end_effector_pose
from ..kinematics import Gen3Kinematics
from ..sew.gen3_geometry import Gen3StereoSewGeometry, sample_gen3_configurations

Vector = NDArray[np.float64]
FIXED_GEOMETRY_TOLERANCE_M = 1e-10


@dataclass(frozen=True)
class ScalarStatistics:
    minimum: float
    maximum: float
    mean: float
    std: float
    variation: float


@dataclass(frozen=True)
class VectorStatistics:
    minimum: Vector
    maximum: Vector
    mean: Vector
    std: Vector
    variation: float

    def __post_init__(self) -> None:
        for name in ("minimum", "maximum", "mean", "std"):
            value = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=np.float64))


@dataclass(frozen=True)
class WarpCompatibilityReport:
    samples: int
    seed: int
    upper_arm_length: ScalarStatistics
    forearm_length: ScalarStatistics
    wrist_to_task: VectorStatistics
    wrist_to_task_max_norm_deviation: float
    tolerance_m: float
    compatible: bool


def _scalar_statistics(values: NDArray[np.float64]) -> ScalarStatistics:
    return ScalarStatistics(float(values.min()), float(values.max()), float(values.mean()), float(values.std()), float(values.max() - values.min()))


def check_warp_fixed_geometry_compatibility(
    robot: Gen3Kinematics,
    gen3_geometry: Gen3StereoSewGeometry,
    *, samples: int = 1000,
    seed: int = 20260912,
    tolerance_m: float = FIXED_GEOMETRY_TOLERANCE_M,
) -> WarpCompatibilityReport:
    """Measure one tested point definition, not every possible Gen3 WARP path."""
    if not isinstance(robot, Gen3Kinematics) or not isinstance(gen3_geometry, Gen3StereoSewGeometry):
        raise ValueError("robot and gen3_geometry must be validated Gen3 instances")
    if samples < 1 or tolerance_m < 0 or not np.isfinite(tolerance_m):
        raise ValueError("samples must be positive and tolerance_m finite nonnegative")
    configurations = sample_gen3_configurations(robot, samples, seed)
    upper, forearm, offsets = [], [], []
    for q in configurations:
        points = gen3_geometry.sew_points(q)
        pinch, hand = gen3_end_effector_pose(q, robot)
        upper.append(np.linalg.norm(points.elbow - points.shoulder))
        forearm.append(np.linalg.norm(points.wrist - points.elbow))
        offsets.append(hand.T @ (pinch - points.wrist))
    upper_values, forearm_values, offset_values = np.asarray(upper), np.asarray(forearm), np.asarray(offsets)
    offset_mean = offset_values.mean(axis=0)
    offset_stats = VectorStatistics(offset_values.min(axis=0), offset_values.max(axis=0), offset_mean, offset_values.std(axis=0), float(np.max(np.linalg.norm(offset_values - offset_mean, axis=1))))
    compatible = (_scalar_statistics(upper_values).variation <= tolerance_m and _scalar_statistics(forearm_values).variation <= tolerance_m and offset_stats.variation <= tolerance_m)
    return WarpCompatibilityReport(samples, seed, _scalar_statistics(upper_values), _scalar_statistics(forearm_values), offset_stats, offset_stats.variation, float(tolerance_m), compatible)
