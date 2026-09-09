"""Deterministic identification of fixed virtual WARP skeletons for Gen3.

This module is diagnostic only: it never changes the production Gen3 or WARP
paths.  It makes the fixed-link assumption explicit and validates it on an
independent configuration set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common.evaluation import gen3_end_effector_pose
from ..kinematics import Gen3Kinematics
from ..sew.gen3_geometry import Gen3StereoSewGeometry, sample_gen3_configurations


Vector = NDArray[np.float64]
EXACT_FIXED_SKELETON_TOLERANCE_M = 1e-10
PRACTICAL_APPROXIMATE_SKELETON_TOLERANCE_M = 1e-3


def _readonly_vector(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return np.frombuffer(vector.tobytes(), dtype=np.float64)


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


@dataclass(frozen=True)
class ResidualStatistics:
    mean_m: float
    median_m: float
    p95_m: float
    maximum_m: float


@dataclass(frozen=True)
class FixedSkeletonParameters:
    shoulder: Vector
    upper_arm_length: float
    forearm_length: float
    wrist_to_task: Vector

    def __post_init__(self) -> None:
        object.__setattr__(self, "shoulder", _readonly_vector(self.shoulder, "shoulder"))
        object.__setattr__(self, "wrist_to_task", _readonly_vector(self.wrist_to_task, "wrist_to_task"))
        if not np.isfinite(self.upper_arm_length) or self.upper_arm_length <= 0.0:
            raise ValueError("upper_arm_length must be finite and positive")
        if not np.isfinite(self.forearm_length) or self.forearm_length <= 0.0:
            raise ValueError("forearm_length must be finite and positive")


@dataclass(frozen=True)
class CandidateReport:
    name: str
    parameters: FixedSkeletonParameters
    shoulder: VectorStatistics
    upper_arm_length: ScalarStatistics
    forearm_length: ScalarStatistics
    wrist_to_task: VectorStatistics
    reconstruction: ResidualStatistics
    upper_proxy_angle_rad: ScalarStatistics
    lower_proxy_angle_rad: ScalarStatistics


@dataclass(frozen=True)
class ProxyFitReport:
    parameters: FixedSkeletonParameters
    train: ResidualStatistics
    validation: ResidualStatistics


@dataclass(frozen=True)
class GeometryProxyReport:
    """A-level parameters derived from one q=0 geometry, never fitted."""
    parameters: FixedSkeletonParameters
    train: ResidualStatistics
    validation: ResidualStatistics


@dataclass(frozen=True)
class IdentificationReport:
    train_samples: int
    validation_samples: int
    train_seed: int
    validation_seed: int
    current_exact_sew: CandidateReport
    s23_e45_w67: CandidateReport
    other_candidates: tuple[CandidateReport, ...]
    s23_q1_only_variation_m: float
    s23_q2_only_variation_m: float
    s23_full_variation_m: float
    geometry_proxy: GeometryProxyReport
    proxy_fit: ProxyFitReport
    verdict: str


def _scalar(values: ArrayLike) -> ScalarStatistics:
    value = np.asarray(values, dtype=float)
    return ScalarStatistics(float(value.min()), float(value.max()), float(value.mean()),
                            float(value.std()), float(value.max() - value.min()))


def _vector(values: ArrayLike) -> VectorStatistics:
    value = np.asarray(values, dtype=float)
    mean = value.mean(axis=0)
    return VectorStatistics(value.min(axis=0), value.max(axis=0), mean, value.std(axis=0),
                            float(np.max(np.linalg.norm(value - mean, axis=1))))


def _residual(actual: ArrayLike, predicted: ArrayLike) -> ResidualStatistics:
    values = np.linalg.norm(np.asarray(actual) - np.asarray(predicted), axis=1)
    return ResidualStatistics(float(values.mean()), float(np.median(values)),
                              float(np.percentile(values, 95)), float(values.max()))


def _unit(vector: Vector) -> Vector:
    return vector / np.linalg.norm(vector)


def proxy_directions(robot: Gen3Kinematics, q: ArrayLike) -> tuple[Vector, Vector]:
    """Validated h3/h5 direction proxies in the native base frame."""
    configuration = np.asarray(q, dtype=float)
    return (
        robot.R_0_i(configuration, 3) @ robot.arm_proxy_axis(3),
        robot.R_0_i(configuration, 5) @ robot.arm_proxy_axis(5),
    )


def _candidate_points(geometry: Gen3StereoSewGeometry, q: Vector, kind: str) -> tuple[Vector, Vector, Vector]:
    points, directions = geometry.joint_axis_lines(q)
    if kind == "current S1/E45/W67":
        return points[0], points[3], points[5]
    if kind == "S23/E45/W67":
        return points[1], points[3], points[5]
    if kind == "S1/S23/E45":
        return points[0], points[1], points[3]
    if kind == "joint1-axis projection(S23)/E45/W67":
        origin, axis = points[0], directions[0]
        s23 = points[1]
        return origin + axis * float(axis @ (s23 - origin)), points[3], points[5]
    raise ValueError(f"unknown candidate {kind!r}")


def candidate_fixed_parameters(robot: Gen3Kinematics,
                               geometry: Gen3StereoSewGeometry,
                               configurations: ArrayLike,
                               kind: str) -> FixedSkeletonParameters:
    """Identify one candidate's fixed means from only the supplied samples."""
    shoulders: list[Vector] = []
    upper: list[float] = []
    lower: list[float] = []
    offsets: list[Vector] = []
    for q in np.asarray(configurations, dtype=float):
        shoulder, elbow, wrist = _candidate_points(geometry, q, kind)
        pinch, hand = gen3_end_effector_pose(q, robot)
        shoulders.append(shoulder)
        upper.append(float(np.linalg.norm(elbow - shoulder)))
        lower.append(float(np.linalg.norm(wrist - elbow)))
        offsets.append(hand.T @ (pinch - wrist))
    return FixedSkeletonParameters(
        np.mean(shoulders, axis=0),
        float(np.mean(upper)),
        float(np.mean(lower)),
        np.mean(offsets, axis=0),
    )


def candidate_report(robot: Gen3Kinematics, geometry: Gen3StereoSewGeometry,
                     configurations: ArrayLike, kind: str,
                     parameters: FixedSkeletonParameters) -> CandidateReport:
    """Evaluate fixed candidate parameters on the supplied samples."""
    qs = np.asarray(configurations, dtype=float)
    shoulders: list[Vector] = []; upper: list[float] = []; lower: list[float] = []
    offsets: list[Vector] = []; actual: list[Vector] = []; predicted: list[Vector] = []
    upper_angles: list[float] = []; lower_angles: list[float] = []
    rows: list[tuple[Vector, Vector, Vector, Vector, Vector, Vector]] = []
    for q in qs:
        s, e, w = _candidate_points(geometry, q, kind)
        pinch, hand = gen3_end_effector_pose(q, robot)
        u, l = _unit(e - s), _unit(w - e)
        u_proxy, l_proxy = proxy_directions(robot, q)
        shoulders.append(s); upper.append(float(np.linalg.norm(e - s))); lower.append(float(np.linalg.norm(w - e)))
        offsets.append(hand.T @ (pinch - w)); actual.append(pinch)
        upper_angles.append(float(np.arccos(np.clip(u @ u_proxy, -1.0, 1.0))))
        lower_angles.append(float(np.arccos(np.clip(l @ l_proxy, -1.0, 1.0))))
        rows.append((s, u, l, hand, pinch, w))
    s_stats, upper_stats, lower_stats, offset_stats = _vector(shoulders), _scalar(upper), _scalar(lower), _vector(offsets)
    for _, u, l, hand, _, _ in rows:
        predicted.append(parameters.shoulder + parameters.upper_arm_length * u +
                         parameters.forearm_length * l + hand @ parameters.wrist_to_task)
    return CandidateReport(kind, parameters, s_stats, upper_stats, lower_stats, offset_stats,
                           _residual(actual, predicted), _scalar(upper_angles), _scalar(lower_angles))


def fit_proxy_fixed_model(robot: Gen3Kinematics, configurations: ArrayLike) -> FixedSkeletonParameters:
    """Fit exactly one globally fixed S, positive link lengths, and p_WT."""
    qs = np.asarray(configurations, dtype=float)
    if qs.ndim != 2 or qs.shape[0] < 4 or qs.shape[1] != 7:
        raise ValueError("configurations must have shape (N>=4, 7)")
    design = np.zeros((3 * len(qs), 8)); target = np.zeros(3 * len(qs))
    for index, q in enumerate(qs):
        upper, lower = proxy_directions(robot, q)
        pinch, hand = gen3_end_effector_pose(q, robot)
        row = slice(3 * index, 3 * index + 3)
        design[row, :3] = np.eye(3)
        design[row, 3] = upper
        design[row, 4] = lower
        design[row, 5:] = hand
        target[row] = pinch
    fitted, *_ = np.linalg.lstsq(design, target, rcond=None)
    if fitted[3] <= 0.0 or fitted[4] <= 0.0:
        raise RuntimeError("unconstrained fixed proxy fit produced non-positive link length")
    return FixedSkeletonParameters(fitted[:3], float(fitted[3]), float(fitted[4]), fitted[5:])


def geometry_derived_proxy_parameters(robot: Gen3Kinematics,
                                      geometry: Gen3StereoSewGeometry) -> FixedSkeletonParameters:
    """A-level fixed proxy model derived directly from q=0 Gen3 geometry."""
    q0 = np.zeros(7)
    points, directions = geometry.joint_axis_lines(q0)
    s23 = points[1]
    joint1_origin, joint1_axis = points[0], directions[0]
    shoulder = joint1_origin + joint1_axis * float(joint1_axis @ (s23 - joint1_origin))
    pinch, hand = gen3_end_effector_pose(q0, robot)
    return FixedSkeletonParameters(shoulder, float(np.linalg.norm(geometry.P[:, 3])),
                                   float(np.linalg.norm(geometry.P[:, 5])), hand.T @ (pinch - points[5]))


def proxy_prediction(robot: Gen3Kinematics, configurations: ArrayLike,
                     parameters: FixedSkeletonParameters) -> NDArray[np.float64]:
    values = []
    for q in np.asarray(configurations, dtype=float):
        upper, lower = proxy_directions(robot, q)
        _, hand = gen3_end_effector_pose(q, robot)
        values.append(parameters.shoulder + parameters.upper_arm_length * upper +
                      parameters.forearm_length * lower + hand @ parameters.wrist_to_task)
    return np.asarray(values)


def identify_fixed_skeleton(robot: Gen3Kinematics, geometry: Gen3StereoSewGeometry, *,
                            samples: int = 1200, train_seed: int = 20260914,
                            validation_seed: int = 20260915) -> IdentificationReport:
    """Fit on one deterministic set and classify only independent validation."""
    if samples < 1000:
        raise ValueError("samples must be at least 1000")
    if train_seed == validation_seed:
        raise ValueError("train_seed and validation_seed must differ for independent validation")
    train_q = sample_gen3_configurations(robot, samples, train_seed)
    validation_q = sample_gen3_configurations(robot, samples, validation_seed)
    names = ("current S1/E45/W67", "S23/E45/W67", "S1/S23/E45",
             "joint1-axis projection(S23)/E45/W67")
    reports = tuple(
        candidate_report(
            robot,
            geometry,
            validation_q,
            name,
            candidate_fixed_parameters(robot, geometry, train_q, name),
        )
        for name in names
    )
    q0 = np.zeros(7)
    full_period = np.linspace(-np.pi, np.pi, samples, endpoint=False)
    q1 = np.repeat(q0[None, :], samples, axis=0); q1[:, 0] = full_period
    q2 = np.repeat(q0[None, :], samples, axis=0)
    q2[:, 1] = np.linspace(
        robot.joint_limits[1, 0], robot.joint_limits[1, 1], samples
    )
    def shoulder_variation(values: NDArray[np.float64]) -> float:
        origin = values.mean(axis=0)
        return float(np.max(np.linalg.norm(values - origin, axis=1)))
    s_q1 = np.asarray([geometry.joint_axis_lines(q)[0][1] for q in q1])
    s_q2 = np.asarray([geometry.joint_axis_lines(q)[0][1] for q in q2])
    s_full = np.asarray([geometry.joint_axis_lines(q)[0][1] for q in validation_q])
    train_actual = np.asarray([gen3_end_effector_pose(q, robot)[0] for q in train_q])
    validation_actual = np.asarray([gen3_end_effector_pose(q, robot)[0] for q in validation_q])
    geometry_parameters = geometry_derived_proxy_parameters(robot, geometry)
    geometry_proxy = GeometryProxyReport(
        geometry_parameters,
        _residual(train_actual, proxy_prediction(robot, train_q, geometry_parameters)),
        _residual(validation_actual, proxy_prediction(robot, validation_q, geometry_parameters)),
    )
    parameters = fit_proxy_fixed_model(robot, train_q)
    fit = ProxyFitReport(parameters, _residual(train_actual, proxy_prediction(robot, train_q, parameters)),
                         _residual(validation_actual, proxy_prediction(robot, validation_q, parameters)))
    maximum = fit.validation.maximum_m
    verdict = ("EXACT_FIXED_SKELETON" if maximum <= EXACT_FIXED_SKELETON_TOLERANCE_M else
               "PRACTICAL_APPROXIMATE_SKELETON" if maximum <= PRACTICAL_APPROXIMATE_SKELETON_TOLERANCE_M else
               "NO_USEFUL_FIXED_SKELETON")
    return IdentificationReport(samples, samples, train_seed, validation_seed, reports[0], reports[1], reports[2:],
                                shoulder_variation(s_q1), shoulder_variation(s_q2), shoulder_variation(s_full),
                                geometry_proxy, fit, verdict)
