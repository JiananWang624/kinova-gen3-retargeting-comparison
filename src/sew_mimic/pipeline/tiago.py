"""Fixed calibrated body-world to TIAGo-base trajectory and evaluation path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from ..angles import angular_difference
from ..common import HumanArmTarget, SolverResult, SolverStatus, compute_human_task_point
from ..config import CONFIG
from ..csv_adapter import HumanCSVAdapter, load_human_trajectory_csv
from ..sew.legacy_adapter import solve_legacy_sew_mimic
from ..tiago import TiagoKinematics, TiagoNumericalSolver, TiagoSewGeometry, TiagoSewSolver


@dataclass(frozen=True)
class TiagoFrame:
    frame: int
    target: HumanArmTarget


@dataclass(frozen=True)
class TiagoTrajectory:
    robot: TiagoKinematics
    geometry: TiagoSewGeometry
    frames: tuple[TiagoFrame, ...]


def sample_frame_indices(total_frames: int, *, start_frame: int = 0,
                         max_frames: int | None = None, stride: int = 1) -> tuple[int, ...]:
    if total_frames < 1 or start_frame < 0 or start_frame >= total_frames or stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("invalid TIAGo frame selection")
    selected = tuple(range(start_frame, total_frames, stride))
    return selected if max_frames is None else selected[:max_frames]


def fixed_body_to_base_transform() -> tuple[np.ndarray, np.ndarray]:
    calibration = CONFIG["tiago"]["calibration"]
    rotation = np.asarray(calibration["R_base_from_body"], dtype=float)
    calibrated = np.asarray(calibration["t_calibrated_m"], dtype=float)
    offset = np.asarray(calibration["user_xyz_offset_base_m"], dtype=float)
    translation = np.asarray(calibration["t_base_from_body_m"], dtype=float)
    shoulder_body = np.asarray(calibration["shoulder_reference_body_m"], dtype=float)
    shoulder_base = np.asarray(calibration["shoulder_reference_base_m"], dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-12, rtol=0):
        raise ValueError("TIAGo R_base_from_body must be a proper 3x3 rotation")
    for name, value in (("t_calibrated", calibrated), ("user_xyz_offset", offset),
                        ("t_base_from_body", translation), ("shoulder_body", shoulder_body),
                        ("shoulder_base", shoulder_base)):
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"TIAGo {name} must be a finite length-3 vector")
    if not np.allclose(calibrated, shoulder_base - rotation @ shoulder_body, atol=1e-10, rtol=0):
        raise ValueError("TIAGo t_calibrated differs from fixed shoulder calibration")
    if not np.allclose(translation, calibrated + offset, atol=1e-10, rtol=0):
        raise ValueError("TIAGo t_base_from_body differs from calibrated plus user offset")
    return rotation, translation


def prepare_tiago_trajectory(input_path: str | Path, *, start_frame: int = 0,
                             max_frames: int | None = None, stride: int = 1,
                             adapter: HumanCSVAdapter | None = None) -> TiagoTrajectory:
    human = load_human_trajectory_csv(input_path, adapter)
    rotation, translation = fixed_body_to_base_transform()
    points = np.stack((human.shoulders, human.elbows, human.wrists), axis=1)
    points_base = np.einsum("ij,tkj->tki", rotation, points) + translation
    hand_base = rotation @ human.hand_orientations
    indices = sample_frame_indices(len(human), start_frame=start_frame,
                                   max_frames=max_frames, stride=stride)
    frames = []
    task = CONFIG["task_point"]
    for index in indices:
        shoulder, elbow, wrist = points_base[index]
        hand = hand_base[index]
        frames.append(TiagoFrame(index, HumanArmTarget(
            shoulder, elbow, wrist, hand,
            compute_human_task_point(wrist, hand, mode=task["mode"],
                                     human_wrist_to_task_offset_m=np.array(task["human_wrist_to_task_offset_m"])),
        )))
    robot = TiagoKinematics()
    return TiagoTrajectory(robot, TiagoSewGeometry(robot), tuple(frames))


def evaluate_tiago_result(frame: TiagoFrame, result: SolverResult,
                          robot: TiagoKinematics, geometry: TiagoSewGeometry,
                          *, method: str = "tiago_sew") -> dict[str, object]:
    row: dict[str, object] = {
        "frame": frame.frame, "method": method, "status": result.status.value,
        "backend": CONFIG["tiago"]["solver"]["backend"] if method == "tiago_sew" else method,
        "model_revision": CONFIG["tiago"]["menagerie_revision"],
        "calibration_revision": CONFIG["tiago"]["calibration"]["revision"],
        "branch_id": result.diagnostics.branch_id,
        "solve_time_ms": result.diagnostics.solve_time_ms,
        "message": result.message,
    }
    for name, value in zip(robot.joint_names, result.q if result.q is not None else [None] * 7):
        row[name] = None if value is None else float(value)
    if result.q is None:
        row.update(ee_position_error_mm=None, ee_orientation_error_deg=None,
                   sew_angle_error_deg=None, joint_limit_margin_deg=None)
        return row
    position, rotation = robot.tcp_pose(result.q)
    row["ee_position_error_mm"] = float(np.linalg.norm(position - frame.target.task_point) * 1000)
    row["ee_orientation_error_deg"] = math.degrees(Rotation.from_matrix(rotation.T @ frame.target.hand_rotation).magnitude())
    row["sew_angle_error_deg"] = math.degrees(abs(angular_difference(
        geometry.psi(result.q), geometry.stereo.forward(
            frame.target.shoulder, frame.target.elbow, frame.target.wrist))))
    margin = np.min(np.minimum(result.q - robot.joint_limits[:, 0], robot.joint_limits[:, 1] - result.q))
    row["joint_limit_margin_deg"] = math.degrees(float(margin))
    return row


def run_tiago_benchmark(trajectory: TiagoTrajectory, *,
                        methods: Iterable[str] = ("tiago_sew",),
                        oracle_frames: Iterable[int] = ()) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected_methods = tuple(methods)
    if not selected_methods or not set(selected_methods) <= {"sew_mimic", "tiago_sew", "numerical_oracle"}:
        raise ValueError("unsupported TIAGo benchmark method")
    backend = CONFIG["tiago"]["solver"]["backend"]
    if "tiago_sew" not in selected_methods:
        solver = None
    elif backend == "tiago_semi_analytic":
        solver = TiagoSewSolver(trajectory.robot, trajectory.geometry)
    elif backend == "tiago_numerical":
        solver = TiagoNumericalSolver(trajectory.robot, trajectory.geometry)
    else:
        raise ValueError(f"unsupported TIAGo production backend {backend}")
    rows = []
    production_rows = []
    previous_baseline_q = np.array(CONFIG["tiago"]["home_q_rad"], dtype=float)
    selected_oracle_frames = set(oracle_frames)
    if "numerical_oracle" in selected_methods and not selected_oracle_frames:
        selected_oracle_frames = {frame.frame for frame in trajectory.frames[:10]}
    oracle = None
    if selected_oracle_frames or "numerical_oracle" in selected_methods:
        from ..tiago.oracle import TiagoNumericalOracle
        oracle = TiagoNumericalOracle()
    for frame in trajectory.frames:
        if "sew_mimic" in selected_methods:
            human = frame.target
            baseline = solve_legacy_sew_mimic(previous_baseline_q, human.shoulder,
                                              human.elbow, human.wrist,
                                              human.hand_rotation, trajectory.robot)
            if baseline.q is not None:
                previous_baseline_q = baseline.q.copy()
            rows.append(evaluate_tiago_result(frame, baseline, trajectory.robot,
                                              trajectory.geometry, method="sew_mimic"))
        if solver is None and "numerical_oracle" not in selected_methods:
            continue
        try:
            target = trajectory.geometry.target(frame.target)
            result = solver.solve(target) if solver is not None else None
        except ValueError as error:
            from ..common import SolverDiagnostics
            result = SolverResult("tiago_sew", SolverStatus.INVALID_INPUT, None,
                                  SolverDiagnostics(), str(error)) if solver is not None else None
            target = None
        check = None
        if oracle is not None and frame.frame in selected_oracle_frames and target is not None:
            check = oracle.solve(target)
        if result is not None:
            row = evaluate_tiago_result(frame, result, trajectory.robot, trajectory.geometry)
            if result.q is not None:
                row["classification"] = "reachable_success"
            elif check is not None:
                row["classification"] = "solver_miss" if check.classification == "reachable" else check.classification
                row["oracle_reason"] = check.reason
                row["oracle_best_residual_norm"] = check.best_residual_norm
            else:
                row["classification"] = "oracle_inconclusive"
            rows.append(row)
            production_rows.append(row)
        if "numerical_oracle" in selected_methods and check is not None:
            from ..common import SolverDiagnostics
            oracle_status = SolverStatus.SUCCESS_EXACT if check.classification == "reachable" else SolverStatus.NUMERICAL_FAILURE
            oracle_result = SolverResult("numerical_oracle", oracle_status,
                                         check.q if oracle_status == SolverStatus.SUCCESS_EXACT else None,
                                         SolverDiagnostics(), check.reason)
            oracle_row = evaluate_tiago_result(frame, oracle_result, trajectory.robot,
                                               trajectory.geometry, method="numerical_oracle")
            oracle_row["classification"] = check.classification
            rows.append(oracle_row)
    times = [row["solve_time_ms"] for row in production_rows if row["solve_time_ms"] is not None]
    success = sum(row["status"] == SolverStatus.SUCCESS_EXACT.value for row in production_rows)
    categories = ("reachable_success", "solver_miss", "joint_limit_failure",
                  "workspace_unreachable", "oracle_inconclusive")
    summary = {"frames": len(production_rows), "baseline_rows": sum(row["method"] == "sew_mimic" for row in rows),
               "oracle_rows": sum(row["method"] == "numerical_oracle" for row in rows),
               **{name: sum(row["classification"] == name for row in production_rows)
                                     for name in categories},
               "mean_ms": float(np.mean(times)) if times else None,
               "median_ms": float(np.median(times)) if times else None,
               "p95_ms": float(np.percentile(times, 95)) if times else None}
    return rows, summary
