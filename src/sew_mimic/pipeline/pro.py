"""Fixed-placement human trajectory and independent Pro MuJoCo evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..angles import angular_difference
from ..common import HumanArmTarget, SolverResult, SolverStatus
from ..config import CONFIG
from ..csv_adapter import load_human_trajectory_csv
from ..pro.model import ProKinematics
from ..pro.solver import ProSewGeometry, ProSewSolver


@dataclass(frozen=True)
class ProFrame:
    frame: int
    target: HumanArmTarget


@dataclass(frozen=True)
class ProTrajectory:
    robot: ProKinematics
    geometry: ProSewGeometry
    frames: tuple[ProFrame, ...]


def prepare_pro_trajectory(input_path: str | Path, indices: list[int] | range) -> ProTrajectory:
    human = load_human_trajectory_csv(input_path)
    calibration = CONFIG["tiago_pro"]["calibration"]
    rotation = np.asarray(calibration["R_world_from_body"], dtype=float)
    translation = np.asarray(calibration["t_world_from_body_m"], dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,) or
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12) or
        not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)):
        raise ValueError("invalid fixed Pro body-to-world calibration")
    if not indices or any(i < 0 or i >= len(human) for i in indices):
        raise ValueError("invalid Pro frame selection")
    if CONFIG["task_point"] != {"mode": "wrist", "human_wrist_to_task_offset_m": [0.0, 0.0, 0.0]}:
        raise ValueError("Pro task requires Human Wrist_XYZ without palm offset")
    frames = []
    for index in indices:
        shoulder = rotation @ human.shoulders[index] + translation
        elbow = rotation @ human.elbows[index] + translation
        wrist = rotation @ human.wrists[index] + translation
        hand = rotation @ human.hand_orientations[index]
        frames.append(ProFrame(index, HumanArmTarget(shoulder, elbow, wrist, hand, wrist)))
    robot = ProKinematics()
    return ProTrajectory(robot, ProSewGeometry(robot), tuple(frames))


def evaluate_pro_result(frame: ProFrame, result: SolverResult,
                        robot: ProKinematics, geometry: ProSewGeometry) -> dict[str, object]:
    row: dict[str, object] = {
        "frame": frame.frame,
        "method": result.method,
        "status": result.status.value,
        "backend": CONFIG["tiago_pro"]["solver"]["backend"],
        "calibration_revision": CONFIG["tiago_pro"]["calibration"]["revision"],
        "position_frame": CONFIG["tiago_pro"]["task_frames"]["position"],
        "orientation_frame": CONFIG["tiago_pro"]["task_frames"]["orientation"],
        "solve_time_ms": result.diagnostics.solve_time_ms,
        "iterations": result.diagnostics.metadata.get("iterations"),
        "attempts": result.diagnostics.metadata.get("attempts"),
        "message": result.message,
    }
    for name, value in zip(robot.joint_names, result.q if result.q is not None else [None] * 7):
        row[name] = None if value is None else float(value)
    if result.q is None:
        row.update(position_error_mm=None, orientation_error_deg=None, sew_error_deg=None,
                   joint_limit_margin_deg=None, j5_margin_deg=None, j6_margin_deg=None,
                   j7_margin_deg=None)
        return row
    position, orientation = robot.task_poses(result.q)
    margins = np.minimum(result.q - robot.joint_limits[:, 0], robot.joint_limits[:, 1] - result.q)
    row.update(
        position_error_mm=float(np.linalg.norm(position - frame.target.wrist) * 1000),
        orientation_error_deg=math.degrees(Rotation.from_matrix(
            frame.target.hand_rotation @ orientation.T).magnitude()),
        sew_error_deg=math.degrees(abs(angular_difference(
            geometry.psi(result.q), geometry.stereo.forward(
                frame.target.shoulder, frame.target.elbow, frame.target.wrist)))),
        joint_limit_margin_deg=math.degrees(float(np.min(margins))),
        j5_margin_deg=math.degrees(float(margins[4])),
        j6_margin_deg=math.degrees(float(margins[5])),
        j7_margin_deg=math.degrees(float(margins[6])),
    )
    return row


def run_pro_trajectory(trajectory: ProTrajectory, *, reset_each_frame: bool = False) -> tuple[list[dict[str, object]], dict[str, object]]:
    solver = ProSewSolver(trajectory.robot, trajectory.geometry)
    rows: list[dict[str, object]] = []
    previous_q: np.ndarray | None = None
    joint_steps = []
    longest = current = 0
    for frame in trajectory.frames:
        if reset_each_frame:
            solver.reset()
        try:
            result = solver.solve(trajectory.geometry.target(frame.target))
        except ValueError as error:
            from ..common import SolverDiagnostics
            result = SolverResult(solver.method, SolverStatus.INVALID_INPUT, None,
                                  SolverDiagnostics(), str(error))
        row = evaluate_pro_result(frame, result, trajectory.robot, trajectory.geometry)
        row["classification"] = "reachable_success" if result.q is not None else "oracle_inconclusive"
        rows.append(row)
        if result.q is not None:
            current += 1
            longest = max(longest, current)
            if previous_q is not None and not reset_each_frame:
                joint_steps.append(float(np.linalg.norm(result.q - previous_q)))
            previous_q = result.q
        else:
            current = 0
            previous_q = None
    times = [float(row["solve_time_ms"]) for row in rows if row["solve_time_ms"] is not None]
    summary = {
        "frames": len(rows),
        "strict_success": sum(row["status"] == SolverStatus.SUCCESS_EXACT.value for row in rows),
        "solver_failure_without_oracle_proof": sum(row["status"] != SolverStatus.SUCCESS_EXACT.value for row in rows),
        "p50_ms": float(np.percentile(times, 50)) if times else None,
        "p95_ms": float(np.percentile(times, 95)) if times else None,
        "iterations_p50": float(np.percentile([row["iterations"] for row in rows if row["iterations"] is not None], 50)) if times else None,
        "iterations_p95": float(np.percentile([row["iterations"] for row in rows if row["iterations"] is not None], 95)) if times else None,
        "longest_success_segment": longest,
        "joint_step_p95_rad": float(np.percentile(joint_steps, 95)) if joint_steps else None,
    }
    return rows, summary
