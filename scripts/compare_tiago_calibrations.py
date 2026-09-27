"""Compare two fixed TIAGo shoulder calibrations on bounded representative frames.

This is a diagnostic only: it never changes config, robot base, or any runtime
calibration. Both placements receive identical orientations, SEW definitions,
joint limits, solver settings, and deterministic oracle search budgets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sew_mimic.config import CONFIG  # noqa: E402
from sew_mimic.pipeline.tiago import prepare_tiago_trajectory  # noqa: E402
from sew_mimic.tiago.oracle import TiagoNumericalOracle  # noqa: E402
from sew_mimic.tiago.solver import TiagoSewSolver, TiagoSewTarget  # noqa: E402


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values)
    return {"p50": float(np.quantile(array, 0.5)),
            "p95": float(np.quantile(array, 0.95))}


def _sew_margin(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """Dimensionless minimum of Stereo-SEW pole and elbow collinearity margins."""
    e_sw = (wrist - shoulder) / np.linalg.norm(wrist - shoulder)
    e_se = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
    sew = CONFIG["tiago"]["sew"]
    pole = np.linalg.norm(np.cross(e_sw - np.asarray(sew["e_t"]),
                                    np.asarray(sew["e_r"]))) / 2
    elbow_margin = np.linalg.norm(np.cross(e_sw, e_se))
    return float(min(pole, elbow_margin))


def _continuity(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    steps = []
    max_joint_steps = []
    for before, after in zip(rows[:-1], rows[1:]):
        if before[field] is None or after[field] is None:
            continue
        q0, q1 = np.asarray(before[field]), np.asarray(after[field])
        change = (q1 - q0 + np.pi) % (2 * np.pi) - np.pi
        frame_gap = int(after["frame"]) - int(before["frame"])
        steps.append(float(np.linalg.norm(change) / frame_gap))
        max_joint_steps.append(float(np.max(np.abs(change)) / frame_gap))
    return {"adjacent_solved_pairs": len(steps),
            "wrapped_joint_l2_rad_per_frame": _quantiles(steps),
            "max_wrapped_joint_deg_per_frame":
                _quantiles(np.rad2deg(max_joint_steps).tolist())}


def _case(trajectory, indices: list[int], offset: np.ndarray, *, name: str,
          oracle_seeds: int, recovery_seeds: int) -> dict[str, object]:
    robot, geometry = trajectory.robot, trajectory.geometry
    oracle = TiagoNumericalOracle()
    solver = TiagoSewSolver(robot, geometry)
    solver.settings = dict(solver.settings)
    solver.settings["recovery_seeds"] = recovery_seeds
    rows = []
    for ordinal, index in enumerate(indices, 1):
        frame = trajectory.frames[index]
        base_target = geometry.target(frame.target)
        target = TiagoSewTarget(base_target.position + offset,
                                base_target.rotation, base_target.psi)
        oracle_result = oracle.solve(target, seeds=oracle_seeds, seed=20260929 + index)
        result = solver.solve(target)
        oracle_q = oracle_result.q if oracle_result.classification == "reachable" else None
        robot_margin = None
        joint_margin_deg = None
        if oracle_q is not None:
            points = robot.sew_points(oracle_q)
            robot_margin = _sew_margin(points.shoulder, points.elbow, points.wrist)
            joint_margin_deg = float(np.rad2deg(np.min(np.minimum(
                oracle_q - robot.joint_limits[:, 0],
                robot.joint_limits[:, 1] - oracle_q))))
        human = frame.target
        semi_points = robot.sew_points(result.q) if result.q is not None else None
        rows.append({
            "frame": index,
            "oracle_classification": oracle_result.classification,
            "oracle_reason": oracle_result.reason,
            "oracle_q_rad": None if oracle_q is None else oracle_q.tolist(),
            "oracle_joint_limit_margin_deg": joint_margin_deg,
            "oracle_robot_sew_singularity_margin": robot_margin,
            "human_sew_singularity_margin": _sew_margin(
                human.shoulder, human.elbow, human.wrist),
            "semi_status": result.status.value,
            "semi_q_rad": None if result.q is None else result.q.tolist(),
            "semi_branch": result.diagnostics.branch_id,
            "semi_solver_miss": oracle_q is not None and result.q is None,
            "oracle_search_miss_but_semi_success": oracle_q is None and result.q is not None,
            "semi_joint_limit_margin_deg": None if result.q is None else
                float(np.rad2deg(result.diagnostics.joint_limit_margin_rad)),
            "semi_robot_sew_singularity_margin": None if result.q is None else
                _sew_margin(semi_points.shoulder, semi_points.elbow, semi_points.wrist),
            "semi_solve_time_ms": float(result.diagnostics.solve_time_ms),
        })
        print(f"{name}: {ordinal}/{len(indices)} frame {index}: "
              f"oracle={oracle_result.classification}, semi={result.status.value}", flush=True)

    oracle_reachable = [row for row in rows if row["oracle_classification"] == "reachable"]
    joint_margins = [row["oracle_joint_limit_margin_deg"] for row in oracle_reachable]
    robot_sew_margins = [row["oracle_robot_sew_singularity_margin"] for row in oracle_reachable]
    return {
        "name": name,
        "fixed_translation_delta_base_m": offset.tolist(),
        "candidate_t_base_from_body_m":
            (np.asarray(CONFIG["tiago"]["calibration"]["t_base_from_body_m"]) + offset).tolist(),
        "frames": len(rows),
        "oracle_reachable": len(oracle_reachable),
        "oracle_joint_limit_failure": sum(row["oracle_classification"] == "joint_limit_failure" for row in rows),
        "oracle_inconclusive": sum(row["oracle_classification"] == "oracle_inconclusive" for row in rows),
        "semi_success": sum(row["semi_q_rad"] is not None for row in rows),
        "semi_solver_miss": sum(row["semi_solver_miss"] for row in rows),
        "oracle_search_miss_but_semi_success":
            sum(row["oracle_search_miss_but_semi_success"] for row in rows),
        "oracle_joint_limit_margin_deg": _quantiles(joint_margins),
        "oracle_robot_sew_singularity_margin": _quantiles(robot_sew_margins),
        "human_sew_singularity_margin":
            _quantiles([row["human_sew_singularity_margin"] for row in rows]),
        "oracle_trajectory_continuity": _continuity(rows, "oracle_q_rad"),
        "semi_trajectory_continuity": _continuity(rows, "semi_q_rad"),
        "semi_solve_time_ms": _quantiles([row["semi_solve_time_ms"] for row in rows]),
        "semi_reachable_solve_time_ms": _quantiles([
            row["semi_solve_time_ms"] for row in oracle_reachable]),
        "rows": rows,
    }


def _local_continuity(trajectory, offset: np.ndarray, recovery_seeds: int) -> dict[str, object]:
    """Check adjacent frames around boundary and candidate-specific strict successes."""
    centers = (620, 688, 1152, 1677, 2967, 3899)
    solver = TiagoSewSolver(trajectory.robot, trajectory.geometry)
    solver.settings = dict(solver.settings)
    solver.settings["recovery_seeds"] = recovery_seeds
    joint_steps, max_steps, windows = [], [], []
    for center in centers:
        solver.reset()
        previous = None
        solved = 0
        local_pairs = 0
        for index in range(center - 2, center + 3):
            original = trajectory.geometry.target(trajectory.frames[index].target)
            target = TiagoSewTarget(original.position + offset, original.rotation, original.psi)
            result = solver.solve(target)
            if result.q is not None:
                solved += 1
            if previous is not None and result.q is not None:
                change = (result.q - previous + np.pi) % (2 * np.pi) - np.pi
                joint_steps.append(float(np.linalg.norm(change)))
                max_steps.append(float(np.rad2deg(np.max(np.abs(change)))))
                local_pairs += 1
            previous = result.q
        windows.append({"center_frame": center, "frames": 5,
                        "strict_semi_solutions": solved,
                        "adjacent_solved_pairs": local_pairs})
    return {"windows": windows, "adjacent_solved_pairs": len(joint_steps),
            "wrapped_joint_l2_rad_per_frame": _quantiles(joint_steps),
            "max_wrapped_joint_deg_per_frame": _quantiles(max_steps)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output" / "tiago_placement" / "calibration_comparison.json")
    parser.add_argument("--equidistant-frames", type=int, default=50)
    parser.add_argument("--oracle-seeds", type=int, default=32)
    parser.add_argument("--recovery-seeds", type=int, default=16)
    args = parser.parse_args()
    if min(args.equidistant_frames, args.oracle_seeds, args.recovery_seeds) < 1:
        parser.error("frame and seed counts must be positive")
    trajectory = prepare_tiago_trajectory(args.input)
    if len(trajectory.frames) <= 3901:
        parser.error("comparison requires the 4,344-frame dataset containing boundary and continuity windows")
    robot = trajectory.robot
    calibration = CONFIG["tiago"]["calibration"]
    home = np.asarray(CONFIG["tiago"]["home_q_rad"])
    anchors = robot.axes_and_anchors(home)[1]
    root, j2_home = anchors[0].copy(), anchors[1].copy()
    baseline = calibration["baseline_j1_arm_root"]
    baseline_translation = np.asarray(baseline["t_base_from_body_m"])
    active_translation = np.asarray(calibration["t_base_from_body_m"])
    baseline_offset = baseline_translation - active_translation
    j2_offset = baseline_translation + j2_home - root - active_translation
    if not np.allclose(root, baseline["shoulder_reference_base_m"], atol=1e-10, rtol=0):
        raise ValueError("preserved J1 baseline does not align with the MuJoCo J1 anchor")
    if not np.allclose(j2_home, np.asarray(calibration["shoulder_reference_base_m"]), atol=1e-10, rtol=0):
        raise ValueError("active calibration does not align with the MuJoCo J2 home anchor")
    indices = sorted(set(np.linspace(0, len(trajectory.frames) - 1,
                                         min(args.equidistant_frames, len(trajectory.frames)),
                                         dtype=int).tolist() + [688, 1677, 2967]))
    comparison = {
        "model_revision": CONFIG["tiago"]["menagerie_revision"],
        "calibration_revision": CONFIG["tiago"]["calibration"]["revision"],
        "fixed_rotation_base_from_body": CONFIG["tiago"]["calibration"]["R_base_from_body"],
        "frame_indices": indices,
        "equidistant_frames_requested": args.equidistant_frames,
        "known_boundary_frames": [688, 1677, 2967],
        "oracle_seeds_per_frame": args.oracle_seeds,
        "semi_recovery_seeds_per_frame": args.recovery_seeds,
        "j2_reference": "J2 anchor at configured home pose; candidate translation is fixed for all frames",
        "target_trajectory_continuity":
            "identical for both placements: one constant translation changes neither point increments nor SEW angles",
        "cases": [],
    }
    for name, offset in (("J1_arm_root_baseline", baseline_offset),
                         ("J2_home_anchor_candidate", j2_offset)):
        case = _case(trajectory, indices, offset, name=name,
                     oracle_seeds=args.oracle_seeds,
                     recovery_seeds=args.recovery_seeds)
        case["short_window_trajectory_continuity"] = _local_continuity(
            trajectory, offset, args.recovery_seeds)
        comparison["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"comparison: {args.output}")
    for case in comparison["cases"]:
        print(f"{case['name']}: oracle reachable {case['oracle_reachable']}, "
              f"semi misses {case['semi_solver_miss']}, "
              f"semi P50/P95 {case['semi_solve_time_ms']}")


if __name__ == "__main__":
    main()
