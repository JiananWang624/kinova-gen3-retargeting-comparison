"""Bounded, fixed-calibration J1/SEW-S=J1 validation on one continuous CSV window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sew_mimic.config import CONFIG  # noqa: E402
from sew_mimic.pipeline.tiago import prepare_tiago_trajectory  # noqa: E402
from sew_mimic.tiago.oracle import TiagoNumericalOracle  # noqa: E402
from sew_mimic.tiago.solver import TiagoSewSolver  # noqa: E402
from sew_mimic.tiago.wrist import wrist_center  # noqa: E402


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"p50": float(np.quantile(values, 0.5)),
            "p95": float(np.quantile(values, 0.95))}


def _sew_margin(points) -> float:
    sw = (points.wrist - points.shoulder)
    se = (points.elbow - points.shoulder)
    sw /= np.linalg.norm(sw)
    se /= np.linalg.norm(se)
    sew = CONFIG["tiago"]["sew"]
    pole = np.linalg.norm(np.cross(sw - np.asarray(sew["e_t"]),
                                    np.asarray(sew["e_r"]))) / 2
    return float(min(pole, np.linalg.norm(np.cross(sw, se))))


def _position_error(robot, target: np.ndarray, seed: np.ndarray) -> float:
    lower, upper = robot.joint_limits.T
    fit = least_squares(lambda q: wrist_center(robot, q) - target,
                        np.clip(seed, lower + 1e-9, upper - 1e-9),
                        bounds=(lower, upper), max_nfev=100,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    return float(np.linalg.norm(fit.fun))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--start", type=int, default=640)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--extra-frames", type=int, nargs="*", default=[1677, 2967])
    parser.add_argument("--oracle-seeds", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output" / "tiago_j1_validation" / "report.json")
    args = parser.parse_args()
    if not 50 <= args.frames <= 100 or args.oracle_seeds < 1:
        parser.error("use 50–100 contiguous frames and at least one oracle seed")
    trajectory = prepare_tiago_trajectory(args.input)
    if (args.start < 0 or args.start + args.frames > len(trajectory.frames) or
        any(index < 0 or index >= len(trajectory.frames) for index in args.extra_frames)):
        parser.error("requested frames are outside the input trajectory")
    robot, geometry = trajectory.robot, trajectory.geometry
    calibration = CONFIG["tiago"]["calibration"]
    if (calibration["reference_joint_name"] != "arm_1_joint" or
        tuple(CONFIG["tiago"]["sew"]["selected"]) != ("J1", "J4", "wrist_center") or
        CONFIG["tiago"]["solver"]["backend"] != "tiago_semi_analytic"):
        raise ValueError("J1 validation requires fixed J1 calibration with configured offset, S=J1, and semi-analytic backend")
    root = robot.axes_and_anchors(np.asarray(CONFIG["tiago"]["home_q_rad"]))[1][0]
    shoulder = np.median([frame.target.shoulder for frame in trajectory.frames], axis=0)
    offset = np.asarray(CONFIG["tiago"]["placement"]["j1_offset_world_m"], dtype=float)
    alignment_error = float(np.linalg.norm(root - offset - shoulder))
    if alignment_error > 1e-6:
        raise ValueError(f"robust shoulder differs from nominal J1 root: {alignment_error} m")

    indices = list(range(args.start, args.start + args.frames))
    extras = sorted(set(args.extra_frames) - set(indices))
    oracle = TiagoNumericalOracle()
    solver = TiagoSewSolver(robot, geometry)
    rng = np.random.default_rng(20260929)
    lower, upper = robot.joint_limits.T
    samples = rng.uniform(lower, upper, size=(10000, 7))
    workspace = np.array([wrist_center(robot, q) for q in samples])
    tree = cKDTree(workspace)
    rows = []
    for ordinal, index in enumerate(indices + extras, 1):
        frame = trajectory.frames[index]
        target = geometry.target(frame.target)
        _, nearest = tree.query(target.position)
        position_error = _position_error(robot, target.position, samples[nearest])
        result = solver.solve(target)
        check = oracle.solve(target, seeds=args.oracle_seeds, seed=20260929 + index)
        strict_errors = None if result.q is None else oracle._errors(result.q, target)
        if strict_errors is not None and not (
            strict_errors[0] < 0.001 and
            strict_errors[1] < np.deg2rad(1) and
            strict_errors[2] < np.deg2rad(1)):
            raise ValueError(f"frame {index} passed solver but failed independent MuJoCo strict FK")
        if result.q is None and check.classification == "reachable":
            print(f"solver miss at frame {index}", flush=True)
        points = robot.sew_points(result.q) if result.q is not None else None
        rows.append({
            "frame": index,
            "in_continuous_window": index in range(args.start, args.start + args.frames),
            "position_only_error_mm": position_error * 1000,
            "oracle_classification_raw": check.classification,
            "oracle_reason": check.reason,
            "strict_reachability_evidence": "reachable" if result.q is not None or
                check.classification == "reachable" else check.classification,
            "oracle_search_miss_but_semi_success":
                result.q is not None and check.classification != "reachable",
            "semi_status": result.status.value,
            "semi_branch": result.diagnostics.branch_id,
            "semi_q_rad": None if result.q is None else result.q.tolist(),
            "semi_solver_miss": result.q is None and check.classification == "reachable",
            "strict_error_mm_deg_deg": None if strict_errors is None else
                [strict_errors[0] * 1000, np.rad2deg(strict_errors[1]),
                 np.rad2deg(strict_errors[2])],
            "joint_limit_margin_deg": None if result.q is None else
                float(np.rad2deg(result.diagnostics.joint_limit_margin_rad)),
            "robot_sew_singularity_margin": None if points is None else _sew_margin(points),
            "human_sew_singularity_margin": _sew_margin(frame.target),
            "solve_time_ms": float(result.diagnostics.solve_time_ms),
        })
        print(f"{ordinal}/{len(indices) + len(extras)} frame {index}: "
              f"{check.classification}, {result.status.value}", flush=True)

    continuous = rows[:len(indices)]
    strict = [row for row in continuous if row["semi_q_rad"] is not None]
    steps = []
    for left, right in zip(continuous[:-1], continuous[1:]):
        if left["semi_q_rad"] is None or right["semi_q_rad"] is None:
            continue
        difference = (np.asarray(right["semi_q_rad"]) - np.asarray(left["semi_q_rad"])
                      + np.pi) % (2 * np.pi) - np.pi
        steps.append(float(np.rad2deg(np.max(np.abs(difference)))))
    report = {
        "model_revision": CONFIG["tiago"]["menagerie_revision"],
        "calibration_revision": calibration["revision"],
        "sew_definition": CONFIG["tiago"]["sew"]["selected"],
        "task_frames": CONFIG["tiago"]["task_frames"],
        "window_start": args.start, "window_frames": args.frames,
        "extra_frames": extras, "oracle_seeds_per_frame": args.oracle_seeds,
        "semi_recovery_seeds": CONFIG["tiago"]["solver"]["recovery_seeds"],
        "robust_shoulder_to_j1_m": float(np.linalg.norm(root - shoulder)),
        "fixed_offset_consistency_error_m": alignment_error,
        "position_only_within_1mm": sum(row["position_only_error_mm"] < 1 for row in continuous),
        "position_only_error_mm": _quantiles([row["position_only_error_mm"] for row in continuous]),
        "oracle_strict_reachable": sum(row["oracle_classification_raw"] == "reachable" for row in continuous),
        "strict_reachable_union": sum(row["strict_reachability_evidence"] == "reachable" for row in continuous),
        "oracle_search_miss_but_semi_success": sum(
            row["oracle_search_miss_but_semi_success"] for row in continuous),
        "semi_strict_success": len(strict),
        "semi_solver_miss": sum(row["semi_solver_miss"] for row in continuous),
        "oracle_failure_caveat":
            "finite multistart failure cannot prove physical unreachability; solver strict FK success is reachable evidence",
        "joint_limit_margin_deg": _quantiles([row["joint_limit_margin_deg"] for row in strict]),
        "robot_sew_singularity_margin": _quantiles([
            row["robot_sew_singularity_margin"] for row in strict]),
        "human_sew_singularity_margin": _quantiles([
            row["human_sew_singularity_margin"] for row in continuous]),
        "adjacent_strict_solution_pairs": len(steps),
        "max_wrapped_joint_step_deg": _quantiles(steps),
        "semi_solve_time_ms": _quantiles([row["solve_time_ms"] for row in continuous]),
        "semi_success_solve_time_ms": _quantiles([row["solve_time_ms"] for row in strict]),
        "strict_error_max_mm_deg_deg": None if not strict else
            np.max([row["strict_error_mm_deg_deg"] for row in strict], axis=0).tolist(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"J1 validation: {args.output}")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
