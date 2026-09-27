"""Probe twelve low-wrist failures with independent bounded and relaxed 7D IK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sew_mimic.config import CONFIG  # noqa: E402
from sew_mimic.pipeline.tiago import prepare_tiago_trajectory  # noqa: E402
from sew_mimic.tiago.oracle import TiagoNumericalOracle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--baseline-results", type=Path,
                        default=ROOT / "output" / "tiago_yaw_bite2" / "yaw_000" / "placement_frames.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output" / "tiago_low_wrist_bite2_limits")
    parser.add_argument("--physical-seeds", type=int, default=128)
    parser.add_argument("--relaxed-seeds", type=int, default=64)
    args = parser.parse_args(argv)
    if args.physical_seeds < 1 or args.relaxed_seeds < 1:
        parser.error("both search budgets must be positive")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if CONFIG["tiago"]["placement"]["j1_offset_world_m"] != [0.0, 0.0, -0.20]:
        parser.error("this diagnostic requires the unchanged [0,0,-0.20] m placement")

    baseline = pd.read_csv(args.baseline_results)
    if (len(baseline) != 301 or baseline["frame"].tolist() != list(range(246, 547)) or
        not (baseline["base_yaw_deg"] == 0).all() or baseline["joint_limit_aware"].any()):
        parser.error("baseline results must be the original yaw=0, frames 246-546 run")
    selected = []
    for event in ("transfer", "withdrawal"):
        candidates = baseline.loc[(baseline["event"] == event) &
                                  (baseline["wrist_height_phase"] == "low") &
                                  (baseline["status"] != "SUCCESS_EXACT")]
        if len(candidates) < 6:
            parser.error(f"fewer than six low-wrist failures in {event}")
        selected.extend(candidates.iloc[np.linspace(0, len(candidates) - 1, 6, dtype=int)]["frame"].tolist())
    selected.sort()
    print(f"selected low-wrist failed frames: {selected}", flush=True)

    trajectory = prepare_tiago_trajectory(args.input, start_frame=246, max_frames=301)
    frames = {frame.frame: frame for frame in trajectory.frames}
    oracle = TiagoNumericalOracle()
    joint_names = CONFIG["tiago"]["joint_names"]
    relaxed_lower = np.full(7, -2 * np.pi)
    relaxed_upper = np.full(7, 2 * np.pi)
    details = []
    for frame_id in selected:
        frame = frames[frame_id]
        target = trajectory.geometry.target(frame.target)
        started = time.perf_counter()
        physical_q, physical_best = oracle._search(
            target, oracle.limits[:, 0], oracle.limits[:, 1],
            seeds=args.physical_seeds, seed=20261007)
        physical_time_s = time.perf_counter() - started
        started = time.perf_counter()
        relaxed_q, relaxed_best = oracle._search(
            target, relaxed_lower, relaxed_upper,
            seeds=args.relaxed_seeds, seed=20261008)
        relaxed_time_s = time.perf_counter() - started

        def errors(q):
            if q is None:
                return None
            position, orientation, sew = oracle._errors(q, target)
            assert position < 0.001 and orientation < np.deg2rad(1) and sew < np.deg2rad(1)
            return {"wrist_position_mm": position * 1000,
                    "gripper_orientation_deg": np.rad2deg(orientation),
                    "stereo_sew_deg": np.rad2deg(sew)}

        physical_errors = errors(physical_q)
        relaxed_errors = errors(relaxed_q)
        violations = {}
        if relaxed_q is not None:
            excess = np.maximum.reduce((oracle.limits[:, 0] - relaxed_q,
                                        relaxed_q - oracle.limits[:, 1],
                                        np.zeros(7)))
            violations = {name: float(np.rad2deg(value))
                          for name, value in zip(joint_names, excess) if value > 1e-8}
        prior = baseline.loc[baseline["frame"] == frame_id].iloc[0]
        detail = {
            "frame": frame_id, "event": prior["event"],
            "wrist_height_world_m": float(frame.target.wrist[2]),
            "semi_analytic_status": prior["status"],
            "physical_found": physical_q is not None,
            "physical_best_residual_norm": physical_best,
            "physical_search_s": physical_time_s,
            "physical_q_rad": None if physical_q is None else physical_q.tolist(),
            "physical_errors": physical_errors,
            "relaxed_found": relaxed_q is not None,
            "relaxed_best_residual_norm": relaxed_best,
            "relaxed_search_s": relaxed_time_s,
            "relaxed_q_rad": None if relaxed_q is None else relaxed_q.tolist(),
            "relaxed_errors": relaxed_errors,
            "violations_deg": violations,
        }
        details.append(detail)
        print(f"frame {frame_id}: physical={detail['physical_found']} "
              f"relaxed={detail['relaxed_found']} violations={violations}", flush=True)

    report = {
        "input": str(args.input.resolve()),
        "baseline_results": str(args.baseline_results.resolve()),
        "frames": selected,
        "robot_offset_world_m": CONFIG["tiago"]["placement"]["j1_offset_world_m"],
        "base_yaw_deg": 0,
        "task_frames": CONFIG["tiago"]["task_frames"],
        "sew_selection": CONFIG["tiago"]["sew"]["selected"],
        "physical_seeds": args.physical_seeds,
        "relaxed_seeds": args.relaxed_seeds,
        "relaxed_bounds_rad": [-2 * np.pi, 2 * np.pi],
        "physical_found_count": sum(item["physical_found"] for item in details),
        "relaxed_found_count": sum(item["relaxed_found"] for item in details),
        "details": details,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = [{"frame": item["frame"], "event": item["event"],
             "wrist_height_world_m": item["wrist_height_world_m"],
             "physical_found": item["physical_found"],
             "relaxed_found": item["relaxed_found"],
             "violated_joints_deg": "; ".join(f"{name} {amount:.2f}"
                                            for name, amount in item["violations_deg"].items()),
             "semi_analytic_status": item["semi_analytic_status"]}
            for item in details]
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    print(f"physical {report['physical_found_count']}/{len(selected)}, "
          f"relaxed {report['relaxed_found_count']}/{len(selected)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
