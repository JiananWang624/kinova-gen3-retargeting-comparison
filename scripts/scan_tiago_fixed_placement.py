"""Run one fixed robot placement on one complete bite without an oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sew_mimic.config import CONFIG  # noqa: E402
from sew_mimic.pipeline.tiago import prepare_tiago_trajectory, run_tiago_benchmark  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--bite-id", type=int, default=2)
    parser.add_argument("--robot-offset-world-m", type=float, nargs=3, required=True,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    offset = np.asarray(args.robot_offset_world_m, dtype=float)
    if not np.all(np.isfinite(offset)):
        parser.error("robot offset must be finite")
    table = pd.read_csv(args.input, usecols=["bite_id", "event"])
    indices = np.flatnonzero(table["bite_id"].to_numpy() == args.bite_id)
    if (not len(indices) or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)) or
        set(table.iloc[indices]["event"]) != {"transfer", "withdrawal"}):
        parser.error("selected bite must be one contiguous transfer plus withdrawal trajectory")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")

    # This override is process-local; config.yaml and human calibration remain unchanged.
    CONFIG["tiago"]["placement"]["j1_offset_world_m"] = offset.tolist()
    trajectory = prepare_tiago_trajectory(args.input, start_frame=int(indices[0]),
                                          max_frames=len(indices))
    heights = np.array([frame.target.wrist[2] for frame in trajectory.frames])
    cuts = np.quantile(heights, [1 / 3, 2 / 3])
    phases = np.where(heights <= cuts[0], "low",
                      np.where(heights <= cuts[1], "middle", "high"))
    print(f"bite {args.bite_id}: frames {indices[0]}-{indices[-1]}, offset {offset.tolist()}, "
          f"wrist-height cuts {cuts.tolist()}", flush=True)
    rows, _ = run_tiago_benchmark(trajectory, methods=("tiago_sew",))
    frame_table = pd.DataFrame(rows)
    frame_table.insert(1, "event", table.iloc[indices]["event"].to_numpy())
    frame_table.insert(2, "wrist_height_phase", phases)
    frame_table.insert(3, "wrist_z_world_m", heights)
    for joint_number in (5, 7):
        joint = f"arm_{joint_number}_joint"
        low, high = trajectory.robot.joint_limits[joint_number - 1]
        frame_table[f"j{joint_number}_limit_margin_deg"] = np.rad2deg(
            np.minimum(frame_table[joint] - low, high - frame_table[joint]))
    success = frame_table["status"].eq("SUCCESS_EXACT")

    def minimum(column: str) -> float | None:
        values = frame_table.loc[success, column].dropna()
        return float(values.min()) if len(values) else None

    def median(column: str) -> float | None:
        values = frame_table.loc[success, column].dropna()
        return float(values.median()) if len(values) else None

    phase_results = {}
    for name in ("low", "middle", "high"):
        selected = phases == name
        count = int(np.sum(selected))
        solved = int(np.sum(success.to_numpy()[selected]))
        phase_results[name] = {"success": solved, "total": count,
                               "success_rate": solved / count}
    times = frame_table["solve_time_ms"].dropna().to_numpy()
    report = {
        "input": str(args.input.resolve()), "bite_id": args.bite_id,
        "frame_start": int(indices[0]), "frame_end": int(indices[-1]),
        "robot_offset_world_m": offset.tolist(),
        "model_sha256": CONFIG["tiago"]["model_sha256"],
        "calibration_revision": CONFIG["tiago"]["calibration"]["revision"],
        "solver_settings": CONFIG["tiago"]["solver"],
        "wrist_height_cut_m": cuts.tolist(),
        "strict_success": int(success.sum()), "total": len(rows),
        "solver_failure_count": int((~success).sum()),
        "phase": phase_results,
        "overall_min_margin_deg": minimum("joint_limit_margin_deg"),
        "j5_min_margin_deg": minimum("j5_limit_margin_deg"),
        "j7_min_margin_deg": minimum("j7_limit_margin_deg"),
        "overall_median_margin_deg": median("joint_limit_margin_deg"),
        "j5_median_margin_deg": median("j5_limit_margin_deg"),
        "j7_median_margin_deg": median("j7_limit_margin_deg"),
        "p50_solve_time_ms": float(np.percentile(times, 50)),
        "p95_solve_time_ms": float(np.percentile(times, 95)),
    }
    args.output_dir.mkdir(parents=True)
    frame_table.to_csv(args.output_dir / "placement_frames.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
