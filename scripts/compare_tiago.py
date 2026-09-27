"""Run fixed-base TIAGo Steel retargeting and optional oracle validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sew_mimic.pipeline.tiago import prepare_tiago_trajectory, run_tiago_benchmark  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--oracle-max-frames", type=int, default=0)
    parser.add_argument("--methods", nargs="+",
                        choices=("sew_mimic", "tiago_sew", "numerical_oracle"),
                        default=("tiago_sew",))
    args = parser.parse_args(argv)
    if args.oracle_max_frames < 0:
        parser.error("--oracle-max-frames must be nonnegative")
    trajectory = prepare_tiago_trajectory(args.input, start_frame=args.start_frame,
                                          max_frames=None if args.all else args.max_frames,
                                          stride=args.stride)
    frame_ids = [item.frame for item in trajectory.frames]
    selected = set()
    if args.oracle_max_frames:
        locations = np.linspace(0, len(frame_ids) - 1,
                                min(len(frame_ids), args.oracle_max_frames), dtype=int)
        selected = {frame_ids[index] for index in locations}
    rows, summary = run_tiago_benchmark(trajectory, methods=args.methods,
                                        oracle_frames=selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = args.output_dir / "comparison_frames.csv"
    summary_path = args.output_dir / "comparison_summary.json"
    pd.DataFrame(rows).to_csv(frames_path, index=False)
    summary_path.write_text(json.dumps({"input_path": str(args.input.resolve()),
                                        "selected_frame_indices": frame_ids,
                                        "oracle_frame_indices": sorted(selected),
                                        "methods_requested": args.methods,
                                        "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
