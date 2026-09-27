"""Compute a bounded Pro right-arm 7D trajectory window for MuJoCo replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sew_mimic.config import CONFIG, project_path
from sew_mimic.pipeline.pro import prepare_pro_trajectory, run_pro_trajectory


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_path(CONFIG["human_csv"]["input_path"]))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--manual-long-run", action="store_true",
                        help="explicitly allow more than 100 frames for manual inspection")
    parser.add_argument("--heldout", type=int, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("output/tiago_pro_run"))
    args = parser.parse_args()
    if args.start_frame < 0 or args.count < 1:
        parser.error("choose a nonnegative start frame and a positive frame count")
    if args.count > 100 and not args.manual_long_run:
        parser.error("more than 100 frames requires explicit --manual-long-run")
    if len(args.heldout) > 10 or any(frame < 0 for frame in args.heldout):
        parser.error("held-out frames must be at most 10 nonnegative indices")
    window = prepare_pro_trajectory(args.input, list(range(args.start_frame, args.start_frame + args.count)))
    rows, summary = run_pro_trajectory(window)
    heldout_rows = []
    heldout_summary = None
    if args.heldout:
        heldout = prepare_pro_trajectory(args.input, args.heldout)
        heldout_rows, heldout_summary = run_pro_trajectory(heldout, reset_each_frame=True)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output / "window_frames.csv", rows)
    _write_rows(args.output / "heldout_frames.csv", heldout_rows)
    placement = CONFIG["tiago_pro"]["placement"]
    report = {"window": summary, "heldout": heldout_summary,
              "calibration_revision": CONFIG["tiago_pro"]["calibration"]["revision"],
              "model_sha256": CONFIG["tiago_pro"]["model_sha256"],
              "placement": placement,
              "window_frames": [args.start_frame, args.start_frame + args.count - 1],
              "heldout_frames": args.heldout}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
