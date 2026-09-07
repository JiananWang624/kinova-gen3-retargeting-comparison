"""Run the fixed-base Gen3 retargeter comparison without visualization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sew_mimic.common.task_point import (  # noqa: E402
    DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M,
    DEFAULT_TASK_POINT_MODE,
)
from sew_mimic.config import load_config  # noqa: E402
from sew_mimic.exact import R2R2R2RSearchConfig  # noqa: E402
from sew_mimic.pipeline import (  # noqa: E402
    capability_metadata,
    prepare_trajectory,
    run_benchmark,
)


def _timing_settings() -> tuple[bool, int]:
    """Read and validate compare-retargeter timing settings from config.yaml."""
    config = load_config()
    benchmark_config = config.get("benchmark")
    timing_config = (
        benchmark_config.get("timing")
        if isinstance(benchmark_config, dict)
        else None
    )
    if not isinstance(timing_config, dict):
        raise ValueError("config.yaml benchmark.timing must be a mapping")
    enabled = timing_config.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("config.yaml benchmark.timing.enabled must be a bool")
    every = timing_config.get("report_every_n_frames")
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError(
            "config.yaml benchmark.timing.report_every_n_frames "
            "must be a positive integer"
        )
    return enabled, every


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare executable fixed-base Gen3 retargeters."
    )
    parser.add_argument("input_positional", type=Path, nargs="?")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--all",
        action="store_true",
        help="intentionally process all selected input frames",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("sew_mimic", "exact_sew", "numerical_oracle"),
        default=("sew_mimic", "exact_sew"),
    )
    parser.add_argument(
        "--exact-branch-policy",
        choices=("canonical", "continuous"),
        default="continuous",
    )
    parser.add_argument("--compare-exact-policies", action="store_true")
    parser.add_argument("--oracle-max-frames", type=int, default=10)
    args = parser.parse_args(argv)
    input_path = args.input or args.input_positional
    if input_path is None:
        parser.error("--input is required")
    timing_enabled, timing_report_every_n_frames = _timing_settings()
    search_config = R2R2R2RSearchConfig()
    prepared = prepare_trajectory(
        input_path,
        start_frame=args.start_frame,
        max_frames=None if args.all else args.max_frames,
        stride=args.stride,
    )
    result = run_benchmark(
        prepared,
        methods=args.methods,
        exact_branch_policy=args.exact_branch_policy,
        compare_exact_policies=args.compare_exact_policies,
        oracle_max_frames=args.oracle_max_frames,
        search_config=search_config,
        timing_enabled=timing_enabled,
        timing_report_every_n_frames=timing_report_every_n_frames,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_output = args.output_dir / "comparison_frames.csv"
    summary_output = args.output_dir / "comparison_summary.json"
    pd.DataFrame([row.to_dict() for row in result.rows]).to_csv(
        frame_output, index=False
    )
    payload = {
        "input_path": str(input_path.resolve()),
        "frame_selection": {
            "start_frame": args.start_frame,
            "max_frames": None if args.all else args.max_frames,
            "stride": args.stride,
            "all": args.all,
            "selected_frame_indices": [frame.frame for frame in prepared.frames],
        },
        "methods_requested": list(args.methods),
        "exact_branch_policy": args.exact_branch_policy,
        "compare_exact_policies": args.compare_exact_policies,
        "oracle_max_frames": args.oracle_max_frames,
        "task_point": {
            "mode": DEFAULT_TASK_POINT_MODE,
            "human_wrist_to_task_offset_m": (
                DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M.tolist()
            ),
        },
        "stereo_sew_reference": {
            "e_t": prepared.stereo.reference.e_t.tolist(),
            "e_r": prepared.stereo.reference.e_r.tolist(),
        },
        "search_mode": search_config.mode,
        "summary": result.summary,
        "capabilities": capability_metadata(prepared),
    }
    summary_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
