"""Benchmark the production stateful Exact-SEW solver on a CSV trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sew_mimic.config import load_config
from sew_mimic.exact import ExactSewConfig, ExactSewSolver, human_arm_to_exact_sew_target
from sew_mimic.exact.acceptance import ORIENTATION_ACCEPTANCE_RAD, POSITION_ACCEPTANCE_M, SEW_ACCEPTANCE_RAD
from sew_mimic.pipeline import prepare_trajectory
from sew_mimic.common import SolverStatus


def _config() -> ExactSewConfig:
    value = load_config().get("exact_sew")
    if not isinstance(value, dict):
        raise ValueError("config.yaml exact_sew must be a mapping")
    return ExactSewConfig(**value)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _statistics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": int(data.size),
        "mean_ms": float(data.mean()),
        "median_ms": float(np.median(data)),
        "p95_ms": float(np.percentile(data, 95)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-frames", type=_positive_int, default=100)
    args = parser.parse_args(argv)
    trajectory = prepare_trajectory(args.input, max_frames=args.max_frames)
    solver = ExactSewSolver(trajectory.robot, trajectory.geometry, trajectory.stereo, config=_config())
    loop_started = time.perf_counter()
    results = [
        solver.solve(human_arm_to_exact_sew_target(frame.target, trajectory.stereo))
        for frame in trajectory.frames
    ]
    wall_time_ms = 1000.0 * (time.perf_counter() - loop_started)
    times = np.asarray([item.diagnostics.solve_time_ms for item in results], dtype=float)
    success = [item for item in results if item.status is SolverStatus.SUCCESS_EXACT]
    local_indices = [
        frame.frame for frame, item in zip(trajectory.frames, results, strict=True)
        if item.diagnostics.metadata.get("fast_path_success")
    ]
    global_indices = [
        frame.frame for frame, item in zip(trajectory.frames, results, strict=True)
        if item.diagnostics.metadata.get("fallback_used")
    ]
    local_times = [
        float(item.diagnostics.solve_time_ms)
        for item in results
        if item.diagnostics.metadata.get("fast_path_success")
    ]
    global_times = [
        float(item.diagnostics.solve_time_ms)
        for item in results
        if item.diagnostics.metadata.get("fallback_used")
    ]
    strict_residuals = bool(results) and len(success) == len(results) and all(
        item.diagnostics.position_error_m is not None
        and item.diagnostics.orientation_error_rad is not None
        and item.diagnostics.sew_error_rad is not None
        and item.diagnostics.position_error_m < POSITION_ACCEPTANCE_M
        and item.diagnostics.orientation_error_rad < ORIENTATION_ACCEPTANCE_RAD
        and item.diagnostics.sew_error_rad < SEW_ACCEPTANCE_RAD
        for item in results
    )
    overall = _statistics([float(value) for value in times])
    local = _statistics(local_times)
    global_ = _statistics(global_times)
    overall_mean = overall["mean_ms"]
    performance_passed = (
        overall_mean is not None
        and overall_mean <= 31.6
        and overall["median_ms"] is not None
        and overall["median_ms"] < 10.0
        and overall["p95_ms"] is not None
        and overall["p95_ms"] < 50.0
    )
    passes = (
        len(results) == 100
        and len(success) == 100
        and strict_residuals
        and performance_passed
    )
    summary = {
        "frames": len(results),
        "expected_frames": 100,
        "success": len(success),
        "strict_residuals": strict_residuals,
        "first_frame_ms": None if not results else float(times[0]),
        "wall_time_ms": wall_time_ms,
        "wall_mean_ms": wall_time_ms / len(results) if results else None,
        "solver_timing_ms": {
            "overall": overall,
            "local": local,
            "global": global_,
        },
        "local_frame_indices": local_indices,
        "local_frame_count": len(local_indices),
        "global_frame_indices": global_indices,
        "global_frame_count": len(global_indices),
        "maximum_residuals": {
            "position_m": max(item.diagnostics.position_error_m for item in success),
            "orientation_rad": max(item.diagnostics.orientation_error_rad for item in success),
            "sew_rad": max(item.diagnostics.sew_error_rad for item in success),
        } if strict_residuals else None,
        "acceptance": {
            "performance_passed": performance_passed,
            "passed": passes,
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
