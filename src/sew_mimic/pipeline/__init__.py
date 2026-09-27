"""Fixed-calibration TIAGo retargeting and evaluation."""

from .tiago import (
    TiagoFrame,
    TiagoTrajectory,
    evaluate_tiago_result,
    fixed_body_to_base_transform,
    prepare_tiago_trajectory,
    run_tiago_benchmark,
    sample_frame_indices,
)

__all__ = [
    "TiagoFrame",
    "TiagoTrajectory",
    "evaluate_tiago_result",
    "fixed_body_to_base_transform",
    "prepare_tiago_trajectory",
    "run_tiago_benchmark",
    "sample_frame_indices",
]
