"""Method dispatch, deterministic summaries, and capability reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Literal

import numpy as np

from ..common import SolverDiagnostics, SolverResult, SolverStatus
from ..exact import (
    R2R2R2RSearchConfig,
    enumerate_exact_sew_candidates,
    human_arm_to_exact_sew_target,
    select_exact_sew_branch,
    solve_exact_sew,
)
from ..sew import StereoSewSingularityError, solve_legacy_sew_mimic
from ..warp import check_warp_fixed_geometry_compatibility
from .evaluator import EvaluationRow, evaluate_result
from .trajectory import PreparedTrajectory


MethodName = Literal["sew_mimic", "exact_sew", "numerical_oracle"]
BranchPolicy = Literal["canonical", "continuous"]
_SUCCESS = {SolverStatus.SUCCESS_EXACT, SolverStatus.SUCCESS_APPROX}


@dataclass(frozen=True)
class BenchmarkResult:
    rows: tuple[EvaluationRow, ...]
    summary: dict[str, object]


def capability_metadata(
    trajectory: PreparedTrajectory,
    *,
    warp_samples: int = 1000,
) -> dict[str, object]:
    """Describe availability separately from per-frame solver outcomes."""
    warp = check_warp_fixed_geometry_compatibility(
        trajectory.robot,
        trajectory.geometry,
        samples=warp_samples,
    )
    return {
        "sew_mimic": {
            "executable_on_gen3": True,
            "role": "baseline",
        },
        "exact_sew": {
            "executable_on_gen3": True,
            "role": "recommended",
        },
        "numerical_oracle": {
            "executable_on_gen3": True,
            "role": "validation_only",
        },
        "warp_csew": {
            "generic_core_reproduced": True,
            "executable_on_current_gen3": bool(warp.compatible),
            "gen3_fixed_link_compatible": bool(warp.compatible),
            "reason": None if warp.compatible else "fixed_link_geometry_incompatible",
            "compatibility": {
                "samples": warp.samples,
                "seed": warp.seed,
                "tolerance_m": warp.tolerance_m,
                "upper_arm_length_variation_m": warp.upper_arm_length.variation,
                "forearm_length_variation_m": warp.forearm_length.variation,
                "wrist_to_task_variation_m": warp.wrist_to_task.variation,
                "wrist_to_task_max_norm_deviation_m": (
                    warp.wrist_to_task_max_norm_deviation
                ),
            },
            "documentation": "docs/WARP_CSEW_CORE.md",
        },
    }


def _failure_result(
    method: str,
    status: SolverStatus,
    started: float,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> SolverResult:
    return SolverResult(
        method,
        status,
        None,
        SolverDiagnostics(
            solve_time_ms=1000.0 * (time.perf_counter() - started),
            metadata={} if metadata is None else metadata,
        ),
        message,
    )


def _with_elapsed_if_missing(
    result: SolverResult,
    started: float,
) -> SolverResult:
    """Supply pipeline wall time for adapters that do not time themselves."""
    if result.diagnostics.solve_time_ms is not None:
        return result
    previous = result.diagnostics
    return SolverResult(
        result.method,
        result.status,
        result.q,
        SolverDiagnostics(
            position_error_m=previous.position_error_m,
            orientation_error_rad=previous.orientation_error_rad,
            sew_error_rad=previous.sew_error_rad,
            joint_limit_margin_rad=previous.joint_limit_margin_rad,
            solve_time_ms=1000.0 * (time.perf_counter() - started),
            branch_id=previous.branch_id,
            metadata=previous.metadata,
        ),
        result.message,
    )


def _result_from_cached_selection(
    candidate_set: Any,
    *,
    branch_policy: BranchPolicy,
    q_previous: np.ndarray | None,
    search_config: R2R2R2RSearchConfig,
    started: float,
) -> SolverResult:
    """Apply the public Phase-5B selector without regenerating candidates."""
    metadata: dict[str, Any] = {
        "constraint_set": "pinch_pose_plus_stereo_sew",
        "branch_policy": branch_policy,
        "candidate_count": len(candidate_set.candidates),
        "joint_limit_valid_candidate_count": sum(
            candidate.joint_limit_valid for candidate in candidate_set.candidates
        ),
        "search_mode": search_config.mode,
        "backend_elapsed_ms": candidate_set.elapsed_ms,
    }
    try:
        outcome = select_exact_sew_branch(
            candidate_set,
            branch_policy=branch_policy,
            q_previous=q_previous,
        )
    except (TypeError, ValueError) as error:
        return _failure_result(
            "exact_sew", SolverStatus.INVALID_INPUT, started, str(error), metadata
        )
    except Exception as error:
        metadata["exception_type"] = type(error).__name__
        return _failure_result(
            "exact_sew", SolverStatus.NUMERICAL_FAILURE, started, str(error), metadata
        )

    if outcome.status is not SolverStatus.SUCCESS_EXACT:
        return _failure_result(
            "exact_sew",
            outcome.status,
            started,
            "no selectable exact branch",
            metadata,
        )

    candidate = outcome.candidate
    assert candidate is not None and outcome.branch_id is not None
    metadata.update(
        backend_branch_identity=outcome.branch_id,
        wrist_search_angle=candidate.wrist_search_angle,
        search_branch=candidate.search_branch,
        backend_metadata=dict(candidate.metadata),
    )
    return SolverResult(
        "exact_sew",
        SolverStatus.SUCCESS_EXACT,
        candidate.q,
        SolverDiagnostics(
            position_error_m=candidate.position_error_m,
            orientation_error_rad=candidate.orientation_error_rad,
            sew_error_rad=candidate.sew_error_rad,
            joint_limit_margin_rad=candidate.joint_limit_margin_rad,
            solve_time_ms=1000.0 * (time.perf_counter() - started),
            branch_id=outcome.branch_id,
            metadata=metadata,
        ),
    )


def run_benchmark(
    trajectory: PreparedTrajectory,
    *,
    methods: Iterable[MethodName] = ("sew_mimic", "exact_sew"),
    exact_branch_policy: BranchPolicy = "continuous",
    compare_exact_policies: bool = False,
    oracle_max_frames: int = 10,
    search_config: R2R2R2RSearchConfig = R2R2R2RSearchConfig(),
    timing_enabled: bool = False,
    timing_report_every_n_frames: int = 10,
) -> BenchmarkResult:
    """Run executable methods, with Method 3 limited to a leading subset.

    Continuous Method 2 history is updated only by successful selected
    configurations. A failed frame leaves the most recent valid configuration
    in place for the next frame.
    """
    selected = tuple(methods)
    supported = {"sew_mimic", "exact_sew", "numerical_oracle"}
    if not selected or not set(selected) <= supported:
        raise ValueError("methods must contain at least one supported method")
    if exact_branch_policy not in ("canonical", "continuous"):
        raise ValueError("invalid exact branch policy")
    if oracle_max_frames < 1:
        raise ValueError("oracle_max_frames must be at least 1")
    if not isinstance(search_config, R2R2R2RSearchConfig):
        raise ValueError("search_config must be R2R2R2RSearchConfig")
    if not isinstance(timing_enabled, bool):
        raise ValueError("timing_enabled must be a bool")
    if (
        isinstance(timing_report_every_n_frames, bool)
        or not isinstance(timing_report_every_n_frames, int)
        or timing_report_every_n_frames < 1
    ):
        raise ValueError("timing_report_every_n_frames must be a positive integer")

    rows: list[EvaluationRow] = []
    q_legacy = np.zeros(7)
    q_continuous: np.ndarray | None = None
    oracle = None
    if "numerical_oracle" in selected:
        from ..exact.numerical_oracle import NumericalExactSewOracle

        oracle = NumericalExactSewOracle(
            trajectory.robot, trajectory.geometry, trajectory.stereo
        )
    oracle_frames = {
        item.frame for item in trajectory.frames[:oracle_max_frames]
    }
    timing_block_elapsed = 0.0
    timing_block_count = 0
    timing_block_start = 1

    for item in trajectory.frames:
        frame_started = time.perf_counter() if timing_enabled else 0.0
        target = item.target
        if "sew_mimic" in selected:
            method_started = time.perf_counter()
            result = solve_legacy_sew_mimic(
                q_legacy,
                target.shoulder,
                target.elbow,
                target.wrist,
                target.hand_rotation,
            )
            result = _with_elapsed_if_missing(result, method_started)
            rows.append(
                evaluate_result(
                    item.frame,
                    "sew_mimic",
                    result,
                    target,
                    trajectory.robot,
                    trajectory.geometry,
                    trajectory.stereo,
                )
            )
            if result.status in _SUCCESS:
                assert result.q is not None
                q_legacy = result.q

        exact_target = None
        target_failure: SolverResult | None = None
        if "exact_sew" in selected or (
            oracle is not None and item.frame in oracle_frames
        ):
            target_started = time.perf_counter()
            try:
                exact_target = human_arm_to_exact_sew_target(
                    target, trajectory.stereo
                )
            except StereoSewSingularityError as error:
                target_failure = _failure_result(
                    "exact_sew", SolverStatus.SEW_SINGULAR, target_started, str(error)
                )
            except (TypeError, ValueError) as error:
                target_failure = _failure_result(
                    "exact_sew", SolverStatus.INVALID_INPUT, target_started, str(error)
                )

        if "exact_sew" in selected:
            policies: tuple[BranchPolicy, ...] = (
                ("canonical", "continuous")
                if compare_exact_policies
                else (exact_branch_policy,)
            )
            cached_candidates = None
            enumeration_failure: SolverResult | None = None
            enumeration_started = time.perf_counter()
            if exact_target is not None and compare_exact_policies:
                try:
                    cached_candidates = enumerate_exact_sew_candidates(
                        exact_target,
                        trajectory.robot,
                        trajectory.geometry,
                        trajectory.stereo,
                        search_config=search_config,
                    )
                except StereoSewSingularityError as error:
                    enumeration_failure = _failure_result(
                        "exact_sew",
                        SolverStatus.SEW_SINGULAR,
                        enumeration_started,
                        str(error),
                    )
                except Exception as error:
                    enumeration_failure = _failure_result(
                        "exact_sew",
                        SolverStatus.NUMERICAL_FAILURE,
                        enumeration_started,
                        str(error),
                        {"exception_type": type(error).__name__},
                    )

            for policy in policies:
                if target_failure is not None:
                    result = target_failure
                elif enumeration_failure is not None:
                    result = enumeration_failure
                elif cached_candidates is not None:
                    result = _result_from_cached_selection(
                        cached_candidates,
                        branch_policy=policy,
                        q_previous=q_continuous if policy == "continuous" else None,
                        search_config=search_config,
                        started=enumeration_started,
                    )
                else:
                    assert exact_target is not None
                    result = solve_exact_sew(
                        exact_target,
                        trajectory.robot,
                        trajectory.geometry,
                        trajectory.stereo,
                        branch_policy=policy,
                        q_previous=(
                            q_continuous if policy == "continuous" else None
                        ),
                        search_config=search_config,
                    )
                label = (
                    f"exact_sew_{policy}" if compare_exact_policies else "exact_sew"
                )
                rows.append(
                    evaluate_result(
                        item.frame,
                        label,
                        result,
                        target,
                        trajectory.robot,
                        trajectory.geometry,
                        trajectory.stereo,
                    )
                )
                if policy == "continuous" and result.status in _SUCCESS:
                    assert result.q is not None
                    q_continuous = result.q

        if oracle is not None and item.frame in oracle_frames:
            if target_failure is not None:
                result = SolverResult(
                    "numerical_exact_sew_oracle",
                    target_failure.status,
                    None,
                    target_failure.diagnostics,
                    target_failure.message,
                )
            else:
                assert exact_target is not None
                try:
                    oracle_started = time.perf_counter()
                    result = oracle.solve_pose_and_sew(exact_target)
                except Exception as error:
                    result = SolverResult(
                        "numerical_exact_sew_oracle",
                        SolverStatus.NUMERICAL_FAILURE,
                        None,
                        SolverDiagnostics(
                            metadata={"exception_type": type(error).__name__}
                        ),
                        str(error),
                    )
                result = _with_elapsed_if_missing(result, oracle_started)
            rows.append(
                evaluate_result(
                    item.frame,
                    "numerical_oracle",
                    result,
                    target,
                    trajectory.robot,
                    trajectory.geometry,
                    trajectory.stereo,
                )
            )

        if timing_enabled:
            timing_block_elapsed += time.perf_counter() - frame_started
            timing_block_count += 1
            if timing_block_count == timing_report_every_n_frames:
                average_ms = 1000.0 * timing_block_elapsed / timing_block_count
                print(
                    f"[timing] frames {timing_block_start}-"
                    f"{timing_block_start + timing_block_count - 1}: "
                    f"average {average_ms:.3f} ms/frame"
                )
                timing_block_start += timing_block_count
                timing_block_elapsed = 0.0
                timing_block_count = 0

    if timing_enabled and timing_block_count:
        average_ms = 1000.0 * timing_block_elapsed / timing_block_count
        print(
            f"[timing] frames {timing_block_start}-"
            f"{timing_block_start + timing_block_count - 1}: "
            f"average {average_ms:.3f} ms/frame"
        )

    rows.sort(key=lambda row: (row.frame, row.method))
    return BenchmarkResult(tuple(rows), summarize_rows(rows))


def _statistics(
    values: Iterable[float],
    *,
    percentiles: tuple[int, ...] = (),
    include_mean: bool = True,
    include_minimum: bool = False,
    include_maximum: bool = True,
) -> dict[str, float | int | None]:
    data = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    result: dict[str, float | int | None] = {"count": int(data.size)}
    if include_mean:
        result["mean"] = None if not data.size else float(data.mean())
    result["median"] = None if not data.size else float(np.median(data))
    for percentile in percentiles:
        result[f"p{percentile}"] = (
            None if not data.size else float(np.percentile(data, percentile))
        )
    if include_minimum:
        result["minimum"] = None if not data.size else float(data.min())
    if include_maximum:
        result["max"] = None if not data.size else float(data.max())
    return result


def summarize_rows(rows: Iterable[EvaluationRow]) -> dict[str, object]:
    """Summarize statuses over all rows and errors over successful rows only."""
    grouped: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)

    methods: dict[str, object] = {}
    for method, values in sorted(grouped.items()):
        values.sort(key=lambda row: row.frame)
        successful = [
            row
            for row in values
            if row.status
            in (SolverStatus.SUCCESS_EXACT.value, SolverStatus.SUCCESS_APPROX.value)
        ]
        jumps = [
            float(
                np.linalg.norm(
                    np.arctan2(
                        np.sin(np.asarray(right.q) - np.asarray(left.q)),
                        np.cos(np.asarray(right.q) - np.asarray(left.q)),
                    )
                )
            )
            for left, right in zip(successful, successful[1:])
        ]
        violations = sum(row.joint_limit_valid is False for row in successful)
        methods[method] = {
            "frames_requested": len(values),
            "success_count": len(successful),
            "success_fraction": len(successful) / len(values) if values else 0.0,
            "status_counts": dict(sorted(Counter(row.status for row in values).items())),
            "error_statistics_scope": "successful_frames_only",
            "ee_position_error_mm": _statistics(
                (row.ee_position_error_mm for row in successful),
                percentiles=(95, 99),
            ),
            "ee_orientation_error_deg": _statistics(
                (row.ee_orientation_error_deg for row in successful),
                percentiles=(95,),
            ),
            "sew_angle_error_deg": _statistics(
                (row.sew_angle_error_deg for row in successful),
                percentiles=(95,),
            ),
            "joint_limits": {
                "statistics_scope": "successful_frames_only",
                "violation_count": violations,
                "violation_fraction": (
                    violations / len(successful) if successful else 0.0
                ),
                "margin_deg": _statistics(
                    (row.joint_limit_margin_deg for row in successful),
                    include_mean=False,
                    include_minimum=True,
                    include_maximum=False,
                ),
            },
            "trajectory_continuity": {
                "branch_switch_count": sum(
                    left.branch_id != right.branch_id
                    for left, right in zip(successful, successful[1:])
                    if left.branch_id is not None and right.branch_id is not None
                ),
                "wrapped_joint_jump_rad": _statistics(
                    jumps,
                    percentiles=(95,),
                    include_mean=False,
                ),
            },
            "solve_time_ms": _statistics(
                (row.solve_time_ms for row in values),
                percentiles=(95,),
                include_maximum=False,
            ),
        }

    return {
        "methods": methods,
        "oracle_agreement": _oracle_agreement(grouped),
    }


def _oracle_agreement(
    grouped: dict[str, list[EvaluationRow]],
) -> dict[str, object]:
    oracle = {row.frame: row for row in grouped.get("numerical_oracle", [])}
    result: dict[str, object] = {}
    for method in ("exact_sew", "exact_sew_canonical", "exact_sew_continuous"):
        exact = {row.frame: row for row in grouped.get(method, [])}
        common = sorted(set(exact) & set(oracle))
        if not common:
            continue
        categories = {
            "both_exact": 0,
            "method2_exact_oracle_nonexact": 0,
            "oracle_exact_method2_nonexact": 0,
            "both_nonexact": 0,
        }
        for frame in common:
            method2_exact = exact[frame].status == SolverStatus.SUCCESS_EXACT.value
            oracle_exact = oracle[frame].status == SolverStatus.SUCCESS_EXACT.value
            if method2_exact and oracle_exact:
                categories["both_exact"] += 1
            elif method2_exact:
                categories["method2_exact_oracle_nonexact"] += 1
            elif oracle_exact:
                categories["oracle_exact_method2_nonexact"] += 1
            else:
                categories["both_nonexact"] += 1
        discrepancies = [
            frame
            for frame in common
            if oracle[frame].status == SolverStatus.SUCCESS_EXACT.value
            and exact[frame].status
            in (
                SolverStatus.NO_VALID_BRANCH.value,
                SolverStatus.NUMERICAL_FAILURE.value,
            )
        ]
        result[method] = {
            "common_frames": len(common),
            "categories": categories,
            "correctness_discrepancy_count": len(discrepancies),
            "correctness_discrepancy_frames": discrepancies,
        }
    return result
