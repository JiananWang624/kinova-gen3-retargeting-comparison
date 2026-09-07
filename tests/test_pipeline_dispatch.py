import numpy as np
import pytest

import sew_mimic.exact.numerical_oracle as numerical_oracle
import sew_mimic.pipeline.benchmark as benchmark
from sew_mimic.common import (
    HumanArmTarget,
    SolverDiagnostics,
    SolverResult,
    SolverStatus,
    gen3_end_effector_pose,
)
from sew_mimic.exact import (
    ExactSewCandidate,
    ExactSewCandidateSet,
    NumericalExactSewOracle,
    human_arm_to_exact_sew_target,
    solve_exact_sew,
)
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.pipeline import PreparedTrajectory, TrajectoryFrame, capability_metadata
from sew_mimic.pipeline.evaluator import EvaluationRow
from sew_mimic.sew import (
    Gen3StereoSewGeometry,
    StereoSew,
    project_stereo_sew_reference,
    solve_legacy_sew_mimic,
)


def _prepared(frame_count: int = 1) -> PreparedTrajectory:
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    q = np.zeros(7)
    points = geometry.sew_points(q)
    position, rotation = gen3_end_effector_pose(q, robot)
    target = HumanArmTarget(
        points.shoulder,
        points.elbow,
        points.wrist,
        rotation,
        position,
    )
    frames = tuple(TrajectoryFrame(index, target) for index in range(frame_count))
    return PreparedTrajectory(robot, geometry, stereo, frames)


def _row_without_fk(
    frame,
    method,
    result,
    target,
    robot,
    geometry,
    stereo,
):
    nan = float("nan")
    q = (nan,) * 7 if result.q is None else tuple(result.q)
    return EvaluationRow(
        frame,
        method,
        result.status.value,
        q,
        nan,
        nan,
        nan,
        None,
        nan,
        result.diagnostics.branch_id,
        result.diagnostics.solve_time_ms or nan,
        result.message,
    )


def test_all_methods_receive_the_same_human_target_object(monkeypatch):
    prepared = _prepared()
    target_ids = []

    def capture(*args):
        target_ids.append(id(args[3]))
        return _row_without_fk(*args)

    success = SolverResult("fake", SolverStatus.SUCCESS_EXACT, np.zeros(7))
    monkeypatch.setattr(benchmark, "evaluate_result", capture)
    monkeypatch.setattr(benchmark, "solve_legacy_sew_mimic", lambda *args: success)
    monkeypatch.setattr(benchmark, "solve_exact_sew", lambda *args, **kwargs: success)

    class FakeOracle:
        def __init__(self, *args):
            pass

        def solve_pose_and_sew(self, target):
            return success

    monkeypatch.setattr(
        numerical_oracle, "NumericalExactSewOracle", FakeOracle
    )
    result = benchmark.run_benchmark(
        prepared,
        methods=("sew_mimic", "exact_sew", "numerical_oracle"),
        oracle_max_frames=1,
    )
    assert len(result.rows) == 3
    assert target_ids == [id(prepared.frames[0].target)] * 3


def test_method0_pipeline_matches_direct_legacy_adapter():
    prepared = _prepared()
    target = prepared.frames[0].target
    direct = solve_legacy_sew_mimic(
        np.zeros(7),
        target.shoulder,
        target.elbow,
        target.wrist,
        target.hand_rotation,
    )
    row = benchmark.run_benchmark(prepared, methods=("sew_mimic",)).rows[0]
    assert row.status == direct.status.value
    assert np.isfinite(row.solve_time_ms)
    if direct.q is not None:
        np.testing.assert_allclose(row.q, direct.q)


def test_benchmark_timing_reports_complete_and_partial_blocks(capsys):
    prepared = _prepared(3)
    benchmark.run_benchmark(
        prepared,
        methods=("sew_mimic",),
        timing_enabled=True,
        timing_report_every_n_frames=2,
    )
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 2
    assert lines[0].startswith("[timing] frames 1-2: average ")
    assert lines[0].endswith(" ms/frame")
    assert lines[1].startswith("[timing] frames 3-3: average ")


def test_benchmark_timing_validation_is_explicit():
    prepared = _prepared()
    with pytest.raises(ValueError, match="timing_enabled must be a bool"):
        benchmark.run_benchmark(prepared, timing_enabled=1)
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.run_benchmark(prepared, timing_report_every_n_frames=0)


def test_method2_pipeline_matches_direct_solver_result():
    prepared = _prepared()
    exact_target = human_arm_to_exact_sew_target(
        prepared.frames[0].target, prepared.stereo
    )
    direct = solve_exact_sew(
        exact_target,
        prepared.robot,
        prepared.geometry,
        prepared.stereo,
        branch_policy="canonical",
    )
    row = benchmark.run_benchmark(
        prepared, methods=("exact_sew",), exact_branch_policy="canonical"
    ).rows[0]
    assert row.status == direct.status.value
    if direct.q is not None:
        np.testing.assert_allclose(row.q, direct.q)


def test_method3_pipeline_matches_direct_oracle_result():
    prepared = _prepared()
    exact_target = human_arm_to_exact_sew_target(
        prepared.frames[0].target, prepared.stereo
    )
    direct = NumericalExactSewOracle(
        prepared.robot, prepared.geometry, prepared.stereo
    ).solve_pose_and_sew(exact_target)

    row = benchmark.run_benchmark(
        prepared, methods=("numerical_oracle",), oracle_max_frames=1
    ).rows[0]
    assert row.status == direct.status.value
    if direct.q is not None:
        np.testing.assert_allclose(row.q, direct.q)


def test_continuous_history_retains_last_success_across_failure(monkeypatch):
    prepared = _prepared(3)
    first = np.full(7, 0.1)
    final = np.full(7, 0.2)
    outcomes = iter(
        (
            SolverResult("exact_sew", SolverStatus.SUCCESS_EXACT, first),
            SolverResult("exact_sew", SolverStatus.NO_VALID_BRANCH, None),
            SolverResult("exact_sew", SolverStatus.SUCCESS_EXACT, final),
        )
    )
    histories = []

    def solve(*args, q_previous=None, **kwargs):
        histories.append(None if q_previous is None else q_previous.copy())
        return next(outcomes)

    monkeypatch.setattr(benchmark, "solve_exact_sew", solve)
    monkeypatch.setattr(benchmark, "evaluate_result", _row_without_fk)
    result = benchmark.run_benchmark(prepared, methods=("exact_sew",))
    assert [row.status for row in result.rows] == [
        "SUCCESS_EXACT",
        "NO_VALID_BRANCH",
        "SUCCESS_EXACT",
    ]
    assert histories[0] is None
    np.testing.assert_array_equal(histories[1], first)
    np.testing.assert_array_equal(histories[2], first)


def _candidate(q, margin, branch):
    return ExactSewCandidate(
        np.asarray(q, dtype=float),
        0.0,
        branch,
        0.0,
        0.0,
        0.0,
        True,
        margin,
        True,
        {},
    )


def _candidate_set(candidates):
    return ExactSewCandidateSet(tuple(candidates), len(candidates), len(candidates), len(candidates), {}, 1.0)


def test_policy_comparison_enumerates_once_and_continuous_is_nearest(monkeypatch):
    prepared = _prepared(2)
    sets = iter(
        (
            _candidate_set(
                (_candidate(np.zeros(7), 2.0, 0), _candidate(np.ones(7), 1.0, 1))
            ),
            _candidate_set(
                (_candidate(np.ones(7), 2.0, 0), _candidate(np.full(7, 0.1), 1.0, 1))
            ),
        )
    )
    calls = []

    def enumerate_once(*args, **kwargs):
        calls.append(1)
        return next(sets)

    monkeypatch.setattr(benchmark, "enumerate_exact_sew_candidates", enumerate_once)
    monkeypatch.setattr(benchmark, "evaluate_result", _row_without_fk)
    result = benchmark.run_benchmark(
        prepared, methods=("exact_sew",), compare_exact_policies=True
    )
    assert len(calls) == 2
    rows = {(row.frame, row.method): row for row in result.rows}
    np.testing.assert_allclose(rows[(1, "exact_sew_canonical")].q, np.ones(7))
    np.testing.assert_allclose(rows[(1, "exact_sew_continuous")].q, np.full(7, 0.1))


def test_cached_backend_exception_is_numerical_failure_and_row_is_retained(monkeypatch):
    prepared = _prepared()
    monkeypatch.setattr(
        benchmark,
        "enumerate_exact_sew_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("backend")),
    )
    monkeypatch.setattr(benchmark, "evaluate_result", _row_without_fk)
    rows = benchmark.run_benchmark(
        prepared, methods=("exact_sew",), compare_exact_policies=True
    ).rows
    assert len(rows) == 2
    assert {row.status for row in rows} == {"NUMERICAL_FAILURE"}


def test_warp_is_incompatible_capability_metadata_and_has_no_frame_rows():
    prepared = _prepared()
    capabilities = capability_metadata(prepared, warp_samples=64)
    assert capabilities["sew_mimic"] == {
        "executable_on_gen3": True,
        "role": "baseline",
    }
    assert capabilities["exact_sew"] == {
        "executable_on_gen3": True,
        "role": "recommended",
    }
    assert capabilities["numerical_oracle"] == {
        "executable_on_gen3": True,
        "role": "validation_only",
    }
    warp = capabilities["warp_csew"]
    assert warp["generic_core_reproduced"] is True
    assert warp["executable_on_current_gen3"] is False
    assert warp["reason"] == "fixed_link_geometry_incompatible"
    rows = benchmark.run_benchmark(prepared, methods=("sew_mimic",)).rows
    assert all(row.method != "warp_csew" for row in rows)
