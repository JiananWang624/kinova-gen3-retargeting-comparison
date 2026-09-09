import json
from pathlib import Path

import numpy as np
import pytest

import sew_mimic.exact.numerical_oracle as numerical_oracle
import sew_mimic.pipeline.benchmark as benchmark
from sew_mimic.common import HumanArmTarget, SolverResult, SolverStatus, gen3_end_effector_pose
from sew_mimic.exact import NumericalExactSewOracle, human_arm_to_exact_sew_target
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.pipeline import PreparedTrajectory, TrajectoryFrame, capability_metadata
from sew_mimic.pipeline.evaluator import EvaluationRow
from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference, solve_legacy_sew_mimic


def _prepared(frame_count: int = 1) -> PreparedTrajectory:
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    q = np.zeros(7)
    points = geometry.sew_points(q)
    position, rotation = gen3_end_effector_pose(q, robot)
    target = HumanArmTarget(points.shoulder, points.elbow, points.wrist, rotation, position)
    return PreparedTrajectory(robot, geometry, stereo,
                              tuple(TrajectoryFrame(index, target) for index in range(frame_count)))


def _row_without_fk(frame, method, result, target, robot, geometry, stereo):
    nan = float("nan")
    q = (nan,) * 7 if result.q is None else tuple(result.q)
    return EvaluationRow(frame, method, result.status.value, q, nan, nan, nan, None, nan,
                         result.diagnostics.branch_id,
                         result.diagnostics.solve_time_ms or nan, result.message)


def test_all_methods_receive_the_same_human_target_object(monkeypatch):
    prepared = _prepared()
    target_ids = []
    success = SolverResult("fake", SolverStatus.SUCCESS_EXACT, np.zeros(7))
    monkeypatch.setattr(benchmark, "evaluate_result",
                        lambda *args: (target_ids.append(id(args[3])) or _row_without_fk(*args)))
    monkeypatch.setattr(benchmark, "solve_legacy_sew_mimic", lambda *args: success)

    class FakeSolver:
        def __init__(self, *args, **kwargs): pass
        def solve(self, target): return success

    class FakeOracle:
        def __init__(self, *args): pass
        def solve_pose_and_sew(self, target): return success

    monkeypatch.setattr(benchmark, "ExactSewSolver", FakeSolver)
    monkeypatch.setattr(numerical_oracle, "NumericalExactSewOracle", FakeOracle)
    result = benchmark.run_benchmark(prepared, methods=("sew_mimic", "exact_sew", "numerical_oracle"), oracle_max_frames=1)
    assert len(result.rows) == 3
    assert target_ids == [id(prepared.frames[0].target)] * 3


def test_method0_pipeline_matches_direct_legacy_adapter():
    prepared = _prepared()
    target = prepared.frames[0].target
    direct = solve_legacy_sew_mimic(np.zeros(7), target.shoulder, target.elbow, target.wrist, target.hand_rotation)
    row = benchmark.run_benchmark(prepared, methods=("sew_mimic",)).rows[0]
    assert row.status == direct.status.value
    assert np.isfinite(row.solve_time_ms)
    if direct.q is not None:
        np.testing.assert_allclose(row.q, direct.q)


def test_benchmark_timing_reports_complete_and_partial_blocks(capsys):
    benchmark.run_benchmark(_prepared(3), methods=("sew_mimic",), timing_enabled=True,
                            timing_report_every_n_frames=2)
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 2
    assert lines[0].startswith("[timing] frames 1-2: average ")
    assert lines[1].startswith("[timing] frames 3-3: average ")


def test_benchmark_timing_validation_is_explicit():
    with pytest.raises(ValueError, match="timing_enabled must be a bool"):
        benchmark.run_benchmark(_prepared(), timing_enabled=1)
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.run_benchmark(_prepared(), timing_report_every_n_frames=0)


def test_method3_pipeline_matches_direct_oracle_result():
    prepared = _prepared()
    target = human_arm_to_exact_sew_target(prepared.frames[0].target, prepared.stereo)
    direct = NumericalExactSewOracle(prepared.robot, prepared.geometry, prepared.stereo).solve_pose_and_sew(target)
    row = benchmark.run_benchmark(prepared, methods=("numerical_oracle",), oracle_max_frames=1).rows[0]
    assert row.status == direct.status.value
    if direct.q is not None:
        np.testing.assert_allclose(row.q, direct.q)


def test_capability_metadata_and_dispatch_exclude_warp_rows():
    prepared = _prepared()
    capabilities = capability_metadata(prepared, warp_samples=64)
    assert capabilities["sew_mimic"] == {"executable_on_gen3": True, "role": "baseline"}
    assert capabilities["exact_sew"] == {"executable_on_gen3": True, "role": "recommended"}
    assert capabilities["numerical_oracle"] == {"executable_on_gen3": True, "role": "validation_only"}
    assert capabilities["warp_csew"]["executable_on_current_gen3"] is False
    assert all(row.method != "warp_csew" for row in benchmark.run_benchmark(prepared, methods=("sew_mimic",)).rows)


def test_exact_cli_smoke_uses_config_and_writes_output(tmp_path: Path):
    from scripts.compare_retargeters import main
    root = Path(__file__).parents[1]
    assert main(["--input", str(root / "data" / "test.csv"), "--max-frames", "1",
                 "--methods", "exact_sew", "--output-dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "comparison_summary.json").read_text(encoding="utf-8"))
    assert payload["exact_sew_config"]["global_partitions"] == 64
    assert payload["methods_requested"] == ["exact_sew"]


def test_exact_acceptance_benchmark_reports_modes_and_rejects_short_run(capsys):
    from scripts.benchmark_exact_sew import main

    root = Path(__file__).parents[1]
    assert main(["--input", str(root / "data" / "test.csv"), "--max-frames", "1"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["frames"] == 1
    assert payload["local_frame_count"] + payload["global_frame_count"] == 1
    assert payload["first_frame_ms"] is not None
    assert payload["wall_time_ms"] > 0.0
    assert payload["acceptance"]["passed"] is False
