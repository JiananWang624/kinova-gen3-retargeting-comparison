"""Production Exact-SEW integration checks."""
from pathlib import Path
import numpy as np
import pytest

from sew_mimic.common import SolverStatus
from sew_mimic.exact import ExactSewConfig, ExactSewSolver, human_arm_to_exact_sew_target, solve_exact_sew
from sew_mimic.exact.acceptance import ORIENTATION_ACCEPTANCE_RAD, POSITION_ACCEPTANCE_M, SEW_ACCEPTANCE_RAD
from sew_mimic.pipeline import prepare_trajectory


def _trajectory(count=3):
    return prepare_trajectory(Path(__file__).parents[1] / "data" / "test.csv", max_frames=count)


def test_stateful_solver_is_exact_and_reset_is_deterministic():
    trajectory = _trajectory()
    targets = [human_arm_to_exact_sew_target(frame.target, trajectory.stereo) for frame in trajectory.frames]
    solver = ExactSewSolver(trajectory.robot, trajectory.geometry, trajectory.stereo)
    first = [solver.solve(target) for target in targets]
    assert all(item.status is SolverStatus.SUCCESS_EXACT for item in first)
    assert first[0].diagnostics.metadata["fallback_used"] is True
    assert all(item.diagnostics.metadata["fast_path_success"] for item in first[1:])
    for item in first:
        assert item.diagnostics.position_error_m < POSITION_ACCEPTANCE_M
        assert item.diagnostics.orientation_error_rad < ORIENTATION_ACCEPTANCE_RAD
        assert item.diagnostics.sew_error_rad < SEW_ACCEPTANCE_RAD
    solver.reset()
    second = [solver.solve(target) for target in targets]
    np.testing.assert_allclose([item.q for item in first], [item.q for item in second], atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"maximum_event_evaluations": 0}, {"global_partitions": 0},
                                      {"radii_rad": (0.01, 0.002)}])
def test_config_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        ExactSewConfig(**kwargs)


def test_single_target_continuous_accepts_previous_configuration():
    trajectory = _trajectory(1)
    target = human_arm_to_exact_sew_target(trajectory.frames[0].target, trajectory.stereo)
    first = solve_exact_sew(target, trajectory.robot, trajectory.geometry, trajectory.stereo)
    assert first.status is SolverStatus.SUCCESS_EXACT
    result = solve_exact_sew(target, trajectory.robot, trajectory.geometry, trajectory.stereo,
                             branch_policy="continuous", q_previous=first.q)
    assert result.status is SolverStatus.SUCCESS_EXACT


def test_stateful_solver_retains_last_success_across_invalid_frame():
    trajectory = _trajectory(2)
    targets = [
        human_arm_to_exact_sew_target(frame.target, trajectory.stereo)
        for frame in trajectory.frames
    ]
    solver = ExactSewSolver(trajectory.robot, trajectory.geometry, trajectory.stereo)
    first = solver.solve(targets[0])
    assert first.status is SolverStatus.SUCCESS_EXACT
    previous_q = solver.state.previous_q.copy()
    previous_wrist_angle = solver.state.previous_wrist_angle

    failure = solver.solve(object())
    assert failure.status is SolverStatus.INVALID_INPUT
    np.testing.assert_array_equal(solver.state.previous_q, previous_q)
    assert solver.state.previous_wrist_angle == previous_wrist_angle

    resumed = solver.solve(targets[1])
    assert resumed.status is SolverStatus.SUCCESS_EXACT
    assert resumed.diagnostics.metadata["fast_path_success"] is True
