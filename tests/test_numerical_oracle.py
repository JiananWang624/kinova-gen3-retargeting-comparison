import numpy as np
import pytest

from sew_mimic.common import ExactSewTarget, SolverStatus, gen3_end_effector_pose
from sew_mimic.exact import NumericalExactSewOracle
from sew_mimic.exact.acceptance import (
    ORIENTATION_ACCEPTANCE_RAD,
    POSITION_ACCEPTANCE_M,
    SEW_ACCEPTANCE_RAD,
)
from sew_mimic.exact.numerical_oracle import NumericalOracleConfig
from sew_mimic.exact.residuals import ExactSewResiduals
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.sew import (
    Gen3StereoSewGeometry,
    StereoSew,
    StereoSewSingularityError,
    project_stereo_sew_reference,
)


@pytest.fixture(scope="module")
def problem():
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    q = np.array([.2, .3, -.4, .5, -.2, .3, .4])
    p, R = gen3_end_effector_pose(q, robot)
    points = geometry.sew_points(q)
    target = ExactSewTarget(
        p,
        R,
        stereo.forward(points.shoulder, points.elbow, points.wrist),
    )
    return NumericalExactSewOracle(robot, geometry, stereo), q, target


def test_pose_and_exact_sew_recover_without_hidden_qtrue_seed(problem):
    oracle, q_true, target = problem
    assert not any(
        np.array_equal(q_true, q) for _, q in oracle.deterministic_seeds
    )
    pose = oracle.solve_pose(target.position, target.rotation)
    exact = oracle.solve_pose_and_sew(target)
    assert pose.status is SolverStatus.SUCCESS_EXACT
    assert exact.status is SolverStatus.SUCCESS_EXACT
    assert exact.diagnostics.sew_error_rad < 1e-5
    assert exact.diagnostics.metadata["n_starts"] == len(oracle.deterministic_seeds)
    assert len(exact.diagnostics.metadata["runs"]) == len(oracle.deterministic_seeds)


def test_oracle_defaults_share_method2_physical_acceptance_limits():
    config = NumericalOracleConfig()
    assert config.exact_position_m == POSITION_ACCEPTANCE_M
    assert config.exact_orientation_rad == ORIENTATION_ACCEPTANCE_RAD
    assert config.exact_sew_rad == SEW_ACCEPTANCE_RAD


def test_repeat_is_deterministic_except_timing(problem):
    oracle, _, target = problem
    first, second = oracle.solve_pose_and_sew(target), oracle.solve_pose_and_sew(target)
    assert first.status is second.status
    assert np.array_equal(first.q, second.q)
    assert first.diagnostics.metadata["best_seed"] == second.diagnostics.metadata["best_seed"]


def test_invalid_seed_and_target_are_explicit(problem):
    oracle, _, target = problem
    invalid_seed = oracle.solve_pose(
        target.position, target.rotation, q_seed=np.zeros(6)
    )
    invalid_target = oracle.solve_pose([np.nan, 0, 0], target.rotation)
    assert invalid_seed.status is SolverStatus.INVALID_INPUT
    assert invalid_target.status is SolverStatus.INVALID_INPUT


def test_impossible_target_is_not_claimed_exact_or_unreachable(problem):
    oracle, _, target = problem
    result = oracle.solve_pose(target.position + [100, 100, 100], target.rotation)
    assert result.status not in (SolverStatus.SUCCESS_EXACT, SolverStatus.UNREACHABLE)
    assert result.status in (SolverStatus.SUCCESS_APPROX, SolverStatus.NUMERICAL_FAILURE)


def test_caller_local_seed_is_additive_and_near_limit_target_stays_bounded(problem):
    oracle, _, _ = problem
    q = np.where(
        np.isfinite(oracle.robot.joint_limits[:, 0]),
        oracle.robot.joint_limits[:, 0] + 1e-5,
        0.1,
    )
    p, R = gen3_end_effector_pose(q, oracle.robot)
    points = oracle.geometry.sew_points(q)
    target = ExactSewTarget(
        p,
        R,
        oracle.stereo.forward(points.shoulder, points.elbow, points.wrist),
    )
    result = oracle.solve_pose_and_sew(target, q_seed=q + 1e-7)
    assert result.status in (SolverStatus.SUCCESS_EXACT, SolverStatus.SUCCESS_APPROX)
    assert np.all(result.q >= oracle.robot.joint_limits[:, 0])
    assert np.all(result.q <= oracle.robot.joint_limits[:, 1])


def test_canonical_key_uses_physical_acceptance_thresholds(problem):
    oracle, _, _ = problem
    residual = ExactSewResiduals(
        np.array([2e-3, 0, 0]),
        np.zeros(3),
        0.0,
        np.zeros(3),
        np.eye(3),
        0.0,
    )
    candidate = {"q": np.zeros(7), "residual": residual, "margin": 0.1}
    assert not oracle._exact(residual, True)  # Cost is irrelevant to exactness.
    assert oracle._candidate_key(candidate, True)[0] == 1


def test_sew_singularity_is_explicit_status(problem):
    original, _, target = problem
    oracle = NumericalExactSewOracle(original.robot, original.geometry, original.stereo)

    class SingularStereo:
        def forward(self, *args):
            raise StereoSewSingularityError("test singularity")

    oracle.stereo = SingularStereo()
    result = oracle.solve_pose_and_sew(target)
    assert result.status is SolverStatus.SEW_SINGULAR
    assert result.diagnostics.solve_time_ms is not None


def test_unexpected_residual_error_is_numerical_failure(problem):
    original, _, target = problem
    oracle = NumericalExactSewOracle(original.robot, original.geometry, original.stereo)

    class BrokenStereo:
        def forward(self, *args):
            raise ValueError("unexpected residual failure")

    oracle.stereo = BrokenStereo()
    result = oracle.solve_pose_and_sew(target)
    assert result.status is SolverStatus.NUMERICAL_FAILURE
    assert result.diagnostics.solve_time_ms is not None
    assert all(
        "ValueError: unexpected residual failure" in run["exception"]
        for run in result.diagnostics.metadata["runs"]
    )


def test_candidate_deduplication_and_canonical_selection_are_order_independent(problem):
    oracle, _, _ = problem
    exact = ExactSewResiduals(
        np.zeros(3), np.zeros(3), 0.0, np.zeros(3), np.eye(3), 0.0
    )
    approximate = ExactSewResiduals(
        np.array([2e-6, 0, 0]),
        np.zeros(3),
        0.0,
        np.zeros(3),
        np.eye(3),
        0.0,
    )
    candidates = [
        {"q": np.array([.2, 0, 0, 0, 0, 0, 0]), "residual": exact, "margin": .1},
        {"q": np.array([.2 + 1e-10, 0, 0, 0, 0, 0, 0]), "residual": exact, "margin": .2},
        {"q": np.array([.1, 0, 0, 0, 0, 0, 0]), "residual": approximate, "margin": .9},
    ]
    forward = oracle._deduplicate_candidates(candidates)
    backward = oracle._deduplicate_candidates(list(reversed(candidates)))
    assert len(forward) == len(backward) == 2
    expected = np.array([0.2, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(oracle._select_best(forward, True)["q"], expected)
    assert np.array_equal(oracle._select_best(backward, True)["q"], expected)
