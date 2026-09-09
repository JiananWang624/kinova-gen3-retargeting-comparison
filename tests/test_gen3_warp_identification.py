"""Focused guards for the diagnostic Gen3 virtual-WARP identification."""
from __future__ import annotations

import numpy as np

from sew_mimic.common.evaluation import gen3_end_effector_pose
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.sew import Gen3StereoSewGeometry
from sew_mimic.sew.gen3_geometry import sample_gen3_configurations
from sew_mimic.warp.identification import (
    FixedSkeletonParameters,
    _candidate_points,
    candidate_fixed_parameters,
    candidate_report,
    fit_proxy_fixed_model,
    geometry_derived_proxy_parameters,
    identify_fixed_skeleton,
    proxy_directions,
    proxy_prediction,
)


def _robot_geometry() -> tuple:
    robot = gen3_kinematics()
    return robot, Gen3StereoSewGeometry.from_robot(robot)


def test_candidate_axis_pair_definitions_are_the_validated_points() -> None:
    robot, geometry = _robot_geometry()
    q = sample_gen3_configurations(robot, 1, 729)[0]
    points, _ = geometry.joint_axis_lines(q)
    s, e, w = _candidate_points(geometry, q, "S23/E45/W67")
    np.testing.assert_allclose(s, points[1], atol=0.0)
    np.testing.assert_allclose(e, points[3], atol=0.0)
    np.testing.assert_allclose(w, points[5], atol=0.0)


def test_s23_candidate_link_lengths_are_configuration_invariant() -> None:
    robot, geometry = _robot_geometry()
    configurations = sample_gen3_configurations(robot, 64, 730)
    parameters = candidate_fixed_parameters(
        robot, geometry, configurations, "S23/E45/W67"
    )
    report = candidate_report(
        robot, geometry, configurations, "S23/E45/W67", parameters
    )
    assert report.upper_arm_length.variation < 1e-12
    assert report.forearm_length.variation < 1e-12
    assert report.wrist_to_task.variation < 1e-12


def test_proxy_directions_follow_the_validated_h3_h5_equations() -> None:
    robot, _ = _robot_geometry()
    q = sample_gen3_configurations(robot, 1, 731)[0]
    upper, lower = proxy_directions(robot, q)
    np.testing.assert_allclose(upper, robot.R_0_i(q, 3) @ robot.arm_proxy_axis(3), atol=0.0)
    np.testing.assert_allclose(lower, robot.R_0_i(q, 5) @ robot.arm_proxy_axis(5), atol=0.0)


def test_proxy_prediction_is_the_fixed_parameter_equation() -> None:
    robot, _ = _robot_geometry()
    q = sample_gen3_configurations(robot, 1, 732)[0]
    parameters = FixedSkeletonParameters(np.array([.1, -.2, .3]), .4, .5, np.array([.02, .03, -.04]))
    upper, lower = proxy_directions(robot, q)
    _, hand = gen3_end_effector_pose(q, robot)
    expected = parameters.shoulder + parameters.upper_arm_length * upper + parameters.forearm_length * lower + hand @ parameters.wrist_to_task
    np.testing.assert_allclose(proxy_prediction(robot, q[None, :], parameters)[0], expected, atol=0.0)


def test_fixed_parameters_validate_shape_finiteness_and_positive_lengths() -> None:
    with np.testing.assert_raises(ValueError):
        FixedSkeletonParameters(np.zeros(2), .4, .5, np.zeros(3))
    with np.testing.assert_raises(ValueError):
        FixedSkeletonParameters(np.zeros(3), 0.0, .5, np.zeros(3))
    with np.testing.assert_raises(ValueError):
        FixedSkeletonParameters(np.zeros(3), .4, .5, np.array([np.nan, 0.0, 0.0]))


def test_train_and_validation_are_independent_and_parameters_are_global() -> None:
    robot, _ = _robot_geometry()
    train = sample_gen3_configurations(robot, 32, 733)
    validation = sample_gen3_configurations(robot, 32, 734)
    assert not np.array_equal(train, validation)
    parameters = fit_proxy_fixed_model(robot, train)
    assert parameters.upper_arm_length > 0.0
    assert parameters.forearm_length > 0.0
    first = proxy_prediction(robot, validation, parameters)
    second = proxy_prediction(robot, validation, parameters)
    np.testing.assert_array_equal(first, second)
    geometry_parameters = geometry_derived_proxy_parameters(robot, _robot_geometry()[1])
    assert geometry_parameters.upper_arm_length > 0.0
    assert geometry_parameters.forearm_length > 0.0


def test_candidate_report_uses_only_training_fixed_parameters() -> None:
    robot, geometry = _robot_geometry()
    train = sample_gen3_configurations(robot, 32, 738)
    validation = sample_gen3_configurations(robot, 32, 739)
    parameters = candidate_fixed_parameters(
        robot, geometry, train, "S23/E45/W67"
    )
    report = candidate_report(
        robot, geometry, validation, "S23/E45/W67", parameters
    )
    np.testing.assert_array_equal(report.parameters.shoulder, parameters.shoulder)
    np.testing.assert_array_equal(
        report.parameters.wrist_to_task, parameters.wrist_to_task
    )
    assert report.parameters.upper_arm_length == parameters.upper_arm_length
    assert report.parameters.forearm_length == parameters.forearm_length


def test_identification_is_deterministic_and_rejects_exact_compatibility() -> None:
    robot, geometry = _robot_geometry()
    first = identify_fixed_skeleton(robot, geometry, samples=1000, train_seed=735, validation_seed=736)
    second = identify_fixed_skeleton(robot, geometry, samples=1000, train_seed=735, validation_seed=736)
    assert first.verdict == "NO_USEFUL_FIXED_SKELETON"
    assert first.train_seed != first.validation_seed
    np.testing.assert_allclose(first.s23_q1_only_variation_m, 0.01175, atol=1e-12)
    assert first.s23_q2_only_variation_m < 1e-12
    np.testing.assert_allclose(first.proxy_fit.parameters.shoulder, second.proxy_fit.parameters.shoulder, atol=0.0)
    np.testing.assert_allclose(first.proxy_fit.parameters.wrist_to_task, second.proxy_fit.parameters.wrist_to_task, atol=0.0)
    assert first.proxy_fit.parameters.upper_arm_length == second.proxy_fit.parameters.upper_arm_length
    assert first.proxy_fit.parameters.forearm_length == second.proxy_fit.parameters.forearm_length


def test_identification_rejects_a_shared_train_validation_seed() -> None:
    robot, geometry = _robot_geometry()
    with np.testing.assert_raises(ValueError):
        identify_fixed_skeleton(robot, geometry, samples=1000, train_seed=737, validation_seed=737)
