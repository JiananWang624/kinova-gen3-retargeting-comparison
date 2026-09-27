"""Focused TIAGo model, fixed calibration, and solver gates."""

import numpy as np
from scipy.spatial.transform import Rotation

from sew_mimic.config import CONFIG
from sew_mimic.pipeline.tiago import fixed_body_to_base_transform, prepare_tiago_trajectory
from sew_mimic.tiago import (TiagoKinematics, TiagoSewGeometry, TiagoSewSolver,
                              TiagoSewTarget, validate_spherical_wrist)
from sew_mimic.tiago.calibration import calibration_report
from sew_mimic.tiago.oracle import TiagoNumericalOracle
from sew_mimic.tiago.warp import diagnose_warp_invariance
from sew_mimic.sew.legacy_adapter import solve_legacy_sew_mimic


def test_fixed_model_and_official_tcp():
    robot = TiagoKinematics()
    assert robot.model.nq == 22
    assert robot.joint_names == tuple(f"arm_{i}_joint" for i in range(1, 8))
    assert np.allclose(robot.model.site_pos[robot.tcp_id], [0, 0, 0.206575], atol=1e-12)
    # PAL's published rpy values are rounded to 1.5708, not mathematical pi/2.
    mount = Rotation.from_quat([-1, 1, 0, 0])
    official = mount * Rotation.from_euler("xyz", [-1.5708, 1.5708, 0])
    site = Rotation.from_quat(robot.model.site_quat[robot.tcp_id], scalar_first=True)
    assert (official.inv() * site).magnitude() < 1e-10
    robot.set_q(np.array(CONFIG["tiago"]["home_q_rad"]))
    assert np.allclose(robot.data.xanchor[robot.joint_ids[0]], [0.09305, 0.014, 0.8875])
    for name, value in CONFIG["tiago"]["fixed_joints"].items():
        joint = robot.fixed_joint_ids[name]
        assert robot.data.qpos[robot.model.jnt_qposadr[joint]] == value


def test_fixed_body_to_base_calibration_does_not_follow_first_frame():
    rotation, translation = fixed_body_to_base_transform()
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, [0.5299635435, -0.0406565780, 0.4919458315])
    trajectory = prepare_tiago_trajectory("data/test.csv", start_frame=100, max_frames=1)
    assert trajectory.frames[0].frame == 100
    assert np.allclose(trajectory.robot.data.xpos[1], np.zeros(3))


def test_sew_identification_is_reproducible_and_solver_independent():
    report = calibration_report("data/test.csv")
    assert report["translation_drift_m"] < 1e-12
    assert report["model_fingerprint_matches_config"]
    assert report["sew_candidates"][0]["candidate"] == "J2/J4/wrist_center"


def test_wrist_hard_gate_and_self_consistency():
    robot = TiagoKinematics()
    gate = validate_spherical_wrist(robot, samples=1000)
    assert gate.passed
    assert gate.maximum_axis_separation_m < 1e-10
    assert gate.maximum_tool_offset_error_m < 1e-10
    geometry = TiagoSewGeometry(robot)
    solver = TiagoSewSolver(robot, geometry)
    rng = np.random.default_rng(20260930)
    for q in rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1], (20, 7)):
        position, rotation = robot.tcp_pose(q)
        result = solver.solve(TiagoSewTarget(position, rotation, geometry.psi(q)))
        assert result.q is not None, result.message
        assert result.diagnostics.position_error_m < 0.001
        assert result.diagnostics.orientation_error_rad < np.deg2rad(1)
        assert result.diagnostics.sew_error_rad < np.deg2rad(1)


def test_validation_oracle_uses_separate_mujoco_data():
    robot = TiagoKinematics()
    geometry = TiagoSewGeometry(robot)
    oracle = TiagoNumericalOracle()
    assert oracle.model is not robot.model
    q = np.array(CONFIG["tiago"]["home_q_rad"])
    position, rotation = robot.tcp_pose(q)
    result = oracle.solve(TiagoSewTarget(position, rotation, geometry.psi(q)), seeds=4)
    assert result.classification == "reachable"


def test_tiago_method_zero_keeps_its_direction_and_orientation_contract():
    robot = TiagoKinematics()
    q = np.array([1.2, -0.5, -1.2, 1.4, 0.4, 0.7, -0.4])
    points = robot.sew_points(q)
    rotation = robot.tcp_pose(q)[1]
    result = solve_legacy_sew_mimic(q, points.shoulder, points.elbow,
                                    points.wrist, rotation, robot)
    assert result.q is not None
    assert result.diagnostics.metadata["upper_arm_error_deg"] < 1e-8
    assert result.diagnostics.metadata["lower_arm_error_deg"] < 1e-8
    assert result.diagnostics.metadata["wrist_rotation_error_deg"] < 1e-8


def test_warp_diagnostic_does_not_claim_executable_fixed_skeleton():
    report = diagnose_warp_invariance(TiagoKinematics(), samples=1000)
    assert not report["tiago_warp_executable"]
    selected = [row for row in report["candidate_diagnostics"] if row["selected_for_production"]]
    assert len(selected) == 1
    assert selected[0]["shoulder_variation_m"] > report["tolerance_m"]
