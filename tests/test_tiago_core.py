"""Focused TIAGo model, fixed calibration, and solver gates."""

import mujoco
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from sew_mimic.config import CONFIG
from sew_mimic.pipeline.tiago import (fixed_body_to_base_transform,
                                      prepare_tiago_trajectory,
                                      run_tiago_benchmark)
from sew_mimic.tiago import (TiagoKinematics, TiagoSewGeometry, TiagoSewSolver,
                              TiagoSewTarget, validate_spherical_wrist)
from sew_mimic.tiago.calibration import calibration_report
from sew_mimic.tiago.oracle import TiagoNumericalOracle
from sew_mimic.tiago.wrist import decompose_wrist, wrist_center
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
    assert np.allclose(robot.data.xanchor[robot.joint_ids[0]], [0.09305, 0.014, 0.6875])
    torso_id = robot.torso_joint_id
    assert robot.data.qpos[robot.model.jnt_qposadr[torso_id]] == 0.0
    base_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert np.allclose(robot.model.body_pos[base_id], [0.0, 0.0, -0.05])
    for name, value in CONFIG["tiago"]["fixed_joints"].items():
        joint = robot.fixed_joint_ids[name]
        assert robot.data.qpos[robot.model.jnt_qposadr[joint]] == value


def test_robot_side_placement_preserves_relative_arm_geometry():
    placed = TiagoKinematics()
    nominal = TiagoKinematics()
    nominal.model.body_pos[nominal.base_id] -= nominal.base_translation_world_m
    nominal.torso_lift_m = CONFIG["tiago"]["placement"]["reference_torso_lift_m"]
    offset = np.asarray(CONFIG["tiago"]["placement"]["j1_offset_world_m"])
    for q in (np.asarray(CONFIG["tiago"]["home_q_rad"]),
              np.mean(placed.joint_limits, axis=1)):
        placed.set_q(q)
        nominal.set_q(q)
        np.testing.assert_allclose(placed.data.xanchor[placed.joint_ids] -
                                   nominal.data.xanchor[nominal.joint_ids],
                                   np.tile(offset, (7, 1)), atol=1e-12)
        np.testing.assert_allclose(placed.data.site_xpos[placed.tcp_id] -
                                   nominal.data.site_xpos[nominal.tcp_id], offset, atol=1e-12)
        np.testing.assert_allclose(placed.data.site_xmat[placed.tcp_id],
                                   nominal.data.site_xmat[nominal.tcp_id], atol=1e-12)


def test_task_frame_config_rejects_tcp_position(monkeypatch):
    robot = TiagoKinematics()
    monkeypatch.setitem(CONFIG["tiago"], "task_frames",
                        {"position": "gripper_grasping_frame", "orientation": "gripper_grasping_frame"})
    with pytest.raises(ValueError, match="spherical wrist_center position"):
        TiagoSewGeometry(robot)


def test_fixed_body_to_base_calibration_does_not_follow_first_frame():
    rotation, translation = fixed_body_to_base_transform()
    assert np.allclose(rotation, np.eye(3))
    calibration = CONFIG["tiago"]["calibration"]
    assert calibration["reference_joint_name"] == "arm_1_joint"
    assert np.allclose(translation, [0.5299635435, -0.0406565780, 0.4919458315])
    assert "t_base_from_body_m" not in calibration
    assert "user_xyz_offset_base_m" not in calibration
    assert np.allclose(translation, calibration["t_calibrated_m"])
    trajectory = prepare_tiago_trajectory("data/test.csv", start_frame=100, max_frames=1)
    assert trajectory.frames[0].frame == 100
    assert not np.any(trajectory.robot.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)


def test_locked_j1_calibration_is_reproducible():
    report = calibration_report("data/test.csv")
    assert report["reference_joint_name"] == "arm_1_joint"
    assert report["translation_drift_m"] < 1e-12
    assert np.allclose(report["shoulder_reference_base_m"], [0.09305, 0.014, 0.8875])
    assert np.allclose(report["robot_j1_placed_world_m"], [0.09305, 0.014, 0.6875])
    assert report["model_fingerprint_matches_config"]
    assert report["sew_definition"] == ["J1", "J4", "wrist_center"]


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
        position = wrist_center(robot, q)
        rotation = robot.tcp_pose(q)[1]
        result = solver.solve(TiagoSewTarget(position, rotation, geometry.psi(q)))
        assert result.q is not None, result.message
        assert result.diagnostics.position_error_m < 0.001
        assert result.diagnostics.orientation_error_rad < np.deg2rad(1)
        assert result.diagnostics.sew_error_rad < np.deg2rad(1)


def test_singular_wrist_enumerates_in_limit_j5_j7_split():
    robot = TiagoKinematics()
    prefix = np.array(CONFIG["tiago"]["home_q_rad"][:4])
    known = np.r_[prefix, [1.5, 0.0, 1.5]]
    rotation = robot.tcp_pose(known)[1]
    branches = decompose_wrist(robot, prefix, rotation)
    assert branches
    for branch in branches:
        assert np.all(branch >= robot.joint_limits[4:, 0] - 1e-12)
        assert np.all(branch <= robot.joint_limits[4:, 1] + 1e-12)
        reconstructed = robot.tcp_pose(np.r_[prefix, branch])[1]
        assert Rotation.from_matrix(reconstructed.T @ rotation).magnitude() < 1e-10
    preferred = decompose_wrist(robot, prefix, rotation,
                                preferred_wrist_q=known[4:])
    np.testing.assert_allclose(preferred[0], known[4:], atol=1e-12)
    geometry = TiagoSewGeometry(robot)
    result = TiagoSewSolver(robot, geometry).solve(TiagoSewTarget(
        wrist_center(robot, known), rotation, geometry.psi(known)))
    assert result.q is not None, result.message
    assert result.diagnostics.orientation_error_rad < 1e-10


def test_validation_oracle_uses_separate_mujoco_data():
    robot = TiagoKinematics()
    geometry = TiagoSewGeometry(robot)
    oracle = TiagoNumericalOracle()
    assert oracle.model is not robot.model
    q = np.array(CONFIG["tiago"]["home_q_rad"])
    position = wrist_center(robot, q)
    rotation = robot.tcp_pose(q)[1]
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
    assert selected[0]["shoulder_variation_m"] < report["tolerance_m"]
    assert selected[0]["upper_length_variation_m"] > report["tolerance_m"]


def test_semi_analytic_csv_targets_use_strict_wrist_center_position():
    trajectory = prepare_tiago_trajectory("data/test.csv")
    solver = TiagoSewSolver(trajectory.robot, trajectory.geometry)
    solver.settings = dict(solver.settings)
    solver.settings["recovery_seeds"] = 16
    for frame_id in (144, 145, 146):
        frame = trajectory.frames[frame_id]
        result = solver.solve(trajectory.geometry.target(frame.target))
        assert result.q is not None, (frame_id, result.message)
        assert result.diagnostics.position_error_m < 0.001
        assert np.linalg.norm(wrist_center(trajectory.robot, result.q) - frame.target.task_point) < 0.001
        assert result.diagnostics.orientation_error_rad < np.deg2rad(1)
        assert result.diagnostics.sew_error_rad < np.deg2rad(1)
        assert np.all(result.q >= trajectory.robot.joint_limits[:, 0] - 1e-12)
        assert np.all(result.q <= trajectory.robot.joint_limits[:, 1] + 1e-12)


def test_fixed_j1_pipeline_and_headless_replay(tmp_path):
    from scripts.replay_tiago import main as replay_main

    trajectory = prepare_tiago_trajectory("data/test.csv", start_frame=145, max_frames=1)
    robot = trajectory.robot
    q_home = np.asarray(CONFIG["tiago"]["home_q_rad"])
    assert np.allclose(robot.sew_points(q_home).shoulder, robot.axes_and_anchors(q_home)[1][0])
    rows, summary = run_tiago_benchmark(trajectory, methods=("tiago_sew",))
    assert summary["reachable_success"] == 1
    assert rows[0]["backend"] == "tiago_semi_analytic"
    assert rows[0]["calibration_revision"] == CONFIG["tiago"]["calibration"]["revision"]
    assert rows[0]["position_target_frame"] == "wrist_center"
    assert rows[0]["orientation_target_frame"] == "gripper_grasping_frame"
    assert rows[0]["wrist_center_position_error_mm"] < 1
    assert "ee_position_error_mm" not in rows[0]
    assert np.all(np.isfinite([rows[0][f"grasping_frame_position_{axis}_m"] for axis in "xyz"]))
    actual_grasp = np.array([rows[0][f"grasping_frame_position_{axis}_m"] for axis in "xyz"])
    assert np.linalg.norm(actual_grasp - trajectory.frames[0].target.task_point) > 0.1
    assert rows[0]["ee_orientation_error_deg"] < 1
    assert rows[0]["sew_angle_error_deg"] < 1
    for name, value in CONFIG["tiago"]["fixed_joints"].items():
        joint_id = robot.fixed_joint_ids[name]
        assert robot.data.qpos[robot.model.jnt_qposadr[joint_id]] == value
    saved = tmp_path / "comparison_frames.csv"
    pd.DataFrame(rows).to_csv(saved, index=False)
    assert replay_main(["--input", "data/test.csv", "--results", str(saved),
                        "--max-frames", "1", "--no-viewer"]) == 0
    stale = saved.with_name("stale_comparison_frames.csv")
    stale_row = dict(rows[0], calibration_revision="prior-placement")
    pd.DataFrame([stale_row]).to_csv(stale, index=False)
    with pytest.raises(SystemExit):
        replay_main(["--input", "data/test.csv", "--results", str(stale),
                     "--max-frames", "1", "--no-viewer"])
