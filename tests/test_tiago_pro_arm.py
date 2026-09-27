"""Focused PAL Pro arm-only geometry and seven-dimensional solver checks."""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from sew_mimic.config import CONFIG, project_path
from sew_mimic.pro import ProKinematics
from sew_mimic.pro._pro_7d_core import ProCore
from sew_mimic.pro.solver import ProSewGeometry, ProSewSolver, ProTarget
from sew_mimic.pipeline.pro import evaluate_pro_result, prepare_pro_trajectory, run_pro_trajectory


# Pinned PAL tiago-pro Xacro joint origins, independent of the generated MJCF.
ORIGINS = np.array([
    [0, 0, 0],
    [0.0371099999999995, 0.02, 0.18],
    [0.019999999999999, -0.17011, -0.037110000000001],
    [0.0199999999999993, -0.0371100000000008, 0.15989],
    [-0.0199999999999971, 0.17011, -0.037110000000002],
    [0, -0.0422, 0.1654],
    [0.07, -0.0534, -0.0422],
])
RPY = np.array([
    [0, 0, 0],
    [-1.57079632679489, 0, -1.5707963267949],
    [1.5707963267949, 0, 0],
    [1.57079632679489, 0, 0],
    [-1.5707963267949, 3.14, 0],
    [-1.5708, 0, 3.14],
    [1.5708, 0, 0],
])


def _official_chain(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    placement = CONFIG["tiago_pro"]["placement"]
    position = (np.asarray(placement["reference_shoulder_world_m"], dtype=float)
                + np.asarray(placement["robot_world_offset_m"], dtype=float))
    rotation = Rotation.from_euler("xyz", [placement["mounting_roll_deg"],
                                            placement["mounting_pitch_deg"],
                                            placement["mounting_yaw_deg"]], degrees=True).as_matrix()
    anchors = []
    for i in range(7):
        position += rotation @ ORIGINS[i]
        rotation = rotation @ Rotation.from_euler("xyz", RPY[i]).as_matrix()
        anchors.append(position.copy())
        axis = np.array([0.0, 0.0, -1.0 if i == 5 else 1.0])
        rotation = rotation @ Rotation.from_rotvec(axis * q[i]).as_matrix()
    tool = position + rotation @ np.array([0.0, 0.0, 0.017])
    grasp = rotation @ Rotation.from_euler("y", -1.57).as_matrix()
    return tool, grasp, np.asarray(anchors)


def _core(robot: ProKinematics) -> ProCore:
    source = robot.geometry_at_zero()
    sew = CONFIG["tiago_pro"]["sew"]
    return ProCore(source["axes"], source["anchors"], source["tool_position"],
                   source["grasp_rotation"], np.asarray(sew["e_t"]),
                   np.asarray(sew["e_r"]), robot.joint_limits)


def test_pro_model_matches_official_right_arm_xacro_fk() -> None:
    robot = ProKinematics()
    assert robot.model.njnt == robot.model.nq == 7
    np.testing.assert_allclose(robot.joint_limits,
                               np.deg2rad(CONFIG["tiago_pro"]["joint_limits_deg"]), atol=1e-9)
    rng = np.random.default_rng(20260927)
    for q in (np.zeros(7), *rng.uniform(robot.joint_limits[:, 0],
                                      robot.joint_limits[:, 1], size=(100, 7))):
        position, grasp, anchors = _official_chain(q)
        mj_position, mj_grasp = robot.task_poses(q)
        np.testing.assert_allclose(mj_position, position, atol=1e-10)
        np.testing.assert_allclose(mj_grasp, grasp, atol=1e-10)
        np.testing.assert_allclose(robot.data.xanchor[robot.joint_ids], anchors, atol=1e-10)


def test_pro_tool_base_and_grasp_frames_are_official() -> None:
    robot = ProKinematics()
    offset, rotation = robot.tool_to_gripper_base()
    np.testing.assert_allclose(offset, 0, atol=1e-12)
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
    tool_pos, tool_rot = robot.pose(np.zeros(7), robot.position_site)
    grasp_pos, grasp_rot = robot.pose(np.zeros(7), robot.orientation_site)
    np.testing.assert_allclose(tool_rot.T @ (grasp_pos - tool_pos), [0, 0, 0.157157], atol=1e-12)
    np.testing.assert_allclose(tool_rot.T @ grasp_rot,
                               Rotation.from_euler("y", -1.57).as_matrix(), atol=1e-12)
    # Official grasp +X points along tool +Z, matching captured forward.
    assert float((tool_rot.T @ grasp_rot)[2, 0]) > 0.999999


def test_pro_wrist_is_not_a_common_center() -> None:
    robot = ProKinematics()
    source = robot.geometry_at_zero()
    axes, anchors = source["axes"], source["anchors"]
    pair_distance = abs(float(np.dot(anchors[6] - anchors[5],
                                    np.cross(axes[5], axes[6])))) / np.linalg.norm(
                                        np.cross(axes[5], axes[6]))
    np.testing.assert_allclose(pair_distance, 0.07, atol=1e-10)
    assert pair_distance > 1e-10  # Old Steel analytic spherical wrist is inapplicable.


def test_cpp_fk_and_exact_jacobian_match_independent_checks() -> None:
    robot = ProKinematics()
    geometry = ProSewGeometry(robot)
    core = _core(robot)
    q = np.array([-0.7, -0.3, 0.4, -0.5, 0.4, 0.2, -0.1])
    goal = np.array([-0.5, -0.2, 0.2, -0.4, 0.2, 0.3, -0.2])
    position, orientation = robot.task_poses(goal)
    psi = geometry.psi(goal)
    answer = core.residual_and_jacobian(q, position, orientation, psi)
    central = np.zeros((7, 7))
    for i in range(7):
        delta = np.zeros(7)
        delta[i] = 1e-6
        plus = core.residual_and_jacobian(q + delta, position, orientation, psi)["residual"]
        minus = core.residual_and_jacobian(q - delta, position, orientation, psi)["residual"]
        central[:, i] = (plus - minus) / (2e-6)
    np.testing.assert_allclose(answer["jacobian"], central, atol=1e-7)
    rng = np.random.default_rng(44)
    for q in rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1], size=(100, 7)):
        expected_p, expected_r = robot.task_poses(q)
        result = core.evaluate(q)
        np.testing.assert_allclose(result["position"], expected_p, atol=1e-10)
        np.testing.assert_allclose(result["orientation"], expected_r, atol=1e-10)
        np.testing.assert_allclose(result["psi"], geometry.psi(q), atol=1e-10)


def test_pro_solver_self_consistent_strict_targets() -> None:
    robot = ProKinematics()
    geometry = ProSewGeometry(robot)
    solver = ProSewSolver(robot, geometry)
    rng = np.random.default_rng(77)
    for q in rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1], size=(100, 7)):
        position, orientation = robot.task_poses(q)
        target = ProTarget(position, orientation, geometry.psi(q))
        result = solver.solve(target)
        assert result.q is not None, result.diagnostics.metadata
        assert result.diagnostics.position_error_m < 0.001
        assert result.diagnostics.orientation_error_rad < np.deg2rad(1)
        assert result.diagnostics.sew_error_rad < np.deg2rad(1)


def test_pro_engineering_sew_continuity() -> None:
    robot = ProKinematics()
    geometry = ProSewGeometry(robot)
    q0 = np.array([-0.6, -0.3, 0.4, -0.5, 0.3, 0.2, -0.2])
    direction = np.array([0.3, 0.1, -0.2, 0.1, 0.2, -0.1, 0.2])
    angles = np.unwrap([geometry.psi(q0 + t * direction) for t in np.linspace(0, 1, 50)])
    assert np.max(np.abs(np.diff(angles))) < 0.1


def test_pro_pipeline_fixed_calibration_and_independent_evaluation() -> None:
    trajectory = prepare_pro_trajectory("data/test.csv", [318, 319, 320])
    rows, summary = run_pro_trajectory(trajectory)
    assert summary["frames"] == 3
    assert summary["strict_success"] == 3
    np.testing.assert_allclose(trajectory.robot.sew_points(np.zeros(7))[0],
                               trajectory.robot.j1_world, atol=1e-12)
    for frame, row in zip(trajectory.frames, rows):
        q = np.array([row[name] for name in trajectory.robot.joint_names])
        from sew_mimic.common import SolverDiagnostics, SolverResult, SolverStatus
        result = SolverResult("tiago_pro_7d", SolverStatus.SUCCESS_EXACT, q, SolverDiagnostics())
        independently_checked = evaluate_pro_result(frame, result, trajectory.robot, trajectory.geometry)
        for key in ("position_error_mm", "orientation_error_deg", "sew_error_deg"):
            np.testing.assert_allclose(row[key], independently_checked[key], atol=1e-9)


def test_pro_viewer_scene_preserves_task_fk_and_fixed_placement() -> None:
    from scripts.replay_tiago_pro import _shoulder_reference_world

    authoritative = ProKinematics()
    scene = ProKinematics(model_path=project_path(
        "assets/pal_tiago_pro_arm/viewer_scene.xml"))
    assert scene.model.njnt == 7
    assert scene.model.nlight >= 1
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "viewer_floor") >= 0
    shoulder = _shoulder_reference_world()
    np.testing.assert_allclose(scene.data.xpos[scene.base_id], scene.data.xanchor[scene.joint_ids[0]], atol=1e-12)
    np.testing.assert_allclose(scene.data.xanchor[scene.joint_ids[0]] - shoulder,
                               CONFIG["tiago_pro"]["placement"]["robot_world_offset_m"], atol=1e-10)
    np.testing.assert_allclose(scene.data.xmat[scene.base_id].reshape(3, 3),
                               Rotation.from_euler("x", 90, degrees=True).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(scene.data.xaxis[scene.joint_ids[0]], [0, -1, 0], atol=1e-12)
    for q in (np.zeros(7), np.array([-0.5, -0.2, 0.1, -0.3, 0.4, 0.2, -0.1])):
        actual_position, actual_orientation = authoritative.task_poses(q)
        scene_position, scene_orientation = scene.task_poses(q)
        np.testing.assert_allclose(scene_position, actual_position, atol=1e-12)
        np.testing.assert_allclose(scene_orientation, actual_orientation, atol=1e-12)


def test_pro_root_yaw_rotates_arm_without_moving_j1(monkeypatch) -> None:
    placement = CONFIG["tiago_pro"]["placement"]
    neutral = ProKinematics()
    neutral_tool, _ = neutral.task_poses(np.zeros(7))
    monkeypatch.setitem(placement, "mounting_yaw_deg", 30.0)
    turned = ProKinematics()
    turned_tool, _ = turned.task_poses(np.zeros(7))
    baseline_rotation = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    expected_rotation = Rotation.from_euler("xyz", [90, 0, 30], degrees=True).as_matrix()
    np.testing.assert_allclose(turned.data.xanchor[turned.joint_ids[0]], neutral.j1_world, atol=1e-12)
    np.testing.assert_allclose(turned.data.xmat[turned.base_id].reshape(3, 3),
                               expected_rotation, atol=1e-12)
    np.testing.assert_allclose(turned_tool - turned.j1_world,
                               expected_rotation @ baseline_rotation.T
                               @ (neutral_tool - neutral.j1_world), atol=1e-12)
    shifted_offset = np.asarray(placement["robot_world_offset_m"], dtype=float) + [0.03, 0, 0]
    monkeypatch.setitem(placement, "robot_world_offset_m", shifted_offset.tolist())
    translated = ProKinematics()
    translated_tool, _ = translated.task_poses(np.zeros(7))
    np.testing.assert_allclose(translated.j1_world - neutral.j1_world, [0.03, 0, 0], atol=1e-12)
    np.testing.assert_allclose(translated_tool - turned_tool, [0.03, 0, 0], atol=1e-12)


def test_pro_debug_frames_use_world_local_axes_and_minimal_labels() -> None:
    from scripts.replay_tiago_pro import _frame, _sphere

    robot = ProKinematics()
    scene = mujoco.MjvScene(robot.model, maxgeom=10)
    origin = np.array([0.1, -0.2, 0.3])
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    _frame(scene, origin, rotation, 0.2, "arm_right_tool_link")
    _sphere(scene, origin, (1, 1, 1, 1))
    _frame(scene, origin, rotation, 0.2, "Hand")
    assert scene.ngeom == 7
    assert scene.geoms[0].label == ""
    assert scene.geoms[3].label == ""
    assert scene.geoms[4].label == "Hand"
    np.testing.assert_allclose(scene.geoms[0].mat[:, 2], rotation[:, 0], atol=1e-12)
