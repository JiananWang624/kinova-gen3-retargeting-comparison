import math

import mujoco
import numpy as np

from sew_mimic.common import HumanArmTarget, evaluate_end_effector, gen3_end_effector_pose
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.sew import (
    Gen3StereoSewGeometry,
    StereoSew,
    project_stereo_sew_reference,
)
from sew_mimic.visualization import (
    build_overlay,
    overlay_base_to_world,
    render_overlay_into_scene,
)


def _case():
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    q = np.zeros(7)
    points = geometry.sew_points(q)
    pinch, aligned = gen3_end_effector_pose(q, robot)
    target = HumanArmTarget(
        points.shoulder,
        points.elbow,
        points.wrist,
        aligned,
        pinch + np.array([0.01, -0.02, 0.03]),
    )
    return robot, geometry, stereo, q, target


def test_human_task_and_aligned_target_overlay_coordinates():
    robot, geometry, stereo, _, target = _case()
    overlay = build_overlay(target, robot, geometry, stereo)
    np.testing.assert_array_equal(overlay.sphere("human_shoulder").position, target.shoulder)
    np.testing.assert_array_equal(overlay.sphere("human_elbow").position, target.elbow)
    np.testing.assert_array_equal(overlay.sphere("human_wrist").position, target.wrist)
    np.testing.assert_array_equal(overlay.sphere("human_task").position, target.task_point)
    np.testing.assert_array_equal(overlay.line("human_upper_arm").start, target.shoulder)
    np.testing.assert_array_equal(overlay.line("human_upper_arm").end, target.elbow)
    np.testing.assert_array_equal(overlay.line("human_forearm").end, target.wrist)
    np.testing.assert_array_equal(overlay.axis("human_hand_frame").origin, target.wrist)
    np.testing.assert_array_equal(
        overlay.axis("target_hand_frame").rotation, target.hand_rotation
    )


def test_actual_pinch_error_and_axes_match_phase7_authoritative_evaluator():
    robot, geometry, stereo, q, target = _case()
    overlay = build_overlay(target, robot, geometry, stereo, q)
    pinch, aligned = gen3_end_effector_pose(q, robot)
    metrics = evaluate_end_effector(q, target, robot)
    np.testing.assert_allclose(overlay.sphere("actual_pinch").position, pinch)
    error = overlay.line("ee_position_error")
    np.testing.assert_allclose(error.start, pinch)
    np.testing.assert_allclose(error.end, target.task_point)
    assert math.isclose(
        1000.0 * np.linalg.norm(error.end - error.start),
        metrics.ee_position_error_mm,
        abs_tol=1e-12,
    )
    actual_axis = overlay.axis("actual_aligned_pinch_frame")
    np.testing.assert_allclose(actual_axis.origin, pinch)
    np.testing.assert_allclose(actual_axis.rotation, aligned)
    np.testing.assert_allclose(
        overlay.axis("target_hand_frame").rotation, target.hand_rotation
    )


def test_human_display_offset_cannot_change_target_robot_q_or_metrics():
    robot, geometry, stereo, q, target = _case()
    offset = np.array([0.4, -0.3, 0.2])
    q_before = q.copy()
    target_before = (
        target.shoulder.copy(),
        target.elbow.copy(),
        target.wrist.copy(),
        target.task_point.copy(),
        target.hand_rotation.copy(),
    )
    metrics_before = evaluate_end_effector(q, target, robot)
    overlay = build_overlay(
        target,
        robot,
        geometry,
        stereo,
        q,
        human_display_offset_m=offset,
        show_sew=True,
    )
    np.testing.assert_allclose(
        overlay.sphere("human_shoulder").position, target.shoulder + offset
    )
    np.testing.assert_allclose(
        overlay.sphere("human_task").position, target.task_point + offset
    )
    np.testing.assert_allclose(overlay.sphere("target_task").position, target.task_point)
    pinch, _ = gen3_end_effector_pose(q, robot)
    np.testing.assert_allclose(overlay.sphere("actual_pinch").position, pinch)
    np.testing.assert_allclose(overlay.line("ee_position_error").end, target.task_point)
    np.testing.assert_array_equal(q, q_before)
    for actual, expected in zip(
        (
            target.shoulder,
            target.elbow,
            target.wrist,
            target.task_point,
            target.hand_rotation,
        ),
        target_before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert evaluate_end_effector(q, target, robot) == metrics_before


def test_sew_overlay_uses_validated_human_and_robot_points_and_normals():
    robot, geometry, stereo, q, target = _case()
    offset = np.array([0.1, 0.2, -0.1])
    overlay = build_overlay(
        target,
        robot,
        geometry,
        stereo,
        q,
        human_display_offset_m=offset,
        show_sew=True,
    )
    points = geometry.sew_points(q)
    np.testing.assert_allclose(
        overlay.sphere("human_sew_shoulder").position, target.shoulder + offset
    )
    np.testing.assert_allclose(
        overlay.sphere("robot_sew_elbow").position, points.elbow
    )
    human_normal = overlay.line("human_sew_normal")
    expected_human_normal = np.cross(
        target.wrist - target.shoulder, target.elbow - target.shoulder
    )
    expected_human_normal /= np.linalg.norm(expected_human_normal)
    np.testing.assert_allclose(
        (human_normal.end - human_normal.start) / 0.1, expected_human_normal
    )
    robot_normal = overlay.line("robot_sew_normal")
    expected_robot_normal = np.cross(
        points.wrist - points.shoulder, points.elbow - points.shoulder
    )
    expected_robot_normal /= np.linalg.norm(expected_robot_normal)
    np.testing.assert_allclose(
        (robot_normal.end - robot_normal.start) / 0.1, expected_robot_normal
    )


def test_base_to_world_transform_and_headless_scene_rendering():
    robot, geometry, stereo, q, target = _case()
    overlay = build_overlay(target, robot, geometry, stereo, q, show_sew=True)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([0.5, -0.4, 0.3])
    world = overlay_base_to_world(overlay, rotation, translation)
    np.testing.assert_allclose(
        world.sphere("target_task").position,
        rotation @ target.task_point + translation,
    )
    np.testing.assert_allclose(
        world.axis("target_hand_frame").rotation,
        rotation @ target.hand_rotation,
    )
    scene = mujoco.MjvScene(robot.model, maxgeom=200)
    render_overlay_into_scene(scene, world)
    assert scene.ngeom == len(world.spheres) + len(world.lines) + 3 * len(world.axes)
    render_overlay_into_scene(scene, world, show_world_frame=True)
    assert scene.ngeom == len(world.spheres) + len(world.lines) + 3 * (len(world.axes) + 1)
