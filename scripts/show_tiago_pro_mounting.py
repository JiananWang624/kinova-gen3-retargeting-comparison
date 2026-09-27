"""Inspect the fixed TIAGo Pro arm mounting without running IK."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sew_mimic.config import CONFIG, project_path
from sew_mimic.csv_adapter import load_human_trajectory_csv
from sew_mimic.pro.model import ProKinematics
from replay_tiago_pro import _frame, _line, _sphere


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--frame", type=int, default=0,
                        help="human Wrist target frame to display; no IK is run")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="0 keeps the Viewer open until manually closed")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if args.frame < 0 or args.seconds < 0:
        parser.error("--frame and --seconds must be nonnegative")

    scene_path = Path(CONFIG["tiago_pro"]["model_path"]).with_name("viewer_scene.xml")
    robot = ProKinematics(model_path=project_path(scene_path))
    robot.set_q(np.zeros(7))
    human = load_human_trajectory_csv(args.input)
    if args.frame >= len(human):
        parser.error(f"frame {args.frame} is outside the {len(human)}-frame input")
    calibration = CONFIG["tiago_pro"]["calibration"]
    world_from_body = np.asarray(calibration["R_world_from_body"], dtype=float)
    translation = np.asarray(calibration["t_world_from_body_m"], dtype=float)
    shoulder = np.asarray(CONFIG["tiago_pro"]["placement"]["reference_shoulder_world_m"], dtype=float)
    offset = np.asarray(CONFIG["tiago_pro"]["placement"]["robot_world_offset_m"], dtype=float)
    frame_shoulder = world_from_body @ human.shoulders[args.frame] + translation
    elbow_target = world_from_body @ human.elbows[args.frame] + translation
    target = world_from_body @ human.wrists[args.frame] + translation
    target_rotation = world_from_body @ human.hand_orientations[args.frame]
    base = robot.data.xpos[robot.base_id].copy()
    base_rotation = robot.data.xmat[robot.base_id].reshape(3, 3).copy()
    j1 = robot.data.xanchor[robot.joint_ids[0]].copy()
    j4 = robot.data.xanchor[robot.joint_ids[3]].copy()
    j1_body = robot.model.jnt_bodyid[robot.joint_ids[0]]
    j1_rotation = robot.data.xmat[j1_body].reshape(3, 3).copy()
    j1_axis = robot.data.xaxis[robot.joint_ids[0]].copy()
    tool = robot.data.site_xpos[robot.position_site].copy()
    tool_rotation = robot.data.site_xmat[robot.position_site].reshape(3, 3).copy()

    def show(name: str, value: np.ndarray) -> None:
        print(f"{name}: {np.array2string(value, precision=6, suppress_small=True)}")

    print("TIAGo Pro arm-only fixed mounting; q=0; world XYZ in metres")
    show("shoulder_reference_world", shoulder)
    show("robot_world_offset", offset)
    show("root mounting RPY degrees", np.asarray([
        CONFIG["tiago_pro"]["placement"]["mounting_roll_deg"],
        CONFIG["tiago_pro"]["placement"]["mounting_pitch_deg"],
        CONFIG["tiago_pro"]["placement"]["mounting_yaw_deg"]], dtype=float))
    show("arm_base world position", base)
    show("arm_base world quaternion WXYZ", Rotation.from_matrix(base_rotation).as_quat(scalar_first=True))
    show("J1 world position", j1)
    show("J1 world quaternion WXYZ", Rotation.from_matrix(j1_rotation).as_quat(scalar_first=True))
    show("J1 - shoulder delta", j1 - shoulder)
    show("J1 axis world", j1_axis)
    show("arm_right_tool_link world position", tool)
    show("arm_right_tool_link world rotation", tool_rotation)
    show(f"Human elbow target frame {args.frame} world position", elbow_target)
    show(f"Human Wrist target frame {args.frame} world position", target)
    print("arm_base and J1 anchors coincide in this reduced arm-only MJCF; "
          "their unlabelled local axes are drawn at different lengths")
    if args.no_viewer:
        return 0

    import mujoco.viewer

    camera = CONFIG["tiago_pro"]["viewer"]
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.lookat[:] = shoulder + np.asarray(camera["camera_lookat_offset_m"], dtype=float)
        viewer.cam.distance = float(camera["camera_distance_m"])
        viewer.cam.azimuth = float(camera["camera_azimuth_deg"])
        viewer.cam.elevation = float(camera["camera_elevation_deg"])
        viewer.user_scn.ngeom = 0
        _frame(viewer.user_scn, np.zeros(3), np.eye(3), 0.25, "", 0.012)
        _sphere(viewer.user_scn, shoulder, (0.9, 0.1, 0.8, 1), 0.025)
        _sphere(viewer.user_scn, frame_shoulder, (1, 0.25, 0.15, 1), 0.017)
        _sphere(viewer.user_scn, elbow_target, (1, 0.75, 0.05, 1), 0.018)
        _line(viewer.user_scn, frame_shoulder, elbow_target, (1, 0.55, 0.05, 1), 3.0)
        _line(viewer.user_scn, elbow_target, target, (1, 0.78, 0.12, 1), 3.0)
        _sphere(viewer.user_scn, base, (0.2, 0.9, 0.9, 1), 0.020)
        _frame(viewer.user_scn, base, base_rotation, 0.08, "", 0.005)
        _sphere(viewer.user_scn, j1, (1, 0.65, 0, 1), 0.014)
        _frame(viewer.user_scn, j1, j1_rotation,
               0.13, "", 0.008)
        _sphere(viewer.user_scn, j4, (0.75, 0.25, 1, 1), 0.016)
        j4_body = robot.model.jnt_bodyid[robot.joint_ids[3]]
        _frame(viewer.user_scn, j4, robot.data.xmat[j4_body].reshape(3, 3),
               0.075, "", 0.005)
        _sphere(viewer.user_scn, tool, (0.1, 0.5, 1, 1), 0.014)
        _frame(viewer.user_scn, tool, tool_rotation, 0.12, "")
        _sphere(viewer.user_scn, target, (0.1, 0.9, 0.2, 1), 0.018, "Hand")
        _frame(viewer.user_scn, target, target_rotation, 0.16, "", 0.011)
        start = time.perf_counter()
        while viewer.is_running():
            if args.seconds and time.perf_counter() - start >= args.seconds:
                break
            viewer.sync()
            time.sleep(0.01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
