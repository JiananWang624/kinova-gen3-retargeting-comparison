"""Independently revalidate saved Pro arm-only results and optionally replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
import pandas as pd

from sew_mimic.common import SolverDiagnostics, SolverResult, SolverStatus
from sew_mimic.config import CONFIG
from sew_mimic.pipeline.pro import evaluate_pro_result, prepare_pro_trajectory


def _sphere(scene: mujoco.MjvScene, point: np.ndarray,
            color: tuple[float, float, float, float], radius: float = 0.012,
            label: str = "") -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.full(3, radius), point, np.eye(3).ravel(), color)
    geom.label = label
    scene.ngeom += 1


def _frame(scene: mujoco.MjvScene, origin: np.ndarray, rotation: np.ndarray,
           length: float, label: str, width: float = 0.008) -> None:
    for axis, color in enumerate(((1.0, 0.1, 0.1, 1.0),
                                  (0.1, 0.8, 0.1, 1.0),
                                  (0.1, 0.35, 1.0, 1.0))):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                           np.zeros(3), origin, np.eye(3).ravel(), color)
        endpoint = origin + length * rotation[:, axis]
        line_width = 4.0 if width >= 0.006 else 2.0
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE,
                             line_width, origin, endpoint)
        geom.rgba = color
        geom.label = label if axis == 0 and label == "Hand" else ""
        scene.ngeom += 1


def _line(scene: mujoco.MjvScene, start: np.ndarray, end: np.ndarray,
          color: tuple[float, float, float, float], width: float = 3.0) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
    geom.rgba = color
    geom.emission = 0.25
    geom.label = ""
    scene.ngeom += 1


def _shoulder_reference_world() -> np.ndarray:
    return np.asarray(CONFIG["tiago_pro"]["placement"]["reference_shoulder_world_m"], dtype=float)


def _print_placement_diagnostic(robot, frame, q: np.ndarray, pose_source: str) -> None:
    robot.set_q(q)
    base = robot.data.xpos[robot.base_id]
    j1 = robot.data.xanchor[robot.joint_ids[0]]
    shoulder = _shoulder_reference_world()
    tool = robot.data.site_xpos[robot.position_site]
    target = frame.target.wrist
    print(f"placement diagnostic at frame {frame.frame} (world XYZ, m; {pose_source}):")
    for name, point in (("arm_base", base), ("Pro J1 / arm root", j1),
                        ("human robust shoulder reference", shoulder),
                        ("configured robot world offset", np.asarray(
                            CONFIG["tiago_pro"]["placement"]["robot_world_offset_m"])),
                        ("J1 - shoulder reference", j1 - shoulder),
                        ("arm_right_tool_link actual", tool),
                        ("human Wrist target", target)):
        print(f"  {name}: {np.array2string(point, precision=5, suppress_small=True)}")
    print("orientation task: canonical hand -> gripper_right_grasping_link; "
          "arm_right_tool_link is the position task frame")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--manual-long-run", action="store_true",
                        help="explicitly allow replaying more than 100 frames")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if args.max_frames < 1 or args.fps <= 0:
        parser.error("choose a positive frame count and fps")
    if args.max_frames > 100 and not args.manual_long_run:
        parser.error("replaying more than 100 frames requires explicit --manual-long-run")
    table = pd.read_csv(args.results).head(args.max_frames)
    if table.empty:
        parser.error("results file contains no Pro rows")
    expected = {"frame", "status", "calibration_revision", "position_frame", "orientation_frame",
                "position_error_mm", "orientation_error_deg", "sew_error_deg",
                *(f"arm_right_{i}_joint" for i in range(1, 8))}
    if not expected <= set(table):
        parser.error(f"results file lacks {sorted(expected - set(table))}")
    indices = table["frame"].astype(int).tolist()
    trajectory = prepare_pro_trajectory(args.input, indices)
    frames = {frame.frame: frame for frame in trajectory.frames}
    loaded = []
    for _, row in table.iterrows():
        frame_id = int(row["frame"])
        if row["calibration_revision"] != CONFIG["tiago_pro"]["calibration"]["revision"]:
            parser.error(f"frame {frame_id} calibration revision differs from current config")
        for field, key in (("position_frame", "position"), ("orientation_frame", "orientation")):
            if row[field] != CONFIG["tiago_pro"]["task_frames"][key]:
                parser.error(f"frame {frame_id} {field} differs from current config")
        status = SolverStatus(row["status"])
        q = np.array([row[name] for name in trajectory.robot.joint_names], dtype=float)
        success = status == SolverStatus.SUCCESS_EXACT
        if success != bool(np.all(np.isfinite(q))):
            parser.error(f"frame {frame_id} has inconsistent q/status")
        result = SolverResult("tiago_pro_7d", status, q if success else None, SolverDiagnostics())
        checked = evaluate_pro_result(frames[frame_id], result, trajectory.robot, trajectory.geometry)
        if success:
            for field in ("position_error_mm", "orientation_error_deg", "sew_error_deg"):
                if not np.isclose(float(row[field]), checked[field], atol=1e-6, rtol=0):
                    parser.error(f"frame {frame_id} stored {field} disagrees with independent MuJoCo FK")
        loaded.append((frames[frame_id], q if success else None))
    print(f"validated {len(loaded)} Pro arm-only replay rows with independent MuJoCo FK")
    robot = trajectory.robot
    first_frame, first_q = loaded[0]
    _print_placement_diagnostic(robot, first_frame,
                                first_q if first_q is not None else np.zeros(7),
                                "saved solution" if first_q is not None else "zero pose; first row failed")
    if args.no_viewer:
        return 0
    import mujoco.viewer

    scene_path = Path(CONFIG["tiago_pro"]["model_path"]).with_name("viewer_scene.xml")
    from sew_mimic.pro.model import ProKinematics
    from sew_mimic.config import project_path
    robot = ProKinematics(model_path=project_path(scene_path))
    last_valid_q = np.zeros(7)
    failed_count = sum(q is None for _, q in loaded)
    if failed_count:
        print(f"{failed_count} failed frames: holding the last valid arm pose (zero pose before the first success); red target markers indicate failure")
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        camera = CONFIG["tiago_pro"]["viewer"]
        viewer.cam.lookat[:] = (_shoulder_reference_world()
                                + np.asarray(camera["camera_lookat_offset_m"], dtype=float))
        viewer.cam.distance = float(camera["camera_distance_m"])
        viewer.cam.azimuth = float(camera["camera_azimuth_deg"])
        viewer.cam.elevation = float(camera["camera_elevation_deg"])
        shoulder_reference = _shoulder_reference_world()
        for frame, q in loaded:
            if not viewer.is_running():
                break
            if q is not None:
                last_valid_q = q
            robot.set_q(last_valid_q)
            viewer.user_scn.ngeom = 0
            _frame(viewer.user_scn, np.zeros(3), np.eye(3), 0.25, "", 0.012)
            base = robot.data.xpos[robot.base_id]
            _sphere(viewer.user_scn, base, (0.2, 0.9, 0.9, 1), 0.019)
            _frame(viewer.user_scn, base, robot.data.xmat[robot.base_id].reshape(3, 3),
                   0.08, "", 0.005)
            j1 = robot.data.xanchor[robot.joint_ids[0]]
            _sphere(viewer.user_scn, j1, (1, 0.65, 0, 1), 0.014)
            j1_body = robot.model.jnt_bodyid[robot.joint_ids[0]]
            _frame(viewer.user_scn, j1, robot.data.xmat[j1_body].reshape(3, 3),
                   0.13, "")
            _sphere(viewer.user_scn, shoulder_reference, (0.9, 0.1, 0.8, 1), 0.013,
                    "")
            _sphere(viewer.user_scn, frame.target.shoulder, (1, 0.2, 0.2, 1), 0.014)
            _sphere(viewer.user_scn, frame.target.elbow, (1, 0.8, 0.2, 1), 0.016)
            _line(viewer.user_scn, frame.target.shoulder, frame.target.elbow,
                  (1, 0.45, 0.08, 1), 3.0)
            _line(viewer.user_scn, frame.target.elbow, frame.target.wrist,
                  (1, 0.78, 0.12, 1), 3.0)
            j4 = robot.data.xanchor[robot.joint_ids[3]]
            _sphere(viewer.user_scn, j4, (0.75, 0.25, 1, 1), 0.016)
            j4_body = robot.model.jnt_bodyid[robot.joint_ids[3]]
            _frame(viewer.user_scn, j4, robot.data.xmat[j4_body].reshape(3, 3),
                   0.075, "", 0.005)
            target = frame.target.wrist
            _sphere(viewer.user_scn, target,
                    (0.1, 0.9, 0.2, 1) if q is not None else (1, 0, 0, 1),
                    0.018, "Hand")
            _frame(viewer.user_scn, target, frame.target.hand_rotation,
                   0.16, "", 0.011)
            tool = robot.data.site_xpos[robot.position_site]
            _sphere(viewer.user_scn, tool, (0.1, 0.5, 1, 1), 0.014)
            _frame(viewer.user_scn, tool,
                   robot.data.site_xmat[robot.position_site].reshape(3, 3),
                   0.10, "", 0.006)
            _frame(viewer.user_scn, tool,
                   robot.data.site_xmat[robot.orientation_site].reshape(3, 3),
                   0.13, "", 0.004)
            for point in robot.sew_points(last_valid_q):
                _sphere(viewer.user_scn, point, (1, 0.4, 0, 1), radius=0.008)
            viewer.sync()
            time.sleep(1 / args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
