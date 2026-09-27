"""Validate precomputed TIAGo results and optionally replay the full robot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sew_mimic.common import SolverDiagnostics, SolverResult, SolverStatus  # noqa: E402
from sew_mimic.config import CONFIG  # noqa: E402
from sew_mimic.pipeline.tiago import (evaluate_tiago_result,  # noqa: E402
                                      prepare_tiago_trajectory)
from sew_mimic.tiago import TiagoKinematics  # noqa: E402
from sew_mimic.tiago.wrist import wrist_center  # noqa: E402

CONFIG_HOME = CONFIG["tiago"]["home_q_rad"]


def _sphere(scene: mujoco.MjvScene, position: np.ndarray, color: tuple[float, ...], radius: float) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.full(3, radius), position, np.eye(3).ravel(), color)
    scene.ngeom += 1


def _world_frame(scene: mujoco.MjvScene, length: float = 0.4) -> None:
    origin = np.zeros(3)
    for axis, color in ((0, (1, 0, 0, 1)),
                        (1, (0, 1, 0, 1)),
                        (2, (0, 0.35, 1, 1))):
        if scene.ngeom >= scene.maxgeom:
            return
        endpoint = origin.copy()
        endpoint[axis] = length
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                           origin, np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.012,
                            origin, endpoint)
        geom.rgba[:] = color
        scene.ngeom += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--method", choices=("sew_mimic", "tiago_sew", "numerical_oracle"),
                        default="tiago_sew")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args(argv)
    if args.max_frames < 1 or args.fps <= 0:
        parser.error("--max-frames and --fps must be positive")
    table = pd.read_csv(args.results)
    table = table.loc[table["method"] == args.method].head(args.max_frames)
    if table.empty:
        parser.error(f"results CSV has no {args.method} rows")
    expected = {"frame", "status", "calibration_revision", "position_target_frame", "orientation_target_frame",
                "wrist_center_position_error_mm", "ee_orientation_error_deg",
                "sew_angle_error_deg", *(f"grasping_frame_position_{axis}_m" for axis in "xyz"),
                *(f"arm_{i}_joint" for i in range(1, 8))}
    if not expected <= set(table.columns):
        parser.error(f"results CSV lacks columns {sorted(expected - set(table.columns))}")
    trajectory = prepare_tiago_trajectory(args.input)
    frames = {item.frame: item for item in trajectory.frames}
    loaded = []
    for _, row in table.iterrows():
        frame_id = int(row["frame"])
        if frame_id not in frames:
            parser.error(f"result frame {frame_id} is outside the input CSV")
        revision = CONFIG["tiago"]["calibration"]["revision"]
        if row["calibration_revision"] != revision:
            parser.error(f"result frame {frame_id} uses calibration {row['calibration_revision']!r}; "
                         f"current configuration requires {revision!r}")
        task_frames = CONFIG["tiago"]["task_frames"]
        for name, key in (("position_target_frame", "position"),
                          ("orientation_target_frame", "orientation")):
            if row[name] != task_frames[key]:
                parser.error(f"result frame {frame_id} has {name}={row[name]!r}; "
                             f"current configuration requires {task_frames[key]!r}")
        status = SolverStatus(row["status"])
        values = np.array([row[name] for name in trajectory.robot.joint_names], dtype=float)
        success = status in (SolverStatus.SUCCESS_EXACT, SolverStatus.SUCCESS_APPROX)
        if success != bool(np.all(np.isfinite(values))):
            parser.error(f"result frame {frame_id} has inconsistent q and status")
        result = SolverResult(args.method, status, values if success else None, SolverDiagnostics())
        checked = evaluate_tiago_result(frames[frame_id], result, trajectory.robot,
                                        trajectory.geometry, method=args.method)
        if success:
            for field in ("wrist_center_position_error_mm", "ee_orientation_error_deg",
                          "sew_angle_error_deg",
                          *(f"grasping_frame_position_{axis}_m" for axis in "xyz")):
                if not np.isclose(checked[field], float(row[field]), atol=1e-6, rtol=0):
                    parser.error(f"frame {frame_id} stored {field} disagrees with MuJoCo FK")
        loaded.append((frames[frame_id], values if success else None, checked))
    print(f"validated {len(loaded)} TIAGo replay rows using independent MuJoCo FK")
    if args.no_viewer:
        return 0
    import mujoco.viewer

    visual = TiagoKinematics(ROOT / "assets" / "pal_tiago" / "scene_position.xml")
    print("MuJoCo world frame: origin=(0, 0, 0), +X=red, +Y=green, +Z=blue")
    with mujoco.viewer.launch_passive(visual.model, visual.data) as viewer:
        viewer.cam.lookat[:] = [0.1, -0.2, 0.9]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -18
        for frame, q, _ in loaded:
            if not viewer.is_running():
                break
            visual.set_q(np.array(CONFIG_HOME if q is None else q))
            viewer.user_scn.ngeom = 0
            _world_frame(viewer.user_scn)
            target = frame.target
            for point, color in ((target.shoulder, (0.9, 0.2, 0.2, 1)),
                                 (target.elbow, (0.9, 0.8, 0.2, 1)),
                                 (target.wrist, (0.2, 0.9, 0.2, 1)),
                                 (target.task_point, (0.2, 0.5, 1.0, 1))):
                _sphere(viewer.user_scn, point, color, 0.014)
            if q is not None:
                _sphere(viewer.user_scn, wrist_center(visual, q), (1, 0, 1, 1), 0.012)
                _sphere(viewer.user_scn, visual.data.site_xpos[visual.tcp_id], (0, 1, 1, 1), 0.009)
                points = trajectory.geometry.robot.sew_points(q)
                for point in (points.shoulder, points.elbow, points.wrist):
                    _sphere(viewer.user_scn, point, (1, 0.5, 0, 1), 0.009)
            viewer.sync()
            time.sleep(1 / args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
