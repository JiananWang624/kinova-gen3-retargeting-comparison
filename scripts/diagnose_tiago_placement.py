"""Inspect the fixed TIAGo placement against the complete human wrist cloud.

The workspace cloud is sampled from the actual MuJoCo model. Its nearest-point
distances are descriptive only; the bounded position-only fit is the stronger
check and neither test establishes reachability of orientation plus SEW.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sew_mimic.config import CONFIG  # noqa: E402
from sew_mimic.pipeline.tiago import prepare_tiago_trajectory  # noqa: E402
from sew_mimic.tiago.wrist import wrist_center  # noqa: E402


def _position_fit(robot, target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, float]:
    lower, upper = robot.joint_limits.T

    def residual(q: np.ndarray) -> np.ndarray:
        return wrist_center(robot, q) - target

    result = least_squares(residual, np.clip(seed, lower + 1e-9, upper - 1e-9),
                           bounds=(lower, upper), max_nfev=100,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    return result.x, float(np.linalg.norm(result.fun))


def _plot(path: Path, shoulders: np.ndarray, wrists: np.ndarray,
          workspace: np.ndarray, root: np.ndarray, selected_shoulder: np.ndarray,
          arm: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    ax = figure.add_subplot(121, projection="3d")
    ax.scatter(*workspace.T, s=0.35, alpha=0.055, c="steelblue", label="MuJoCo wrist-center samples")
    ax.scatter(*wrists.T, s=1, alpha=0.24, c="darkorange", label="all human wrist targets")
    ax.scatter(*root, s=80, c="black", marker="x", label="TIAGo arm root")
    ax.scatter(*selected_shoulder, s=65, c="purple", marker="D",
               label=f"selected SEW S ({CONFIG['tiago']['sew']['selected'][0]})")
    ax.scatter(*np.median(shoulders, axis=0), s=65, c="crimson", marker="+",
               label="median human shoulder")
    ax.plot(*arm.T, c="black", linewidth=2, label="home arm joint anchors")
    ax.set(xlabel="base X (m)", ylabel="base Y (m)", zlabel="base Z (m)",
           title="Fixed TIAGo placement / full target cloud")
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="upper center", fontsize=7)

    ax = figure.add_subplot(122)
    ax.scatter(workspace[:, 1], workspace[:, 2], s=0.35, alpha=0.055, c="steelblue")
    ax.scatter(wrists[:, 1], wrists[:, 2], s=1, alpha=0.24, c="darkorange")
    ax.scatter(shoulders[:, 1], shoulders[:, 2], s=0.5, alpha=0.1, c="crimson")
    ax.scatter(root[1], root[2], s=80, c="black", marker="x")
    ax.scatter(selected_shoulder[1], selected_shoulder[2], s=65, c="purple", marker="D")
    ax.plot(arm[:, 1], arm[:, 2], c="black", linewidth=2)
    ax.set(xlabel="base Y (m)", ylabel="base Z (m)",
           title="Side projection (right-arm reach)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _viewer(robot, wrists: np.ndarray, workspace: np.ndarray,
            root: np.ndarray, shoulder: np.ndarray) -> None:
    import mujoco.viewer

    robot.set_q(np.asarray(CONFIG["tiago"]["home_q_rad"], dtype=float))
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        viewer.cam.lookat[:] = [0.2, 0.1, 0.85]
        viewer.cam.distance = 2.3
        viewer.user_scn.ngeom = 0
        for points, rgba, radius in ((workspace[::max(1, len(workspace) // 500)],
                                      (0.2, 0.45, 0.9, 0.12), 0.005),
                                     (wrists[::max(1, len(wrists) // 1200)],
                                      (1, 0.45, 0.05, 0.6), 0.008),
                                     (np.array([root, shoulder]),
                                      (1, 0, 0, 1), 0.025)):
            for point in points:
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                    break
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.full(3, radius), point, np.eye(3).ravel(), rgba)
                viewer.user_scn.ngeom += 1
        while viewer.is_running():
            viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "tiago_j1_placement")
    parser.add_argument("--workspace-samples", type=int, default=20000)
    parser.add_argument("--position-check-frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.workspace_samples < 1 or args.position_check_frames < 1:
        parser.error("sample counts must be positive")

    trajectory = prepare_tiago_trajectory(args.input)
    robot = trajectory.robot
    calibration = CONFIG["tiago"]["calibration"]
    targets = tuple(frame.target for frame in trajectory.frames)
    shoulders = np.array([target.shoulder for target in targets])
    wrists = np.array([target.task_point for target in targets])
    root = robot.axes_and_anchors(np.asarray(CONFIG["tiago"]["home_q_rad"]))[1][0]
    shoulder = np.asarray(calibration["shoulder_reference_base_m"])
    selected_shoulder = robot.sew_points(np.asarray(CONFIG["tiago"]["home_q_rad"])).shoulder
    robot_offset = np.asarray(CONFIG["tiago"]["placement"]["j1_offset_world_m"])
    if not np.allclose(selected_shoulder, shoulder + robot_offset, atol=1e-10, rtol=0):
        raise ValueError(f"selected shoulder {selected_shoulder} differs from nominal calibration plus robot offset {shoulder + robot_offset}")
    median_shoulder = np.median(shoulders, axis=0)
    if not np.allclose(median_shoulder, shoulder, atol=1e-6, rtol=0):
        raise ValueError(f"transformed median shoulder {median_shoulder} differs from fixed human calibration {shoulder}")

    rng = np.random.default_rng(args.seed)
    lower, upper = robot.joint_limits.T
    configurations = rng.uniform(lower, upper, size=(args.workspace_samples, 7))
    workspace = np.array([wrist_center(robot, q) for q in configurations])
    tree = cKDTree(workspace)
    nearest, _ = tree.query(wrists)
    check_indices = np.linspace(0, len(wrists) - 1,
                                min(args.position_check_frames, len(wrists)), dtype=int)
    errors = []
    for index in check_indices:
        _, nearest_index = tree.query(wrists[index])
        _, error = _position_fit(robot, wrists[index], configurations[nearest_index])
        errors.append(error)
    errors = np.asarray(errors)
    robot.axes_and_anchors(np.asarray(CONFIG["tiago"]["home_q_rad"]))
    arm = np.vstack((root, robot.data.xanchor[robot.joint_ids],
                     robot.data.site_xpos[robot.tcp_id]))

    report = {
        "model_revision": CONFIG["tiago"]["menagerie_revision"],
        "calibration_revision": CONFIG["tiago"]["calibration"]["revision"],
        "task_frames": CONFIG["tiago"]["task_frames"],
        "target_frames": len(wrists), "workspace_samples": args.workspace_samples,
        "sample_seed": args.seed,
        "arm_root_base_m": root.tolist(),
        "calibration_reference_joint_name": calibration["reference_joint_name"],
        "selected_sew_shoulder_at_home_base_m": selected_shoulder.tolist(),
        "selected_sew_shoulder_from_arm_root_m": (selected_shoulder - root).tolist(),
        "selected_sew_shoulder_to_arm_root_distance_m":
            float(np.linalg.norm(selected_shoulder - root)),
        "median_human_shoulder_base_m": median_shoulder.tolist(),
        "shoulder_median_to_root_m": float(np.linalg.norm(median_shoulder - root)),
        "shoulder_median_to_calibration_reference_m":
            float(np.linalg.norm(median_shoulder - shoulder)),
        "wrist_bounds_base_m": [wrists.min(axis=0).tolist(), wrists.max(axis=0).tolist()],
        "workspace_sample_bounds_base_m": [workspace.min(axis=0).tolist(), workspace.max(axis=0).tolist()],
        "nearest_workspace_sample_distance_mm_p50_p95_max":
            (np.quantile(nearest, [0.5, 0.95, 1]) * 1000).tolist(),
        "position_only_checked_frames": check_indices.tolist(),
        "position_only_error_mm_p50_p95_max":
            (np.quantile(errors, [0.5, 0.95, 1]) * 1000).tolist(),
        "position_only_within_1mm": int(np.count_nonzero(errors <= 0.001)),
        "interpretation": "Position-only feasibility does not establish orientation or SEW feasibility.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot(args.output / "placement.png", shoulders, wrists, workspace, root,
          selected_shoulder, arm)
    print(json.dumps(report, indent=2))
    print(f"plot: {args.output / 'placement.png'}")
    if args.viewer:
        _viewer(robot, wrists, workspace, root, shoulder)


if __name__ == "__main__":
    main()
