"""Write a reproducible PAL Pro arm-only geometry and task-frame report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sew_mimic.config import CONFIG
from sew_mimic.pro import ProKinematics
from sew_mimic.pro.solver import ProSewGeometry


def _skew_line_distance(p: np.ndarray, a: np.ndarray,
                        q: np.ndarray, b: np.ndarray) -> float:
    cross = np.cross(a, b)
    magnitude = float(np.linalg.norm(cross))
    if magnitude < 1e-8:
        return float(np.linalg.norm(np.cross(q - p, a)))
    return abs(float(np.dot(q - p, cross))) / magnitude


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/tiago_pro_model/geometry_report.json"))
    args = parser.parse_args()
    robot = ProKinematics()
    geometry = ProSewGeometry(robot)
    q0 = np.zeros(7)
    source = robot.geometry_at_zero()
    axes, anchors = source["axes"], source["anchors"]
    joints = []
    for i, (name, joint_id) in enumerate(zip(robot.joint_names, robot.joint_ids)):
        body_id = int(robot.model.jnt_bodyid[joint_id])
        joints.append({
            "name": name,
            "origin_in_parent_m": robot.model.body_pos[body_id].tolist(),
            "fixed_rotation_in_parent_xyzw": Rotation.from_quat(
                robot.model.body_quat[body_id], scalar_first=True).as_quat().tolist(),
            "axis_in_joint_frame": robot.model.jnt_axis[joint_id].tolist(),
            "physical_limit_deg": np.rad2deg(robot.joint_limits[i]).tolist(),
            "q0_anchor_world_m": anchors[i].tolist(),
            "q0_axis_world": axes[i].tolist(),
        })
    robot.set_q(q0)
    tool_p, tool_r = robot.pose(q0, robot.position_site)
    base_p, base_r = robot.pose(q0, robot.gripper_base_site)
    grasp_p, grasp_r = robot.pose(q0, robot.orientation_site)
    q_ref = np.array([-0.6, -0.3, 0.4, -0.5, 0.3, 0.2, -0.2])
    direction = np.array([0.3, 0.1, -0.2, 0.1, 0.2, -0.1, 0.2])
    psi = []
    sew_plane_margin = []
    reference_margin = []
    for fraction in np.linspace(0, 1, 50):
        q = q_ref + fraction * direction
        s, e, w = robot.sew_points(q)
        u = (w - s) / np.linalg.norm(w - s)
        v = (e - s) / np.linalg.norm(e - s)
        sew_plane_margin.append(float(np.linalg.norm(np.cross(u, v))))
        reference_margin.append(float(np.linalg.norm(np.cross(
            u - geometry.stereo.reference.e_t, geometry.stereo.reference.e_r))))
        psi.append(geometry.psi(q))
    pairs = {f"J{i+1}/J{j+1}": _skew_line_distance(
        anchors[i], axes[i], anchors[j], axes[j]) for i, j in ((4, 5), (5, 6), (4, 6))}
    report = {
        "model_path": str(robot.model_path),
        "model_sha256": CONFIG["tiago_pro"]["model_sha256"],
        "source_revisions": CONFIG["tiago_pro"]["sources"],
        "joint_count": robot.model.njnt,
        "joints": joints,
        "arm_base_world_position_m": robot.model.body_pos[robot.base_id].tolist(),
        "arm_base_world_rotation_xyzw": Rotation.from_quat(
            robot.model.body_quat[robot.base_id], scalar_first=True).as_quat().tolist(),
        "task_frames": CONFIG["tiago_pro"]["task_frames"],
        "sew_definition": CONFIG["tiago_pro"]["sew"]["selected"],
        "tool_to_gripper_base_position_m": (tool_r.T @ (base_p - tool_p)).tolist(),
        "tool_to_gripper_base_rotation": (tool_r.T @ base_r).tolist(),
        "tool_to_grasp_position_m": (tool_r.T @ (grasp_p - tool_p)).tolist(),
        "tool_to_grasp_rotation": (tool_r.T @ grasp_r).tolist(),
        "wrist_axis_pair_min_distance_m": pairs,
        "common_center_wrist_decomposition_applicable": max(pairs.values()) <= 1e-10,
        "engineering_sew_path": {
            "samples": len(psi),
            "minimum_plane_margin": min(sew_plane_margin),
            "minimum_reference_margin": min(reference_margin),
            "maximum_wrapped_step_rad": float(np.max(np.abs(np.diff(np.unwrap(psi))))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Pro arm-only geometry report: {args.output}")
    print(f"J6/J7 distance: {pairs['J6/J7']:.9f} m; common-center wrist decomposition applicable: {report['common_center_wrist_decomposition_applicable']}")


if __name__ == "__main__":
    main()
