"""Authoritative MuJoCo mapping for the PAL TIAGo Pro right arm only."""

from __future__ import annotations

from pathlib import Path
import hashlib

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import CONFIG, project_path


class ProKinematics:
    dof = 7

    def __init__(self, model_path: str | Path | None = None):
        settings = CONFIG["tiago_pro"]
        self.model_path = project_path(model_path or settings["model_path"])
        if model_path is None:
            actual = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            if actual != settings["model_sha256"]:
                raise ValueError(f"Pro model fingerprint mismatch: {actual}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(settings["joint_names"])
        if len(self.joint_names) != 7 or self.model.njnt != 7 or self.model.nq != 7:
            raise ValueError("Pro arm-only model must contain exactly seven joints/qpos")
        self.joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ], dtype=int)
        if np.any(self.joint_ids < 0) or len(set(self.joint_ids)) != 7:
            raise ValueError("Pro model joint names differ from configured right arm")
        if not np.all(self.model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("all Pro arm joints must be revolute")
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.joint_limits = self.model.jnt_range[self.joint_ids].copy()
        limits = np.deg2rad(np.asarray(settings["joint_limits_deg"], dtype=float))
        if limits.shape != (7, 2) or not np.allclose(self.joint_limits, limits, atol=1e-6, rtol=0):
            raise ValueError("Pro MJCF limits differ from official configured physical limits")
        self.base_id = self._body_id("arm_base")
        if self.model.body_parentid[self.base_id] != 0:
            raise ValueError("Pro arm_base must be fixed directly under MuJoCo world")
        placement = settings["placement"]
        shoulder_world = np.asarray(placement["reference_shoulder_world_m"], dtype=float)
        offset_world = np.asarray(placement["robot_world_offset_m"], dtype=float)
        mounting_rpy_deg = np.asarray([placement["mounting_roll_deg"],
                                        placement["mounting_pitch_deg"],
                                        placement["mounting_yaw_deg"]], dtype=float)
        calibration = settings["calibration"]
        calibrated_shoulder_world = (np.asarray(calibration["R_world_from_body"], dtype=float)
                                     @ np.asarray(calibration["shoulder_reference_body_m"], dtype=float)
                                     + np.asarray(calibration["t_world_from_body_m"], dtype=float))
        if (shoulder_world.shape != (3,) or offset_world.shape != (3,) or
            not np.all(np.isfinite(shoulder_world)) or not np.all(np.isfinite(offset_world)) or
            not np.all(np.isfinite(mounting_rpy_deg)) or
            not np.allclose(shoulder_world, calibrated_shoulder_world, atol=1e-9, rtol=0)):
            raise ValueError("invalid Pro shoulder reference, world offset or root mounting RPY")
        self.j1_world = shoulder_world + offset_world
        self.model.body_pos[self.base_id] = self.j1_world
        self.model.body_quat[self.base_id] = Rotation.from_euler(
            "xyz", mounting_rpy_deg, degrees=True).as_quat(scalar_first=True)
        frames = settings["task_frames"]
        self.position_site = self._site_id(frames["position"])
        self.orientation_site = self._site_id(frames["orientation"])
        self.gripper_base_site = self._site_id(frames["gripper_base"])
        self.set_q(np.zeros(7))
        if not np.allclose(self.data.xanchor[self.joint_ids[0]], self.j1_world, atol=1e-9):
            raise ValueError("Pro J1 anchor differs from fixed standalone shoulder-plus-offset target")

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Pro model is missing body {name}")
        return body_id

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Pro model is missing site {name}")
        return site_id

    def set_q(self, q: np.ndarray, *, enforce_limits: bool = False) -> None:
        values = np.asarray(q, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("Pro q must be a finite length-7 vector")
        if enforce_limits and np.any((values < self.joint_limits[:, 0]) | (values > self.joint_limits[:, 1])):
            raise ValueError("Pro q violates physical joint limits")
        self.data.qpos[self.qpos_indices] = values
        mujoco.mj_forward(self.model, self.data)

    def pose(self, q: np.ndarray, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        self.set_q(q)
        return (self.data.site_xpos[site_id].copy(),
                self.data.site_xmat[site_id].reshape(3, 3).copy())

    def task_poses(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.set_q(q)
        return (self.data.site_xpos[self.position_site].copy(),
                self.data.site_xmat[self.orientation_site].reshape(3, 3).copy())

    def sew_points(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.set_q(q)
        anchors = self.data.xanchor[self.joint_ids]
        return (anchors[0].copy(), anchors[3].copy(),
                self.data.site_xpos[self.position_site].copy())

    def geometry_at_zero(self) -> dict[str, np.ndarray]:
        self.set_q(np.zeros(7))
        return {
            "axes": self.data.xaxis[self.joint_ids].copy(),
            "anchors": self.data.xanchor[self.joint_ids].copy(),
            "tool_position": self.data.site_xpos[self.position_site].copy(),
            "grasp_rotation": self.data.site_xmat[self.orientation_site].reshape(3, 3).copy(),
        }

    def tool_to_gripper_base(self) -> tuple[np.ndarray, np.ndarray]:
        self.set_q(np.zeros(7))
        tool_pos = self.data.site_xpos[self.position_site]
        tool_rot = self.data.site_xmat[self.position_site].reshape(3, 3)
        base_pos = self.data.site_xpos[self.gripper_base_site]
        base_rot = self.data.site_xmat[self.gripper_base_site].reshape(3, 3)
        return tool_rot.T @ (base_pos - tool_pos), tool_rot.T @ base_rot
