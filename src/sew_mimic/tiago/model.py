"""Fixed-base TIAGo Steel arm geometry from the vendored MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import CONFIG, project_path


ARM_JOINT_NAMES = tuple(f"arm_{i}_joint" for i in range(1, 8))
FIXED_JOINTS = {
    "head_1_joint": 0.0,
    "head_2_joint": 0.0,
    "gripper_left_finger_joint": 0.03,
    "gripper_right_finger_joint": 0.03,
}


@dataclass(frozen=True)
class SewPoints:
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray


class TiagoKinematics:
    """One MuJoCo data instance, seven arm qpos, and fixed non-arm joints."""

    dof = 7

    def __init__(self, model_path: str | Path | None = None, *, validate_config: bool = True):
        settings = CONFIG["tiago"]
        self.model_path = project_path(model_path or settings["model_path"])
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "reference") >= 0:
            raise ValueError("TIAGo base must be fixed; free joint 'reference' exists")
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, settings["base_frame"])
        if self.base_id < 0 or self.model.body_parentid[self.base_id] != 0:
            raise ValueError("TIAGo base body must exist directly under world")
        self.torso_joint_id = self._joint_id("torso_lift_joint")
        if self.model.jnt_type[self.torso_joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError("TIAGo torso lift must be a slide joint")
        placement = settings["placement"]
        reference_torso = float(placement["reference_torso_lift_m"])
        offset = np.asarray(placement["j1_offset_world_m"], dtype=float)
        torso_range = self.model.jnt_range[self.torso_joint_id]
        if not np.allclose(torso_range, [0.0, 0.35], atol=1e-12, rtol=0):
            raise ValueError("TIAGo torso range differs from validated [0, 0.35] m")
        if (offset.shape != (3,) or not np.all(np.isfinite(offset)) or
            not np.isfinite(reference_torso) or
            not torso_range[0] <= reference_torso <= torso_range[1]):
            raise ValueError("invalid TIAGo J1 placement or reference torso lift")
        self.torso_lift_m = float(np.clip(reference_torso + offset[2], *torso_range))
        self.base_translation_world_m = offset - np.array([0.0, 0.0, self.torso_lift_m - reference_torso])
        self.model.body_pos[self.base_id] += self.base_translation_world_m
        self.joint_names = ARM_JOINT_NAMES
        self.joint_ids = np.array([self._joint_id(name) for name in ARM_JOINT_NAMES])
        if not np.all(self.model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("all TIAGo arm joints must be hinges")
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.joint_limits = np.array(self.model.jnt_range[self.joint_ids], dtype=float)
        self.axes = np.array(self.model.jnt_axis[self.joint_ids], dtype=float)
        self.joint_body_ids = self.model.jnt_bodyid[self.joint_ids].astype(int)
        self.fixed_parent_to_child = np.repeat(np.eye(4)[None, :, :], 7, axis=0)
        for i, body in enumerate(self.joint_body_ids):
            self.fixed_parent_to_child[i, :3, :3] = Rotation.from_quat(
                self.model.body_quat[body], scalar_first=True).as_matrix()
            self.fixed_parent_to_child[i, :3, 3] = self.model.body_pos[body]
        self.fixed_joint_ids = {name: self._joint_id(name) for name in FIXED_JOINTS}
        self.non_arm_joint_ids = np.array([i for i in range(self.model.njnt)
                                           if i not in set(self.joint_ids)], dtype=int)
        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, settings["tcp_site"])
        if self.tcp_id < 0:
            raise ValueError(f"TIAGo model has no {settings['tcp_site']} site")
        if self.model.site_bodyid[self.tcp_id] != self.model.jnt_bodyid[self.joint_ids[-1]]:
            raise ValueError("grasping frame must be attached to arm_7_link")
        self.ee_rotation_in_7 = Rotation.from_quat(self.model.site_quat[self.tcp_id], scalar_first=True).as_matrix()
        self.R_robot_align = np.eye(3)
        self.reset_fixed()
        if validate_config:
            expected = settings["model_sha256"]
            actual = hashlib.sha256(project_path(settings["fingerprint_path"]).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"TIAGo model fingerprint mismatch: config={expected}, model={actual}")
            configured = np.asarray(settings["joint_limits_rad"], dtype=float)
            if configured.shape != (7, 2) or not np.allclose(configured, self.joint_limits, atol=1e-12, rtol=0):
                raise ValueError("TIAGo joint limits differ between config and MJCF")
            configured_fixed = settings["fixed_joints"]
            if configured_fixed != FIXED_JOINTS:
                raise ValueError("TIAGo fixed joint positions differ from validated constants")
            for name, joint_id in self.fixed_joint_ids.items():
                low, high = self.model.jnt_range[joint_id]
                if self.model.jnt_limited[joint_id] and not low <= FIXED_JOINTS[name] <= high:
                    raise ValueError(f"fixed TIAGo joint {name} violates MJCF limits")

    def _joint_id(self, name: str) -> int:
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if value < 0:
            raise ValueError(f"TIAGo model has no joint {name}")
        return value

    def reset_fixed(self) -> None:
        self.data.qpos[self.model.jnt_qposadr[self.non_arm_joint_ids]] = 0.0
        self.data.qpos[self.model.jnt_qposadr[self.torso_joint_id]] = self.torso_lift_m
        for name, value in FIXED_JOINTS.items():
            self.data.qpos[self.model.jnt_qposadr[self.fixed_joint_ids[name]]] = value
        mujoco.mj_forward(self.model, self.data)

    def set_q(self, q: np.ndarray, *, enforce_limits: bool = False) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("TIAGo arm q must be a finite length-7 vector")
        if enforce_limits and np.any((q < self.joint_limits[:, 0]) | (q > self.joint_limits[:, 1])):
            raise ValueError("TIAGo arm q violates MJCF joint limits")
        self.data.qpos[self.qpos_indices] = q
        self.reset_fixed()

    def tcp_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.set_q(q)
        return self.data.site_xpos[self.tcp_id].copy(), self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()

    def R_0_i(self, q: np.ndarray, i: int) -> np.ndarray:
        if not 0 <= i <= 7:
            raise ValueError("TIAGo frame index must be in [0, 7]")
        self.set_q(q)
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link") if i == 0 else self.joint_body_ids[i - 1]
        return self.data.xmat[body].reshape(3, 3).copy()

    def aligned_ee_rotation(self, q: np.ndarray) -> np.ndarray:
        return self.tcp_pose(q)[1]

    def arm_proxy_axis(self, joint_number: int) -> np.ndarray:
        signs = CONFIG["tiago"]["baseline_proxy_signs"]
        if joint_number == 3:
            return float(signs["h3"]) * self.axes[2]
        if joint_number == 5:
            return float(signs["h5"]) * self.axes[4]
        raise ValueError("TIAGo baseline proxy exists only for h3 and h5")

    def axes_and_anchors(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.set_q(q)
        return self.data.xaxis[self.joint_ids].copy(), self.data.xanchor[self.joint_ids].copy()

    def sew_points(self, q: np.ndarray, selection: tuple[str, str, str] | None = None) -> SewPoints:
        """Evaluate model-anchored points; selection is fixed by calibration."""
        selection = selection or tuple(CONFIG["tiago"]["sew"]["selected"])
        axes, anchors = self.axes_and_anchors(q)
        def point(name: str) -> np.ndarray:
            if name.startswith("J") and len(name) == 2 and name[1:].isdigit():
                return anchors[int(name[1:]) - 1].copy()
            if name in ("J12", "J34", "J45"):
                a, b = {"J12": (0, 1), "J34": (2, 3), "J45": (3, 4)}[name]
                return closest_axis_midpoint(anchors[a], axes[a], anchors[b], axes[b])[0]
            if name == "wrist_center":
                return anchors[5].copy()
            raise ValueError(f"unknown TIAGo SEW point {name}")
        return SewPoints(*(point(name) for name in selection))

    def site_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.set_q(q)
        linear = np.zeros((3, self.model.nv))
        angular = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, linear, angular, self.tcp_id)
        dof_indices = self.model.jnt_dofadr[self.joint_ids]
        return linear[:, dof_indices], angular[:, dof_indices]


def closest_axis_midpoint(p: np.ndarray, a: np.ndarray, q: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Nearest points of two nonparallel infinite joint-axis lines."""
    matrix = np.column_stack((a, -b))
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, q - p, rcond=None)
    if rank != 2:
        raise ValueError("candidate joint axes are parallel")
    first, second = p + coefficients[0] * a, q + coefficients[1] * b
    return (first + second) / 2, float(np.linalg.norm(first - second))
