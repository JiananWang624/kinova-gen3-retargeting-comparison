# PAL TIAGo Pro right arm-only model

`tiago_pro_arm.xml` is a fixed-base **kinematic** MuJoCo model of the right
`tiago-pro` 7-DoF arm with `spherical-wrist`, tool changer, and
`pal-pro-gripper` (`calibration_tool=False`). Its seven revolute joints retain
the default (`limits_v2=False`, `joint_reflect=1`) physical limits, not the
`pal-sea-arm-standalone` continuous-joint limits. J1 is at the local origin;
runtime placement supplies the fixed world transform.

Official inputs pinned for reproducibility:

- [pal_sea_arm `78b544fd`](https://github.com/pal-robotics/pal_sea_arm/tree/78b544fd2d96b8edbc2472e68f653ea5cce15423): arm joint Xacro origins, axes, limits, spherical-wrist and tool-changer geometry, and arm meshes.
- [pal_pro_gripper `ed944571`](https://github.com/pal-robotics/pal_pro_gripper/tree/ed9445716b9534b6d387d88eaf88f0bcb781b5cd): gripper Xacro mount, grasping frame and meshes.
- [tiago_pro_robot `59eff255`](https://github.com/pal-robotics/tiago_pro_robot/tree/59eff25559b0f071ac5cf6d3c4c72c2e33b8951a): confirms right-arm joint configuration (`reflect=1`, `joint_reflect=1`, `limits_v2=False`); its torso-mounted arm rotation is **not** used for standalone placement.
- [PAL standalone arm Xacro](https://github.com/pal-robotics/pal_sea_arm/blob/78b544fd2d96b8edbc2472e68f653ea5cce15423/pal_sea_arm_description/robots/pal_sea_arm.urdf.xacro): defines a separate pedestal/root assembly, not this project's human-right-arm retargeting mount. The local J1 origin here is zero. Runtime places J1 at `reference_shoulder_world_m + robot_world_offset_m` and rotates only the fixed root by configurable RPY, currently Gen3's `Rx(+90°)` baseline. The standalone wrapper's pedestal translations and standalone continuous-joint limits are not imported.
- [pal_mjlab `fb3802e5`](https://github.com/pal-robotics/pal_mjlab/tree/fb3802e53f64dec7946d5a8c1b9bbc2267985887/src/pal_mjlab/robots/pal_tiago_pro/xmls): MuJoCo inertial data, fixed neutral gripper-finger poses and visual subtree. Its `ee_right` is **not** used as a task frame.

The generation script `scripts/build_tiago_pro_arm.py` requires checked-out
copies at exactly these commits and reconstructs the arm joint transforms,
axes, hard limits, tool frame and grasp frame directly from the pinned Xacro
values. It takes PAL's MuJoCo right-arm/gripper visual and inertial subtree,
removes unrelated whole-body/left-arm components, replaces rounded joint
transforms and ranges with Xacro values, fixes gripper fingers at their neutral
poses, and drops PAL MJLab's simplified collision primitives. Thus this is a
kinematics/visualization asset, **not** an authoritative collision or dynamics
model. No robot geometry is fitted or approximated for IK.

Task frame sites (kept distinct even when positions coincide):

- `arm_right_tool_link`: J7 → tool changer `xyz=(0,0,0.017)` m, identity rotation.
- `gripper_right_base_link`: tool → Pro gripper base is the official identity transform.
- `gripper_right_grasping_link`: gripper base → grasp `xyz=(0,0,0.157157)` m, `rpy=(0,-1.57,0)` rad.

The arm and gripper Xacro headers/package metadata and the PAL MJLab XML
identify Apache License 2.0; see the included `LICENSE` copied from
`pal_pro_gripper/LICENSE.txt` (identical Apache-2.0 text in PAL MJLab). Original
mesh files are redistributed unchanged from the pinned PAL source trees.
