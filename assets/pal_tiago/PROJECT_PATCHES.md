# TIAGo Steel model provenance

This directory is copied from MuJoCo Menagerie `pal_tiago` at commit
`c96a32d` (release v2026.9.2). The upstream model and meshes are distributed
under the included Apache-2.0 `LICENSE`.

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d/pal_tiago

Project changes:

- `tiago.xml`: removed the floating `reference` joint to fix the base at the
  world origin. Added `gripper_grasping_frame` to `arm_7_link`. The site is the
  composition of the model's gripper mount with PAL's gripper-local origin
  `xyz=(0,0,-0.13)`, `rpy=(-1.5708,1.5708,0)` from
  https://github.com/pal-robotics/pal_gripper/blob/humble-devel/pal_gripper_description/urdf/gripper.urdf.xacro.
- `scene_position.xml`: moved the ground from `z=-0.99` to `z=0` for the
  fixed-base model.
- `tiago_position.xml` and `tiago_velocity.xml`: removed upstream keyframes
  whose qpos vectors included the removed seven free-base coordinates.

The solver loads `tiago.xml` and sets the torso, head, fingers, and wheels to
their configured fixed values for every FK evaluation. The optional viewer
loads `scene_position.xml` for lighting, ground, and full-body rendering.
