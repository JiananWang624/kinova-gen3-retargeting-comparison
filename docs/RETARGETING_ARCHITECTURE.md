# Retargeting Architecture

The repository keeps four retargeting methods separate. A status is exact only
with respect to the constraints declared by that method.

## Method 0 — Legacy SEW-Mimic

This is the verified regression baseline implemented by
`sew_mimic.retarget.sew_mimic`. It matches human upper-arm direction,
lower-arm direction, and hand orientation. It does not constrain the actual
Gen3 `pinch_site` position. The shared-result adapter delegates to the legacy
implementation without changing its inputs, joint solution, subproblems, or
failure behavior for direct callers.

## Method 1 — WARP-cSEW + Legacy SEW-Mimic

The generic fixed-link corrected-skeleton core is reproduced and validated.
The original Exact-SEW point definition fails the fixed-link gate. Phase W1
also found that none of the tested kinematically meaningful virtual definitions
provides a fixed skeleton consistent with the established h3/h5 proxies below
the practical position threshold. Consequently there is no executable Gen3
WARP trajectory path in this repository. WARP materials explicitly demonstrate
Dual-Kinova3; the missing capability here means its Kinova-specific
parameterization has not yet been reproduced, not that Gen3 is inherently
incompatible or that all virtual definitions have been disproved.

## Method 2 — Exact Pose + Stereo-SEW Gen3 IK

This is the recommended fixed-base Kinova Gen3 solver. It
match the target pinch position, aligned pinch orientation, and human
Stereo-SEW parameter without moving the Gen3 root per frame. Unreachable
fixed-base targets must be reported explicitly.

## Method 3 — Numerical Exact Pose + SEW

This numerical method is a validation oracle only. It does not silently
replace Method 2, and least-squares solutions must never be labelled exact.

## Validation authority

For every executable method, final end-effector position and orientation errors
must be evaluated using the real MuJoCo-derived `Gen3Kinematics` `pinch_site`
forward kinematics. Visualization offsets must never enter mathematical
calculations.

## Human task point

The canonical human task point is `t_h`. The default configuration uses
`task_point.mode: wrist`, so `t_h = w_h`. No anatomical wrist-to-palm distance
is assumed. The optional `wrist_plus_hand_offset` mode uses

```text
t_h = w_h + H_h @ p_human_WT
```

where `p_human_WT` is a fixed metre-valued vector expressed in the canonical
human hand frame. Its default is `[0, 0, 0]`.

## Metric frame and mounting

CSV positions first become absolute body/MuJoCo-world coordinates through
`HumanCSVAdapter`. The fixed mounting then defines `R_world_from_base` and
`p_world_of_base`, including `robot.world_offset_m`. Before solving or
evaluation, human points and orientations are expressed in native Gen3 base
frame 0:

```text
p_base = R_world_from_base.T @ (p_world - p_world_of_base)
H_base = R_world_from_base.T @ H_world
```

`Gen3Kinematics.ee_transform(q)` returns the MuJoCo-derived `pinch_site` pose
in the same native base frame. Position error compares its translation with
`t_h_base`; orientation error compares
`ee_transform(q).R @ R_robot_align` with `H_h_base`. Comparing both poses in
world coordinates would give identical errors under the common rigid mounting
transform.

`robot.world_offset_m` is part of the physical fixed root pose, not a display
offset. It changes the human-target coordinates relative to the fixed Gen3 and
therefore changes position error. Camera/look-at offsets are visualization-only
and never enter these calculations.

Method 0 can remain `SUCCESS_EXACT` while reporting nonzero end-effector
position error: its exact status covers only upper-arm direction, lower-arm
direction, and aligned hand orientation. Pinch-position error is an external
evaluation metric, not a Method-0 solve constraint.
