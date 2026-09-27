# TIAGo retargeting context

This is the short semantic guardrail for future changes. It intentionally omits
installation, CLI usage, performance history, and general architecture; see
`README.md` and `HANDOFF.md` for those details.

| Mark | Meaning |
|---|---|
| **LOCKED** | Validated project convention; do not change as a tuning parameter. |
| **VERIFIED** | Supported by current code, tests, model geometry, or source data. |
| **DIAGNOSTIC** | Useful evidence, but not a proof or production definition. |
| **UNKNOWN** | Current evidence is insufficient. |

## 1. Human input data semantics

| CSV field | Actual meaning | Status |
|---|---|---|
| `Wrist_X/Y/Z` | Independent `FeederArm:Wrist` position marker in Motive global coordinates. | **VERIFIED** |
| `Wrist_Rx/Ry/Rz` | Orientation derived from the Fork rigid body fixed to the hand; it is not the local orientation of the Wrist position marker. | **LOCKED** |
| `Hand_X/Y/Z` | Virtual point generated from the Fork rigid-body center; not an independent marker. | **VERIFIED** |

The Fork was held in a repeatable pose relative to the hand, so its rigid-body
orientation is the project proxy for hand/wrist orientation. The preprocessing
formula is:

```text
R_fork = Rz(Wrist_Rz) @ Ry(Wrist_Ry) @ Rx(Wrist_Rx)
p_hand = p_fork + R_fork @ [0, -0.1, 0] m
```

The angles are extrinsic XYZ in degrees, equivalently
`R_fork = Rz @ Ry @ Rx`. Fork local `-Y` is therefore the captured hand/fork
forward direction. `p_fork` and `Wrist_X/Y/Z` are different origins:
**never use `Hand_XYZ - Wrist_XYZ` to infer hand forward**. `Hand_X/Y/Z` is
only a visualization and coordinate-consistency diagnostic, not an independent
calibration observation. **LOCKED**

## 2. Human coordinate chain

Positions first enter the body-world basis as:

```text
p_body = R_body_from_csv @ (0.001 * p_csv_mm)
```

Orientation uses the complete current chain:

```text
R_wrist_csv = extrinsic XYZ(Wrist_Rx, Wrist_Ry, Wrist_Rz)

H_body =
    R_body_from_csv
    @ R_wrist_csv
    @ R_input_align
```

```text
R_body_from_csv =
[[0, 0, 1],
 [1, 0, 0],
 [0, 1, 0]]

R_input_align =
[[ 0, 1, 0],
 [-1, 0, 0],
 [ 0, 0, 1]]
```

The right product defines the canonical hand basis:

```text
canonical hand +X = Fork local -Y   # forward
canonical hand +Y = Fork local +X   # palm convention
canonical hand +Z = Fork local +Z   # thumb/right-handed convention
```

`R_input_align` is **LOCKED**. Do not refit it from TIAGo reachability or IK
success rates.

## 3. Fixed human-to-TIAGo placement

The robust human shoulder reference is the coordinate-wise median of all 4,344
adapted `Shoulder_X/Y/Z` samples after mm-to-m conversion and
`R_body_from_csv`:

```text
shoulder_reference_body_m =
[-0.4369135435, 0.0546565780, 0.3955541685]
```

It is aligned once to the nominal TIAGo `arm_1_joint` anchor at `home_q_rad`
with the reference torso lift (0.15 m). The robot is then placed using a
separate fixed J1 offset. The J1 anchor remains the final SEW `S`:

```text
p_nominal_world = R_base_from_body @ p_body + t_calibrated
H_nominal_world = R_base_from_body @ H_body

t_calibrated = shoulder_reference_base
             - R_base_from_body @ shoulder_reference_body
p_J1_placed_world = shoulder_reference_base + j1_offset_world
```

For a requested robot J1 Z offset, the model uses the torso's MJCF legal range
first, then applies any remainder to the fixed base:

```text
torso_lift = clip(reference_torso_lift + j1_offset_z, 0, 0.35) m
base_translation_z = j1_offset_z - (torso_lift - reference_torso_lift)
```

X/Y offsets are applied directly to the fixed base. The production kinematics
and independent oracle implement this same split.

Current configured values are:

```text
base_world_yaw_deg       = 0
R_base_from_body         = I
shoulder_reference_base  = [0.09305, 0.014, 0.8875] m
t_calibrated             = [0.5299635435, -0.0406565780, 0.4919458315] m
j1_offset_world          = [0, 0, -0.20] m
current torso lift       = 0 m
current fixed base z     = -0.05 m
p_J1_placed_world        = [0.09305, 0.014, 0.6875] m
```

The human targets stay in the nominal world frame. The base is placed once
and must not move per frame. Normal pipeline/replay reads the saved transform
and never recalibrates. The below-ground base position is for kinematic
retargeting only; collision and dynamics are not evaluated. **LOCKED**

The committed placement report confirms shoulder median and nominal J1
coincide to numerical precision and reports successful bounded position-only
samples under an older placement.
**DIAGNOSTIC caveat:** that report was generated before the position task was
changed from grasping-frame/TCP position to wrist-center position. The current
diagnostic script uses the wrist center, but the committed report has not been
regenerated. Use the report to reject an obvious placement error, not as a
current task-frame benchmark.

## 4. Current TIAGo task

```text
Human Wrist_XYZ
    -> TIAGo J5/J6/J7 spherical wrist-center position

Canonical human hand orientation
    -> TIAGo gripper_grasping_frame orientation

Human Stereo-SEW
    -> TIAGo Stereo-SEW
```

The position task **does not** require
`gripper_grasping_frame position == Human Wrist_XYZ`. The grasping frame is
about 20.7 cm beyond the spherical wrist center; the former definition
artificially consumed workspace by placing the long gripper at the anatomical
wrist. The wrist center now owns position, while `gripper_grasping_frame` owns
full 3-D orientation. Its position remains diagnostic only. No additional
TIAGo robot-side orientation alignment is applied (`R_robot_align = I`).
**LOCKED**

## 5. TIAGo Stereo-SEW points

The authoritative configured selection is:

```text
S = J1 anchor / arm-root
E = J4 anchor
W = common J5/J6/J7 spherical wrist center
```

These are mathematical points for evaluating Stereo-SEW. Keep the following
concepts distinct:

- The human shoulder calibration reference is aligned to nominal J1; placed
  J1 differs by the fixed robot-side offset. Calibration and the SEW
  definition remain separate responsibilities.
- The wrist center is the common spherical-wrist axis point; it is not the
  gripper TCP/site.
- The arm-root/J1 point is not the wrist center or the gripper frame.

The selected tuple in `config.yaml` is `[J1, J4, wrist_center]`. **LOCKED**

## 6. Semi-analytic solver structure

```text
J1-J4: bounded numerical solve of wrist-center position + Stereo-SEW
J5-J7: analytic spherical-wrist orientation decomposition
```

Every result is post-validated against wrist-center position, official
grasping-frame orientation, Stereo-SEW, and physical MJCF joint limits. An
out-of-limit analytic branch may seed one bounded local 7-D refinement; there
is no silent per-frame numerical-oracle fallback.

At `J6 = 0`, wrist orientation fixes only the combined J5/J7 rotation. The old
Euler representative could exceed limits even when another J5/J7 split was
valid. The current decomposition must enumerate/reallocate a feasible J5/J7
split, optionally preferring the previous wrist state. The regression test
`test_singular_wrist_enumerates_in_limit_j5_j7_split` protects this behavior.
**VERIFIED**

## 7. Correct interpretation of failures

Position-only diagnostics cannot establish whether a fixed placement is good
for the complete task. The difficult constraint is the combination:

```text
wrist-center position
+ full gripper orientation
+ Stereo-SEW
+ TIAGo physical joint limits
```

Keep these outcomes separate:

| Outcome | Meaning |
|---|---|
| `solver_miss` | An independent oracle found a strict in-limit solution but production did not. |
| `joint_limit_failure` | Strict solution was found only after relaxing physical limits. |
| `workspace_unreachable` | Independent geometric evidence proves the position is outside the workspace. |
| `oracle_inconclusive` | Finite oracle search did not establish any stronger conclusion. |

Oracle/multistart failure alone never proves physical unreachability. Known
failed examples show J5/J7 limit conflicts, but bounded searches are not a
global infeasibility proof. **LOCKED classification rule / DIAGNOSTIC evidence**

## 8. Minimal Gen3 history

Gen3 used a geometry-derived robot-side `R_robot_align` to convert its raw
`pinch_site` tool frame into the canonical hand frame. TIAGo instead uses the
official `gripper_grasping_frame` directly and its current robot-side alignment
is identity. Do not transplant the Gen3 matrix.

Gen3 Exact-SEW also relied on the Gen3-specific R-2R-2R-2R axis structure and
virtual intersections. That analytic structure is not the TIAGo kinematic
model and cannot be copied into the TIAGo solver. **LOCKED**

## 9. Do not do these

| Do not | Why |
|---|---|
| Derive forward from `Hand - Wrist` | Hand originates at the Fork center; Wrist is a separate marker. |
| Refit `R_input_align` using IK success | It is a capture-coordinate convention, not a solver parameter. |
| Align gripper position to human Wrist | Position belongs to the spherical wrist center. |
| Move the base per frame | Placement is one fixed calibrated rigid transform. |
| Call oracle failure unreachable | Finite search is inconclusive without independent geometric proof. |
| Add the Gen3 `R_robot_align` to TIAGo | It encodes different robot tool geometry. |
| Conflate wrist center, TCP, arm-root, or SEW S | They are distinct model points with different roles. |

## 10. Sources of truth

| Authority | What it controls |
|---|---|
| `config.yaml` | Runtime task frames, transforms, locked SEW tuple, model and solver settings. |
| `src/sew_mimic/csv_adapter.py` | CSV position conversion and the human orientation chain. |
| `src/sew_mimic/pipeline/tiago.py` | Fixed body-to-base transform and human target construction. |
| `src/sew_mimic/tiago/model.py` | MuJoCo joint/frame mapping and S/E/W point evaluation. |
| `src/sew_mimic/tiago/solver.py` | Production task residuals, strict validation, and solver flow. |
| `src/sew_mimic/tiago/wrist.py` | Spherical wrist center, hard gate, and analytic branch enumeration. |
| `src/sew_mimic/tiago/calibration.py` | Explicit calibration check/write behavior. |
| `scripts/diagnose_tiago_placement.py` | Current placement and wrist-center position-only diagnostic. |
| `tests/test_tiago_core.py` | Task-frame, wrist hard-gate, J6=0 branch, and fixed-calibration regressions. |
| `F:\code\optitrack_data_preprecessing\process_fork_hand.py` | Original Fork Euler conversion and virtual Hand generation. |
| `output/tiago_j1_placement/report.json` | Historical placement/TCP-position diagnostic; see caveat above. |
| `README.md`, `HANDOFF.md` | User workflow, bounded evidence, engineering state, and remaining limits. |

When prose conflicts with executable definitions, use current `config.yaml`
and current tested implementation, then update the prose. The two known naming
traps are that CSV `Wrist_R*` describes the Fork orientation while CSV
`Wrist_*` describes an independent position marker, and that the committed
placement report predates the wrist-center task definition.
