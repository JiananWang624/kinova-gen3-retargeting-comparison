# TIAGo Steel arm retargeting

For the short, authoritative map of human-data semantics, coordinate frames,
SEW points, task frames, and historical traps, read [TIAGO_CONTEXT.md](TIAGO_CONTEXT.md) first.

This repository retargets a human arm trajectory to the fixed-base, single-arm
**TIAGo Steel** MuJoCo model. The only optimized joints are `arm_1_joint` through
`arm_7_joint`; under the current placement torso lift is fixed at 0 m, head at
neutral, and each parallel gripper finger at 0.030 m. The fixed base origin is
5 cm below the MuJoCo ground plane and never moves per frame. The position
task aligns the validated J5/J6/J7 spherical wrist center with human
`Wrist_X/Y/Z`; the orientation task aligns the model's
`gripper_grasping_frame` with the transformed human wrist Euler orientation.
The grasping frame's position is diagnostic, not a human-wrist position target.

The model and assets are vendored from [MuJoCo Menagerie `pal_tiago`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/pal_tiago)
at commit `c96a32d` (v2026.9.2). See [the model notes](assets/pal_tiago/README.md),
[project patches](assets/pal_tiago/PROJECT_PATCHES.md), and
[license](assets/pal_tiago/LICENSE). The grasping site follows the
[PAL gripper definition](https://github.com/pal-robotics/pal_gripper/blob/humble-devel/pal_gripper_description/urdf/gripper.urdf.xacro).

## Fixed geometry and calibration

`config.yaml` is the runtime authority. The CSV adapter first converts the
human data into body-world coordinates. One stored rigid transform then maps
every frame into the nominal TIAGo base frame, which coincides with MuJoCo
world before robot placement:

```text
p_nominal_world = R_base_from_body @ p_body + t_calibrated_m
H_nominal_world = R_base_from_body @ H_body
```

The current rotation is identity. The robust median human shoulder is
`[-0.4369135435, 0.0546565780, 0.3955541685]` m in body-world; the TIAGo
J1 arm-root is `[0.09305, 0.014, 0.8875]` m at the reference torso pose. Their
one-time alignment gives `t_calibrated_m = [0.5299635435, -0.0406565780,
0.4919458315]` m. Human targets retain this transform. The fixed robot-side
`tiago.placement.j1_offset_world_m = [0,0,-0.20]` m lowers J1 relative to
its reference pose: torso lift goes from 0.15 m to 0, and the fixed base
origin goes from ground level to -0.05 m. Normal comparison and replay never
recalibrate or move the base per frame. This below-ground placement is a
kinematic experiment, not a collision-safe physical mounting proposal.

Placement decomposes the requested J1 offset by using the torso's legal range
first. For Z, `torso = clip(reference_torso + offset_z, 0, 0.35)` m; the base
then receives the remaining `offset_z - (torso - reference_torso)`. X/Y
offsets go directly to the fixed base. This rule is shared by the kinematics
model and validation oracle.

`tiago.task_frames` names the two robot evaluation frames explicitly:
`position: wrist_center` (the verified J5/J6/J7 common center) and
`orientation: gripper_grasping_frame`. `task_point.mode: wrist` selects the
human CSV wrist position; `Hand_X/Y/Z` is not used. Strict acceptance uses
`<1 mm` wrist-center position, `<1°` grasping-frame orientation, `<1°`
Stereo-SEW, and physical joint limits.

The CSV `Wrist_Rx/Ry/Rz` are the hand-mounted Fork rigid-body orientation,
exported as extrinsic XYZ Euler angles in degrees. The capture convention
locks canonical hand `+X` to Fork local `-Y` through `R_input_align` in
`config.yaml`; it is not fitted from TIAGo IK success. `Hand_X/Y/Z` is a
virtual point, `p_fork + R_fork @ [0,-0.1,0]` m. CSV `Wrist_X/Y/Z` is a
separate marker, so `Hand - Wrist` is not that forward vector. The virtual
Hand point is useful only for visualization and coordinate-consistency checks,
not as an independent calibration observation.

The locked Stereo-SEW points are **S=J1 arm-root, E=J4 anchor, W=the validated
spherical-wrist center**. The reference pair is `e_t=[-1,0,0]`, `e_r=[0,1,0]`.
The former J1/J2 placement experiment did **not** compare these shoulder
definitions: it changed the alignment while keeping `S=J2`. It is not used
to select the final geometry.

## Methods

| Role | Implementation | Status |
|---|---|---|
| Method 0 | TIAGo SEW-Mimic baseline | Executable comparison |
| Method 1 | Generic WARP corrected-skeleton core | Diagnostic only: selected TIAGo fixed skeleton fails invariance |
| Method 2 | `tiago_sew`, stateful semi-analytic IK | Active fast path |
| Method 3 | Independent MuJoCo/SciPy numerical oracle | Validation only |

Method 2 numerically solves J1–J4 for wrist-center position and Stereo-SEW,
then enumerates exact spherical-wrist branches for J5–J7. A real-model
geometric hard gate must pass before this backend starts. If an analytic wrist
branch just exceeds a physical limit, one bounded local 7D correction is
allowed; every returned solution is checked against MuJoCo wrist-center
position, grasping-frame orientation, SEW, and joint-limit thresholds. There is
no silent per-frame oracle or numerical-backend fallback. The separate
`tiago_numerical` backend remains a
correctness reference and possible fallback candidate, not the active path.

## Install and run

Use Python 3.11 and MuJoCo 3.1.6 from the repository root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .[test]
.venv\Scripts\python.exe -m pytest -q tests/test_tiago_core.py
```

The default CLI processes a bounded number of frames without an oracle:

```powershell
.venv\Scripts\python.exe scripts\compare_tiago.py --start-frame 144 --max-frames 32 --output-dir output\tiago_run
.venv\Scripts\python.exe scripts\replay_tiago.py --results output\tiago_run\comparison_frames.csv --max-frames 32 --no-viewer
```

Omit `--no-viewer` to view the fixed full robot, human overlay, S/E/W, human
wrist target, robot wrist center, and grasping frame in MuJoCo. Replay
independently recomputes and checks saved errors from MuJoCo FK. To inspect
calibration and placement without an IK run:

```powershell
.venv\Scripts\python.exe -m sew_mimic.tiago.calibration --check
.venv\Scripts\python.exe scripts\diagnose_tiago_placement.py
```

The placement diagnostic writes a 3D plot and JSON report. Add `--viewer` for
an interactive full-robot point-cloud view. `calibration --write` is the only
explicit command that rewrites stored model/calibration values; it preserves
the locked J1/J4/wrist-center SEW definition.

## Bounded validation and limits

The same consecutive CSV frames 0–199 were compared before and after the
task-frame correction at the former zero user offset, with fixed calibration
and SEW. Strict successes rose
from **3/200** (old TCP-position target) to **32/200** (new wrist-center target),
forming one continuous solved run, frames 144–175. Solve-time P50/P95 changed
from **140/203 ms** to **173/218 ms**. Across the new run's 31 adjacent solved
pairs, P95 maximum wrapped-joint step was **1.13°/frame**. Maximum independently
recomputed errors were **0.819 mm wrist-center position, 0.366° grasping-frame
orientation, and 0.0063° SEW**. The 168 remaining failures have not been proven
unreachable. Four targeted failed-frame oracle probes found exact solutions
only after relaxing limits; finite search cannot prove no in-limit solution.
Those historical comparison results are in `output/tiago_wrist_center_200/`.

In the historical human-target `+0.20 m` Z-offset run, the production
semi-analytic backend returned strict solutions on **690/2000 (34.5%)** of
CSV frames 0–1999. P50/P95 solve time was **264/617 ms**. The other 1310
frames are `oracle_inconclusive`, not proven unreachable; no oracle was run.
Results are in `output/tiago_z020_first2000/`. Success varied sharply between
100-frame blocks, so the earlier 95/100 placement-scan window must not be
treated as a full-trajectory success estimate. The current fixed robot-side
J1 `-0.20 m` placement is geometrically equivalent for arm FK and relative
target pose, but the 2000-frame solver has not been rerun under this config.
Replay rejects results saved with an earlier calibration revision.

The committed [older J1 validation report](output/tiago_j1_validation/report.json)
uses the *previous* TCP-position task and is retained only as historical
evidence of calibration, SEW geometry, and the wrist hard gate. Its success
counts and pose errors must not be compared with the new task definition.
No full 4,344-frame retargeting or all-data oracle sweep was performed. See
[HANDOFF.md](HANDOFF.md) for engineering details and remaining limits.
