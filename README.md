# TIAGo Steel arm retargeting

This repository retargets a human arm trajectory to the fixed-base, single-arm
**TIAGo Steel** MuJoCo model. The only optimized joints are `arm_1_joint` through
`arm_7_joint`; torso lift is fixed at 0.15 m, head at neutral, and each parallel
gripper finger at 0.030 m. The base remains on the ground and never moves to
follow a target. All pose metrics use the model's `gripper_grasping_frame` site
as the single TCP.

The model and assets are vendored from [MuJoCo Menagerie `pal_tiago`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/pal_tiago)
at commit `c96a32d` (v2026.9.2). See [the model notes](assets/pal_tiago/README.md),
[project patches](assets/pal_tiago/PROJECT_PATCHES.md), and
[license](assets/pal_tiago/LICENSE). The grasping site follows the
[PAL gripper definition](https://github.com/pal-robotics/pal_gripper/blob/humble-devel/pal_gripper_description/urdf/gripper.urdf.xacro).

## Fixed geometry and calibration

`config.yaml` is the runtime authority. The CSV adapter first converts the
human data into body-world coordinates. One stored rigid transform then maps
every frame into TIAGo base coordinates:

```text
p_base = R_base_from_body @ p_body + t_base_from_body
H_base = R_base_from_body @ H_body
```

The current rotation is identity. The robust median human shoulder is
`[-0.4369135435, 0.0546565780, 0.3955541685]` m in body-world; the TIAGo
J1 arm-root is `[0.09305, 0.014, 0.8875]` m in base coordinates. Their
one-time alignment gives `t_base_from_body = [0.5299635435, -0.0406565780,
0.4919458315]` m with zero user offset. Normal comparison and replay only
read this transform; they never recalibrate or move the base per frame.

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
allowed; every returned solution is checked against independent MuJoCo pose,
SEW, and joint-limit thresholds. There is no silent per-frame oracle or
numerical-backend fallback. The separate `tiago_numerical` backend remains a
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
.venv\Scripts\python.exe scripts\compare_tiago.py --start-frame 640 --max-frames 60 --output-dir output\tiago_run
.venv\Scripts\python.exe scripts\replay_tiago.py --results output\tiago_run\comparison_frames.csv --max-frames 60 --no-viewer
```

Omit `--no-viewer` to view the fixed full robot, human overlay, S/E/W, target,
and actual TCP in MuJoCo. Replay independently recomputes and checks saved
errors from MuJoCo FK. To inspect calibration and placement without an IK run:

```powershell
.venv\Scripts\python.exe -m sew_mimic.tiago.calibration --check
.venv\Scripts\python.exe scripts\diagnose_tiago_placement.py
```

The placement diagnostic writes a 3D plot and JSON report. Add `--viewer` for
an interactive full-robot point-cloud view. `calibration --write` is the only
explicit command that rewrites stored model/calibration values; it preserves
the locked J1/J4/wrist-center SEW definition.

## Bounded validation and limits

The committed [J1 validation report](output/tiago_j1_validation/report.json)
covers consecutive frames 640–699, plus two historical boundary frames. The
robust shoulder and J1 arm-root coincide to floating-point precision;
position-only fitting succeeds on all 60 continuous targets. Twenty-one
frames have strict full pose+SEW+physical-limit solutions. The independent
oracle found 17 of them; the semi-analytic solver missed none and found four
additional strict solutions that the finite oracle search missed. Its returned
solutions stayed below **1 mm position, 1° orientation, and 1° SEW error**.
Solve-time P50/P95 on this window was approximately **159/172 ms**; the P95
maximum wrapped joint step across adjacent solved frames was **0.71°/frame**.

Most strict solutions are close to a joint limit (median limit margin is
effectively zero). An oracle search failure is not a proof of unreachability;
the report preserves its raw classification and separate strict-solution
evidence. No full 4,344-frame retargeting run or all-data oracle sweep was
performed. See [HANDOFF.md](HANDOFF.md) for engineering details and boundaries.
