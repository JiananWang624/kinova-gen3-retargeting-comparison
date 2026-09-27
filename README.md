# TIAGo Pro right-arm retargeting

This branch uses one fixed-base, arm-only **TIAGo Pro right arm** with the PAL Pro gripper. Only `arm_right_1_joint` through `arm_right_7_joint` are solved. The robot model, C++ solver, benchmark and replay are Pro-only; no Steel whole-body or Gen3 runtime path remains. The MJCF is a kinematic/visualization model, not a collision or dynamics model. Official-source pins, generation details and license are in [PROJECT_PATCHES.md](assets/pal_tiago_pro_arm/PROJECT_PATCHES.md).

## Locked task and placement

| Input | Pro target |
|---|---|
| Human `Wrist_X/Y/Z` | `arm_right_tool_link` **position** |
| Canonical hand orientation | `gripper_right_grasping_link` **orientation** |
| Human shoulder/elbow/wrist | Stereo-SEW with robot S=J1, E=J4, W=`arm_right_tool_link` |

`W=tool_link` is an engineering SEW point matched to the human wrist position, not a physical spherical-wrist center. Pro J6/J7 axes are 70 mm apart; the former Steel spherical-wrist analytic decomposition does **not** apply. The task is solved as seven variables/seven constraints by a stateful C++ solver; MuJoCo independently checks returned solutions. Strict acceptance is tool position `<1 mm`, grasp orientation `<1°`, Stereo-SEW `<1°`, and all physical joint limits.

The current fixed placement in [config.yaml](config.yaml) is:

```yaml
reference_shoulder_world_m: [0.09305, 0.014, 0.8875]
robot_world_offset_m: [0.0, 0.20, 0.0]
mounting_roll_deg: 90.0
mounting_pitch_deg: 0.0
mounting_yaw_deg: 0.0
```

Thus `J1_world = reference_shoulder_world + robot_world_offset = [0.09305, 0.214, 0.8875] m`. `Rx(+90°)` rotates only the fixed arm root, following the old Gen3 human-right-arm experiment layout. It is **not** the official whole-robot torso mount. Neither the base nor human calibration moves per frame. These placement values are locked for the current experiment; they are not a claim of globally optimal mounting.

Human CSV positions are millimetres and use the fixed body-world transform. `Wrist_Rx/Ry/Rz` are extrinsic XYZ **degrees** from a hand-held Fork rigid body. The orientation chain is:

```text
p_body  = R_body_from_csv @ (0.001 * p_csv_mm)
H_body  = R_body_from_csv @ R_wrist_csv @ R_input_align
p_world = R_world_from_body @ p_body + t_world_from_body
H_world = R_world_from_body @ H_body
```

`R_body_from_csv` and `R_input_align` are in `config.yaml`. The latter maps canonical hand +X to Fork local −Y, the captured forward direction. Do not refit it using IK outcomes or add an unverified robot-side orientation alignment. `Hand_X/Y/Z` is a *virtual* point projected from the Fork center by `R_fork @ [0,−0.1,0] m`; `Wrist_X/Y/Z` is a separate marker. Consequently `Hand − Wrist` is **not** a forward-direction measurement.

## Install and manual inspection

装配查看、指定帧数计算及 MuJoCo 回放的可复制指令另见 [COMMANDS.md](COMMANDS.md)。

Use Python 3.11 from the repository root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .[test]
.venv\Scripts\python.exe -m pytest -q tests/test_tiago_pro_arm.py tests/test_human_input.py tests/test_stereo_sew.py tests/test_solver_contracts.py tests/test_geometry.py
```

View the fixed root at q=0 without running IK:

```powershell
.venv\Scripts\python.exe scripts\show_tiago_pro_mounting.py --frame 0
```

Compute and replay exactly the first 100, 500, 1000 or 2000 frames by setting `--count N` and `--max-frames N` to the same value. Runs above 100 frames require `--manual-long-run` **in both commands**; this keeps focused automated validation bounded.

```powershell
.venv\Scripts\python.exe scripts\benchmark_tiago_pro.py --start-frame 0 --count 500 --manual-long-run --output output\tiago_pro_first500
.venv\Scripts\python.exe scripts\replay_tiago_pro.py --results output\tiago_pro_first500\window_frames.csv --max-frames 500 --manual-long-run
```

`--no-viewer` on replay validates saved rows against independent MuJoCo FK without opening a window. Failed frames hold the last valid arm pose in Viewer and show a red target; the arm's held pose is **not** a solved pose for those frames. The mounting Viewer shows world/base/J1/tool/target frames, human shoulder–elbow–hand segments and colored points; only the hand carries a text label. The purple sphere is robot J4 (elbow), not an extra target.

The retained [500-frame result](output/tiago_pro_first500/summary.json) records the current placement and calibration revision: **494/500 strict successes**, six solver failures without oracle proof, longest success segment 492 frames, and P50/P95 solve times about **0.22/0.23 ms** on the machine used. Timing is machine-dependent. This is a bounded manual-visualization run, not full-dataset coverage. A solver miss cannot be distinguished from infeasibility without a successful independent oracle; failed search alone does not prove unreachability.

For implementation details, limitations, and migration history, read [HANDOFF.md](HANDOFF.md). Stereo-SEW math is documented in [STEREO_SEW.md](docs/STEREO_SEW.md).
