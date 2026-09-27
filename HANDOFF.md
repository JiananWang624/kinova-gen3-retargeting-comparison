# TIAGo Pro arm-only handoff

## Current authority

- Branch: `tiago-pro` (local). Runtime authority: `config.yaml`; current calibration revision `pro-rx90-y020-v1`.
- Fixed PAL Pro **right** arm, default `limits_v2=False`, `arm_type=tiago-pro`, `wrist_model=spherical-wrist`, tool changer and fixed Pro gripper. Exact PAL source commits, model-generation patch and license: `assets/pal_tiago_pro_arm/PROJECT_PATCHES.md`. Do not substitute standalone continuous-joint limits, Steel joints, or PAL torso-mount rotation.
- Root-only mounting: `Rx(+90°)`, no pitch/yaw; J1 world = robust shoulder world `[0.09305, 0.014, 0.8875]` + fixed offset `[0, 0.20, 0]` = `[0.09305, 0.214, 0.8875]` m. This reproduces the old Gen3 **spatial layout**, not its joint geometry. `arm_base` and J1 anchors coincide in the reduced model. Do not interpret pedestal mesh depth as J1 placement error; inspect numerical J1 anchor.
- Human-to-world transform and root placement are both fixed over a run. Do not re-calibrate or move base per frame. Robust shoulder median in body-world is `[-0.4369135435, 0.0546565780, 0.3955541685]` m; configured `R_world_from_body=I`, `t_world_from_body=[0.5299635435,-0.0406565780,0.4919458315]` m.
- Task: human Wrist position → `arm_right_tool_link` position; canonical hand orientation → `gripper_right_grasping_link` rotation; Stereo-SEW S=J1/E=J4/W=`arm_right_tool_link`. W is an **engineering correspondence point**, not a spherical wrist center. Tool link and gripper base have identity relative transform in the pinned model; the official grasp frame has its own fixed transform, including 0.157157 m translation and `rpy=(0,-1.57,0)` rad. Do not force grasp-frame **position** to human wrist.
- `Wrist_Rx/Ry/Rz` comes from the Fork rigid body, not the separate wrist position marker. Extrinsic XYZ degrees: `R_wrist=Rz(Rz) @ Ry(Ry) @ Rx(Rx)`; `H_body=R_body_from_csv @ R_wrist @ R_input_align`. Canonical hand +X maps to Fork local −Y. The synthetic Hand point is `p_fork + R_fork @ [0,-0.1,0] m`; **never** infer forward from `Hand − Wrist`, or fit `R_input_align` by solver success. Pro has no extra Gen3 `R_robot_align`.

## Model and solver facts

- Generated MJCF contains exactly seven arm qpos; the gripper is fixed. MuJoCo FK and the pinned Xacro chain agree on 100 deterministic legal configurations; Pro model and limits are checked at load time. The Pro tool frame and gripper-base frame coincide only because the pinned fixed transform is identity; retain the distinct frame names.
- J6/J7 axes are separated by **70 mm**. The old Steel J1–J4 plus analytic J5–J7 common-center decomposition is inapplicable; do not relax the geometric hard gate or invent a wrist center. The Pro path uses a cached-geometry C++ seven-dimensional bounded solver, previous-frame continuation and deterministic failure recovery. MuJoCo performs independent final FK/error checks. No per-frame numerical oracle or hidden fallback.
- Strict thresholds: `<1 mm` tool position, `<1°` grasp orientation, `<1°` Stereo-SEW, physical PAL joint limits. Fork orientation is a full 3D hand-pose proxy, but exact anatomical palm/thumb axis alignment has no independent marker calibration. Preserve the current convention unless new capture evidence exists.
- Retained current-placement result: `output/tiago_pro_first500/` has **494/500** strict successes, six unresolved failures, longest continuous success run 492, joint-step P95 about `0.04796 rad`, and P50/P95 solver time about `0.22/0.23 ms` on the current machine. `summary.json` stores model hash, calibration revision and placement; replay checks saved results against current MuJoCo FK. This run is not a physical-reachability proof or a whole-dataset benchmark. Oracle search failure must remain inconclusive unless independent evidence proves otherwise.

## Why the migration happened

Steel had many low-wrist failures under the combined strict wrist-position, full orientation, SEW and physical joint-limit task, though position alone was generally reachable. Prior placement/yaw/branch/order experiments did not resolve it. In 12 sampled low-wrist failures, relaxed-limit exact solutions violated mainly J4 upper limits, sometimes J5/J6; bounded physical-limit oracle search found none, **not** a mathematical impossibility proof. This motivated Pro geometry, not a change to human coordinate semantics. Steel's results, scanners and whole-body code have been removed from this branch; use Git history or the `tiago` branch if historical comparison is needed. Avoid restoring Steel/Pro runtime compatibility layers.

## Working map and cautions

| Area | Authoritative files |
|---|---|
| PAL model and provenance | `assets/pal_tiago_pro_arm/`, `scripts/build_tiago_pro_arm.py` |
| 7D compiled core | `cpp/pro_7d_core.cpp`, `src/sew_mimic/pro/{model,solver}.py` |
| Human input / task / evaluation | `src/sew_mimic/{csv_adapter,human_input}.py`, `src/sew_mimic/pipeline/pro.py` |
| Mounting, compute, replay | `scripts/show_tiago_pro_mounting.py`, `scripts/benchmark_tiago_pro.py`, `scripts/replay_tiago_pro.py` |
| Focused regression | `tests/test_tiago_pro_arm.py` and shared human/SEW/contract tests |

The visualization-only MJCF gives a dark checkerboard floor and lighting. Mounting Viewer runs at q=0 with no IK; replay draws world/base/J1/tool/target frames, robust shoulder, robot J4 and human shoulder–elbow–hand lines. The purple sphere marks **robot J4**. Failed replay frames hold the last valid q, so inspect status/target color rather than interpreting a still arm as a valid solution.

Do not scan placements or run a full 4,344-frame trajectory merely to validate a small code change. Use focused tests and the retained first-500 replay. If placement, frames, model or calibration intentionally change, bump `calibration.revision` and regenerate results; do not replay historical q under a new transform. If Pro solver coverage becomes a research issue, distinguish **solver failure**, **oracle-proven solver miss**, **joint-limit evidence**, and **independently proven workspace unreachability**. No collision avoidance, dynamics, ROS or hardware control is implemented.
