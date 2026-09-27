# TIAGo Steel engineering handoff

Start with [TIAGO_CONTEXT.md](TIAGO_CONTEXT.md) for the concise locked
coordinate, data-semantic, SEW, and task-frame conventions.

Code and `config.yaml` are the runtime authority; this file records the
validated scientific choices and outstanding limits. The project is
TIAGo-only: a fixed-base, single 7-DoF Steel arm with parallel gripper.

## Locked production path

- MuJoCo Menagerie `pal_tiago` v2026.9.2, commit `c96a32d`, vendored with its
  license and project patch record. MuJoCo 3.1.6 and Python 3.11 are the
  validated environment.
- Only `arm_1_joint`–`arm_7_joint` are solved. Base is fixed; the current torso lift is 0 m,
  head 0, and both finger joints 0.030 m. Replay rewrites fixed joints on
  each arm update. The validated J5/J6/J7 spherical wrist center is the
  position frame; `gripper_grasping_frame` is the orientation frame and its
  position is diagnostic only.
- The body-world to nominal-base/world transform is fixed across all frames:

  ```text
  p_nominal_world = R_base_from_body @ p_body + t_calibrated_m
  H_nominal_world = R_base_from_body @ H_body
  ```

  `R=I` and `t_calibrated_m=[0.5299635435,-0.0406565780,0.4919458315]` m
  align the robust human shoulder median to nominal J1 arm-root at torso 0.15 m.
  Human targets remain fixed. The robot-side
  `tiago.placement.j1_offset_world_m=[0,0,-0.20]` m lowers J1 by 0.20 m:
  torso 0.15 to 0 m plus a fixed base-origin shift of -0.05 m. Only explicit
  `calibration --write` may recompute the saved human transform; comparison
  and replay must not do so or move the base per frame. The base origin is
  below the ground plane; collision and dynamics are out of scope.
  Placement always uses the legal torso range first:
  `torso=clip(reference_torso + offset_z, 0, 0.35)` m, then assigns the
  remainder `offset_z - (torso - reference_torso)` to fixed base Z. X/Y
  offsets are applied directly to the fixed base. The solver model and
  independent oracle use the same decomposition.
- Stereo-SEW is locked to `S=J1` arm-root, `E=J4` anchor, and `W=wrist_center`
  with `e_t=[-1,0,0]`, `e_r=[0,1,0]`. A prior J1/J2 placement experiment kept
  `S=J2` in both conditions, so it does not decide the final S definition.
- `task_point.mode: wrist` takes human `Wrist_X/Y/Z` after the fixed transform.
  `tiago.task_frames.position: wrist_center` matches the robot wrist center
  to that point, while `tiago.task_frames.orientation: gripper_grasping_frame`
  matches its complete 3D rotation to transformed `Wrist_Rx/Ry/Rz`. The
  human `Hand_X/Y/Z` columns are not used. No grasping-frame position constraint
  is imposed at the human wrist.
- Capture orientation is locked: `Wrist_Rx/Ry/Rz` are extrinsic XYZ degrees
  from the hand-mounted Fork rigid body. The existing `R_input_align` maps
  canonical hand `+X` to Fork local `-Y`; do not refit it using TIAGo IK.
  `Hand_X/Y/Z = p_fork + R_fork @ [0,-0.1,0]` m is a virtual forward point,
  while `Wrist_X/Y/Z` is a separate marker. Never use `Hand - Wrist` as a
  calibration direction or treat Hand as an independent observation.
- The active backend is `tiago_semi_analytic`, called directly by the TIAGo
  pipeline. It has a real-model spherical-wrist hard gate. J1–J4 solve 3D
  wrist-center position plus 1D Stereo-SEW; J5–J7 use exact wrist branches.
  A projected out-of-limit branch may receive one bounded local 7D refinement
  before strict MuJoCo post-validation. It does not silently call the oracle
  or the separate numerical solver. `tiago_numerical` remains a development
  correctness reference and possible fallback candidate, not an automatic
  per-frame fallback.
- Acceptance thresholds are wrist-center position `<1 mm`, grasping-frame
  orientation `<1°`, and Stereo-SEW `<1°`, with physical MJCF arm-joint limits. Failure has `q=None`
  and an explicit status. Oracle search failure alone does not prove
  unreachability.

## Evidence and caveats

The corrected wrist-center task was run on consecutive frames 0–199 at the
former zero user offset with no oracle sweep. The pre-change
`output/tiago_first500/` results provide the
same-frame TCP-position baseline; the new results are in
`output/tiago_wrist_center_200/`. Both use fixed J1 calibration, fixed base,
the same Stereo-SEW definition, 16 recovery seeds, and strict thresholds.

| Metric on frames 0–199 | Old TCP position = Wrist | New wrist center = Wrist |
|---|---:|---:|
| Strict successes | 3/200 (1.5%) | 32/200 (16%) |
| Consecutive solved run | 144–146 | 144–175 |
| Solve time P50 / P95 | 140 / 203 ms | 173 / 218 ms |
| Adjacent solved-frame pairs | 2 | 31 |
| P95 maximum wrapped-joint step | 0.322°/frame | 1.132°/frame |

On the 32 new strict solutions, maximum independently replayed errors were
0.819 mm wrist-center position, 0.366° grasping-frame orientation, and
0.0063° Stereo-SEW. Median joint-limit margin is 0.416°; the minimum is
effectively zero. The actual grasping frame is about 0.206575 m from the
human wrist on those frames, as intended. Four targeted failed frames
(0, 143, 176, 199) gave exact solutions with *relaxed* limits under eight-seed
oracle search, but no bounded strict solution was found. This suggests a
limit-related difficulty; finite search does not prove physical infeasibility
or exclude solver misses on the other 164 failed frames. All four have bounded
wrist-position-only solutions below 1 mm, so their observed failure arises
only when orientation, SEW, and physical limits are enforced together. The
16% success rate is a substantial relative improvement, not sufficient full-trajectory
coverage. No full 4,344-frame run was authorized or performed.

At the former human-target `+0.20 m` Z offset, a bounded
production run on frames 0–1999 returned **690/2000 (34.5%)** strict
solutions and 1310 `oracle_inconclusive` solver failures. No oracle was run.
P50/P95 solve time was 263.6/617.4 ms; maximum accepted wrist-center,
orientation, and SEW errors were 0.966 mm, 0.681 degrees, and 0.664 degrees.
Success varied from 0 to 100 per 100-frame block. The result files are in
`output/tiago_z020_first2000/`; the earlier 95/100 placement-scan window
does not generalize to all 2000 frames. The new robot-side J1 `-0.20 m`
placement preserves the same arm/target relative geometry in FK; no new
2000-frame IK run has been made. No full 4,344-frame run was made.

A fixed-placement sensitivity scan on the complete second bite (frames
246–546, 301 frames including transfer and withdrawal) tested Z=-0.10/-0.20/-0.30 m,
then X and Y separately at the best Z=-0.20 m. The current [0,0,-0.20] m
placement had the highest strict count, 118/301. Its low/middle/high wrist-height
tertiles solved 0/101, 78/100, and 40/100. Moving Y to -0.30 m recovered
61/101 low frames but lost all 100 high frames and solved only 111/301 overall.
No single-axis candidate improved total success without shifting failures to
another height phase, so no combination was tested and production placement
was not changed. Per-offset summaries and frame-level results are in
`output/tiago_placement_bite2/`; the reproducible bounded runner is
`scripts/scan_tiago_fixed_placement.py`. No oracle was run, so solver failures
are not proven physically unreachable.

A follow-up fixed-base-yaw scan rotated `base_link` about its world-positioned
origin by -20/-10/0/+10/+20 degrees with offset [0,0,-0.20] m unchanged.
Strict counts on the same 301 frames were 92/112/118/117/104 respectively;
all five solved 0/101 low-wrist frames and retained a near-zero minimum J7
limit margin. The best nonzero yaw was +10 degrees, one success below yaw=0.
An experimental, opt-in wrist-branch selector that ranks strictly valid
solutions by minimum J5/J7 margin, then overall margin, then continuity was
compared at yaw=0 and +10. Both runs produced exactly the same per-frame
statuses and joint vectors as their original-solver controls. Among the 118
yaw=0 successes, no frame had a second strict in-limit wrist branch for its
selected J1-J4 prefix. Production yaw and solver selection remain unchanged.
Results are in `output/tiago_yaw_bite2/`; no oracle or other trajectory was run.

A subsequent limit-focused diagnostic used twelve equally spaced low-wrist
failures from bite 2 (six transfer, six withdrawal). An independent MuJoCo/7D
oracle with 128 physical-limit starts found 0/12 strict solutions; with the
existing relaxed range [-2π,2π] and 64 starts it found 12/12, all satisfying
the original wrist-center, full-orientation, and Stereo-SEW thresholds.
The first strict relaxed solution exceeded the J4 upper limit by 5.6–25.9°
on every frame; J5 exceeded its lower limit by about 58° on the first three
transfer frames, J6 exceeded its lower limit by 6.6–38.7° on four withdrawal
frames, and J7 was not violated in these relaxed solutions. A separate local
7D search seeded by clipping each relaxed solution to physical limits also
found 0/12 strict solutions. This is strong qualitative evidence of a
joint-range bottleneck in these sampled low-wrist poses, especially J4, but
finite search cannot prove that physical-limit solutions do not exist.
It does not support a specific, stable J5/J7-only explanation. See
`output/tiago_low_wrist_bite2_limits/` and
`scripts/diagnose_tiago_low_wrist_limits.py`; production settings were not
changed.

With capture alignment held fixed, a focused semi-analytic probe on failed
frames 0, 143, 176, 180, and 199 found one converged J1–J4 solution family
per frame and enumerated both nonsingular wrist branches. The closer branch
exceeded J7 by 1.39–32.01 degrees; the other exceeded J5 by 51.77–59.08
degrees. Bounded local refinement from both projected branches did not find
a strict solution. This is evidence of a limit conflict, not proof of
unreachability or exhaustive branch coverage. Profiling failed frame 180
showed 18 four-variable least-squares searches and 957 prefix-residual calls;
recovery multistarts, not wrist decomposition, dominate failure-path time.
The exact J6=0 singular case now enumerates feasible J5/J7 angle splits
within physical limits instead of relying on SciPy's arbitrary Euler split;
a synthetic in-limit wrist pose and the full semi-analytic solve have focused
regression coverage. A four-recovery-seed experiment preserved all 32 strict
solutions in frames 140–199, but the configured 16 seeds remain unchanged
because that window cannot establish recovery completeness elsewhere.

The committed [historical J1 validation report](output/tiago_j1_validation/report.json)
uses frames 640–699 consecutively and checks frames 1677 and 2967 separately.
It uses the **previous TCP-position task** and eight deterministic oracle
starts per frame. Its success counts and pose errors must not be interpreted
as evidence for the corrected wrist-center task.

| Metric on the 60-frame window | Result |
|---|---:|
| Robust shoulder to fixed J1 anchor | ~1.1e-16 m |
| Position-only fitting within 1 mm | 60/60 |
| Independent oracle strict solutions | 17 |
| Strict solutions found by active solver | 21 |
| Solver misses on oracle-reachable frames | 0 |
| Additional active-solver strict solutions missed by finite oracle search | 4 |
| Maximum independent MuJoCo errors: position / orientation / SEW | 0.984 mm / 0.708° / 0.111° |
| Solve time P50 / P95 | 159 / 172 ms |
| P95 maximum wrapped joint step across adjacent solved frames | 0.707°/frame |

The robot SEW singularity margin among strict solutions has median 0.612
(dimensionless). Joint-limit margin median is effectively 0°: many valid
solutions sit at a limit, and no dynamics/collision guarantee is implied.
Only 19 adjacent solved-frame pairs contribute to the continuity statistic.
The oracle's raw `joint_limit_failure` when the active solver has an
independently validated strict solution is a finite-search miss, not a
physical impossibility. The committed report keeps raw classification and
strict reachability evidence separate.

The separate [placement diagnostic](scripts/diagnose_tiago_placement.py)
now samples the wrist-center workspace and performs bounded position-only
fits. The committed `output/tiago_j1_placement/` plot/report used the old TCP
workspace and are historical; they were not regenerated in this change. The
selected J1 shoulder is fixed, but the selected fixed-skeleton WARP hypothesis
is **not** executable under the
configured 1 mm invariance threshold: its upper-link length varies by about
87.5 mm across the diagnostic sample. WARP corrected-skeleton code is
retained only as a generic research core/diagnostic.

No all-data oracle sweep has been performed. Do not infer full-dataset success
rates from either bounded window.

## Entry points and focused checks

- [`src/sew_mimic/tiago/model.py`](src/sew_mimic/tiago/model.py): fixed joints,
  MuJoCo FK, joint axes/anchors, TCP, SEW points, qpos mapping.
- [`src/sew_mimic/tiago/solver.py`](src/sew_mimic/tiago/solver.py): stateful
  semi-analytic and separate numerical solvers; [`wrist.py`](src/sew_mimic/tiago/wrist.py)
  implements the hard gate and exact decomposition.
- [`src/sew_mimic/tiago/oracle.py`](src/sew_mimic/tiago/oracle.py): independent
  MuJoCo/SciPy validation; never production fallback.
- [`src/sew_mimic/pipeline/tiago.py`](src/sew_mimic/pipeline/tiago.py): fixed
  transform, target preparation, benchmark, classification and metrics.
- [`scripts/compare_tiago.py`](scripts/compare_tiago.py),
  [`scripts/replay_tiago.py`](scripts/replay_tiago.py): bounded comparison and
  independent MuJoCo-FK replay.
- [`scripts/validate_tiago_j1.py`](scripts/validate_tiago_j1.py): bounded
  coherent J1 validation (not a full-dataset acceptance test).

Focused commands from the repository root:

```powershell
.venv\Scripts\python.exe -m sew_mimic.tiago.calibration --check
.venv\Scripts\python.exe -m pytest -q tests/test_tiago_core.py
.venv\Scripts\python.exe scripts\compare_tiago.py --start-frame 145 --max-frames 1 --output-dir output\tiago_check
.venv\Scripts\python.exe scripts\replay_tiago.py --results output\tiago_check\comparison_frames.csv --max-frames 1 --no-viewer
```

The CLI does not invoke the oracle unless explicitly requested. Keep the
single-robot direct path simple; do not reintroduce a Gen3 compatibility layer,
runtime plugin registry, base motion, torso/head/gripper retargeting, collision
control, or hardware integration.
