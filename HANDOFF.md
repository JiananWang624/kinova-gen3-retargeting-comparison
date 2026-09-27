# TIAGo Steel engineering handoff

Code and `config.yaml` are the runtime authority; this file records the
validated scientific choices and outstanding limits. The project is
TIAGo-only: a fixed-base, single 7-DoF Steel arm with parallel gripper.

## Locked production path

- MuJoCo Menagerie `pal_tiago` v2026.9.2, commit `c96a32d`, vendored with its
  license and project patch record. MuJoCo 3.1.6 and Python 3.11 are the
  validated environment.
- Only `arm_1_joint`–`arm_7_joint` are solved. Base is fixed; torso is 0.15 m,
  head 0, and both finger joints 0.030 m. Replay rewrites fixed joints on
  each arm update. `gripper_grasping_frame` is the sole TCP.
- The body-world to TIAGo-base transform is fixed across all frames:

  ```text
  p_base = R_base_from_body @ p_body + t_base_from_body
  H_base = R_base_from_body @ H_body
  ```

  `R=I` and `t=[0.5299635435,-0.0406565780,0.4919458315]` m align the
  robust human shoulder median to J1 arm-root. `user_xyz_offset_base_m` is
  expressed in TIAGo base coordinates and currently zero. Only explicit
  `calibration --write` may recompute saved values; comparison and replay
  must not do so or move the base.
- Stereo-SEW is locked to `S=J1` arm-root, `E=J4` anchor, and `W=wrist_center`
  with `e_t=[-1,0,0]`, `e_r=[0,1,0]`. A prior J1/J2 placement experiment kept
  `S=J2` in both conditions, so it does not decide the final S definition.
- The active backend is `tiago_semi_analytic`, called directly by the TIAGo
  pipeline. It has a real-model spherical-wrist hard gate. J1–J4 solve 3D
  wrist-center position plus 1D Stereo-SEW; J5–J7 use exact wrist branches.
  A projected out-of-limit branch may receive one bounded local 7D refinement
  before strict MuJoCo post-validation. It does not silently call the oracle
  or the separate numerical solver. `tiago_numerical` remains a development
  correctness reference and possible fallback candidate, not an automatic
  per-frame fallback.
- Acceptance thresholds are position `<1 mm`, TCP orientation `<1°`, and
  Stereo-SEW `<1°`, with physical MJCF arm-joint limits. Failure has `q=None`
  and an explicit status. Oracle search failure alone does not prove
  unreachability.

## Evidence and caveats

The committed [J1 validation report](output/tiago_j1_validation/report.json)
uses frames 640–699 consecutively and checks frames 1677 and 2967 separately.
It uses eight deterministic oracle starts per frame, not a broad sweep.

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
generates a MuJoCo-workspace/target-cloud plot and performs 100 bounded
position-only fits. Its generated J1 plot and report are in
`output/tiago_j1_placement/`. The selected J1 shoulder is fixed, but the
selected fixed-skeleton WARP hypothesis is **not** executable under the
configured 1 mm invariance threshold: its upper-link length varies by about
87.5 mm across the diagnostic sample. WARP corrected-skeleton code is
retained only as a generic research core/diagnostic.

No full 4,344-frame production retargeting run or all-data oracle sweep has
been performed. Do not infer full-dataset success rates from this window.

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
.venv\Scripts\python.exe scripts\compare_tiago.py --start-frame 688 --max-frames 1 --output-dir output\tiago_check
.venv\Scripts\python.exe scripts\replay_tiago.py --results output\tiago_check\comparison_frames.csv --max-frames 1 --no-viewer
```

The CLI does not invoke the oracle unless explicitly requested. Keep the
single-robot direct path simple; do not reintroduce a Gen3 compatibility layer,
runtime plugin registry, base motion, torso/head/gripper retargeting, collision
control, or hardware integration.
