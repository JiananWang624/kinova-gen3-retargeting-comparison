# Engineering handoff

This is the final implementation snapshot for the Kinova Gen3 SEW retargeting
repository. Detailed derivations and phase evidence remain in `docs/`; this file
records the operational state a future developer needs.

## Environment and validation state

- Validated on Windows with Python 3.11.15 and MuJoCo 3.1.6.
- Dependencies are listed in `requirements.txt`; MuJoCo is pinned, while the
  remaining packages are not fully locked.
- The Phase 9 starting point had 289 passing tests.
- The Phase 9 final suite has 291 passing tests (`52.09 s` in a recorded run).
- Run the final suite with `.venv\Scripts\python.exe -m pytest -q`.
- No CI workflow was added because this exact dependency set and MuJoCo setup
  have only been validated in the Windows project environment. Avoid adding a
  speculative cross-platform workflow without first reproducing the suite on
  its target runner.

## Final architecture

```text
src/sew_mimic/
  common/         HumanArmTarget, ExactSewTarget, status/result contracts,
                  task-point calculation, authoritative pinch-site metrics
  sew/            legacy adapter, StereoSew, validated Gen3 SEW geometry
  exact/          production Method 2 backend/selection/solver; lazy Method 3
  warp/           generic corrected-skeleton core and compatibility gate
  pipeline/       mounted CSV preparation, method dispatch, evaluation, summary
  visualization/  precomputed replay, consistency gate, display overlays
```

Core legacy files (`geometry.py`, `kinematics.py`, `human_input.py`,
`csv_adapter.py`, `mounting.py`, `retarget.py`, and `metrics.py`) remain the
verified baseline. Method 2 production modules do not import the numerical
oracle. The pipeline loads Method 3 only when dispatching an explicitly
requested oracle run.

## Capability contract

| Method | Name | Executable on current Gen3 | Role/status |
|---|---|---:|---|
| 0 | `sew_mimic` | Yes | Baseline |
| 1 | `warp_csew` | No | Generic core reproduced; `fixed_link_geometry_incompatible` |
| 2 | `exact_sew` | Yes | Recommended |
| 3 | `numerical_oracle` | Yes | Validation only |

`pipeline.capability_metadata()` is the machine-readable authority for this
table. Do not add fake Method 1 Gen3 rows or silently route Method 2 through
Method 3.

## Coordinate and task conventions

- Positions are metres; joint angles and internal angular errors are radians.
- CSV positions and wrist Euler rotations retain the established adapter
  convention in `config.yaml`: extrinsic XYZ, input angles in degrees,
  `rotation_body_from_csv`, then right-multiplied `rotation_input_align`.
- The fixed Gen3 root uses the established `Rx(+90deg)` mounting and configured
  world offset. Per-frame root motion is prohibited for executable fixed-base
  results.
- Human targets are transformed into native Gen3 base frame before solving and
  evaluation.
- The default task point is `t_h = Wrist_XYZ`. Optional calibrated mode is
  `t_h = w_h + H_h @ p_human_WT`; the default offset is zero.
- Target orientation is the canonical human hand frame. Robot orientation is
  the MuJoCo `pinch_site` rotation right-multiplied by the established
  `R_robot_align`.
- Final position and orientation errors always come from real MuJoCo-derived
  `Gen3Kinematics` pinch-site FK.
- Display offsets are applied only to copied visualization geometry.

## Stereo-SEW and Gen3 geometry

The project reference pair, in the shared native Gen3 base frame, is:

```text
e_t = [0, 0, -1]
e_r = [1, 0, 0]
```

Changing either vector changes the zero/sign convention for `psi` and requires
new mathematical validation. The validated R-2R-2R-2R PoE representation
matches MuJoCo FK, including the established Gen3 axes and negative h3/h5 proxy
signs. Do not change those conventions from visual intuition.

The WARP compatibility audit found that the selected Gen3 upper-arm proxy
length varies by approximately `0.19513668 m` over deterministic valid samples.
That violates WARP's fixed-link model. Forearm and wrist-to-task variations are
only floating-point noise, but they do not repair the upper-arm incompatibility.

## Exact-SEW solver contract

Method 2 targets:

```text
p_pinch(q) = t_h
R_pinch(q) @ R_robot_align = H_h
psi_robot(q) = psi_h
```

An exact candidate must satisfy the project acceptance thresholds:

- position error `< 1e-3 m` (1 mm)
- orientation error `< 1 degree`
- Stereo-SEW error `< 1 degree`

Method 2 returns only `SUCCESS_EXACT` for accepted configurations; it never
labels least-squares output exact. Failures retain explicit shared statuses and
`q=None`.

The official fixed-grid root search remains available as
`reference_fixed_grid`. Production uses the deterministic `event_aware` search
because the fixed grid missed a validated narrow exact branch. This is a
documented, tested departure from the official sampling policy, not a change to
the R-2R-2R-2R equations. Candidate generation is independent of branch
selection. Canonical selection is deterministic and history-free; continuous
selection greedily chooses the wrapped nearest configuration to the last
successful `q`.

The optimized backend averaged about 2.4 seconds per frame on a ten-frame
validation in the validated Windows environment. Runtime remains
environment-dependent; do not run all 4,344 input frames as routine validation.

## Reproduction commands

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\measure_baseline.py --input data\test.csv
.venv\Scripts\python.exe scripts\validate_gen3_sew_geometry.py
.venv\Scripts\python.exe scripts\validate_warp_core.py
.venv\Scripts\python.exe scripts\validate_exact_sew_backend.py --count 3
.venv\Scripts\python.exe scripts\validate_exact_sew_solver.py --input data\test.csv --count 3 --oracle-count 3
.venv\Scripts\python.exe scripts\compare_retargeters.py --input data\test.csv --methods sew_mimic exact_sew numerical_oracle --exact-branch-policy continuous --max-frames 100 --oracle-max-frames 10
.venv\Scripts\python.exe scripts\replay_compare.py --input data\test.csv --results output\comparison_frames.csv --method exact_sew --max-frames 3 --no-viewer
```

The comparison command regenerates ignored output CSV/JSON files. Replay uses
those precomputed joint values, checks them against the authoritative evaluator,
and does not run IK.

## Known remaining limitations

- Method 2 event-aware runtime remains approximately 2.4 seconds per frame in
  the latest short validation and is still unsuitable for real-time control.
- The anatomical meaning of `Wrist_XYZ` is dataset-dependent until calibrated.
- Generic WARP cannot be applied exactly to the current fixed-link Gen3 model.
- Visualization normally requires precomputed Method 2 results.
- Collision avoidance, dynamics, global trajectory optimization, ROS, and real
  robot execution are outside the validated scope.

No correctness blocker remains after the Phase 9 release validation. See
`docs/FINAL_ENGINEERING_REPORT.md` for the recorded evidence.
