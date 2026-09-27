# Engineering handoff for coding agents

Read this file before modifying the repository. It is the concise operational
snapshot; `README.md` is the user-facing guide and `docs/` contains derivations
and phase evidence. If prose conflicts with behavior, use this authority order:

1. current code and tests;
2. `config.yaml` for runtime configuration;
3. this handoff;
4. `README.md` and detailed documentation.

Historical phase wording may describe an implementation that has since been
replaced. Do not restore an old path merely because it remains discussed in a
phase document.

## 1. Current snapshot

- Snapshot reviewed: 2026-09-27.
- HEAD at review start: `52d74327d98a89b82012d311dde8d6dba066b843`
  (`feat: validate Gen3 feasibility for Method 1 WARP-cSEW`).
- Worktree was clean before this documentation update.
- Required runtime: Python `>=3.11,<3.12`; validated MuJoCo version: `3.1.6`.
- Build: scikit-build-core + pybind11 + C++17 private extension.
- Current collection: 289 pytest tests. This documentation-only review ran
  collection and focused consistency checks, not the full suite.
- Recommended executable Gen3 retargeter: **Method 2 Exact-SEW**.

Important recent evolution:

- Method 0 remains the unchanged regression baseline.
- Method 2 was reproduced, then cut over to one compiled stateful production
  path for trajectory speed.
- Method 3 remains a lazily loaded validation oracle only.
- Method 1 has a validated generic WARP core and Gen3 geometry investigations,
  but no reproduced executable Kinova trajectory path in this repository.

## 2. Capability contract

| Method | Machine name | Current repository capability | Role |
|---|---|---|---|
| 0 | `sew_mimic` | Executable on Gen3 | Regression baseline |
| 1 | `warp_csew` | Generic core only; Kinova path not reproduced | Paper comparison/research |
| 2 | `exact_sew` | Executable on fixed-base Gen3 | Recommended production method |
| 3 | `numerical_oracle` | Executable on Gen3 | Validation only |

`sew_mimic.pipeline.capability_metadata()` is the machine-readable capability
authority. The pipeline creates no Method 1 trajectory rows.

### WARP wording must remain precise

Do **not** say that Kinova Gen3 is inherently incompatible with WARP. The WARP
materials explicitly demonstrate Dual-Kinova3, so an author-side Kinova path
exists. What this repository established is narrower:

- the tested `S1/E45/W67` model fails fixed-link invariance;
- `S23/E45/W67` has invariant virtual lengths/offset but its shoulder moves
  with q1 and does not directly satisfy the required fixed model;
- the best tested global h3/h5 proxy fit retained about `15.349 mm` mean and
  `25.901 mm` maximum independent-validation position error;
- the tested candidate set is classified `NO_USEFUL_FIXED_SKELETON` under the
  project's 1 mm practical threshold;
- public material does not specify enough Kinova-specific S/E/W points,
  calibration, or adaptation detail to reconstruct the demonstrated path.

Therefore: generic WARP-cSEW is reproduced, tested Gen3 hypotheses are
documented, but a Gen3 WARP trajectory method is **not reproduced here**. Do
not fabricate an approximate Method 1 or present a tested hypothesis as the
authors' parameterization. See `docs/WARP_CSEW_CORE.md`.

## 3. Non-negotiable scientific conventions

These are validated definitions, not tuning knobs. Change them only with a
mathematical derivation and new regression evidence.

### Input and mounting

- CSV positions are scaled by `0.001` from millimetres to metres.
- Wrist Euler convention is extrinsic `xyz`, in degrees.
- CSV-world to body/MuJoCo-world rotation:

  ```text
  R_body_from_csv = [[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]]
  ```

- Motive wrist rigid-body to canonical hand-frame right alignment:

  ```text
  R_input_align = [[ 0, 1, 0],
                   [-1, 0, 0],
                   [ 0, 0, 1]]
  ```

- Fixed robot mounting is `Rx(+90deg)`.
- Current physical root world offset is `[0.0, 0.15, 0.2] m`.
- The Gen3 root is fixed for an executable trajectory. Never move it per frame
  to make an unreachable target appear reachable.

Human targets are expressed in native Gen3 base frame before solving:

```text
p_base = R_world_from_base.T @ (p_world - p_world_of_base)
H_base = R_world_from_base.T @ H_world
```

The configured world offset is part of the physical mounting, not a display
offset. Camera, overlay, and human-display offsets must never enter solver
inputs or evaluation.

### Task point and end-effector orientation

- Default task point: `t_h = Wrist_XYZ` (`task_point.mode: wrist`).
- No anatomical wrist-to-palm distance is assumed by default.
- Optional calibrated mode is
  `t_h = w_h + H_h @ p_human_WT`, with the offset in canonical hand frame.
- Physical robot position is the MuJoCo-derived `pinch_site` position.
- Aligned robot hand orientation is
  `R_pinch_aligned(q) = R_pinch(q) @ R_robot_align`.
- Final pose errors must always be recomputed using real
  `Gen3Kinematics`/MuJoCo `pinch_site` FK, never only internal PoE residuals.

### Stereo-SEW and Gen3 geometry

The project reference in native Gen3 base frame is fixed to:

```text
e_t = [0, 0, -1]
e_r = [1, 0, 0]
```

Changing either vector changes the numerical zero/sign of `psi`. The validated
Gen3 R-2R-2R-2R axes, PoE offsets, `R_7T`, `R_robot_align`, and negative h3/h5
proxy signs must also remain unchanged without new proof.

## 4. Method 2 production path

The authoritative trajectory path is one `ExactSewSolver` instance reused for
all selected frames:

```text
prepared HumanArmTarget
  -> human_arm_to_exact_sew_target()
  -> stateful ExactSewSolver
       -> conservative local event continuation when history is valid
       -> deterministic compiled global event recovery otherwise
  -> deterministic joint-limit representative
  -> authoritative MuJoCo pinch-site + Stereo-SEW acceptance
  -> SolverResult and pipeline evaluation row
```

Implementation ownership:

- `cpp/exact_sew_core.cpp`: compiled event/root core.
- `src/sew_mimic/exact/cpp_solver.py`: state, continuation, global recovery,
  deterministic selection, joint representation, authoritative acceptance.
- `src/sew_mimic/exact/solver.py`: public single-target adapters.
- `src/sew_mimic/pipeline/benchmark.py`: one solver per trajectory and method
  dispatch.
- `src/sew_mimic/exact/numerical_oracle.py`: Method 3 only; never a fallback.

`solve_exact_sew()` remains the stable one-target API. It constructs the same
compiled global solver and supports `canonical` and `continuous` selection via
`q_previous`. For trajectories, prefer a reused `ExactSewSolver`.

Pipeline/CLI deliberately expose no legacy search, tracker, or branch-policy
switches. Local continuation and bounded global recovery are always enabled.
The only runtime Method 2 controls are the exact keys under `exact_sew` in
`config.yaml`, loaded strictly into `ExactSewConfig`:

```text
radii_rad
local_partitions_min
global_partitions
maximum_event_evaluations
maximum_wrapped_joint_step_rad
```

Do not add silent numerical-IK fallback, hidden clipping, or an alternative
production solver path.

## 5. Solver result semantics

Shared statuses are:

```text
SUCCESS_EXACT
SUCCESS_APPROX
UNREACHABLE
JOINT_LIMIT
SEW_SINGULAR
NO_VALID_BRANCH
INVALID_INPUT
NUMERICAL_FAILURE
LEGACY_FAILURE
```

For Methods 2 and 3, authoritative acceptance constants in
`src/sew_mimic/exact/acceptance.py` are strict inequalities:

```text
position error           < 1e-3 m
aligned orientation error < 1 degree
Stereo-SEW error          < 1 degree
```

`SUCCESS_EXACT` means all constraints claimed by that method passed its
post-validation tolerances; it does not mean algebraically zero. Method 2 does
not return a least-squares result as exact and never routes through Method 3.
Every failure must retain an explicit status and `q=None`.

Method 0 can be `SUCCESS_EXACT` while having large pinch-position error because
its declared constraint set is arm directions plus aligned hand orientation,
not absolute pinch position.

## 6. Package map and public APIs

```text
src/sew_mimic/
  common/         HumanArmTarget, ExactSewTarget, statuses/results, evaluation
  sew/            legacy adapter, StereoSew, validated Gen3 SEW geometry
  exact/          compiled Method 2 and lazily loaded Method 3 oracle
  warp/           generic core, compatibility and identification diagnostics
  pipeline/       shared trajectory preparation, dispatch, evaluation, summary
  visualization/  precomputed replay and display-only overlays
```

Major stable imports:

- `sew_mimic.common`: `HumanArmTarget`, `ExactSewTarget`, `SolverStatus`,
  `SolverResult`.
- `sew_mimic.geometry`: `sp3`, `SP3Result` and legacy subproblems.
- `sew_mimic.sew`: `solve_legacy_sew_mimic`, `StereoSew`,
  `StereoSewReference`, `Gen3StereoSewGeometry`.
- `sew_mimic.exact`: `ExactSewConfig`, `ExactSewSolver`,
  `solve_exact_sew`, `retarget_exact_sew`, and lazy
  `NumericalExactSewOracle`.
- `sew_mimic.warp`: generic skeleton construction, adaptive offset,
  compatibility and identification APIs.
- `sew_mimic.pipeline`: `prepare_trajectory`, `run_benchmark`,
  `capability_metadata`, evaluation and summary types.
- `sew_mimic.visualization`: overlay construction, replay preparation,
  consistency validation, and optional viewer.

Do not rewrite `retarget.py`; it is the preserved Method 0 baseline. Keep new
method work in its existing modular package.

## 7. Reproduction and daily commands

From the repository root on Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .[test]
.venv\Scripts\python.exe -m pytest -q
```

The editable install is required to build the private `_exact_sew_core` C++
extension. Useful bounded workflows:

```powershell
# Production Method 2 timing/continuation diagnostics
.venv\Scripts\python.exe scripts\benchmark_exact_sew.py --input data\test.csv --max-frames 100

# Unified comparison; Method 3 is limited to the first 10 selected frames
.venv\Scripts\python.exe scripts\compare_retargeters.py --input data\test.csv --methods sew_mimic exact_sew numerical_oracle --max-frames 100 --oracle-max-frames 10

# Generic WARP and tested Gen3 skeleton evidence
.venv\Scripts\python.exe scripts\validate_warp_core.py
.venv\Scripts\python.exe scripts\identify_gen3_warp_skeleton.py --samples 1000

# Headless consistency replay of precomputed Method 2 rows
.venv\Scripts\python.exe scripts\replay_compare.py --input data\test.csv --results output\comparison_frames.csv --method exact_sew --max-frames 3 --no-viewer
```

`compare_retargeters.py` supports `--start-frame`, `--max-frames`, `--stride`,
and explicit `--all`. Avoid `--all` for expensive validation unless it is
actually required. Method 3 is normally bounded by `--oracle-max-frames`.

Generated comparison/baseline CSV and JSON files under `output/` are
regenerable and ignored by exact filename. Replay uses precomputed q values and
does not run Method 2 IK during playback.

## 8. Recorded validation evidence

The authoritative measured snapshot is in
`docs/FINAL_ENGINEERING_REPORT.md`. Highlights:

- Method 0: complete 4,344-frame baseline succeeded, with expected large
  nonzero pinch-position mismatch and zero aligned-orientation error.
- Method 2: recorded 100-frame run was 100/100 `SUCCESS_EXACT`, with one global
  first frame and 99 local-continuation frames.
- Recorded Method 2 solve time: mean `8.477 ms`, median `8.624 ms`, P95
  `14.208 ms` on the validation machine. Re-benchmark on deployment hardware.
- Method 3: 10/10 exact on the recorded comparison subset, with zero correctness
  discrepancy against Method 2.
- Generic WARP core: 1,000/1,000 exact compatible synthetic cases.
- Tested Gen3 WARP proxy models: no useful fixed skeleton below 1 mm; this is
  not proof that the authors' Kinova path is invalid.

## 9. Known limitations and safe next steps

- `Wrist_XYZ` anatomical meaning is dataset-dependent until calibrated.
- No reproduced executable Gen3 WARP path exists in this repository.
- Method 2 does not include collision avoidance, dynamics, global trajectory
  optimization, ROS, or real-robot safety/control.
- Visualization normally requires precomputed Method 2 results.
- Hosted CI is absent; dependencies other than MuJoCo are not fully locked.

Before changing solver mathematics or conventions:

1. inspect the relevant implementation, tests, and detailed document;
2. state the exact invariant being changed;
3. add a mathematical regression test;
4. run focused tests first, then broader tests only if the impact warrants it;
5. preserve explicit failure status and Method 2/Method 3 isolation.

## 10. Documentation map

- `README.md`: installation and everyday commands.
- `docs/RETARGETING_ARCHITECTURE.md`: method separation and frame conventions.
- `docs/GEN3_STEREO_SEW_GEOMETRY.md`: validated R-2R-2R-2R/PoE mapping.
- `docs/GEN3_EXACT_SEW_SOLVER.md`: current compiled solver behavior.
- `docs/NUMERICAL_EXACT_SEW_ORACLE.md`: Method 3 contract.
- `docs/WARP_CSEW_CORE.md`: generic core, Gen3 hypotheses, and corrected WARP
  conclusion.
- `docs/RETARGETING_COMPARISON.md`: pipeline outputs and statistics.
- `docs/MUJOCO_RETARGETING_VISUALIZATION.md`: precomputed replay conventions.
- `docs/FINAL_ENGINEERING_REPORT.md`: recorded acceptance results and limits.
