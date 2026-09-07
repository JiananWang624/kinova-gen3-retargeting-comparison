# SEW-Mimic and Exact-SEW for Kinova Gen3

This repository reproduces SEW-Mimic and extends it with independently tested
retargeting methods for the fixed-base Kinova Gen3 7-DoF arm. Human
shoulder/elbow/wrist geometry and hand orientation are converted to robot joint
configurations and evaluated with the real MuJoCo-derived `pinch_site` forward
kinematics.

The recommended Gen3 method is **Method 2: Exact-SEW**. The original SEW-Mimic
implementation remains unchanged as the regression baseline.

## Methods and capabilities

| Method | Package / API | Gen3 capability | Role |
|---|---|---|---|
| 0: SEW-Mimic | `sew_mimic.sew.solve_legacy_sew_mimic` | Executable | Baseline |
| 1: WARP-cSEW | `sew_mimic.warp` | Not executable on the current Gen3 | Generic fixed-link core reproduction |
| 2: Exact-SEW | `sew_mimic.exact.solve_exact_sew` | Executable | Recommended fixed-base solver |
| 3: numerical exact-pose + SEW | `sew_mimic.exact.NumericalExactSewOracle` | Executable | Validation only |

Method 1 reproduces generic c-SEW corrected-skeleton geometry, but the
validated Gen3 upper-arm proxy length is configuration-dependent. The robot
therefore fails WARP's fixed-link compatibility requirement. This repository
does **not** claim a Gen3 WARP reproduction and exposes no Gen3 WARP trajectory
solver.

## Architecture

```text
src/sew_mimic/
  common/         shared targets, statuses, task point, and FK evaluation
  sew/            legacy Method 0 adapter and Stereo-SEW representation
  exact/          Method 2 candidates/selection and lazy Method 3 oracle
  warp/           generic WARP core and Gen3 compatibility gate
  pipeline/       mounted trajectory preparation, dispatch, and benchmark
  visualization/  precomputed replay and display-only overlays
```

Candidate generation and trajectory branch selection are separate. Method 2
uses deterministic event-aware candidate discovery and supports canonical or
nearest-previous-configuration selection. It never falls back to Method 3.

## Installation on Windows

The validated environment is Python 3.11 with MuJoCo 3.1.6.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run commands from the repository root. The scripts add `src/` to their import
path; no package installation step is required.

## Validate

Run the complete automated suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Measure the preserved Method 0 baseline over the input CSV:

```powershell
.venv\Scripts\python.exe scripts\measure_baseline.py --input data\test.csv
```

Compare Method 0, recommended Method 2, and the bounded Method 3 oracle:

```powershell
.venv\Scripts\python.exe scripts\compare_retargeters.py `
  --input data\test.csv `
  --methods sew_mimic exact_sew numerical_oracle `
  --exact-branch-policy continuous `
  --max-frames 100 `
  --oracle-max-frames 10
```

This writes regenerable `output/comparison_frames.csv` and
`output/comparison_summary.json`. The optimized Method 2 backend averaged about
2.4 seconds per frame in the latest ten-frame Windows validation, so bounded
validation remains the normal workflow.

Replay precomputed Method 2 results interactively:

```powershell
.venv\Scripts\python.exe scripts\replay_compare.py `
  --input data\test.csv `
  --results output\comparison_frames.csv `
  --method exact_sew `
  --max-frames 100
```

Add `--no-viewer` for headless replay-file and evaluator consistency checks.
Use `--method sew_mimic` to inspect the baseline. Replay never runs Method 2
IK; it uses precomputed configurations and independently verifies their stored
metrics.

## Validated findings

- Method 0 preserves exact aligned hand orientation but has a large nonzero
  physical pinch-position mismatch because position is not its solve
  constraint.
- Method 2 matches the fixed-base pinch position, aligned pinch orientation,
  and human Stereo-SEW angle below the exact thresholds on the validated
  trajectory subset.
- Method 3 agrees with Method 2 on the validation subset and remains an oracle,
  never a production fallback.
- Generic WARP fixed-link synthetic cases are exact, while the current Gen3 is
  deterministically reported incompatible.

The configured human task point defaults to `Wrist_X/Y/Z`. Its anatomical
meaning remains dataset-dependent unless a wrist-to-task offset is calibrated.
Visualization offsets never affect targets, solver inputs, or metrics.

## Detailed documentation

- [Retargeting architecture](docs/RETARGETING_ARCHITECTURE.md)
- [Stereo-SEW convention](docs/STEREO_SEW.md)
- [Validated Gen3 geometry](docs/GEN3_STEREO_SEW_GEOMETRY.md)
- [Exact-SEW backend](docs/GEN3_EXACT_SEW_BACKEND.md)
- [Exact-SEW solver and branch policies](docs/GEN3_EXACT_SEW_SOLVER.md)
- [Numerical oracle](docs/NUMERICAL_EXACT_SEW_ORACLE.md)
- [Generic WARP core and compatibility](docs/WARP_CSEW_CORE.md)
- [Unified comparison](docs/RETARGETING_COMPARISON.md)
- [MuJoCo replay](docs/MUJOCO_RETARGETING_VISUALIZATION.md)
- [Engineering handoff](HANDOFF.md)
