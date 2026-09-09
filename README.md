# Kinova Gen3 Retargeting Comparison

This repository compares three independently tested retargeting methods for the
fixed-base Kinova Gen3 7-DoF arm, retaining SEW-Mimic as the regression baseline
and including a numerical Exact-SEW validation oracle. Human
shoulder/elbow/wrist geometry and hand orientation are converted to robot joint
configurations and evaluated with the real MuJoCo-derived `pinch_site` forward
kinematics.

The recommended Gen3 method is **Method 2: Exact-SEW**. The original SEW-Mimic
implementation remains unchanged as the regression baseline.

## Methods and capabilities

| Method | Package / API | Gen3 capability | Role |
|---|---|---|---|
| 0: SEW-Mimic | `sew_mimic.sew.solve_legacy_sew_mimic` | Executable | Baseline |
| 1: WARP-cSEW | `sew_mimic.warp` | Kinova path not yet reproduced in this repository | Generic fixed-link core reproduction |
| 2: Exact-SEW | `sew_mimic.exact.solve_exact_sew` | Executable | Recommended fixed-base solver |
| 3: numerical exact-pose + SEW | `sew_mimic.exact.NumericalExactSewOracle` | Executable | Validation only |

Method 1 reproduces generic c-SEW corrected-skeleton geometry. WARP materials
explicitly demonstrate Dual-Kinova3, so this project does not claim that Gen3
is inherently incompatible with WARP. Phase W1 found only that the tested
`S1/E45/W67`, `S23/E45/W67`, and h3/h5-based fixed models were insufficient:
the best global proxy model retained about 15.35 mm mean / 25.90 mm maximum
independent-validation position error. The authors' exact Kinova-specific
parameterization is not available in enough public detail to reconstruct here;
therefore this repository has not yet reproduced a Gen3 WARP trajectory path.

## Architecture

```text
src/sew_mimic/
  common/         shared targets, statuses, task point, and FK evaluation
  sew/            legacy Method 0 adapter and Stereo-SEW representation
  exact/          compiled Method 2 solver and lazy Method 3 oracle
  warp/           generic WARP core and tested Gen3 geometry diagnostics
  pipeline/       mounted trajectory preparation, dispatch, and benchmark
  visualization/  precomputed replay and display-only overlays
```

Method 2 uses one stateful compiled event solver per trajectory. It applies
strict MuJoCo-derived pinch-site acceptance and never falls back to Method 3.

## Installation on Windows

The validated environment is Python 3.11 with MuJoCo 3.1.6.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .[test]
```

Run commands from the repository root. The scripts add `src/` to their import
path; no package installation step is required.

## Validate

Run the complete automated suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Reproduce the Method 1 Gen3 virtual-skeleton identification:

```powershell
.venv\Scripts\python.exe scripts\identify_gen3_warp_skeleton.py --samples 1000
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
  --max-frames 100 `
  --oracle-max-frames 10
```

This writes regenerable `output/comparison_frames.csv` and
`output/comparison_summary.json`. The Phase 3 stateful C++ path recorded about
8.48 ms per frame on the 100-frame benchmark; rerun the benchmark on the target
machine before treating timing as a release measurement.

Exact-SEW continuation and bounded global recovery are always enabled for a
trajectory. Its sole runtime controls are the `exact_sew` mapping in
`config.yaml`.

### 选择方法和帧范围

`--methods` 可以指定一个或多个方法；`--start-frame` 从 0 开始计数，
`--max-frames` 指定最多处理多少帧，`--stride` 指定帧间隔。

只运行方法 2 的第 100 帧：

```powershell
.venv\Scripts\python.exe scripts\compare_retargeters.py `
  --input data\test.csv `
  --methods exact_sew `
  --start-frame 100 `
  --max-frames 1
```

运行方法 0 的第 100–199 帧（包含两端）：

```powershell
.venv\Scripts\python.exe scripts\compare_retargeters.py `
  --input data\test.csv `
  --methods sew_mimic `
  --start-frame 100 `
  --max-frames 100
```

运行方法 2 的全部 CSV 帧：

```powershell
.venv\Scripts\python.exe scripts\compare_retargeters.py `
  --input data\test.csv `
  --methods exact_sew `
  --all
```

同时运行方法 0 和方法 2 的全部 CSV 帧：

```powershell
.venv\Scripts\python.exe scripts\compare_retargeters.py `
  --input data\test.csv `
  --methods sew_mimic exact_sew `
  --all
```

方法 3 是验证用数值 oracle，默认只运行前 10 帧。若确实需要让它覆盖
全部 CSV 帧，必须把 `--oracle-max-frames` 设置为数据帧数；这通常会非常慢：

```powershell
.venv\Scripts\python.exe scripts\compare_retargeters.py `
  --input data\test.csv `
  --methods numerical_oracle `
  --all `
  --oracle-max-frames 4344
```

上面的 `4344` 需要替换成实际 CSV 的总帧数。比较结果统一写入
`output/comparison_frames.csv` 和 `output/comparison_summary.json`。

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

Replay 已经计算完成的全部帧：

```powershell
.venv\Scripts\python.exe scripts\replay_compare.py `
  --input data\test.csv `
  --results output\comparison_frames.csv `
  --method exact_sew `
  --all
```

如果只想无窗口验证全部结果，在命令末尾加 `--no-viewer`。回放方法 0 时将
`--method exact_sew` 改为 `--method sew_mimic`；回放方法 3 时改为
`--method numerical_oracle`。

## Validated findings

- Method 0 preserves exact aligned hand orientation but has a large nonzero
  physical pinch-position mismatch because position is not its solve
  constraint.
- Method 2 matches the fixed-base pinch position, aligned pinch orientation,
  and human Stereo-SEW angle below the exact thresholds on the validated
  trajectory subset.
- Method 3 agrees with Method 2 on the validation subset and remains an oracle,
  never a production fallback.
- Generic WARP fixed-link synthetic cases are exact. The tested Gen3 virtual
  definitions do not supply a fixed skeleton consistent with the established
  h3/h5 proxies below the 1 mm practical threshold; this does not contradict
  the Dual-Kinova3 WARP demonstration or rule out its unpublished details.

The configured human task point defaults to `Wrist_X/Y/Z`. Its anatomical
meaning remains dataset-dependent unless a wrist-to-task offset is calibrated.
Visualization offsets never affect targets, solver inputs, or metrics.

## Detailed documentation

- [Retargeting architecture](docs/RETARGETING_ARCHITECTURE.md)
- [Stereo-SEW convention](docs/STEREO_SEW.md)
- [Validated Gen3 geometry](docs/GEN3_STEREO_SEW_GEOMETRY.md)
- [Exact-SEW solver and branch policies](docs/GEN3_EXACT_SEW_SOLVER.md)
- [Numerical oracle](docs/NUMERICAL_EXACT_SEW_ORACLE.md)
- [Generic WARP core and compatibility](docs/WARP_CSEW_CORE.md)
- [Unified comparison](docs/RETARGETING_COMPARISON.md)
- [MuJoCo replay](docs/MUJOCO_RETARGETING_VISUALIZATION.md)
- [Engineering handoff](HANDOFF.md)
