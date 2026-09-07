# Final engineering report

## Repository state

Phase 9 was validated from commit
`63d7172cde67044abe6236e69c125014b924d807` on Windows with Python 3.11.15 and
MuJoCo 3.1.6. The working tree already contained the completed Phase 5-8 files;
Phase 9 did not commit, push, merge, or tag them.

Important module tree:

```text
src/sew_mimic/
  common/         evaluation.py, status.py, task_point.py, types.py
  sew/            gen3_geometry.py, legacy_adapter.py, stereo.py
  exact/          stereo_backend.py, root_search.py, branch_selection.py,
                  solver.py, numerical_oracle.py
  warp/           geometry.py, skeleton.py, compatibility.py
  pipeline/       trajectory.py, evaluator.py, benchmark.py
  visualization/  overlay.py, replay.py
```

Generated `baseline_metrics.csv`, `comparison_frames.csv`, and
`comparison_summary.json` are ignored by their exact names and remain available
under `output/`. Existing tracked diagnostic artifacts were not removed.

## Automated tests

- Starting suite: `289 passed in 52.38s`.
- Suite after API/documentation hardening: `291 passed in 52.09s` in the
  recorded report run.
- Added release coverage imports all documented major APIs and verifies that
  importing Method 2 does not load Method 3.
- Existing tests retain the pinned SP3 reference, Stereo-SEW round trip, Gen3
  PoE/MuJoCo agreement, selected reference pair, narrow event-aware root,
  deterministic branches, Gen3 WARP incompatibility, authoritative evaluator,
  and replay consistency regressions.

## Capability matrix

| Method | Machine name | Executable on current Gen3 | Final role |
|---|---|---:|---|
| 0 | `sew_mimic` | Yes | `baseline` |
| 1 | `warp_csew` | No | Generic core reproduced; `fixed_link_geometry_incompatible` |
| 2 | `exact_sew` | Yes | `recommended` |
| 3 | `numerical_oracle` | Yes | `validation_only` |

The pipeline emits this contract through `capability_metadata()`. It creates no
Method 1 Gen3 trajectory rows. Method 2 has no numerical-oracle import or
fallback; the benchmark dispatcher loads Method 3 only for explicitly requested
oracle validation.

## Numerical validation

### Method 0

The complete 4,344-frame baseline succeeded with no joint-limit violations.
Mean/median/P95/max pinch-position error was
`130.717 / 114.071 / 251.710 / 334.899 mm`; aligned orientation error was zero.
This large position mismatch is expected because Method 0 does not constrain
physical pinch position.

On the shared 100-frame comparison subset, Method 0 was 100/100 successful,
with mean `90.294 mm` and maximum `110.638 mm` pinch-position error and zero
aligned orientation error.

### Method 2

The shared 100-frame continuous-policy comparison was 100/100
`SUCCESS_EXACT`. Maximum errors were:

- pinch position: `5.623e-11 m`
- aligned orientation: `0 rad` as reported by the authoritative metric
- Stereo-SEW: `1.221e-13 rad` (`6.997e-12 deg`)

There were no joint-limit violations or branch switches. Wrapped joint jumps
had median `0.016140 rad`, P95 `0.020390 rad`, and maximum `0.024542 rad`.
Mean solve time was `3.393 s/frame` in this run.

The deterministic three-target backend check was exact on all targets. Its
pinned narrow case was missed by fixed grids of 200, 400, and 800 samples but
found exactly by event-aware search, preserving the documented production
departure.

### Method 3

The first ten shared comparison frames were 10/10 `SUCCESS_EXACT`. All ten
overlapped Method 2 exact successes; correctness discrepancy count was zero.
The oracle was used only for validation and not for candidate generation,
selection, fallback, or replay.

### WARP

The generic fixed-link corrected-skeleton validation was exact on 1,000/1,000
synthetic compatible cases. Maximum palm error was zero, maximum upper/forearm
length errors were `1.665e-16 / 2.776e-16 m`, and maximum Stereo-SEW error was
`8.882e-16 rad`.

The current Gen3 is incompatible: sampled upper-arm proxy length varied by
`0.19513668 m`, exceeding the `1e-10 m` fixed-geometry tolerance. No
approximate Gen3 WARP path exists.

## Validated commands

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\measure_baseline.py --input data\test.csv
.venv\Scripts\python.exe scripts\validate_gen3_sew_geometry.py
.venv\Scripts\python.exe scripts\validate_warp_core.py
.venv\Scripts\python.exe scripts\validate_exact_sew_backend.py --count 3
.venv\Scripts\python.exe scripts\validate_exact_sew_solver.py --input data\test.csv --count 3 --oracle-count 3
.venv\Scripts\python.exe scripts\compare_retargeters.py --input data\test.csv --methods sew_mimic exact_sew numerical_oracle --exact-branch-policy continuous --max-frames 100 --oracle-max-frames 10
.venv\Scripts\python.exe scripts\replay_compare.py --input data\test.csv --results output\comparison_frames.csv --method sew_mimic --max-frames 3 --no-viewer
.venv\Scripts\python.exe scripts\replay_compare.py --input data\test.csv --results output\comparison_frames.csv --method exact_sew --max-frames 3 --no-viewer
```

Both replay commands loaded only precomputed configurations and passed the
authoritative evaluator consistency gate. Interactive viewer launch remains a
manual smoke test.

## Public API summary

- `sew_mimic.common`: shared targets, explicit solver status/result contracts,
  task point, and authoritative evaluation.
- `sew_mimic.geometry`: SP3 and legacy geometric subproblems.
- `sew_mimic.sew`: legacy Method 0 adapter, `StereoSew`, reference type, and
  validated Gen3 geometry.
- `sew_mimic.exact`: candidate enumeration, branch selection, production solve
  and retarget APIs, plus lazily loaded numerical oracle.
- `sew_mimic.warp`: generic geometry/skeleton core and compatibility gate.
- `sew_mimic.pipeline`: trajectory preparation, benchmark dispatch, summaries,
  evaluation, and capability metadata.
- `sew_mimic.visualization`: headless overlay construction, precomputed replay,
  consistency validation, and optional MuJoCo viewer.

## Known limitations and release conclusion

- Optimized event-aware Method 2 remains approximately 2.4 seconds per frame in
  the latest short Windows validation and is not real-time.
- `Wrist_XYZ` anatomical meaning is dataset-dependent unless calibrated.
- Generic fixed-link WARP cannot be applied exactly to the current Gen3 model.
- Visualization normally requires precomputed Method 2 results.
- CI was not added because the non-MuJoCo dependencies are not locked and this
  exact suite has not been reproduced on a hosted Linux or Windows runner.

The final diff contains release/API/docs hardening only for Phase 9; no solver
mathematics, coordinate convention, alignment, mounting, or tolerance changed.
No correctness blocker remains.
