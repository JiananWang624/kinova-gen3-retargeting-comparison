# Retargeting comparison

`scripts/compare_retargeters.py` evaluates the fixed-base Gen3 using one
mounted-base preprocessing pass. Each sampled human frame creates one shared
`HumanArmTarget`, including the configured physical task point.

Method 0 (`sew_mimic`) remains the legacy regression baseline. Method 2
(`exact_sew`) is the recommended fixed-base pinch-pose plus Stereo-SEW method.
Method 3 (`numerical_oracle`) is validation-only and runs only its deterministic
requested subset; it never seeds or selects any production method.

WARP-cSEW is reported only as capability metadata. Its generic core exists, but
this repository has not reproduced the Kinova-specific parameterization shown
in the public Dual-Kinova3 demonstration, so it creates no executable per-frame
comparison rows.

Run the bounded validation comparison from the repository root:

```powershell
.venv\Scripts\python.exe scripts\compare_retargeters.py `
  --input data\test.csv `
  --methods sew_mimic exact_sew `
  --max-frames 100
```

Add `numerical_oracle --oracle-max-frames 10` to the method list to validate a
deterministic leading subset of the same selected frames. Method 3 does not run
on every frame unless the requested subset covers every selected frame.

Use `--all` intentionally for a complete trajectory. Method 2 creates one
stateful compiled solver for the selected frames. Local continuation preserves
the last successful state; uncertainty uses deterministic global recovery and
a failed result never replaces that state. Its controls are the single
`exact_sew` mapping in `config.yaml`.

Output is `output/comparison_frames.csv` and
`output/comparison_summary.json` unless `--output-dir` is supplied. Failed
frames remain in the CSV and denominator statistics with empty joint/error
fields. Error and joint-limit statistics explicitly use successful frames
only. Every successful row independently recomputes the physical pinch-site
pose, joint limits, and robot Stereo-SEW from the MuJoCo-derived FK.

The summary reports the reproduced generic WARP-cSEW core separately from
solver outcomes. Its original capability gate measures the Exact-SEW
`S1/E45/W67` definition, whose upper-arm length varies by about 0.195 m. The
separate Phase W1 identification also tests `S23/E45/W67` and the established
h3/h5 direction proxies: although both candidate link lengths are fixed,
`S23` moves with joint 1 and the best global fixed proxy model has centimetre-
scale independent-validation error. These results reject only the tested
definitions; they do not establish inherent Gen3 incompatibility. See
`docs/WARP_CSEW_CORE.md`. This project has not yet reproduced the demonstrated
Kinova WARP path and creates no WARP trajectory rows.
