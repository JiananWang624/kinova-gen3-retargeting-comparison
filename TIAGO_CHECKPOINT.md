# TIAGo Steel checkpoint (2026-09-27)

The `tiago` branch intentionally retains all Gen3 implementation, tests, and
assets. Do not start second-stage Gen3 cleanup or run all 4,344 retargeting
frames until fixed placement and the production performance route are agreed.

The fixed calibration maps the robust median human shoulder to the MuJoCo
`arm_1_joint` anchor at `[0.09305, 0.014, 0.8875]` m. It is never recomputed
during comparison or replay. The selected SEW shoulder point is `J2`, which is
0.130255 m from this arm-root at the configured home pose; aligning the human
shoulder to J2 would be a different calibration decision and is **not** applied.

`scripts/diagnose_tiago_placement.py` generates an all-4,344-point target
cloud, a deterministic joint-limited MuJoCo TCP workspace sample, a home-pose
arm overlay, and a bounded position-only test on 100 evenly spaced targets.
With the current calibration, the median human shoulder and arm-root agree to
numerical precision, and 100/100 position-only targets fit within 1 mm under
physical joint limits (maximum observed error 0.00234 mm). This does **not**
prove full pose-plus-SEW reachability or rule out an effect of placement on
the 86/100 full-task joint-limit classifications.

The numerical backend remains a correctness reference and potential reliable
fallback, not a qualified production solver: 100-frame mean/P95 were
approximately 5.95/11.05 s. The semi-analytic route now tries exact spherical
wrist branches first, then one bounded local refinement from a projected
out-of-limit branch. The previously missed oracle-reachable frames 688, 1677,
and 2967 pass strict MuJoCo TCP, orientation, and SEW thresholds. The 100-frame
oracle comparison has 14 reachable successes, zero solver misses, and 86
`joint_limit_failure` classifications; a finite multi-start search cannot by
itself prove that every one of those 86 is physically unreachable. The
semi-analytic performance route is still under assessment, especially failure
latency. No backend is yet designated final production.

To refresh only the placement diagnostics (no retargeting solve and no config
write), run:

```powershell
.venv\Scripts\python.exe scripts\diagnose_tiago_placement.py
```

Add `--viewer` to inspect the full TIAGo model and sampled point clouds in
MuJoCo. Headless plot and JSON report are in `output/tiago_placement/`.
