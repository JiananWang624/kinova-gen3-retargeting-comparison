# TIAGo Steel checkpoint (2026-09-27)

The `tiago` branch intentionally retains all Gen3 implementation, tests, and
assets. Do not start second-stage Gen3 cleanup or run all 4,344 retargeting
frames until fixed placement and the production performance route are agreed.

The original fixed calibration maps the robust median human shoulder to the
MuJoCo `arm_1_joint` anchor at `[0.09305, 0.014, 0.8875]` m. Its full
translation and shoulder reference remain in `config.yaml` as
`baseline_j1_arm_root`, and its diagnostic plot/report remain in
`output/tiago_placement/`. Both candidate transforms are fixed across frames;
neither moves the base or recalibrates during comparison or replay.

The active calibration now maps that human reference to the selected SEW
shoulder `J2` **at the configured home pose**. The home-pose J2 anchor is
`[0.21945241193369783, 0.008738796968015413, 0.8565]` m, 0.130255 m
from J1. Its fixed `t_base_from_body` is
`[0.6563659554336978, -0.045917781031984586, 0.4609458315]` m. The J2
anchor itself moves with J1; only its *home-pose reference* defines this
one-time fixed calibration.

`scripts/diagnose_tiago_placement.py` generates an all-4,344-point target
cloud, a deterministic joint-limited MuJoCo TCP workspace sample, a home-pose
arm overlay, and a bounded position-only test on 100 evenly spaced targets.
Both J1 baseline and active J2 alignment fit 100/100 position-only targets
within 1 mm under physical joint limits. Position-only reachability does not
prove full pose-plus-SEW reachability.

`scripts/compare_tiago_calibrations.py` compared both fixed placements on
the **same** 50 evenly spaced frames plus the known boundary frames 688,
1677, and 2967. The independent MuJoCo/SciPy full pose-plus-SEW oracle found
8 strict solutions with J1 and 17 with J2; the semi-analytic solver missed
none of those reachable frames in either case. J2 improved the oracle joint
limit margin median from 1.26° to 2.83° and overall semi-analytic P95 solve
time from 943 ms to 323 ms. Its robot SEW singularity margin was lower
(minimum 0.436 versus 0.685), but remained well clear of zero. On six
five-frame local continuity windows, solved-joint P95 maximum wrapped step
was 0.69°/frame for J2 versus 1.07°/frame for J1; the sets of solvable
windows differ, so this continuity comparison is descriptive rather than a
paired proof. Full per-frame and summary data are in
`output/tiago_placement/calibration_comparison.json`. This evidence met the
requested condition for switching the active *fixed* calibration to J2.

The numerical backend remains a correctness reference and potential reliable
fallback, not a qualified production solver: its earlier 100-frame mean/P95
were approximately 5.95/11.05 s. The active evaluation path is now
`tiago_semi_analytic`: exact spherical-wrist branches first, then one bounded
local refinement from a projected out-of-limit branch. It passed 1,000/1,000
self-consistency targets and all oracle-reachable frames in the J1 100-frame
and J1/J2 53-frame gates. It is **not yet fully production-qualified**;
the full 4,344-frame run and Gen3 cleanup remain paused. An oracle report of
`joint_limit_failure` means relaxed-limit strict success plus failure of a
finite physical-limit search; it is not a proof of physical impossibility.

To refresh only the placement diagnostics (no retargeting solve and no config
write), run:

```powershell
.venv\Scripts\python.exe scripts\diagnose_tiago_placement.py
```

Add `--viewer` to inspect the full TIAGo model and sampled point clouds in
MuJoCo. Active J2 headless plot and JSON report are in
`output/tiago_placement/active_j2/`.
