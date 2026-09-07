# MuJoCo retargeting visualization

`scripts/replay_compare.py` replays stored Phase 7 configurations. It reads the
original human CSV through `prepare_trajectory()` and requires a matching
method/frame row in `comparison_frames.csv`; it never runs IK during normal
playback. This is important for Method 2 because optimized event-aware
Exact-SEW solving still took about 2.4 seconds per frame in the latest ten-frame
Windows validation.

Before opening the viewer, replay runs the Phase 7 authoritative evaluator on
each stored successful q and rejects stale results whose position,
orientation, SEW, or joint-margin metrics disagree with the selected input.
This verification is FK-only and does not invoke candidate generation or IK.

Supported Gen3 playback labels are `sew_mimic`, `exact_sew` (the default), and
`numerical_oracle` (validation only). WARP-cSEW has a reproduced generic core,
but the present Gen3 fails its fixed-link compatibility gate, so
`warp_csew` is rejected rather than treated as a failed IK trajectory.

## Overlay convention

All headless primitives are constructed in the same fixed Gen3 base frame used
by Phase 7. Only the final MuJoCo renderer converts them to the mounted world
frame. The overlays show:

- human shoulder, elbow, wrist, task point, arm segments, and hand-frame axes;
- the unmodified target task point and target hand-frame axes;
- the physical `pinch_site` and its established aligned hand-frame axes;
- the actual-pinch-to-target-task position-error vector;
- optionally, validated human and robot S/E/W triangles and oriented SEW-plane
  normals;
- optional finite human-task and robot-pinch trails.

Axes use red X, green Y, and blue Z. The position-error segment endpoints are
the same points used by the Phase 7 authoritative evaluator.

`--human-display-offset X Y Z` translates only copied human display geometry,
including its displayed SEW geometry and trail. It never changes the
`HumanArmTarget`, true target marker/frame, stored q, FK, solver input, or
metrics.

If a stored row has no q, the status and human/target overlays remain visible.
After at least one success, the viewer holds the last valid robot pose for
display only and suppresses the current-frame error vector. If the first frame
fails, no robot pose is presented as valid. Held poses never feed a solver,
metric, or trajectory state.

## Commands

Method 2 with SEW geometry and 30-point trails:

```cmd
.venv\Scripts\python.exe scripts\replay_compare.py ^
  --input data\test.csv ^
  --results output\comparison_frames.csv ^
  --method exact_sew ^
  --max-frames 100 ^
  --show-sew ^
  --trail-length 30
```

Method 0 uses the same precomputed workflow:

```cmd
.venv\Scripts\python.exe scripts\replay_compare.py ^
  --input data\test.csv ^
  --results output\comparison_frames.csv ^
  --method sew_mimic ^
  --max-frames 20
```

Use `--fps`, `--start-frame`, `--stride`, and `--loop` for display timing and
selection. FPS never changes input or results. `--no-viewer` validates files,
constructs every overlay headlessly, and prints stored per-frame diagnostics.
The `--no-show-human`, `--no-show-target`, `--no-show-sew`, and
`--no-show-error` forms disable individual layers.
