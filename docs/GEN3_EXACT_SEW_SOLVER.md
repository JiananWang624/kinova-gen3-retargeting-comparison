# Gen3 Exact-SEW Solver (Method 2)

Method 2 is the fixed-base Kinova Gen3 production retargeter. It solves the
pinch-site pose plus Stereo-SEW task exactly; it does not move the robot root,
alter the human target, refine candidates numerically, or fall back to the
Method-3 numerical oracle.

## Target contract

`human_arm_to_exact_sew_target()` is the single conversion from an existing
`HumanArmTarget` to `ExactSewTarget`:

```text
position = human.task_point
rotation = human.hand_rotation
psi = StereoSew(final_reference).forward(
    human.shoulder, human.elbow, human.wrist
)
```

All values must already be expressed in the established Gen3 base frame. No
additional alignment, mounting, or task-point offset is applied here. A
Stereo-SEW singularity produces `SEW_SINGULAR`.

## Exact-only solver semantics

`solve_exact_sew()` calls the Phase-5A candidate enumerator once and then
selects one branch. A selectable candidate must be marked exact, possess a
valid joint-limit representative, and satisfy the authoritative MuJoCo
pinch-site thresholds:

- position error `< 1e-3 m` (1 mm)
- orientation error `< 1 degree`
- Stereo-SEW error `< 1 degree`

Method 2 only returns `SUCCESS_EXACT`; it never returns `SUCCESS_APPROX`.
Approximate subproblem results, clipping, numerical refinement, and numerical
IK fallback are not part of this method.

The production search mode is `event_aware`. The official 200-sample policy is
retained as `reference_fixed_grid` compatibility mode.

## Branch policies

Canonical selection is deterministic and history-independent. It ranks
selectable candidates by:

1. largest minimum joint-limit margin;
2. smallest normalized authoritative residual;
3. the existing Phase-5A candidate order.

The normalized residual is the sum of squared ratios to the three exact
acceptance thresholds.

Continuous selection minimizes the sum of squared wrapped joint differences
to `q_previous`. Ties use the canonical margin, residual, and Phase-5A-order
rules. When `q_previous` is absent, continuous selection is exactly canonical.
This is greedy per-frame selection, not global trajectory optimization.
History affects selection only and is never passed into candidate generation.

Public branch IDs use the fixed Phase-5A lexical branch slot
`4*i + 2*j + k` plus the wrist-root ordinal within that same slot in the
unfiltered Phase-5A candidate order. The continuously varying wrist-search
angle is recorded separately in diagnostics metadata.

## Status mapping

- `INVALID_INPUT`: malformed target, policy, or previous configuration.
- `SEW_SINGULAR`: the configured Stereo-SEW representation is singular.
- `JOINT_LIMIT`: exact geometric candidates exist, but none is joint-valid.
- `NO_VALID_BRANCH`: enumeration completes with no exact geometric candidate.
- `NUMERICAL_FAILURE`: the backend fails abnormally or returns an internally
  inconsistent exact/joint-valid set that fails authoritative tolerances.

Every failure has `q=None`. Method 2 does not infer `UNREACHABLE` from an empty
candidate set.

## Mounted CSV validation

`scripts/validate_exact_sew_solver.py` evaluated the first 100 consecutive
frames of `data/test.csv`. Each frame was mounted and converted through the
existing CSV/task-point pipeline, enumerated once, and then supplied to both
selection policies.

| Measurement | Canonical | Continuous |
|---|---:|---:|
| `SUCCESS_EXACT` | 100/100 | 100/100 |
| Branch switches | 2 | 0 |
| Wrapped jump median (rad) | 0.016164 | 0.016140 |
| Wrapped jump P95 (rad) | 0.021288 | 0.020390 |
| Wrapped jump maximum (rad) | 5.346846 | 0.024542 |
| Solve time mean (ms) | 3779.31 | 3779.33 |
| Solve time median (ms) | 3887.56 | 3887.57 |
| Solve time P95 (ms) | 4254.00 | 4254.01 |

All continuous choices independently satisfied the documented local
minimum-distance rule. Both policies had maximum errors of `5.623e-11 m`,
`3.360e-10 rad` orientation, and `1.221e-13 rad` SEW. The minimum selected
joint-limit margin was `0.133333 rad`. Candidate count was exactly eight on
every frame. A deterministic 10-frame oracle subset produced 10 oracle-exact /
Method-2-exact results and no production misses.

The table above records the original pre-optimization validation. The backend
now hoists target invariants, caches exact repeated angle evaluations within a
frame, and evaluates only the requested alignment slot. A ten-frame rerun in
the validated Windows environment averaged about 2.4 seconds per frame;
runtime remains environment-dependent and a full trajectory is still expensive.
