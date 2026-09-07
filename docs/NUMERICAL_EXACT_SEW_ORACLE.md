# Numerical Exact-SEW oracle

Method 3 is a deterministic numerical validation oracle, not the production
Exact-SEW IK solver. It accepts an `ExactSewTarget(position, rotation, psi)`
in the native fixed Gen3 base frame. Position and orientation are evaluated
from MuJoCo-derived `pinch_site` FK with the established `R_robot_align`;
robot psi uses the Phase-3 virtual `S,E,W` geometry and Phase-2 Stereo-SEW.

It uses bounded SciPy `least_squares` starts (zero, validated nominal/home,
limit midpoint when distinct, and six fixed-seed samples), with separate
pose-only and pose-plus-SEW APIs. The fixed starts are target-independent;
an optional caller seed is additive and is labelled in candidate diagnostics.
Residuals are scaled by 0.1 m, 0.2 rad, and 0.2 rad for optimization only.
Physical post-validation defaults to the shared Method 2/3 limits: position
< 1e-3 m, orientation < 1 degree, and SEW < 1 degree. Callers may override the
oracle configuration explicitly. A finite nonexact candidate is
`SUCCESS_APPROX`, never exact. Numerical failure is not a reachability proof, so the oracle does not
infer `UNREACHABLE` from optimizer failure.

Candidates retain seed ID, optimizer termination/status/message/nfev/cost,
post-validated physical residuals and q. Equivalent canonical q values are
deduplicated. They are ordered deterministically: exact first, normalized
physical residual (by the exact acceptance thresholds, not optimizer scales),
larger finite joint margin, wrapped distance to zero reference, and
lexicographic q. `SEW_SINGULAR` is returned when all full-task final
evaluations are stereographically singular; invalid tasks or seeds are
`INVALID_INPUT`; unusable numerical runs are `NUMERICAL_FAILURE`.
Unexpected residual exceptions are recorded per start and are not converted
into ordinary approximate candidates. Phase 3 established a positive robot
workspace margin from the configured stereographic half-line; Phase 4 tests
the oracle's explicit singular-status propagation without fabricating an
unreachable Gen3 configuration.

The validation script runs 100 unbiased synthetic targets in each mode by
default, a clearly labelled local perturbed-seed control, and a near-limit
case. For CSV data it mounts once from the first shoulder, converts every
selected human frame into that unchanged base, and never moves the root per
frame. It reports the two modes separately and does not interpret a numerical
failure as a reachability certificate. Multi-start coverage is intentionally
finite. This oracle validates future analytical IK; it does not replace that
production solver.

On the initial deterministic validation run, all 100 synthetic pose-only and
all 100 synthetic pose-plus-SEW targets were `SUCCESS_EXACT`. Their maximum
exact errors were respectively `2.542e-14 m / 8.292e-14 rad` and
`1.579e-15 m / 1.909e-15 rad / 2.220e-15 rad`. The local perturbed-seed and
near-limit controls were both exact; the latter retained a `1.0e-5 rad`
margin. On the first 100 fixed-base CSV frames, both modes were exact for all
frames, with zero pose-exact/SEW-nonexact cases. Full-task CSV median/P95
errors were `1.532e-16 / 2.852e-16 m`, `3.323e-16 / 6.009e-16 rad`, and
`4.441e-16 / 8.882e-16 rad`; median/P95 solve time was `1773 / 1940 ms`.
