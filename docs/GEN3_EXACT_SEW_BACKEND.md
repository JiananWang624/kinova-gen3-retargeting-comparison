# Gen3 Exact-SEW candidate backend

This Phase-5A module enumerates all strict geometric candidates for the
fixed-base Gen3 R-2R-2R-2R family. It does not select a final branch; Phase
5B owns canonical and trajectory selection. It ports pinned
[Stereo-SEW R-2R-2R-2R IK](https://github.com/rpiRobotics/stereo-sew/blob/d6917478037b924e1292e65a8f52398da3948851/%2BSEW_IK/IK_R_2R_2R_2R.m)
and [IK-Geo one-dimensional search](https://github.com/rpiRobotics/ik-geo/blob/a3a1675e1f01ad6f8f15f2cc787fa01472082a11/matlab/robot_IK_helpers/search_1D.m).

`ExactSewTarget` uses aligned real-pinch orientation. The native target is
`R_07 = R_target R_robot_align^T R_7T^T`; position and wrapped psi are
unchanged. The target gate is FK-tested against 500 authoritative MuJoCo and
Phase-3 PoE configurations.

The official nested lattice has fixed slot `4*i + 2*j + k`, for SP3 q1 branch
`i`, SP2(q2,q3) branch `j`, and SP2(q4,q5) branch `k`. The wrist construction
uses `StereoSew.inverse(S,W,psi)`, then the official rotated wrist-to-elbow
vector. q6/q7 use the official SP1 relations. SP3 must report exact; SP2 and
SP1 are independently checked against their defining full-vector equations.

Feasibility uses normalized signed margins: SP3 is
`1 - (C/(2 r1 r2))^2`; SP4 is `1 - (b/||a||)^2`; SP2 is the minimum of its two
SP4 reductions. The hierarchical deterministic event search first localizes
SP3 feasible intervals, then q23 children, then q45 children, and evaluates
alignment only in q45 leaves. Defaults are 64 initial partitions, depth 24,
minimum width `1e-12 rad`, a 50,000-callback cap for each bounded event
localization, feasibility boundary xtol `1e-12`, alignment xtol `1e-15`, and
alignment construction residual tolerance `1e-8`. These search values remain
unchanged because looser experimental values dropped validated tangent and
near-limit branches. Candidate diagnostics aggregate actual evaluations and
budget exhaustion across the hierarchy.

LS values may be used only as continuous trace witnesses during event
localization. They never become candidates: each root recomputes strict
subproblems, then checks real MuJoCo-derived pinch position/orientation and
Stereo-SEW psi against the project limits of `1e-3 m`, `1 degree`, and
`1 degree`. Unlimited joints
wrap to `[-pi,pi)`; limited joints choose a deterministic equivalent `q+2pi*k`
only when integer bounds derived from the actual interval admit it. Nothing is
clipped. Candidates order by wrist angle, slot, then q. Target-invariant
geometry is computed once per enumeration, exact repeated angle evaluations
are cached within that frame, and alignment leaves compute only their requested
lexical slot. These changes reduce duplicate work without changing coverage or
subproblem equations.

## Departure from the reference fixed-grid root discovery

The official R-2R-2R-2R equations are retained. Gen3's nearly parallel
`P[:,5]`/`H[:,4]` geometry creates exact q4/q5 feasibility intervals only a
few microradians wide. For the pinned target generated from
`q=[0.2,0.3,-0.4,0.5,-0.2,0.3,0.4]`, uniform 200-, 400-, and 800-sample
searches all miss a real exact solution. Production therefore uses
deterministic, hierarchical feasibility-boundary discovery; the official
200-sample behavior remains available as `reference_fixed_grid` mode.

This implementation is **R-2R-2R-2R Stereo-SEW IK with a Gen3-specific
robust exact-feasibility search strategy**, not a byte-for-byte reproduction
of reference root discovery. Other explicit differences are tighter SciPy
bracket tolerances, sampled-zero/tangent handling, fixed inactive/tangent
slots, and strict subproblem plus final post-validation. Approximate
subproblem solutions are never accepted as exact candidates. Method 3's
least-squares oracle is validation-only and is never imported by production
candidate generation.

The deterministic 100-target validation (`seed=20260911`) reported 99
oracle-exact/production-exact, 0 oracle-exact/production-miss, 1
production-exact/oracle-nonexact, and 100/100 production exact and joint-valid
coverage. Exact candidate counts were mean 8.12, median 8, P95 8, maximum 16;
joint-valid counts were mean 8.04, median 8, P95 8, maximum 16. Event callback
counts were mean 9,776.53, median 9,291.5, and P95 12,440.6. Production solve
times were mean 4,061.17 ms, median 3,796.64 ms, and P95 5,754.49 ms. Maximum
accepted errors were `1.5122e-9 m`, `9.0298e-9 rad`, and `8.9521e-11 rad`.
These timings and evaluation counts are environment-specific; rerun the CLI
for current values.
