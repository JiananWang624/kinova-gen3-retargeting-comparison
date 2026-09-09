# Generic WARP c-SEW Core

This package reproduces the corrected-skeleton portion of WARP for a generic
robot satisfying WARP's fixed-link arm assumptions. It intentionally stops at
the corrected `(shoulder, elbow, wrist, hand rotation)` geometry and does not
call SEW-Mimic or produce robot joint angles.

Reference: [WARP: Whole-Body Retargeting for Learning from Offline Human
Demonstrations](https://arxiv.org/html/2606.29940v2), equations 2–3.

## Fixed geometry

`WarpArmGeometry` contains one shoulder position, fixed positive upper-arm and
forearm lengths, and a fixed wrist-to-task vector expressed in the hand frame.
The values are immutable and are never recomputed from a candidate robot
configuration.

For each human arm, the scale-independent directions are

```text
u_h = unit(E_h - S_h)
l_h = unit(W_h - E_h)
```

and the predicted robot task point is

```text
t_hat = S_r + L_SE u_h + L_EW l_h + H_h p_WT.
```

For `N >= 1`, `compute_adaptive_offset()` returns

```text
offset = centroid(t_h) - centroid(t_hat).
```

`N=2` is the paper's bimanual equation. `N=1` is an explicit single-arm
adaptation. The function returns the offset and never moves a robot or mutates
geometry.

## Corrected skeleton

`construct_warp_skeleton()` receives the target task point already expressed
in the desired robot frame. Human `S/E/W`, human hand orientation, the robot
shoulder, the target point, and the supplied Stereo-SEW reference must all use
that same orientation-aligned coordinate frame. It applies no hidden transform
or adaptive offset. The caller is responsible for transforming all inputs
consistently and applying any virtual placement exactly once.

Hand orientation transfers directly, `H_r = H_h`, and wrist position follows
from the hard palm constraint:

```text
W_r = t_target_robot - H_r p_WT
W_r + H_r p_WT = t_target_robot.
```

The supplied `StereoSew` instance transfers the human redundancy parameter:

```text
psi_h = stereo.forward(S_h, E_h, W_h)
n_hat = stereo.inverse(S_r, W_r, psi_h).plane_normal.
```

The elbow is reconstructed through the validated SP3 implementation:

```text
e_SW = unit(W_r - S_r)
theta = sp3(L_SE e_SW, W_r - S_r, n_hat, L_EW)
E_r = S_r + R(n_hat, theta) (L_SE e_SW).
```

Only SP3 roots already classified exact are considered. The WARP branch rule
is `theta > 1e-12 rad`; least-squares roots and exact nonpositive roots produce
distinct non-success statuses. Exact results are independently checked at
`1e-12 m` for palm and link lengths and `1e-10 rad` for Stereo-SEW transfer.

The validation script generated 1,000 deterministic exact fixed-link cases
whose human upper/forearm lengths differ from the robot geometry. All 1,000
succeeded. Maximum observed errors were:

- palm: `0 m`
- upper-arm length: `1.665e-16 m`
- forearm length: `2.776e-16 m`
- Stereo-SEW: `8.882e-16 rad`

## Previous Gen3 point definition

Exact WARP assumes configuration-invariant `S`, `L_SE`, `L_EW`, and `p_WT`.
`check_warp_fixed_geometry_compatibility()` evaluates these quantities over
deterministic mechanically valid configurations using the validated Gen3
Exact-SEW `S1/E45/W67` points and aligned MuJoCo pinch-site FK.

For 1,000 configurations with seed `20260912`:

```text
L_SE min/max       0.3545435742 / 0.5496802542 m
L_SE std/variation 0.0597750822 / 0.1951366800 m
L_EW variation     2.776e-16 m
p_WT mean          [0.167455, 0, 0] m
p_WT max deviation 1.976e-15 m
```

This proves only that the Exact-SEW point definition is not a WARP fixed-link
skeleton. It does not by itself rule out a different virtual Gen3 skeleton.

## Phase W1 virtual-skeleton identification

`scripts/identify_gen3_warp_skeleton.py` tests the more natural consecutive
axis-intersection candidate

```text
S23 = intersection of axes 2/3
E45 = intersection of axes 4/5
W67 = intersection of axes 6/7.
```

On 1,000 deterministic mechanically valid validation configurations (seed
`20260915`), this candidate gives:

```text
L_SE variation       2.220e-16 m
L_EW variation       2.776e-16 m
p_WT max deviation   1.885e-15 m
```

The lengths and aligned-hand-frame wrist offset are therefore invariant to
floating-point precision. The shoulder is not invariant in the native base
frame: `S23` moves on a circle of radius `0.01175 m` when q1 varies. It is fixed
relative to a joint-1-local frame, but that frame depends on q1, which is part
of the arm configuration being solved. It is not the shoulder point WARP
requires to be known before the arm solve.

The decisive test uses the existing, independently validated SEW-Mimic
directions

```text
u_proxy = R_0_3(q) (-h3)
l_proxy = R_0_5(q) (-h5)
```

in one global fixed model:

```text
p_pinch(q) = S_fixed + L_SE u_proxy(q) + L_EW l_proxy(q)
             + H_robot(q) p_WT.
```

The geometry-derived parameters are

```text
S_fixed = [0, 0, 0.28481] m
L_SE    = 0.420953132902 m
L_EW    = 0.314360194952 m
p_WT    = [0.167455, 0, 0] m
```

and have `15.33 mm` mean / `24.84 mm` maximum validation residual. A separate
least-squares fit on 1,000 configurations (seed `20260914`) identifies

```text
S_fixed = [-0.000171546, 0.000654709, 0.284987292] m
L_SE    = 0.420730334 m
L_EW    = 0.314475843 m
p_WT    = [0.167161286, 0.000676738, 0.000284848] m
```

and, without refitting on the independent validation set, gives:

```text
mean     15.349 mm
median   16.886 mm
P95      24.484 mm
maximum  25.901 mm
```

The mathematical-exact threshold is `1e-10 m`. The separate practical
classification threshold is the project's `1 mm` position threshold; it is
never used to call a model mathematically exact. The result for the tested
candidate set is therefore `NO_USEFUL_FIXED_SKELETON`, not an exact or
practical approximate result for those definitions.

A fixed projection of `S23` onto the joint-1 axis reconstructs position within
`0.364 mm` on the independent validation set when its own configuration-
dependent link direction is used, but
its `L_SE` still varies by `0.711 mm` and its direction differs from the h3
proxy by as much as `0.0582 rad`. It does not satisfy the required proxy model.
The earlier `S1/S23/E45` candidate fails through a configuration-dependent
hand-frame offset (about `0.395 m` maximum deviation).

## What the public material establishes

The [WARP paper](https://arxiv.org/html/2606.29940v2#S3) specifies fixed robot
`L_SE`, `L_EW`, and `p_WT` read from robot geometry, applies an adaptive body
offset, constructs a corrected c-SEW skeleton, and then passes that skeleton to
SEW-Mimic. The WARP paper/project materials explicitly demonstrate
Dual-Kinova3, so that demonstration is direct evidence that the authors have a
Kinova path.

The [SEW-Mimic paper](https://arxiv.org/html/2602.01632v1#S3) supports h3/h5 as
the generic robot direction proxies. The available public material does not,
however, provide enough Kinova-specific implementation detail to reproduce the
demonstrated path: the exact S/E/W points, fixed virtual link parameters,
calibration choices, and any robot-specific adaptation are **NOT SPECIFIED IN
ENOUGH DETAIL TO RECONSTRUCT THE AUTHORS' PARAMETERIZATION**. In particular,
neither `S1` nor `S23` is source-specified as the exact Kinova WARP shoulder;
both are project hypotheses tested here.

The previous broad statement “Gen3 is WARP incompatible” is incorrect. This
project established only that `S1/E45/W67`, `S23/E45/W67`, the other listed
virtual definitions, and the tested h3/h5 fixed models do not reproduce the
demonstrated Kinova path exactly. Method 2 Exact-SEW remains the validated
executable method in this repository while the authors' Kinova parameterization
is unknown here; a Gen3 WARP implementation cannot be claimed as reproduced
without that missing detail or an independently validated equivalent.
