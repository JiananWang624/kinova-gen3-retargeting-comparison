# Stereo-SEW definition

The project uses the stereographic shoulder-elbow-wrist representation from
`sew_mimic.sew.stereo`. A shoulder S, elbow E, and wrist W define an oriented
half-plane. The reference is an explicit orthonormal pair `(e_t,e_r)`;
`e_t` determines the stereographic singular half-line and `e_r` fixes the
zero/sign convention for the wrapped redundancy angle `psi`.

The TIAGo Pro engineering task definition is fixed in `config.yaml`:

```text
S = J1 arm-root anchor
E = J4 anchor
W = arm_right_tool_link position (human Wrist task correspondence)
e_t = [-1, 0, 0]
e_r = [ 0, 1, 0]
```

Human S/E/W use the same stored body-world to MuJoCo-world rigid transform
before computing target `psi`; the robot points come directly from
MuJoCo joint geometry. W is not a physical spherical-wrist center. The solver
does not select S/E/W based on IK success.
The returned angle is wrapped to `[-pi,pi]`, and comparisons use a wrapped
angular difference. Degenerate shoulder-wrist or shoulder-elbow directions,
collinear S/E/W, and the stereographic pole raise explicit singularity errors.

The reported SEW singularity margin is the minimum of the normalized pole
margin and the normalized elbow-collinearity margin. Zero is singular.
This representation describes a redundancy coordinate only; it is not by
itself an IK solver. See `src/sew_mimic/sew/stereo.py` and
`tests/test_stereo_sew.py` for the exact numerical contract.
