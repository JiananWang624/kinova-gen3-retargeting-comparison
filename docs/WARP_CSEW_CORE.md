# Generic WARP corrected-skeleton core

The retained `sew_mimic.warp` code computes the corrected-skeleton geometry
for a robot with a **fixed** shoulder, upper/forearm lengths, and wrist-to-TCP
offset. It does not solve TIAGo joint angles and is not a production fallback.
Its input is a human shoulder, elbow, wrist, and hand orientation; its output
is corrected geometric points for a supplied immutable `WarpArmGeometry`.
Reference: [WARP, equations 2–3](https://arxiv.org/html/2606.29940v2).

The assumptions must be checked on a concrete robot before calling the core
an executable retargeter. `sew_mimic.tiago.warp.diagnose_warp_invariance`
samples actual TIAGo MuJoCo FK and reports variation of the shoulder, both
segment lengths, and wrist-to-TCP offset for physical candidate S/E/W points.
For the locked `J1/J4/wrist_center` definition, J1 is fixed and the lower
segment/tool offset are invariant, but the upper segment varies by about
87.5 mm in the diagnostic sample. That fails the configured 1 mm fixed-skeleton
threshold, so TIAGo WARP remains **diagnostic-only**. No approximate fixed
skeleton is inserted into the production path.

The generic corrected-skeleton implementation and its tests are retained for
research reproducibility; all robot-specific runtime behavior remains TIAGo
only. See `src/sew_mimic/warp/geometry.py`, `skeleton.py`, and
`tests/test_warp_geometry.py` / `tests/test_warp_skeleton.py`.
