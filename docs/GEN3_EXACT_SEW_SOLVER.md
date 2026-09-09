# Gen3 Exact-SEW solver

Method 2 solves the established pinch pose plus Stereo-SEW constraint with the
private compiled `_exact_sew_core` event solver. `ExactSewSolver` is stateful:
one instance is used for a complete pipeline trajectory, performs conservative
local continuation after a successful frame, and uses deterministic global
event recovery whenever continuation is uncertain.

Python remains responsible for target conversion, deterministic joint-limit
representation, and the authoritative MuJoCo-derived `pinch_site` residual.
Only candidates below the strict 1 mm / 1 degree / 1 degree thresholds return
`SUCCESS_EXACT`. There is no Method-3 or SciPy fallback.

The only runtime settings are `exact_sew` in `config.yaml`: local radii,
minimum local partitions, global partitions, event evaluation budget, and
maximum wrapped joint step. They load strictly into `ExactSewConfig`.
