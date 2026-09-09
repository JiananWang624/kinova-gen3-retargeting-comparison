# Engineering handoff

Method 2 is now a single production path: `ExactSewSolver` reuses compiled
local continuation and deterministic global event recovery for an entire
trajectory. The C++ core keeps the established R-2R-2R-2R mathematics; Python
performs task conversion, joint representation, and authoritative
MuJoCo-derived `pinch_site` acceptance. It never routes through Method 3 or a
SciPy optimizer.

The public `solve_exact_sew()` remains available for individual targets. It
uses the same compiled global core and supports `canonical` and `continuous`
selection with `q_previous`. Pipeline and CLI intentionally expose no legacy
search/tracker/policy switches.

Install the validated Python 3.11 environment with:

```powershell
.venv\Scripts\python.exe -m pip install -e .[test]
```

Runtime Exact-SEW controls live solely in `config.yaml` under `exact_sew` and
load strictly into `ExactSewConfig`. Keep coordinate conventions, SEW
references, root tolerances, joint axes, proxy signs, and failure semantics
unchanged.

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\benchmark_exact_sew.py --input data\test.csv --max-frames 100
.venv\Scripts\python.exe scripts\compare_retargeters.py --input data\test.csv --methods sew_mimic exact_sew --max-frames 100
```
