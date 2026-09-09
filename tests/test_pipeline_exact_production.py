from pathlib import Path

import sew_mimic.pipeline.benchmark as benchmark
from sew_mimic.common import SolverResult, SolverStatus
from sew_mimic.pipeline import prepare_trajectory


def test_pipeline_creates_one_stateful_exact_solver(monkeypatch):
    prepared = prepare_trajectory(Path(__file__).parents[1] / "data" / "test.csv", max_frames=2)
    instances = []

    class FakeSolver:
        def __init__(self, *args, **kwargs):
            instances.append(self)

        def solve(self, target):
            return SolverResult("exact_sew", SolverStatus.NO_VALID_BRANCH, None)

    monkeypatch.setattr(benchmark, "ExactSewSolver", FakeSolver)
    result = benchmark.run_benchmark(prepared, methods=("exact_sew",))
    assert len(instances) == 1
    assert len(result.rows) == 2
