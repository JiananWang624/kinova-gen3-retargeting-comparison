from pathlib import Path

import numpy as np
import pandas as pd

from sew_mimic.common import SolverDiagnostics, SolverResult, SolverStatus
import sew_mimic.pipeline.benchmark as benchmark
from sew_mimic.pipeline import evaluate_result, prepare_trajectory
import scripts.replay_compare as replay_compare


def test_cli_headless_smoke_uses_precomputed_results_and_never_calls_ik(
    monkeypatch, capsys, tmp_path
):
    root = Path(__file__).resolve().parents[1]
    prepared = prepare_trajectory(root / "data" / "test.csv", max_frames=2)
    rows = []
    for frame in prepared.frames:
        result = SolverResult(
            "exact_sew",
            SolverStatus.SUCCESS_EXACT,
            np.zeros(7),
            SolverDiagnostics(solve_time_ms=0.0),
        )
        rows.append(
            evaluate_result(
                frame.frame,
                "exact_sew",
                result,
                frame.target,
                prepared.robot,
                prepared.geometry,
                prepared.stereo,
            ).to_dict()
        )
    results = tmp_path / "comparison_frames.csv"
    pd.DataFrame(rows).to_csv(results, index=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not invoke Method-2 IK")

    monkeypatch.setattr(benchmark, "ExactSewSolver", forbidden)
    monkeypatch.setattr(replay_compare, "replay_in_mujoco", forbidden)
    assert replay_compare.main(
        [
            "--input",
            str(root / "data" / "test.csv"),
            "--results",
            str(results),
            "--method",
            "exact_sew",
            "--start-frame",
            "0",
            "--max-frames",
            "2",
            "--stride",
            "1",
            "--fps",
            "60",
            "--show-sew",
            "--show-human",
            "--show-target",
            "--show-error",
            "--trail-length",
            "2",
            "--human-display-offset",
            "0.1",
            "0",
            "0",
            "--no-viewer",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "precomputed replay" in output
    assert "frame=0 method=exact_sew status=SUCCESS_EXACT" in output
    assert "frame=1 method=exact_sew status=SUCCESS_EXACT" in output
