import json
from pathlib import Path

import pandas as pd

from scripts.compare_retargeters import main


def test_cli_writes_reloadable_bounded_method0_outputs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    assert main(["--input", str(root / "data" / "test.csv"), "--max-frames", "1",
                 "--methods", "sew_mimic", "--output-dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "comparison_summary.json").read_text(encoding="utf-8"))
    assert payload["frame_selection"]["selected_frame_indices"] == [0]
    assert payload["task_point"]["mode"] in ("wrist", "wrist_plus_hand_offset")
    assert payload["exact_sew_config"]["global_partitions"] == 64
    assert payload["capabilities"]["warp_csew"]["generic_core_reproduced"] is True
    frames = pd.read_csv(tmp_path / "comparison_frames.csv")
    assert list(frames["frame"]) == [0]
    assert list(frames["method"]) == ["sew_mimic"]
    assert {f"q{index}" for index in range(1, 8)} <= set(frames.columns)
