from pathlib import Path

import numpy as np

from sew_mimic.config import CONFIG_PATH, PROJECT_ROOT, load_config, project_path
from sew_mimic.csv_adapter import (
    R_BODY_FROM_CSV,
    R_INPUT_ALIGN,
    WRIST_EULER_CONVENTION,
    WRIST_EULER_DEGREES,
    WRIST_EULER_ORDER,
)
from sew_mimic.mounting import DEFAULT_ROBOT_WORLD_OFFSET
from sew_mimic.common.task_point import (
    DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M,
    DEFAULT_TASK_POINT_MODE,
)


def test_project_config_loads_from_the_single_root_yaml() -> None:
    config = load_config()

    assert CONFIG_PATH == PROJECT_ROOT / "config.yaml"
    assert set(config) == {
        "robot",
        "human_csv",
        "task_point",
        "stereo_sew",
        "benchmark",
        "replay_csv",
        "synthetic_replay",
        "first_frame_validation",
        "mounting_validation",
        "self_consistency",
        "wrist_validation",
    }
    assert project_path(config["robot"]["model_path"]).is_file()
    assert config["benchmark"]["timing"] == {
        "enabled": True,
        "report_every_n_frames": 10,
    }


def test_configured_coordinate_and_mounting_defaults_preserve_behavior() -> None:
    np.testing.assert_array_equal(
        R_BODY_FROM_CSV,
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    np.testing.assert_array_equal(
        R_INPUT_ALIGN,
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert WRIST_EULER_ORDER == "xyz"
    assert WRIST_EULER_DEGREES is True
    assert WRIST_EULER_CONVENTION == "extrinsic"
    offset = np.asarray(DEFAULT_ROBOT_WORLD_OFFSET, dtype=float)
    assert offset.shape == (3,)
    assert np.all(np.isfinite(offset))
    assert DEFAULT_TASK_POINT_MODE == "wrist"
    np.testing.assert_array_equal(
        DEFAULT_HUMAN_WRIST_TO_TASK_OFFSET_M, np.zeros(3)
    )


def test_load_config_rejects_a_non_mapping_document(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    try:
        load_config(path)
    except ValueError as error:
        assert "must contain a YAML mapping" in str(error)
    else:
        raise AssertionError("load_config accepted a non-mapping YAML document")
