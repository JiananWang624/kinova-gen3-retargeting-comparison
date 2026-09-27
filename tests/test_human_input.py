from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from sew_mimic.csv_adapter import (
    R_BODY_FROM_CSV,
    R_INPUT_ALIGN,
    HumanCSVAdapter,
    REQUIRED_COLUMNS,
    load_human_trajectory_csv,
)
from sew_mimic.geometry import rot
from sew_mimic.human_input import (
    compute_lower_arm_direction,
    compute_upper_arm_direction,
    transform_human_to_robot_body_frame,
    wrist_euler_to_rotation,
)


RNG = np.random.default_rng(20260831)
AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _random_unit() -> np.ndarray:
    vector = RNG.normal(size=3)
    return vector / np.linalg.norm(vector)


def test_random_upper_and_lower_arm_directions_follow_paper_definition() -> None:
    for _ in range(200):
        shoulder = RNG.normal(size=3)
        upper_direction = _random_unit()
        upper_length = RNG.uniform(0.1, 1.0)
        elbow = shoulder + upper_length * upper_direction
        lower_direction = _random_unit()
        lower_length = RNG.uniform(0.1, 1.0)
        wrist = elbow + lower_length * lower_direction

        np.testing.assert_allclose(
            compute_upper_arm_direction(shoulder, elbow), upper_direction, atol=8e-16
        )
        np.testing.assert_allclose(
            compute_lower_arm_direction(elbow, wrist), lower_direction, atol=8e-16
        )


@pytest.mark.parametrize(
    ("function", "start", "end"),
    [
        (compute_upper_arm_direction, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        (compute_lower_arm_direction, [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]),
    ],
)
def test_arm_direction_rejects_coincident_keypoints(function, start, end) -> None:
    with pytest.raises(ValueError):
        function(start, end)


@pytest.mark.parametrize("order", ["xyz", "zyx", "xzx"])
@pytest.mark.parametrize("convention", ["intrinsic", "extrinsic"])
@pytest.mark.parametrize("degrees", [False, True])
def test_random_wrist_euler_conventions_have_expected_composition(
    order: str, convention: str, degrees: bool
) -> None:
    for _ in range(50):
        radians = RNG.uniform(-np.pi, np.pi, size=3)
        angles = np.degrees(radians) if degrees else radians
        actual = wrist_euler_to_rotation(
            angles,
            order=order,
            degrees=degrees,
            convention=convention,
        )

        rotations = [rot(AXES[axis], angle) for axis, angle in zip(order, radians)]
        if convention == "intrinsic":
            expected = rotations[0] @ rotations[1] @ rotations[2]
        else:
            expected = rotations[2] @ rotations[1] @ rotations[0]

        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
        np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=2e-15)
        assert np.linalg.det(actual) == pytest.approx(1.0, abs=2e-15)


def test_identity_human_to_robot_body_transform_preserves_synthetic_input() -> None:
    shoulder = np.array([0.2, -0.3, 1.4])
    elbow = np.array([0.4, -0.5, 1.1])
    wrist = np.array([0.7, -0.2, 0.9])
    hand = wrist_euler_to_rotation(
        [10.0, -20.0, 30.0],
        order="xyz",
        degrees=True,
        convention="extrinsic",
    )

    transformed = transform_human_to_robot_body_frame(
        shoulder,
        elbow,
        wrist,
        hand,
        rotation_robot_from_human=np.eye(3),
        translation_robot_from_human=np.zeros(3),
    )

    for actual, expected in zip(transformed, (shoulder, elbow, wrist, hand)):
        np.testing.assert_allclose(actual, expected, atol=0.0)


def test_random_frame_transform_rotates_arm_directions_and_hand_orientation() -> None:
    for _ in range(100):
        shoulder = RNG.normal(size=3)
        elbow = shoulder + RNG.normal(size=3)
        wrist = elbow + RNG.normal(size=3)
        hand = rot(_random_unit(), RNG.uniform(-np.pi, np.pi))
        frame_rotation = rot(_random_unit(), RNG.uniform(-np.pi, np.pi))
        frame_translation = RNG.normal(size=3)

        s_robot, e_robot, w_robot, h_robot = transform_human_to_robot_body_frame(
            shoulder,
            elbow,
            wrist,
            hand,
            rotation_robot_from_human=frame_rotation,
            translation_robot_from_human=frame_translation,
        )

        np.testing.assert_allclose(
            compute_upper_arm_direction(s_robot, e_robot),
            frame_rotation @ compute_upper_arm_direction(shoulder, elbow),
            atol=2e-15,
        )
        np.testing.assert_allclose(
            compute_lower_arm_direction(e_robot, w_robot),
            frame_rotation @ compute_lower_arm_direction(elbow, wrist),
            atol=2e-15,
        )
        np.testing.assert_allclose(h_robot, frame_rotation @ hand, atol=2e-15)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"order": "XYZ", "degrees": False, "convention": "intrinsic"}, "order"),
        ({"order": "xyz", "degrees": "yes", "convention": "intrinsic"}, "degrees"),
        ({"order": "xyz", "degrees": False, "convention": "moving"}, "convention"),
    ],
)
def test_wrist_euler_requires_explicit_valid_convention(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        wrist_euler_to_rotation([0.1, 0.2, 0.3], **kwargs)


def test_csv_adapter_uses_extrinsic_xyz_degrees_and_separate_input_alignment(
    tmp_path,
) -> None:
    row = {
        "Shoulder_X": 100.0,
        "Shoulder_Y": 200.0,
        "Shoulder_Z": 300.0,
        "Elbow_X": 400.0,
        "Elbow_Y": 500.0,
        "Elbow_Z": 600.0,
        "Wrist_X": 700.0,
        "Wrist_Y": 800.0,
        "Wrist_Z": 900.0,
        "Wrist_Rx": 20.0,
        "Wrist_Ry": -30.0,
        "Wrist_Rz": 40.0,
    }
    csv_path = tmp_path / "human.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    frame_rotation = rot([0.0, 0.0, 1.0], 0.5)
    adapter = HumanCSVAdapter(frame_rotation, position_scale=0.001)

    trajectory = load_human_trajectory_csv(csv_path, adapter)

    np.testing.assert_allclose(
        trajectory.shoulders[0], frame_rotation @ np.array([0.1, 0.2, 0.3])
    )
    np.testing.assert_allclose(
        trajectory.elbows[0], frame_rotation @ np.array([0.4, 0.5, 0.6])
    )
    np.testing.assert_allclose(
        trajectory.wrists[0], frame_rotation @ np.array([0.7, 0.8, 0.9])
    )
    raw_hand = wrist_euler_to_rotation(
        [20.0, -30.0, 40.0], order="xyz", degrees=True, convention="extrinsic"
    )
    expected_hand = frame_rotation @ raw_hand @ R_INPUT_ALIGN
    np.testing.assert_allclose(trajectory.hand_orientations[0], expected_hand, atol=2e-15)


def test_csv_adapter_default_matches_supplied_absolute_body_frame() -> None:
    adapter = HumanCSVAdapter()

    np.testing.assert_array_equal(
        adapter.rotation_body_from_csv,
        R_BODY_FROM_CSV,
    )
    np.testing.assert_array_equal(
        adapter.rotation_body_from_csv @ np.eye(3),
        np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])),
    )
    np.testing.assert_allclose(
        adapter.rotation_body_from_csv.T @ adapter.rotation_body_from_csv,
        np.eye(3),
        atol=0.0,
    )
    assert np.linalg.det(adapter.rotation_body_from_csv) == pytest.approx(1.0)
    np.testing.assert_array_equal(adapter.rotation_input_align, R_INPUT_ALIGN)
    np.testing.assert_array_equal(
        R_INPUT_ALIGN,
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert adapter.position_scale == 0.001


def test_csv_adapter_rejects_previous_improper_reflection() -> None:
    previous_mapping = np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )

    with pytest.raises(ValueError, match=r"determinant \+1"):
        HumanCSVAdapter(rotation_body_from_csv=previous_mapping)


def test_csv_adapter_proper_frame_orientation_stays_in_so3(tmp_path) -> None:
    row = {column: 0.0 for column in REQUIRED_COLUMNS}
    row["Elbow_Y"] = -300.0
    row["Wrist_Y"] = -600.0
    row["Wrist_Rx"] = 0.2
    row["Wrist_Ry"] = -0.3
    row["Wrist_Rz"] = 0.4
    csv_path = tmp_path / "handedness.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    hand = load_human_trajectory_csv(csv_path).hand_orientations[0]
    raw_hand = wrist_euler_to_rotation(
        [0.2, -0.3, 0.4], order="xyz", degrees=True, convention="extrinsic"
    )
    expected = R_BODY_FROM_CSV @ raw_hand @ R_INPUT_ALIGN

    np.testing.assert_allclose(hand, expected, atol=2e-15)
    np.testing.assert_allclose(hand.T @ hand, np.eye(3), atol=2e-15)
    assert np.linalg.det(hand) == pytest.approx(1.0, abs=2e-15)


def test_input_alignment_maps_canonical_axes_to_motive_minus_y_plus_x_plus_z() -> None:
    raw_device_orientation = rot([0.3, -0.4, 0.5], 0.7)
    canonical = raw_device_orientation @ R_INPUT_ALIGN

    np.testing.assert_allclose(canonical[:, 0], -raw_device_orientation[:, 1])
    np.testing.assert_allclose(canonical[:, 1], raw_device_orientation[:, 0])
    np.testing.assert_allclose(canonical[:, 2], raw_device_orientation[:, 2])
    np.testing.assert_allclose(canonical.T @ canonical, np.eye(3), atol=2e-15)
    assert np.linalg.det(canonical) == pytest.approx(1.0, abs=2e-15)


def test_input_alignment_is_supported_by_full_human_device_trajectory() -> None:
    table = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "test.csv")
    rotations_csv = Rotation.from_euler(
        "xyz",
        table.loc[:, ["Wrist_Rx", "Wrist_Ry", "Wrist_Rz"]].to_numpy(),
        degrees=True,
    ).as_matrix()
    pointing_csv = -rotations_csv[:, :, 1]
    lower_arm_csv = (
        table.loc[:, ["Wrist_X", "Wrist_Y", "Wrist_Z"]].to_numpy()
        - table.loc[:, ["Elbow_X", "Elbow_Y", "Elbow_Z"]].to_numpy()
    )
    lower_arm_csv /= np.linalg.norm(lower_arm_csv, axis=1)[:, None]
    palm_normal_from_geometry = np.cross(lower_arm_csv, pointing_csv)
    palm_normal_from_geometry /= np.linalg.norm(
        palm_normal_from_geometry, axis=1
    )[:, None]
    device_plus_x = rotations_csv[:, :, 0]
    median_cosine = float(
        np.median(np.sum(palm_normal_from_geometry * device_plus_x, axis=1))
    )

    assert median_cosine > 0.99


def test_csv_adapter_rejects_missing_or_nonfinite_required_values(tmp_path) -> None:
    complete = {column: 0.1 for column in REQUIRED_COLUMNS}
    complete["Elbow_X"] = 1.0
    complete["Wrist_Y"] = 1.0

    missing_path = tmp_path / "missing.csv"
    pd.DataFrame([{key: value for key, value in complete.items() if key != "Wrist_Rz"}]).to_csv(
        missing_path, index=False
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_human_trajectory_csv(missing_path)

    nonfinite_path = tmp_path / "nonfinite.csv"
    complete["Wrist_Rz"] = np.nan
    pd.DataFrame([complete]).to_csv(nonfinite_path, index=False)
    with pytest.raises(ValueError, match=r"non-finite.*rows \[0\]"):
        load_human_trajectory_csv(nonfinite_path)
