import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sew_mimic.common import (
    HumanArmTarget,
    SolverStatus,
    evaluate_end_effector,
    gen3_end_effector_pose,
)
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.pipeline import PreparedTrajectory, TrajectoryFrame
from sew_mimic.sew import (
    Gen3StereoSewGeometry,
    StereoSew,
    project_stereo_sew_reference,
)
from sew_mimic.visualization import (
    ComparisonRow,
    ReplayOptions,
    ReplaySequence,
    ReplayState,
    TrailBuffer,
    build_replay_frames,
    load_comparison_rows,
    prepare_replay_sequence,
    validate_replay_consistency,
)


FIELDS = [
    "frame",
    "method",
    "status",
    "ee_position_error_mm",
    "ee_orientation_error_deg",
    "sew_angle_error_deg",
    "joint_limit_margin_deg",
    "branch_id",
    "solve_time_ms",
    "message",
    "diagnostics_json",
    *(f"q{index}" for index in range(1, 8)),
]


def _csv_row(frame, method, status, q=None):
    row = {
        "frame": frame,
        "method": method,
        "status": status,
        "ee_position_error_mm": "" if q is None else "1.25",
        "ee_orientation_error_deg": "" if q is None else "2.5",
        "sew_angle_error_deg": "" if q is None else "3.75",
        "joint_limit_margin_deg": "" if q is None else "4.0",
        "branch_id": "branch-a" if q is not None else "",
        "solve_time_ms": "5.0",
        "message": "",
        "diagnostics_json": json.dumps({"metadata": {"candidate_count": 4}}),
    }
    row.update(
        {
            f"q{index}": "" if q is None else q[index - 1]
            for index in range(1, 8)
        }
    )
    return row


def _write(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("method", ("sew_mimic", "exact_sew"))
def test_method0_and_method2_precomputed_rows_load(tmp_path, method):
    q = np.arange(7, dtype=float) / 10.0
    path = tmp_path / "comparison.csv"
    _write(path, [_csv_row(3, method, "SUCCESS_EXACT", q)])
    row = load_comparison_rows(path, method, (3,))[3]
    assert row.method == method
    assert row.status == SolverStatus.SUCCESS_EXACT.value
    np.testing.assert_allclose(row.q, q)
    assert row.ee_position_error_mm == 1.25
    assert row.diagnostics["metadata"]["candidate_count"] == 4


def test_failed_frame_hold_policy_never_fabricates_q_or_error_vector():
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    points = geometry.sew_points(np.zeros(7))
    pinch, rotation = gen3_end_effector_pose(np.zeros(7), robot)
    target = HumanArmTarget(
        points.shoulder, points.elbow, points.wrist, rotation, pinch
    )
    prepared = PreparedTrajectory(
        robot,
        geometry,
        stereo,
        tuple(TrajectoryFrame(frame, target) for frame in range(3)),
    )

    def result(frame, q):
        metrics = None if q is None else evaluate_end_effector(q, target, robot)
        return ComparisonRow(
            frame,
            "exact_sew",
            "SUCCESS_EXACT" if q is not None else "UNREACHABLE",
            q,
            None if metrics is None else metrics.ee_position_error_mm,
            None if metrics is None else metrics.ee_orientation_error_deg,
            0.0 if q is not None else None,
            None if metrics is None else metrics.joint_limit_margin_deg,
            None,
            None,
            None,
            {},
        )

    q = np.zeros(7)
    sequence = ReplaySequence(
        prepared,
        "exact_sew",
        (result(0, None), result(1, q), result(2, None)),
    )
    frames = build_replay_frames(sequence)
    assert frames[0].display_q is None
    assert frames[0].held_robot_pose is False
    assert not any(line.name == "ee_position_error" for line in frames[0].overlay.lines)
    np.testing.assert_allclose(frames[1].display_q, q)
    assert frames[1].held_robot_pose is False
    np.testing.assert_allclose(frames[2].display_q, q)
    assert frames[2].held_robot_pose is True
    assert not any(line.name == "ee_position_error" for line in frames[2].overlay.lines)


def test_replay_state_and_trails_are_copying_and_strictly_bounded():
    failed = ComparisonRow(
        0, "exact_sew", "UNREACHABLE", None, None, None, None, None, None, None, None, {}
    )
    state = ReplayState()
    assert state.advance(failed).display_q is None
    trail = TrailBuffer(2)
    value = np.zeros(3)
    trail.append(value)
    value[:] = 99.0
    trail.append(np.ones(3))
    trail.append(np.full(3, 2.0))
    values = trail.values()
    assert len(values) == 2
    np.testing.assert_array_equal(values[0], np.ones(3))
    np.testing.assert_array_equal(values[1], np.full(3, 2.0))
    disabled = TrailBuffer(0)
    disabled.append(np.ones(3))
    assert disabled.values() == ()


def test_in_memory_failure_row_cannot_carry_a_fabricated_configuration():
    with pytest.raises(ValueError, match="failed rows must omit q"):
        ComparisonRow(
            0,
            "exact_sew",
            "UNREACHABLE",
            np.zeros(7),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            {},
        )


def test_prepare_replay_sequence_uses_deterministic_source_frame_indices():
    root = Path(__file__).resolve().parents[1]
    sequence = prepare_replay_sequence(
        root / "data" / "test.csv",
        root / "output" / "comparison_frames.csv",
        method="exact_sew",
        start_frame=2,
        max_frames=3,
        stride=2,
    )
    assert tuple(frame.frame for frame in sequence.prepared.frames) == (2, 4, 6)
    assert tuple(row.frame for row in sequence.rows) == (2, 4, 6)


def test_replay_consistency_rejects_stale_phase7_metrics():
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    stereo = StereoSew(project_stereo_sew_reference())
    q = np.zeros(7)
    points = geometry.sew_points(q)
    pinch, rotation = gen3_end_effector_pose(q, robot)
    target = HumanArmTarget(
        points.shoulder, points.elbow, points.wrist, rotation, pinch
    )
    prepared = PreparedTrajectory(
        robot, geometry, stereo, (TrajectoryFrame(0, target),)
    )
    stale = ComparisonRow(
        0,
        "exact_sew",
        "SUCCESS_EXACT",
        q,
        123.0,
        0.0,
        0.0,
        0.0,
        None,
        1.0,
        None,
        {},
    )
    with pytest.raises(ValueError, match="stored ee_position_error_mm disagrees"):
        validate_replay_consistency(ReplaySequence(prepared, "exact_sew", (stale,)))


def test_result_loading_rejects_warp_and_missing_selected_rows(tmp_path):
    path = tmp_path / "comparison.csv"
    _write(path, [])
    with pytest.raises(ValueError, match="has not reproduced.*Kinova parameterization"):
        load_comparison_rows(path, "warp_csew", (0,))
    with pytest.raises(ValueError, match="selected frames"):
        load_comparison_rows(path, "exact_sew", (0,))
