"""Precomputed comparison loading, headless replay state, and MuJoCo rendering."""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Sequence

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common import SolverDiagnostics, SolverResult, SolverStatus, gen3_end_effector_pose
from ..config import CONFIG
from ..pipeline import PreparedTrajectory, prepare_trajectory
from ..pipeline.evaluator import evaluate_result
from .overlay import OverlayFrame, build_overlay, overlay_base_to_world


Vector = NDArray[np.float64]
SUPPORTED_PLAYBACK_METHODS = ("sew_mimic", "exact_sew", "numerical_oracle")
_SUCCESS = {SolverStatus.SUCCESS_EXACT.value, SolverStatus.SUCCESS_APPROX.value}
_REQUIRED_COLUMNS = {
    "frame",
    "method",
    "status",
    "ee_position_error_mm",
    "ee_orientation_error_deg",
    "sew_angle_error_deg",
    "joint_limit_margin_deg",
    "branch_id",
    *(f"q{index}" for index in range(1, 8)),
}
_AXIS_COLORS = (
    (1.0, 0.1, 0.1, 1.0),
    (0.1, 1.0, 0.1, 1.0),
    (0.1, 0.35, 1.0, 1.0),
)
_AXIS_WIDTH_M = 0.003
_WORLD_FRAME_LENGTH_M = 0.20
_STORED_METRIC_ABSOLUTE_TOLERANCE = 1e-8
_STORED_METRIC_RELATIVE_TOLERANCE = 1e-10


def _optional_float(value: str, name: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"comparison CSV {name} must be numeric") from error
    return number if np.isfinite(number) else None


def _optional_text(value: str) -> str | None:
    text = value.strip()
    return text or None


def _configuration(item: dict[str, str], success: bool) -> Vector | None:
    values = np.asarray(
        [
            np.nan if not item[f"q{index}"].strip() else float(item[f"q{index}"])
            for index in range(1, 8)
        ],
        dtype=float,
    )
    finite = np.isfinite(values)
    if success:
        if not np.all(finite):
            raise ValueError("successful comparison row requires seven finite q values")
        return values
    if np.any(finite):
        raise ValueError("failed comparison row must not contain a robot q")
    return None


@dataclass(frozen=True)
class ComparisonRow:
    """One validated Phase-7 row used strictly for display."""

    frame: int
    method: str
    status: str
    q: Vector | None
    ee_position_error_mm: float | None
    ee_orientation_error_deg: float | None
    sew_angle_error_deg: float | None
    joint_limit_margin_deg: float | None
    branch_id: str | None
    solve_time_ms: float | None
    message: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, int) or self.frame < 0:
            raise ValueError("frame must be a nonnegative integer")
        if self.method not in SUPPORTED_PLAYBACK_METHODS:
            raise ValueError("method is not an executable Gen3 playback method")
        if self.status not in {status.value for status in SolverStatus}:
            raise ValueError("status must be a SolverStatus value")
        configuration = None if self.q is None else np.asarray(self.q, dtype=float)
        if configuration is not None:
            if configuration.shape != (7,) or not np.all(np.isfinite(configuration)):
                raise ValueError("q must be finite with shape (7,) when present")
            configuration = configuration.copy()
        if (self.status in _SUCCESS) != (configuration is not None):
            raise ValueError("successful rows require q and failed rows must omit q")
        object.__setattr__(self, "q", configuration)
        for name in (
            "ee_position_error_mm",
            "ee_orientation_error_deg",
            "sew_angle_error_deg",
            "joint_limit_margin_deg",
            "solve_time_ms",
        ):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when present")
        if configuration is not None and any(
            getattr(self, name) is None
            for name in (
                "ee_position_error_mm",
                "ee_orientation_error_deg",
                "joint_limit_margin_deg",
            )
        ):
            raise ValueError(
                "successful rows require position, orientation, and joint-margin metrics"
            )
        if not isinstance(self.diagnostics, dict):
            raise ValueError("diagnostics must be a dictionary")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def successful(self) -> bool:
        return self.status in _SUCCESS


def load_comparison_rows(
    path: str | Path,
    method: str,
    selected_frames: Sequence[int],
) -> dict[int, ComparisonRow]:
    """Load exactly the requested precomputed method/frame rows.

    No solver is imported or invoked here. Missing frames and invalid result
    contracts fail before an interactive viewer can open.
    """
    if method == "warp_csew":
        raise ValueError(
            "warp_csew has a reproduced generic core but is not executable "
            "because this repository has not reproduced the demonstrated "
            "Kinova parameterization"
        )
    if method not in SUPPORTED_PLAYBACK_METHODS:
        raise ValueError(
            f"unsupported playback method {method!r}; expected one of "
            f"{SUPPORTED_PLAYBACK_METHODS}"
        )
    frame_ids = tuple(int(frame) for frame in selected_frames)
    if len(frame_ids) != len(set(frame_ids)) or any(frame < 0 for frame in frame_ids):
        raise ValueError("selected_frames must contain unique nonnegative integers")

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set() if reader.fieldnames is None else set(reader.fieldnames)
        missing_columns = sorted(_REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(
                f"comparison CSV is missing required columns: {missing_columns}"
            )
        raw_rows = list(reader)

    allowed_statuses = {status.value for status in SolverStatus}
    rows: dict[int, ComparisonRow] = {}
    for item in raw_rows:
        if item["method"] != method:
            continue
        try:
            frame_value = float(item["frame"])
        except ValueError as error:
            raise ValueError("comparison CSV frame must be an integer") from error
        frame = int(frame_value)
        if frame_value != frame or frame < 0:
            raise ValueError("comparison CSV frame must be a nonnegative integer")
        if item["status"] not in allowed_statuses:
            raise ValueError("comparison CSV contains an invalid SolverStatus")
        success = item["status"] in _SUCCESS
        try:
            q = _configuration(item, success)
        except ValueError as error:
            raise ValueError(f"invalid q at frame {frame}: {error}") from error
        diagnostics_text = item.get("diagnostics_json", "").strip()
        try:
            diagnostics = {} if not diagnostics_text else json.loads(diagnostics_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"comparison CSV diagnostics_json is invalid at frame {frame}"
            ) from error
        if not isinstance(diagnostics, dict):
            raise ValueError("comparison CSV diagnostics_json must contain an object")
        if frame in rows:
            raise ValueError("comparison CSV contains duplicate method/frame rows")
        rows[frame] = ComparisonRow(
            frame=frame,
            method=method,
            status=item["status"],
            q=q,
            ee_position_error_mm=_optional_float(
                item["ee_position_error_mm"], "ee_position_error_mm"
            ),
            ee_orientation_error_deg=_optional_float(
                item["ee_orientation_error_deg"], "ee_orientation_error_deg"
            ),
            sew_angle_error_deg=_optional_float(
                item["sew_angle_error_deg"], "sew_angle_error_deg"
            ),
            joint_limit_margin_deg=_optional_float(
                item["joint_limit_margin_deg"], "joint_limit_margin_deg"
            ),
            branch_id=_optional_text(item["branch_id"]),
            solve_time_ms=_optional_float(item.get("solve_time_ms", ""), "solve_time_ms"),
            message=_optional_text(item.get("message", "")),
            diagnostics=diagnostics,
        )

    missing_frames = sorted(set(frame_ids) - set(rows))
    if missing_frames:
        raise ValueError(
            f"comparison CSV has no {method!r} row for selected frames: "
            f"{missing_frames}"
        )
    return {frame: rows[frame] for frame in frame_ids}


@dataclass(frozen=True)
class ReplaySequence:
    prepared: PreparedTrajectory
    method: str
    rows: tuple[ComparisonRow, ...]


def prepare_replay_sequence(
    input_path: str | Path,
    results_path: str | Path,
    *,
    method: str = "exact_sew",
    start_frame: int = 0,
    max_frames: int | None = 100,
    stride: int = 1,
) -> ReplaySequence:
    """Apply Phase-7 preprocessing once and align stored rows by source frame."""
    prepared = prepare_trajectory(
        input_path,
        start_frame=start_frame,
        max_frames=max_frames,
        stride=stride,
    )
    frame_ids = tuple(frame.frame for frame in prepared.frames)
    loaded = load_comparison_rows(results_path, method, frame_ids)
    sequence = ReplaySequence(
        prepared,
        method,
        tuple(loaded[frame] for frame in frame_ids),
    )
    validate_replay_consistency(sequence)
    return sequence


def _metric_matches(stored: float | None, authoritative: float) -> bool:
    if not np.isfinite(authoritative):
        return stored is None
    if stored is None:
        return False
    return bool(
        np.isclose(
            stored,
            authoritative,
            atol=_STORED_METRIC_ABSOLUTE_TOLERANCE,
            rtol=_STORED_METRIC_RELATIVE_TOLERANCE,
        )
    )


def validate_replay_consistency(sequence: ReplaySequence) -> None:
    """Reject stale/mismatched results using the Phase-7 authoritative evaluator."""
    if len(sequence.prepared.frames) != len(sequence.rows):
        raise ValueError("prepared targets and comparison rows must have equal length")
    for target_frame, row in zip(
        sequence.prepared.frames, sequence.rows, strict=True
    ):
        if target_frame.frame != row.frame:
            raise ValueError("prepared target and comparison row frames do not match")
        if row.q is None:
            continue
        evaluated = evaluate_result(
            row.frame,
            row.method,
            SolverResult(
                row.method,
                SolverStatus(row.status),
                row.q,
                SolverDiagnostics(),
            ),
            target_frame.target,
            sequence.prepared.robot,
            sequence.prepared.geometry,
            sequence.prepared.stereo,
        )
        comparisons = (
            ("ee_position_error_mm", row.ee_position_error_mm, evaluated.ee_position_error_mm),
            (
                "ee_orientation_error_deg",
                row.ee_orientation_error_deg,
                evaluated.ee_orientation_error_deg,
            ),
            ("sew_angle_error_deg", row.sew_angle_error_deg, evaluated.sew_angle_error_deg),
            (
                "joint_limit_margin_deg",
                row.joint_limit_margin_deg,
                evaluated.joint_limit_margin_deg,
            ),
        )
        for name, stored, authoritative in comparisons:
            if not _metric_matches(stored, authoritative):
                raise ValueError(
                    f"stored {name} disagrees with authoritative evaluation "
                    f"at frame {row.frame}: stored={stored}, "
                    f"authoritative={authoritative}"
                )


class TrailBuffer:
    """A bounded copy-owning point history; length zero disables the trail."""

    def __init__(self, max_length: int = 0) -> None:
        if max_length < 0:
            raise ValueError("max_length must be nonnegative")
        self._values: deque[Vector] = deque(maxlen=max_length)

    def append(self, value: ArrayLike) -> None:
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("trail points must be finite with shape (3,)")
        self._values.append(point.copy())

    def values(self) -> tuple[Vector, ...]:
        return tuple(value.copy() for value in self._values)


@dataclass(frozen=True)
class ReplayStep:
    display_q: Vector | None
    held: bool


@dataclass
class ReplayState:
    """Track a display pose without feeding it into numerical state."""

    last_successful_q: Vector | None = None

    def advance(self, row: ComparisonRow) -> ReplayStep:
        if row.successful:
            assert row.q is not None
            self.last_successful_q = row.q.copy()
            return ReplayStep(row.q.copy(), False)
        if self.last_successful_q is None:
            return ReplayStep(None, False)
        return ReplayStep(self.last_successful_q.copy(), True)


@dataclass(frozen=True)
class ReplayOptions:
    fps: float = 30.0
    loop: bool = False
    show_human: bool = True
    show_target: bool = True
    show_sew: bool = False
    show_error: bool = True
    trail_length: int = 0
    human_display_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be positive and finite")
        if self.trail_length < 0:
            raise ValueError("trail_length must be nonnegative")
        offset = np.asarray(self.human_display_offset_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("human_display_offset_m must be finite with shape (3,)")


@dataclass(frozen=True)
class ReplayDisplayFrame:
    frame: int
    result: ComparisonRow
    display_q: Vector | None
    held_robot_pose: bool
    overlay: OverlayFrame


def build_replay_frames(
    sequence: ReplaySequence,
    options: ReplayOptions = ReplayOptions(),
) -> tuple[ReplayDisplayFrame, ...]:
    """Create every display frame headlessly, including bounded trails."""
    validate_replay_consistency(sequence)
    state = ReplayState()
    human_trail = TrailBuffer(options.trail_length)
    robot_trail = TrailBuffer(options.trail_length)
    display_frames: list[ReplayDisplayFrame] = []
    for target_frame, row in zip(
        sequence.prepared.frames, sequence.rows, strict=True
    ):
        if target_frame.frame != row.frame:
            raise ValueError("prepared target and comparison row frames do not match")
        step = state.advance(row)
        human_trail.append(target_frame.target.task_point)
        if row.q is not None:
            pinch, _ = gen3_end_effector_pose(row.q, sequence.prepared.robot)
            robot_trail.append(pinch)
        overlay = build_overlay(
            target_frame.target,
            sequence.prepared.robot,
            sequence.prepared.geometry,
            sequence.prepared.stereo,
            step.display_q,
            current_q_available=row.q is not None,
            human_display_offset_m=options.human_display_offset_m,
            show_human=options.show_human,
            show_target=options.show_target,
            show_sew=options.show_sew,
            show_error=options.show_error,
            human_task_trail=human_trail.values(),
            robot_pinch_trail=robot_trail.values(),
        )
        display_frames.append(
            ReplayDisplayFrame(
                target_frame.frame,
                row,
                step.display_q,
                step.held,
                overlay,
            )
        )
    return tuple(display_frames)


def format_frame_diagnostics(frame: ReplayDisplayFrame) -> str:
    """Format stored Phase-7 diagnostics without recomputing solver work."""
    row = frame.result
    metadata = row.diagnostics.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    extras = []
    for key in ("candidate_count", "wrist_search_angle"):
        if key in metadata:
            extras.append(f"{key}={metadata[key]}")
    parts = [
        f"frame={row.frame}",
        f"method={row.method}",
        f"status={row.status}",
        f"position_error_mm={row.ee_position_error_mm}",
        f"orientation_error_deg={row.ee_orientation_error_deg}",
        f"sew_error_deg={row.sew_angle_error_deg}",
        f"joint_limit_margin_deg={row.joint_limit_margin_deg}",
        f"branch_id={row.branch_id}",
        f"robot_pose={'held' if frame.held_robot_pose else 'current' if row.q is not None else 'unavailable'}",
        *extras,
    ]
    return " ".join(parts)


def render_overlay_into_scene(
    scene: Any,
    overlay: OverlayFrame,
    *,
    show_world_frame: bool = False,
) -> None:
    """Replace a MuJoCo user scene with one finite set of overlay primitives."""
    required = (
        len(overlay.spheres)
        + len(overlay.lines)
        + 3 * len(overlay.axes)
        + (3 if show_world_frame else 0)
    )
    if required > len(scene.geoms):
        raise ValueError(
            f"overlay needs {required} user geoms but scene capacity is {len(scene.geoms)}"
        )
    scene.ngeom = 0

    def next_geom() -> Any:
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return geom

    for primitive in overlay.spheres:
        mujoco.mjv_initGeom(
            next_geom(),
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([primitive.radius_m, 0.0, 0.0]),
            primitive.position,
            np.eye(3).ravel(),
            np.asarray(primitive.color),
        )
    for primitive in overlay.lines:
        geom = next_geom()
        geom_type = (
            mujoco.mjtGeom.mjGEOM_ARROW
            if primitive.kind == "arrow"
            else mujoco.mjtGeom.mjGEOM_CAPSULE
        )
        mujoco.mjv_connector(
            geom,
            geom_type,
            primitive.width_m,
            primitive.start,
            primitive.end,
        )
        geom.rgba[:] = primitive.color
        geom.emission = 1.0
    for primitive in overlay.axes:
        for axis, color in enumerate(_AXIS_COLORS):
            geom = next_geom()
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                _AXIS_WIDTH_M,
                primitive.origin,
                primitive.origin + primitive.length_m * primitive.rotation[:, axis],
            )
            geom.rgba[:] = color
            geom.emission = 1.0
    if show_world_frame:
        origin = np.zeros(3)
        for axis, color in enumerate(_AXIS_COLORS):
            geom = next_geom()
            endpoint = origin.copy()
            endpoint[axis] = _WORLD_FRAME_LENGTH_M
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                _AXIS_WIDTH_M,
                origin,
                endpoint,
            )
            geom.rgba[:] = color
            geom.emission = 1.0


def _set_robot_configuration(
    prepared: PreparedTrajectory,
    data: mujoco.MjData,
    q: Vector,
) -> None:
    for index, joint_id in enumerate(prepared.robot.joint_ids):
        qpos_address = prepared.robot.model.jnt_qposadr[int(joint_id)]
        data.qpos[qpos_address] = q[index]
    mujoco.mj_forward(prepared.robot.model, data)


def replay_in_mujoco(
    sequence: ReplaySequence,
    options: ReplayOptions = ReplayOptions(),
) -> None:
    """Replay stored configurations once, or loop until the viewer closes."""
    import mujoco.viewer

    displays = build_replay_frames(sequence, options)
    model = sequence.prepared.robot.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot_body_ids = set(int(value) for value in sequence.prepared.robot.frame_body_ids)
    robot_geom_mask = np.asarray(
        [int(body_id) in robot_body_ids for body_id in model.geom_bodyid], dtype=bool
    )
    original_robot_alpha = model.geom_rgba[robot_geom_mask, 3].copy()
    camera_config = CONFIG["replay_csv"]

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            first_target = sequence.prepared.frames[0].target
            base_id = int(sequence.prepared.robot.frame_body_ids[0])
            rotation_world_from_base = data.xmat[base_id].reshape(3, 3).copy()
            position_world_of_base = data.xpos[base_id].copy()
            viewer.cam.lookat[:] = (
                rotation_world_from_base @ first_target.shoulder
                + position_world_of_base
                + np.asarray(camera_config["camera_lookat_offset_m"], dtype=float)
            )
            viewer.cam.distance = float(camera_config["camera_distance_m"])
            viewer.cam.azimuth = float(camera_config["camera_azimuth_deg"])
            viewer.cam.elevation = float(camera_config["camera_elevation_deg"])

            while viewer.is_running():
                cycle_started = time.perf_counter()
                for index, display in enumerate(displays):
                    if not viewer.is_running():
                        return
                    visible = display.display_q is not None
                    model.geom_rgba[robot_geom_mask, 3] = (
                        original_robot_alpha if visible else 0.0
                    )
                    if display.display_q is not None:
                        _set_robot_configuration(
                            sequence.prepared, data, display.display_q
                        )
                    world_overlay = overlay_base_to_world(
                        display.overlay,
                        data.xmat[base_id].reshape(3, 3),
                        data.xpos[base_id],
                    )
                    render_overlay_into_scene(
                        viewer.user_scn,
                        world_overlay,
                        show_world_frame=True,
                    )
                    print(format_frame_diagnostics(display))
                    viewer.sync()
                    deadline = cycle_started + (index + 1) / options.fps
                    remaining = deadline - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)
                if not options.loop:
                    break
    finally:
        model.geom_rgba[robot_geom_mask, 3] = original_robot_alpha
