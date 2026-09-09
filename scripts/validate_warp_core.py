"""Validate generic WARP construction and one tested Gen3 point definition."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.common import HumanArmTarget  # noqa: E402
from sew_mimic.geometry import rot  # noqa: E402
from sew_mimic.kinematics import gen3_kinematics  # noqa: E402
from sew_mimic.sew import (  # noqa: E402
    Gen3StereoSewGeometry,
    StereoSew,
    project_stereo_sew_reference,
)
from sew_mimic.warp import (  # noqa: E402
    WarpArmGeometry,
    WarpSkeletonStatus,
    check_warp_fixed_geometry_compatibility,
    construct_warp_skeleton,
)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _synthetic_human_target(
    rng: np.random.Generator, geometry: WarpArmGeometry
) -> tuple[HumanArmTarget, np.ndarray]:
    upper_direction = _unit(rng.normal(size=3))
    transverse = rng.normal(size=3)
    transverse -= float(transverse @ upper_direction) * upper_direction
    transverse = _unit(transverse)
    bend = rng.uniform(0.35, 2.35)
    forearm_direction = (
        np.cos(bend) * upper_direction + np.sin(bend) * transverse
    )
    robot_elbow = geometry.shoulder + geometry.upper_arm_length * upper_direction
    wrist = robot_elbow + geometry.forearm_length * forearm_direction
    human_upper_length = rng.uniform(0.24, 0.34)
    human_forearm_length = rng.uniform(0.38, 0.50)
    human_shoulder = (
        wrist
        - human_upper_length * upper_direction
        - human_forearm_length * forearm_direction
    )
    elbow = human_shoulder + human_upper_length * upper_direction
    hand_axis = _unit(rng.normal(size=3))
    hand = rot(hand_axis, rng.uniform(-np.pi, np.pi))
    task = wrist + hand @ geometry.wrist_to_task
    return (
        HumanArmTarget(human_shoulder, elbow, wrist, hand, task),
        task,
    )


def validate_generic(count: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    stereo = StereoSew(project_stereo_sew_reference())
    geometry = WarpArmGeometry(
        np.array([0.12, -0.08, 0.31]),
        0.41,
        0.32,
        np.array([0.09, -0.015, 0.025]),
    )
    palm_errors: list[float] = []
    upper_errors: list[float] = []
    forearm_errors: list[float] = []
    sew_errors: list[float] = []
    for index in range(count):
        human, target = _synthetic_human_target(rng, geometry)
        result = construct_warp_skeleton(human, target, geometry, stereo)
        if result.status is not WarpSkeletonStatus.SUCCESS_EXACT:
            raise RuntimeError(
                f"generic WARP case {index} failed: "
                f"{result.status.value}: {result.reason}"
            )
        assert result.palm_error_m is not None
        assert result.upper_length_error_m is not None
        assert result.forearm_length_error_m is not None
        assert result.sew_error_rad is not None
        palm_errors.append(result.palm_error_m)
        upper_errors.append(result.upper_length_error_m)
        forearm_errors.append(result.forearm_length_error_m)
        sew_errors.append(result.sew_error_rad)
    return {
        "cases": count,
        "exact_successes": len(palm_errors),
        "max_palm_error_m": max(palm_errors, default=float("nan")),
        "max_upper_length_error_m": max(upper_errors, default=float("nan")),
        "max_forearm_length_error_m": max(forearm_errors, default=float("nan")),
        "max_sew_error_rad": max(sew_errors, default=float("nan")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-cases", type=int, default=1000)
    parser.add_argument("--compatibility-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260912)
    arguments = parser.parse_args()
    if arguments.synthetic_cases < 1 or arguments.compatibility_samples < 1:
        raise SystemExit("sample counts must be positive")

    generic = validate_generic(arguments.synthetic_cases, arguments.seed)
    print("A. GENERIC WARP CORE")
    print(generic)

    robot = gen3_kinematics()
    report = check_warp_fixed_geometry_compatibility(
        robot,
        Gen3StereoSewGeometry.from_robot(robot),
        samples=arguments.compatibility_samples,
        seed=arguments.seed,
    )
    print("B. TESTED GEN3 S1/E45/W67 DEFINITION")
    print(
        {
            "samples": report.samples,
            "seed": report.seed,
            "L_SE": vars(report.upper_arm_length),
            "L_EW": vars(report.forearm_length),
            "p_WT_min": report.wrist_to_task.minimum.tolist(),
            "p_WT_max": report.wrist_to_task.maximum.tolist(),
            "p_WT_mean": report.wrist_to_task.mean.tolist(),
            "p_WT_std": report.wrist_to_task.std.tolist(),
            "p_WT_max_norm_deviation": report.wrist_to_task_max_norm_deviation,
            "tolerance_m": report.tolerance_m,
        }
    )
    print("Tested S1/E45/W67 fixed geometry:", "PASS" if report.compatible else "FAIL")
    if report.compatible:
        raise RuntimeError("S1/E45/W67 unexpectedly passed its fixed-link gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
