"""Explicit fixed TIAGo J1 calibration checks; never select SEW points at runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from ..config import CONFIG, project_path
from ..csv_adapter import load_human_trajectory_csv
from .model import TiagoKinematics


def calibration_report(csv_path: str | Path) -> dict[str, object]:
    settings = CONFIG["tiago"]
    robot = TiagoKinematics(validate_config=False)
    human = load_human_trajectory_csv(csv_path)
    shoulder_body = np.median(human.shoulders, axis=0)
    q = np.array(settings["home_q_rad"], dtype=float)
    transform = settings["calibration"]
    reference_joint = transform["reference_joint_name"]
    if transform["reference_pose"] != "home_q_rad" or reference_joint not in robot.joint_names:
        raise ValueError("TIAGo calibration must use a named arm joint at home_q_rad")
    shoulder_base = robot.axes_and_anchors(q)[1][robot.joint_names.index(reference_joint)]
    rotation = np.asarray(transform["R_base_from_body"], dtype=float)
    offset = np.asarray(transform["user_xyz_offset_base_m"], dtype=float)
    translation = shoulder_base - rotation @ shoulder_body + offset
    measured = np.asarray(transform["t_base_from_body_m"], dtype=float)
    selected = settings["sew"]["selected"]
    if (reference_joint != "arm_1_joint" or
        selected != ["J1", "J4", "wrist_center"]):
        raise ValueError("TIAGo calibration requires locked J1 / J4 / wrist_center geometry")
    return {"frames": len(human), "reference_joint_name": reference_joint,
            "shoulder_reference_body_m": shoulder_body.tolist(),
            "shoulder_reference_base_m": shoulder_base.tolist(),
            "t_base_from_body_m": translation.tolist(),
            "translation_drift_m": float(np.linalg.norm(translation - measured)),
            "model_sha256": hashlib.sha256(project_path(settings["fingerprint_path"]).read_bytes()).hexdigest(),
            "model_fingerprint_matches_config": settings["model_sha256"] == hashlib.sha256(project_path(settings["fingerprint_path"]).read_bytes()).hexdigest(),
            "sew_definition": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check fixed TIAGo calibration and SEW geometry")
    parser.add_argument("--input", default=CONFIG["human_csv"]["input_path"])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = calibration_report(project_path(args.input))
    print(json.dumps(report, indent=2))
    if args.write:
        configuration = yaml.safe_load(Path(project_path("config.yaml")).read_text(encoding="utf-8"))
        settings = configuration["tiago"]
        model = TiagoKinematics(validate_config=False)
        settings["model_sha256"] = report["model_sha256"]
        settings["joint_limits_rad"] = model.joint_limits.tolist()
        calibration = settings["calibration"]
        calibration["shoulder_reference_body_m"] = report["shoulder_reference_body_m"]
        calibration["shoulder_reference_base_m"] = report["shoulder_reference_base_m"]
        rotation = np.asarray(calibration["R_base_from_body"], dtype=float)
        base = np.asarray(report["shoulder_reference_base_m"])
        body = np.asarray(report["shoulder_reference_body_m"])
        calibrated = base - rotation @ body
        calibration["t_calibrated_m"] = calibrated.tolist()
        calibration["t_base_from_body_m"] = (calibrated + np.asarray(calibration["user_xyz_offset_base_m"])).tolist()
        Path(project_path("config.yaml")).write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
