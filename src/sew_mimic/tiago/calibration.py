"""Read-only TIAGo calibration checks and deterministic SEW geometry search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from ..config import CONFIG, project_path
from ..csv_adapter import load_human_trajectory_csv
from ..sew.stereo import StereoSew, StereoSewReference
from .model import TiagoKinematics


def _directions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shoulder, elbow, wrist = points[:, 0], points[:, 1], points[:, 2]
    sw, se = wrist - shoulder, elbow - shoulder
    sw /= np.linalg.norm(sw, axis=1)[:, None]
    se /= np.linalg.norm(se, axis=1)[:, None]
    return sw, np.linalg.norm(np.cross(sw, se), axis=1)


def _reference_candidates() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    axes = np.eye(3)
    return tuple((sign * axes[i], axes[j]) for sign in (-1, 1) for i in range(3) for j in range(3) if i != j)


def identify_sew(robot: TiagoKinematics, human_points: np.ndarray, *,
                 samples: int = 300, seed: int = 20260927) -> list[dict[str, object]]:
    """Rank physical point definitions without consulting either IK backend."""
    rng = np.random.default_rng(seed)
    q_samples = rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1], (samples, 7))
    human_sw, human_elbow_margin = _directions(np.asarray(human_points, dtype=float).copy())
    output = []
    order = CONFIG["tiago"]["sew"]["candidate_order"]
    for ordinal, label in enumerate(order):
        selection = tuple(label.split("/"))
        try:
            point_sets = [robot.sew_points(q, selection) for q in q_samples]
            points = np.array([[p.shoulder, p.elbow, p.wrist] for p in point_sets])
            robot_sw, robot_elbow_margin = _directions(points)
            if not np.all(np.isfinite(robot_sw)) or not np.all(np.isfinite(robot_elbow_margin)):
                raise ValueError("non-finite point definition")
            best = None
            for e_t, e_r in _reference_candidates():
                robot_pole = np.linalg.norm(np.cross(robot_sw - e_t, e_r), axis=1) / 2
                human_pole = np.linalg.norm(np.cross(human_sw - e_t, e_r), axis=1) / 2
                margin = np.r_[np.minimum(robot_pole, robot_elbow_margin),
                               np.minimum(human_pole, human_elbow_margin)]
                score = float(np.percentile(margin, 1))
                record = (score, float(np.percentile(margin, 5)),
                          e_t.tolist(), e_r.tolist())
                if best is None or record[:2] > best[:2]:
                    best = record
            assert best is not None
            stereo = StereoSew(StereoSewReference(np.array(best[2]), np.array(best[3])))
            path_q = q_samples[:40]
            path_psi = [stereo.forward(p[0], p[1], p[2]) for p in points[:40]]
            # Local perturbations test continuity without interpreting random
            # independent configurations as a physical trajectory.
            deltas = np.minimum(1e-4, robot.joint_limits[:, 1] - path_q)
            moved = np.array([robot.sew_points(q + delta, selection)
                              for q, delta in zip(path_q, deltas)])
            moved_points = np.array([[p.shoulder, p.elbow, p.wrist] for p in moved])
            displacement = np.linalg.norm(moved_points - points[:40], axis=2)
            moved_psi = [stereo.forward(p[0], p[1], p[2]) for p in moved_points]
            psi_change = np.abs(np.angle(np.exp(1j * (np.array(moved_psi) - path_psi))))
            continuity = float(np.percentile(displacement, 95))
            psi_continuity = float(np.percentile(psi_change, 95))
            if not np.isfinite(continuity + psi_continuity) or continuity > 0.01 or psi_continuity > 0.1:
                raise ValueError("SEW points or psi discontinuous under small joint perturbations")
            conditions = []
            for q in path_q[:20]:
                p0, r0 = robot.tcp_pose(q)
                psi0 = stereo.forward(*(vars(robot.sew_points(q, selection)).values()))
                jacobian = np.empty((7, 7))
                for column in range(7):
                    perturbed = q.copy()
                    perturbed[column] += 1e-6
                    p1, r1 = robot.tcp_pose(perturbed)
                    pset = robot.sew_points(perturbed, selection)
                    psi1 = stereo.forward(pset.shoulder, pset.elbow, pset.wrist)
                    from scipy.spatial.transform import Rotation
                    jacobian[:, column] = np.r_[(p1 - p0) / 1e-6,
                                                  0.2 * Rotation.from_matrix(r1 @ r0.T).as_rotvec() / 1e-6,
                                                  0.2 * np.angle(np.exp(1j * (psi1 - psi0))) / 1e-6]
                conditions.append(float(np.linalg.cond(jacobian)))
            output.append({"candidate": label, "order": ordinal, "valid": True,
                           "margin_p1": best[0], "margin_p5": best[1],
                           "e_t": best[2], "e_r": best[3],
                           "point_continuity_p95_m": continuity,
                           "psi_continuity_p95_rad": psi_continuity,
                           "jacobian_condition_p95": float(np.percentile(conditions, 95))})
        except (ValueError, FloatingPointError) as error:
            output.append({"candidate": label, "order": ordinal, "valid": False,
                           "reason": str(error)})
    output.sort(key=lambda item: (not item["valid"], -item.get("margin_p1", -1),
                                  -item.get("margin_p5", -1),
                                  item.get("jacobian_condition_p95", float("inf")),
                                  item.get("psi_continuity_p95_rad", float("inf")),
                                  item["order"]))
    return output


def calibration_report(csv_path: str | Path) -> dict[str, object]:
    settings = CONFIG["tiago"]
    robot = TiagoKinematics(validate_config=False)
    human = load_human_trajectory_csv(csv_path)
    shoulder_body = np.median(human.shoulders, axis=0)
    q = np.array(settings["home_q_rad"], dtype=float)
    shoulder_base = robot.axes_and_anchors(q)[1][0]
    transform = settings["calibration"]
    rotation = np.asarray(transform["R_base_from_body"], dtype=float)
    offset = np.asarray(transform["user_xyz_offset_base_m"], dtype=float)
    translation = shoulder_base - rotation @ shoulder_body + offset
    measured = np.asarray(transform["t_base_from_body_m"], dtype=float)
    body_points = np.stack((human.shoulders, human.elbows, human.wrists), axis=1)
    base_points = np.einsum("ij,tkj->tki", rotation, body_points) + translation
    selection = identify_sew(robot, base_points)
    validation = identify_sew(robot, base_points, seed=20260928)
    return {"frames": len(human), "shoulder_reference_body_m": shoulder_body.tolist(),
            "shoulder_reference_base_m": shoulder_base.tolist(),
            "t_base_from_body_m": translation.tolist(),
            "translation_drift_m": float(np.linalg.norm(translation - measured)),
            "model_sha256": hashlib.sha256(project_path(settings["fingerprint_path"]).read_bytes()).hexdigest(),
            "model_fingerprint_matches_config": settings["model_sha256"] == hashlib.sha256(project_path(settings["fingerprint_path"]).read_bytes()).hexdigest(),
            "sew_candidates": selection,
            "validation_sew_candidates": validation}


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
        chosen = report["sew_candidates"][0]
        if not chosen["valid"]:
            raise ValueError("no valid TIAGo SEW point definition")
        if chosen["candidate"] != report["validation_sew_candidates"][0]["candidate"]:
            raise ValueError("TIAGo SEW selection differs between calibration and validation seeds")
        settings["sew"]["selected"] = chosen["candidate"].split("/")
        settings["sew"]["e_t"] = chosen["e_t"]
        settings["sew"]["e_r"] = chosen["e_r"]
        settings["sew"]["identification"] = {
            "calibration_seed": 20260927, "validation_seed": 20260928,
            "samples_per_split": 300,
            "calibration_candidates": report["sew_candidates"],
            "validation_candidates": report["validation_sew_candidates"],
        }
        Path(project_path("config.yaml")).write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
