"""TIAGo fixed-skeleton WARP diagnostic, separate from production SEW ranking."""

from __future__ import annotations

import numpy as np

from ..config import CONFIG
from .model import TiagoKinematics


def diagnose_warp_invariance(robot: TiagoKinematics, *, samples: int = 1000,
                             seed: int = 20261001) -> dict[str, object]:
    settings = CONFIG["tiago"]["warp"]
    rng = np.random.default_rng(seed)
    configurations = rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1], (samples, 7))
    output = []
    for label in CONFIG["tiago"]["sew"]["candidate_order"]:
        definition = tuple(label.split("/"))
        shoulders, upper, lower, tool_offsets = [], [], [], []
        for q in configurations:
            points = robot.sew_points(q, definition)
            tcp, rotation = robot.tcp_pose(q)
            shoulders.append(points.shoulder)
            upper.append(np.linalg.norm(points.elbow - points.shoulder))
            lower.append(np.linalg.norm(points.wrist - points.elbow))
            tool_offsets.append(rotation.T @ (tcp - points.wrist))
        shoulders = np.asarray(shoulders)
        upper, lower, tool_offsets = map(np.asarray, (upper, lower, tool_offsets))
        shoulder_variation = float(np.max(np.linalg.norm(shoulders - shoulders[0], axis=1)))
        upper_variation = float(np.ptp(upper))
        lower_variation = float(np.ptp(lower))
        tool_variation = float(np.max(np.linalg.norm(tool_offsets - tool_offsets[0], axis=1)))
        tolerance = float(settings["fixed_skeleton_tolerance_m"])
        output.append({"candidate": label, "selected_for_production": label == "/".join(CONFIG["tiago"]["sew"]["selected"]),
                       "shoulder_variation_m": shoulder_variation,
                       "upper_length_variation_m": upper_variation,
                       "lower_length_variation_m": lower_variation,
                       "wrist_to_tcp_variation_m": tool_variation,
                       "executable": max(shoulder_variation, upper_variation,
                                         lower_variation, tool_variation) < tolerance})
    return {"samples": samples, "seed": seed,
            "tolerance_m": settings["fixed_skeleton_tolerance_m"],
            "candidate_diagnostics": output,
            "tiago_warp_executable": any(row["executable"] for row in output)}
