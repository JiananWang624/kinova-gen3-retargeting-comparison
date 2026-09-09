"""Identify and independently validate a fixed virtual WARP skeleton for Gen3."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sew_mimic.kinematics import gen3_kinematics  # noqa: E402
from sew_mimic.sew import Gen3StereoSewGeometry  # noqa: E402
from sew_mimic.warp.identification import (  # noqa: E402
    EXACT_FIXED_SKELETON_TOLERANCE_M,
    PRACTICAL_APPROXIMATE_SKELETON_TOLERANCE_M,
    identify_fixed_skeleton,
)


def _scalar(stats) -> str:
    return f"min={stats.minimum:.6g}, max={stats.maximum:.6g}, mean={stats.mean:.6g}, std={stats.std:.3g}, variation={stats.variation:.3g} m"


def _residual(stats) -> str:
    return (f"mean={stats.mean_m:.6g} m ({1000*stats.mean_m:.4g} mm), "
            f"median={stats.median_m:.6g}, P95={stats.p95_m:.6g}, max={stats.maximum_m:.6g} m ({1000*stats.maximum_m:.4g} mm)")


def _vector(stats) -> str:
    return (f"min={stats.minimum.tolist()}, max={stats.maximum.tolist()}, "
            f"mean={stats.mean.tolist()}, std={stats.std.tolist()}, "
            f"max-deviation={stats.variation:.6g} m")


def _candidate(report) -> None:
    print(report.name)
    print("  S:", _vector(report.shoulder))
    print("  L_SE:", _scalar(report.upper_arm_length))
    print("  L_EW:", _scalar(report.forearm_length))
    print("  p_WT:", _vector(report.wrist_to_task))
    print("  reconstruction:", _residual(report.reconstruction))
    print("  link-vs-h3/h5 proxy angles (mean/max rad):", f"{report.upper_proxy_angle_rad.mean:.6g}/{report.upper_proxy_angle_rad.maximum:.6g}, {report.lower_proxy_angle_rad.mean:.6g}/{report.lower_proxy_angle_rad.maximum:.6g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--train-seed", type=int, default=20260914)
    parser.add_argument("--validation-seed", type=int, default=20260915)
    args = parser.parse_args()
    robot = gen3_kinematics(); geometry = Gen3StereoSewGeometry.from_robot(robot)
    report = identify_fixed_skeleton(robot, geometry, samples=args.samples,
                                     train_seed=args.train_seed, validation_seed=args.validation_seed)
    print("CURRENT EXACT-SEW S/E/W")
    _candidate(report.current_exact_sew)
    print("\nCANDIDATE S23/E45/W67")
    _candidate(report.s23_e45_w67)
    print("  S23 variation q1-only/q2-only/full:", f"{report.s23_q1_only_variation_m:.6g}, {report.s23_q2_only_variation_m:.6g}, {report.s23_full_variation_m:.6g} m")
    print("\nSEW-MIMIC PROXY FIXED-MODEL TEST")
    a = report.geometry_proxy
    print("  A geometry-derived (q=0, not fitted) S, L_SE, L_EW, p_WT:", a.parameters.shoulder.tolist(), a.parameters.upper_arm_length, a.parameters.forearm_length, a.parameters.wrist_to_task.tolist())
    print("  A train:", _residual(a.train))
    print("  A validation:", _residual(a.validation))
    p = report.proxy_fit.parameters
    print("  B global least-squares S, L_SE, L_EW, p_WT:", p.shoulder.tolist(), p.upper_arm_length, p.forearm_length, p.wrist_to_task.tolist())
    print("  B train:", _residual(report.proxy_fit.train))
    print("  B validation:", _residual(report.proxy_fit.validation))
    print("\nOTHER CANDIDATES")
    print("  candidate | S-var m | L_SE-var m | L_EW-var m | p_WT-var m | reconstruction max mm | h3/h5 proxy max rad")
    for candidate in report.other_candidates:
        print("  " + candidate.name + " | " + " | ".join((
            f"{candidate.shoulder.variation:.3g}", f"{candidate.upper_arm_length.variation:.3g}",
            f"{candidate.forearm_length.variation:.3g}", f"{candidate.wrist_to_task.variation:.3g}",
            f"{1000 * candidate.reconstruction.maximum_m:.3g}",
            f"{candidate.upper_proxy_angle_rad.maximum:.3g}/{candidate.lower_proxy_angle_rad.maximum:.3g}",
        )))
    print("  Note: reconstruction uses fixed means identified on the training set and each candidate's link directions;")
    print("        it is evaluated on the validation set and is not the h3/h5 proxy equation above.")
    print("\nFINAL VERDICT")
    print(report.verdict)
    print("  thresholds: exact max validation <=", EXACT_FIXED_SKELETON_TOLERANCE_M,
          "m; practical <=", PRACTICAL_APPROXIMATE_SKELETON_TOLERANCE_M, "m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
