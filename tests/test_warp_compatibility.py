from __future__ import annotations

import numpy as np

from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.sew import Gen3StereoSewGeometry
from sew_mimic.warp.compatibility import check_warp_fixed_geometry_compatibility


def test_s1_e45_w67_is_deterministically_not_fixed_link_geometry() -> None:
    robot = gen3_kinematics()
    report = check_warp_fixed_geometry_compatibility(robot, Gen3StereoSewGeometry.from_robot(robot))
    assert not report.compatible
    assert 0.354 < report.upper_arm_length.minimum < .355
    assert .549 < report.upper_arm_length.maximum < .551
    assert .195 < report.upper_arm_length.variation < .196
    assert report.forearm_length.variation < 1e-12
    assert report.wrist_to_task_max_norm_deviation < 1e-12


def test_analytic_q2_upper_arm_dependence_matches_validated_geometry() -> None:
    robot = gen3_kinematics()
    geometry = Gen3StereoSewGeometry.from_robot(robot)
    for q2 in (0.0, .5, -.5, 2.24, -2.24):
        q = np.zeros(7); q[1] = q2
        points = geometry.sew_points(q)
        actual_sq = np.linalg.norm(points.elbow - points.shoulder) ** 2
        expected_sq = .42076**2 + .02450**2 + .12838**2 + 2 * .42076 * .12838 * np.cos(q2)
        np.testing.assert_allclose(actual_sq, expected_sq, atol=2e-12)
