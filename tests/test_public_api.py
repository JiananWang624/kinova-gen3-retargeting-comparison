"""Release-level import smoke tests for the documented public APIs."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_documented_public_apis_import() -> None:
    from sew_mimic.common import (  # noqa: F401
        ExactSewTarget,
        HumanArmTarget,
        SolverResult,
        SolverStatus,
    )
    from sew_mimic.exact import (  # noqa: F401
            NumericalExactSewOracle,
            ExactSewSolver,
        retarget_exact_sew,
        solve_exact_sew,
    )
    from sew_mimic.geometry import SP3Result, sp3  # noqa: F401
    from sew_mimic.pipeline import (  # noqa: F401
        capability_metadata,
        prepare_trajectory,
        run_benchmark,
    )
    from sew_mimic.sew import (  # noqa: F401
        Gen3StereoSewGeometry,
        StereoSew,
        StereoSewReference,
        solve_legacy_sew_mimic,
    )
    from sew_mimic.visualization import (  # noqa: F401
        build_overlay,
        prepare_replay_sequence,
        replay_in_mujoco,
    )
    from sew_mimic.warp import (  # noqa: F401
        check_warp_fixed_geometry_compatibility,
        compute_adaptive_offset,
        construct_warp_skeleton,
    )


def test_method2_import_does_not_load_numerical_oracle() -> None:
    code = """
import sys
sys.path.insert(0, 'src')
from sew_mimic.exact import ExactSewSolver, solve_exact_sew
from sew_mimic.kinematics import gen3_kinematics
from sew_mimic.pipeline import PreparedTrajectory, run_benchmark
from sew_mimic.sew import Gen3StereoSewGeometry, StereoSew, project_stereo_sew_reference
robot = gen3_kinematics()
prepared = PreparedTrajectory(
    robot,
    Gen3StereoSewGeometry.from_robot(robot),
    StereoSew(project_stereo_sew_reference()),
    (),
)
run_benchmark(prepared, methods=('exact_sew',))
assert 'sew_mimic.exact.numerical_oracle' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
