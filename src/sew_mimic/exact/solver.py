"""Public Method-2 Exact-SEW adapters."""
from __future__ import annotations

from typing import Literal
import numpy as np
from numpy.typing import ArrayLike

from ..common import ExactSewTarget, HumanArmTarget, SolverDiagnostics, SolverResult, SolverStatus
from ..kinematics import Gen3Kinematics
from ..sew import Gen3StereoSewGeometry, StereoSew, StereoSewSingularityError
from .cpp_solver import ExactSewConfig, ExactSewSolver

BranchPolicy = Literal["canonical", "continuous"]


def human_arm_to_exact_sew_target(human_target: HumanArmTarget, stereo: StereoSew) -> ExactSewTarget:
    """Use the established task point, hand frame, and human Stereo-SEW angle."""
    if not isinstance(human_target, HumanArmTarget):
        raise ValueError("human_target must be a HumanArmTarget")
    return ExactSewTarget(human_target.task_point, human_target.hand_rotation,
                          stereo.forward(human_target.shoulder, human_target.elbow, human_target.wrist))


def _invalid(message: str) -> SolverResult:
    return SolverResult("exact_sew", SolverStatus.INVALID_INPUT, None,
                        SolverDiagnostics(metadata={"constraint_set": "pinch_pose_plus_stereo_sew"}), message)


def solve_exact_sew(target: ExactSewTarget, robot: Gen3Kinematics,
                    geometry: Gen3StereoSewGeometry, stereo: StereoSew, *,
                    branch_policy: BranchPolicy = "canonical", q_previous: ArrayLike | None = None,
                    config: ExactSewConfig = ExactSewConfig()) -> SolverResult:
    """Solve one target through the compiled global event core.

    ``continuous`` selects the global exact branch nearest ``q_previous``;
    a stateful :class:`ExactSewSolver` is preferred for trajectories.
    """
    if branch_policy not in ("canonical", "continuous"):
        return _invalid("branch_policy must be 'canonical' or 'continuous'")
    if not isinstance(config, ExactSewConfig):
        return _invalid("config must be ExactSewConfig")
    previous = None
    if q_previous is not None:
        previous = np.asarray(q_previous, dtype=float)
        if previous.shape != (7,) or not np.all(np.isfinite(previous)):
            return _invalid("q_previous must be finite with shape (7,)")
    solver = ExactSewSolver(robot, geometry, stereo, config=config)
    if branch_policy == "continuous" and previous is not None:
        solver.state.previous_q = previous.copy()
    result = solver.solve(target)
    result.diagnostics.metadata.update(branch_policy=branch_policy)
    return result


def retarget_exact_sew(human_target: HumanArmTarget, robot: Gen3Kinematics,
                       geometry: Gen3StereoSewGeometry, stereo: StereoSew, *,
                       branch_policy: BranchPolicy = "canonical", q_previous: ArrayLike | None = None,
                       config: ExactSewConfig = ExactSewConfig()) -> SolverResult:
    """Human-facing thin wrapper over :func:`solve_exact_sew`."""
    try:
        target = human_arm_to_exact_sew_target(human_target, stereo)
    except StereoSewSingularityError as error:
        return SolverResult("exact_sew", SolverStatus.SEW_SINGULAR, None, SolverDiagnostics(), str(error))
    except (TypeError, ValueError) as error:
        return _invalid(str(error))
    return solve_exact_sew(target, robot, geometry, stereo, branch_policy=branch_policy,
                           q_previous=q_previous, config=config)
