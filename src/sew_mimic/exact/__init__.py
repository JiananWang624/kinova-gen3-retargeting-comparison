"""Exact-SEW production APIs and the lazily loaded Method-3 oracle."""
from typing import TYPE_CHECKING, Any

from .cpp_solver import ExactSewConfig, ExactSewSolver, NativeStereoSewTarget, to_native_stereo_sew_target
from .residuals import ExactSewResiduals, robot_exact_sew_residuals, so3_log
from .solver import human_arm_to_exact_sew_target, retarget_exact_sew, solve_exact_sew

if TYPE_CHECKING:
    from .numerical_oracle import NumericalExactSewOracle, NumericalOracleConfig


def __getattr__(name: str) -> Any:
    if name in ("NumericalExactSewOracle", "NumericalOracleConfig"):
        from .numerical_oracle import NumericalExactSewOracle, NumericalOracleConfig
        globals().update(NumericalExactSewOracle=NumericalExactSewOracle,
                         NumericalOracleConfig=NumericalOracleConfig)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ExactSewConfig", "ExactSewSolver", "ExactSewResiduals", "NativeStereoSewTarget",
           "NumericalExactSewOracle", "NumericalOracleConfig", "human_arm_to_exact_sew_target",
           "retarget_exact_sew", "robot_exact_sew_residuals", "solve_exact_sew", "so3_log",
           "to_native_stereo_sew_target"]
