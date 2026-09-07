"""Deterministic multi-start numerical oracle for Method 3 only."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from ..angles import wrap_to_pi
from ..common import (
    ExactSewTarget,
    SolverDiagnostics,
    SolverResult,
    SolverStatus,
    joint_limit_margin,
)
from ..kinematics import Gen3Kinematics, gen3_kinematics
from ..sew import (
    Gen3StereoSewGeometry,
    StereoSew,
    StereoSewSingularityError,
    project_stereo_sew_reference,
    sample_gen3_configurations,
)
from .residuals import ExactSewResiduals, robot_exact_sew_residuals
from .acceptance import (
    ORIENTATION_ACCEPTANCE_RAD,
    POSITION_ACCEPTANCE_M,
    SEW_ACCEPTANCE_RAD,
)

Vector = NDArray[np.float64]
_PENALTY = 1e6


@dataclass(frozen=True)
class NumericalOracleConfig:
    """Optimizer units and strict physical post-validation thresholds."""
    position_scale_m: float = 0.1
    orientation_scale_rad: float = 0.2
    sew_scale_rad: float = 0.2
    exact_position_m: float = POSITION_ACCEPTANCE_M
    exact_orientation_rad: float = ORIENTATION_ACCEPTANCE_RAD
    exact_sew_rad: float = SEW_ACCEPTANCE_RAD
    xtol: float = 1e-12
    ftol: float = 1e-12
    gtol: float = 1e-12
    max_nfev: int = 500


class NumericalExactSewOracle:
    """Validation oracle; a least-squares result is exact only post-checked."""

    method = "numerical_exact_sew_oracle"

    def __init__(
        self,
        robot: Gen3Kinematics | None = None,
        geometry: Gen3StereoSewGeometry | None = None,
        stereo: StereoSew | None = None,
        config: NumericalOracleConfig = NumericalOracleConfig(),
    ) -> None:
        self.robot = gen3_kinematics() if robot is None else robot
        self.geometry = (
            Gen3StereoSewGeometry.from_robot(self.robot)
            if geometry is None
            else geometry
        )
        self.stereo = (
            StereoSew(project_stereo_sew_reference()) if stereo is None else stereo
        )
        self.config = config
        positive_values = (
            config.position_scale_m,
            config.orientation_scale_rad,
            config.sew_scale_rad,
            config.exact_position_m,
            config.exact_orientation_rad,
            config.exact_sew_rad,
            config.xtol,
            config.ftol,
            config.gtol,
        )
        if (
            any(value <= 0.0 or not np.isfinite(value) for value in positive_values)
            or config.max_nfev < 1
        ):
            raise ValueError("oracle scales, tolerances, and max_nfev must be positive")
        self._lower, self._upper = self._validated_bounds()
        self.q_reference = np.zeros(7)
        self._fixed_seeds = self._build_fixed_seeds()

    def _validated_bounds(self) -> tuple[Vector, Vector]:
        bounds = np.asarray(self.robot.joint_limits, dtype=float)
        if bounds.shape != (7, 2) or np.any(np.isnan(bounds)):
            raise ValueError("robot joint_limits must have shape (7,2) without NaN")
        lower, upper = bounds[:, 0].copy(), bounds[:, 1].copy()
        if np.any(lower > upper):
            raise ValueError("robot joint limits must satisfy lower <= upper")
        return lower, upper

    def _build_fixed_seeds(self) -> tuple[tuple[str, Vector], ...]:
        midpoint = np.zeros(7)
        finite_pair = np.isfinite(self._lower) & np.isfinite(self._upper)
        midpoint[finite_pair] = 0.5 * (self._lower[finite_pair] + self._upper[finite_pair])
        nominal = np.array(
            [0.0, 0.26179939, np.pi, -2.26892803, 0.0, 0.95993109, np.pi / 2.0]
        )
        values: list[tuple[str, Vector]] = [
            ("zero", np.zeros(7)),
            ("midpoint", midpoint),
            ("nominal", nominal),
        ]
        for index, value in enumerate(sample_gen3_configurations(self.robot, 6, 20260908)):
            values.append((f"fixed_{index}", value))
        unique: list[tuple[str, Vector]] = []
        for name, value in values:
            duplicate = any(
                np.allclose(value, old, atol=0.0, rtol=0.0) for _, old in unique
            )
            if self._in_bounds(value) and not duplicate:
                unique.append((name, value.copy()))
        return tuple(unique)

    @property
    def deterministic_seeds(self) -> tuple[tuple[str, Vector], ...]:
        """Inspectable target-independent default starts, copied for callers."""
        return tuple((name, q.copy()) for name, q in self._fixed_seeds)

    def _in_bounds(self, q: ArrayLike) -> bool:
        value = np.asarray(q, dtype=float)
        return (
            value.shape == (7,)
            and np.all(np.isfinite(value))
            and bool(
                np.all(value >= self._lower) and np.all(value <= self._upper)
            )
        )

    def _canonicalize(self, q: Vector) -> Vector:
        result = q.copy()
        unlimited = ~np.isfinite(self._lower) & ~np.isfinite(self._upper)
        result[unlimited] = [wrap_to_pi(value) for value in result[unlimited]]
        return result

    def solve_pose(
        self,
        position: ArrayLike,
        rotation: ArrayLike,
        q_seed: ArrayLike | None = None,
    ) -> SolverResult:
        try:
            return self._solve(ExactSewTarget(position, rotation, 0.0), False, q_seed)
        except (TypeError, ValueError) as error:
            return self._failure(SolverStatus.INVALID_INPUT, str(error))

    def solve_pose_and_sew(
        self,
        target: ExactSewTarget | ArrayLike,
        rotation: ArrayLike | None = None,
        psi: float | None = None,
        q_seed: ArrayLike | None = None,
    ) -> SolverResult:
        try:
            task = (
                target
                if isinstance(target, ExactSewTarget)
                else ExactSewTarget(target, rotation, psi)  # type: ignore[arg-type]
            )
            return self._solve(task, True, q_seed)
        except (TypeError, ValueError) as error:
            return self._failure(SolverStatus.INVALID_INPUT, str(error))

    def _failure(
        self,
        status: SolverStatus,
        message: str,
        metadata: dict | None = None,
        solve_time_ms: float | None = None,
    ) -> SolverResult:
        diagnostics = SolverDiagnostics(
            solve_time_ms=solve_time_ms,
            metadata={} if metadata is None else metadata,
        )
        return SolverResult(self.method, status, None, diagnostics, message)

    def _seeds(self, q_seed: ArrayLike | None) -> tuple[tuple[str, Vector], ...]:
        values = list(self._fixed_seeds)
        if q_seed is not None:
            q = np.asarray(q_seed, dtype=float)
            if not self._in_bounds(q):
                raise ValueError(
                    "q_seed must be finite, shape (7,), and within mechanical limits"
                )
            if not any(np.array_equal(q, old) for _, old in values):
                values.append(("caller", q.copy()))
        return tuple(values)

    def _scaled(self, residual: ExactSewResiduals, include_sew: bool) -> Vector:
        pieces = [
            residual.position / self.config.position_scale_m,
            residual.rotation / self.config.orientation_scale_rad,
        ]
        if include_sew:
            assert residual.sew is not None
            pieces.append(np.array([residual.sew / self.config.sew_scale_rad]))
        return np.concatenate(pieces)

    def _exact(self, residual: ExactSewResiduals, include_sew: bool) -> bool:
        return (
            residual.position_error_m < self.config.exact_position_m
            and residual.orientation_error_rad < self.config.exact_orientation_rad
            and (
                not include_sew
                or (
                    residual.sew_error_rad is not None
                    and residual.sew_error_rad < self.config.exact_sew_rad
                )
            )
        )

    def _candidate_key(self, candidate: dict, include_sew: bool) -> tuple:
        r: ExactSewResiduals = candidate["residual"]
        norm = (
            r.position_error_m / self.config.exact_position_m
            + r.orientation_error_rad / self.config.exact_orientation_rad
        )
        if include_sew:
            norm += (r.sew_error_rad or 0.0) / self.config.exact_sew_rad
        wrapped_delta = np.array(
            [wrap_to_pi(a - b) for a, b in zip(candidate["q"], self.q_reference)]
        )
        distance = float(np.linalg.norm(wrapped_delta))
        return (
            0 if self._exact(r, include_sew) else 1,
            norm,
            -candidate["margin"],
            distance,
            tuple(candidate["q"].tolist()),
        )

    def _deduplicate_candidates(self, candidates: Iterable[dict]) -> list[dict]:
        """Collapse canonical periodic configurations independently of run order."""
        unique: list[dict] = []
        for candidate in sorted(candidates, key=lambda item: tuple(item["q"].tolist())):
            if not any(np.allclose(candidate["q"], old["q"], atol=1e-9, rtol=0.0) for old in unique):
                unique.append(candidate)
        return unique

    def _select_best(self, candidates: Iterable[dict], include_sew: bool) -> dict:
        """Choose the documented canonical candidate independent of scipy order."""
        return min(candidates, key=lambda item: self._candidate_key(item, include_sew))

    def _solve(
        self,
        target: ExactSewTarget,
        include_sew: bool,
        q_seed: ArrayLike | None,
    ) -> SolverResult:
        starts = self._seeds(q_seed)
        candidates: list[dict] = []
        run_metadata: list[dict] = []
        singular_count = 0
        started = time.perf_counter()
        for seed_id, seed in starts:
            singular_seen = False

            def objective(q: Vector) -> Vector:
                nonlocal singular_seen
                try:
                    physical = robot_exact_sew_residuals(
                        q,
                        target,
                        self.robot,
                        self.geometry,
                        self.stereo,
                        include_sew=include_sew,
                    )
                    return self._scaled(physical, include_sew)
                except StereoSewSingularityError:
                    singular_seen = True
                    return np.full(7 if include_sew else 6, _PENALTY)

            try:
                outcome = least_squares(
                    objective,
                    seed,
                    bounds=(self._lower, self._upper),
                    xtol=self.config.xtol,
                    ftol=self.config.ftol,
                    gtol=self.config.gtol,
                    max_nfev=self.config.max_nfev,
                    x_scale="jac",
                )
                q = self._canonicalize(np.asarray(outcome.x, dtype=float))
                record = {
                    "seed_id": seed_id,
                    "status": int(outcome.status),
                    "success": bool(outcome.success),
                    "message": str(outcome.message),
                    "nfev": int(outcome.nfev),
                    "cost": float(outcome.cost),
                    "singular_seen": singular_seen,
                }
                if self._in_bounds(q):
                    try:
                        residual = robot_exact_sew_residuals(
                            q,
                            target,
                            self.robot,
                            self.geometry,
                            self.stereo,
                            include_sew=include_sew,
                        )
                    except StereoSewSingularityError:
                        singular_count += 1
                        record["final_singular"] = True
                    else:
                        candidate = {
                            "q": q,
                            "seed_id": seed_id,
                            "result": outcome,
                            "residual": residual,
                            "margin": joint_limit_margin(q, self.robot),
                        }
                        candidates.append(candidate)
                        record.update(
                            {
                                "position_error_m": residual.position_error_m,
                                "orientation_error_rad": residual.orientation_error_rad,
                                "sew_error_rad": residual.sew_error_rad,
                                "q": q.tolist(),
                            }
                        )
                run_metadata.append(record)
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as error:
                run_metadata.append(
                    {
                        "seed_id": seed_id,
                        "exception": f"{type(error).__name__}: {error}",
                        "singular_seen": singular_seen,
                    }
                )
        elapsed = 1000.0 * (time.perf_counter() - started)
        # Equivalent periodic solutions are canonicalized first, then collapsed by q.
        unique = self._deduplicate_candidates(candidates)
        if not unique:
            status = (
                SolverStatus.SEW_SINGULAR
                if include_sew and singular_count == len(starts)
                else SolverStatus.NUMERICAL_FAILURE
            )
            return self._failure(
                status,
                "no finite post-validated optimizer candidate",
                {"runs": run_metadata, "n_starts": len(starts)},
                solve_time_ms=elapsed,
            )
        best = self._select_best(unique, include_sew)
        residual: ExactSewResiduals = best["residual"]
        status = (
            SolverStatus.SUCCESS_EXACT
            if self._exact(residual, include_sew)
            else SolverStatus.SUCCESS_APPROX
        )
        diagnostics = SolverDiagnostics(
            position_error_m=residual.position_error_m,
            orientation_error_rad=residual.orientation_error_rad,
            sew_error_rad=residual.sew_error_rad,
            joint_limit_margin_rad=best["margin"],
            solve_time_ms=elapsed,
            metadata={
                "runs": run_metadata,
                "n_starts": len(starts),
                "n_candidates": len(unique),
                "best_seed": best["seed_id"],
                "best_nfev": int(best["result"].nfev),
                "best_cost": float(best["result"].cost),
                "sew_singular_evaluations": singular_count,
            },
        )
        return SolverResult(
            self.method,
            status,
            best["q"],
            diagnostics,
            "post-validated numerical least-squares candidate",
        )
