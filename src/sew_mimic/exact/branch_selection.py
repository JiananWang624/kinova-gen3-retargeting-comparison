"""Deterministic policy selection over already-enumerated Exact-SEW branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..angles import angular_difference
from ..common import SolverStatus
from .stereo_backend import ExactSewCandidate, ExactSewCandidateSet
from .acceptance import (
    ORIENTATION_ACCEPTANCE_RAD,
    POSITION_ACCEPTANCE_M,
    SEW_ACCEPTANCE_RAD,
)


BranchPolicy = Literal["canonical", "continuous"]


@dataclass(frozen=True)
class BranchSelectionOutcome:
    """A selection result usable by both policy evaluation and the solver."""

    status: SolverStatus
    candidate: ExactSewCandidate | None
    candidate_index: int | None
    branch_id: str | None


def _validate_policy(policy: str) -> BranchPolicy:
    if policy not in ("canonical", "continuous"):
        raise ValueError("branch_policy must be 'canonical' or 'continuous'")
    return policy


def _previous_configuration(value: ArrayLike | None) -> np.ndarray | None:
    if value is None:
        return None
    q = np.asarray(value, dtype=float)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("q_previous must be finite with shape (7,)")
    return q.copy()


def normalized_authoritative_residual(candidate: ExactSewCandidate) -> float:
    """Squared residual normalized by the Method-2 acceptance thresholds."""
    return float(
        (candidate.position_error_m / POSITION_ACCEPTANCE_M) ** 2
        + (candidate.orientation_error_rad / ORIENTATION_ACCEPTANCE_RAD) ** 2
        + (candidate.sew_error_rad / SEW_ACCEPTANCE_RAD) ** 2
    )


def candidate_passes_authoritative_thresholds(candidate: ExactSewCandidate) -> bool:
    return bool(
        candidate.position_error_m < POSITION_ACCEPTANCE_M
        and candidate.orientation_error_rad < ORIENTATION_ACCEPTANCE_RAD
        and candidate.sew_error_rad < SEW_ACCEPTANCE_RAD
    )


def candidate_branch_id(
    candidates: ExactSewCandidateSet, candidate_index: int
) -> str:
    """Return the Phase-5A slot plus its deterministic wrist-root ordinal.

    The ordinal is counted in the original Phase-5A ordering among candidates
    with the same fixed lexical slot. It is therefore independent of Method-2
    filtering and does not encode the continuously changing wrist-root angle.
    """
    if not isinstance(candidates, ExactSewCandidateSet):
        raise ValueError("candidates must be an ExactSewCandidateSet")
    if not isinstance(candidate_index, int) or not 0 <= candidate_index < len(candidates.candidates):
        raise ValueError("candidate_index is outside the candidate set")
    candidate = candidates.candidates[candidate_index]
    root_ordinal = sum(
        previous.search_branch == candidate.search_branch
        for previous in candidates.candidates[:candidate_index]
    )
    return (
        f"r2r2r2r2r:slot={candidate.search_branch}:"
        f"wrist_root={root_ordinal}"
    )


def _canonical_key(item: tuple[int, ExactSewCandidate]) -> tuple[float, float, int]:
    index, candidate = item
    return (-candidate.joint_limit_margin_rad, normalized_authoritative_residual(candidate), index)


def _continuous_key(
    item: tuple[int, ExactSewCandidate], q_previous: np.ndarray
) -> tuple[float, float, float, int]:
    index, candidate = item
    distance = sum(
        angular_difference(float(current), float(previous)) ** 2
        for current, previous in zip(candidate.q, q_previous, strict=True)
    )
    return (
        distance,
        -candidate.joint_limit_margin_rad,
        normalized_authoritative_residual(candidate),
        index,
    )


def select_exact_sew_branch(
    candidates: ExactSewCandidateSet,
    *,
    branch_policy: BranchPolicy = "canonical",
    q_previous: ArrayLike | None = None,
) -> BranchSelectionOutcome:
    """Select one exact, joint-valid, physically postvalidated candidate.

    Candidate-array order is the deterministic Phase-5A ordering and is the
    final tie-breaker.  ``continuous`` falls back exactly to canonical when
    no previous configuration is given.
    """
    policy = _validate_policy(branch_policy)
    previous = _previous_configuration(q_previous)
    if not isinstance(candidates, ExactSewCandidateSet):
        raise ValueError("candidates must be an ExactSewCandidateSet")

    exact = [(index, candidate) for index, candidate in enumerate(candidates.candidates) if candidate.exact]
    if not exact:
        return BranchSelectionOutcome(SolverStatus.NO_VALID_BRANCH, None, None, None)
    joint_valid = [(index, candidate) for index, candidate in exact if candidate.joint_limit_valid]
    if not joint_valid:
        return BranchSelectionOutcome(SolverStatus.JOINT_LIMIT, None, None, None)
    selectable = [
        (index, candidate)
        for index, candidate in joint_valid
        if candidate_passes_authoritative_thresholds(candidate)
    ]
    if not selectable:
        return BranchSelectionOutcome(SolverStatus.NUMERICAL_FAILURE, None, None, None)

    if policy == "continuous" and previous is not None:
        index, candidate = min(selectable, key=lambda item: _continuous_key(item, previous))
    else:
        index, candidate = min(selectable, key=_canonical_key)
    return BranchSelectionOutcome(
        SolverStatus.SUCCESS_EXACT,
        candidate,
        index,
        candidate_branch_id(candidates, index),
    )
