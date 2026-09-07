from types import MappingProxyType

import numpy as np
import pytest

from sew_mimic.common import SolverStatus
from sew_mimic.exact.branch_selection import (
    candidate_branch_id,
    select_exact_sew_branch,
)
from sew_mimic.exact.acceptance import (
    ORIENTATION_ACCEPTANCE_RAD,
    POSITION_ACCEPTANCE_M,
    SEW_ACCEPTANCE_RAD,
)
from sew_mimic.exact.stereo_backend import ExactSewCandidate, ExactSewCandidateSet


def _candidate(
    value: float,
    *,
    margin: float = 0.1,
    exact: bool = True,
    valid: bool = True,
    position: float = 0.0,
    orientation: float = 0.0,
    sew: float = 0.0,
    slot: int = 0,
    wrist: float = -0.2,
) -> ExactSewCandidate:
    return ExactSewCandidate(
        np.full(7, value), wrist, slot, position, orientation, sew,
        valid, margin, exact, {},
    )


def _set(*candidates: ExactSewCandidate) -> ExactSewCandidateSet:
    return ExactSewCandidateSet(tuple(candidates), len(candidates), len(candidates),
                                sum(c.joint_limit_valid for c in candidates),
                                MappingProxyType({}), 0.0)


def test_canonical_ranks_margin_then_normalized_residual_then_phase5a_order():
    lower_error = _candidate(1.0, margin=.2, position=1e-8)
    larger_margin = _candidate(2.0, margin=.3, position=9e-7)
    selected = select_exact_sew_branch(_set(lower_error, larger_margin))
    assert selected.status is SolverStatus.SUCCESS_EXACT
    assert np.array_equal(selected.candidate.q, larger_margin.q)

    first = _candidate(3.0, margin=.2, slot=2)
    second = _candidate(4.0, margin=.2, slot=1)
    assert select_exact_sew_branch(_set(first, second)).candidate_index == 0


def test_continuous_uses_wrapped_joint_distance_and_first_frame_is_canonical():
    canonical = _candidate(-3.0, margin=.4, slot=0)
    wrapped_near = _candidate(3.1, margin=.1, slot=1)
    candidates = _set(canonical, wrapped_near)
    assert select_exact_sew_branch(candidates, branch_policy="continuous").candidate_index == 0
    outcome = select_exact_sew_branch(
        candidates, branch_policy="continuous", q_previous=np.full(7, -3.18)
    )
    assert outcome.candidate_index == 1

    # The same configuration expressed with either full-turn sign remains
    # exactly zero wrapped distance from the previous frame.
    for turn in (-2.0 * np.pi, 2.0 * np.pi):
        full_turn = _candidate(turn, margin=.1, slot=2)
        assert select_exact_sew_branch(
            _set(full_turn, canonical),
            branch_policy="continuous",
            q_previous=np.zeros(7),
        ).candidate_index == 0


def test_selection_failure_mapping_and_invalid_previous():
    assert select_exact_sew_branch(_set(_candidate(0, exact=False))).status is SolverStatus.NO_VALID_BRANCH
    assert select_exact_sew_branch(_set(_candidate(0, valid=False))).status is SolverStatus.JOINT_LIMIT
    assert select_exact_sew_branch(_set(_candidate(0, position=POSITION_ACCEPTANCE_M))).status is SolverStatus.NUMERICAL_FAILURE
    with pytest.raises(ValueError, match="q_previous"):
        select_exact_sew_branch(_set(_candidate(0)), q_previous=np.zeros(6))
    with pytest.raises(ValueError, match="branch_policy"):
        select_exact_sew_branch(_set(_candidate(0)), branch_policy="other")


@pytest.mark.parametrize(
    ("field", "boundary"),
    (
        ("position", POSITION_ACCEPTANCE_M),
        ("orientation", ORIENTATION_ACCEPTANCE_RAD),
        ("sew", SEW_ACCEPTANCE_RAD),
    ),
)
def test_project_acceptance_thresholds_are_strict(field, boundary):
    kwargs = {field: boundary}
    assert select_exact_sew_branch(_set(_candidate(0, **kwargs))).status is SolverStatus.NUMERICAL_FAILURE
    kwargs[field] = boundary * (1.0 - 1e-12)
    assert select_exact_sew_branch(_set(_candidate(0, **kwargs))).status is SolverStatus.SUCCESS_EXACT


def test_invalid_residual_candidate_does_not_poison_valid_selectable_candidate():
    invalid = _candidate(0, margin=1.0, position=POSITION_ACCEPTANCE_M)
    valid = _candidate(1, margin=.1)
    outcome = select_exact_sew_branch(_set(invalid, valid))
    assert outcome.status is SolverStatus.SUCCESS_EXACT
    assert outcome.candidate_index == 1


def test_canonical_is_history_independent_and_repeated_selection_is_deterministic():
    candidates = _set(_candidate(.1, margin=.2, slot=3), _candidate(.2, margin=.1, slot=4))
    first = select_exact_sew_branch(candidates)
    second = select_exact_sew_branch(candidates)
    with_irrelevant_history = select_exact_sew_branch(candidates, q_previous=np.full(7, 2.5))
    continuous_first = select_exact_sew_branch(candidates, branch_policy="continuous")
    assert (first.candidate_index, first.branch_id) == (second.candidate_index, second.branch_id)
    assert with_irrelevant_history == first
    assert continuous_first == first


def test_short_trajectory_continuous_matches_independent_wrapped_distance_rule():
    trajectory = (
        _set(_candidate(-2.8, margin=.2), _candidate(.1, margin=.1)),
        _set(_candidate(-2.9, margin=.1), _candidate(.2, margin=.4)),
        _set(_candidate(3.05, margin=.2), _candidate(-.4, margin=.3)),
    )
    previous = None
    for candidates in trajectory:
        outcome = select_exact_sew_branch(candidates, branch_policy="continuous", q_previous=previous)
        if previous is None:
            expected = select_exact_sew_branch(candidates, branch_policy="canonical").candidate_index
        else:
            expected = min(
                range(len(candidates.candidates)),
                key=lambda index: (
                    sum((np.arctan2(np.sin(current - prior), np.cos(current - prior))) ** 2
                        for current, prior in zip(candidates.candidates[index].q, previous, strict=True)),
                    -candidates.candidates[index].joint_limit_margin_rad,
                    0.0,
                    index,
                ),
            )
        assert outcome.candidate_index == expected
        previous = outcome.candidate.q


def test_branch_identity_uses_slot_and_unfiltered_wrist_root_ordinal():
    inactive_for_selection = _candidate(0, slot=5, exact=False, wrist=-.4)
    first = _candidate(1, slot=6, wrist=-.3)
    second = _candidate(2, slot=6, wrist=-.2)
    candidates = _set(inactive_for_selection, first, second)
    assert candidate_branch_id(candidates, 1) == "r2r2r2r2r:slot=6:wrist_root=0"
    assert candidate_branch_id(candidates, 2) == "r2r2r2r2r:slot=6:wrist_root=1"
