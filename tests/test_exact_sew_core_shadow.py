"""Focused numerical tests for the private production C++ core."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sew_mimic.exact import _exact_sew_core as core
from sew_mimic.geometry import SP3_EXACT_TOL, rot, sp1, sp2, sp3


RNG = np.random.default_rng(20260909)


def _unit() -> np.ndarray:
    value = RNG.normal(size=3)
    return value / np.linalg.norm(value)


def _sp3_tuple(result: object) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[float, ...], bool]:
    return (tuple(result["angles"]), tuple(result["is_exact"]), tuple(result["residuals"]), bool(result["degenerate"]))  # type: ignore[index]


def test_rot_and_sp1_match_python_on_regular_geometry() -> None:
    for _ in range(100):
        axis = _unit() * RNG.uniform(0.1, 10.0)
        angle = RNG.uniform(-4.0 * math.pi, 4.0 * math.pi)
        vector = RNG.normal(size=3)
        np.testing.assert_allclose(core.rot(axis, angle), rot(axis, angle), atol=3e-14, rtol=3e-14)
        target = rot(axis, angle) @ vector
        assert core.sp1(vector, target, axis) == pytest.approx(sp1(vector, target, axis), abs=3e-14)


def test_sp2_matches_python_on_regular_geometry() -> None:
    for _ in range(100):
        while True:
            first, second = _unit(), _unit()
            if np.linalg.norm(np.cross(first, second)) > 0.2:
                break
        target = RNG.normal(size=3)
        q1, q2 = RNG.uniform(-math.pi, math.pi, size=2)
        p1, p2 = rot(first, -q1) @ target, rot(second, -q2) @ target
        np.testing.assert_allclose(core.sp2(p1, p2, first * 3.0, second * 7.0), sp2(p1, p2, first * 3.0, second * 7.0), atol=5e-13, rtol=5e-13)


@pytest.mark.parametrize(
    "arguments",
    [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0),
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0),
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0 + 5e-11),
        ([0.0, 0.0, 2.0], [1.0, 0.0, 3.0], [0.0, 0.0, 4.0], math.sqrt(2.0)),
        ([5e-5, 0.0, 0.0], [1e8, 0.0, 0.0], [0.0, 0.0, 1.0], 1e8),
        ([1.0, 0.0, 1e13], [1.0, 0.0, 1e13], [0.0, 0.0, 1.0], 1.0),
        ([1e308, 0.0, 0.0], [0.0, 1e308, 0.0], [0.0, 0.0, 1e308], 0.0),
    ],
)
def test_sp3_matches_python_for_exact_degenerate_and_inexact_cases(arguments: tuple[object, object, object, float]) -> None:
    expected = sp3(*arguments)
    actual = _sp3_tuple(core.sp3(*arguments, exact_tolerance=SP3_EXACT_TOL))
    assert actual[0] == pytest.approx(expected.angles, abs=3e-12)
    assert actual[1] == expected.is_exact
    assert actual[2] == pytest.approx(expected.residuals, abs=1e-9, rel=3e-13)
    assert actual[3] is expected.degenerate


def test_sp3_matches_python_on_random_exact_inputs() -> None:
    for _ in range(200):
        first, second, axis = RNG.normal(size=(3, 3))
        angle = RNG.uniform(-math.pi, math.pi)
        distance = float(np.linalg.norm(rot(axis, angle) @ first - second))
        expected = sp3(first, second, axis, distance)
        actual = _sp3_tuple(core.sp3(first, second, axis, distance))
        assert actual[0] == pytest.approx(expected.angles, abs=5e-12)
        assert actual[1] == expected.is_exact
        assert actual[2] == pytest.approx(expected.residuals, abs=2e-12)
        assert actual[3] is expected.degenerate


def test_sp3_fixed_regression_matches_python_root() -> None:
    arguments = (
        [455.59492433, -397.40013771, -1057.81920247],
        [0.00271005, 0.00144601, 0.00095721],
        [-0.05466334, 0.18777522, 0.10299625],
        1218.3892026246222,
    )
    expected = sp3(*arguments)
    actual = _sp3_tuple(core.sp3(*arguments))
    assert actual[0] == pytest.approx(expected.angles, abs=1e-8)
    assert actual[1] == expected.is_exact


def test_sp3_disparate_scale_regression_matches_python() -> None:
    arguments = (
        [-1138603.96947231, 236394.43401868, -1268478.70841927],
        [-4.09267739e-09, 6.92573600e-10, 4.71246868e-09],
        [-0.03101863, -0.00919972, 0.01321274],
        1720854.3115110449,
    )
    expected = sp3(*arguments)
    actual = _sp3_tuple(core.sp3(*arguments))
    assert actual[0] == pytest.approx(expected.angles, abs=1e-8)
    assert actual[1] == expected.is_exact


@pytest.mark.parametrize(("seed", "exponent_range"), [(20260910, (-3.0, 3.0)), (20260911, (-6.0, 6.0))])
def test_sp3_independent_cross_scale_random_differential_is_deterministic(
    seed: int, exponent_range: tuple[float, float]
) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(100):
        first, second, axis = rng.normal(size=(3, 3))
        first *= 10.0 ** rng.uniform(*exponent_range)
        second *= 10.0 ** rng.uniform(*exponent_range)
        axis *= 10.0 ** rng.uniform(*exponent_range)
        angle = rng.uniform(-math.pi, math.pi)
        distance = float(np.linalg.norm(rot(axis, angle) @ first - second))
        expected = sp3(first, second, axis, distance)
        actual = _sp3_tuple(core.sp3(first, second, axis, distance))
        assert actual[0] == pytest.approx(expected.angles, abs=1e-8)
        assert actual[1] == expected.is_exact


@pytest.mark.parametrize(
    "function,args",
    [
        (core.rot, ([0.0, 0.0, 0.0], 0.5)),
        (core.sp1, ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])),
        (core.sp2, ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0])),
        (core.sp3, ([1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0)),
    ],
)
def test_core_rejects_degenerate_or_invalid_inputs(function: object, args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        function(*args)  # type: ignore[operator]


def test_private_deterministic_brent_solver() -> None:
    root = core._test_brent_quadratic(1.0, 0.0, -2.0, 1.0, 2.0, 1e-14, 128)
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-13)


def test_private_brent_handles_tiny_signs_and_endpoint_roots() -> None:
    assert core._test_brent_quadratic(0.0, 1e-300, 0.0, -1.0, 1.0, 1e-14, 128) == 0.0
    assert core._test_brent_quadratic(0.0, 1.0, 0.0, 0.0, 1.0, 1e-14, 128) == 0.0


def test_private_brent_rejects_unbracketed_tiny_values_and_invalid_controls() -> None:
    with pytest.raises(ValueError, match="bracket"):
        core._test_brent_quadratic(0.0, 1e-300, 1e-300, 0.0, 1.0, 1e-14, 128)
    with pytest.raises(ValueError, match="tolerance"):
        core._test_brent_quadratic(1.0, 0.0, -2.0, 1.0, 2.0, 0.0, 128)
    with pytest.raises(ValueError, match="tolerance"):
        core._test_brent_quadratic(1.0, 0.0, -2.0, 1.0, 2.0, math.inf, 128)
    with pytest.raises(ValueError, match="maximum_iterations"):
        core._test_brent_quadratic(1.0, 0.0, -2.0, 1.0, 2.0, 1e-14, 0)


def test_event_root_normalization_sorts_and_deduplicates_by_smallest_residual() -> None:
    roots = core._test_normalize_native_roots(
        [
            {"angle": -0.1, "slot": 2, "residual": 0.4, "q": [2.0] * 7},
            {"angle": -0.2, "slot": 3, "residual": 0.2, "q": [3.0] * 7},
            {"angle": -0.1 + 5e-9, "slot": 2, "residual": 0.1, "q": [1.0] * 7},
            {"angle": -0.1, "slot": 1, "residual": 0.3, "q": [4.0] * 7},
        ]
    )
    assert [(float(root["angle"]), int(root["slot"])) for root in roots] == [
        (-0.2, 3),
        (-0.1, 1),
        (-0.1 + 5e-9, 2),
    ]
    assert float(roots[2]["residual"]) == pytest.approx(0.1)
    assert list(roots[2]["q"]) == [1.0] * 7


def test_final_sp1_gate_rejects_wrong_terminal_angles_before_root_acceptance() -> None:
    h = np.zeros((3, 7))
    h[:, 0] = [1.0, 0.0, 0.0]
    h[:, 1] = [0.0, 1.0, 0.0]
    h[:, 2] = [0.0, 0.0, 1.0]
    h[:, 3] = [1.0, 0.0, 0.0]
    h[:, 4] = [0.0, 1.0, 0.0]
    h[:, 5] = [1.0, 0.0, 0.0]
    h[:, 6] = [0.0, 1.0, 0.0]
    r07 = np.eye(3)
    assert core._test_final_sp1_valid([0.0] * 7, h, r07) is True
    invalid = [0.0] * 7
    invalid[5] = 0.1
    assert core._test_final_sp1_valid(invalid, h, r07) is False
