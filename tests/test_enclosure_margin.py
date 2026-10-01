"""CPU tests of the rule every enclosure check uses (certified_head.selftest.margin).

A bound must clear the FP64 reference logit by the margin to count as enclosing
it: the reference has its own rounding error, so a bound that only reaches it is
not confirmed. NaN is a miss. The row's lower bound on its largest logit follows
the same rule (it once used the margin as a tolerance instead).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip('torch')

from certified_head.selftest import (
    lower_misses,
    margin,
    row_lower_misses,
    upper_misses,
)

X = torch.tensor([[1.0, -2.0, 300.0], [0.0, 5.5, -7.25]], dtype=torch.float64)


def test_margin_is_relative_with_a_floor() -> None:
    assert torch.equal(margin(X), 1e-9 * (1 + X.abs()))
    assert bool((margin(X) > 0).all())


def test_a_bound_equal_to_the_reference_is_a_miss() -> None:
    assert bool(lower_misses(X, X).all())
    assert bool(upper_misses(X, X).all())


def test_bounds_two_margins_away() -> None:
    s = margin(X)
    assert not bool(lower_misses(X - 2 * s, X).any())
    assert bool(lower_misses(X + 2 * s, X).all())
    assert not bool(upper_misses(X + 2 * s, X).any())
    assert bool(upper_misses(X - 2 * s, X).all())


def test_nan_bounds_are_misses() -> None:
    nan = torch.full_like(X, float('nan'))
    assert bool(lower_misses(nan, X).all())
    assert bool(upper_misses(nan, X).all())
    assert bool(row_lower_misses(torch.full((2,), float('nan')), X).all())


def test_float32_bounds_are_compared_in_float64() -> None:
    s = margin(X)
    assert not bool(lower_misses((X - 2 * s).float() - 1e-3, X).any())
    assert bool(upper_misses(X.float().double() - 2 * s, X).all())


def test_row_lower_bound_uses_the_margin_not_a_tolerance() -> None:
    top = X.max(dim=1).values
    assert bool(row_lower_misses(top, X).all())
    assert not bool(row_lower_misses(top - 2 * margin(top), X).any())
    # Passed the old tolerance rule (lower <= max(x + margin)), fails this one.
    half = top + 0.5 * margin(top)
    assert bool((half <= (X + margin(X)).max(dim=1).values).all())
    assert bool(row_lower_misses(half, X).all())
