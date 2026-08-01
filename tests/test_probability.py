"""Tests for Kalshi midpoint probability calculation."""

import math

import pytest

from kalshi_nba.clean.probability import add_midpoint_probability, brier_score, midpoint_probability


def test_midpoint_probability_unit_interval() -> None:
    assert midpoint_probability(0.40, 0.50) == pytest.approx(0.45)


def test_midpoint_probability_cents_scale() -> None:
    assert midpoint_probability(40, 50) == pytest.approx(0.45)


def test_midpoint_probability_missing_sides() -> None:
    assert midpoint_probability(None, 0.5) is None
    assert midpoint_probability(0.5, None) is None


def test_midpoint_probability_ask_below_bid() -> None:
    with pytest.raises(ValueError, match="yes_ask"):
        midpoint_probability(0.6, 0.5)


def test_add_midpoint_probability_frame() -> None:
    import pandas as pd

    frame = pd.DataFrame({"yes_bid": [0.2, 55], "yes_ask": [0.3, 65]})
    out = add_midpoint_probability(frame)
    assert out["kalshi_prob"].tolist() == pytest.approx([0.25, 0.60])


def test_brier_score_perfect_and_wrong() -> None:
    assert brier_score([1, 0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([1, 0], [0, 1]) == pytest.approx(1.0)
    assert math.isclose(brier_score([1], [0.7]), (0.3**2))
