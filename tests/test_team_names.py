"""Tests for team-name conversion between Kalshi and Basketball Reference."""

import pytest

from kalshi_nba.clean.team_names import normalize_team_name, to_canonical_team


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Los Angeles Lakers", "los angeles lakers"),
        ("  LA-Clippers ", "la clippers"),
        ("Philadelphia 76ers", "philadelphia 76ers"),
        ("Portland Trail Blazers", "portland trail blazers"),
    ],
)
def test_normalize_team_name(raw: str, expected: str) -> None:
    assert normalize_team_name(raw) == expected


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("LAL", "Los Angeles Lakers"),
        ("la lakers", "Los Angeles Lakers"),
        ("Los Angeles Clippers", "Los Angeles Clippers"),
        ("LAC", "Los Angeles Clippers"),
        ("Golden State Warriors", "Golden State Warriors"),
        ("GSW", "Golden State Warriors"),
        ("OKC", "Oklahoma City Thunder"),
        ("76ers", "Philadelphia 76ers"),
        ("sixers", "Philadelphia 76ers"),
        ("Trail Blazers", "Portland Trail Blazers"),
        ("NYK", "New York Knicks"),
        ("Boston Celtics", "Boston Celtics"),
    ],
)
def test_to_canonical_team(raw: str, canonical: str) -> None:
    assert to_canonical_team(raw) == canonical


def test_to_canonical_team_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unrecognized team name"):
        to_canonical_team("Springfield Isotopes")


def test_normalize_rejects_none() -> None:
    with pytest.raises(ValueError):
        normalize_team_name(None)  # type: ignore[arg-type]
