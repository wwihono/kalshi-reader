"""Tests for rolling pregame feature construction (no leakage)."""

import pandas as pd
import pytest

from kalshi_nba.features.rolling import (
    add_rest_features,
    build_pregame_features,
    games_to_team_rows,
    rolling_team_stats,
)


def _toy_schedule() -> pd.DataFrame:
    """Three teams, five games — enough for short rolling windows."""
    return pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                [
                    "2025-10-01",
                    "2025-10-03",
                    "2025-10-04",
                    "2025-10-06",
                    "2025-10-08",
                ]
            ),
            "home_team": ["A", "B", "A", "C", "A"],
            "visitor_team": ["B", "C", "C", "B", "B"],
            "home_pts": [100, 110, 95, 105, 120],
            "visitor_pts": [90, 100, 100, 100, 110],
        }
    )


def test_games_to_team_rows_expands_to_two_rows_per_game() -> None:
    rows = games_to_team_rows(_toy_schedule())
    assert len(rows) == 10
    assert set(rows["is_home"]) == {0, 1}


def test_rolling_stats_shift_excludes_current_game() -> None:
    stats = rolling_team_stats(_toy_schedule(), windows=(2,))
    team_a = stats[stats["team"] == "A"].sort_values("game_date")

    # First game for A has no prior history
    first = team_a.iloc[0]
    assert pd.isna(first["win_pct_l2"])
    assert pd.isna(first["avg_point_diff_l2"])

    # Second A game should reflect only the first result (home win by +10)
    second = team_a.iloc[1]
    assert second["win_pct_l2"] == pytest.approx(1.0)
    assert second["avg_point_diff_l2"] == pytest.approx(10.0)


def test_rest_days_and_back_to_back() -> None:
    rows = games_to_team_rows(_toy_schedule())
    with_rest = add_rest_features(rows)
    team_c = with_rest[with_rest["team"] == "C"].sort_values("game_date")

    # C plays on 10/03 then 10/04 -> back-to-back on second game
    assert pd.isna(team_c.iloc[0]["rest_days"])
    assert team_c.iloc[1]["rest_days"] == 1
    assert int(team_c.iloc[1]["is_back_to_back"]) == 1


def test_build_pregame_features_home_minus_visitor_and_target() -> None:
    features = build_pregame_features(_toy_schedule(), windows=(2,))
    assert "diff_win_pct_l2" in features.columns
    assert "diff_rest_days" in features.columns
    assert "home_win" in features.columns

    # Game on 10/08: A home vs B, A won
    last = features.sort_values("game_date").iloc[-1]
    assert last["home_team"] == "A"
    assert last["visitor_team"] == "B"
    assert int(last["home_win"]) == 1
