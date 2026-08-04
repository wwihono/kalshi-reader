"""Tests for the EDA summary and merge helpers.

Uses small hand-built frames where every expected value can be checked
by hand, so the numbers quoted in the report can be trusted.
"""

import numpy as np
import pandas as pd
import pytest

from kalshi_nba.eda import (
    build_event_table,
    categorical_summary,
    merge_events_and_results,
    missingness_summary,
    seven_number_summary,
)


def test_missingness_summary_counts_and_percentages():
    frame = pd.DataFrame(
        {
            "full": [1, 2, 3, 4],
            "half": [1.0, None, 3.0, None],
            "empty": [None, None, None, None],
        }
    )
    out = missingness_summary(frame)
    assert out.loc["full", "n_missing"] == 0
    assert out.loc["half", "n_missing"] == 2
    assert out.loc["half", "pct_missing"] == pytest.approx(50.0)
    assert out.loc["empty", "pct_missing"] == pytest.approx(100.0)


def test_seven_number_summary_matches_hand_computation():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = seven_number_summary(frame, ["x"])
    row = out.loc["x"]
    assert row["mean"] == pytest.approx(3.0)
    assert row["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert row["min"] == 1.0
    assert row["q1"] == 2.0
    assert row["median"] == 3.0
    assert row["q3"] == 4.0
    assert row["max"] == 5.0


def test_categorical_summary_includes_missing_values():
    frame = pd.DataFrame({"result": ["yes", "no", "yes", None]})
    out = categorical_summary(frame, "result")
    counts = dict(zip(out["result"], out["count"], strict=True))
    assert counts["yes"] == 2
    assert counts["no"] == 1
    assert out["count"].sum() == 4  # NaN row counted too


def _events_fixture() -> pd.DataFrame:
    markets = pd.DataFrame(
        {
            "ticker": ["E1-HOU", "E1-PHX", "E2-BOS"],
            "event_ticker": ["E1", "E1", "E2"],
            "title": ["t1", "t1", "t2"],
            "game_date": pd.to_datetime(["2026-01-05"] * 2 + ["2026-01-06"]),
            "visitor_team": ["Phoenix Suns", "Phoenix Suns", "New York Knicks"],
            "home_team": ["Houston Rockets", "Houston Rockets", "Boston Celtics"],
            "status": ["finalized"] * 3,
            "result": ["yes", "no", "no"],
            "volume_fp": [100.0, 90.0, 50.0],
            "open_interest_fp": [0.0, 0.0, 0.0],
            "settlement_value_dollars": [1.0, 0.0, 0.0],
            "occurrence_datetime": [pd.NaT] * 3,
            "yes_is_home": [True, False, True],
        }
    )
    pregame = pd.DataFrame(
        {
            "ticker": ["E1-HOU", "E2-BOS"],
            "kalshi_home_prob": [0.725, 0.40],
            "pregame_yes_bid": [0.72, 0.39],
            "pregame_yes_ask": [0.73, 0.41],
            "cutoff_ts": [1, 2],
            "end_period_ts": [1, 2],
        }
    )
    return build_event_table(markets, pregame)


def test_build_event_table_keeps_home_side_only():
    events = _events_fixture()
    assert len(events) == 2
    assert set(events["ticker"]) == {"E1-HOU", "E2-BOS"}
    hou = events[events["ticker"] == "E1-HOU"].iloc[0]
    assert hou["kalshi_home_prob"] == pytest.approx(0.725)
    assert hou["kalshi_home_win"] == 1
    bos = events[events["ticker"] == "E2-BOS"].iloc[0]
    assert bos["kalshi_home_win"] == 0


def test_merge_events_and_results_inner_join_and_target():
    events = _events_fixture()
    schedule = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-01-05", "2026-01-07"]),
            "visitor_team": ["Phoenix Suns", "Utah Jazz"],
            "home_team": ["Houston Rockets", "Miami Heat"],
            "visitor_pts": [97, 100],
            "home_pts": [100, 90],
        }
    )
    merged = merge_events_and_results(events, schedule)
    # Only the Jan 5 game exists in both sources
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["home_win"] == 1
    assert row["home_margin"] == 3
    assert row["kalshi_home_win"] == row["home_win"]


def test_merge_events_and_results_rejects_duplicate_games():
    events = _events_fixture()
    schedule = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "visitor_team": ["Phoenix Suns"] * 2,
            "home_team": ["Houston Rockets"] * 2,
            "visitor_pts": [97, 97],
            "home_pts": [100, 100],
        }
    )
    with pytest.raises(pd.errors.MergeError):
        merge_events_and_results(events, schedule)
