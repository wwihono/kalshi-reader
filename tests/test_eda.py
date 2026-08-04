"""Tests for EDA summary helpers and Kalshi–BR joining."""

import json
from pathlib import Path

import pandas as pd

from kalshi_nba.clean.kalshi_parse import (
    enrich_kalshi_markets,
    game_level_kalshi,
    markets_records_to_frame,
)
from kalshi_nba.eda.join_eda import join_kalshi_results
from kalshi_nba.eda.summarize import (
    categorical_summary,
    dataset_shape,
    has_any_missing,
    missingness_report,
    seven_number_summary,
)


ROOT = Path(__file__).resolve().parents[1]
KALSHI_FIX = ROOT / "data" / "fixtures" / "kalshi_markets_sample.json"
BR_FIX = ROOT / "data" / "fixtures" / "br_schedule_sample.csv"


def test_dataset_shape_and_missingness() -> None:
    frame = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
    assert dataset_shape(frame) == {"n_rows": 3, "n_columns": 2}
    assert has_any_missing(frame) is True
    report = missingness_report(frame)
    assert report.loc[report["column"] == "a", "n_missing"].iloc[0] == 1
    assert report.loc[report["column"] == "b", "n_missing"].iloc[0] == 0

    complete = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert has_any_missing(complete) is False
    assert bool(complete.isna().any().any()) is False


def test_seven_number_summary() -> None:
    stats = seven_number_summary(pd.Series([1, 2, 3, 4, 5]))
    assert stats["min"] == 1
    assert stats["max"] == 5
    assert stats["median"] == 3
    assert stats["count"] == 5


def test_categorical_summary() -> None:
    summary = categorical_summary(pd.Series(["a", "b", "a"]))
    assert summary.iloc[0]["value"] == "a"
    assert summary.iloc[0]["count"] == 2


def test_join_kalshi_results_fixture() -> None:
    records = json.loads(KALSHI_FIX.read_text(encoding="utf-8"))
    markets = enrich_kalshi_markets(markets_records_to_frame(records))
    games = game_level_kalshi(markets)
    results = pd.read_csv(BR_FIX, parse_dates=["game_date"])
    joined = join_kalshi_results(games, results)

    assert len(joined) == 2
    assert set(joined["home_team"]) == {"Los Angeles Lakers", "Phoenix Suns"}
    assert joined["home_win"].tolist() == [1, 1]
    assert joined["total_volume"].sum() == 4200.0
