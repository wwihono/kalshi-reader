"""Exploratory data analysis helpers: summaries, cleaning, and merging.

These helpers answer the EDA questions in the Part 2 report: dataset
sizes, missing data, seven-number summaries for quantitative variables,
value counts for categorical variables, and construction of the
per-game event table that joins Kalshi markets with Basketball
Reference results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kalshi_nba.clean.markets import add_event_fields, markets_to_frame


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Count missing values per column (n_missing, pct_missing).

    ``pct_missing`` is a percentage of the total row count.
    """
    n_missing = frame.isna().sum()
    out = pd.DataFrame(
        {
            "n_missing": n_missing,
            "pct_missing": (n_missing / len(frame) * 100).round(2),
        }
    )
    out.index.name = "column"
    return out


def seven_number_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Seven-number summary (mean/std/min/q1/median/q3/max) per column."""
    rows = {}
    for col in columns:
        series = pd.to_numeric(frame[col], errors="raise")
        rows[col] = {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "q1": series.quantile(0.25),
            "median": series.median(),
            "q3": series.quantile(0.75),
            "max": series.max(),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "variable"
    return out


def categorical_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Unique values of a categorical column with counts (incl. missing)."""
    counts = frame[column].value_counts(dropna=False)
    out = counts.rename("count").to_frame()
    out.index.name = column
    return out.reset_index()


def load_raw_markets(raw_dir: Path) -> pd.DataFrame:
    """Load recent + historical market JSON into one tidy typed frame."""
    kalshi_dir = Path(raw_dir) / "kalshi"
    recent = json.loads((kalshi_dir / "markets_recent.json").read_text())
    historical = json.loads((kalshi_dir / "markets_historical.json").read_text())
    return add_event_fields(markets_to_frame(recent + historical))


def load_pregame_prices(raw_dir: Path) -> pd.DataFrame:
    """Load the pregame candle records saved by scripts/collect_candles.py."""
    path = Path(raw_dir) / "kalshi" / "pregame_prices.json"
    records = json.loads(path.read_text())
    frame = pd.DataFrame(records).drop_duplicates(subset="ticker")
    frame = frame.rename(
        columns={
            "yes_bid": "pregame_yes_bid",
            "yes_ask": "pregame_yes_ask",
            "price": "pregame_price",
            "midpoint": "kalshi_home_prob",
        }
    )
    return frame.reset_index(drop=True)


def load_schedule(raw_dir: Path) -> pd.DataFrame:
    """Load the cleaned Basketball Reference schedule CSV."""
    path = Path(raw_dir) / "basketball_reference" / "nba_2026_schedule.csv"
    return pd.read_csv(path, parse_dates=["game_date"])


def build_event_table(markets: pd.DataFrame, pregame: pd.DataFrame) -> pd.DataFrame:
    """One row per Kalshi event, from the home-team side of the market.

    Keeps the market whose Yes contract pays out when the home team
    wins, and attaches the final pregame implied probability.
    """
    home_side = markets[markets["yes_is_home"]].copy()
    events = home_side[
        [
            "event_ticker",
            "ticker",
            "title",
            "game_date",
            "visitor_team",
            "home_team",
            "status",
            "result",
            "volume_fp",
            "open_interest_fp",
            "settlement_value_dollars",
            "occurrence_datetime",
        ]
    ].copy()
    events = events.merge(
        pregame[
            [
                "ticker",
                "kalshi_home_prob",
                "pregame_yes_bid",
                "pregame_yes_ask",
                "cutoff_ts",
                "end_period_ts",
            ]
        ],
        on="ticker",
        how="left",
    )
    events["kalshi_home_win"] = (events["result"] == "yes").astype("Int64")
    return events.sort_values("game_date").reset_index(drop=True)


def merge_events_and_results(
    events: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join Kalshi events with Basketball Reference results.

    Joins on game date plus canonical home/visitor team names; rows on
    either side without a partner (preseason exhibitions, unmatched
    markets) are dropped.
    """
    results = schedule.copy()
    results["home_win"] = (results["home_pts"] > results["visitor_pts"]).astype(int)
    merged = events.merge(
        results,
        on=["game_date", "home_team", "visitor_team"],
        how="inner",
        validate="one_to_one",
    )
    merged["home_margin"] = merged["home_pts"] - merged["visitor_pts"]
    return merged.sort_values("game_date").reset_index(drop=True)
