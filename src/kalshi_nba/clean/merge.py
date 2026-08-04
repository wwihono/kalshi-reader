"""Merge Kalshi markets with Basketball Reference results."""

from __future__ import annotations

import pandas as pd

from .team_names import to_canonical_team


def _canonical_series(values: pd.Series) -> pd.Series:
    return values.map(lambda x: to_canonical_team(x) if pd.notna(x) else pd.NA)


def prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize result rows for joining."""
    required = {"game_date", "home_team", "visitor_team", "home_pts", "visitor_pts"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"results missing columns: {sorted(missing)}")

    out = results.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.normalize()
    out["home_team"] = _canonical_series(out["home_team"])
    out["visitor_team"] = _canonical_series(out["visitor_team"])
    out = out.dropna(subset=["game_date", "home_team", "visitor_team", "home_pts", "visitor_pts"])
    out["home_win"] = (out["home_pts"] > out["visitor_pts"]).astype(int)
    return out


def prepare_kalshi_markets(markets: pd.DataFrame) -> pd.DataFrame:
    """Normalize Kalshi market rows for joining.

    Expects columns that identify the game sides and schedule time. Flexible
    aliases are accepted for team and datetime fields.
    """
    out = markets.copy()

    date_candidates = (
        "game_date",
        "occurrence_datetime",
        "close_time",
        "expected_expiration_time",
    )
    date_col = next((c for c in date_candidates if c in out.columns), None)
    if date_col is None:
        raise KeyError("markets missing a datetime column")

    out["game_date"] = pd.to_datetime(out[date_col], utc=True, errors="coerce").dt.tz_convert(None)
    out["game_date"] = out["game_date"].dt.normalize()

    # Prefer explicit home/visitor if present; otherwise parse from subtitles later.
    if "home_team" in out.columns:
        out["home_team"] = _canonical_series(out["home_team"])
    if "visitor_team" in out.columns:
        out["visitor_team"] = _canonical_series(out["visitor_team"])

    return out


def merge_kalshi_and_results(
    markets: pd.DataFrame,
    results: pd.DataFrame,
    *,
    how: str = "inner",
) -> pd.DataFrame:
    """Join markets to results on game_date + home_team + visitor_team."""
    m = prepare_kalshi_markets(markets)
    r = prepare_results(results)

    join_cols = ["game_date", "home_team", "visitor_team"]
    for col in join_cols:
        if col not in m.columns:
            raise KeyError(f"prepared markets missing join column: {col}")

    merged = m.merge(r, on=join_cols, how=how, suffixes=("_kalshi", "_br"))
    return merged.reset_index(drop=True)
