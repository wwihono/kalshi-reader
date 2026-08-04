"""Join Kalshi markets with Basketball Reference results for EDA."""

from __future__ import annotations

import pandas as pd

from kalshi_nba.clean.merge import prepare_results
from kalshi_nba.clean.team_names import to_canonical_team


def _team_pair_key(team_a: str, team_b: str) -> str:
    """Order-independent key for a two-team matchup."""
    return "|".join(sorted((team_a, team_b)))


def _normalize_br(results: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and prepare Basketball Reference rows."""
    out = results.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.drop_duplicates(subset=["game_date", "home_team", "visitor_team"], keep="first")
    return prepare_results(out)


def join_kalshi_results(
    kalshi_games: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join game-level Kalshi rows to BR results.

    Matching uses ``game_date`` plus the unordered team pair when Kalshi home /
    visitor sides are unknown (``vs`` titles). When Kalshi provides home/visitor
    (``at`` titles), those columns are used directly.
    """
    br = _normalize_br(results)
    br = br.copy()
    br["team_key"] = [
        _team_pair_key(h, v) for h, v in zip(br["home_team"], br["visitor_team"], strict=True)
    ]
    br_by_pair = {
        (row.game_date, row.team_key): (row.home_team, row.visitor_team)
        for row in br.itertuples(index=False)
    }

    kg = kalshi_games.copy()
    kg["game_date"] = pd.to_datetime(kg["game_date"], errors="coerce").dt.normalize()

    filled_home: list[object] = []
    filled_visitor: list[object] = []
    for row in kg.itertuples(index=False):
        home = getattr(row, "home_team", pd.NA)
        visitor = getattr(row, "visitor_team", pd.NA)
        if pd.isna(home) or pd.isna(visitor):
            t1, t2 = getattr(row, "team_1", pd.NA), getattr(row, "team_2", pd.NA)
            if pd.notna(t1) and pd.notna(t2):
                key = (row.game_date, _team_pair_key(str(t1), str(t2)))
                if key in br_by_pair:
                    home, visitor = br_by_pair[key]
        filled_home.append(home)
        filled_visitor.append(visitor)
    kg["home_team"] = filled_home
    kg["visitor_team"] = filled_visitor

    for col in ("home_team", "visitor_team", "team_1", "team_2"):
        if col in kg.columns:
            kg[col] = kg[col].map(lambda x: to_canonical_team(x) if pd.notna(x) else pd.NA)

    merged = kg.merge(
        br.drop(columns=["team_key"]),
        on=["game_date", "home_team", "visitor_team"],
        how="inner",
        suffixes=("_kalshi", "_br"),
    )
    return merged.sort_values("game_date").reset_index(drop=True)
