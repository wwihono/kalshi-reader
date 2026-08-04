"""Rolling pregame performance and rest features (no leakage)."""

from __future__ import annotations

import pandas as pd


def _require_columns(frame: pd.DataFrame, cols: set[str]) -> None:
    missing = cols - set(frame.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")


def games_to_team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Expand one row per game into two team-perspective rows."""
    _require_columns(
        games,
        {"game_date", "home_team", "visitor_team", "home_pts", "visitor_pts"},
    )
    base = games.copy()
    base["game_date"] = pd.to_datetime(base["game_date"])

    home = pd.DataFrame(
        {
            "game_date": base["game_date"],
            "team": base["home_team"],
            "opponent": base["visitor_team"],
            "is_home": 1,
            "points_for": base["home_pts"],
            "points_against": base["visitor_pts"],
        }
    )
    visitor = pd.DataFrame(
        {
            "game_date": base["game_date"],
            "team": base["visitor_team"],
            "opponent": base["home_team"],
            "is_home": 0,
            "points_for": base["visitor_pts"],
            "points_against": base["home_pts"],
        }
    )
    rows = pd.concat([home, visitor], ignore_index=True)
    rows["won"] = (rows["points_for"] > rows["points_against"]).astype(int)
    rows["point_diff"] = rows["points_for"] - rows["points_against"]
    return rows.sort_values(["team", "game_date"]).reset_index(drop=True)


def rolling_team_stats(games: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Compute shifted rolling win% and point differential for each team.

    Uses only previously completed games (shift(1) before rolling).
    """
    rows = games_to_team_rows(games)
    pieces: list[pd.DataFrame] = []

    for team, group in rows.groupby("team", sort=False):
        g = group.sort_values("game_date").copy()
        for window in windows:
            # shift(1) excludes the current game from the rolling window
            g[f"win_pct_l{window}"] = (
                g["won"].shift(1).rolling(window=window, min_periods=1).mean()
            )
            g[f"avg_point_diff_l{window}"] = (
                g["point_diff"].shift(1).rolling(window=window, min_periods=1).mean()
            )
        g["team"] = team
        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


def add_rest_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Add rest days and back-to-back indicators from prior game dates."""
    _require_columns(team_rows, {"team", "game_date"})
    out_parts: list[pd.DataFrame] = []
    for _, group in team_rows.groupby("team", sort=False):
        g = group.sort_values("game_date").copy()
        prev = g["game_date"].shift(1)
        delta = (g["game_date"] - prev).dt.days
        g["rest_days"] = delta
        g["is_back_to_back"] = (delta == 1).astype("Int64")
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


def build_pregame_features(games: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Build home-minus-visitor feature rows for modeling.

    Target ``home_win`` is 1 when the home team wins.
    """
    stats = rolling_team_stats(games, windows=windows)
    stats = add_rest_features(stats)

    feature_cols = []
    for window in windows:
        feature_cols.extend([f"win_pct_l{window}", f"avg_point_diff_l{window}"])
    feature_cols.extend(["rest_days", "is_back_to_back"])

    home = stats[stats["is_home"] == 1][
        ["game_date", "team", "opponent", *feature_cols]
    ].rename(columns={"team": "home_team", "opponent": "visitor_team"})
    visitor = stats[stats["is_home"] == 0][["game_date", "team", *feature_cols]].rename(
        columns={"team": "visitor_team"}
    )

    merged = home.merge(
        visitor,
        on=["game_date", "visitor_team"],
        how="inner",
        suffixes=("_home", "_visitor"),
    )

    out = merged[["game_date", "home_team", "visitor_team"]].copy()
    for col in feature_cols:
        out[f"diff_{col}"] = merged[f"{col}_home"] - merged[f"{col}_visitor"]
    out["home_indicator"] = 1  # all rows are home-perspective targets

    # Attach target from original scores
    score_lookup = games.copy()
    score_lookup["game_date"] = pd.to_datetime(score_lookup["game_date"])
    score_lookup["home_win"] = (
        score_lookup["home_pts"] > score_lookup["visitor_pts"]
    ).astype(int)
    out = out.merge(
        score_lookup[["game_date", "home_team", "visitor_team", "home_win"]],
        on=["game_date", "home_team", "visitor_team"],
        how="left",
    )
    return out.sort_values("game_date").reset_index(drop=True)
