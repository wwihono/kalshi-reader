"""Pregame feature construction and chronological splitting.

Every feature for a game is computed using only games completed strictly
before it, so no future information can leak into training or evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "game_date",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split games chronologically into train / validation / test sets.

    The earliest ``train_frac`` of games form the training set, the next
    ``val_frac`` the validation set, and the remainder the test set. Rows
    are sorted (stably) by ``date_col`` first so a shuffled input cannot
    leak future games into the training period.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Fractions must be in (0, 1) and sum to less than 1")
    ordered = df.sort_values(date_col, kind="stable").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    return train, val, test


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape one-row-per-game into one-row-per-team-per-game (long format)."""
    home = pd.DataFrame(
        {
            "game_id": games.index,
            "game_date": games["game_date"].values,
            "team": games["home_team"].values,
            "points_for": games["home_points"].values,
            "points_against": games["away_points"].values,
            "is_home": 1,
        }
    )
    away = pd.DataFrame(
        {
            "game_id": games.index,
            "game_date": games["game_date"].values,
            "team": games["away_team"].values,
            "points_for": games["away_points"].values,
            "points_against": games["home_points"].values,
            "is_home": 0,
        }
    )
    log = pd.concat([home, away], ignore_index=True)
    log["won"] = (log["points_for"] > log["points_against"]).astype(int)
    log["point_diff"] = log["points_for"] - log["points_against"]
    return log.sort_values(["team", "game_date"], kind="stable").reset_index(drop=True)


def rolling_team_features(games: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Per-team pregame rolling features from previously completed games only.

    For each team-game row, computes over each window ``w`` in ``windows``:

    - ``win_pct_last{w}``: win percentage over the previous ``w`` games
    - ``avg_point_diff_last{w}``: average point differential over the
      previous ``w`` games

    plus ``rest_days`` (days since the team's previous game) and
    ``back_to_back`` (played on the previous calendar day). The current
    game is excluded via ``shift(1)``, and early-season rows with no
    history hold NaN so they can be filtered explicitly.
    """
    log = _team_game_log(games)
    grouped = log.groupby("team", sort=False)
    for w in windows:
        shifted_won = grouped["won"].transform(lambda s: s.shift(1))
        shifted_diff = grouped["point_diff"].transform(lambda s: s.shift(1))
        log[f"win_pct_last{w}"] = (
            shifted_won.groupby(log["team"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        log[f"avg_point_diff_last{w}"] = (
            shifted_diff.groupby(log["team"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
    prev_date = grouped["game_date"].transform(lambda s: s.shift(1))
    rest = (pd.to_datetime(log["game_date"]) - pd.to_datetime(prev_date)).dt.days
    log["rest_days"] = rest
    log["back_to_back"] = (rest == 1).astype("float").where(rest.notna())
    return log


def compute_elo_ratings(
    games: pd.DataFrame,
    k: float = ELO_K,
    home_advantage: float = ELO_HOME_ADVANTAGE,
    initial: float = ELO_INITIAL,
) -> pd.DataFrame:
    """Sequential Elo ratings; returns pregame ratings for both teams.

    Games must be sorted chronologically. The returned frame carries, for
    each game, ``home_elo_pre`` and ``away_elo_pre`` — the ratings *before*
    the game is played — so they are valid pregame features.
    """
    ratings: dict[str, float] = {}
    home_pre, away_pre = [], []
    ordered = games.sort_values("game_date", kind="stable")
    for _, row in ordered.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh = ratings.get(h, initial)
        ra = ratings.get(a, initial)
        home_pre.append(rh)
        away_pre.append(ra)
        expected_home = 1.0 / (1.0 + 10.0 ** ((ra - (rh + home_advantage)) / 400.0))
        actual_home = 1.0 if row["home_points"] > row["away_points"] else 0.0
        delta = k * (actual_home - expected_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    result = ordered.copy()
    result["home_elo_pre"] = home_pre
    result["away_elo_pre"] = away_pre
    return result


def build_game_features(games: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    """Assemble the model matrix: home-minus-visitor feature differences.

    Input needs one row per game with ``game_date``, ``home_team``,
    ``away_team``, ``home_points``, ``away_points``. Output adds, for each
    rolling window, ``diff_win_pct_last{w}`` and
    ``diff_avg_point_diff_last{w}``, plus ``diff_elo``, ``diff_rest_days``,
    ``diff_back_to_back``, and the target ``home_win``. Rows without full
    pregame history (early-season games) are dropped.
    """
    games = games.sort_values("game_date", kind="stable").reset_index(drop=True)
    with_elo = compute_elo_ratings(games)
    log = rolling_team_features(games, windows=windows)

    feature_cols = [f"win_pct_last{w}" for w in windows]
    feature_cols += [f"avg_point_diff_last{w}" for w in windows]
    feature_cols += ["rest_days", "back_to_back"]

    home_log = log[log["is_home"] == 1].set_index("game_id")[feature_cols]
    away_log = log[log["is_home"] == 0].set_index("game_id")[feature_cols]

    out = with_elo.copy()
    for col in feature_cols:
        out[f"diff_{col}"] = home_log[col].reindex(out.index) - away_log[col].reindex(out.index)
    out["diff_elo"] = out["home_elo_pre"] - out["away_elo_pre"]
    out["home_win"] = (out["home_points"] > out["away_points"]).astype(int)

    diff_cols = [f"diff_{col}" for col in feature_cols]
    out = out.dropna(subset=diff_cols).reset_index(drop=True)
    return out


FEATURE_COLUMNS = [
    "diff_win_pct_last5",
    "diff_win_pct_last10",
    "diff_avg_point_diff_last5",
    "diff_avg_point_diff_last10",
    "diff_rest_days",
    "diff_back_to_back",
    "diff_elo",
]
TARGET_COLUMN = "home_win"
