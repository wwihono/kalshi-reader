"""Simple Elo-style team ratings updated chronologically."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(
    games: pd.DataFrame,
    *,
    k: float = 20.0,
    initial: float = 1500.0,
) -> pd.DataFrame:
    """Return pregame Elo for home and visitor on each game row.

    Ratings are updated after each game so pregame values never include the
    current contest.
    """
    required = {"game_date", "home_team", "visitor_team", "home_pts", "visitor_pts"}
    missing = required - set(games.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    ordered = games.sort_values("game_date").reset_index(drop=True)
    ratings: dict[str, float] = defaultdict(lambda: initial)

    home_elo: list[float] = []
    visitor_elo: list[float] = []

    for row in ordered.itertuples(index=False):
        home = row.home_team
        visitor = row.visitor_team
        home_elo.append(ratings[home])
        visitor_elo.append(ratings[visitor])

        home_score = 1.0 if row.home_pts > row.visitor_pts else 0.0
        visitor_score = 1.0 - home_score
        exp_home = expected_score(ratings[home], ratings[visitor])
        exp_visitor = 1.0 - exp_home

        ratings[home] = ratings[home] + k * (home_score - exp_home)
        ratings[visitor] = ratings[visitor] + k * (visitor_score - exp_visitor)

    out = ordered.copy()
    out["home_elo_pre"] = home_elo
    out["visitor_elo_pre"] = visitor_elo
    out["diff_elo"] = out["home_elo_pre"] - out["visitor_elo_pre"]
    return out
