"""End-to-end analysis skeleton (features -> split -> metrics placeholders)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kalshi_nba.features.elo import compute_elo_ratings
from kalshi_nba.features.rolling import build_pregame_features
from kalshi_nba.models.split import chronological_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games",
        type=Path,
        required=True,
        help="CSV of cleaned games with scores (home/visitor teams + pts)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/pregame_features.csv"),
    )
    args = parser.parse_args()

    games = pd.read_csv(args.games, parse_dates=["game_date"])
    features = build_pregame_features(games)
    elo = compute_elo_ratings(games)[
        ["game_date", "home_team", "visitor_team", "diff_elo"]
    ]
    features = features.merge(elo, on=["game_date", "home_team", "visitor_team"], how="left")

    split = chronological_split(features)
    print(
        f"Rows={len(features)} | train={len(split.train)} "
        f"val={len(split.validation)} test={len(split.test)}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
