"""Assemble structured EDA findings used by the CLI and PDF report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from kalshi_nba.clean.kalshi_parse import (
    enrich_kalshi_markets,
    game_level_kalshi,
    markets_records_to_frame,
)
from kalshi_nba.eda.join_eda import join_kalshi_results
from kalshi_nba.eda.plots import (
    plot_home_win_rate_by_month,
    plot_joined_home_win_by_volume_quartile,
    plot_joined_volume_vs_margin,
    plot_volume_distribution,
)
from kalshi_nba.eda.summarize import (
    categorical_summary,
    dataset_shape,
    has_any_missing,
    missingness_report,
    seven_number_summary,
)


def load_raw_kalshi(raw_dir: Path) -> pd.DataFrame:
    """Load and enrich recent + historical Kalshi market JSON files."""
    records: list[dict[str, Any]] = []
    for name in ("markets_recent.json", "markets_historical.json"):
        path = raw_dir / "kalshi" / name
        if path.exists():
            records.extend(json.loads(path.read_text(encoding="utf-8")))
    frame = markets_records_to_frame(records)
    return enrich_kalshi_markets(frame)


def load_raw_br(raw_dir: Path) -> pd.DataFrame:
    """Load Basketball Reference schedule CSV."""
    path = raw_dir / "basketball_reference" / "nba_2026_schedule.csv"
    frame = pd.read_csv(path)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame = frame.drop_duplicates(
        subset=["game_date", "home_team", "visitor_team"], keep="first"
    )
    return frame.reset_index(drop=True)


def build_eda_payload(
    raw_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """Run core EDA computations and write required figures."""
    markets = load_raw_kalshi(raw_dir)
    results = load_raw_br(raw_dir)
    games = game_level_kalshi(markets)
    joined = join_kalshi_results(games, results)

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "kalshi_volume": str(
            plot_volume_distribution(markets, figures_dir / "kalshi_volume_hist.png")
        ),
        "br_home_win_month": str(
            plot_home_win_rate_by_month(results, figures_dir / "br_home_win_by_month.png")
        ),
        "joined_volume_margin": str(
            plot_joined_volume_vs_margin(joined, figures_dir / "joined_volume_vs_margin.png")
        ),
        "joined_home_by_volume": str(
            plot_joined_home_win_by_volume_quartile(
                joined, figures_dir / "joined_home_win_by_volume.png"
            )
        ),
    }

    kalshi_interest_quant = {
        "volume_fp": seven_number_summary(markets["volume_fp"]),
        "last_price_dollars": seven_number_summary(markets["last_price_dollars"]),
        "settlement_value_dollars": seven_number_summary(markets["settlement_value_dollars"]),
    }
    kalshi_interest_cat = {
        "status": categorical_summary(markets["status"]).to_dict(orient="records"),
        "result": categorical_summary(markets["result"]).to_dict(orient="records"),
        "title_format": categorical_summary(markets["title_format"]).to_dict(orient="records"),
    }

    results = results.copy()
    results["home_win"] = (results["home_pts"] > results["visitor_pts"]).astype(int)
    results["point_margin"] = results["home_pts"] - results["visitor_pts"]
    br_interest_quant = {
        "home_pts": seven_number_summary(results["home_pts"]),
        "visitor_pts": seven_number_summary(results["visitor_pts"]),
        "point_margin": seven_number_summary(results["point_margin"]),
    }
    br_interest_cat = {
        "home_win": categorical_summary(results["home_win"]).to_dict(orient="records"),
        "home_team": categorical_summary(results["home_team"]).head(10).to_dict(orient="records"),
    }

    joined_interest_quant = {
        "total_volume": seven_number_summary(joined["total_volume"]),
        "home_pts": seven_number_summary(joined["home_pts"]),
        "visitor_pts": seven_number_summary(joined["visitor_pts"]),
        "home_win": seven_number_summary(joined["home_win"]),
    }
    joined_interest_cat = {
        "home_win": categorical_summary(joined["home_win"]).to_dict(orient="records"),
        "title_format": categorical_summary(joined["title_format"]).to_dict(orient="records"),
    }

    payload: dict[str, Any] = {
        "kalshi": {
            "shape": dataset_shape(markets),
            "row_meaning": "One Yes-contract market for a team in an NBA game.",
            "column_meaning": "Market metadata, prices, volume, settlement, parsed teams/dates.",
            "has_missing": has_any_missing(markets),
            "missingness": missingness_report(markets).to_dict(orient="records"),
            "quant": kalshi_interest_quant,
            "categorical": kalshi_interest_cat,
            "n_events": int(markets["event_ticker"].nunique()) if len(markets) else 0,
            "n_yes_teams_mapped": int(markets["yes_team"].notna().sum()),
        },
        "basketball_reference": {
            "shape": dataset_shape(results),
            "row_meaning": "One NBA game (regular season or playoffs) with final score.",
            "column_meaning": "Game date, visitor/home teams, and final points.",
            "has_missing": has_any_missing(results),
            "missingness": missingness_report(results).to_dict(orient="records"),
            "quant": br_interest_quant,
            "categorical": br_interest_cat,
            "date_min": str(results["game_date"].min().date()) if len(results) else None,
            "date_max": str(results["game_date"].max().date()) if len(results) else None,
        },
        "joined": {
            "shape": dataset_shape(joined),
            "row_meaning": "One NBA game present in both Kalshi and Basketball Reference.",
            "column_meaning": "Kalshi volume/event fields joined to BR scores and home_win.",
            "has_missing": has_any_missing(joined),
            "missingness": missingness_report(joined).to_dict(orient="records"),
            "quant": joined_interest_quant,
            "categorical": joined_interest_cat,
            "relation": (
                "Kalshi KXNBAGAME contracts and Basketball Reference box scores describe the "
                "same NBA games. They are linked on game_date + home_team + visitor_team after "
                "team-name normalization (and team-pair matching for Kalshi 'vs' titles)."
            ),
            "match_rate_vs_kalshi_events": (
                float(len(joined) / games["event_ticker"].nunique())
                if len(games)
                else 0.0
            ),
            "match_rate_vs_br_games": float(len(joined) / len(results)) if len(results) else 0.0,
        },
        "figures": figure_paths,
        "n_kalshi_game_rows": int(len(games)),
    }
    return payload


def save_payload(payload: dict[str, Any], path: Path) -> Path:
    """Persist EDA payload as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
