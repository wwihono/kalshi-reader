"""End-to-end pipeline: collect -> clean -> features -> model -> simulate.

Stages can be run individually so the analysis is reproducible from files
saved under data/ once collection has happened:

    python scripts/run_pipeline.py collect     # needs network access
    python scripts/run_pipeline.py analyze     # runs from saved data
    python scripts/run_pipeline.py all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import calibration, cleaning, data_collection, features, modeling, simulation, visualization

PROCESSED_DIR = Path("data/processed")
MERGED_PATH = PROCESSED_DIR / "merged_games.csv"


def collect() -> None:
    """Download Kalshi markets and the Basketball Reference schedule."""
    print("Fetching Kalshi markets (live + historical)...")
    markets = data_collection.fetch_all_markets()
    data_collection.save_raw_json(markets, "kalshi_markets")
    print(f"  saved {len(markets)} markets to data/raw/kalshi_markets.json")

    print("Scraping Basketball Reference schedule...")
    schedule = data_collection.fetch_bbref_schedule()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(PROCESSED_DIR / "bbref_schedule.csv", index=False)
    print(f"  saved {len(schedule)} completed games to data/processed/bbref_schedule.csv")


def clean() -> None:
    """Normalize team names, compute pregame probabilities, merge sources."""
    raw_markets = json.loads((data_collection.RAW_DIR / "kalshi_markets.json").read_text())
    schedule = pd.read_csv(PROCESSED_DIR / "bbref_schedule.csv", parse_dates=["game_date"])
    schedule["game_date"] = schedule["game_date"].dt.date
    schedule["home_team"] = schedule["home_team_name"].map(cleaning.normalize_team_name)
    schedule["away_team"] = schedule["away_team_name"].map(cleaning.normalize_team_name)

    rows = []
    for market in raw_markets:
        try:
            rows.append(
                {
                    "ticker": market["ticker"],
                    "event_ticker": market.get("event_ticker"),
                    "game_date": cleaning.parse_kalshi_timestamp(
                        market["occurrence_datetime"]
                    ).date(),
                    "home_team": cleaning.normalize_team_name(market["no_sub_title"]),
                    "away_team": cleaning.normalize_team_name(market["yes_sub_title"]),
                    "kalshi_prob": cleaning.implied_probability(
                        market["yes_bid"], market["yes_ask"]
                    ),
                    "volume": market.get("volume_fp"),
                    "settlement_value_dollars": market.get("settlement_value_dollars"),
                }
            )
        except (KeyError, ValueError):
            continue  # incomplete records are dropped per the work plan
    markets_df = pd.DataFrame(rows)
    merged = cleaning.merge_markets_with_results(markets_df, schedule)
    merged.to_csv(MERGED_PATH, index=False)
    print(f"Merged {len(merged)} games -> {MERGED_PATH}")


def analyze() -> None:
    """Features, chronological split, model tuning, comparison, simulation."""
    merged = pd.read_csv(MERGED_PATH, parse_dates=["game_date"])
    modeled = features.build_game_features(merged)
    train, val, test = features.chronological_split(modeled)
    X_cols, y_col = features.FEATURE_COLUMNS, features.TARGET_COLUMN

    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test games")
    tuned = modeling.tune_all_models(
        train[X_cols], train[y_col], val[X_cols], val[y_col]
    )
    comparison = modeling.compare_on_test(
        tuned, test[X_cols], test[y_col], kalshi_probs=test["kalshi_prob"].to_numpy()
    )
    print("\nTest-period comparison:\n", comparison.round(4))
    comparison.to_csv(PROCESSED_DIR / "model_comparison.csv")
    visualization.plot_model_comparison(comparison)

    table = calibration.calibration_table(
        merged["kalshi_prob"].to_numpy(), merged["home_win"].to_numpy()
    )
    visualization.plot_calibration(table)

    with_quartiles = calibration.add_quartile_column(merged, "volume", "volume_quartile")
    grouped = calibration.grouped_market_performance(with_quartiles, "volume_quartile")
    visualization.plot_grouped_metric(grouped, "volume_quartile")

    best_name = comparison.drop(index="kalshi", errors="ignore")["log_loss"].idxmin()
    test = test.copy()
    test["model_prob"] = tuned[best_name].estimator.predict_proba(test[X_cols])[:, 1]
    sweep = simulation.threshold_sweep(test)
    print("\nSimulated returns:\n", sweep.round(4))
    sweep.to_csv(PROCESSED_DIR / "simulation_summary.csv", index=False)
    trades_by_threshold = {
        t: simulation.simulate_trades(test, t) for t in (0.05, 0.10, 0.15)
    }
    visualization.plot_cumulative_returns(trades_by_threshold)
    print("\nFigures written to results/figures/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["collect", "clean", "analyze", "all"], help="pipeline stage"
    )
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        collect()
    if args.stage in ("clean", "all"):
        clean()
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
