"""CLI entrypoint to download Kalshi and Basketball Reference raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalshi_nba.collect.basketball_reference import fetch_season_schedule
from kalshi_nba.collect.kalshi import (
    fetch_historical_markets,
    fetch_markets,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for raw downloads",
    )
    parser.add_argument(
        "--skip-kalshi",
        action="store_true",
        help="Skip Kalshi API downloads",
    )
    parser.add_argument(
        "--skip-br",
        action="store_true",
        help="Skip Basketball Reference downloads",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_kalshi:
        markets = fetch_markets()
        historical = fetch_historical_markets()
        save_json(markets, out_dir / "kalshi" / "markets_recent.json")
        save_json(historical, out_dir / "kalshi" / "markets_historical.json")
        print(f"Saved {len(markets)} recent and {len(historical)} historical markets")

    if not args.skip_br:
        schedule = fetch_season_schedule()
        br_path = out_dir / "basketball_reference" / "nba_2026_schedule.csv"
        br_path.parent.mkdir(parents=True, exist_ok=True)
        schedule.to_csv(br_path, index=False)
        print(f"Saved schedule with {len(schedule)} rows to {br_path}")


if __name__ == "__main__":
    main()
