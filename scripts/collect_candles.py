"""Download final pregame Kalshi prices for each NBA game.

For every Kalshi event, this script picks the market whose Yes side is
the home team, works out the scheduled tip-off (from the market's
occurrence_datetime when present, otherwise from the Basketball
Reference start time), and stores the last 1-minute candle ending at
least 15 minutes before tip-off. Results land in
``data/raw/kalshi/pregame_prices.json`` for the EDA to consume.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from kalshi_nba.clean.markets import add_event_fields, markets_to_frame
from kalshi_nba.clean.probability import pregame_price_from_candles
from kalshi_nba.clean.tipoff import add_tip_datetime
from kalshi_nba.collect.kalshi import (
    fetch_candlesticks,
    fetch_historical_candlesticks,
    save_json,
)

CUTOFF_MINUTES = 15
WINDOW_HOURS = 6
MAX_RETRIES = 6


def _fetch_with_backoff(fetch, *args, **kwargs):
    """Call a fetch function, backing off exponentially on HTTP 429."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            return fetch(*args, **kwargs)
        except requests.HTTPError as err:
            status = err.response.status_code if err.response is not None else None
            if status != 429 or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def load_markets(raw_dir: Path) -> tuple[pd.DataFrame, set[str]]:
    """Load raw market JSON files; return tidy frame + recent-ticker set."""
    recent = json.loads((raw_dir / "kalshi" / "markets_recent.json").read_text())
    historical = json.loads(
        (raw_dir / "kalshi" / "markets_historical.json").read_text()
    )
    frame = add_event_fields(markets_to_frame(recent + historical))
    recent_tickers = {m["ticker"] for m in recent}
    return frame, recent_tickers


def build_home_market_table(markets: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per event: home-side market plus scheduled tip-off time."""
    home_side = markets[markets["yes_is_home"]].copy()

    schedule = add_tip_datetime(schedule)
    tips = schedule[["game_date", "home_team", "visitor_team", "tip_datetime_utc"]]
    merged = home_side.merge(
        tips,
        on=["game_date", "home_team", "visitor_team"],
        how="left",
    )
    # Prefer Kalshi's own scheduled time when available.
    merged["tip_utc"] = merged["occurrence_datetime"].fillna(
        merged["tip_datetime_utc"]
    )
    return merged.drop_duplicates(subset="ticker").reset_index(drop=True)


def fetch_pregame_record(
    row: pd.Series,
    *,
    recent_tickers: set[str],
    session: requests.Session,
) -> dict | None:
    """Fetch candles for one market and select the last pregame candle."""
    tip = row["tip_utc"]
    if pd.isna(tip):
        return None
    cutoff_ts = int(tip.timestamp()) - CUTOFF_MINUTES * 60
    start_ts = cutoff_ts - WINDOW_HOURS * 3600

    fetch = (
        fetch_candlesticks
        if row["ticker"] in recent_tickers
        else fetch_historical_candlesticks
    )
    candles = _fetch_with_backoff(
        fetch, row["ticker"], start_ts=start_ts, end_ts=cutoff_ts, session=session
    )
    if not candles:
        # Thinly traded / early-listed markets: retry with hourly candles
        # over the week before tip-off.
        candles = _fetch_with_backoff(
            fetch,
            row["ticker"],
            start_ts=cutoff_ts - 7 * 24 * 3600,
            end_ts=cutoff_ts,
            period_interval=60,
            session=session,
        )
    selected = pregame_price_from_candles(candles, cutoff_ts)
    if selected is None:
        return None
    return {
        "ticker": row["ticker"],
        "event_ticker": row["event_ticker"],
        "cutoff_ts": cutoff_ts,
        **selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Seconds to sleep between API requests",
    )
    args = parser.parse_args()

    markets, recent_tickers = load_markets(args.raw_dir)
    schedule = pd.read_csv(
        args.raw_dir / "basketball_reference" / "nba_2026_schedule.csv",
        parse_dates=["game_date"],
    )
    table = build_home_market_table(markets, schedule)
    print(f"{len(table)} events; {table['tip_utc'].notna().sum()} with tip-off times")

    session = requests.Session()
    records: list[dict] = []
    skipped: list[str] = []
    for i, (_, row) in enumerate(table.iterrows(), start=1):
        try:
            record = fetch_pregame_record(
                row, recent_tickers=recent_tickers, session=session
            )
        except requests.HTTPError as err:
            print(f"HTTP error for {row['ticker']}: {err}")
            record = None
        if record is None:
            skipped.append(row["ticker"])
        else:
            records.append(record)
        if i % 100 == 0:
            print(f"  {i}/{len(table)} events processed")
        time.sleep(args.sleep)

    out_path = args.raw_dir / "kalshi" / "pregame_prices.json"
    save_json(records, out_path)
    print(f"Saved {len(records)} pregame prices to {out_path}")
    print(f"Skipped {len(skipped)} events (no tip time or no candles)")
    if skipped:
        save_json(skipped, args.raw_dir / "kalshi" / "pregame_skipped.json")


if __name__ == "__main__":
    main()
