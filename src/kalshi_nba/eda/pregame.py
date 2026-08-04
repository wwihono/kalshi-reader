"""Extract pregame Kalshi midpoint probabilities from candlesticks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from kalshi_nba.clean.probability import midpoint_probability
from kalshi_nba.collect.kalshi import fetch_candlesticks


def _candle_bid_ask(candle: dict[str, Any]) -> tuple[float | None, float | None]:
    """Pull yes bid/ask close prices from a candlestick payload."""
    yes_bid = candle.get("yes_bid") or {}
    yes_ask = candle.get("yes_ask") or {}
    bid = yes_bid.get("close_dollars", yes_bid.get("close"))
    ask = yes_ask.get("close_dollars", yes_ask.get("close"))
    if bid is None and ask is None:
        price = candle.get("price") or {}
        close = price.get("close_dollars", price.get("close"))
        if close is None:
            return None, None
        value = float(close)
        return value, value
    return (float(bid) if bid is not None else None, float(ask) if ask is not None else None)


def pregame_midpoint_for_ticker(
    ticker: str,
    tip_time: datetime,
    *,
    minutes_before: int = 15,
    lookback_hours: int = 6,
    period_interval: int = 60,
    session: requests.Session | None = None,
) -> float | None:
    """Return midpoint probability from the last candle ≥ minutes_before tip."""
    if tip_time.tzinfo is None:
        tip_time = tip_time.replace(tzinfo=timezone.utc)
    end = tip_time - timedelta(minutes=minutes_before)
    start = tip_time - timedelta(hours=lookback_hours)
    candles = fetch_candlesticks(
        ticker,
        start_ts=int(start.timestamp()),
        end_ts=int(end.timestamp()),
        period_interval=period_interval,
        session=session,
    )
    if not candles:
        return None
    candles = sorted(candles, key=lambda c: int(c.get("end_period_ts", 0)))
    bid, ask = _candle_bid_ask(candles[-1])
    if bid is None or ask is None:
        return None
    try:
        return midpoint_probability(bid, ask)
    except ValueError:
        return None


def attach_home_pregame_probability(
    markets: pd.DataFrame,
    joined: pd.DataFrame,
    *,
    max_workers: int = 8,
    limit: int | None = None,
) -> pd.DataFrame:
    """Attach home-team pregame Kalshi probabilities onto joined game rows.

    Uses the Yes contract whose ``yes_team`` matches the home team. ``limit``
    optionally caps how many games are queried (useful for tests / smoke runs).
    """
    out = joined.copy()
    if out.empty:
        out["kalshi_home_prob"] = pd.Series(dtype=float)
        return out

    lookup: dict[tuple[str, str], str] = {}
    for _, row in markets.dropna(subset=["event_ticker", "yes_team", "ticker"]).iterrows():
        lookup[(str(row["event_ticker"]), str(row["yes_team"]))] = str(row["ticker"])

    tip_times: dict[str, datetime] = {}
    for time_col in ("occurrence_datetime", "expected_expiration_time"):
        if time_col not in markets.columns:
            continue
        for _, row in markets.dropna(subset=["event_ticker", time_col]).iterrows():
            event = str(row["event_ticker"])
            if event in tip_times and time_col != "occurrence_datetime":
                continue
            tip_times[event] = pd.to_datetime(row[time_col], utc=True).to_pydatetime()

    targets = out.dropna(subset=["event_ticker", "home_team"]).copy()
    if limit is not None:
        targets = targets.head(int(limit))

    jobs: list[tuple[str, str, datetime]] = []
    for _, row in targets.iterrows():
        event = str(row["event_ticker"])
        ticker = lookup.get((event, str(row["home_team"])))
        tip = tip_times.get(event)
        if ticker and tip:
            jobs.append((event, ticker, tip))

    probs: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(pregame_midpoint_for_ticker, ticker, tip): event
            for event, ticker, tip in jobs
        }
        for fut in as_completed(futures):
            event = futures[fut]
            try:
                probs[event] = fut.result()
            except Exception:
                probs[event] = None

    out["kalshi_home_prob"] = out["event_ticker"].map(probs)
    return out
