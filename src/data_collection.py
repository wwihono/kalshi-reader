"""Data collection: Kalshi public API and Basketball Reference schedule.

Kalshi splits NBA game-winner markets (series ``KXNBAGAME``) between a live
endpoint and a historical archive, with a cutoff endpoint indicating which
source holds which markets. Both endpoints paginate with cursors. Raw
responses are saved to ``data/raw/`` so the analysis can be reproduced
without re-hitting the API.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXNBAGAME"
BBREF_SCHEDULE_URL = "https://www.basketball-reference.com/leagues/NBA_2026_games.html"

RAW_DIR = Path("data/raw")


def _get_json(url: str, params: dict[str, Any] | None = None, retries: int = 3) -> dict:
    """GET with basic retry/backoff; Kalshi's public endpoints need no auth."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def iterate_paginated(
    url: str, params: dict[str, Any], items_key: str
) -> Iterator[dict]:
    """Follow Kalshi cursor pagination until the returned cursor is empty."""
    cursor: str | None = None
    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        payload = _get_json(url, page_params)
        yield from payload.get(items_key, [])
        cursor = payload.get("cursor")
        if not cursor:
            break


def fetch_markets(
    series_ticker: str = SERIES_TICKER, limit: int = 1000, historical: bool = False
) -> list[dict]:
    """Fetch all markets for a series from the live or historical endpoint."""
    path = "historical/markets" if historical else "markets"
    url = f"{KALSHI_BASE_URL}/{path}"
    params = {"series_ticker": series_ticker, "limit": limit}
    return list(iterate_paginated(url, params, items_key="markets"))


def fetch_historical_cutoff() -> dict:
    """Cutoff metadata describing which markets live in the archive."""
    return _get_json(f"{KALSHI_BASE_URL}/historical/cutoff")


def fetch_all_markets(series_ticker: str = SERIES_TICKER) -> list[dict]:
    """Combine live and archived markets for the series, deduplicated by ticker."""
    live = fetch_markets(series_ticker, historical=False)
    archived = fetch_markets(series_ticker, historical=True)
    by_ticker: dict[str, dict] = {}
    for market in archived + live:  # live records win on overlap
        by_ticker[market["ticker"]] = market
    return list(by_ticker.values())


def fetch_candlesticks(
    market_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval_minutes: int = 1,
) -> list[dict]:
    """One-minute price history for a single market."""
    url = f"{KALSHI_BASE_URL}/historical/markets/{market_ticker}/candlesticks"
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval_minutes,
    }
    payload = _get_json(url, params)
    return payload.get("candlesticks", [])


def last_pregame_candle(
    candles: list[dict], game_start_ts: int, buffer_minutes: int = 15
) -> dict | None:
    """The final candle at least ``buffer_minutes`` before the scheduled start.

    Guarantees no live-game information enters the pregame price. Candles
    must carry an ``end_period_ts`` unix-seconds field; returns ``None``
    when no candle qualifies.
    """
    cutoff = game_start_ts - buffer_minutes * 60
    eligible = [c for c in candles if c.get("end_period_ts", 0) <= cutoff]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c["end_period_ts"])


def save_raw_json(payload: Any, name: str, raw_dir: Path = RAW_DIR) -> Path:
    """Persist an original API response under data/raw/ for reproducibility."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def fetch_bbref_schedule(url: str = BBREF_SCHEDULE_URL) -> pd.DataFrame:
    """Scrape and combine Basketball Reference's monthly schedule tables.

    The season page links to one page per month; each holds a single
    schedule table. Tables are collected programmatically (never copied by
    hand) and concatenated, keeping only completed games.
    """
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(response.text, "lxml")
    month_links = {
        a["href"]
        for a in soup.select("div.filter a[href]")
        if "/leagues/NBA_2026_games-" in a["href"]
    }
    frames = []
    pages = sorted(month_links) or [url]
    for page in pages:
        page_url = page if page.startswith("http") else f"https://www.basketball-reference.com{page}"
        tables = pd.read_html(page_url)
        if tables:
            frames.append(tables[0])
        time.sleep(3)  # stay under Basketball Reference's rate limit
    schedule = pd.concat(frames, ignore_index=True)
    return normalize_bbref_schedule(schedule)


def normalize_bbref_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize a raw Basketball Reference schedule table.

    Renames columns, parses dates, drops header/unplayed rows, and returns
    one row per completed game with the columns the rest of the pipeline
    expects (team names are normalized separately in ``src.cleaning``).
    """
    df = raw.rename(
        columns={
            "Date": "game_date",
            "Visitor/Neutral": "away_team_name",
            "Home/Neutral": "home_team_name",
            "PTS": "away_points",
            "PTS.1": "home_points",
        }
    )
    df = df[df["game_date"].astype(str).str.lower() != "date"]  # repeated header rows
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df = df.dropna(subset=["game_date", "away_points", "home_points"])
    df["away_points"] = df["away_points"].astype(int)
    df["home_points"] = df["home_points"].astype(int)
    keep = ["game_date", "away_team_name", "home_team_name", "away_points", "home_points"]
    return df[keep].reset_index(drop=True)
