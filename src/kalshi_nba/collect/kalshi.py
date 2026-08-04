"""Kalshi public API collectors for KXNBAGAME markets.

Endpoints (no auth required for public market data):
- Recent: https://external-api.kalshi.com/trade-api/v2/markets
- Historical: https://external-api.kalshi.com/trade-api/v2/historical/markets
- Cutoff: used to decide which source holds a given market
- Candlesticks: one-minute price history per market
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXNBAGAME"
DEFAULT_LIMIT = 1000
DEFAULT_TIMEOUT = 30


def _get_json(
    url: str,
    params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    client = session or requests.Session()
    response = client.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Kalshi response type: {type(payload)!r}")
    return payload


def paginate(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    items_key: str = "markets",
    session: requests.Session | None = None,
    sleep_s: float = 0.0,
) -> Iterator[dict[str, Any]]:
    """Yield items across cursor-paginated Kalshi list endpoints."""
    params = dict(params or {})
    cursor: str | None = None

    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor

        payload = _get_json(url, params=page_params, session=session)
        items = payload.get(items_key, [])
        if not isinstance(items, list):
            raise ValueError(f"Expected list under '{items_key}'")

        for item in items:
            if isinstance(item, dict):
                yield item

        cursor = payload.get("cursor") or None
        if not cursor:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)


def get_cutoff(session: requests.Session | None = None) -> dict[str, Any]:
    """Return Kalshi historical-data cutoff metadata."""
    return _get_json(f"{BASE_URL}/historical/cutoff", session=session)


def fetch_markets(
    *,
    series_ticker: str = SERIES_TICKER,
    limit: int = DEFAULT_LIMIT,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent markets for a series (live/recent endpoint)."""
    url = f"{BASE_URL}/markets"
    params = {"series_ticker": series_ticker, "limit": limit}
    return list(paginate(url, params=params, session=session))


def fetch_historical_markets(
    *,
    series_ticker: str = SERIES_TICKER,
    limit: int = DEFAULT_LIMIT,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch archived markets for a series (historical endpoint)."""
    url = f"{BASE_URL}/historical/markets"
    params = {"series_ticker": series_ticker, "limit": limit}
    return list(paginate(url, params=params, session=session))


def fetch_candlesticks(
    ticker: str,
    *,
    start_ts: int | None = None,
    end_ts: int | None = None,
    period_interval: int = 1,
    series_ticker: str = SERIES_TICKER,
    historical: bool | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch candlesticks for a market ticker.

    Live/recent markets use ``/series/{series}/markets/{ticker}/candlesticks``.
    Archived markets use ``/historical/markets/{ticker}/candlesticks``. When
    ``historical`` is None, try live first and fall back to historical on 404.
    """
    params: dict[str, Any] = {"period_interval": period_interval}
    if start_ts is not None:
        params["start_ts"] = start_ts
    if end_ts is not None:
        params["end_ts"] = end_ts

    live_url = f"{BASE_URL}/series/{series_ticker}/markets/{ticker}/candlesticks"
    hist_url = f"{BASE_URL}/historical/markets/{ticker}/candlesticks"

    urls: list[str]
    if historical is True:
        urls = [hist_url]
    elif historical is False:
        urls = [live_url]
    else:
        urls = [live_url, hist_url]

    last_error: Exception | None = None
    for url in urls:
        try:
            payload = _get_json(url, params=params, session=session)
        except requests.HTTPError as exc:
            last_error = exc
            response = exc.response
            if response is not None and response.status_code == 404 and url != urls[-1]:
                continue
            raise
        candles = payload.get("candlesticks", [])
        if not isinstance(candles, list):
            raise ValueError("Expected list under 'candlesticks'")
        return [c for c in candles if isinstance(c, dict)]

    if last_error is not None:
        raise last_error
    return []


def save_json(data: Any, path: str | Path) -> Path:
    """Persist raw API payloads for reproducibility."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out
