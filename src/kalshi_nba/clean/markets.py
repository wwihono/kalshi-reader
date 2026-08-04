"""Tidy raw Kalshi KXNBAGAME market records into typed DataFrames.

Raw API records are messy in several ways this module papers over:

- Prices, volumes, and open interest arrive as strings
  (``"0.3400"``, ``"48467702.82"``).
- ``occurrence_datetime`` is present only for a minority of markets
  (playoff-era records); the game date must instead be parsed from the
  event ticker (e.g. ``KXNBAGAME-26JAN05PHXHOU`` -> 2026-01-05).
- Team labels in titles/subtitles are ambiguous ("LA", "Los Angeles"),
  so teams are decoded from the 3-letter abbreviations embedded in the
  event ticker instead.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# KXNBAGAME-<YY><MON><DD><VISITOR><HOME>, all abbreviations 3 letters.
EVENT_TICKER_RE = re.compile(
    r"^KXNBAGAME-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})"
    r"(?P<visitor>[A-Z]{3})(?P<home>[A-Z]{3})$"
)

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Ticker abbreviations observed in KXNBAGAME event tickers. Non-NBA
# exhibition opponents (e.g. GUA = Guangzhou) are intentionally absent
# and map to <NA> so they can be filtered out.
ABBREV_TO_TEAM: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BKN": "Brooklyn Nets",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

# Raw string columns converted to floats in the tidy frame.
NUMERIC_COLUMNS = (
    "yes_bid_dollars",
    "yes_ask_dollars",
    "last_price_dollars",
    "settlement_value_dollars",
    "liquidity_dollars",
    "volume_fp",
    "volume_24h_fp",
    "open_interest_fp",
)

DATETIME_COLUMNS = (
    "occurrence_datetime",
    "open_time",
    "close_time",
    "expected_expiration_time",
    "settlement_ts",
)

KEEP_COLUMNS = (
    "ticker",
    "event_ticker",
    "title",
    "yes_sub_title",
    "no_sub_title",
    "status",
    "result",
    *NUMERIC_COLUMNS,
    *DATETIME_COLUMNS,
)


def parse_event_ticker(event_ticker: str) -> dict[str, Any]:
    """Decode game date and team abbreviations from an event ticker.

    Raises ValueError when the ticker does not follow the
    ``KXNBAGAME-YYMONDDVISHOM`` pattern.
    """
    match = EVENT_TICKER_RE.match(str(event_ticker))
    if match is None:
        raise ValueError(f"Unparseable event ticker: {event_ticker!r}")
    year = 2000 + int(match.group("yy"))
    month = _MONTHS[match.group("mon")]
    day = int(match.group("dd"))
    return {
        "game_date": pd.Timestamp(year=year, month=month, day=day),
        "visitor_abbrev": match.group("visitor"),
        "home_abbrev": match.group("home"),
    }


def markets_to_frame(markets: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw market dicts to a typed DataFrame (one row per market).

    Duplicated tickers (markets present in both the recent and historical
    endpoints) are dropped, keeping the first occurrence.
    """
    frame = pd.DataFrame(markets)
    if frame.empty:
        raise ValueError("no market records provided")
    frame = frame.drop_duplicates(subset="ticker", keep="first")

    keep = [c for c in KEEP_COLUMNS if c in frame.columns]
    out = frame[keep].copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in DATETIME_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(
                out[col], format="ISO8601", utc=True, errors="coerce"
            )
    return out.reset_index(drop=True)


def add_event_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach game date, teams, and side flags decoded from tickers.

    Adds ``game_date``, ``visitor_abbrev``, ``home_abbrev``,
    ``visitor_team``, ``home_team`` (pd.NA for non-NBA opponents),
    ``yes_abbrev``, and ``yes_is_home``.
    """
    out = frame.copy()
    parsed = out["event_ticker"].map(parse_event_ticker)
    out["game_date"] = parsed.map(lambda d: d["game_date"])
    out["visitor_abbrev"] = parsed.map(lambda d: d["visitor_abbrev"])
    out["home_abbrev"] = parsed.map(lambda d: d["home_abbrev"])
    out["visitor_team"] = out["visitor_abbrev"].map(ABBREV_TO_TEAM).astype("object")
    out["home_team"] = out["home_abbrev"].map(ABBREV_TO_TEAM).astype("object")
    out["yes_abbrev"] = out["ticker"].str.rsplit("-", n=1).str[1]
    out["yes_is_home"] = out["yes_abbrev"] == out["home_abbrev"]
    return out
