"""Cleaning and merging utilities.

Includes team-name normalization (Kalshi and Basketball Reference use
different naming formats), Kalshi price -> probability conversion,
timestamp handling, and the market/schedule merge.
"""

from __future__ import annotations

import pandas as pd

# Canonical three-letter codes keyed by every spelling the two data sources
# use: full names (Basketball Reference), cities and nicknames (Kalshi
# subtitles), and common abbreviations.
_TEAM_ALIASES: dict[str, str] = {
    # Full names (Basketball Reference schedule tables)
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
    # Cities (Kalshi yes_sub_title / no_sub_title style)
    "atlanta": "ATL",
    "boston": "BOS",
    "brooklyn": "BKN",
    "charlotte": "CHA",
    "chicago": "CHI",
    "cleveland": "CLE",
    "dallas": "DAL",
    "denver": "DEN",
    "detroit": "DET",
    "golden state": "GSW",
    "houston": "HOU",
    "indiana": "IND",
    "memphis": "MEM",
    "miami": "MIA",
    "milwaukee": "MIL",
    "minnesota": "MIN",
    "new orleans": "NOP",
    "new york": "NYK",
    "oklahoma city": "OKC",
    "orlando": "ORL",
    "philadelphia": "PHI",
    "phoenix": "PHX",
    "portland": "POR",
    "sacramento": "SAC",
    "san antonio": "SAS",
    "toronto": "TOR",
    "utah": "UTA",
    "washington": "WAS",
    # Nicknames
    "hawks": "ATL",
    "celtics": "BOS",
    "nets": "BKN",
    "hornets": "CHA",
    "bulls": "CHI",
    "cavaliers": "CLE",
    "cavs": "CLE",
    "mavericks": "DAL",
    "mavs": "DAL",
    "nuggets": "DEN",
    "pistons": "DET",
    "warriors": "GSW",
    "rockets": "HOU",
    "pacers": "IND",
    "clippers": "LAC",
    "lakers": "LAL",
    "grizzlies": "MEM",
    "heat": "MIA",
    "bucks": "MIL",
    "timberwolves": "MIN",
    "wolves": "MIN",
    "pelicans": "NOP",
    "knicks": "NYK",
    "thunder": "OKC",
    "magic": "ORL",
    "76ers": "PHI",
    "sixers": "PHI",
    "suns": "PHX",
    "trail blazers": "POR",
    "blazers": "POR",
    "kings": "SAC",
    "spurs": "SAS",
    "raptors": "TOR",
    "jazz": "UTA",
    "wizards": "WAS",
    # Alternate abbreviations seen in the wild
    "bkn": "BKN",
    "brk": "BKN",
    "cha": "CHA",
    "cho": "CHA",
    "gsw": "GSW",
    "gs": "GSW",
    "nop": "NOP",
    "no": "NOP",
    "nyk": "NYK",
    "ny": "NYK",
    "okc": "OKC",
    "phx": "PHX",
    "pho": "PHX",
    "sas": "SAS",
    "sa": "SAS",
    "uta": "UTA",
    "was": "WAS",
    "wsh": "WAS",
}

# Every canonical code maps to itself so already-normalized values pass through.
_TEAM_ALIASES.update({code.lower(): code for code in set(_TEAM_ALIASES.values())})


def normalize_team_name(name: str) -> str:
    """Convert any team spelling from either data source to a canonical code.

    >>> normalize_team_name("Boston Celtics")
    'BOS'
    >>> normalize_team_name("Oklahoma City")
    'OKC'

    Raises ``KeyError`` for unrecognized names so bad joins fail loudly
    instead of silently dropping games.
    """
    if not isinstance(name, str) or not name.strip():
        raise KeyError(f"Not a team name: {name!r}")
    key = " ".join(name.strip().lower().split())
    if key not in _TEAM_ALIASES:
        raise KeyError(f"Unrecognized team name: {name!r}")
    return _TEAM_ALIASES[key]


def implied_probability(yes_bid: float, yes_ask: float, price_scale: float = 100.0) -> float:
    """Kalshi market-implied probability: midpoint of the yes bid/ask spread.

    Kalshi quotes prices in cents (0-100), so the default ``price_scale``
    converts the midpoint to a probability in [0, 1]. Pass ``price_scale=1``
    for prices already expressed in dollars.
    """
    if yes_bid is None or yes_ask is None:
        raise ValueError("yes_bid and yes_ask are both required")
    bid, ask = float(yes_bid), float(yes_ask)
    if bid < 0 or ask < 0:
        raise ValueError(f"Prices must be non-negative, got bid={bid}, ask={ask}")
    if bid > ask:
        raise ValueError(f"Crossed market: bid={bid} > ask={ask}")
    prob = (bid + ask) / 2.0 / price_scale
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"Implied probability {prob} outside [0, 1]")
    return prob


def parse_kalshi_timestamp(value: str | int | float) -> pd.Timestamp:
    """Parse a Kalshi timestamp (ISO-8601 string or unix seconds) to UTC."""
    if isinstance(value, (int, float)):
        return pd.Timestamp(value, unit="s", tz="UTC")
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def settlement_to_outcome(settlement_value_dollars: float) -> int:
    """Map a contract settlement value ($0 or $1) to a binary outcome."""
    value = float(settlement_value_dollars)
    if value not in (0.0, 1.0):
        raise ValueError(f"Unexpected settlement value: {value}")
    return int(value)


def merge_markets_with_results(
    markets: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    """Merge Kalshi markets with Basketball Reference results.

    Both frames must already carry normalized team codes:

    - ``markets``: one row per market with ``game_date`` (date), ``home_team``,
      ``away_team`` and Kalshi fields.
    - ``schedule``: one row per game with ``game_date``, ``home_team``,
      ``away_team``, ``home_points``, ``away_points``.

    Only games present in both sources are kept.
    """
    merged = markets.merge(
        schedule,
        on=["game_date", "home_team", "away_team"],
        how="inner",
        validate="many_to_one",
    )
    merged["home_win"] = (merged["home_points"] > merged["away_points"]).astype(int)
    return merged
