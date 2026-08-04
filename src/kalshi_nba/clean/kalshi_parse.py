"""Parse Kalshi KXNBAGAME market records into joinable game rows."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd

from .team_names import normalize_team_name, to_canonical_team

TITLE_AT = re.compile(
    r"(?:Game\s+\d+:\s*)?(?P<visitor>.+?)\s+at\s+(?P<home>.+?)\s+Winner\??\s*$",
    re.IGNORECASE,
)
TITLE_VS = re.compile(
    r"(?:Game\s+\d+:\s*)?(?P<team_a>.+?)\s+vs\.?\s+(?P<team_b>.+?)\s+Winner\??\s*$",
    re.IGNORECASE,
)
TICKER_RE = re.compile(
    r"^KXNBAGAME-(?P<yymmdd>\d{2}[A-Z]{3}\d{2})(?P<code_a>[A-Z]{3})(?P<code_b>[A-Z]{3})-"
    r"(?P<yes>[A-Z]{3})$"
)

# Extra Kalshi display labels beyond the shared CANONICAL_TEAMS map.
KALSHI_LABELS: dict[str, str] = {
    "los angeles l": "Los Angeles Lakers",
    "los angeles c": "Los Angeles Clippers",
    "new york k": "New York Knicks",
    "oklahoma city": "Oklahoma City Thunder",
    "san antonio": "San Antonio Spurs",
    "golden state": "Golden State Warriors",
    "minnesota": "Minnesota Timberwolves",
    "cleveland": "Cleveland Cavaliers",
    "indiana": "Indiana Pacers",
    "denver": "Denver Nuggets",
    "new york": "New York Knicks",
    "detroit": "Detroit Pistons",
    "boston": "Boston Celtics",
    "orlando": "Orlando Magic",
    "houston": "Houston Rockets",
    "philadelphia": "Philadelphia 76ers",
    "atlanta": "Atlanta Hawks",
    "miami": "Miami Heat",
    "toronto": "Toronto Raptors",
    "memphis": "Memphis Grizzlies",
    "portland": "Portland Trail Blazers",
    "phoenix": "Phoenix Suns",
    "milwaukee": "Milwaukee Bucks",
    "dallas": "Dallas Mavericks",
    "charlotte": "Charlotte Hornets",
    "chicago": "Chicago Bulls",
    "sacramento": "Sacramento Kings",
    "utah": "Utah Jazz",
    "washington": "Washington Wizards",
    "brooklyn": "Brooklyn Nets",
    "new orleans": "New Orleans Pelicans",
}


def kalshi_label_to_team(label: str) -> str:
    """Map a Kalshi yes_sub_title / title fragment to a canonical team."""
    key = normalize_team_name(label)
    if key in KALSHI_LABELS:
        return KALSHI_LABELS[key]
    return to_canonical_team(label)


def parse_game_date_from_ticker(ticker: str) -> pd.Timestamp | None:
    """Extract the calendar game date encoded in a KXNBAGAME ticker."""
    match = TICKER_RE.match(ticker)
    if not match:
        return None
    raw = match.group("yymmdd")
    try:
        parsed = datetime.strptime(raw, "%y%b%d")
    except ValueError:
        return None
    return pd.Timestamp(parsed.date())


def parse_title_teams(title: str) -> tuple[str | None, str | None, str]:
    """Return (visitor, home, format) when possible.

    ``at`` titles encode visitor/home. ``vs`` titles leave home/visitor unknown
    (returned as team_a / team_b with format ``vs``).
    """
    text = (title or "").strip()
    at_match = TITLE_AT.search(text)
    if at_match:
        return at_match.group("visitor"), at_match.group("home"), "at"
    vs_match = TITLE_VS.search(text)
    if vs_match:
        return vs_match.group("team_a"), vs_match.group("team_b"), "vs"
    return None, None, "other"


def markets_records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten raw Kalshi market JSON dicts into a tabular DataFrame."""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    # Prefer first occurrence when recent/historical overlap on ticker.
    if "ticker" in frame.columns:
        frame = frame.drop_duplicates(subset=["ticker"], keep="first")
    return frame.reset_index(drop=True)


def enrich_kalshi_markets(markets: pd.DataFrame) -> pd.DataFrame:
    """Add parsed teams, game_date, yes-team, and numeric price/volume fields."""
    if markets.empty:
        return markets.copy()

    out = markets.copy()
    parsed = out["title"].map(parse_title_teams)
    out["title_team_a"] = [p[0] for p in parsed]
    out["title_team_b"] = [p[1] for p in parsed]
    out["title_format"] = [p[2] for p in parsed]

    ticker_dates = out["ticker"].map(parse_game_date_from_ticker)
    occ = pd.to_datetime(out.get("occurrence_datetime"), utc=True, errors="coerce")
    exp = pd.to_datetime(out.get("expected_expiration_time"), utc=True, errors="coerce")
    # Tip times are UTC evening; normalize to a US-evening calendar date via ticker first.
    out["game_date"] = ticker_dates
    missing_date = out["game_date"].isna()
    out.loc[missing_date, "game_date"] = occ[missing_date].dt.tz_convert(None).dt.normalize()
    still_missing = out["game_date"].isna()
    out.loc[still_missing, "game_date"] = exp[still_missing].dt.tz_convert(None).dt.normalize()

    def _safe_label(value: Any) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return kalshi_label_to_team(str(value))
        except KeyError:
            return None

    out["yes_team"] = out["yes_sub_title"].map(_safe_label)
    # Fall back to the 3-letter ticker suffix when the display label is ambiguous.
    ticker_yes = out["ticker"].map(
        lambda t: _safe_label(TICKER_RE.match(str(t)).group("yes"))
        if isinstance(t, str) and TICKER_RE.match(str(t))
        else None
    )
    out["yes_team"] = out["yes_team"].fillna(ticker_yes)
    out["team_a"] = out["title_team_a"].map(_safe_label)
    out["team_b"] = out["title_team_b"].map(_safe_label)
    # For vs titles, recover teams from ticker codes when labels fail.
    ticker_a = out["ticker"].map(
        lambda t: _safe_label(TICKER_RE.match(str(t)).group("code_a"))
        if isinstance(t, str) and TICKER_RE.match(str(t))
        else None
    )
    ticker_b = out["ticker"].map(
        lambda t: _safe_label(TICKER_RE.match(str(t)).group("code_b"))
        if isinstance(t, str) and TICKER_RE.match(str(t))
        else None
    )
    out["team_a"] = out["team_a"].fillna(ticker_a)
    out["team_b"] = out["team_b"].fillna(ticker_b)

    # For "at" titles, team_a/visitor and team_b/home are reliable.
    out["visitor_team"] = pd.NA
    out["home_team"] = pd.NA
    at_mask = out["title_format"] == "at"
    out.loc[at_mask, "visitor_team"] = out.loc[at_mask, "team_a"]
    out.loc[at_mask, "home_team"] = out.loc[at_mask, "team_b"]

    for col in ("volume_fp", "last_price_dollars", "settlement_value_dollars"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["yes_won"] = (out.get("result") == "yes").astype("Int64")
    return out.reset_index(drop=True)


def game_level_kalshi(markets: pd.DataFrame) -> pd.DataFrame:
    """Collapse two yes-contracts per game into one row keyed by event_ticker."""
    required = {"event_ticker", "game_date", "yes_team", "volume_fp"}
    missing = required - set(markets.columns)
    if missing:
        raise KeyError(f"markets missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for event_ticker, group in markets.groupby("event_ticker", sort=False):
        teams = sorted({t for t in group["yes_team"].dropna().unique()})
        if len(teams) != 2:
            continue
        home = group["home_team"].dropna()
        visitor = group["visitor_team"].dropna()
        home_team = home.iloc[0] if len(home) else pd.NA
        visitor_team = visitor.iloc[0] if len(visitor) else pd.NA
        rows.append(
            {
                "event_ticker": event_ticker,
                "game_date": group["game_date"].iloc[0],
                "team_1": teams[0],
                "team_2": teams[1],
                "home_team": home_team,
                "visitor_team": visitor_team,
                "title_format": group["title_format"].iloc[0],
                "total_volume": float(group["volume_fp"].fillna(0).sum()),
                "n_contracts": int(len(group)),
            }
        )
    return pd.DataFrame(rows)
