"""Team-name normalization between Kalshi and Basketball Reference."""

from __future__ import annotations

import re
import unicodedata

# Canonical NBA team names used for joins across sources.
CANONICAL_TEAMS: dict[str, str] = {
    # City / full names
    "atlanta hawks": "Atlanta Hawks",
    "boston celtics": "Boston Celtics",
    "brooklyn nets": "Brooklyn Nets",
    "charlotte hornets": "Charlotte Hornets",
    "chicago bulls": "Chicago Bulls",
    "cleveland cavaliers": "Cleveland Cavaliers",
    "dallas mavericks": "Dallas Mavericks",
    "denver nuggets": "Denver Nuggets",
    "detroit pistons": "Detroit Pistons",
    "golden state warriors": "Golden State Warriors",
    "houston rockets": "Houston Rockets",
    "indiana pacers": "Indiana Pacers",
    "la clippers": "Los Angeles Clippers",
    "los angeles clippers": "Los Angeles Clippers",
    "la lakers": "Los Angeles Lakers",
    "los angeles lakers": "Los Angeles Lakers",
    "memphis grizzlies": "Memphis Grizzlies",
    "miami heat": "Miami Heat",
    "milwaukee bucks": "Milwaukee Bucks",
    "minnesota timberwolves": "Minnesota Timberwolves",
    "new orleans pelicans": "New Orleans Pelicans",
    "new york knicks": "New York Knicks",
    "oklahoma city thunder": "Oklahoma City Thunder",
    "orlando magic": "Orlando Magic",
    "philadelphia 76ers": "Philadelphia 76ers",
    "phoenix suns": "Phoenix Suns",
    "portland trail blazers": "Portland Trail Blazers",
    "sacramento kings": "Sacramento Kings",
    "san antonio spurs": "San Antonio Spurs",
    "toronto raptors": "Toronto Raptors",
    "utah jazz": "Utah Jazz",
    "washington wizards": "Washington Wizards",
    # Common nicknames / abbreviations
    "hawks": "Atlanta Hawks",
    "atl": "Atlanta Hawks",
    "celtics": "Boston Celtics",
    "bos": "Boston Celtics",
    "nets": "Brooklyn Nets",
    "bkn": "Brooklyn Nets",
    "hornets": "Charlotte Hornets",
    "cha": "Charlotte Hornets",
    "bulls": "Chicago Bulls",
    "chi": "Chicago Bulls",
    "cavaliers": "Cleveland Cavaliers",
    "cavs": "Cleveland Cavaliers",
    "cle": "Cleveland Cavaliers",
    "mavericks": "Dallas Mavericks",
    "mavs": "Dallas Mavericks",
    "dal": "Dallas Mavericks",
    "nuggets": "Denver Nuggets",
    "den": "Denver Nuggets",
    "pistons": "Detroit Pistons",
    "det": "Detroit Pistons",
    "warriors": "Golden State Warriors",
    "gsw": "Golden State Warriors",
    "gs": "Golden State Warriors",
    "rockets": "Houston Rockets",
    "hou": "Houston Rockets",
    "pacers": "Indiana Pacers",
    "ind": "Indiana Pacers",
    "clippers": "Los Angeles Clippers",
    "lac": "Los Angeles Clippers",
    "lakers": "Los Angeles Lakers",
    "lal": "Los Angeles Lakers",
    "grizzlies": "Memphis Grizzlies",
    "mem": "Memphis Grizzlies",
    "heat": "Miami Heat",
    "mia": "Miami Heat",
    "bucks": "Milwaukee Bucks",
    "mil": "Milwaukee Bucks",
    "timberwolves": "Minnesota Timberwolves",
    "wolves": "Minnesota Timberwolves",
    "min": "Minnesota Timberwolves",
    "pelicans": "New Orleans Pelicans",
    "nop": "New Orleans Pelicans",
    "no": "New Orleans Pelicans",
    "knicks": "New York Knicks",
    "nyk": "New York Knicks",
    "ny": "New York Knicks",
    "thunder": "Oklahoma City Thunder",
    "okc": "Oklahoma City Thunder",
    "magic": "Orlando Magic",
    "orl": "Orlando Magic",
    "76ers": "Philadelphia 76ers",
    "sixers": "Philadelphia 76ers",
    "phi": "Philadelphia 76ers",
    "suns": "Phoenix Suns",
    "phx": "Phoenix Suns",
    "phoenix": "Phoenix Suns",
    "trail blazers": "Portland Trail Blazers",
    "blazers": "Portland Trail Blazers",
    "por": "Portland Trail Blazers",
    "kings": "Sacramento Kings",
    "sac": "Sacramento Kings",
    "spurs": "San Antonio Spurs",
    "sas": "San Antonio Spurs",
    "sa": "San Antonio Spurs",
    "raptors": "Toronto Raptors",
    "tor": "Toronto Raptors",
    "jazz": "Utah Jazz",
    "uta": "Utah Jazz",
    "utah": "Utah Jazz",
    "wizards": "Washington Wizards",
    "was": "Washington Wizards",
    "wsh": "Washington Wizards",
}


def normalize_team_name(name: str) -> str:
    """Lowercase, strip accents/punctuation, and collapse whitespace."""
    if name is None:
        raise ValueError("team name must not be None")
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_canonical_team(name: str) -> str:
    """Map a source-specific team label to a canonical NBA team name.

    Raises KeyError when the name cannot be mapped.
    """
    key = normalize_team_name(name)
    if key in CANONICAL_TEAMS:
        return CANONICAL_TEAMS[key]

    # Try last token (nickname) when a longer phrase is present
    tokens = key.split()
    if len(tokens) >= 2:
        nickname = tokens[-1]
        if nickname in CANONICAL_TEAMS:
            return CANONICAL_TEAMS[nickname]
        # Portland Trail Blazers / Philadelphia 76ers style
        two = " ".join(tokens[-2:])
        if two in CANONICAL_TEAMS:
            return CANONICAL_TEAMS[two]

    raise KeyError(f"Unrecognized team name: {name!r}")
