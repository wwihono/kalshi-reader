"""Scheduled tip-off times from Basketball Reference "Start (ET)" strings.

Most Kalshi KXNBAGAME records are missing ``occurrence_datetime``, so the
scheduled start has to be reconstructed from the Basketball Reference
schedule ("7:30p" style strings, US Eastern time).
"""

from __future__ import annotations

import re
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")

_START_ET_RE = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})(?P<ampm>[ap])m?$")


def parse_start_et(text: str) -> tuple[int, int]:
    """Parse a Basketball Reference start string into (hour24, minute).

    Accepts "7:30p", "10:00a", or "7:30pm". Raises ValueError otherwise.
    """
    match = _START_ET_RE.match(str(text).strip().lower())
    if match is None:
        raise ValueError(f"Unparseable start time: {text!r}")
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if not (1 <= hour <= 12) or not (0 <= minute <= 59):
        raise ValueError(f"Start time out of range: {text!r}")
    if match.group("ampm") == "p" and hour != 12:
        hour += 12
    if match.group("ampm") == "a" and hour == 12:
        hour = 0
    return hour, minute


def tip_datetime_utc(game_date: pd.Timestamp, start_et: str) -> pd.Timestamp:
    """Combine a game date and an Eastern start string into a UTC timestamp."""
    hour, minute = parse_start_et(start_et)
    local = pd.Timestamp(
        year=game_date.year,
        month=game_date.month,
        day=game_date.day,
        hour=hour,
        minute=minute,
        tz=EASTERN,
    )
    return local.tz_convert("UTC")


def add_tip_datetime(
    schedule: pd.DataFrame,
    *,
    date_col: str = "game_date",
    start_col: str = "start_et",
    out_col: str = "tip_datetime_utc",
) -> pd.DataFrame:
    """Attach a UTC tip-off timestamp column to a schedule frame."""
    out = schedule.copy()
    dates = pd.to_datetime(out[date_col])
    out[out_col] = [
        tip_datetime_utc(date, start) if pd.notna(start) else pd.NaT
        for date, start in zip(dates, out[start_col], strict=True)
    ]
    return out
