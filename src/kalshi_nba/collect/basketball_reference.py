"""Basketball Reference NBA schedule/results collectors."""

from __future__ import annotations

from collections.abc import Iterable
from io import StringIO
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

SEASON_SCHEDULE_URL = "https://www.basketball-reference.com/leagues/NBA_2026_games.html"
DEFAULT_TIMEOUT = 30
USER_AGENT = (
    "kalshi-nba-research/0.1 (+https://github.com/wwihono/kalshi-reader; academic use)"
)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _month_links(html: str, base_url: str = SEASON_SCHEDULE_URL) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for anchor in soup.select("div.filter a"):
        href = anchor.get("href")
        if not href:
            continue
        if href.startswith("http"):
            links.append(href)
        else:
            # Monthly pages are under /leagues/...
            links.append(urljoin(base_url, href))
    # The landing page duplicates the first month's table, so only fall
    # back to it when no month links were found at all.
    if not links:
        links.append(base_url)
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for link in links:
        if link not in seen:
            seen.add(link)
            ordered.append(link)
    return ordered


def parse_schedule_table(html: str) -> pd.DataFrame:
    """Parse the main schedule table from a Basketball Reference HTML page."""
    tables = pd.read_html(StringIO(html), attrs={"id": "schedule"})
    if not tables:
        raise ValueError("No schedule table found")
    frame = tables[0].copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def fetch_season_schedule(
    season_url: str = SEASON_SCHEDULE_URL,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download and concatenate monthly NBA schedule/result tables."""
    client = session or _session()
    landing = client.get(season_url, timeout=DEFAULT_TIMEOUT)
    landing.raise_for_status()

    month_urls = _month_links(landing.text, base_url=season_url)
    frames: list[pd.DataFrame] = []
    for url in month_urls:
        response = client.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        try:
            frames.append(parse_schedule_table(response.text))
        except ValueError:
            continue

    if not frames:
        raise RuntimeError("Failed to parse any monthly schedule tables")

    combined = pd.concat(frames, ignore_index=True)
    return clean_schedule_frame(combined)


def clean_schedule_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Basketball Reference schedule columns."""
    rename = {
        "Date": "game_date",
        "Start (ET)": "start_et",
        "Visitor/Neutral": "visitor_team",
        "Home/Neutral": "home_team",
        "PTS": "visitor_pts",
        "PTS.1": "home_pts",
        "OT": "overtime",
    }
    out = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})

    # Drop repeated header rows sometimes embedded in the HTML table
    if "game_date" in out.columns:
        out = out[out["game_date"].astype(str).str.lower() != "date"]

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    for col in ("visitor_pts", "home_pts"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    keep = [
        c
        for c in (
            "game_date",
            "start_et",
            "visitor_team",
            "home_team",
            "visitor_pts",
            "home_pts",
            "overtime",
        )
        if c in out.columns
    ]
    # NBA teams play at most once per day, so identical rows are always
    # scrape artifacts (e.g. a month table parsed twice).
    return out[keep].drop_duplicates().reset_index(drop=True)


def concat_monthly_tables(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine pre-downloaded monthly tables."""
    frames = list(tables)
    if not frames:
        raise ValueError("No monthly tables provided")
    return clean_schedule_frame(pd.concat(frames, ignore_index=True))
