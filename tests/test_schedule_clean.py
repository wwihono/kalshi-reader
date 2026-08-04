"""Tests for Basketball Reference schedule cleaning."""

import pandas as pd

from kalshi_nba.collect.basketball_reference import clean_schedule_frame


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2025-10-21", "2025-10-21", "Date", "2025-10-22"],
            "Start (ET)": ["7:30p", "7:30p", "Start (ET)", "10:00p"],
            "Visitor/Neutral": ["Houston Rockets", "Houston Rockets", "x", "Bulls"],
            "Home/Neutral": ["OKC Thunder", "OKC Thunder", "y", "Kings"],
            "PTS": ["124", "124", "PTS", "111"],
            "PTS.1": ["125", "125", "PTS", "119"],
        }
    )


def test_clean_schedule_frame_drops_header_rows_and_duplicates():
    out = clean_schedule_frame(_raw_frame())
    # Repeated header row removed, duplicated game removed
    assert len(out) == 2
    assert out["visitor_pts"].dtype.kind in "if"
    assert out.loc[0, "home_pts"] == 125
    assert out.loc[0, "start_et"] == "7:30p"


def test_clean_schedule_frame_parses_dates():
    out = clean_schedule_frame(_raw_frame())
    assert out["game_date"].dt.year.tolist() == [2025, 2025]
