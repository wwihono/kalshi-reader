"""Tests for Basketball Reference start-time parsing."""

import pandas as pd
import pytest

from kalshi_nba.clean.tipoff import add_tip_datetime, parse_start_et, tip_datetime_utc


def test_parse_start_et_evening():
    assert parse_start_et("7:30p") == (19, 30)


def test_parse_start_et_morning_and_noon_midnight_rules():
    assert parse_start_et("10:00a") == (10, 0)
    assert parse_start_et("12:00p") == (12, 0)
    assert parse_start_et("12:30a") == (0, 30)


def test_parse_start_et_accepts_pm_suffix():
    assert parse_start_et("7:30pm") == (19, 30)


def test_parse_start_et_rejects_garbage():
    with pytest.raises(ValueError):
        parse_start_et("late")
    with pytest.raises(ValueError):
        parse_start_et("13:00p")


def test_tip_datetime_utc_handles_est():
    # January is UTC-5: 7:30 PM ET -> 00:30 UTC next day
    tip = tip_datetime_utc(pd.Timestamp("2026-01-05"), "7:30p")
    assert tip == pd.Timestamp("2026-01-06T00:30:00", tz="UTC")


def test_tip_datetime_utc_handles_edt():
    # April is UTC-4 (daylight saving): 7:30 PM ET -> 23:30 UTC same day
    tip = tip_datetime_utc(pd.Timestamp("2026-04-10"), "7:30p")
    assert tip == pd.Timestamp("2026-04-10T23:30:00", tz="UTC")


def test_add_tip_datetime_keeps_missing_starts_missing():
    schedule = pd.DataFrame(
        {
            "game_date": ["2026-01-05", "2026-01-06"],
            "start_et": ["7:30p", None],
        }
    )
    out = add_tip_datetime(schedule)
    assert out.loc[0, "tip_datetime_utc"] == pd.Timestamp(
        "2026-01-06T00:30:00", tz="UTC"
    )
    assert pd.isna(out.loc[1, "tip_datetime_utc"])
