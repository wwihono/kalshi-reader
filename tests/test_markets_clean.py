"""Tests for tidying raw Kalshi market records."""

import pandas as pd
import pytest

from kalshi_nba.clean.markets import (
    add_event_fields,
    markets_to_frame,
    parse_event_ticker,
)


def _raw_market(**overrides) -> dict:
    base = {
        "ticker": "KXNBAGAME-26JAN05PHXHOU-HOU",
        "event_ticker": "KXNBAGAME-26JAN05PHXHOU",
        "title": "Phoenix vs Houston Winner?",
        "yes_sub_title": "Houston",
        "no_sub_title": "Phoenix",
        "status": "finalized",
        "result": "yes",
        "yes_bid_dollars": "0.7200",
        "yes_ask_dollars": "0.7300",
        "settlement_value_dollars": "1.0000",
        "volume_fp": "5559753.00",
        "open_interest_fp": "0.00",
        "occurrence_datetime": None,
        "open_time": "2026-01-03T19:06:00Z",
        "close_time": "2026-01-06T03:50:20.942795Z",
    }
    base.update(overrides)
    return base


def test_parse_event_ticker_decodes_date_and_teams():
    parsed = parse_event_ticker("KXNBAGAME-26JAN05PHXHOU")
    assert parsed["game_date"] == pd.Timestamp("2026-01-05")
    assert parsed["visitor_abbrev"] == "PHX"
    assert parsed["home_abbrev"] == "HOU"


def test_parse_event_ticker_rejects_garbage():
    with pytest.raises(ValueError):
        parse_event_ticker("KXNBASERIES-26JAN05PHXHOU")


def test_markets_to_frame_converts_types_and_dedupes():
    raw = [_raw_market(), _raw_market(), _raw_market(ticker="OTHER-TICKER")]
    frame = markets_to_frame(raw)
    # Duplicate ticker dropped
    assert len(frame) == 2
    assert frame["yes_bid_dollars"].dtype == float
    assert frame.loc[0, "yes_bid_dollars"] == pytest.approx(0.72)
    assert frame.loc[0, "volume_fp"] == pytest.approx(5559753.0)
    # Missing occurrence stays missing, close_time parses with tz
    assert pd.isna(frame.loc[0, "occurrence_datetime"])
    assert frame.loc[0, "close_time"].tzinfo is not None


def test_add_event_fields_resolves_teams_and_side():
    frame = markets_to_frame([_raw_market()])
    out = add_event_fields(frame)
    row = out.iloc[0]
    assert row["game_date"] == pd.Timestamp("2026-01-05")
    assert row["visitor_team"] == "Phoenix Suns"
    assert row["home_team"] == "Houston Rockets"
    assert row["yes_abbrev"] == "HOU"
    assert bool(row["yes_is_home"]) is True


def test_add_event_fields_flags_non_nba_opponents_as_missing():
    raw = _raw_market(
        ticker="KXNBAGAME-25OCT13GUAMIN-MIN",
        event_ticker="KXNBAGAME-25OCT13GUAMIN",
    )
    out = add_event_fields(markets_to_frame([raw]))
    assert pd.isna(out.loc[0, "visitor_team"])
    assert out.loc[0, "home_team"] == "Minnesota Timberwolves"


def test_add_event_fields_disambiguates_la_teams():
    clippers = _raw_market(
        ticker="KXNBAGAME-25NOV06LACPHX-LAC",
        event_ticker="KXNBAGAME-25NOV06LACPHX",
        yes_sub_title="LA",
    )
    lakers = _raw_market(
        ticker="KXNBAGAME-25NOV03LALPOR-LAL",
        event_ticker="KXNBAGAME-25NOV03LALPOR",
        yes_sub_title="Los Angeles",
    )
    out = add_event_fields(markets_to_frame([clippers, lakers]))
    assert out.loc[0, "visitor_team"] == "Los Angeles Clippers"
    assert out.loc[1, "visitor_team"] == "Los Angeles Lakers"
    # Both are visitor-side markets in these events
    assert not out["yes_is_home"].any()
