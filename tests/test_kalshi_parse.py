"""Tests for Kalshi market parsing and enrichment."""

import json
from pathlib import Path

import pandas as pd

from kalshi_nba.clean.kalshi_parse import (
    enrich_kalshi_markets,
    game_level_kalshi,
    kalshi_label_to_team,
    markets_records_to_frame,
    parse_game_date_from_ticker,
    parse_title_teams,
)


FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "kalshi_markets_sample.json"


def test_parse_title_at_and_vs() -> None:
    visitor, home, fmt = parse_title_teams("Houston at Los Angeles L Winner?")
    assert fmt == "at"
    assert visitor == "Houston"
    assert home == "Los Angeles L"

    a, b, fmt = parse_title_teams("LA vs Phoenix Winner?")
    assert fmt == "vs"
    assert a == "LA"
    assert b == "Phoenix"


def test_parse_game_date_from_ticker() -> None:
    date = parse_game_date_from_ticker("KXNBAGAME-25OCT21HOULAL-HOU")
    assert date == pd.Timestamp("2025-10-21")


def test_kalshi_label_to_team() -> None:
    assert kalshi_label_to_team("Los Angeles L") == "Los Angeles Lakers"
    assert kalshi_label_to_team("Oklahoma City") == "Oklahoma City Thunder"
    assert kalshi_label_to_team("LAC") == "Los Angeles Clippers"


def test_enrich_and_game_level_from_fixture() -> None:
    records = json.loads(FIXTURE.read_text(encoding="utf-8"))
    markets = enrich_kalshi_markets(markets_records_to_frame(records))
    assert len(markets) == 4
    assert markets["yes_team"].isna().sum() == 0
    assert set(markets.loc[markets["title_format"] == "at", "home_team"]) == {
        "Los Angeles Lakers"
    }

    games = game_level_kalshi(markets)
    assert len(games) == 2
    hou_lal = games.loc[games["event_ticker"] == "KXNBAGAME-25OCT21HOULAL"].iloc[0]
    assert hou_lal["total_volume"] == 3000.0
    assert hou_lal["home_team"] == "Los Angeles Lakers"
    assert hou_lal["visitor_team"] == "Houston Rockets"
