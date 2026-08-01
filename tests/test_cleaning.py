"""Tests for team-name conversion, probability calculation, and merging."""

import datetime

import pandas as pd
import pytest

from src.cleaning import (
    implied_probability,
    merge_markets_with_results,
    normalize_team_name,
    parse_kalshi_timestamp,
    settlement_to_outcome,
)


class TestNormalizeTeamName:
    def test_full_names_from_basketball_reference(self):
        assert normalize_team_name("Boston Celtics") == "BOS"
        assert normalize_team_name("Golden State Warriors") == "GSW"
        assert normalize_team_name("Portland Trail Blazers") == "POR"

    def test_city_style_from_kalshi_subtitles(self):
        assert normalize_team_name("Oklahoma City") == "OKC"
        assert normalize_team_name("New Orleans") == "NOP"
        assert normalize_team_name("Golden State") == "GSW"

    def test_nicknames(self):
        assert normalize_team_name("76ers") == "PHI"
        assert normalize_team_name("Trail Blazers") == "POR"
        assert normalize_team_name("Timberwolves") == "MIN"

    def test_case_and_whitespace_insensitive(self):
        assert normalize_team_name("  boston   celtics ") == "BOS"
        assert normalize_team_name("LAKERS") == "LAL"

    def test_canonical_codes_pass_through(self):
        assert normalize_team_name("BOS") == "BOS"
        assert normalize_team_name("lal") == "LAL"

    def test_alternate_abbreviations(self):
        assert normalize_team_name("BRK") == "BKN"
        assert normalize_team_name("PHO") == "PHX"
        assert normalize_team_name("CHO") == "CHA"

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            normalize_team_name("Seattle SuperSonics")

    def test_empty_or_non_string_raises(self):
        with pytest.raises(KeyError):
            normalize_team_name("")
        with pytest.raises(KeyError):
            normalize_team_name(None)

    def test_all_thirty_teams_covered(self):
        full_names = [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans",
            "New York Knicks", "Oklahoma City Thunder", "Orlando Magic",
            "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
            "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz",
            "Washington Wizards",
        ]
        codes = {normalize_team_name(name) for name in full_names}
        assert len(codes) == 30


class TestImpliedProbability:
    def test_midpoint_of_bid_ask_in_cents(self):
        assert implied_probability(58, 62) == pytest.approx(0.60)

    def test_equal_bid_ask(self):
        assert implied_probability(50, 50) == pytest.approx(0.50)

    def test_dollar_scale(self):
        assert implied_probability(0.58, 0.62, price_scale=1) == pytest.approx(0.60)

    def test_extremes_allowed(self):
        assert implied_probability(0, 0) == 0.0
        assert implied_probability(100, 100) == 1.0

    def test_crossed_market_raises(self):
        with pytest.raises(ValueError):
            implied_probability(62, 58)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError):
            implied_probability(-1, 50)

    def test_missing_price_raises(self):
        with pytest.raises(ValueError):
            implied_probability(None, 50)

    def test_out_of_range_probability_raises(self):
        with pytest.raises(ValueError):
            implied_probability(150, 160)


class TestTimestampsAndSettlement:
    def test_parse_iso_string(self):
        ts = parse_kalshi_timestamp("2026-01-15T19:30:00Z")
        assert ts == pd.Timestamp("2026-01-15 19:30:00", tz="UTC")

    def test_parse_unix_seconds(self):
        ts = parse_kalshi_timestamp(1768505400)
        assert ts.tzinfo is not None
        assert ts.year == 2026

    def test_naive_string_assumed_utc(self):
        ts = parse_kalshi_timestamp("2026-01-15 19:30:00")
        assert str(ts.tz) == "UTC"

    def test_settlement_to_outcome(self):
        assert settlement_to_outcome(1.0) == 1
        assert settlement_to_outcome(0.0) == 0
        with pytest.raises(ValueError):
            settlement_to_outcome(0.5)


class TestMergeMarketsWithResults:
    def _frames(self):
        markets = pd.DataFrame(
            {
                "game_date": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)],
                "home_team": ["BOS", "LAL"],
                "away_team": ["NYK", "GSW"],
                "kalshi_prob": [0.65, 0.45],
            }
        )
        schedule = pd.DataFrame(
            {
                "game_date": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 3)],
                "home_team": ["BOS", "MIA"],
                "away_team": ["NYK", "CHI"],
                "home_points": [110, 99],
                "away_points": [104, 101],
            }
        )
        return markets, schedule

    def test_inner_join_keeps_only_matched_games(self):
        markets, schedule = self._frames()
        merged = merge_markets_with_results(markets, schedule)
        assert len(merged) == 1
        assert merged.loc[0, "home_team"] == "BOS"

    def test_home_win_target(self):
        markets, schedule = self._frames()
        merged = merge_markets_with_results(markets, schedule)
        assert merged.loc[0, "home_win"] == 1
