"""Tests for chronological splitting, rolling features, rest days, and Elo."""

import datetime

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ELO_INITIAL,
    build_game_features,
    chronological_split,
    compute_elo_ratings,
    rolling_team_features,
)


def _games(rows):
    """rows: (date, home, away, home_pts, away_pts)"""
    return pd.DataFrame(
        rows, columns=["game_date", "home_team", "away_team", "home_points", "away_points"]
    )


class TestChronologicalSplit:
    def _df(self, n=100):
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        return pd.DataFrame({"game_date": dates, "value": range(n)})

    def test_60_20_20_sizes(self):
        train, val, test = chronological_split(self._df(100))
        assert (len(train), len(val), len(test)) == (60, 20, 20)

    def test_no_future_leakage_between_periods(self):
        train, val, test = chronological_split(self._df(100))
        assert train["game_date"].max() <= val["game_date"].min()
        assert val["game_date"].max() <= test["game_date"].min()

    def test_shuffled_input_is_sorted_first(self):
        df = self._df(50).sample(frac=1.0, random_state=7)
        train, val, test = chronological_split(df)
        assert train["game_date"].is_monotonic_increasing
        assert train["game_date"].max() < test["game_date"].min()

    def test_all_rows_preserved_exactly_once(self):
        df = self._df(83)
        train, val, test = chronological_split(df)
        combined = pd.concat([train, val, test])
        assert len(combined) == 83
        assert sorted(combined["value"]) == list(range(83))

    def test_invalid_fractions_raise(self):
        with pytest.raises(ValueError):
            chronological_split(self._df(10), train_frac=0.8, val_frac=0.3)
        with pytest.raises(ValueError):
            chronological_split(self._df(10), train_frac=0.0)


class TestRollingTeamFeatures:
    def _sample(self):
        d = datetime.date
        return _games(
            [
                (d(2026, 1, 1), "BOS", "NYK", 100, 90),   # BOS win +10
                (d(2026, 1, 3), "BOS", "MIA", 95, 100),   # BOS loss -5
                (d(2026, 1, 4), "BOS", "CHI", 105, 100),  # BOS win +5
                (d(2026, 1, 8), "BOS", "LAL", 99, 101),   # BOS loss -2
            ]
        )

    def test_first_game_has_no_history(self):
        log = rolling_team_features(self._sample())
        bos = log[log["team"] == "BOS"].sort_values("game_date")
        assert np.isnan(bos.iloc[0]["win_pct_last5"])
        assert np.isnan(bos.iloc[0]["rest_days"])

    def test_rolling_win_pct_excludes_current_game(self):
        log = rolling_team_features(self._sample())
        bos = log[log["team"] == "BOS"].sort_values("game_date")
        # Before game 2: 1-0. Before game 3: 1-1. Before game 4: 2-1.
        assert bos.iloc[1]["win_pct_last5"] == pytest.approx(1.0)
        assert bos.iloc[2]["win_pct_last5"] == pytest.approx(0.5)
        assert bos.iloc[3]["win_pct_last5"] == pytest.approx(2 / 3)

    def test_rolling_point_diff(self):
        log = rolling_team_features(self._sample())
        bos = log[log["team"] == "BOS"].sort_values("game_date")
        # Before game 3: mean(+10, -5) = 2.5. Before game 4: mean(+10, -5, +5) = 10/3.
        assert bos.iloc[2]["avg_point_diff_last5"] == pytest.approx(2.5)
        assert bos.iloc[3]["avg_point_diff_last5"] == pytest.approx(10 / 3)

    def test_window_caps_history(self):
        d = datetime.date
        rows = []
        for i in range(8):
            home_pts = 110 if i % 2 == 0 else 90  # BOS wins on even i, loses on odd i
            rows.append((d(2026, 1, i + 1), "BOS", "NYK", home_pts, 100))
        log = rolling_team_features(_games(rows), windows=(5,))
        bos = log[log["team"] == "BOS"].sort_values("game_date")
        # Before game 8 (index 7), last 5 games are i=2..6 -> wins at i=2,4,6 -> 3/5.
        assert bos.iloc[7]["win_pct_last5"] == pytest.approx(3 / 5)

    def test_rest_days_and_back_to_back(self):
        log = rolling_team_features(self._sample())
        bos = log[log["team"] == "BOS"].sort_values("game_date")
        assert bos.iloc[1]["rest_days"] == 2
        assert bos.iloc[2]["rest_days"] == 1
        assert bos.iloc[2]["back_to_back"] == 1.0
        assert bos.iloc[3]["rest_days"] == 4
        assert bos.iloc[3]["back_to_back"] == 0.0

    def test_teams_tracked_independently(self):
        d = datetime.date
        games = _games(
            [
                (d(2026, 1, 1), "BOS", "NYK", 100, 90),
                (d(2026, 1, 2), "MIA", "NYK", 100, 110),
                (d(2026, 1, 5), "NYK", "BOS", 105, 100),
            ]
        )
        log = rolling_team_features(games)
        nyk = log[log["team"] == "NYK"].sort_values("game_date")
        # NYK before game 3: lost to BOS, beat MIA -> 0.5
        assert nyk.iloc[2]["win_pct_last5"] == pytest.approx(0.5)


class TestElo:
    def _one_game(self):
        return _games([(datetime.date(2026, 1, 1), "BOS", "NYK", 100, 90)])

    def test_pregame_ratings_start_at_initial(self):
        rated = compute_elo_ratings(self._one_game())
        assert rated.iloc[0]["home_elo_pre"] == ELO_INITIAL
        assert rated.iloc[0]["away_elo_pre"] == ELO_INITIAL

    def test_ratings_are_pregame_not_postgame(self):
        d = datetime.date
        games = _games(
            [
                (d(2026, 1, 1), "BOS", "NYK", 100, 90),
                (d(2026, 1, 2), "BOS", "NYK", 100, 90),
            ]
        )
        rated = compute_elo_ratings(games)
        # Winner's rating rises before the rematch; loser's falls.
        assert rated.iloc[1]["home_elo_pre"] > ELO_INITIAL
        assert rated.iloc[1]["away_elo_pre"] < ELO_INITIAL

    def test_zero_sum_updates(self):
        d = datetime.date
        games = _games(
            [
                (d(2026, 1, 1), "BOS", "NYK", 100, 90),
                (d(2026, 1, 2), "BOS", "NYK", 100, 90),
            ]
        )
        rated = compute_elo_ratings(games)
        # Whatever the winner gained after game 1, the loser lost exactly.
        winner_gain = rated.iloc[1]["home_elo_pre"] - ELO_INITIAL
        loser_loss = rated.iloc[1]["away_elo_pre"] - ELO_INITIAL
        assert winner_gain == pytest.approx(-loser_loss)
        assert winner_gain > 0

    def test_favorite_gains_less_than_underdog(self):
        d = datetime.date
        # BOS beats NYK twice, building a rating gap; then they split results.
        games = _games(
            [
                (d(2026, 1, 1), "BOS", "NYK", 100, 90),
                (d(2026, 1, 2), "BOS", "NYK", 100, 90),
                (d(2026, 1, 3), "BOS", "NYK", 100, 90),
            ]
        )
        rated = compute_elo_ratings(games)
        gain_first = rated.iloc[1]["home_elo_pre"] - rated.iloc[0]["home_elo_pre"]
        gain_second = rated.iloc[2]["home_elo_pre"] - rated.iloc[1]["home_elo_pre"]
        # As BOS becomes the bigger favorite, each win moves its rating less.
        assert gain_second < gain_first


class TestBuildGameFeatures:
    def _season(self, n_games=40):
        """Round-robin-ish schedule among four teams with varied outcomes."""
        rng = np.random.default_rng(0)
        teams = ["BOS", "NYK", "MIA", "CHI"]
        rows = []
        date = datetime.date(2026, 1, 1)
        for i in range(n_games):
            home, away = rng.choice(teams, size=2, replace=False)
            home_pts = int(rng.integers(90, 125))
            away_pts = int(rng.integers(90, 125))
            if home_pts == away_pts:
                home_pts += 1
            rows.append((date + datetime.timedelta(days=i // 2), home, away, home_pts, away_pts))
        return _games(rows)

    def test_produces_diff_columns_and_target(self):
        out = build_game_features(self._season())
        for col in [
            "diff_win_pct_last5",
            "diff_win_pct_last10",
            "diff_avg_point_diff_last5",
            "diff_avg_point_diff_last10",
            "diff_rest_days",
            "diff_back_to_back",
            "diff_elo",
            "home_win",
        ]:
            assert col in out.columns, col

    def test_no_nans_in_model_matrix(self):
        out = build_game_features(self._season())
        diff_cols = [c for c in out.columns if c.startswith("diff_")]
        assert not out[diff_cols].isna().any().any()

    def test_target_matches_scores(self):
        out = build_game_features(self._season())
        expected = (out["home_points"] > out["away_points"]).astype(int)
        assert (out["home_win"] == expected).all()

    def test_diff_is_home_minus_away(self):
        d = datetime.date
        games = _games(
            [
                (d(2026, 1, 1), "BOS", "NYK", 100, 90),  # BOS wins
                (d(2026, 1, 2), "MIA", "CHI", 90, 100),  # CHI wins
                (d(2026, 1, 3), "BOS", "CHI", 100, 99),  # both 1-0 entering
                (d(2026, 1, 4), "NYK", "MIA", 100, 99),  # both 0-1 entering
            ]
        )
        out = build_game_features(games)
        game3 = out[(out["home_team"] == "BOS") & (out["away_team"] == "CHI")].iloc[0]
        assert game3["diff_win_pct_last5"] == pytest.approx(0.0)  # 1.0 - 1.0
        game4 = out[(out["home_team"] == "NYK") & (out["away_team"] == "MIA")].iloc[0]
        assert game4["diff_win_pct_last5"] == pytest.approx(0.0)  # 0.0 - 0.0
