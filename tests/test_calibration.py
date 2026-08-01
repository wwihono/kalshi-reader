"""Tests for calibration binning and grouped market performance."""

import numpy as np
import pandas as pd
import pytest

from src.calibration import add_quartile_column, calibration_table, grouped_market_performance


class TestCalibrationTable:
    def test_default_decile_bins(self):
        table = calibration_table(np.array([0.05, 0.55, 0.95]), np.array([0, 1, 1]))
        assert len(table) == 10
        assert table["n"].sum() == 3

    def test_bin_assignment_and_win_rate(self):
        probs = np.array([0.62, 0.65, 0.68, 0.61])
        outcomes = np.array([1, 1, 0, 1])
        table = calibration_table(probs, outcomes)
        row = table[np.isclose(table["bin_low"], 0.6)].iloc[0]
        assert row["n"] == 4
        assert row["avg_market_prob"] == pytest.approx(0.64)
        assert row["win_rate"] == pytest.approx(0.75)

    def test_perfectly_calibrated_market_has_small_gaps(self):
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.05, 0.95, size=20000)
        outcomes = (rng.random(20000) < probs).astype(float)
        table = calibration_table(probs, outcomes).dropna()
        assert (table["calibration_gap"].abs() < 0.03).all()

    def test_probability_of_one_lands_in_top_bin(self):
        table = calibration_table(np.array([1.0]), np.array([1]))
        assert table.iloc[-1]["n"] == 1

    def test_empty_bins_are_nan_not_error(self):
        table = calibration_table(np.array([0.55]), np.array([1]))
        assert np.isnan(table.iloc[0]["avg_market_prob"])


class TestGroupedPerformance:
    def test_metrics_per_group(self):
        df = pd.DataFrame(
            {
                "bucket": ["low", "low", "high", "high"],
                "kalshi_prob": [0.6, 0.7, 0.9, 0.8],
                "home_win": [0, 1, 1, 1],
            }
        )
        grouped = grouped_market_performance(df, "bucket").set_index("bucket")
        assert grouped.loc["high", "accuracy"] == 1.0
        assert grouped.loc["low", "accuracy"] == 0.5
        assert grouped.loc["high", "brier_score"] < grouped.loc["low", "brier_score"]

    def test_add_quartile_column(self):
        df = pd.DataFrame({"volume": range(100)})
        out = add_quartile_column(df, "volume", "volume_quartile")
        counts = out["volume_quartile"].value_counts()
        assert set(counts.index) == {"Q1", "Q2", "Q3", "Q4"}
        assert (counts == 25).all()
