"""Tests for the threshold-based trading simulation and fee model."""

import numpy as np
import pandas as pd
import pytest

from src.simulation import (
    cumulative_profit,
    kalshi_fee,
    simulate_trades,
    summarize_simulation,
    threshold_sweep,
)


class TestKalshiFee:
    def test_rounds_up_to_next_cent(self):
        # 0.07 * 0.5 * 0.5 = 0.0175 -> $0.02
        assert kalshi_fee(0.5) == pytest.approx(0.02)

    def test_cheaper_at_extremes(self):
        assert kalshi_fee(0.05) < kalshi_fee(0.5)
        assert kalshi_fee(0.0) == 0.0
        assert kalshi_fee(1.0) == 0.0

    def test_invalid_price_raises(self):
        with pytest.raises(ValueError):
            kalshi_fee(1.5)


def _test_frame():
    return pd.DataFrame(
        {
            "model_prob": [0.70, 0.40, 0.52, 0.90],
            "kalshi_prob": [0.55, 0.60, 0.50, 0.88],
            "home_win": [1, 0, 1, 0],
        }
    )


class TestSimulateTrades:
    def test_only_edges_above_threshold_trade(self):
        trades = simulate_trades(_test_frame(), threshold=0.10)
        # Edges: +0.15, -0.20, +0.02, +0.02 -> two trades.
        assert len(trades) == 2
        assert list(trades["side"]) == ["yes", "no"]

    def test_yes_trade_profit_when_team_wins(self):
        trades = simulate_trades(_test_frame(), threshold=0.10, include_fees=False)
        yes = trades[trades["side"] == "yes"].iloc[0]
        # Bought Yes at 0.55, settled at $1.
        assert yes["profit"] == pytest.approx(0.45)

    def test_no_trade_profit_when_team_loses(self):
        trades = simulate_trades(_test_frame(), threshold=0.10, include_fees=False)
        no = trades[trades["side"] == "no"].iloc[0]
        # Bought No at 1 - 0.60 = 0.40; home team lost, so No settled at $1.
        assert no["profit"] == pytest.approx(0.60)

    def test_fees_reduce_profit(self):
        with_fees = simulate_trades(_test_frame(), threshold=0.10)
        without = simulate_trades(_test_frame(), threshold=0.10, include_fees=False)
        assert (with_fees["profit"] < without["profit"]).all()

    def test_losing_trade_costs_price_plus_fee(self):
        df = pd.DataFrame({"model_prob": [0.80], "kalshi_prob": [0.60], "home_win": [0]})
        trades = simulate_trades(df, threshold=0.10)
        assert trades.iloc[0]["profit"] == pytest.approx(-0.60 - trades.iloc[0]["fee"])

    def test_no_qualifying_trades_returns_empty_frame(self):
        trades = simulate_trades(_test_frame(), threshold=0.50)
        assert trades.empty
        assert "profit" in trades.columns

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            simulate_trades(_test_frame(), threshold=0.0)


class TestSummaries:
    def test_summarize_counts_and_roi(self):
        trades = simulate_trades(_test_frame(), threshold=0.10, include_fees=False)
        summary = summarize_simulation(trades)
        assert summary["n_trades"] == 2
        assert summary["total_profit"] == pytest.approx(0.45 + 0.60)
        assert summary["roi"] == pytest.approx((0.45 + 0.60) / (0.55 + 0.40))

    def test_empty_summary(self):
        summary = summarize_simulation(simulate_trades(_test_frame(), threshold=0.50))
        assert summary["n_trades"] == 0
        assert np.isnan(summary["roi"])

    def test_threshold_sweep_covers_spec_thresholds(self):
        sweep = threshold_sweep(_test_frame())
        assert list(sweep["threshold"]) == [0.05, 0.10, 0.15]
        # Higher thresholds can never produce more trades.
        assert sweep["n_trades"].is_monotonic_decreasing

    def test_cumulative_profit_running_total(self):
        trades = simulate_trades(_test_frame(), threshold=0.10, include_fees=False)
        cum = cumulative_profit(trades)
        assert cum.iloc[-1] == pytest.approx(trades["profit"].sum())
