"""Tests for pregame price selection from Kalshi candlesticks."""

import pytest

from kalshi_nba.clean.probability import (
    candle_close_values,
    pregame_price_from_candles,
)


def _live_candle(ts: int, bid: float, ask: float) -> dict:
    return {
        "end_period_ts": ts,
        "yes_bid": {"close_dollars": f"{bid:.4f}"},
        "yes_ask": {"close_dollars": f"{ask:.4f}"},
        "price": {"close_dollars": f"{(bid + ask) / 2:.4f}"},
    }


def _historical_candle(ts: int, bid: float, ask: float) -> dict:
    return {
        "end_period_ts": ts,
        "yes_bid": {"close": f"{bid:.4f}"},
        "yes_ask": {"close": f"{ask:.4f}"},
        "price": {"close": f"{(bid + ask) / 2:.4f}"},
    }


def test_candle_close_values_live_schema():
    values = candle_close_values(_live_candle(100, 0.33, 0.34))
    assert values["yes_bid"] == pytest.approx(0.33)
    assert values["yes_ask"] == pytest.approx(0.34)


def test_candle_close_values_historical_schema():
    values = candle_close_values(_historical_candle(100, 0.72, 0.73))
    assert values["yes_bid"] == pytest.approx(0.72)
    assert values["yes_ask"] == pytest.approx(0.73)


def test_candle_close_values_missing_sections():
    assert candle_close_values({"end_period_ts": 1}) == {
        "yes_bid": None,
        "yes_ask": None,
        "price": None,
    }


def test_pregame_price_selects_last_candle_before_cutoff():
    candles = [
        _live_candle(100, 0.40, 0.42),
        _live_candle(200, 0.50, 0.52),
        _live_candle(300, 0.60, 0.62),  # after cutoff: must be ignored
    ]
    selected = pregame_price_from_candles(candles, cutoff_ts=250)
    assert selected is not None
    assert selected["end_period_ts"] == 200
    assert selected["midpoint"] == pytest.approx(0.51)


def test_pregame_price_handles_unordered_candles():
    candles = [
        _historical_candle(200, 0.50, 0.52),
        _historical_candle(100, 0.40, 0.42),
    ]
    selected = pregame_price_from_candles(candles, cutoff_ts=1000)
    assert selected is not None
    assert selected["end_period_ts"] == 200


def test_pregame_price_returns_none_when_all_after_cutoff():
    candles = [_live_candle(500, 0.4, 0.5)]
    assert pregame_price_from_candles(candles, cutoff_ts=250) is None


def test_pregame_price_returns_none_for_empty_input():
    assert pregame_price_from_candles([], cutoff_ts=250) is None
