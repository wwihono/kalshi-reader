"""Probability helpers for Kalshi market prices."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def midpoint_probability(yes_bid: float | None, yes_ask: float | None) -> float | None:
    """Return the midpoint of yes_bid and yes_ask as an implied probability.

    Prices are expected on a 0–1 scale (or 0–100 and will be scaled down when
    either value is > 1). Returns None if either side is missing.
    """
    if yes_bid is None or yes_ask is None:
        return None
    if pd.isna(yes_bid) or pd.isna(yes_ask):
        return None

    bid = float(yes_bid)
    ask = float(yes_ask)
    if bid < 0 or ask < 0:
        raise ValueError("bid/ask must be non-negative")
    if ask < bid:
        raise ValueError("yes_ask must be >= yes_bid")

    # Accept either 0–1 or 0–100 market conventions
    if bid > 1 or ask > 1:
        bid /= 100.0
        ask /= 100.0

    if bid > 1 or ask > 1:
        raise ValueError("bid/ask out of valid probability range")

    return (bid + ask) / 2.0


def add_midpoint_probability(
    frame: pd.DataFrame,
    *,
    bid_col: str = "yes_bid",
    ask_col: str = "yes_ask",
    out_col: str = "kalshi_prob",
) -> pd.DataFrame:
    """Attach midpoint implied probabilities to a markets DataFrame."""
    out = frame.copy()
    out[out_col] = [
        midpoint_probability(b, a) for b, a in zip(out[bid_col], out[ask_col], strict=True)
    ]
    return out


def candle_close_values(candle: dict) -> dict[str, float | None]:
    """Extract close-of-candle bid/ask/price from a Kalshi candlestick.

    Handles both schemas returned by the public API: the live endpoint
    uses ``close_dollars`` keys while the historical endpoint uses
    ``close``. Missing sides come back as None.
    """

    def _close(section: object) -> float | None:
        if not isinstance(section, dict):
            return None
        raw = section.get("close_dollars", section.get("close"))
        if raw is None:
            return None
        return float(raw)

    return {
        "yes_bid": _close(candle.get("yes_bid")),
        "yes_ask": _close(candle.get("yes_ask")),
        "price": _close(candle.get("price")),
    }


def pregame_price_from_candles(
    candles: Iterable[dict],
    cutoff_ts: int,
) -> dict[str, float | None] | None:
    """Select the last candle at or before ``cutoff_ts`` and summarize it.

    Returns a dict with end_period_ts, yes_bid, yes_ask, price, and the
    bid/ask midpoint probability, or None when no candle qualifies.
    """
    best: dict | None = None
    for candle in candles:
        ts = candle.get("end_period_ts")
        if ts is None or ts > cutoff_ts:
            continue
        if best is None or ts > best["end_period_ts"]:
            best = candle
    if best is None:
        return None

    values = candle_close_values(best)
    midpoint = midpoint_probability(values["yes_bid"], values["yes_ask"])
    return {
        "end_period_ts": int(best["end_period_ts"]),
        "yes_bid": values["yes_bid"],
        "yes_ask": values["yes_ask"],
        "price": values["price"],
        "midpoint": midpoint,
    }


def brier_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    """Mean squared error between outcomes and predicted probabilities."""
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_prob), dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if yt.size == 0:
        raise ValueError("inputs must be non-empty")
    return float(np.mean((yp - yt) ** 2))
