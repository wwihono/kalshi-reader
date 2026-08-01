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


def brier_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    """Mean squared error between outcomes and predicted probabilities."""
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_prob), dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if yt.size == 0:
        raise ValueError("inputs must be non-empty")
    return float(np.mean((yp - yt) ** 2))
