"""Kalshi calibration analysis helpers (RQ1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kalshi_nba.models.evaluate import classification_metrics


def calibration_table(
    probabilities: pd.Series | np.ndarray | list[float],
    outcomes: pd.Series | np.ndarray | list[float],
    *,
    bin_width: float = 0.10,
) -> pd.DataFrame:
    """Bin probabilities and compare mean forecast to empirical win rate."""
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if probs.shape != y.shape:
        raise ValueError("probabilities and outcomes must align")
    if bin_width <= 0 or bin_width > 1:
        raise ValueError("bin_width must be in (0, 1]")

    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    # Ensure 1.0 is included as an edge despite float drift
    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0)

    bins = pd.cut(probs, bins=edges, include_lowest=True, right=True)
    frame = pd.DataFrame({"prob": probs, "outcome": y, "bin": bins})
    grouped = (
        frame.groupby("bin", observed=False)
        .agg(
            n=("outcome", "size"),
            avg_prob=("prob", "mean"),
            empirical_rate=("outcome", "mean"),
        )
        .reset_index()
    )
    grouped["abs_error"] = (grouped["avg_prob"] - grouped["empirical_rate"]).abs()
    return grouped


def overall_market_metrics(
    probabilities: pd.Series | np.ndarray | list[float],
    outcomes: pd.Series | np.ndarray | list[float],
) -> dict[str, float]:
    return classification_metrics(outcomes, probabilities)
