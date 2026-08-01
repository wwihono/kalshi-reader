"""Calibration analysis (RQ1) and conditional accuracy grouping (RQ3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling import brier_score


def calibration_table(
    probs: np.ndarray,
    outcomes: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bin market probabilities and compare with observed win rates.

    Default bins are the deciles 0.0-0.1, ..., 0.9-1.0. For each bin the
    table reports the number of contracts, the average quoted probability,
    and the fraction that settled at $1 — a well-calibrated market shows
    the last two closely matching.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if bin_edges is None:
        bin_edges = np.linspace(0.0, 1.0, 11)
    # Index of the bin each probability falls into; the top edge is inclusive.
    idx = np.clip(np.digitize(probs, bin_edges, right=False) - 1, 0, len(bin_edges) - 2)
    rows = []
    for b in range(len(bin_edges) - 1):
        mask = idx == b
        rows.append(
            {
                "bin_low": bin_edges[b],
                "bin_high": bin_edges[b + 1],
                "n": int(mask.sum()),
                "avg_market_prob": float(probs[mask].mean()) if mask.any() else np.nan,
                "win_rate": float(outcomes[mask].mean()) if mask.any() else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table["calibration_gap"] = table["win_rate"] - table["avg_market_prob"]
    return table


def grouped_market_performance(
    df: pd.DataFrame,
    group_col: str,
    prob_col: str = "kalshi_prob",
    outcome_col: str = "home_win",
) -> pd.DataFrame:
    """Accuracy and Brier score of the market within each group (RQ3).

    ``group_col`` can hold any precomputed grouping: volume quartiles,
    probability ranges, rest-difference buckets, home-court status, or
    recent-form difference buckets.
    """
    rows = []
    for group, sub in df.groupby(group_col, observed=True):
        probs = sub[prob_col].to_numpy(dtype=float)
        outcomes = sub[outcome_col].to_numpy(dtype=float)
        preds = (probs > 0.5).astype(float)
        rows.append(
            {
                group_col: group,
                "n": len(sub),
                "accuracy": float(np.mean(preds == outcomes)),
                "brier_score": brier_score(outcomes, probs),
            }
        )
    return pd.DataFrame(rows)


def add_quartile_column(df: pd.DataFrame, source_col: str, new_col: str) -> pd.DataFrame:
    """Label rows by quartile of ``source_col`` (e.g. market volume)."""
    out = df.copy()
    out[new_col] = pd.qcut(out[source_col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    return out
