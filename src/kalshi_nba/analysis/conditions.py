"""Conditioned accuracy analysis (RQ3)."""

from __future__ import annotations

import pandas as pd

from kalshi_nba.models.evaluate import brier_score


def group_accuracy(
    frame: pd.DataFrame,
    *,
    group_col: str,
    prob_col: str = "kalshi_prob",
    outcome_col: str = "home_win",
) -> pd.DataFrame:
    """Compute accuracy and Brier score within each group."""
    if group_col not in frame.columns:
        raise KeyError(f"missing group column: {group_col}")

    rows: list[dict] = []
    for key, group in frame.groupby(group_col, dropna=False):
        y = group[outcome_col].to_numpy()
        p = group[prob_col].to_numpy(dtype=float)
        yhat = (p >= 0.5).astype(int)
        rows.append(
            {
                group_col: key,
                "n": len(group),
                "accuracy": float((yhat == y).mean()) if len(group) else float("nan"),
                "brier": brier_score(y, p) if len(group) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def volume_quartiles(frame: pd.DataFrame, volume_col: str = "volume_fp") -> pd.Series:
    """Label rows by market-volume quartile (1=lowest, 4=highest)."""
    return pd.qcut(frame[volume_col], 4, labels=[1, 2, 3, 4], duplicates="drop")
