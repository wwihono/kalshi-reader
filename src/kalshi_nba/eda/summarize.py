"""Summary statistics and missingness helpers for EDA."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def dataset_shape(frame: pd.DataFrame) -> dict[str, Any]:
    """Return row/column counts for a dataset."""
    return {"n_rows": int(len(frame)), "n_columns": int(frame.shape[1])}


def missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages."""
    if frame.empty:
        return pd.DataFrame(columns=["column", "n_missing", "pct_missing"])
    n = len(frame)
    rows = []
    for col in frame.columns:
        n_missing = int(frame[col].isna().sum())
        rows.append(
            {
                "column": col,
                "n_missing": n_missing,
                "pct_missing": float(n_missing / n) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("n_missing", ascending=False).reset_index(drop=True)


def has_any_missing(frame: pd.DataFrame) -> bool:
    """Return True if any NA values are present (Pythonic check)."""
    return bool(frame.isna().any().any())


def seven_number_summary(series: pd.Series) -> dict[str, float]:
    """Mean, std, min, Q1, median, Q3, max for a quantitative variable."""
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "q3": float("nan"),
            "max": float("nan"),
            "count": 0.0,
        }
    arr = values.to_numpy()
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q3": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "count": float(len(arr)),
    }


def categorical_summary(series: pd.Series) -> pd.DataFrame:
    """Unique values and counts for a categorical variable."""
    counts = series.fillna("<NA>").astype(str).value_counts(dropna=False)
    return (
        counts.rename_axis("value")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
