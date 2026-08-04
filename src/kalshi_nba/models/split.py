"""Chronological train / validation / test splitting."""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd


class ChronoSplit(NamedTuple):
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame,
    *,
    date_col: str = "game_date",
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> ChronoSplit:
    """Split rows by time: earliest train, middle validation, latest test.

    Fractions must sum to 1.0 (within a small tolerance). Empty partitions are
    allowed only when the input itself is empty.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {total}")
    if min(train_frac, val_frac, test_frac) < 0:
        raise ValueError("fractions must be non-negative")
    if date_col not in frame.columns:
        raise KeyError(f"missing date column: {date_col}")

    ordered = frame.sort_values(date_col).reset_index(drop=True)
    n = len(ordered)
    if n == 0:
        empty = ordered.copy()
        return ChronoSplit(empty, empty.copy(), empty.copy())

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    # Ensure the final partition receives any remainder from integer truncation
    if test_frac > 0 and val_end >= n and n >= 3:
        val_end = n - 1
    if train_frac > 0 and train_end == 0 and n >= 3:
        train_end = 1
        val_end = max(val_end, 2)

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    return ChronoSplit(train=train, validation=validation, test=test)
