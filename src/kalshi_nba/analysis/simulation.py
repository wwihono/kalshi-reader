"""Simulated trading when model and Kalshi disagree (RQ4)."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def simulate_threshold_trades(
    frame: pd.DataFrame,
    *,
    model_prob_col: str = "model_prob",
    kalshi_price_col: str = "kalshi_prob",
    outcome_col: str = "yes_won",
    thresholds: Iterable[float] = (0.05, 0.10, 0.15),
    fee: float = 0.0,
) -> pd.DataFrame:
    """Simulate buying one Yes contract when model - price >= threshold.

    Profit for a purchased contract is ``settlement - price - fee``, where
    settlement is 1 if the Yes side wins else 0.
    """
    summaries: list[dict] = []
    for threshold in thresholds:
        edge = frame[model_prob_col] - frame[kalshi_price_col]
        selected = frame.loc[edge >= threshold].copy()
        if selected.empty:
            summaries.append(
                {
                    "threshold": threshold,
                    "n_trades": 0,
                    "total_profit": 0.0,
                    "roi": 0.0,
                }
            )
            continue

        settlement = selected[outcome_col].astype(float)
        price = selected[kalshi_price_col].astype(float)
        profit = settlement - price - fee
        stake = float(price.sum())
        total_profit = float(profit.sum())
        summaries.append(
            {
                "threshold": threshold,
                "n_trades": int(len(selected)),
                "total_profit": total_profit,
                "roi": (total_profit / stake) if stake else 0.0,
            }
        )
    return pd.DataFrame(summaries)


def cumulative_profit_series(
    frame: pd.DataFrame,
    *,
    model_prob_col: str = "model_prob",
    kalshi_price_col: str = "kalshi_prob",
    outcome_col: str = "yes_won",
    threshold: float = 0.05,
    fee: float = 0.0,
    date_col: str = "game_date",
) -> pd.DataFrame:
    """Return cumulative simulated profit over time for one threshold."""
    ordered = frame.sort_values(date_col).copy()
    edge = ordered[model_prob_col] - ordered[kalshi_price_col]
    traded = ordered.loc[edge >= threshold].copy()
    if traded.empty:
        return pd.DataFrame(columns=[date_col, "profit", "cumulative_profit"])

    settlement = traded[outcome_col].astype(float)
    price = traded[kalshi_price_col].astype(float)
    traded["profit"] = settlement - price - fee
    traded["cumulative_profit"] = traded["profit"].cumsum()
    return traded[[date_col, "profit", "cumulative_profit"]].reset_index(drop=True)
