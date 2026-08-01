"""Historical trading simulation (RQ4). No real trades are placed.

For each test-period game we compare the model probability with the Kalshi
purchase price. When the estimated edge clears a threshold (5%, 10%, 15%)
we simulate buying one contract and settle it against the actual result,
net of transaction costs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def kalshi_fee(price: float, contracts: int = 1) -> float:
    """Kalshi's trading fee: ceil(0.07 * contracts * P * (1-P)), in dollars.

    ``price`` is the contract price in dollars (0-1). The ceiling is taken
    to the next cent, matching Kalshi's published fee schedule.
    """
    if not 0.0 <= price <= 1.0:
        raise ValueError(f"Price must be in [0, 1], got {price}")
    raw = 0.07 * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0


def simulate_trades(
    df: pd.DataFrame,
    threshold: float,
    model_prob_col: str = "model_prob",
    market_prob_col: str = "kalshi_prob",
    outcome_col: str = "home_win",
    include_fees: bool = True,
) -> pd.DataFrame:
    """Simulate one-contract trades wherever the model edge clears ``threshold``.

    Buys the Yes contract when ``model_prob - price >= threshold`` and the
    No contract when ``price - model_prob >= threshold``. Profit per trade
    is settlement value minus purchase price minus fees. Returns one row
    per simulated trade with ``side``, ``price``, ``edge``, ``fee``, and
    ``profit`` columns (empty frame with those columns when no edge clears
    the threshold).
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    trades = []
    for idx, row in df.iterrows():
        model_p = float(row[model_prob_col])
        market_p = float(row[market_prob_col])
        outcome = int(row[outcome_col])
        edge = model_p - market_p
        if edge >= threshold:
            side, price, settled = "yes", market_p, float(outcome)
        elif -edge >= threshold:
            side, price, settled = "no", 1.0 - market_p, float(1 - outcome)
        else:
            continue
        fee = kalshi_fee(price) if include_fees else 0.0
        trades.append(
            {
                "index": idx,
                "side": side,
                "price": price,
                "edge": abs(edge),
                "fee": fee,
                "profit": settled - price - fee,
            }
        )
    columns = ["index", "side", "price", "edge", "fee", "profit"]
    return pd.DataFrame(trades, columns=columns)


def summarize_simulation(trades: pd.DataFrame) -> dict[str, float]:
    """Aggregate a trade log: positions, total P&L, and return on investment."""
    if trades.empty:
        return {"n_trades": 0, "total_cost": 0.0, "total_profit": 0.0, "roi": float("nan")}
    total_cost = float((trades["price"] + trades["fee"]).sum())
    total_profit = float(trades["profit"].sum())
    return {
        "n_trades": int(len(trades)),
        "total_cost": total_cost,
        "total_profit": total_profit,
        "roi": total_profit / total_cost,
    }


def threshold_sweep(
    df: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.15),
    **kwargs,
) -> pd.DataFrame:
    """Run the simulation for each edge threshold and tabulate the summaries."""
    rows = []
    for threshold in thresholds:
        trades = simulate_trades(df, threshold, **kwargs)
        rows.append({"threshold": threshold, **summarize_simulation(trades)})
    return pd.DataFrame(rows)


def cumulative_profit(trades: pd.DataFrame) -> pd.Series:
    """Running total of profit in trade order, for the cumulative-returns plot."""
    return trades["profit"].cumsum()
