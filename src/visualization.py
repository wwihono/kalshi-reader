"""Plots for the report: calibration, group comparisons, model metrics,
and cumulative simulated returns. Each function saves a PNG to
``results/figures/`` and returns the matplotlib figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd

FIGURES_DIR = Path("results/figures")


def _save(fig: plt.Figure, filename: str, figures_dir: Path = FIGURES_DIR) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def plot_calibration(table: pd.DataFrame, filename: str = "calibration.png") -> plt.Figure:
    """Reliability diagram: average market probability vs observed win rate."""
    fig, ax = plt.subplots(figsize=(6, 6))
    populated = table.dropna(subset=["avg_market_prob", "win_rate"])
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.scatter(populated["avg_market_prob"], populated["win_rate"], zorder=3)
    for _, row in populated.iterrows():
        ax.annotate(f"n={row['n']}", (row["avg_market_prob"], row["win_rate"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Average Kalshi probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Kalshi NBA market calibration")
    ax.legend()
    _save(fig, filename)
    return fig


def plot_grouped_metric(
    grouped: pd.DataFrame,
    group_col: str,
    metric: str = "brier_score",
    filename: str | None = None,
) -> plt.Figure:
    """Bar chart of a market metric across game groups (RQ3)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(grouped[group_col].astype(str), grouped[metric])
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Kalshi {metric.replace('_', ' ')} by {group_col}")
    _save(fig, filename or f"{metric}_by_{group_col}.png")
    return fig


def plot_model_comparison(
    comparison: pd.DataFrame, filename: str = "model_comparison.png"
) -> plt.Figure:
    """Grouped bars comparing models (and Kalshi) on the test metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["accuracy", "brier_score", "log_loss"]):
        ax.bar(comparison.index.astype(str), comparison[metric])
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Test-period performance: models vs Kalshi")
    _save(fig, filename)
    return fig


def plot_cumulative_returns(
    trades_by_threshold: dict[float, pd.DataFrame],
    filename: str = "cumulative_returns.png",
) -> plt.Figure:
    """Cumulative simulated profit over trade sequence, one line per threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for threshold, trades in sorted(trades_by_threshold.items()):
        if trades.empty:
            continue
        ax.plot(
            range(1, len(trades) + 1),
            trades["profit"].cumsum(),
            label=f"edge ≥ {threshold:.0%} ({len(trades)} trades)",
        )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Cumulative profit ($ per 1-contract positions)")
    ax.set_title("Simulated returns by edge threshold")
    ax.legend()
    _save(fig, filename)
    return fig
