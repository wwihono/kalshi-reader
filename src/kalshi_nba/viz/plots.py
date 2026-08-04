"""Lightweight plotting helpers (matplotlib / seaborn)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_or_show(fig: plt.Figure, path: str | Path | None) -> None:
    if path is None:
        plt.show()
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_calibration(calib: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.scatter(calib["avg_prob"], calib["empirical_rate"], s=calib["n"] * 3)
    ax.set_xlabel("Mean Kalshi probability")
    ax.set_ylabel("Empirical win rate")
    ax.set_title("Kalshi calibration")
    ax.legend()
    _save_or_show(fig, path)
    return fig


def plot_model_comparison(metrics: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    melted = metrics.melt(id_vars=["model"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=melted, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("Model vs market metrics")
    _save_or_show(fig, path)
    return fig


def plot_group_metric(
    grouped: pd.DataFrame,
    *,
    group_col: str,
    metric: str = "brier",
    path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=grouped, x=group_col, y=metric, ax=ax)
    ax.set_title(f"{metric} by {group_col}")
    _save_or_show(fig, path)
    return fig


def plot_cumulative_returns(
    series: pd.DataFrame,
    *,
    date_col: str = "game_date",
    path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series[date_col], series["cumulative_profit"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative profit")
    ax.set_title("Simulated cumulative returns")
    fig.autofmt_xdate()
    _save_or_show(fig, path)
    return fig
