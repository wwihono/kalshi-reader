"""Lightweight plotting helpers (matplotlib / seaborn)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def plot_probability_hist(
    probabilities: pd.Series, path: str | Path | None = None
) -> plt.Figure:
    """Histogram of final pregame implied home-win probabilities."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(probabilities.dropna(), bins=20, range=(0, 1), edgecolor="white")
    ax.axvline(0.5, color="gray", linestyle="--", label="50/50 game")
    ax.set_xlabel("Kalshi pregame implied home-win probability")
    ax.set_ylabel("Number of games")
    ax.set_title("Distribution of Kalshi pregame home-win probabilities")
    ax.legend()
    _save_or_show(fig, path)
    return fig


def plot_volume_hist(
    volumes: pd.Series, path: str | Path | None = None
) -> plt.Figure:
    """Histogram of per-market contract volume on a log x-axis."""
    positive = volumes.dropna()
    positive = positive[positive > 0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(
        np.log10(positive.min()), np.log10(positive.max()), 30
    )
    ax.hist(positive, bins=bins, edgecolor="white")
    ax.set_xscale("log")
    median = positive.median()
    ax.axvline(median, color="tab:red", linestyle="--", label=f"median = {median:,.0f}")
    ax.set_xlabel("Contracts traded per market (log scale)")
    ax.set_ylabel("Number of markets")
    ax.set_title("Kalshi NBA market volume distribution")
    ax.legend()
    _save_or_show(fig, path)
    return fig


def plot_margin_hist(
    margins: pd.Series, path: str | Path | None = None
) -> plt.Figure:
    """Histogram of home-team winning margins with the home-win share."""
    values = margins.dropna()
    home_win_rate = (values > 0).mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=30, edgecolor="white")
    ax.axvline(0, color="tab:red", linestyle="--", label="tie line (margin = 0)")
    ax.set_xlabel("Home points minus visitor points")
    ax.set_ylabel("Number of games")
    ax.set_title(
        f"Home winning margins, 2025-26 NBA season "
        f"(home wins {home_win_rate:.1%} of games)"
    )
    ax.legend()
    _save_or_show(fig, path)
    return fig


def plot_monthly_games(
    schedule: pd.DataFrame,
    *,
    date_col: str = "game_date",
    home_win_col: str = "home_win",
    path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of games per month with the monthly home-win rate."""
    frame = schedule.copy()
    frame["month"] = pd.to_datetime(frame[date_col]).dt.to_period("M").astype(str)
    grouped = frame.groupby("month").agg(
        n_games=(date_col, "size"), home_win_rate=(home_win_col, "mean")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(grouped.index, grouped["n_games"], color="tab:blue", label="games played")
    ax.set_xlabel("Month")
    ax.set_ylabel("Games played", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(
        grouped.index,
        grouped["home_win_rate"],
        color="tab:red",
        marker="o",
        label="home win rate",
    )
    ax2.set_ylabel("Home win rate", color="tab:red")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Games per month and monthly home-win rate")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.autofmt_xdate()
    _save_or_show(fig, path)
    return fig


def plot_prob_vs_margin(
    merged: pd.DataFrame,
    *,
    prob_col: str = "kalshi_home_prob",
    margin_col: str = "home_margin",
    path: str | Path | None = None,
) -> plt.Figure:
    """Scatter of market probability against realized home margin."""
    frame = merged.dropna(subset=[prob_col, margin_col])
    won = frame[frame[margin_col] > 0]
    lost = frame[frame[margin_col] < 0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(won[prob_col], won[margin_col], s=10, alpha=0.4, label="home win")
    ax.scatter(lost[prob_col], lost[margin_col], s=10, alpha=0.4, label="home loss")
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Kalshi pregame implied home-win probability")
    ax.set_ylabel("Home winning margin (points)")
    ax.set_title("Market confidence vs. realized margin")
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
