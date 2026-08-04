"""EDA visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def plot_volume_distribution(
    markets: pd.DataFrame,
    path: str | Path,
    *,
    volume_col: str = "volume_fp",
) -> Path:
    """Histogram of Kalshi contract volume (log scale)."""
    values = pd.to_numeric(markets[volume_col], errors="coerce").dropna()
    values = values[values > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=40, color="#1f4e79", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Contract volume (log scale)")
    ax.set_ylabel("Number of contracts")
    ax.set_title("Distribution of Kalshi KXNBAGAME Contract Volume")
    fig.text(
        0.5,
        -0.05,
        "Caption: Most NBA game contracts show high traded volume, with a long right tail.",
        ha="center",
        fontsize=9,
    )
    return _save(fig, path)


def plot_home_win_rate_by_month(
    results: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Bar chart of monthly home-win rate in Basketball Reference results."""
    frame = results.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    if "home_win" not in frame.columns:
        frame["home_win"] = (frame["home_pts"] > frame["visitor_pts"]).astype(int)
    frame["month"] = frame["game_date"].dt.to_period("M").astype(str)
    summary = frame.groupby("month", as_index=False)["home_win"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=summary, x="month", y="home_win", color="#c45c26", ax=ax)
    ax.axhline(0.5, color="gray", linestyle="--", label="50% baseline")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Home win rate")
    ax.set_title("Monthly Home-Team Win Rate (2025–26 NBA Season)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.text(
        0.5,
        -0.08,
        "Caption: Home teams win more than half of games in most months of the season.",
        ha="center",
        fontsize=9,
    )
    return _save(fig, path)


def plot_joined_volume_vs_margin(
    joined: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Scatter of total Kalshi volume against absolute point margin."""
    frame = joined.copy()
    frame["abs_margin"] = (frame["home_pts"] - frame["visitor_pts"]).abs()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(
        frame["abs_margin"],
        frame["total_volume"],
        alpha=0.35,
        s=18,
        color="#2f6f4e",
        label="Joined games",
    )
    ax.set_xlabel("Absolute point margin")
    ax.set_ylabel("Total Kalshi volume (both contracts)")
    ax.set_yscale("log")
    ax.set_title("Kalshi Volume vs Game Margin (Joined Dataset)")
    ax.legend()
    fig.text(
        0.5,
        -0.05,
        "Caption: Traded volume varies widely at every margin; blowouts are not clearly thinner.",
        ha="center",
        fontsize=9,
    )
    return _save(fig, path)


def plot_joined_home_win_by_volume_quartile(
    joined: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Home-win rate across Kalshi volume quartiles on the joined data."""
    frame = joined.copy()
    frame = frame.dropna(subset=["total_volume", "home_win"])
    frame["volume_quartile"] = pd.qcut(
        frame["total_volume"],
        q=4,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
    )
    summary = frame.groupby("volume_quartile", observed=True)["home_win"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=summary, x="volume_quartile", y="home_win", color="#1f4e79", ax=ax)
    ax.axhline(0.5, color="gray", linestyle="--", label="50% baseline")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Kalshi volume quartile")
    ax.set_ylabel("Home win rate")
    ax.set_title("Home Win Rate by Kalshi Volume Quartile")
    ax.legend()
    fig.text(
        0.5,
        -0.05,
        "Caption: Home-win rates stay near the seasonal baseline across volume quartiles.",
        ha="center",
        fontsize=9,
    )
    return _save(fig, path)
