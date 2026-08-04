"""Run the full EDA: load raw data, clean, summarize, and plot.

Outputs:
- ``data/processed/``: cleaned Kalshi event table, schedule with the
  home_win target, and the merged per-game table used by later parts.
- ``reports/eda_summary.txt``: dataset sizes, missingness tables,
  seven-number summaries, and categorical counts quoted in the report.
- ``reports/figures/``: the figures embedded in the EDA report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from kalshi_nba.analysis.calibration import calibration_table
from kalshi_nba.eda import (
    build_event_table,
    categorical_summary,
    load_pregame_prices,
    load_raw_markets,
    load_schedule,
    merge_events_and_results,
    missingness_summary,
    seven_number_summary,
)
from kalshi_nba.viz.plots import (
    plot_calibration,
    plot_margin_hist,
    plot_monthly_games,
    plot_prob_vs_margin,
    plot_probability_hist,
    plot_volume_hist,
)

matplotlib.use("Agg")

KALSHI_QUANT_VARS = [
    "kalshi_home_prob",
    "volume_fp",
    "settlement_value_dollars",
]
BR_QUANT_VARS = ["home_pts", "visitor_pts", "home_margin"]


def _section(lines: list[str], title: str) -> None:
    lines.append("")
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)


def _frame(lines: list[str], frame: pd.DataFrame) -> None:
    lines.append(frame.to_string())


def summarize_kalshi(
    lines: list[str], markets: pd.DataFrame, events: pd.DataFrame
) -> None:
    """EDA answers for the Kalshi market dataset."""
    _section(lines, "KALSHI MARKET DATASET (raw markets -> tidy frame)")
    lines.append(
        f"Markets (rows): {len(markets)} | columns: {markets.shape[1]}"
    )
    lines.append(
        f"Events (games): {markets['event_ticker'].nunique()} "
        f"| home-side event rows: {len(events)}"
    )
    lines.append("\n-- Missing data (tidy market frame) --")
    _frame(lines, missingness_summary(markets))
    lines.append("\n-- Missing data (per-event table incl. pregame prices) --")
    _frame(lines, missingness_summary(events))
    lines.append("\n-- Seven-number summary: variables of interest --")
    _frame(lines, seven_number_summary(events, KALSHI_QUANT_VARS))
    lines.append("\n-- Categorical: market result --")
    _frame(lines, categorical_summary(events, "result"))
    lines.append("\n-- Categorical: market status --")
    _frame(lines, categorical_summary(events, "status"))


def summarize_schedule(lines: list[str], schedule: pd.DataFrame) -> None:
    """EDA answers for the Basketball Reference dataset."""
    _section(lines, "BASKETBALL REFERENCE SCHEDULE DATASET")
    lines.append(f"Rows: {len(schedule)} | columns: {schedule.shape[1]}")
    lines.append(f"Date range: {schedule['game_date'].min().date()} "
                 f"to {schedule['game_date'].max().date()}")
    lines.append("\n-- Missing data --")
    _frame(lines, missingness_summary(schedule))
    lines.append(f"Duplicated rows: {schedule.duplicated().sum()}")
    frame = schedule.copy()
    frame["home_margin"] = frame["home_pts"] - frame["visitor_pts"]
    lines.append("\n-- Seven-number summary: variables of interest --")
    _frame(lines, seven_number_summary(frame, BR_QUANT_VARS))
    lines.append("\n-- Categorical: home_win --")
    frame["home_win"] = (frame["home_pts"] > frame["visitor_pts"]).astype(int)
    _frame(lines, categorical_summary(frame, "home_win"))
    lines.append("\n-- Categorical: home_team (games hosted) --")
    _frame(lines, categorical_summary(frame, "home_team"))


def summarize_merged(lines: list[str], merged: pd.DataFrame) -> None:
    """EDA answers for the merged Kalshi + results table."""
    _section(lines, "MERGED DATASET (Kalshi events x Basketball Reference)")
    lines.append(f"Rows: {len(merged)} | columns: {merged.shape[1]}")
    lines.append("\n-- Missing data --")
    _frame(lines, missingness_summary(merged))
    agree = (merged["kalshi_home_win"] == merged["home_win"]).mean()
    lines.append(
        f"\nKalshi settlement vs. Basketball Reference winner agreement: "
        f"{agree:.4%}"
    )
    quant = ["kalshi_home_prob", "volume_fp", "home_margin"]
    lines.append("\n-- Seven-number summary: variables of interest --")
    _frame(lines, seven_number_summary(merged, quant))
    with_prob = merged.dropna(subset=["kalshi_home_prob"])
    lines.append(
        f"\nGames with a pregame probability: {len(with_prob)} "
        f"({len(with_prob) / len(merged):.1%})"
    )
    lines.append("\n-- Calibration preview (10% bins) --")
    _frame(
        lines,
        calibration_table(
            with_prob["kalshi_home_prob"], with_prob["home_win"]
        ),
    )


def make_figures(
    fig_dir: Path,
    events: pd.DataFrame,
    schedule: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    """Write all report figures to fig_dir."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_probability_hist(
        events["kalshi_home_prob"], fig_dir / "fig1_kalshi_prob_hist.png"
    )
    plot_volume_hist(events["volume_fp"], fig_dir / "fig2_kalshi_volume_hist.png")

    frame = schedule.copy()
    frame["home_margin"] = frame["home_pts"] - frame["visitor_pts"]
    frame["home_win"] = (frame["home_margin"] > 0).astype(int)
    plot_margin_hist(frame["home_margin"], fig_dir / "fig3_br_margin_hist.png")
    plot_monthly_games(frame, path=fig_dir / "fig4_br_monthly_games.png")

    with_prob = merged.dropna(subset=["kalshi_home_prob"])
    calib = calibration_table(
        with_prob["kalshi_home_prob"], with_prob["home_win"]
    )
    plot_calibration(calib, fig_dir / "fig5_calibration_preview.png")
    plot_prob_vs_margin(merged, path=fig_dir / "fig6_prob_vs_margin.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    markets = load_raw_markets(args.raw_dir)
    pregame = load_pregame_prices(args.raw_dir)
    schedule = load_schedule(args.raw_dir)
    events = build_event_table(markets, pregame)
    merged = merge_events_and_results(events, schedule)

    lines: list[str] = ["EDA summary (auto-generated by scripts/run_eda.py)"]
    summarize_kalshi(lines, markets, events)
    summarize_schedule(lines, schedule)
    summarize_merged(lines, merged)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.processed_dir / "kalshi_events.csv", index=False)
    merged.to_csv(args.processed_dir / "merged_games.csv", index=False)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.reports_dir / "eda_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    make_figures(args.reports_dir / "figures", events, schedule, merged)

    print(f"Wrote {summary_path}")
    print(f"Wrote processed tables to {args.processed_dir}")
    print(f"Wrote figures to {args.reports_dir / 'figures'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
