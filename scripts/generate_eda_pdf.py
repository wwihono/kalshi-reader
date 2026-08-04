"""Generate reports/eda.pdf from an EDA JSON payload and figure PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO_URL = "https://github.com/wwihono/kalshi-reader"


def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "EDATitle",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=16,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "EDAH1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=13,
            leading=18,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "EDABody",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "EDACaption",
            parent=base["BodyText"],
            fontName="Times-Italic",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "mono": ParagraphStyle(
            "EDAMono",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=9,
            leading=12,
            spaceAfter=6,
        ),
    }
    return styles


def _fmt_seven(stats: dict) -> str:
    return (
        f"mean={stats['mean']:.4g}, std={stats['std']:.4g}, min={stats['min']:.4g}, "
        f"Q1={stats['q1']:.4g}, median={stats['median']:.4g}, Q3={stats['q3']:.4g}, "
        f"max={stats['max']:.4g} (n={int(stats['count'])})"
    )


def _cat_lines(records: list[dict], limit: int = 12) -> str:
    parts = [f"{r['value']}: {r['count']}" for r in records[:limit]]
    if len(records) > limit:
        parts.append(f"... ({len(records) - limit} more)")
    return "; ".join(parts)


def _add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 10)
    page = canvas.getPageNumber()
    canvas.drawCentredString(letter[0] / 2.0, 0.6 * inch, str(page))
    canvas.restoreState()


def build_story(payload: dict, styles) -> list:
    k = payload["kalshi"]
    br = payload["basketball_reference"]
    joined = payload["joined"]
    figs = payload["figures"]
    story: list = []

    story.append(
        Paragraph(
            "Can Data Beat the Market? Predicting NBA Games and Evaluating Kalshi Odds — "
            "Exploratory Data Analysis",
            styles["title"],
        )
    )
    story.append(
        Paragraph(
            f"Author: Winston Wihono<br/>GitHub repository: "
            f'<link href="{REPO_URL}">{REPO_URL}</link>',
            styles["body"],
        )
    )

    story.append(Paragraph("Summary of Research Questions", styles["h1"]))
    story.append(
        Paragraph(
            "1. How accurately do Kalshi’s pregame NBA prices represent the actual probability "
            "of each team winning? I will compare market-implied probabilities shortly before "
            "tip-off with realized game results (calibration by probability bins).",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "2. Can an NBA prediction model outperform Kalshi’s market probabilities? I will "
            "train logistic regression, random forest, and gradient boosting models on "
            "pregame features and compare accuracy, Brier score, and log loss on a "
            "chronological test set.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "3. Under what conditions are Kalshi’s NBA predictions most and least accurate? "
            "I will condition market accuracy and Brier score on volume, probability range, "
            "rest differences, and recent form gaps.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "4. Would differences between the model and Kalshi’s prices produce positive "
            "simulated returns? When model − Kalshi exceeds 5%, 10%, or 15%, I will simulate "
            "buying one contract and report historical ROI (no real-money trading).",
            styles["body"],
        )
    )

    story.append(Paragraph("Motivation", styles["h1"]))
    story.append(
        Paragraph(
            "Prediction markets such as Kalshi let traders buy and sell contracts whose prices "
            "can be read as approximate event probabilities. Those prices reflect information, "
            "liquidity, and opinion—not a guaranteed true probability. NBA games are a strong "
            "test case because teams play often and leave measurable pregame signals (form, "
            "rest, home court, and score differentials). This project evaluates whether Kalshi "
            "NBA prices are calibrated, when they err, and whether models that disagree with "
            "the market retain an edge on unseen games. It does not assume profitable trading "
            "is possible; it tests that claim carefully with held-out chronological evaluation.",
            styles["body"],
        )
    )

    story.append(Paragraph("Dataset", styles["h1"]))
    story.append(
        Paragraph(
            "Kalshi NBA Market Data. Public API markets for series ticker KXNBAGAME are "
            "downloaded from the recent markets endpoint and the historical markets endpoint "
            "(selected using Kalshi’s historical cutoff). Important fields include ticker, "
            "event_ticker, title, yes_sub_title, occurrence_datetime / "
            "expected_expiration_time, volume_fp, result, and settlement_value_dollars. "
            "Candlestick endpoints supply pregame bid/ask history for later modeling; this EDA "
            "focuses on market structure, volume, settlement, and join quality with results.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "NBA Schedule and Results Data. Basketball Reference’s 2025–26 schedule pages "
            "provide game_date, visitor_team, home_team, visitor_pts, and home_pts. Monthly "
            "HTML tables are scraped and concatenated in Python, then de-duplicated because "
            "landing/month pages can repeat early-season games.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            f"How the datasets relate: {joined['relation']}",
            styles["body"],
        )
    )

    story.append(Paragraph("Method", styles["h1"]))
    story.append(
        Paragraph(
            "EDA steps: (1) paginate and save raw Kalshi JSON plus the Basketball Reference "
            "schedule CSV; (2) parse Kalshi titles/tickers into game dates and team labels, "
            "mapping short city names and ticker suffixes to canonical NBA names; "
            "(3) collapse the two Yes-contracts per game into game-level rows; (4) inner-join "
            "to Basketball Reference on date + home/visitor (using unordered team-pair matching "
            "when Kalshi titles use “vs”); (5) compute missingness, seven-number summaries, "
            "and categorical counts for variables tied to the research questions; "
            "(6) produce at least two visualizations for each analysis table (source tables and "
            "the joined table). All steps are implemented as runnable Python modules under "
            "src/kalshi_nba with scripts/run_eda.py as the entrypoint.",
            styles["body"],
        )
    )

    story.append(Paragraph("EDA Results", styles["h1"]))
    story.append(Paragraph("Kalshi markets (contract-level)", styles["h1"]))
    story.append(
        Paragraph(
            f"The Kalshi table has {k['shape']['n_rows']} rows and {k['shape']['n_columns']} "
            f"columns after de-duplicating tickers across recent/historical pulls. "
            f"Rows represent {k['row_meaning']} Columns represent {k['column_meaning']} "
            f"There are {k['n_events']} distinct event_ticker values (games).",
            styles["body"],
        )
    )
    miss_k = [m for m in k["missingness"] if m["n_missing"] > 0][:8]
    if k["has_missing"]:
        detail = "; ".join(
            f"{m['column']}={m['n_missing']} ({100 * m['pct_missing']:.1f}%)" for m in miss_k
        )
        story.append(
            Paragraph(
                f"Missing data is present (confirmed with DataFrame.isna().any().any()). "
                f"Highest-missing fields include: {detail}. Plan: prefer ticker-encoded "
                f"game dates and expected_expiration_time when occurrence_datetime is absent; "
                f"drop non-NBA exhibition contracts that fail team mapping; require complete "
                f"join keys before modeling; obtain pregame probabilities from candlesticks "
                f"rather than post-settlement yes_bid/yes_ask (which collapse to 0/1).",
                styles["body"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No missing values were detected via frame.isna().any().any().",
                styles["body"],
            )
        )

    story.append(
        Paragraph(
            "Variables of interest: volume_fp (liquidity / participation for RQ3), "
            "last_price_dollars and settlement_value_dollars / result (outcome labels for RQ1), "
            "status and title_format (data-quality diagnostics for messy-data challenge). "
            "These columns directly support calibration, conditioning on volume, and merge QA.",
            styles["body"],
        )
    )
    for name, stats in k["quant"].items():
        story.append(Paragraph(f"{name}: {_fmt_seven(stats)}", styles["mono"]))
    for name, records in k["categorical"].items():
        story.append(Paragraph(f"{name}: {_cat_lines(records)}", styles["mono"]))

    if Path(figs["kalshi_volume"]).exists():
        story.append(Image(figs["kalshi_volume"], width=6.3 * inch, height=3.5 * inch))
        story.append(
            Paragraph(
                "Figure 1. Histogram of Kalshi contract volume (log scale).",
                styles["caption"],
            )
        )
        story.append(
            Paragraph(
                "Figure 1 shows volume_fp across Yes-contracts. A log scale is used because "
                "volume spans several orders of magnitude. Readers should take away that "
                "liquidity is generally high but uneven—useful later when conditioning "
                "calibration error on volume quartiles.",
                styles["body"],
            )
        )

    story.append(Paragraph("Basketball Reference results", styles["h1"]))
    story.append(
        Paragraph(
            f"After de-duplication, the schedule table has {br['shape']['n_rows']} rows and "
            f"{br['shape']['n_columns']} columns covering {br['date_min']} to {br['date_max']}. "
            f"Rows represent {br['row_meaning']} Columns represent {br['column_meaning']}",
            styles["body"],
        )
    )
    if br["has_missing"]:
        miss_br = [m for m in br["missingness"] if m["n_missing"] > 0]
        detail = "; ".join(
            f"{m['column']}={m['n_missing']} ({100 * m['pct_missing']:.1f}%)" for m in miss_br
        )
        story.append(
            Paragraph(
                f"Missing values were found with isna(): {detail}. Rows lacking scores would be "
                f"excluded before feature construction; in the current pull, completed games "
                f"are retained after dropping duplicate schedule rows.",
                styles["body"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No missing values remain after de-duplication "
                "(verified with results.isna().any().any() == False).",
                styles["body"],
            )
        )
    story.append(
        Paragraph(
            "Variables of interest: home_pts, visitor_pts, and derived point_margin / home_win. "
            "These define the modeling target (home_win) and support rest/form features built "
            "from prior games for RQ2–RQ4.",
            styles["body"],
        )
    )
    for name, stats in br["quant"].items():
        story.append(Paragraph(f"{name}: {_fmt_seven(stats)}", styles["mono"]))
    for name, records in br["categorical"].items():
        story.append(Paragraph(f"{name}: {_cat_lines(records)}", styles["mono"]))

    if Path(figs["br_home_win_month"]).exists():
        story.append(Image(figs["br_home_win_month"], width=6.3 * inch, height=3.5 * inch))
        story.append(
            Paragraph(
                "Figure 2. Monthly home-team win rate in the 2025–26 Basketball Reference schedule.",
                styles["caption"],
            )
        )
        story.append(
            Paragraph(
                "Figure 2 summarizes the binary home_win target over time. A bar chart by month "
                "is appropriate for a seasonal rate. The takeaway is that home-court advantage "
                "is present but not constant—motivation for including a home indicator and "
                "checking whether Kalshi prices already absorb it.",
                styles["body"],
            )
        )

    story.append(Paragraph("Joined Kalshi–Basketball Reference dataset", styles["h1"]))
    story.append(
        Paragraph(
            f"The joined analysis table has {joined['shape']['n_rows']} rows and "
            f"{joined['shape']['n_columns']} columns. Rows represent {joined['row_meaning']} "
            f"Match rate versus Kalshi game events is "
            f"{100 * joined['match_rate_vs_kalshi_events']:.1f}%; versus BR games "
            f"{100 * joined['match_rate_vs_br_games']:.1f}%. Unmatched Kalshi events include "
            f"non-NBA exhibitions (e.g., Guangzhou) and any remaining label mismatches.",
            styles["body"],
        )
    )
    if joined["has_missing"]:
        miss_j = [m for m in joined["missingness"] if m["n_missing"] > 0][:8]
        detail = "; ".join(
            f"{m['column']}={m['n_missing']} ({100 * m['pct_missing']:.1f}%)" for m in miss_j
        )
        story.append(
            Paragraph(
                f"Missingness on the join (isna()): {detail}. For modeling, only rows with "
                f"complete home_win, teams, and (later) pregame kalshi_prob will be retained.",
                styles["body"],
            )
        )
    else:
        story.append(
            Paragraph(
                "The joined modeling columns checked here have no missing values "
                "(isna().any().any() is False on the retained frame).",
                styles["body"],
            )
        )
    story.append(
        Paragraph(
            "Variables of interest on the join: total_volume (RQ3 liquidity), home_win (RQ1–RQ4 "
            "outcome), home_pts/visitor_pts (feature construction / margin diagnostics), and "
            "title_format (merge-quality monitor). These are the minimum fields needed to "
            "study calibration and volume-conditioned accuracy before model training.",
            styles["body"],
        )
    )
    for name, stats in joined["quant"].items():
        story.append(Paragraph(f"{name}: {_fmt_seven(stats)}", styles["mono"]))
    for name, records in joined["categorical"].items():
        story.append(Paragraph(f"{name}: {_cat_lines(records)}", styles["mono"]))

    if Path(figs["joined_volume_margin"]).exists():
        story.append(Image(figs["joined_volume_margin"], width=6.3 * inch, height=3.5 * inch))
        story.append(
            Paragraph(
                "Figure 3. Scatterplot of total Kalshi volume versus absolute point margin.",
                styles["caption"],
            )
        )
        story.append(
            Paragraph(
                "Figure 3 jointly displays market participation and game competitiveness. A "
                "scatterplot is used because both variables are continuous. The visual suggests "
                "volume is not a simple function of final margin, so volume-based conditioning "
                "in RQ3 is not just a proxy for blowouts.",
                styles["body"],
            )
        )
    if Path(figs["joined_home_by_volume"]).exists():
        story.append(Image(figs["joined_home_by_volume"], width=6.3 * inch, height=3.5 * inch))
        story.append(
            Paragraph(
                "Figure 4. Home win rate by Kalshi total-volume quartile on joined games.",
                styles["caption"],
            )
        )
        story.append(
            Paragraph(
                "Figure 4 is an early look at RQ3-style conditioning: does outcome frequency "
                "shift with liquidity? Bars by quartile communicate group rates clearly. "
                "Rates remain near the seasonal home-win baseline, so later work should "
                "compare Brier/calibration—not only win rate—across the same groups.",
                styles["body"],
            )
        )

    story.append(Paragraph("Challenge Goals Update", styles["h1"]))
    story.append(
        Paragraph(
            "Messy Data (retained): Implemented pagination across Kalshi recent/historical "
            "endpoints, nested JSON flattening, title/ticker parsing for “at” and “vs” formats, "
            "timestamp fallbacks when occurrence_datetime is missing, Basketball Reference HTML "
            "table scraping with duplicate-row removal, and canonical team-name normalization "
            "for join keys. Candlestick pulls for true pregame midpoints are wired in "
            "kalshi_nba.eda.pregame and will be scaled up for modeling.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Machine Learning (unchanged plan): Still targeting logistic regression, random "
            "forest, and gradient boosting with validation-set tuning and one chronological "
            "test evaluation against Kalshi. No models are trained in this EDA deliverable; "
            "feature/split utilities already exist in the repository scaffold.",
            styles["body"],
        )
    )

    story.append(Paragraph("Work Plan Evaluation", styles["h1"]))
    story.append(
        Paragraph(
            "The proposal estimated ~3 hours to download/inspect and ~5 hours to clean/merge. "
            "Those estimates were somewhat optimistic for Kalshi’s split live/historical "
            "candlestick APIs and inconsistent title formats (“at” vs “vs”, truncated city "
            "labels). Collecting markets/results was faster than expected (~minutes once "
            "scripts existed), but parsing/join QA took longer than the raw download step. "
            "Feature engineering (~5h), modeling (~6h), and final visualizations/report polish "
            "(~5h) remain. Updated remaining effort is concentrated on bulk pregame candlestick "
            "collection, leakage-safe features, model tuning, and final report figures—not on "
            "basic schedule scraping.",
            styles["body"],
        )
    )

    story.append(Paragraph("Testing", styles["h1"]))
    story.append(
        Paragraph(
            "EDA code is tested with pytest using small fixture files under data/fixtures and "
            "assert-style unit tests in tests/test_eda.py and tests/test_kalshi_parse.py. Tests "
            "check title/ticker parsing, seven-number summaries, missingness detection via "
            "isna(), game-level collapse to two teams per event, and successful inner joins on "
            "synthetic Kalshi+BR frames. Visual outputs are smoke-checked by running "
            "scripts/run_eda.py end-to-end on the saved raw pull. These tests support trusting "
            "reported counts and join rates because the same helper functions produce both the "
            "fixtures’ expected values and the full-data EDA payload.",
            styles["body"],
        )
    )

    story.append(Paragraph("Collaboration", styles["h1"]))
    story.append(
        Paragraph(
            "No collaborators beyond course staff/team context. Resources consulted: Kalshi "
            "public API documentation (markets, historical cutoff/candlesticks), Basketball "
            "Reference schedule pages, pandas/matplotlib documentation, and the course CSE 163 "
            "code quality guide / flake8 expectations.",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Appendix: EDA numeric payload excerpt", styles["h1"]))
    story.append(
        Paragraph(
            "The machine-readable payload written by scripts/run_eda.py is summarized above; "
            "full JSON is saved under data/processed/eda_summary.json for reproducibility.",
            styles["body"],
        )
    )
    summary_table = [
        ["Dataset", "Rows", "Cols", "Has missing"],
        [
            "Kalshi markets",
            str(k["shape"]["n_rows"]),
            str(k["shape"]["n_columns"]),
            str(k["has_missing"]),
        ],
        [
            "Basketball Reference",
            str(br["shape"]["n_rows"]),
            str(br["shape"]["n_columns"]),
            str(br["has_missing"]),
        ],
        [
            "Joined games",
            str(joined["shape"]["n_rows"]),
            str(joined["shape"]["n_columns"]),
            str(joined["has_missing"]),
        ],
    ]
    table = Table(summary_table, hAlign="LEFT", colWidths=[2.3 * inch, 1 * inch, 1 * inch, 1.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))
    return story


def write_pdf(payload_path: Path, out_path: Path) -> Path:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    styles = _styles()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title="eda",
        author="Winston Wihono",
    )
    doc.build(build_story(payload, styles), onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload",
        type=Path,
        default=Path("data/processed/eda_summary.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/eda.pdf"),
    )
    args = parser.parse_args()
    path = write_pdf(args.payload, args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
