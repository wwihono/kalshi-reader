# Can Data Beat the Market? Predicting NBA Games and Evaluating Kalshi Odds

Author: Winston Wihono

This project evaluates how well Kalshi's pregame NBA market prices reflect true win
probabilities, and whether machine-learning models built from pregame features can
outperform the market.

## Research Questions

1. **Kalshi calibration** — Do contracts priced near, say, 60% win about 60% of the time?
2. **Model performance** — Can logistic regression / random forest / gradient boosting
   models beat Kalshi's probabilities on accuracy, Brier score, and log loss?
3. **Conditions affecting accuracy** — Is the market more or less accurate depending on
   volume, probability range, home-court status, rest differences, and recent form?
4. **Simulated returns** — Would trading on model-vs-market disagreements of at least
   5%, 10%, or 15% have produced positive simulated returns (no real trades)?

## Data Sources

- **Kalshi NBA game-winner markets** (`KXNBAGAME` series) via Kalshi's public
  trade API — live markets, archived/historical markets, and one-minute candlesticks.
  The final available price at least 15 minutes before tipoff is used so no live-game
  information leaks into the analysis.
- **Basketball Reference** 2025–26 NBA schedule and results
  (https://www.basketball-reference.com/leagues/NBA_2026_games.html), collected
  programmatically from the monthly schedule tables.

Raw API responses are saved under `data/raw/` and merged/processed tables under
`data/processed/`.

## Repository Structure

```
├── src/                      # Analysis package
│   ├── data_collection.py    # Kalshi API (pagination, cutoff, candlesticks) + Basketball Reference scraping
│   ├── cleaning.py           # Team-name normalization, timestamps, probability calc, merging
│   ├── features.py           # Rolling win %, point diff, Elo, rest days, chronological split
│   ├── modeling.py           # Model tuning/selection and evaluation metrics (accuracy, Brier, log loss)
│   ├── calibration.py        # Calibration binning and conditional-accuracy grouping (RQ1, RQ3)
│   ├── simulation.py         # Threshold-based trading simulation with fees (RQ4)
│   └── visualization.py      # Calibration plots, comparison charts, cumulative returns
├── scripts/
│   └── run_pipeline.py       # End-to-end pipeline: collect → clean → features → model → simulate
├── tests/                    # Pytest suite for the important pure functions
├── data/
│   ├── raw/                  # Saved API responses / scraped HTML tables (gitignored)
│   └── processed/            # Cleaned and merged datasets (gitignored)
├── results/
│   └── figures/              # Generated plots (gitignored)
├── requirements.txt
└── pyproject.toml
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Tests

The test suite covers the functions the spec calls out as important: team-name
conversion, probability calculation, chronological splitting, and rolling-feature
construction — plus Elo updates, evaluation metrics, calibration binning, and the
trading simulation.

```bash
pytest
```

## Running the Pipeline

```bash
python scripts/run_pipeline.py --help
```

The pipeline downloads Kalshi markets (following pagination cursors and the
live/historical cutoff), scrapes the Basketball Reference schedule, merges the two
sources on date + home team + visiting team, builds leakage-free pregame features,
trains and tunes models on a chronological 60/20/20 split, compares them with Kalshi
on the untouched test period, and simulates threshold-based trading. Network access
to Kalshi and Basketball Reference is required for the collection step; all later
steps run from files saved in `data/`.

## Method Summary

- **Kalshi probability**: midpoint of the final pregame `yes_bid`/`yes_ask` (at least
  15 minutes before tipoff), interpreted as the market's win probability for the Yes team.
- **Features** (computed only from previously completed games): win % over the last 5
  and 10 games, average point differential over the last 5 and 10 games, Elo-style
  rating, rest days, back-to-back indicator, home court. Model columns are
  home-minus-visitor differences; the target is 1 when the home team wins.
- **Split**: chronological — earliest 60% train, next 20% validation, latest 20% test.
  Models are tuned on validation and evaluated once on the test period, with Kalshi
  scored on exactly the same test games.
- **Simulation**: buy one contract when `model probability − Kalshi price` clears a
  5/10/15% threshold; profit = settlement value − purchase price − transaction costs.
