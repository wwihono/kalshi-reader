# Kalshi NBA Odds Evaluation

Data-driven analysis of Kalshi NBA game-winner markets (`KXNBAGAME`): calibration of pregame prices, ML models that try to beat the market, conditions where the market is strongest/weakest, and simulated returns when model and market disagree.

Spec: [`docs/Evaluating_Kalshi_Odds.pdf`](docs/Evaluating_Kalshi_Odds.pdf)

## Research questions

1. **Calibration** — Do Kalshi pregame prices match realized win rates?
2. **Model vs market** — Can logistic regression / random forest / gradient boosting beat Kalshi on accuracy, Brier score, and log loss?
3. **Conditions** — How do volume, probability range, rest, and form gaps affect market accuracy?
4. **Simulated returns** — When model − Kalshi ≥ 5% / 10% / 15%, do simulated contract purchases show positive historical ROI?

## Repository layout

```text
src/kalshi_nba/
  collect/      # Kalshi API + Basketball Reference downloaders
  clean/        # team names, midpoint probability, merges
  features/     # rolling form, rest, Elo (no leakage)
  models/       # chronological split, training, metrics
  analysis/     # calibration, conditioned accuracy, sim trading
  viz/          # plots
tests/          # unit tests for core helpers
scripts/        # CLI entrypoints
data/raw/       # raw API / HTML downloads (gitignored)
data/processed/ # cleaned / feature tables (gitignored)
docs/           # assignment spec
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
```

## Tests

Tests cover the helpers called out in the work plan:

- team-name conversion
- midpoint probability calculation
- chronological 60/20/20 splitting
- rolling feature construction (shifted windows / rest)

```bash
pytest
# with coverage:
pytest --cov=kalshi_nba --cov-report=term-missing
```

## Data sources

| Source | Role |
|--------|------|
| [Kalshi markets API](https://external-api.kalshi.com/trade-api/v2/markets?series_ticker=KXNBAGAME&limit=1000) | Recent `KXNBAGAME` contracts |
| [Kalshi historical markets](https://external-api.kalshi.com/trade-api/v2/historical/markets?series_ticker=KXNBAGAME&limit=1000) | Archived contracts |
| Kalshi candlesticks | Final pregame price ≥ 15 minutes before tip-off |
| [Basketball Reference 2025–26 schedule](https://www.basketball-reference.com/leagues/NBA_2026_games.html) | Results + schedule |

Download helpers:

```bash
python scripts/collect_data.py --out-dir data/raw
```

## Method sketch

1. Paginate Kalshi recent + historical markets; pull 1-minute candles; take last price ≥ 15 minutes before `occurrence_datetime`.
2. Scrape monthly Basketball Reference tables; normalize team names; merge on date + home + visitor.
3. Build pregame features from prior games only: rolling win% / point diff (5 & 10), Elo, rest / B2B, home indicator. Model columns are home − visitor differences; target = home win.
4. Chronological split: 60% train / 20% validation / 20% test. Tune on validation; evaluate once on test against Kalshi.
5. Calibration bins, conditioned Brier/accuracy, and thresholded simulated trades (no real money).

## Exploratory Data Analysis (Part 2)

After downloading raw data, run:

```bash
python scripts/collect_data.py --out-dir data/raw
python scripts/run_eda.py
```

This writes:

- `data/processed/eda_summary.json` — shapes, missingness, summaries
- `reports/figures/*.png` — EDA plots
- `reports/eda.pdf` — Gradescope report (also requires public GitHub link)

EDA modules live under `src/kalshi_nba/eda/` (summaries, join QA, plots). Parsing helpers for Kalshi titles/tickers are in `src/kalshi_nba/clean/kalshi_parse.py`. Tests cover fixtures in `data/fixtures/`.

```bash
pytest
flake8 src scripts tests
```

## Status

EDA deliverable is in place: raw collectors, Kalshi/BR cleaning + join, summary/plot pipeline, `reports/eda.pdf`, and unit tests. Next steps are bulk pregame candlestick probabilities, full model training, conditioned accuracy, and simulated-return analysis.
