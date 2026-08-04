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
- event-ticker parsing and market tidying (dates, teams, LA disambiguation)
- tip-off time parsing (Eastern time, DST)
- pregame candle selection (both live/historical candle schemas)
- EDA summaries (missingness, seven-number, categorical) and the
  Kalshi-to-results merge

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
# then pull final pregame prices (one candlestick request per game):
python scripts/collect_candles.py --raw-dir data/raw
```

## EDA (Part 2)

The EDA report is [`reports/eda.pdf`](reports/eda.pdf). Reproduce every
number and figure in it with:

```bash
python scripts/run_eda.py
```

This writes `reports/eda_summary.txt` (dataset sizes, missingness,
seven-number summaries, categorical counts, calibration preview),
`reports/figures/*.png`, and the processed tables
`data/processed/kalshi_events.csv` / `data/processed/merged_games.csv`.

## Method sketch

1. Paginate Kalshi recent + historical markets; pull 1-minute candles; take last price ≥ 15 minutes before `occurrence_datetime`.
2. Scrape monthly Basketball Reference tables; normalize team names; merge on date + home + visitor.
3. Build pregame features from prior games only: rolling win% / point diff (5 & 10), Elo, rest / B2B, home indicator. Model columns are home − visitor differences; target = home win.
4. Chronological split: 60% train / 20% validation / 20% test. Tune on validation; evaluate once on test against Kalshi.
5. Calibration bins, conditioned Brier/accuracy, and thresholded simulated trades (no real money).

## Status

Scaffold is **test-ready**: package layout, dependencies, collectors/cleaners/feature/model stubs with working core utilities, and a passing unit-test suite. Next steps are live data pulls, merge QA, full model runs, and report figures.
