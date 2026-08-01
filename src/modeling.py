"""Model training, hyperparameter tuning, and evaluation metrics.

Models are tuned on the validation period only and compared with Kalshi
once on the untouched chronological test period (accuracy, Brier score,
log loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Fraction of games where the >0.5 side actually won."""
    y_true = np.asarray(y_true, dtype=float)
    preds = (np.asarray(probs, dtype=float) > 0.5).astype(float)
    return float(np.mean(preds == y_true))


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - y_true) ** 2))


def log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """Negative mean log-likelihood, with clipping for 0/1 probabilities."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)))


def evaluate_probabilities(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """All three comparison metrics at once."""
    return {
        "accuracy": accuracy(y_true, probs),
        "brier_score": brier_score(y_true, probs),
        "log_loss": log_loss(y_true, probs),
    }


def default_model_grids() -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
    """The three model families from the spec with their parameter grids."""
    return {
        "logistic_regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=0),
            {
                "n_estimators": [100, 300],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [1, 5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=0),
            {
                "n_estimators": [100, 300],
                "max_depth": [2, 3],
                "learning_rate": [0.03, 0.1],
            },
        ),
    }


@dataclass
class TuningResult:
    name: str
    estimator: BaseEstimator
    params: dict[str, Any]
    val_metrics: dict[str, float] = field(default_factory=dict)


def tune_model(
    estimator: BaseEstimator,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    selection_metric: str = "log_loss",
) -> TuningResult:
    """Grid-search one model family; select by validation ``selection_metric``.

    Lower is better for both supported metrics (``log_loss``, ``brier_score``).
    """
    best: TuningResult | None = None
    for params in ParameterGrid(param_grid):
        model = clone(estimator).set_params(**params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_probabilities(y_val, probs)
        if best is None or metrics[selection_metric] < best.val_metrics[selection_metric]:
            best = TuningResult(
                name=type(estimator).__name__,
                estimator=model,
                params=dict(params),
                val_metrics=metrics,
            )
    assert best is not None, "param_grid produced no candidates"
    return best


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_grids: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]] | None = None,
    selection_metric: str = "log_loss",
) -> dict[str, TuningResult]:
    """Tune every model family; returns the best configuration of each."""
    model_grids = model_grids or default_model_grids()
    results: dict[str, TuningResult] = {}
    for name, (estimator, grid) in model_grids.items():
        result = tune_model(
            estimator, grid, X_train, y_train, X_val, y_val, selection_metric
        )
        result.name = name
        results[name] = result
    return results


def compare_on_test(
    tuned: dict[str, TuningResult],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    kalshi_probs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score each tuned model (and optionally Kalshi) on the same test games."""
    rows = []
    for name, result in tuned.items():
        probs = result.estimator.predict_proba(X_test)[:, 1]
        rows.append({"model": name, **evaluate_probabilities(y_test, probs)})
    if kalshi_probs is not None:
        rows.append({"model": "kalshi", **evaluate_probabilities(y_test, kalshi_probs)})
    return pd.DataFrame(rows).set_index("model")
