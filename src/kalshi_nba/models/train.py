"""Train and select classification models on chronological validation data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import classification_metrics


def _model_candidates() -> dict[str, list[Pipeline]]:
    """Small searchable grids for logistic regression, RF, and GBM."""
    candidates: dict[str, list[Pipeline]] = {
        "logistic_regression": [
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=c,
                            max_iter=2000,
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            for c in (0.1, 1.0, 10.0)
        ],
        "random_forest": [
            Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=n,
                            max_depth=depth,
                            random_state=42,
                        ),
                    )
                ]
            )
            for n in (100, 200)
            for depth in (3, 5, None)
        ],
        "gradient_boosting": [
            Pipeline(
                [
                    (
                        "clf",
                        GradientBoostingClassifier(
                            n_estimators=n,
                            max_depth=depth,
                            learning_rate=lr,
                            random_state=42,
                        ),
                    )
                ]
            )
            for n in (100, 200)
            for depth in (2, 3)
            for lr in (0.05, 0.1)
        ],
    }
    return candidates


def tune_and_select_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "home_win",
    metric: str = "brier",
) -> dict[str, Any]:
    """Fit candidate models and return the best by validation metric.

    Lower is better for ``brier`` and ``log_loss``; higher is better for
    ``accuracy``.
    """
    if metric not in {"brier", "log_loss", "accuracy"}:
        raise ValueError(f"unsupported metric: {metric}")

    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train[target_col].to_numpy(dtype=int)
    x_val = validation[feature_cols].to_numpy(dtype=float)
    y_val = validation[target_col].to_numpy(dtype=int)

    # Drop rows with NaNs from early-season rolling windows
    train_mask = ~np.isnan(x_train).any(axis=1)
    val_mask = ~np.isnan(x_val).any(axis=1)
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_val, y_val = x_val[val_mask], y_val[val_mask]

    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("train/validation sets have no complete feature rows")

    lower_is_better = metric in {"brier", "log_loss"}
    best: dict[str, Any] | None = None
    leaderboard: list[dict[str, Any]] = []

    for family, models in _model_candidates().items():
        for model in models:
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_val)[:, 1]
            scores = classification_metrics(y_val, proba)
            row = {"family": family, "params": model.get_params(), **scores}
            leaderboard.append(row)

            current = scores[metric]
            if best is None:
                best = {"model": model, "family": family, "metrics": scores}
                continue

            champion = best["metrics"][metric]
            improved = current < champion if lower_is_better else current > champion
            if improved:
                best = {"model": model, "family": family, "metrics": scores}

    assert best is not None
    best["leaderboard"] = sorted(
        leaderboard,
        key=lambda r: r[metric],
        reverse=not lower_is_better,
    )
    return best
