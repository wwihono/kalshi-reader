"""Tests for evaluation metrics and model tuning/selection."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.modeling import (
    accuracy,
    brier_score,
    compare_on_test,
    evaluate_probabilities,
    log_loss,
    tune_all_models,
    tune_model,
)


class TestMetrics:
    def test_accuracy_perfect_and_worst(self):
        y = np.array([1, 0, 1, 0])
        assert accuracy(y, np.array([0.9, 0.1, 0.8, 0.2])) == 1.0
        assert accuracy(y, np.array([0.1, 0.9, 0.2, 0.8])) == 0.0

    def test_brier_known_values(self):
        assert brier_score([1, 0], [1.0, 0.0]) == 0.0
        assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
        assert brier_score([1], [0.7]) == pytest.approx(0.09)

    def test_log_loss_known_value(self):
        assert log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))

    def test_log_loss_clips_extreme_probabilities(self):
        value = log_loss([1], [0.0])  # would be inf without clipping
        assert np.isfinite(value)

    def test_lower_brier_for_better_forecast(self):
        y = [1, 1, 0, 0]
        good = brier_score(y, [0.8, 0.7, 0.2, 0.3])
        bad = brier_score(y, [0.55, 0.55, 0.45, 0.45])
        assert good < bad

    def test_evaluate_probabilities_keys(self):
        metrics = evaluate_probabilities([1, 0], [0.6, 0.4])
        assert set(metrics) == {"accuracy", "brier_score", "log_loss"}


def _separable_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    logits = 2.0 * X["x1"] - 1.0 * X["x2"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int))
    return X, y


class TestTuning:
    def test_tune_model_selects_lowest_val_log_loss(self):
        X, y = _separable_data()
        X_train, y_train = X.iloc[:240], y.iloc[:240]
        X_val, y_val = X.iloc[240:320], y.iloc[240:320]
        result = tune_model(
            LogisticRegression(max_iter=1000),
            {"C": [0.001, 1.0]},
            X_train, y_train, X_val, y_val,
        )
        # With informative features, heavy regularization (C=0.001) should lose.
        assert result.params["C"] == 1.0
        assert result.val_metrics["log_loss"] > 0

    def test_tune_all_models_and_compare(self):
        X, y = _separable_data()
        X_train, y_train = X.iloc[:240], y.iloc[:240]
        X_val, y_val = X.iloc[240:320], y.iloc[240:320]
        X_test, y_test = X.iloc[320:], y.iloc[320:]
        grids = {
            "logreg_a": (LogisticRegression(max_iter=1000), {"C": [0.1, 1.0]}),
            "logreg_b": (LogisticRegression(max_iter=1000), {"C": [10.0]}),
        }
        tuned = tune_all_models(X_train, y_train, X_val, y_val, model_grids=grids)
        assert set(tuned) == {"logreg_a", "logreg_b"}
        kalshi = np.full(len(y_test), 0.5)
        comparison = compare_on_test(tuned, X_test, y_test, kalshi_probs=kalshi)
        assert list(comparison.index) == ["logreg_a", "logreg_b", "kalshi"]
        # A fitted model must beat the uninformative 0.5 benchmark here.
        assert comparison.loc["logreg_a", "log_loss"] < comparison.loc["kalshi", "log_loss"]
