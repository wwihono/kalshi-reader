"""Shared evaluation metrics for models and Kalshi probabilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def brier_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    return float(brier_score_loss(np.asarray(list(y_true)), np.asarray(list(y_prob))))


def log_loss_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    yt = np.asarray(list(y_true))
    yp = np.clip(np.asarray(list(y_prob), dtype=float), 1e-15, 1 - 1e-15)
    return float(log_loss(yt, yp, labels=[0, 1]))


def classification_metrics(y_true: Iterable[float], y_prob: Iterable[float]) -> dict[str, float]:
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_prob), dtype=float)
    yhat = (yp >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(yt, yhat)),
        "brier": brier_score(yt, yp),
        "log_loss": log_loss_score(yt, yp),
    }
