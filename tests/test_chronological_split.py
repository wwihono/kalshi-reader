"""Tests for chronological train/validation/test splitting."""

import pandas as pd
import pytest

from kalshi_nba.models.split import chronological_split


def _sample_games(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2025-10-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "game_date": dates,
            "home_win": [i % 2 for i in range(n)],
        }
    )


def test_chronological_split_fractions_and_order() -> None:
    frame = _sample_games(10)
    split = chronological_split(frame, train_frac=0.6, val_frac=0.2, test_frac=0.2)

    assert len(split.train) == 6
    assert len(split.validation) == 2
    assert len(split.test) == 2

    assert split.train["game_date"].is_monotonic_increasing
    assert split.validation["game_date"].is_monotonic_increasing
    assert split.test["game_date"].is_monotonic_increasing

    assert split.train["game_date"].max() < split.validation["game_date"].min()
    assert split.validation["game_date"].max() < split.test["game_date"].min()


def test_chronological_split_uses_earliest_for_train() -> None:
    frame = _sample_games(5).sample(frac=1.0, random_state=0)  # shuffled
    split = chronological_split(frame, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    assert split.train["game_date"].iloc[0] == frame["game_date"].min()
    assert split.test["game_date"].iloc[-1] == frame["game_date"].max()


def test_chronological_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        chronological_split(_sample_games(), train_frac=0.5, val_frac=0.2, test_frac=0.2)


def test_chronological_split_empty() -> None:
    empty = pd.DataFrame(columns=["game_date", "home_win"])
    split = chronological_split(empty)
    assert len(split.train) == 0
    assert len(split.validation) == 0
    assert len(split.test) == 0
