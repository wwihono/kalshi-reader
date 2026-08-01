"""Research-question analyses: calibration, conditions, simulated returns."""

from .calibration import calibration_table
from .conditions import group_accuracy
from .simulation import simulate_threshold_trades

__all__ = [
    "calibration_table",
    "group_accuracy",
    "simulate_threshold_trades",
]
