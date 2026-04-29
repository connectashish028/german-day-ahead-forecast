"""Rolling-origin backtesting against the German TSO day-ahead baseline."""

from .baselines import seasonal_naive_predict, tso_baseline_predict
from .harness import BacktestResult, quick_backtest, run_backtest
from .loader import (
    BERLIN,
    issue_time_for,
    load_smard_15min,
    slice_history,
    target_index_for,
)
from .metrics import mae, mape, rmse, skill_score
from .types import PredictFn

__all__ = [
    "BERLIN",
    "BacktestResult",
    "PredictFn",
    "issue_time_for",
    "load_smard_15min",
    "mae",
    "mape",
    "quick_backtest",
    "rmse",
    "run_backtest",
    "seasonal_naive_predict",
    "skill_score",
    "slice_history",
    "target_index_for",
    "tso_baseline_predict",
]
