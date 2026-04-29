"""Forecast error metrics + skill score vs TSO baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true - y_pred).abs().mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute percentage error in percent. Skips zero/NaN denominators."""
    mask = y_true.abs() > 1e-9
    if not mask.any():
        return float("nan")
    return float(((y_true[mask] - y_pred[mask]).abs() / y_true[mask].abs()).mean() * 100)


def skill_score(mae_model: float, mae_baseline: float) -> float:
    """1 - MAE_model / MAE_baseline. >0 means the model beats the baseline."""
    if mae_baseline <= 0:
        return float("nan")
    return 1.0 - mae_model / mae_baseline
