"""Reference predictors plugged into the backtest harness.

These exist for two reasons:
1. The TSO baseline is the operational benchmark every model is measured against.
2. The seasonal-naive baseline is the harness's sanity check — it must score
   *worse* than the TSO. If it doesn't, the harness has a bug.
"""

from __future__ import annotations

import pandas as pd

from .loader import target_index_for

TSO_FORECAST_COL = "fc_cons__grid_load"
ACTUAL_LOAD_COL = "actual_cons__grid_load"


def _delivery_target_index(issue_time: pd.Timestamp):
    delivery_local = issue_time.tz_convert("Europe/Berlin").normalize() + pd.Timedelta(days=1)
    return target_index_for(delivery_local.date())


def tso_baseline_predict(df: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """Return the TSO's published day-ahead forecast for the delivery day.

    The TSO publishes the day-D forecast around D-1 10:00 CET, so by issue time
    D-1 12:00 CET the entire 96-step series for day D is already known. The
    harness passes the full DataFrame; selecting the delivery-day slice from
    `fc_cons__grid_load` is leakage-free.
    """
    target_idx = _delivery_target_index(issue_time)
    return df[TSO_FORECAST_COL].reindex(target_idx).rename("y_tso")


def seasonal_naive_predict(df: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """Predict each quarter-hour with the actual load from the same QH one week ago.

    Uses only timestamps strictly less than `issue_time` (D-7 is six days before
    issue time, well in the past).
    """
    target_idx = _delivery_target_index(issue_time)
    lagged_idx = target_idx - pd.Timedelta(days=7)
    history = df.loc[df.index < issue_time, ACTUAL_LOAD_COL]
    vals = history.reindex(lagged_idx).to_numpy()
    return pd.Series(vals, index=target_idx, name="y_seasonal_naive")
