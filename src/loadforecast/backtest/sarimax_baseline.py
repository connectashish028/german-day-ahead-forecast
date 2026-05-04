"""SARIMAX-on-residual baseline.

The TSO already encodes the dominant daily/weekly load seasonality. This
predictor only models the residual `actual - tso_fc`, which is mostly
weather-driven noise plus occasional structural error. We expect SARIMAX
to score *worse* than the TSO baseline (it has no weather input) but
*better* than seasonal-naive — that is the harness's M3 sanity check.

Implementation:
- Take the last `history_days` of residuals strictly before issue_time.
- Fit SARIMAX(1,0,0)x(1,0,0,96) — one AR lag, one daily-seasonal AR lag.
- Forecast `48 + 96 = 144` quarter-hours ahead, take the trailing 96
  (the delivery day), and add back to the TSO forecast.

If the fit fails or produces NaN, fall back to predicting zero residual
(i.e. the raw TSO forecast) so the harness still runs end-to-end.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .baselines import ACTUAL_LOAD_COL, TSO_FORECAST_COL, _delivery_target_index

QH_PER_DAY = 96
GAP_QH = 48  # issue_time D-1 12:00 -> delivery start D 00:00 = 12h = 48 QH


def sarimax_residual_predict(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    history_days: int = 14,
) -> pd.Series:
    """Forecast residual with SARIMAX, add to TSO forecast for the delivery day."""
    target_idx = _delivery_target_index(issue_time)
    tso_fc = df[TSO_FORECAST_COL].reindex(target_idx)

    # Build the in-sample residual series (strictly pre-issue).
    cutoff = issue_time
    pre = df.loc[df.index < cutoff, [ACTUAL_LOAD_COL, TSO_FORECAST_COL]].dropna()
    history_start = cutoff - pd.Timedelta(days=history_days)
    pre = pre.loc[pre.index >= history_start]
    residual = (pre[ACTUAL_LOAD_COL] - pre[TSO_FORECAST_COL]).asfreq("15min")

    if residual.notna().sum() < QH_PER_DAY * 7:
        # Not enough history — fall back to TSO.
        return tso_fc.rename("y_sarimax_residual")

    residual = residual.interpolate(limit=4).dropna()

    # Fit + forecast. statsmodels SARIMAX is heavyweight; suppress its noise.
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    horizon = GAP_QH + QH_PER_DAY  # 144
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            model = SARIMAX(
                residual.to_numpy(),
                order=(1, 0, 0),
                seasonal_order=(1, 0, 0, QH_PER_DAY),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=50)
            fc = np.asarray(fit.forecast(steps=horizon))
    except Exception:
        return tso_fc.rename("y_sarimax_residual")

    if not np.isfinite(fc).all():
        return tso_fc.rename("y_sarimax_residual")

    delivery_residual_fc = fc[GAP_QH:]
    pred = tso_fc.to_numpy() + delivery_residual_fc
    return pd.Series(pred, index=target_idx, name="y_sarimax_residual")
