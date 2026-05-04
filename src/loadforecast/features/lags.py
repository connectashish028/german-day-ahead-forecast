"""Lag and rolling-window features.

All helpers respect availability: a value at timestamp `t` is allowed to enter
the feature only if `t < issue_time + offset` for the column's availability
rule (see `availability.py`). For lags this is enforced by reading from a
`usable_columns(...)` masked frame.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

QH_PER_DAY = 96


def lag_at(series: pd.Series, target_idx: pd.DatetimeIndex, lag_days: float) -> pd.Series:
    """Value of `series` at (target timestamp - lag_days). Returns NaN if unavailable."""
    lagged_idx = target_idx - pd.Timedelta(days=lag_days)
    return series.reindex(lagged_idx).set_axis(target_idx)


def rolling_stat(
    series: pd.Series,
    target_idx: pd.DatetimeIndex,
    *,
    window_days: float,
    stat: str,
    end_offset_days: float = 0.0,
) -> pd.Series:
    """Compute rolling stat on `series` over `[t - end_offset - window, t - end_offset)`
    for each `t` in `target_idx`. NaN where window data is missing.

    `stat` is one of "mean", "std", "min", "max".
    """
    if stat not in {"mean", "std", "min", "max"}:
        raise ValueError(f"unsupported stat: {stat}")

    end = target_idx - pd.Timedelta(days=end_offset_days)
    start = end - pd.Timedelta(days=window_days)

    out = []
    for s, e in zip(start, end, strict=True):
        window = series.loc[(series.index >= s) & (series.index < e)]
        if window.empty:
            out.append(float("nan"))
        else:
            out.append(getattr(window, stat)())
    return pd.Series(out, index=target_idx)


def lag_features(
    masked_df: pd.DataFrame,
    target_idx: pd.DatetimeIndex,
    column: str,
    lag_days_list: Iterable[float],
    *,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of lag features for a single column at multiple lags."""
    pfx = prefix or column
    s = masked_df[column]
    return pd.DataFrame(
        {f"{pfx}__lag_{int(ld * QH_PER_DAY)}qh": lag_at(s, target_idx, ld) for ld in lag_days_list}
    )
