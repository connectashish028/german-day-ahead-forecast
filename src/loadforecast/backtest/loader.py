"""Data loading + Berlin-day window helpers for the backtest harness."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

BERLIN = "Europe/Berlin"
GATE_HOUR_LOCAL = 12  # German day-ahead auction gate closure: D-1 12:00 CET/CEST


def load_smard_15min(path: str | Path = "smard_merged_15min.parquet") -> pd.DataFrame:
    """Load the merged SMARD 15-min parquet with a tz-aware UTC datetime index."""
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    return df


def issue_time_for(delivery_date: date) -> pd.Timestamp:
    """Forecast issue time = D-1 12:00 Europe/Berlin, returned as UTC."""
    d_minus_1 = pd.Timestamp(delivery_date, tz=BERLIN) - pd.Timedelta(days=1)
    issue_local = d_minus_1.replace(hour=GATE_HOUR_LOCAL, minute=0, second=0, microsecond=0)
    return issue_local.tz_convert("UTC")


def target_index_for(delivery_date: date) -> pd.DatetimeIndex:
    """The 96 quarter-hour UTC timestamps covering Berlin calendar day `delivery_date`."""
    start_local = pd.Timestamp(delivery_date, tz=BERLIN)
    end_local = start_local + pd.Timedelta(days=1)
    idx_local = pd.date_range(start_local, end_local, freq="15min", inclusive="left")
    return idx_local.tz_convert("UTC")


def slice_history(df: pd.DataFrame, issue_time: pd.Timestamp) -> pd.DataFrame:
    """Return rows strictly before issue_time. Defensive copy."""
    return df.loc[df.index < issue_time].copy()
