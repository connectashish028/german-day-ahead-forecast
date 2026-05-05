"""Why is the P10/P90 band missing on a specific delivery day?
Run: python scripts/diagnose_tomorrow_band.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.dataset import build_window
from loadforecast.models.predict import lstm_quantile_predict_full

DELIVERY = pd.Timestamp("2026-05-06").date()

df = load_smard_15min("smard_merged_15min.parquet")
issue = issue_time_for(DELIVERY)
print(f"delivery: {DELIVERY}, issue: {issue}")

w = build_window(df, issue, include_weather=True)
print(f"X_enc NaN: {np.isnan(w.X_enc).sum()}, shape: {w.X_enc.shape}")
print(f"X_dec NaN: {np.isnan(w.X_dec).sum()}, shape: {w.X_dec.shape}")

# Where are the encoder NaNs?
ENC_NAMES = ("load", "residual", "hour_sin", "hour_cos",
             "dow_sin", "dow_cos",
             "weather__temperature_2m", "weather__shortwave_radiation",
             "weather__wind_speed_100m", "weather__cloud_cover")
nan_mask = np.isnan(w.X_enc)
print("\nNaN per encoder feature:")
for i, name in enumerate(ENC_NAMES):
    n = nan_mask[:, i].sum()
    if n:
        rows = np.where(nan_mask[:, i])[0]
        print(f"  {name:<35s} {n} NaN at row indices: {rows.tolist()}")

out = lstm_quantile_predict_full(df, issue)
band = out["p90"] - out["p10"]
print(f"band width: min={band.min():.1f}, mean={band.mean():.1f}, max={band.max():.1f}")
print("first 3 rows:")
print(out.head(3))
