"""Smoke-test build_price_window: build one window for a known good
delivery date in the holdout, sanity-check shapes and value ranges."""
from __future__ import annotations

from datetime import date

import numpy as np

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.price_dataset import (
    PRICE_DEC_FEATURE_NAMES,
    PRICE_ENC_FEATURE_NAMES,
    build_price_window,
)

df = load_smard_15min("smard_merged_15min.parquet")
issue = issue_time_for(date(2026, 4, 15))
print(f"issue: {issue}")

w = build_price_window(df, issue, include_weather=True)
print(f"X_enc shape: {w.X_enc.shape}, NaN: {int(np.isnan(w.X_enc).sum())}")
print(f"X_dec shape: {w.X_dec.shape}, NaN: {int(np.isnan(w.X_dec).sum())}")
print(f"y_price shape: {w.y_price.shape}, NaN: {int(np.isnan(w.y_price).sum())}")
print()

print("Encoder feature stats (col mean / min / max):")
all_enc_names = list(PRICE_ENC_FEATURE_NAMES) + [
    "weather__t2m", "weather__sw_rad", "weather__wind100m", "weather__cloud",
]
for i, name in enumerate(all_enc_names):
    col = w.X_enc[:, i]
    print(f"  {name:<28s}  mean={col.mean():>9.2f}  min={col.min():>9.2f}  max={col.max():>9.2f}")

print()
print(f"Target price (€/MWh): mean={w.y_price.mean():.1f}, "
      f"min={w.y_price.min():.1f}, max={w.y_price.max():.1f}")
