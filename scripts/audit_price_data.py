"""Pre-modelling audit of the German day-ahead price data.

Things we want to confirm before designing the price forecaster:

1. Coverage: any large gaps?
2. Resolution: pre-Oct-2025 was hourly (forward-filled to 15min);
   post-Oct-2025 was native 15min. Does that show in the data?
3. Distribution: how non-stationary are prices? (Senior DS focus.)
4. Negative-price frequency: rare or chronic?
5. Daily/weekly seasonality strength.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

df = pd.read_parquet("smard_merged_15min.parquet")
PRICE = "price__germany_luxembourg"

px = df[PRICE].dropna()
print(f"Price column: {PRICE}")
print(f"  range: {px.index.min()} -> {px.index.max()}")
print(f"  rows:  {len(px):,}  (NaN: {df[PRICE].isna().sum():,})")
print()

# 1. Coverage gaps (any 24h+ NaN runs?)
nan_run = df[PRICE].isna().astype(int).groupby(
    df[PRICE].notna().cumsum()
).sum()
big_gaps = nan_run[nan_run >= 96]
print(f"  long NaN runs (>=24h): {len(big_gaps)}")
print()

# 2. Resolution check: does each hour have 4 distinct prices, or only 1?
# Post-Oct-2025 EPEX intraday went 15-min; before that, day-ahead was hourly.
px_2025_h1 = px["2025-01":"2025-06"]
px_2025_h2 = px["2025-11":"2026-04"]
unique_per_hour_h1 = px_2025_h1.groupby(px_2025_h1.index.floor("h")).nunique().mean()
unique_per_hour_h2 = px_2025_h2.groupby(px_2025_h2.index.floor("h")).nunique().mean()
print(f"  unique prices/hour, 2025-H1 (pre-15min auction): {unique_per_hour_h1:.2f}")
print(f"  unique prices/hour, 2025-H2 (post-15min auction): {unique_per_hour_h2:.2f}")
print()

# 3. Distribution
print("  distribution (€/MWh):")
print(f"    mean   : {px.mean():>8.1f}")
print(f"    median : {px.median():>8.1f}")
print(f"    std    : {px.std():>8.1f}")
print(f"    min    : {px.min():>8.1f}")
print(f"    p1     : {px.quantile(0.01):>8.1f}")
print(f"    p99    : {px.quantile(0.99):>8.1f}")
print(f"    max    : {px.max():>8.1f}")
print()

# 4. Negative-price frequency
neg = px[px < 0]
print(f"  negative-price quarter-hours: {len(neg):,} / {len(px):,} = "
      f"{len(neg)/len(px)*100:.1f} %")
print(f"  most negative day: {neg.idxmin()}  ({neg.min():.1f} €/MWh)")
print()

# 5. Year-over-year drift (mean by year)
print("  annual mean €/MWh:")
for y, v in px.groupby(px.index.year).mean().items():
    print(f"    {y}: {v:>7.2f}")
print()

# 6. Hourly mean profile (Berlin local hour)
hourly = px.tz_convert("Europe/Berlin").groupby(
    px.tz_convert("Europe/Berlin").index.hour
).mean()
print("  hourly mean profile (€/MWh, Berlin local):")
print(f"    overnight (0-6h): {hourly.iloc[0:6].mean():>7.2f}")
print(f"    morning (6-12h):  {hourly.iloc[6:12].mean():>7.2f}")
print(f"    midday (12-15h):  {hourly.iloc[12:15].mean():>7.2f}  "
      f"(usually low — solar)")
print(f"    evening (17-21h): {hourly.iloc[17:21].mean():>7.2f}  "
      f"(usually high — peak demand)")
