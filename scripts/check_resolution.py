"""Investigate the 4x difference in negative-price QH counts.

EPEX moved from 60-min to 15-min day-ahead auctions on 2025-10-01.
Energy-Charts may be returning hourly data for older years, repeated as
1 obs/hour rather than 4 obs/hour.
"""
from __future__ import annotations
import pandas as pd

df = pd.read_parquet("smard_merged_15min.parquet")
price = df["price__germany_luxembourg"]

# Count non-NaN observations per year
print("=== Non-NaN observations per year for price__germany_luxembourg ===")
for y in (2022, 2023, 2024, 2025, 2026):
    sub = price.loc[str(y)]
    print(f"  {y}: {sub.notna().sum():>6d} obs (NaN: {sub.isna().sum():>5d})")

print("\n=== First 12 prices in 2022 ===")
print(price.loc["2022-01-01":"2022-01-01 03:00"].head(15))

print("\n=== First 12 prices in 2024 ===")
print(price.loc["2024-06-15":"2024-06-15 03:00"].head(15))

print("\n=== First 12 prices in 2025 (after Oct 2025 → 15-min auctions) ===")
print(price.loc["2025-12-01":"2025-12-01 03:00"].head(15))

print("\n=== First 12 prices on 2026-05-01 ===")
print(price.loc["2026-05-01":"2026-05-01 03:00"].head(15))

# Compare actual_cons__grid_load (SMARD API) vs price (energy-charts) coverage
print("\n=== Coverage comparison ===")
for col in ["actual_cons__grid_load", "price__germany_luxembourg", "fc_cons__grid_load"]:
    s = df[col]
    print(f"  {col:<35s}  non-null: {s.notna().sum():>6d}  total: {len(s):>6d}")
