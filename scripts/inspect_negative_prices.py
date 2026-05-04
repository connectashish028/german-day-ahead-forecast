"""How often have German day-ahead prices gone deeply negative?

Cross-references SMARD `price__germany_luxembourg` against:
  - hour-of-day (negatives concentrated 11-15h => solar mid-day glut)
  - holiday flag / bridge day (low industrial demand)
  - VRE generation (PV + wind)
"""
from __future__ import annotations
import pandas as pd

from loadforecast.backtest import load_smard_15min
from loadforecast.features.calendar import (
    is_federal_holiday, is_bridge_day, population_weighted_holiday_fraction,
)

df = load_smard_15min("smard_merged_15min.parquet")
price = df["price__germany_luxembourg"].dropna()
print(f"Price coverage: {price.index.min()}  ->  {price.index.max()}")
print(f"#observations: {len(price)}")
print()

# Top 20 most negative half-hours
print("=== 20 most negative German day-ahead prices ===")
worst = price.sort_values().head(20)
worst_local = worst.copy()
worst_local.index = worst.index.tz_convert("Europe/Berlin")

cal_idx = price.index
hol_frac = population_weighted_holiday_fraction(cal_idx)
fed = is_federal_holiday(cal_idx)
bridge = is_bridge_day(cal_idx)

# Solar+wind actual generation
vre_cols = [c for c in df.columns if c.startswith(("actual_gen__photovoltaic", "actual_gen__wind"))]
vre = df[vre_cols].sum(axis=1) if vre_cols else None

for ts_utc, eur in worst.items():
    local = ts_utc.tz_convert("Europe/Berlin")
    h = hol_frac.loc[ts_utc]
    is_fed = fed.loc[ts_utc]
    is_br = bridge.loc[ts_utc]
    v = float(vre.loc[ts_utc]) / 1000 if vre is not None and ts_utc in vre.index else float("nan")
    print(
        f"  {local!s:<28} {eur:>9.2f} EUR/MWh  "
        f"hol_frac={h:>4.0%}  federal={'Y' if is_fed else '-'}  bridge={'Y' if is_br else '-'}  "
        f"VRE={v:>5.1f} GW"
    )

# Aggregate counts of negative prices by year and by hour
print("\n=== Negative price incidence by year ===")
neg = price[price < 0]
neg_year = neg.groupby(neg.index.year).agg(["count", "min", "mean"])
neg_year.columns = ["n_quarter_hours", "min_eur", "mean_eur"]
print(neg_year)

print("\n=== Negative prices by hour-of-day (Europe/Berlin) ===")
local_idx = price.index.tz_convert("Europe/Berlin")
hour = pd.Series(local_idx.hour, index=price.index)
neg_by_hour = (price < 0).groupby(hour).sum()
print(neg_by_hour.to_string())

# Holiday/bridge co-incidence with price < -100
print("\n=== Steep negatives (< -100 EUR/MWh): holiday context ===")
steep = price[price < -100]
steep_idx = steep.index
print(f"Total steep-negative quarter-hours: {len(steep)}")
print(f"  Federal holiday:      {fed.loc[steep_idx].sum():>5d}  ({fed.loc[steep_idx].mean():>5.1%})")
print(f"  Bridge day:           {bridge.loc[steep_idx].sum():>5d}  ({bridge.loc[steep_idx].mean():>5.1%})")
print(f"  Hol fraction > 0:     {(hol_frac.loc[steep_idx] > 0).sum():>5d}  ({(hol_frac.loc[steep_idx] > 0).mean():>5.1%})")
print(f"  Sat/Sun:              {((steep_idx.tz_convert('Europe/Berlin').dayofweek >= 5)).sum():>5d}")

# Specifically check April 30 / May 1 each year
print("\n=== Around May Day each year (Erster Mai = federal holiday) ===")
for year in (2023, 2024, 2025, 2026):
    win_start = pd.Timestamp(f"{year}-04-29", tz="UTC")
    win_end = pd.Timestamp(f"{year}-05-02", tz="UTC")
    sub = price.loc[win_start:win_end]
    if sub.empty:
        continue
    mn = sub.min()
    mn_ts = sub.idxmin().tz_convert("Europe/Berlin")
    print(f"  {year}: min {mn:>8.2f} EUR/MWh at {mn_ts}")
