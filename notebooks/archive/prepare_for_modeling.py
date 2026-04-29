"""
Prepare the merged SMARD parquet for EDA + LSTM modeling.

What this does:
  1. Loads the 15-min merged parquet.
  2. Drops columns that are 100% NaN or otherwise unusable.
  3. Interpolates small gaps (<= 1h) in each column; larger gaps are reported.
  4. Resamples 15-min -> hourly with per-column aggregation rules:
       - prices, forecasts, state variables -> mean
       - energy flows (MWh over the interval) -> sum
  5. Restricts to 2023-01-01 onwards.
  6. Builds two convenience columns that are the analyst-report headliners:
       - actual_residual_load = grid_load - (wind_onshore + wind_offshore + pv)
       - fc_residual_load_target = fc_cons__grid_load - fc_gen__photovoltaics_and_wind
  7. Writes smard_clean_hourly.parquet.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

INPUT_PARQUET  = Path("smard_merged_15min.parquet")
OUTPUT_PARQUET = Path("smard_clean_hourly.parquet")
START_DATE     = "2023-01-01"   # exclude 2022 gas-crisis regime
MAX_GAP_HOURS  = 1              # interpolate gaps up to this size, flag bigger ones


# Columns to drop outright. Rationale per column:
DROP_COLS = [
    "price__de_at_lu",  # deprecated bidding zone, 100% NaN post-2018
    # 'price__de_lu_neighbours' is SMARD's own weighted-average neighbor price;
    # potentially useful but redundant with the individual country columns.
    # Keep it — it's a free engineered feature.
]

# Aggregation rule per column when resampling 15-min -> 1h.
#   - 'mean': for instantaneous / forecast / price variables
#   - 'sum':  for energy flows measured in MWh over each interval
# In this SMARD export, MWh fields are labeled that way in the original CSV.
# We already stripped the unit suffix during cleaning, but the pattern is:
#   actual_cons__* and actual_gen__* are in MWh per 15-min interval -> sum
#   prices, forecasts -> mean
SUM_COLS_PREFIXES = ("actual_cons__", "actual_gen__")


def pick_agg_rule(col: str) -> str:
    return "sum" if col.startswith(SUM_COLS_PREFIXES) else "mean"


def report_gaps(df: pd.DataFrame, name: str) -> None:
    """Print a gap summary for each column before resampling."""
    print(f"\n--- {name} gap summary ---")
    total = len(df)
    for c in df.columns:
        n_nan = df[c].isna().sum()
        if n_nan == 0:
            continue
        # Find the longest consecutive run of NaNs, in 15-min steps.
        isna = df[c].isna().to_numpy()
        max_run = 0
        run = 0
        for v in isna:
            if v:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        print(f"  {c:<55} {n_nan:>5} NaN ({n_nan/total*100:.2f}%), "
              f"longest run = {max_run} x 15min "
              f"({max_run*15/60:.1f}h)")


def main():
    if not INPUT_PARQUET.exists():
        raise SystemExit(f"missing {INPUT_PARQUET}")

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"loaded: {df.shape}, {df.index.min()} -> {df.index.max()}")
    assert df.index.is_monotonic_increasing, "index not sorted"
    assert df.index.is_unique, "duplicate timestamps — DST handling likely failed"

    # ---- 1. drop 100%-NaN and other unusable columns ----
    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    to_drop = sorted(set(DROP_COLS + empty_cols))
    print(f"\ndropping {len(to_drop)} columns:")
    for c in to_drop:
        print(f"  - {c} ({df[c].notna().sum()} non-null)")
    df = df.drop(columns=to_drop, errors="ignore")

    # ---- 2. report gaps BEFORE interpolation so you know what was filled ----
    report_gaps(df, "pre-interpolation")

    # ---- 3. interpolate small gaps ----
    # Linear interpolation for numeric gaps up to MAX_GAP_HOURS. Anything
    # longer stays as NaN (fixing it properly needs context you don't have).
    # Use 'limit' in 15-min steps (4 steps = 1h).
    limit_steps = int(MAX_GAP_HOURS * 4)
    df = df.interpolate(method="time", limit=limit_steps, limit_area="inside")

    remaining = df.isna().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        print(f"\nremaining NaNs after interpolation (will survive into output):")
        for c, n in remaining.items():
            print(f"  {c:<55} {n:>5}")

    # ---- 4. resample 15-min -> hourly ----
    agg = {c: pick_agg_rule(c) for c in df.columns}
    print(f"\nresampling to 1h ({sum(v=='sum' for v in agg.values())} sum, "
          f"{sum(v=='mean' for v in agg.values())} mean cols)")
    hourly = df.resample("1h").agg(agg)

    # ---- 5. restrict to 2023+ (skip the gas crisis regime) ----
    before = len(hourly)
    hourly = hourly.loc[START_DATE:]
    print(f"restricted to >= {START_DATE}: {before} -> {len(hourly)} rows")

    # ---- 6. convenience engineered columns ----
    # Actual residual load: what the fossil fleet had to cover.
    hourly["actual_residual_load"] = (
        hourly["actual_cons__grid_load_incl_hydro_pumped_storage"]
        - hourly["actual_gen__wind_onshore"]
        - hourly["actual_gen__wind_offshore"]
        - hourly["actual_gen__photovoltaics"]
    )

    # Forecast residual load for the target hour (your #1 price predictor).
    # Uses the combined wind+solar forecast column since it has better coverage.
    hourly["fc_residual_load"] = (
        hourly["fc_cons__grid_load"] - hourly["fc_gen__photovoltaics_and_wind"]
    )

    # ---- 7. final sanity checks ----
    print(f"\nfinal shape: {hourly.shape}")
    print(f"range: {hourly.index.min()} -> {hourly.index.max()}")
    print(f"expected hours: {int((hourly.index.max() - hourly.index.min()).total_seconds() / 3600) + 1}")
    print(f"actual rows:    {len(hourly)}")

    any_nan = hourly.isna().sum()
    any_nan = any_nan[any_nan > 0]
    if any_nan.empty:
        print("no NaNs remaining — clean")
    else:
        print("columns with NaNs remaining:")
        for c, n in any_nan.items():
            print(f"  {c:<55} {n:>5}")

    print("\nprice sanity (should be roughly: mean ~60-100, some negatives, some >300):")
    print(hourly["price__germany_luxembourg"].describe())

    hourly.to_parquet(OUTPUT_PARQUET)
    print(f"\nwrote {OUTPUT_PARQUET} ({OUTPUT_PARQUET.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()