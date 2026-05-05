"""Sanity-check: auto-fetched downloadcenter values should match the
manual-CSV values within rounding for any overlapping date."""
from __future__ import annotations

import pandas as pd

from loadforecast.data.schema import COLUMN_BY_NAME
from loadforecast.data.sources import smard_downloadcenter

PARQUET = "smard_merged_15min.parquet"

start = pd.Timestamp("2026-04-01", tz="UTC")
end = pd.Timestamp("2026-04-02", tz="UTC")

existing = pd.read_parquet(PARQUET)
existing = existing.loc[(existing.index >= start) & (existing.index < end)]

for name in ("fc_cons__grid_load", "fc_cons__residual_load"):
    auto = smard_downloadcenter.fetch(COLUMN_BY_NAME[name], start, end)
    manual = existing[name]
    diff = (auto - manual).abs()
    print(f"\n{name}:")
    print(f"  auto:   n={len(auto)}, mean={auto.mean():.2f}")
    print(f"  manual: n={len(manual)}, mean={manual.mean():.2f}")
    print(f"  abs diff: max={diff.max():.4f}, mean={diff.mean():.6f}")
    if diff.max() < 1.0:
        print(f"  -> MATCH (within 1 MW)")
    else:
        print(f"  !! mismatch — investigate")
