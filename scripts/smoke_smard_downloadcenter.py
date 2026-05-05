"""Smoke-test the smard_downloadcenter source against the live API."""
from __future__ import annotations

import pandas as pd

from loadforecast.data.schema import COLUMN_BY_NAME
from loadforecast.data.sources import smard_downloadcenter

start = pd.Timestamp("2026-04-01", tz="UTC")
end = pd.Timestamp("2026-04-02", tz="UTC")

for name in ("fc_cons__grid_load", "fc_cons__residual_load"):
    col = COLUMN_BY_NAME[name]
    s = smard_downloadcenter.fetch(col, start, end)
    print(f"\n{name}: len={len(s)}")
    print(f"  index range: {s.index.min()}  ->  {s.index.max()}")
    print(f"  values: min={s.min():.0f}  max={s.max():.0f}  mean={s.mean():.0f}")
    print(f"  sample: {s.iloc[36:40].to_dict()}")
