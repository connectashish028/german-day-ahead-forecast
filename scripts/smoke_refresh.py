"""Smoke-test energy-charts source: fetch a few days of DE/LU price + solar."""
from __future__ import annotations
import pandas as pd

from loadforecast.data.schema import COLUMN_BY_NAME
from loadforecast.data.sources import energy_charts

start = pd.Timestamp("2026-04-30", tz="UTC")
end = pd.Timestamp("2026-05-03", tz="UTC")

for col_name in ("price__germany_luxembourg", "price__france", "actual_gen__photovoltaics"):
    col = COLUMN_BY_NAME[col_name]
    s = energy_charts.fetch(col, start, end)
    print(f"{col_name}: {len(s)} obs, {s.notna().sum()} non-NaN, "
          f"min={s.min():.2f} max={s.max():.2f}, "
          f"first={s.index[0]}, last={s.index[-1]}")
