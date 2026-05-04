"""Smoke-test sources/smard_csv.py against the freshly downloaded forecast CSVs."""
from __future__ import annotations
import pandas as pd

from loadforecast.data.schema import COLUMN_BY_NAME
from loadforecast.data.sources import smard_csv

start = pd.Timestamp("2026-04-30", tz="UTC")
end = pd.Timestamp("2026-05-04", tz="UTC")

for col_name in [
    "fc_cons__grid_load",
    "fc_cons__residual_load",
    "fc_gen__total",
    "fc_gen__photovoltaics_and_wind",
    "fc_gen__wind_onshore",
    "fc_gen__photovoltaics",
    "fc_gen__other",
]:
    col = COLUMN_BY_NAME[col_name]
    s = smard_csv.fetch(col, start, end)
    print(
        f"{col_name:<35s} {len(s):>4d} obs  "
        f"min={s.min():>10.2f}  max={s.max():>10.2f}  "
        f"first={s.index[0]}  last={s.index[-1]}"
    )
