"""Smoke-test sources/open_meteo.py end-to-end."""
import pandas as pd

from loadforecast.data.schema import COLUMN_BY_NAME
from loadforecast.data.sources import open_meteo

start = pd.Timestamp("2024-06-14", tz="UTC")
end   = pd.Timestamp("2024-06-16", tz="UTC")

for col_name in (
    "weather__temperature_2m",
    "weather__shortwave_radiation",
    "weather__wind_speed_100m",
    "weather__cloud_cover",
):
    s = open_meteo.fetch(COLUMN_BY_NAME[col_name], start, end)
    print(
        f"{col_name:<35s} {len(s):>4d} pts  "
        f"min={s.min():>6.2f}  max={s.max():>6.2f}  "
        f"first={s.iloc[0]:.2f}  last={s.iloc[-1]:.2f}"
    )
