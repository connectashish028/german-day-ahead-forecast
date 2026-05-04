"""ENTSO-E Transparency Platform source.

Uses entsoe-py (https://github.com/EnergieID/entsoe-py). The token comes
from the env var ENTSOE_TOKEN (or a `.env` file at the project root).
Get one by emailing transparency@entsoe.eu after registering at
https://transparency.entsoe.eu/.

Methods we hit:
  - query_load(country_code, start, end)            -> 15-min actuals
  - query_load_forecast(country_code, start, end)   -> day-ahead total load forecast
  - query_generation_forecast(country_code, ...)    -> day-ahead total gen forecast
  - query_wind_and_solar_forecast(country_code, ...)-> day-ahead wind + solar forecast (DataFrame)

ENTSO-E's native resolution for DE_LU is 15-min from 2018 onward; for
earlier data it's hourly. We resample-up to 15-min by forward-fill so
the parquet stays uniform.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def _client():
    token = os.getenv("ENTSOE_TOKEN")
    if not token:
        raise RuntimeError(
            "ENTSOE_TOKEN not set. Put it in a .env file at the project root, "
            "e.g.  ENTSOE_TOKEN=<your-token>"
        )
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=token)


def _ensure_qh(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Reindex to a strict 15-min UTC grid, forward-filling at most one step
    (to convert hourly data to 15-min without leaking gaps)."""
    s = s.copy()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    target = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="15min", inclusive="left")
    return s.reindex(target).ffill(limit=3)


def fetch(column, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Dispatch on column.fetch_kwargs['method']."""
    kw = column.fetch_kwargs
    method = kw["method"]
    cc = kw["country_code"]
    client = _client()

    # entsoe-py expects pandas Timestamps with timezone — we pass tz-aware UTC
    # which the lib handles fine.
    s_or_df = getattr(client, f"query_{method}")(cc, start=start, end=end)

    if isinstance(s_or_df, pd.DataFrame):
        # wind_and_solar_forecast returns a DataFrame with columns like
        # ['Solar', 'Wind Offshore', 'Wind Onshore']. We sum to a single
        # photovoltaics_and_wind series to match our schema.
        if column.name == "fc_gen__photovoltaics_and_wind":
            s = s_or_df.sum(axis=1, min_count=1).rename(column.name)
        else:
            # query_load returns DataFrame with a single column "Actual Load"
            s = s_or_df.iloc[:, 0].rename(column.name)
    else:
        s = s_or_df.rename(column.name)

    return _ensure_qh(s, start, end).rename(column.name)
