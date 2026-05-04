"""Open-Meteo NWP weather source.

Returns hourly forecasts for major German load centres, population-weighted
into a single national series per variable. Free, no auth, CC-BY licence.

Why population-weighted
-----------------------
A heatwave in NRW (~22% of pop) hits demand more than the same heatwave in
Saarland (~1%). The simple average across cities under-weights load-heavy
regions. We use the same `BUNDESLAND_POPULATION` weights as the holiday
fraction logic, mapped from each city to its Bundesland.

Leakage safety
--------------
We use the `/historical-forecast` endpoint, which returns NWP forecasts as
they were *issued* (not actuals observed in hindsight). This is the
forecast a real-time system would have had at issue time T. Open-Meteo
serves the most-recent-issued forecast for each timestamp, so for any
parquet timestamp `t` the value reflects the most recent NWP forecast
that pre-dates `t`. That's leakage-safe for both encoder (history) and
decoder (delivery day) usage.

Variables
---------
- temperature_2m         (degC)      heating/cooling load driver
- shortwave_radiation    (W/m^2)     PV generation driver
- wind_speed_100m        (km/h)      wind generation driver
- cloud_cover            (%)         general weather signal
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from ..schema import Column

# 6 largest German load centres + the Bundesland they sit in.
CITIES: dict[str, dict] = {
    "berlin":     {"lat": 52.52, "lon": 13.41, "land": "BE"},
    "hamburg":    {"lat": 53.55, "lon":  9.99, "land": "HH"},
    "munich":     {"lat": 48.14, "lon": 11.58, "land": "BY"},
    "cologne":    {"lat": 50.94, "lon":  6.96, "land": "NW"},
    "frankfurt":  {"lat": 50.11, "lon":  8.68, "land": "HE"},
    "stuttgart":  {"lat": 48.78, "lon":  9.18, "land": "BW"},
}

VARS = ("temperature_2m", "shortwave_radiation", "wind_speed_100m", "cloud_cover")

HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
TIMEOUT = 60
MAX_RETRIES = 3


def _request(params: dict) -> dict:
    """GET with simple retry on 5xx / network errors."""
    for attempt in range(MAX_RETRIES):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            r = requests.get(HISTORICAL_URL, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                continue
            r.raise_for_status()
        except requests.RequestException:
            continue
    raise RuntimeError(f"open-meteo failed after {MAX_RETRIES} attempts: {params}")


def _fetch_city(
    city_key: str, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    """One city's hourly forecast over [start, end] (UTC)."""
    cfg = CITIES[city_key]
    payload = _request({
        "latitude": cfg["lat"],
        "longitude": cfg["lon"],
        "start_date": start.tz_convert("UTC").strftime("%Y-%m-%d"),
        "end_date":   end.tz_convert("UTC").strftime("%Y-%m-%d"),
        "hourly": ",".join(VARS),
        "models": "best_match",
        "timezone": "UTC",
    })
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")[list(VARS)]


def _population_weighted(per_city: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Population-weight each variable across the 6 cities."""
    from ...features.calendar import BUNDESLAND_POPULATION

    weights = {c: BUNDESLAND_POPULATION[CITIES[c]["land"]] for c in per_city}
    total = sum(weights.values())

    aligned = pd.concat(per_city.values(), axis=1, keys=per_city.keys())
    out = pd.DataFrame(index=aligned.index)
    for var in VARS:
        cols = [(c, var) for c in per_city]
        # weighted average per timestamp
        w_arr = pd.Series({c: weights[c] / total for c in per_city})
        sub = aligned[cols]
        sub.columns = [c[0] for c in sub.columns]
        out[var] = (sub * w_arr).sum(axis=1)
    return out


_CACHE: dict[str, pd.DataFrame] = {}


def _load_germany(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch all 6 cities, return a population-weighted national hourly DataFrame.

    Cached: a wider window covers any narrower fetch later in the same process.
    """
    key = "DE_pop_weighted"
    if key in _CACHE:
        cached = _CACHE[key]
        if cached.index.min() <= start and cached.index.max() >= end:
            return cached
    per_city: dict[str, pd.DataFrame] = {}
    for city in CITIES:
        per_city[city] = _fetch_city(city, start, end)
        time.sleep(0.05)  # be polite
    df = _population_weighted(per_city)
    _CACHE[key] = df
    return df


def fetch(column: Column, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return one weather variable as a 15-min UTC Series over [start, end).

    Open-Meteo serves hourly. We forward-fill within hour onto a 15-min
    grid (limit=3) so it lines up with the rest of the parquet.
    """
    var = column.fetch_kwargs["variable"]
    de = _load_germany(start, end)
    s = de[var]
    grid = pd.date_range(start, end, freq="15min", inclusive="left")
    return s.reindex(grid).ffill(limit=3).rename(column.name)
