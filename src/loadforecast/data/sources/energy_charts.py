"""Energy-Charts source. https://api.energy-charts.info/

Free, no auth, public CC-BY licence. Provides:
  - /price?bzn=DE-LU      day-ahead prices for any bidding zone
  - /public_power         actual generation by source (Germany)
  - /total_power          total generation (Germany)

We use it for prices (15 zones) and actual generation. ENTSO-E covers
the load + forecast columns separately.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

BASE_URL = "https://api.energy-charts.info"
TIMEOUT = 60
RETRY_DELAYS = [1.0, 3.0, 8.0]


def _request(path: str, params: dict) -> dict:
    """GET with simple exponential-backoff retry on 429/5xx."""
    last_err = None
    for delay in [0, *RETRY_DELAYS]:
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(f"{BASE_URL}/{path}", params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"energy-charts {path} failed: {last_err}")


def _to_series(unix_seconds: list[int], values: list[float | None], name: str) -> pd.Series:
    idx = pd.to_datetime(unix_seconds, unit="s", utc=True)
    s = pd.Series(values, index=idx, name=name, dtype="float64")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _date_str(ts: pd.Timestamp) -> str:
    """Energy-Charts wants plain YYYY-MM-DD in Berlin local. We pass UTC dates;
    they're slightly larger windows but always supersets of what we want."""
    return ts.tz_convert("UTC").strftime("%Y-%m-%d")


def fetch_price(bzn: str, start: pd.Timestamp, end: pd.Timestamp, *, name: str) -> pd.Series:
    """Day-ahead price for a bidding zone over [start, end] (UTC, 15-min)."""
    payload = _request("price", {
        "bzn": bzn,
        "start": _date_str(start),
        "end":   _date_str(end),
    })
    s = _to_series(payload["unix_seconds"], payload["price"], name)
    return s.loc[(s.index >= start) & (s.index < end)]


def fetch_public_power(
    production_type: str, start: pd.Timestamp, end: pd.Timestamp, *, name: str,
) -> pd.Series:
    """Actual generation for a single production type, Germany, 15-min."""
    payload = _request("public_power", {
        "country": "de",
        "start": _date_str(start),
        "end":   _date_str(end),
    })
    # `production_types` is a list of dicts: {name, data}
    for pt in payload.get("production_types", []):
        if pt.get("name") == production_type:
            s = _to_series(payload["unix_seconds"], pt["data"], name)
            return s.loc[(s.index >= start) & (s.index < end)]
    available = sorted({p.get("name") for p in payload.get("production_types", [])})
    raise KeyError(
        f"production_type {production_type!r} not in /public_power response. "
        f"Available: {available}"
    )


def fetch(column, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Dispatch on `column.fetch_kwargs['endpoint']`."""
    kw = column.fetch_kwargs
    ep = kw["endpoint"]
    if ep == "price":
        return fetch_price(kw["bzn"], start, end, name=column.name)
    if ep == "public_power":
        return fetch_public_power(kw["production_type"], start, end, name=column.name)
    raise ValueError(f"unknown endpoint for energy-charts: {ep!r}")
