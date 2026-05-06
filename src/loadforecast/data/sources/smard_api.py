"""SMARD chart_data API source for known filter IDs (no auth).

The legacy `smard_client.py` already had filter IDs for total grid load
(410), residual load (4359), and most generation sources. This module
wraps those into the unified `fetch(column, start, end)` interface used
by `refresh.py`.

Use this only for the columns Energy-Charts can't give us *and* whose
SMARD filter ID we already know. For TSO consumption forecasts, see
`sources/smard_downloadcenter.py` (different SMARD endpoint).
"""

from __future__ import annotations

import time

import pandas as pd
import requests

BASE = "https://www.smard.de/app/chart_data"
TIMEOUT = 30


def _fetch_filter(filter_id: int, region: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch all chunks for [start, end) of a given filter+region."""
    idx_url = f"{BASE}/{filter_id}/{region}/index_quarterhour.json"
    r = requests.get(idx_url, timeout=TIMEOUT)
    r.raise_for_status()
    all_ts = sorted(r.json().get("timestamps", []))
    start_ms = int(start.tz_convert("UTC").timestamp() * 1000)
    end_ms = int(end.tz_convert("UTC").timestamp() * 1000)
    needed = []
    for i, t in enumerate(all_ts):
        if t > end_ms:
            break
        next_t = all_ts[i + 1] if i + 1 < len(all_ts) else float("inf")
        if next_t > start_ms:
            needed.append(t)

    rows: list[list] = []
    for t in needed:
        url = f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_quarterhour_{t}.json"
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            rows.extend(r.json().get("series", []))
        except requests.HTTPError:
            continue
        time.sleep(0.04)

    if not rows:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(rows, columns=["t_ms", "v"])
    s = pd.Series(
        df["v"].astype("float64").values,
        index=pd.to_datetime(df["t_ms"], unit="ms", utc=True),
        dtype="float64",
    )
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.loc[(s.index >= start) & (s.index < end)]


def fetch(column, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    kw = column.fetch_kwargs
    filter_id = kw["filter_id"]
    region = kw.get("region", "DE-LU")
    s = _fetch_filter(filter_id, region, start, end)
    return s.rename(column.name)
