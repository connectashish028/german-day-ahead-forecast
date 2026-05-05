"""SMARD downloadcenter JSON-API source for TSO forecast columns.

Replaces the manual-CSV path in `sources/smard_csv.py` for the columns
the model actually uses. Hits the same endpoint the smard.de download
form uses behind the scenes, so no auth and no email registration.

Module IDs were captured from a real browser request (DevTools network
panel) — see `scripts/probe_smard_downloadcenter.py` for the discovery
trail. Only the IDs the model actually needs are wired up here:

  6000411  Forecasted consumption: total grid load   (fc_cons__grid_load)
  6004362  Forecasted consumption: residual load     (fc_cons__residual_load)

The forecasted-generation IDs (`fc_gen__*`) aren't covered yet — the
model doesn't use them, but the manual-CSV fallback still works for
completeness if a future feature wants them.
"""
from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd
import requests

URL = "https://www.smard.de/nip-download-manager/nip/download/market-data"
TIMEOUT = 60

# Schema column name -> SMARD module ID.
MODULE_IDS: dict[str, int] = {
    "fc_cons__grid_load":     6000411,
    "fc_cons__residual_load": 6004362,
}


def _request(module_id: int, start: pd.Timestamp, end: pd.Timestamp, region: str) -> str:
    payload = {
        "request_form": [
            {
                "format": "CSV",
                "moduleIds": [module_id],
                "region": region,
                "timestamp_from": int(start.tz_convert("UTC").timestamp() * 1000),
                "timestamp_to":   int(end.tz_convert("UTC").timestamp() * 1000),
                "type": "discrete",
                "language": "en",
                "resolution": "quarterhour",
            }
        ]
    }
    r = requests.post(URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


@lru_cache(maxsize=32)
def _fetch_cached(module_id: int, start_iso: str, end_iso: str, region: str) -> pd.DataFrame:
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    body = _request(module_id, start, end, region)
    # Response is English-locale CSV: comma=thousands, dot=decimal.
    # Encoding is utf-8-sig (leading BOM); skip blank lines between rows.
    df = pd.read_csv(
        io.StringIO(body), sep=";", decimal=".", thousands=",",
        encoding="utf-8", skip_blank_lines=True,
    )
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    if df.empty:
        return df

    # Berlin-local timestamps like "Apr 1, 2026 2:00 AM". Single-digit days
    # mean format= without zero-pad on Windows; let pandas infer instead.
    ts = pd.to_datetime(df["Start date"], errors="coerce")
    ts = ts.dt.tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward",
    )
    df.index = ts.dt.tz_convert("UTC")
    df.index.name = "timestamp"
    return df.sort_index()


def fetch(column, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return the requested column as a tz-aware UTC Series clipped to [start, end)."""
    name = column.name
    if name not in MODULE_IDS:
        raise ValueError(
            f"smard_downloadcenter has no module ID for {name!r}. "
            f"Known: {list(MODULE_IDS)}"
        )
    region = column.fetch_kwargs.get("region", "DE-LU")
    df = _fetch_cached(MODULE_IDS[name], start.isoformat(), end.isoformat(), region)
    if df.empty:
        return pd.Series(dtype="float64", name=name)

    value_cols = [c for c in df.columns if c not in ("Start date", "End date")]
    if not value_cols:
        return pd.Series(dtype="float64", name=name)

    s = pd.to_numeric(df[value_cols[0]], errors="coerce").rename(name)
    return s.loc[(s.index >= start) & (s.index < end)]
