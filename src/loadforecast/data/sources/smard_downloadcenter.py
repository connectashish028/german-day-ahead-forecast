"""SMARD downloadcenter JSON-API source for TSO forecast columns.

Hits the same endpoint the smard.de download form uses behind the
scenes — no auth, no email registration, no manual CSV step.

Module IDs were captured from a real browser request (DevTools network
panel) — see `scripts/probe_smard_downloadcenter.py` for the discovery
trail. Only the IDs the model actually needs are wired up here:

  6000411  Forecasted consumption: total grid load   (fc_cons__grid_load)
  6004362  Forecasted consumption: residual load     (fc_cons__residual_load)

To add another TSO column (e.g. forecasted generation), capture its
module ID from a real browser DevTools request and append it to
`MODULE_IDS` below.
"""
from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd
import requests

URL = "https://www.smard.de/nip-download-manager/nip/download/market-data"
TIMEOUT = 60
# SMARD's downloadcenter caps each request at ~90 days. A multi-year fetch
# silently returns an empty body, so we chunk anything bigger than this.
MAX_CHUNK_DAYS = 90

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


def _parse_chunk(body: str) -> pd.DataFrame:
    """Parse one CSV response into a tz-aware UTC-indexed frame.

    Note: SMARD encodes unpublished forecast slots as ``-`` (a single
    hyphen). Mixed with thousand-separated numerics like ``11,667.99``,
    pandas leaves the value column as object dtype, and a naive
    `pd.to_numeric` returns all-NaN. We strip the placeholders and the
    thousands separators explicitly here so downstream code gets clean
    floats.
    """
    df = pd.read_csv(
        io.StringIO(body), sep=";",
        encoding="utf-8", skip_blank_lines=True, dtype=str,
    )
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    if df.empty or "Start date" not in df.columns:
        return pd.DataFrame()

    ts = pd.to_datetime(df["Start date"], errors="coerce")
    df = df[ts.notna()].copy()
    ts = ts.dropna()
    if df.empty:
        return df

    # Coerce every value column to float, handling "-" sentinels and the
    # English thousand-separator format SMARD uses.
    for col in df.columns:
        if col in ("Start date", "End date"):
            continue
        s = df[col].astype(str).str.strip()
        s = s.where(s != "-", "")
        s = s.str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")

    ts = ts.dt.tz_localize(
        "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward",
    )
    df.index = ts.dt.tz_convert("UTC")
    df.index.name = "timestamp"
    return df.sort_index()


@lru_cache(maxsize=64)
def _fetch_cached(module_id: int, start_iso: str, end_iso: str, region: str) -> pd.DataFrame:
    """Fetch [start, end) in MAX_CHUNK_DAYS-sized chunks and concat."""
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    chunks: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=MAX_CHUNK_DAYS), end)
        body = _request(module_id, cursor, chunk_end, region)
        chunk = _parse_chunk(body)
        if not chunk.empty:
            chunks.append(chunk)
        cursor = chunk_end
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


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
