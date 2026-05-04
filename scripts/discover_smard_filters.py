"""Map SMARD filter IDs to our parquet column names by value matching.

For each candidate filter ID, fetch one chunk of data and compare the values
against every price/forecast column in the parquet at the same timestamps.
The column with near-zero diff is the match.
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import requests

PARQUET = "smard_merged_15min.parquet"
BASE = "https://www.smard.de/app/chart_data"
REGION = "DE-LU"


def fetch_one_chunk(filter_id: int, region: str = REGION) -> pd.Series | None:
    """Get the most-recent quarterhour chunk for a filter and return as Series."""
    try:
        idx = requests.get(
            f"{BASE}/{filter_id}/{region}/index_quarterhour.json", timeout=30
        ).json()
    except Exception as e:
        print(f"  {filter_id}: index error: {e}", file=sys.stderr)
        return None
    if not idx.get("timestamps"):
        return None
    # Pick a chunk roughly mid-2024 so it overlaps our parquet
    target_ms = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp() * 1000)
    chunk_ts = max(t for t in idx["timestamps"] if t <= target_ms)
    try:
        rows = requests.get(
            f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_quarterhour_{chunk_ts}.json",
            timeout=30,
        ).json().get("series", [])
    except Exception as e:
        print(f"  {filter_id}: chunk error: {e}", file=sys.stderr)
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t_ms", "v"])
    df["ts"] = pd.to_datetime(df["t_ms"], unit="ms", utc=True)
    return df.set_index("ts")["v"]


def best_match(api_series: pd.Series, parquet: pd.DataFrame, candidate_cols: list[str]) -> tuple[str, float] | None:
    """Find the parquet column whose values most closely match the API series."""
    # Align on index
    common_idx = api_series.index.intersection(parquet.index)
    if len(common_idx) < 50:
        return None
    api_aligned = api_series.reindex(common_idx)
    best_col, best_score = None, float("inf")
    for col in candidate_cols:
        pq = parquet[col].reindex(common_idx)
        # Mean absolute relative difference (handles NaN)
        diff = (api_aligned - pq).abs()
        if diff.notna().sum() < 50:
            continue
        score = float(diff.mean())
        if score < best_score:
            best_score = score
            best_col = col
    return (best_col, best_score) if best_col else None


def main() -> None:
    print("Loading parquet…")
    df = pd.read_parquet(PARQUET)
    price_cols = [c for c in df.columns if c.startswith("price__")]
    fc_cols = [c for c in df.columns if c.startswith("fc_")]
    actual_cons_extra = [c for c in df.columns if c == "actual_cons__grid_load_incl_hydro_pumped_storage"]

    print(f"Looking for matches against {len(price_cols)} price cols, {len(fc_cols)} forecast cols, {len(actual_cons_extra)} extras.")

    # Probe ranges:
    # 4169-4185 should cover all 17 price filters (DE/LU + 14 neighbours + DE/AT/LU + DE/LU neighbours).
    # Forecast IDs are typically in 6000+ range.
    # Let's also probe a few known-likely forecast IDs.
    price_candidates = list(range(4169, 4186))
    forecast_candidates = [
        # known guesses
        2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506,
        4180, 5078, 6046, 5097, 6004, 6045, 6047,
        # SMARD forecast filter IDs (from observed URL patterns)
        2500, 2501, 410, 4359,
    ]
    extra_candidates = [411, 412, 413, 414, 415, 416, 4358, 4360]

    print(f"\n=== Mapping price filter IDs (4169-4185) to columns ===")
    for fid in price_candidates:
        s = fetch_one_chunk(fid)
        if s is None:
            print(f"  {fid}: no data")
            continue
        m = best_match(s, df, price_cols)
        if m is None:
            print(f"  {fid}: no overlap")
            continue
        col, score = m
        marker = "✓" if score < 0.01 else "?" if score < 1.0 else "✗"
        print(f"  {fid}: {marker}  closest = {col:<40} mean|diff|={score:.3f}")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
