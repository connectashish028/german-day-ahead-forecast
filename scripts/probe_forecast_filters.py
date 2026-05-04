"""Find SMARD filter IDs for the TSO consumption + generation forecasts by
matching values against the parquet at a known timestamp.
"""
from __future__ import annotations
import time
import requests
import pandas as pd

PARQUET = "smard_merged_15min.parquet"
df = pd.read_parquet(PARQUET)

# Known target values at one specific timestamp
TS = pd.Timestamp("2024-06-15 12:00", tz="UTC")
targets = {c: df.loc[TS, c] for c in df.columns if c.startswith(("fc_", "actual_cons__grid_load_incl"))}
print("Targets at", TS)
for k, v in targets.items():
    print(f"  {k}: {v:.2f}")
print()

# Probe likely filter IDs across known SMARD ranges
# Common forecast IDs reported in community: 6004, 6045, 6046, 6047, 5097, 5078, 4071+
RANGES = list(range(2000, 2100)) + list(range(5000, 5200)) + list(range(6000, 6100)) + list(range(8000, 8300))

print(f"Probing {len(RANGES)} candidate filter IDs (region=DE-LU)…")
matched = []
for fid in RANGES:
    try:
        idx_resp = requests.get(
            f"https://www.smard.de/app/chart_data/{fid}/DE-LU/index_quarterhour.json", timeout=10
        )
        if idx_resp.status_code != 200:
            continue
        idx = idx_resp.json()
        if not idx.get("timestamps"):
            continue
        # Pick the chunk containing TS (chunks are 1 week)
        ts_ms = int(TS.timestamp() * 1000)
        chunks = [t for t in idx["timestamps"] if t <= ts_ms]
        if not chunks:
            continue
        chunk_ts = max(chunks)
        rows = requests.get(
            f"https://www.smard.de/app/chart_data/{fid}/DE-LU/{fid}_DE-LU_quarterhour_{chunk_ts}.json",
            timeout=15,
        ).json().get("series", [])
        if not rows:
            continue
        # Find the row at TS
        df_chunk = pd.DataFrame(rows, columns=["t_ms", "v"])
        match_row = df_chunk[df_chunk["t_ms"] == ts_ms]
        if match_row.empty:
            continue
        v = match_row.iloc[0]["v"]
        if v is None or not isinstance(v, (int, float)):
            continue
        # Compare to targets
        for col, target_val in targets.items():
            if pd.notna(target_val) and abs(v - target_val) < 0.5:
                print(f"  MATCH: filter {fid} -> {col} (api={v:.2f}, parquet={target_val:.2f})")
                matched.append((fid, col, v, target_val))
        time.sleep(0.05)
    except Exception:
        continue

print(f"\nTotal matches: {len(matched)}")
