"""Focused SMARD probe: hunt for TSO forecast filter IDs by value matching."""
from __future__ import annotations
import time
import requests
import pandas as pd

PARQUET = "smard_merged_15min.parquet"
df = pd.read_parquet(PARQUET)
TS = pd.Timestamp("2024-06-15 12:00", tz="UTC")
ts_ms = int(TS.timestamp() * 1000)

targets = {c: df.loc[TS, c] for c in df.columns if c.startswith("fc_") or c.startswith("actual_")}
print(f"{len(targets)} target columns at {TS}")
print()

# Targeted ranges where SMARD forecast IDs are known to live
ranges = list(range(2100, 2500)) + list(range(5200, 5500)) + list(range(6000, 6500)) + list(range(8200, 8500))

print(f"Probing {len(ranges)} candidate filter IDs (region DE-LU)…")
matched = {}
for fid in ranges:
    try:
        r = requests.get(
            f"https://www.smard.de/app/chart_data/{fid}/DE-LU/index_quarterhour.json",
            timeout=8,
        )
        if r.status_code != 200:
            continue
        idx = r.json()
        if not idx.get("timestamps"):
            continue
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
        df_chunk = pd.DataFrame(rows, columns=["t_ms", "v"])
        match_row = df_chunk[df_chunk["t_ms"] == ts_ms]
        if match_row.empty:
            continue
        v = match_row.iloc[0]["v"]
        if v is None:
            continue
        for col, target_val in targets.items():
            if pd.notna(target_val) and abs(v - target_val) < 0.5 and col not in matched:
                matched[col] = (fid, v, target_val)
                print(f"  MATCH: {fid} -> {col}  api={v:.2f} parquet={target_val:.2f}")
        time.sleep(0.04)
    except Exception:
        continue

print()
print(f"=== Summary: {len(matched)} unique columns matched ===")
for col, (fid, av, pv) in sorted(matched.items()):
    print(f"  {fid}: {col}")
print()
print("=== Unmatched columns ===")
for col in sorted(targets):
    if col not in matched:
        print(f"  {col}  (target={targets[col]:.2f})")
