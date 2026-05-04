"""Re-probe SMARD with region=DE (not DE-LU) for the missing forecast columns."""
from __future__ import annotations
import time
import requests
import pandas as pd

PARQUET = "smard_merged_15min.parquet"
df = pd.read_parquet(PARQUET)
TS = pd.Timestamp("2024-06-15 12:00", tz="UTC")
ts_ms = int(TS.timestamp() * 1000)

targets = {c: df.loc[TS, c] for c in df.columns if c.startswith("fc_") or c.startswith("actual_cons__grid_load")}
print(f"{len(targets)} target columns at {TS}")
print()

# Targeted ranges. Forecasts on SMARD might be under region 'DE' (not 'DE-LU')
# or filter IDs around 4350-4400, 6000-7000.
ranges = list(range(4350, 4500)) + list(range(2000, 2300)) + list(range(5000, 5300)) + list(range(6000, 6300))
regions = ["DE", "DE-LU"]

print(f"Probing {len(ranges)} filter IDs x {len(regions)} regions...")
matched = {}
for region in regions:
    for fid in ranges:
        try:
            r = requests.get(
                f"https://www.smard.de/app/chart_data/{fid}/{region}/index_quarterhour.json",
                timeout=6,
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
                f"https://www.smard.de/app/chart_data/{fid}/{region}/{fid}_{region}_quarterhour_{chunk_ts}.json",
                timeout=10,
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
                    matched[col] = (fid, region, v, target_val)
                    print(f"  MATCH: {fid}/{region:<8s} -> {col}  api={v:.2f} parquet={target_val:.2f}")
            time.sleep(0.04)
        except Exception:
            continue

print()
print(f"=== Summary: {len(matched)}/{len(targets)} columns matched ===")
for col, (fid, region, av, pv) in sorted(matched.items()):
    print(f"  {fid:>4d} / {region:<6s}  {col}")
print()
print("=== Still unmatched ===")
for col in sorted(targets):
    if col not in matched:
        print(f"  {col}")
