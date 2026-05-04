"""Sample a few values from SMARD filters and try to match against parquet columns by recent value."""
from __future__ import annotations
import sys
import requests
import pandas as pd

PARQUET = "smard_merged_15min.parquet"

def fetch_recent_value(fid: int, region: str = "DE-LU"):
    try:
        idx = requests.get(
            f"https://www.smard.de/app/chart_data/{fid}/{region}/index_quarterhour.json", timeout=20
        ).json()
        if not idx.get("timestamps"):
            return None
        # Pick a chunk in mid-2024
        target_ms = int(pd.Timestamp("2024-06-01", tz="UTC").timestamp() * 1000)
        chunks = [t for t in idx["timestamps"] if t <= target_ms]
        if not chunks:
            return None
        chunk_ts = max(chunks)
        rows = requests.get(
            f"https://www.smard.de/app/chart_data/{fid}/{region}/{fid}_{region}_quarterhour_{chunk_ts}.json",
            timeout=30,
        ).json().get("series", [])
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["t_ms", "v"])
        df["ts"] = pd.to_datetime(df["t_ms"], unit="ms", utc=True)
        return df.set_index("ts")["v"].dropna()
    except Exception as e:
        print(f"  {fid}/{region}: {e}", file=sys.stderr)
        return None


def main() -> None:
    df = pd.read_parquet(PARQUET)
    price_cols = [c for c in df.columns if c.startswith("price__")]

    print("=== Probing filter 4169-4200, region DE-LU ===")
    for fid in range(4169, 4201):
        s = fetch_recent_value(fid)
        if s is None or s.empty:
            continue
        ts = s.index[100] if len(s) > 100 else s.index[0]
        v = s.iloc[100] if len(s) > 100 else s.iloc[0]
        # Find best parquet column match for this single value
        if ts in df.index:
            row = df.loc[ts, price_cols]
            close = row[(row - v).abs() < 0.01]
            match = list(close.index) if len(close) > 0 else "no_match"
        else:
            match = f"ts {ts} not in parquet"
        print(f"  {fid}: ts={ts} v={v}  match={match}")


if __name__ == "__main__":
    main()
