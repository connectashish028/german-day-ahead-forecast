"""Compare the freshly rebuilt parquet against the old CSVs for the overlapping
window (2022-01-01 -> 2026-03-02). Any column where mean|diff| is meaningful
points to a source change between the old SMARD-CSV pipeline and the new
energy-charts/SMARD-API pipeline.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from loadforecast.data.data_clean import parse_smard_csv

PARQUET = "smard_merged_15min.parquet"

# Old CSVs (the pre-2026-03-02 ones — same prefix, different end date)
OLD_CSVS = {
    "Actual_consumption": Path("Actual_consumption_202201010000_202603020000_Quarterhour.csv"),
    "Actual_generation":  Path("Actual_generation_202201010000_202603020000_Quarterhour.csv"),
    "Day-ahead_prices":   Path("Day-ahead_prices_202201010000_202603020000_Quarterhour.csv"),
}

# Map (csv prefix, csv-cleaned-suffix) -> parquet column name
COL_MAP = {
    ("Actual_consumption", "grid_load"):                  "actual_cons__grid_load",
    ("Actual_consumption", "residual_load"):              "actual_cons__residual_load",
    ("Actual_generation",  "biomass"):                    "actual_gen__biomass",
    ("Actual_generation",  "wind_offshore"):              "actual_gen__wind_offshore",
    ("Actual_generation",  "wind_onshore"):               "actual_gen__wind_onshore",
    ("Actual_generation",  "photovoltaics"):              "actual_gen__photovoltaics",
    ("Actual_generation",  "lignite"):                    "actual_gen__lignite",
    ("Actual_generation",  "hard_coal"):                  "actual_gen__hard_coal",
    ("Day-ahead_prices",   "germany_luxembourg"):         "price__germany_luxembourg",
    ("Day-ahead_prices",   "france"):                     "price__france",
    ("Day-ahead_prices",   "netherlands"):                "price__netherlands",
    ("Day-ahead_prices",   "austria"):                    "price__austria",
    ("Day-ahead_prices",   "switzerland"):                "price__switzerland",
}


def _clean(name: str) -> str:
    """Same cleanup data_clean.py uses, normalised to a short suffix."""
    import re
    n = re.sub(r"\[.*?\]", "", name)
    n = re.sub(r"Original resolutions?", "", n, flags=re.IGNORECASE)
    n = n.strip().lower()
    n = re.sub(r"[^a-z0-9]+", "_", n).strip("_")
    return n


def main() -> None:
    new = pd.read_parquet(PARQUET)
    print(f"New parquet:  rows={len(new)}, cols={new.shape[1]}")
    print(f"  range: {new.index.min()}  ->  {new.index.max()}")

    overlap_end = pd.Timestamp("2026-03-01 23:45", tz="UTC")
    overlap_start = pd.Timestamp("2022-01-01", tz="UTC")
    new_overlap = new.loc[overlap_start:overlap_end]
    print(f"\nOverlap window: {overlap_start} -> {overlap_end}  ({len(new_overlap)} rows)")
    print()

    rows = []
    for prefix, path in OLD_CSVS.items():
        if not path.exists():
            print(f"  !! missing CSV: {path}")
            continue
        df_csv = parse_smard_csv(path)
        df_csv = df_csv.set_index("Start date")
        for col_csv in df_csv.columns:
            if col_csv == "End date":
                continue
            suffix = _clean(col_csv)
            new_name = COL_MAP.get((prefix, suffix))
            if new_name is None or new_name not in new.columns:
                continue
            old_series = pd.to_numeric(df_csv[col_csv], errors="coerce")
            new_series = new_overlap[new_name]
            common_idx = old_series.index.intersection(new_series.index)
            if len(common_idx) < 100:
                continue
            o = old_series.reindex(common_idx)
            n = new_series.reindex(common_idx)
            diff = (o - n)
            both = o.notna() & n.notna()
            if not both.any():
                continue
            rows.append({
                "column": new_name,
                "n_compared": int(both.sum()),
                "old_only": int(o.notna().sum() - both.sum()),
                "new_only": int(n.notna().sum() - both.sum()),
                "mean_abs_diff": float(diff.abs().where(both).mean()),
                "max_abs_diff": float(diff.abs().where(both).max()),
                "old_mean": float(o.where(both).mean()),
                "new_mean": float(n.where(both).mean()),
            })
    out = pd.DataFrame(rows).sort_values("mean_abs_diff", ascending=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
