"""
Clean and join SMARD CSV exports into a single parquet file.

Handles all 4 files from the Download Center:
  - Actual_consumption_*.csv
  - Actual_generation_*.csv
  - Day-ahead_prices_*.csv
  - Forecasted_consumption_*.csv
  - Forecasted_generation_Day-Ahead_*.csv

SMARD CSV quirks this handles:
  - Semicolon delimiter, comma thousands separator ("11,273.25" -> 11273.25)
  - Date format "Jan 1, 2022 12:00 AM" (CET/CEST local time)
  - Missing values as "-" or blank
  - All files share "Start date" / "End date" columns -> join on Start date
  - Column names like "[MWh] Original resolutions" are trimmed for readability
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---------------- config ----------------
INPUT_DIR = Path(".")           # folder containing the CSVs
OUTPUT_PARQUET = Path("smard_merged_15min.parquet")
OUTPUT_CSV_PREVIEW = Path("smard_merged_15min_head.csv")  # small preview file
TIMEZONE = "Europe/Berlin"      # SMARD exports are in local German time
RESOLUTION = "15min"            # "15min" or "1h" (aggregation below)
AGG_RULE = "smart"              # "sum", "mean", or "smart" (sum for MWh, mean for prices)

# Columns to drop from the merged output. Matches any column whose name contains
# one of these substrings (case-insensitive), so "nuclear" catches both
# "actual_gen__nuclear" and "fc_gen__nuclear" if either exists.
DROP_COLUMNS = ["nuclear"]

# Map filename prefixes to short, code-friendly source tags.
# Any CSV not matched falls back to a slug of its filename.
SOURCE_TAG_PATTERNS = {
    r"^Actual_consumption":            "actual_cons",
    r"^Actual_generation":             "actual_gen",
    r"^Day-ahead_prices":              "price",
    r"^Forecasted_consumption":        "fc_cons",
    r"^Forecasted_generation":         "fc_gen",
}


# ---------------- helpers ----------------
def tag_for(filename: str) -> str:
    for pat, tag in SOURCE_TAG_PATTERNS.items():
        if re.match(pat, filename):
            return tag
    return re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_")


def clean_col_name(col: str, tag: str) -> str:
    """Turn 'grid load [MWh] Original resolutions' -> 'actual_cons__grid_load'."""
    # strip unit brackets and "Original resolutions" suffix
    name = re.sub(r"\[.*?\]", "", col)
    name = re.sub(r"Original resolutions?", "", name, flags=re.IGNORECASE)
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return f"{tag}__{name}" if name else tag


def parse_smard_csv(path: Path) -> pd.DataFrame:
    # SMARD uses ';' delimiter and ',' as thousands separator with '.' decimal.
    df = pd.read_csv(
        path,
        sep=";",
        thousands=",",
        decimal=".",
        na_values=["-", "", "n/a", "N/A"],
        dtype=str,           # read as str first, then coerce numerics ourselves
        encoding="utf-8",
    )

    # Parse timestamps. SMARD format: "Jan 1, 2022 12:00 AM"
    for c in ("Start date", "End date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%b %d, %Y %I:%M %p", errors="coerce")

    # Localize to Berlin time, then convert to UTC for a canonical index.
    # SMARD timestamps are "wall clock" in Berlin. DST quirks:
    #   - Oct fall-back: the hour 02:00-03:00 appears twice. SMARD exports the
    #     summer (CEST) occurrence first, then the winter (CET) one, so we mark
    #     the first duplicate as DST=True and the second as DST=False.
    #   - Mar spring-forward: the hour 02:00-03:00 doesn't exist — those rows
    #     shouldn't be in SMARD files, but we use shift_forward just in case.
    # `infer` fails on quarter-hour data because pandas can't tell summer from
    # winter from 15-min stamps alone, hence the explicit boolean array below.
    def _resolve_ambiguous(series: pd.Series) -> pd.array:
        """Build the per-row DST flag: True = DST (first occurrence), False = standard."""
        # Find rows that share a local timestamp with another row (the duplicates).
        dup_mask = series.duplicated(keep=False)
        # Within each duplicate group, the earlier-indexed row is DST (summer).
        is_first_of_dup = dup_mask & ~series.duplicated(keep="first")
        # Default to True (DST) for non-duplicated rows — they're unambiguous
        # anyway, so the flag doesn't matter.
        flags = pd.Series(True, index=series.index)
        flags[dup_mask & ~is_first_of_dup] = False  # second occurrence -> standard time
        return flags.values

    for col in ("Start date", "End date"):
        if col not in df.columns:
            continue
        flags = _resolve_ambiguous(df[col])
        df[col] = (
            df[col]
            .dt.tz_localize(TIMEZONE, ambiguous=flags, nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )

    # Coerce all non-date columns to numeric. pd.read_csv with thousands="," handled
    # most of it, but because we read as str we do it explicitly here.
    for c in df.columns:
        if c in ("Start date", "End date"):
            continue
        # strip thousand separators that survived (belt-and-braces)
        s = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
        df[c] = pd.to_numeric(s, errors="coerce")

    return df


def load_and_tag(path: Path) -> pd.DataFrame:
    tag = tag_for(path.name)
    df = parse_smard_csv(path)
    # Rename value columns with the tag prefix.
    rename = {c: clean_col_name(c, tag) for c in df.columns
              if c not in ("Start date", "End date")}
    df = df.rename(columns=rename)
    # Keep Start date as the join key; drop End date (reconstructible from resolution).
    df = df.drop(columns=["End date"], errors="ignore")
    df = df.rename(columns={"Start date": "timestamp"})
    return df.set_index("timestamp").sort_index()


def infer_agg(colname: str) -> str:
    """Return 'sum' for energy flows (MWh), 'mean' for prices/power."""
    if colname.startswith("price__") or "price" in colname:
        return "mean"
    return "sum"   # MWh quantities add up cleanly; MW power would use mean


# ---------------- main ----------------
def main():
    csv_paths = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSVs found in {INPUT_DIR.resolve()}")

    frames = []
    for p in csv_paths:
        print(f"loading {p.name}")
        df = load_and_tag(p)
        print(f"  {df.shape[0]:>7} rows, {df.shape[1]} cols, "
              f"{df.index.min()} -> {df.index.max()}")
        frames.append(df)

    # Outer join on timestamp so gaps in any single series don't drop other series.
    merged = pd.concat(frames, axis=1, join="outer").sort_index()

    # Deduplicate columns that might appear in multiple files (unlikely but safe).
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # Drop unwanted columns (substring match, case-insensitive).
    if DROP_COLUMNS:
        patterns = [p.lower() for p in DROP_COLUMNS]
        to_drop = [c for c in merged.columns
                   if any(p in c.lower() for p in patterns)]
        if to_drop:
            print(f"\ndropping {len(to_drop)} column(s): {to_drop}")
            merged = merged.drop(columns=to_drop)

    print(f"\nmerged 15-min shape: {merged.shape}")
    print(f"range: {merged.index.min()} -> {merged.index.max()}")
    print(f"columns ({len(merged.columns)}):")
    for c in merged.columns:
        nn = merged[c].notna().sum()
        print(f"  {c:<60} {nn:>7} non-null")

    # Optional hourly aggregation
    if RESOLUTION == "1h":
        agg = {c: infer_agg(c) for c in merged.columns} if AGG_RULE == "smart" else AGG_RULE
        merged = merged.resample("1h").agg(agg)
        print(f"\nresampled to hourly: {merged.shape}")

    # Write outputs
    merged.to_parquet(OUTPUT_PARQUET)
    merged.head(200).to_csv(OUTPUT_CSV_PREVIEW)
    print(f"\nwrote {OUTPUT_PARQUET} ({OUTPUT_PARQUET.stat().st_size/1e6:.1f} MB)")
    print(f"wrote {OUTPUT_CSV_PREVIEW} (first 200 rows, for sanity check)")


if __name__ == "__main__":
    main()