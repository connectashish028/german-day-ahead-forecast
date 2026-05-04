"""SMARD CSV-download source.

For columns that aren't reachable via Energy-Charts API (the TSO load
+ generation forecasts), we fall back to manual CSV exports from
https://www.smard.de/en/downloadcenter/download-market-data.

The user drops the CSVs into the project root. We auto-discover them
by filename prefix:

  Forecasted_consumption_*.csv      -> fc_cons__*
  Forecasted_generation_Day-Ahead_*.csv  -> fc_gen__*

This is a pragmatic stop-gap until ENTSO-E credentials arrive. The
parsing reuses `data.data_clean` semantics (semicolon delimiter,
comma thousands separator, Berlin-local "Jan 1, 2022 12:00 AM" dates,
DST-aware tz localisation).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd

from ..data_clean import clean_col_name, parse_smard_csv

# Filename prefix -> column-name prefix mapping. The CSV's value-column
# headers (e.g. "grid load [MWh] Original resolutions") get cleaned and
# joined with the prefix below to match `schema.COLUMNS`.
PREFIXES: dict[str, str] = {
    "Forecasted_consumption":         "fc_cons",
    "Forecasted_generation_Day-Ahead": "fc_gen",
}

# Map "cleaned column name" (after stripping units etc.) to schema-column suffix.
# Lower-cased + non-alnum-to-underscore version of the CSV header.
COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "fc_cons": {
        "grid_load": "grid_load",
        "residual_load": "residual_load",
    },
    "fc_gen": {
        "total": "total",
        "photovoltaics_and_wind": "photovoltaics_and_wind",
        "wind_offshore": "wind_offshore",
        "wind_onshore": "wind_onshore",
        "photovoltaics": "photovoltaics",
        "other": "other",
    },
}


def _project_root() -> Path:
    # The CSVs live next to pyproject.toml.
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not locate project root.")


def _newest_csv_for_prefix(prefix: str) -> Path:
    root = _project_root()
    matches = sorted(root.glob(f"{prefix}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV with prefix {prefix!r} found in {root}")
    # Filenames encode their date range so lex-sort = chronological-end-sort.
    return matches[-1]


@cache
def _load_prefix(prefix: str) -> pd.DataFrame:
    """Parse the newest CSV with the given prefix and return a UTC-indexed DataFrame
    whose columns are the schema-suffixes (e.g. 'grid_load', 'residual_load')."""
    path = _newest_csv_for_prefix(prefix)
    df = parse_smard_csv(path)

    # Rename every value column from "grid load [MWh] Original resolutions" -> "grid_load".
    # parse_smard_csv returns tz-aware UTC timestamps; keep them tz-aware.
    idx = pd.DatetimeIndex(df["Start date"], name="timestamp")
    out = pd.DataFrame(index=idx)

    rename_map = COLUMN_RENAMES[PREFIXES[prefix]]
    for raw_col in df.columns:
        if raw_col in ("Start date", "End date"):
            continue
        cleaned = clean_col_name(raw_col, tag="").lstrip("_")
        if cleaned in rename_map:
            out[rename_map[cleaned]] = pd.to_numeric(df[raw_col].values, errors="coerce")
        # silently skip columns we don't need
    return out.sort_index()


def fetch(column, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return the requested column as a tz-aware UTC Series clipped to [start, end)."""
    name = column.name  # e.g. "fc_cons__grid_load"
    prefix_short, suffix = name.split("__", 1)
    # Find the file prefix that maps to this column-name prefix.
    file_prefix = next((fp for fp, sp in PREFIXES.items() if sp == prefix_short), None)
    if file_prefix is None:
        raise ValueError(f"No SMARD CSV prefix maps to schema column {name!r}")

    df = _load_prefix(file_prefix)
    if suffix not in df.columns:
        raise KeyError(
            f"Column {suffix!r} not found in {file_prefix} CSV. "
            f"Available: {list(df.columns)}"
        )
    s = df[suffix].rename(name)
    s = s.loc[(s.index >= start) & (s.index < end)]
    return s
