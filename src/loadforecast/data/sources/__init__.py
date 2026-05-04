"""Data sources for `refresh.py`. Each module exposes:

- `fetch(column, start, end) -> pd.Series`
  Returns a tz-aware UTC pandas Series indexed at 15-minute resolution
  for the requested column over [start, end].
"""
