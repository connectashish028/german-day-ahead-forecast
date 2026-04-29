"""Type aliases for the backtest harness."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

PredictFn = Callable[[pd.DataFrame, pd.Timestamp], pd.Series]
"""A predictor takes (full_df, issue_time_utc) and returns a Series of length 96
indexed by tz-aware UTC target timestamps covering one Berlin calendar day at
quarter-hour resolution.

The harness passes the *full* DataFrame, not a time-sliced view, because some
columns (the TSO day-ahead forecast, NWP weather forecasts) are legitimately
known at issue time even though their target timestamps are in the future.

It is the predictor's (or its underlying feature pipeline's) responsibility to
respect availability per column:
- Actuals (consumption, generation): only timestamps strictly < issue_time.
- TSO day-ahead forecast columns: published ~D-1 10:00 CET, so the full delivery
  day is available at issue time D-1 12:00 CET.
- Day-ahead prices: published ~D-1 12:45 CET → available only as lagged features
  (D-2 prices and earlier).
- Weather forecasts: ICON-EU runs at 00/06/12/18 UTC; the run available at
  issue time D-1 12:00 CET (10:00 UTC winter, 11:00 UTC CEST) is the 06 UTC run.

The leakage tests in `tests/test_no_leakage.py` (Milestone 2) enforce these
rules feature-by-feature.
"""
