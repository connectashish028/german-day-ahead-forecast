"""Per-column data-availability rules.

This module encodes a single source of truth for *when* each kind of column in
the SMARD master frame is known to a forecaster issuing predictions at issue
time T. The leakage tests in `tests/test_no_leakage.py` rely on these rules.

The rules are categorical, keyed by column-name prefix, because the SMARD
column convention is consistent:

    actual_cons__*     — measured consumption, ground truth.
    actual_gen__*      — measured generation by source.
    fc_cons__*         — TSO day-ahead consumption forecast (published ~D-1 10:00 CET).
    fc_gen__*          — TSO day-ahead generation forecast (published ~D-1 10:00 CET).
    price__*           — exchange day-ahead clearing prices (published ~D-1 12:45 CET).
    weather__*         — Open-Meteo NWP forecast at issue time. By construction
                         (the `historical-forecast` API returns the most recent
                         issued forecast), values for any timestamp `ts` reflect
                         a forecast that pre-dates `ts`. We treat them as
                         "forecast" (visible up to T + 48h).

The forecaster's issue time T is fixed at D-1 12:00 Europe/Berlin.

Availability semantics — a column's value at row timestamp `ts` is "available
at issue time T" iff:

    ACTUAL_*       :  ts < T                   (no future actuals, ever)
    FC_*           :  ts < T + 36h             (TSO publishes the full day-D
                                                series before T; the +36h cap
                                                is a defensive belt-and-braces
                                                covering D 00:00 -> D+1 00:00
                                                Berlin local in any timezone.)
    PRICE_*        :  ts < T + 12h             (Prices for delivery day D
                                                clear ~D-1 12:45, after T.
                                                But prices for delivery D-1
                                                (auctioned at D-2 12:00,
                                                published D-2 12:45) ARE
                                                known at T. So timestamps
                                                up to D 00:00 Berlin = T + 12h
                                                are usable.)
    OTHER          :  ts < T                   (default to the strictest rule.)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

ACTUAL_PREFIXES = ("actual_cons__", "actual_gen__")
FORECAST_PREFIXES = ("fc_cons__", "fc_gen__", "weather__")
PRICE_PREFIXES = ("price__",)


class ColumnKind:
    ACTUAL = "actual"
    FORECAST = "forecast"
    PRICE = "price"
    OTHER = "other"


@dataclass(frozen=True)
class AvailabilityRule:
    kind: str
    max_age_offset: pd.Timedelta  # value's timestamp must satisfy ts < issue_time + offset


RULES: dict[str, AvailabilityRule] = {
    ColumnKind.ACTUAL: AvailabilityRule(ColumnKind.ACTUAL, pd.Timedelta(0)),
    # TSO forecast for delivery day D extends to D+1 00:00 Berlin = at most ~36h
    # after issue time T = D-1 12:00 Berlin. Use 48h as a defensive cap.
    ColumnKind.FORECAST: AvailabilityRule(ColumnKind.FORECAST, pd.Timedelta(hours=48)),
    # Day-ahead prices for delivery day D clear ~D-1 12:45 (after T = D-1 12:00).
    # But prices for delivery D-1 cleared at D-2 12:45 — they ARE available at T.
    # The available window is therefore ts < D 00:00 Berlin = T + 12h.
    ColumnKind.PRICE: AvailabilityRule(ColumnKind.PRICE, pd.Timedelta(hours=12)),
    ColumnKind.OTHER: AvailabilityRule(ColumnKind.OTHER, pd.Timedelta(0)),
}


def classify_column(col: str) -> str:
    if col.startswith(ACTUAL_PREFIXES):
        return ColumnKind.ACTUAL
    if col.startswith(FORECAST_PREFIXES):
        return ColumnKind.FORECAST
    if col.startswith(PRICE_PREFIXES):
        return ColumnKind.PRICE
    return ColumnKind.OTHER


def is_available_at(col: str, ts: pd.Timestamp, issue_time: pd.Timestamp) -> bool:
    """True if column `col`'s value at timestamp `ts` is known to a forecaster
    issuing predictions at `issue_time`."""
    rule = RULES[classify_column(col)]
    return ts < issue_time + rule.max_age_offset


def usable_slice(df: pd.DataFrame, col: str, issue_time: pd.Timestamp) -> pd.Series:
    """Return `df[col]` restricted to rows whose timestamp is available at `issue_time`.

    Out-of-window values are replaced with NaN (rather than dropped) so the
    returned Series keeps the input index, which makes downstream alignment
    cleaner.
    """
    rule = RULES[classify_column(col)]
    cutoff = issue_time + rule.max_age_offset
    s = df[col].copy()
    s.loc[s.index >= cutoff] = float("nan")
    return s


def usable_columns(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    include: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of `df` (or the `include` subset) with future-leaking values masked."""
    cols = list(df.columns) if include is None else list(include)
    return pd.DataFrame({c: usable_slice(df, c, issue_time) for c in cols})
