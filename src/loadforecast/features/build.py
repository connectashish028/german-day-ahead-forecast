"""Build the leakage-safe feature matrix for a single delivery day.

The returned frame has 96 rows (one per quarter-hour of the Berlin calendar
delivery day) and is intended as the `X` for a tabular model, or as the
"future-known covariates" for a seq2seq decoder.
"""

from __future__ import annotations

import pandas as pd

from ..backtest.loader import target_index_for
from .availability import usable_columns
from .calendar import calendar_features
from .lags import lag_at, lag_features, rolling_stat

ACTUAL_LOAD = "actual_cons__grid_load"
ACTUAL_RESIDUAL = "actual_cons__residual_load"
TSO_LOAD_FC = "fc_cons__grid_load"
TSO_RESIDUAL_FC = "fc_cons__residual_load"
TSO_VRE_FC = "fc_gen__photovoltaics_and_wind"
TSO_TOTAL_GEN_FC = "fc_gen__total"
DE_PRICE = "price__germany_luxembourg"
NEIGHBOUR_PRICES = (
    "price__france",
    "price__netherlands",
    "price__austria",
    "price__czech_republic",
    "price__poland",
    "price__switzerland",
)

# NOTE: D-1 (lag 1d) lookups are only half-available at issue time D-1 12:00 —
# the second-half of D-1 (12:00 onwards in Berlin local) is post-issue.
# Standard practice: use the most-recent fully-available day instead (D-2),
# plus weekly seasonal lags.
LAG_DAYS_LOAD = (2.0, 7.0, 14.0)
LAG_DAYS_PRICE = (2.0, 7.0)  # day-ahead prices for D-1 are not yet known at issue time
ROLLING_WINDOWS_DAYS = (1.0, 7.0)


def _delivery_target_index(issue_time: pd.Timestamp) -> pd.DatetimeIndex:
    delivery_local = issue_time.tz_convert("Europe/Berlin").normalize() + pd.Timedelta(days=1)
    return target_index_for(delivery_local.date())


def build_target_day_features(df: pd.DataFrame, issue_time: pd.Timestamp) -> pd.DataFrame:
    """Return a DataFrame of features indexed by the 96 target timestamps for the
    delivery day, with no future-leaking values.

    Sections:
      - Calendar: deterministic from the index.
      - TSO forecast: directly usable (entire delivery day is known by issue time).
      - Lag features on actual load and residual: D-1 lags use the most recent
        full day available before issue_time.
      - Rolling stats on actual load: 1d / 7d windows ending before issue_time.
      - Day-ahead price lags: D-2 and D-7 (D-1 not yet known).
    """
    target_idx = _delivery_target_index(issue_time)

    # Mask future-leaking values according to availability rules.
    needed_cols = (
        ACTUAL_LOAD,
        ACTUAL_RESIDUAL,
        TSO_LOAD_FC,
        TSO_RESIDUAL_FC,
        TSO_VRE_FC,
        TSO_TOTAL_GEN_FC,
        DE_PRICE,
        *NEIGHBOUR_PRICES,
    )
    masked = usable_columns(df, issue_time, include=needed_cols)

    parts: list[pd.DataFrame] = []

    # Calendar
    parts.append(calendar_features(target_idx))

    # TSO forecast features for the delivery day itself
    tso_block = pd.DataFrame(
        {
            "tso_load_fc": masked[TSO_LOAD_FC].reindex(target_idx),
            "tso_residual_fc": masked[TSO_RESIDUAL_FC].reindex(target_idx),
            "tso_vre_fc": masked[TSO_VRE_FC].reindex(target_idx),
            "tso_total_gen_fc": masked[TSO_TOTAL_GEN_FC].reindex(target_idx),
        }
    )
    tso_block["tso_residual_share"] = tso_block["tso_residual_fc"] / tso_block["tso_load_fc"].replace(
        0, float("nan")
    )
    parts.append(tso_block)

    # Lag features on actual load + residual (1d, 7d, 14d)
    parts.append(lag_features(masked, target_idx, ACTUAL_LOAD, LAG_DAYS_LOAD, prefix="load"))
    parts.append(lag_features(masked, target_idx, ACTUAL_RESIDUAL, LAG_DAYS_LOAD, prefix="residual"))

    # Rolling stats on actual load (1d/7d mean/std), ending strictly before issue_time
    end_offset = (target_idx[0] - issue_time) / pd.Timedelta(days=1)  # days from issue->target start
    rolling_block = pd.DataFrame(index=target_idx)
    for w in ROLLING_WINDOWS_DAYS:
        for stat in ("mean", "std"):
            rolling_block[f"load_roll_{int(w)}d_{stat}"] = rolling_stat(
                masked[ACTUAL_LOAD],
                target_idx,
                window_days=w,
                stat=stat,
                end_offset_days=end_offset,
            )
    parts.append(rolling_block)

    # Day-ahead price lags (D-2 and D-7)
    price_lag_block = lag_features(masked, target_idx, DE_PRICE, LAG_DAYS_PRICE, prefix="de_price")
    for col in NEIGHBOUR_PRICES:
        nb_pfx = col.replace("price__", "price_")
        nb_block = lag_features(masked, target_idx, col, LAG_DAYS_PRICE, prefix=nb_pfx)
        price_lag_block = pd.concat([price_lag_block, nb_block], axis=1)
    parts.append(price_lag_block)

    # The residual-of-TSO target lag (1d, 7d): residual = actual_load - TSO_fc.
    # Useful both as a feature (recent residuals predict near-term residual) and
    # because the model's *target* will be this same residual on the delivery day.
    residual_recent = masked[ACTUAL_LOAD] - masked[TSO_LOAD_FC]
    parts.append(
        pd.DataFrame(
            {
                f"tso_residual_err__lag_{int(ld * 96)}qh": lag_at(residual_recent, target_idx, ld)
                for ld in (2.0, 7.0)
            }
        )
    )

    out = pd.concat(parts, axis=1)
    out.index.name = "target_ts"
    return out


def target_residual(df: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """The training target: `actual_load - TSO_forecast` over the 96 delivery-day QHs.

    This requires post-issue actuals and is therefore only usable in training/
    backtesting (where ground truth is known). The forecaster never receives this
    in production.
    """
    target_idx = _delivery_target_index(issue_time)
    return (df[ACTUAL_LOAD] - df[TSO_LOAD_FC]).reindex(target_idx).rename("y_residual")


__all__ = ["build_target_day_features", "target_residual"]
