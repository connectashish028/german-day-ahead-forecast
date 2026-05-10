"""Drift monitor — daily passive check that production models still work.

Runs in the daily GitHub Action after smoke_tomorrow_predict.py. For
yesterday's delivery day (where actuals are now realised), computes:

  - load model P50 MAE  (MW per quarter-hour, vs realised load)
  - price model P50 MAE (EUR/MWh, vs realised clearing price)

Appends one row per day to `backtest_results/drift_log.csv`. Reading the
log gives the model-health trace over time. A 14-day rolling mean
crossing 1.5× the original holdout baseline is the signal that
retraining is genuinely warranted.

This is a one-way observability layer — nothing breaks if drift fires.
The decision to retrain is human-gated (look at the trace, decide).
"""
from __future__ import annotations

import datetime as dt
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.predict import (
    lstm_quantile_predict_full,
    price_quantile_predict_full,
)

PARQUET = "smard_merged_15min.parquet"
ACTUAL_LOAD = "actual_cons__grid_load"
PRICE_COL = "price__germany_luxembourg"
LOG_CSV = Path("backtest_results/drift_log.csv")

# Historical holdout baselines (the production-MAE numbers in the README).
LOAD_BASELINE_MAE_MW = 393.0     # 70-day load holdout
PRICE_BASELINE_MAE = 23.8        # 61-day price holdout, EUR/MWh
DRIFT_MULT = 1.5
ROLL_WINDOW_DAYS = 14


def _most_recent_delivery_with_full_actuals(df: pd.DataFrame) -> date | None:
    """Walk back from yesterday until we find a date with all 96 quarter-
    hour actuals realised (no NaN). Caps at 14 days back."""
    today = dt.datetime.now(ZoneInfo("Europe/Berlin")).date()
    for offset in range(1, 14):
        d = today - timedelta(days=offset)
        target_idx = pd.date_range(
            start=pd.Timestamp(d, tz="Europe/Berlin").tz_convert("UTC"),
            periods=96, freq="15min",
        )
        if df[ACTUAL_LOAD].reindex(target_idx).notna().all():
            return d
    return None


def _compute_day_mae(df: pd.DataFrame, delivery_date: date) -> dict:
    """Predict and score both models for a delivery day. Returns None for
    a model whose forecast couldn't be built (NaN somewhere)."""
    issue = issue_time_for(delivery_date)
    target_idx = pd.date_range(
        start=pd.Timestamp(delivery_date, tz="Europe/Berlin").tz_convert("UTC"),
        periods=96, freq="15min",
    )

    load_actual = df[ACTUAL_LOAD].reindex(target_idx).to_numpy()
    load_fc = lstm_quantile_predict_full(df, issue)
    load_mae = (
        float(np.abs(load_actual - load_fc["p50"].to_numpy()).mean())
        if not load_fc["p50"].isna().any() and not np.isnan(load_actual).any()
        else None
    )

    price_actual = df[PRICE_COL].reindex(target_idx).to_numpy()
    price_fc = price_quantile_predict_full(df, issue)
    price_mae = (
        float(np.abs(price_actual - price_fc["p50"].to_numpy()).mean())
        if not price_fc["p50"].isna().any() and not np.isnan(price_actual).any()
        else None
    )

    return {"load_mae_mw": load_mae, "price_mae_eur": price_mae}


def _append_to_log(delivery: date, row: dict) -> pd.DataFrame:
    """Upsert one row per delivery date. Most recent at the bottom."""
    new = {"delivery_date": str(delivery), **row}
    if LOG_CSV.exists():
        log = pd.read_csv(LOG_CSV)
        log = log[log["delivery_date"] != str(delivery)]
    else:
        log = pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([new])], ignore_index=True)
    log["delivery_date"] = pd.to_datetime(log["delivery_date"])
    log = log.sort_values("delivery_date").reset_index(drop=True)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_CSV, index=False)
    return log


def _report_drift(log: pd.DataFrame) -> None:
    """Print rolling stats + a clear OK / ALERT banner."""
    if len(log) < 5:
        print(f"  log has {len(log)} rows — need >=5 for rolling stats.")
        return

    load_roll = log["load_mae_mw"].rolling(
        ROLL_WINDOW_DAYS, min_periods=5,
    ).mean().iloc[-1]
    price_roll = log["price_mae_eur"].rolling(
        ROLL_WINDOW_DAYS, min_periods=5,
    ).mean().iloc[-1]

    load_thresh = LOAD_BASELINE_MAE_MW * DRIFT_MULT
    price_thresh = PRICE_BASELINE_MAE * DRIFT_MULT
    load_alert = bool(not pd.isna(load_roll) and load_roll > load_thresh)
    price_alert = bool(not pd.isna(price_roll) and price_roll > price_thresh)

    print()
    print(f"  load  14d rolling: {load_roll:>6.1f} MW    "
          f"(baseline {LOAD_BASELINE_MAE_MW:.0f}, alert >{load_thresh:.0f})  "
          f"{'!! ALERT' if load_alert else 'ok'}")
    print(f"  price 14d rolling: {price_roll:>6.2f} EUR  "
          f"(baseline {PRICE_BASELINE_MAE:.1f}, alert >{price_thresh:.1f}) "
          f"{'!! ALERT' if price_alert else 'ok'}")

    if load_alert or price_alert:
        print("\n!! Drift detected. Time to look at the trace and consider retraining.")


def main() -> None:
    print(f"Drift monitor — loading {PARQUET}")
    df = load_smard_15min(PARQUET)

    delivery = _most_recent_delivery_with_full_actuals(df)
    if delivery is None:
        print("No recent date has full actuals. Skipping.")
        return

    print(f"Scoring delivery {delivery}...")
    row = _compute_day_mae(df, delivery)
    print(f"  load  MAE: "
          + (f"{row['load_mae_mw']:.1f} MW" if row['load_mae_mw'] is not None else "NaN — model couldn't predict"))
    print(f"  price MAE: "
          + (f"{row['price_mae_eur']:.2f} EUR/MWh" if row['price_mae_eur'] is not None else "NaN — model couldn't predict"))

    log = _append_to_log(delivery, row)
    print(f"Wrote {LOG_CSV} ({len(log)} rows)")

    _report_drift(log)


if __name__ == "__main__":
    main()
