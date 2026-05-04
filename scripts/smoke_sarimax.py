"""Smoke-test the SARIMAX baseline on a single date."""
from datetime import date
import time
import numpy as np

from loadforecast.backtest import load_smard_15min, issue_time_for
from loadforecast.backtest.sarimax_baseline import sarimax_residual_predict
from loadforecast.backtest.baselines import tso_baseline_predict, seasonal_naive_predict


def main() -> None:
    df = load_smard_15min("smard_merged_15min.parquet")
    for d in [date(2025, 6, 15), date(2025, 12, 1), date(2026, 2, 14)]:
        issue = issue_time_for(d)
        t0 = time.time()
        pred = sarimax_residual_predict(df, issue)
        dt = time.time() - t0
        tso = tso_baseline_predict(df, issue)
        naive = seasonal_naive_predict(df, issue)
        actual = df["actual_cons__grid_load"].reindex(pred.index)
        print(
            f"{d} | fit {dt:5.2f}s | "
            f"MAE tso={np.abs(actual - tso).mean():6.1f}  "
            f"sar={np.abs(actual - pred).mean():6.1f}  "
            f"naive={np.abs(actual - naive).mean():6.1f}  "
            f"NaN={pred.isna().sum()}/96"
        )


if __name__ == "__main__":
    main()
