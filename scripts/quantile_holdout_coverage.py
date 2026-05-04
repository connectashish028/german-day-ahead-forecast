"""Run the quantile model on the 70-date holdout window and report:
  - empirical coverage of [p10, p90]
  - mean interval width
  - per-day p50 skill
  - quantile crossings
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.backtest.baselines import tso_baseline_predict
from loadforecast.models.predict import lstm_quantile_predict_full

PARQUET = "smard_merged_15min.parquet"


def _drange(start, end, step=7):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def main() -> None:
    df = load_smard_15min(PARQUET)
    deliveries = list(_drange(date(2025, 1, 1), date(2026, 4, 30), 7))
    print(f"Backtesting quantile model on {len(deliveries)} delivery dates...")

    rows: list[dict] = []
    for d in deliveries:
        issue = issue_time_for(d)
        target_idx = pd.date_range(
            issue.tz_convert("Europe/Berlin").normalize() + pd.Timedelta(days=1),
            periods=96, freq="15min", tz="Europe/Berlin",
        ).tz_convert("UTC")
        actual = df["actual_cons__grid_load"].reindex(target_idx)
        if actual.isna().any():
            continue
        out = lstm_quantile_predict_full(df, issue)
        if out["p50"].isna().any():
            continue
        for ts, p10, p50, p90, y in zip(
            out.index, out["p10"], out["p50"], out["p90"], actual,
            strict=True,
        ):
            rows.append({
                "issue_date": str(d),
                "target_ts": ts,
                "y_true": float(y),
                "p10": float(p10),
                "p50": float(p50),
                "p90": float(p90),
            })

    bt = pd.DataFrame(rows)
    bt["inside"] = (bt["y_true"] >= bt["p10"]) & (bt["y_true"] <= bt["p90"])
    bt["width"] = bt["p90"] - bt["p10"]
    bt["abs_err_p50"] = (bt["y_true"] - bt["p50"]).abs()
    bt["crossing"] = (bt["p50"] < bt["p10"]) | (bt["p90"] < bt["p50"])

    cov = bt["inside"].mean()
    mean_w = bt["width"].mean()
    crossings = bt["crossing"].mean()
    p50_mae = bt["abs_err_p50"].mean()

    # TSO MAE for skill comparison
    tso_mae = float(np.abs(
        df["actual_cons__grid_load"].reindex(bt["target_ts"])
        - df["fc_cons__grid_load"].reindex(bt["target_ts"])
    ).mean())

    print(f"\nHoldout (n_days = {bt['issue_date'].nunique()}):")
    print(f"  Interval [p10, p90] empirical coverage: {cov:.3%}   (target 80%)")
    print(f"  Mean interval width:                    {mean_w:>7.1f} MW")
    print(f"  Quantile crossings:                     {crossings:.3%}")
    print(f"  P50 MAE:                                {p50_mae:>7.1f} MW")
    print(f"  TSO MAE (same window):                  {tso_mae:>7.1f} MW")
    print(f"  P50 skill:                              {1 - p50_mae / tso_mae:+.4f}")

    bt.to_csv("backtest_results/lstm_quantile_full_step7.csv", index=False)
    print(f"\nWrote backtest_results/lstm_quantile_full_step7.csv")


if __name__ == "__main__":
    main()
