"""Backtest the price quantile model on the holdout window
2026-03-01 → 2026-04-30, vs two naive baselines.

Naive baselines:
  - "yesterday-same-quarter-hour" (1-day lag)
  - "last-week-same-quarter-hour" (7-day lag)

Reports:
  - P50 MAE / MAPE
  - 80 % band coverage and width
  - **Spread MAE**: |actual_spread - predicted_spread| where spread = max - min
    within the day. This is what battery-dispatch P&L cares about.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.dataset import FeatureScaler
from loadforecast.models.price_dataset import PRICE, build_price_window

PARQUET = "smard_merged_15min.parquet"
MODEL_DIR = Path("model_checkpoints/price_quantile_v4")
OUT_CSV = Path("backtest_results/price_quantile_holdout.csv")
OUT_CSV_MASKED = Path("backtest_results/price_quantile_holdout_vre_masked.csv")

# Decoder feature indices (must match PRICE_DEC_FEATURE_NAMES order).
VRE_FC_COL_IDX = 1
VRE_PRESENT_COL_IDX = 2
VRE_RATIO_COL_IDX = 3
VRE_PCTILE_COL_IDX = 4


def _drange(start: date, end: date, step: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def _load_model():
    from tensorflow import keras
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    scaler_npz = np.load(MODEL_DIR / "scaler.npz")
    scaler = FeatureScaler(
        enc_mean=scaler_npz["enc_mean"], enc_std=scaler_npz["enc_std"],
        dec_mean=scaler_npz["dec_mean"], dec_std=scaler_npz["dec_std"],
        y_mean=float(scaler_npz["y_mean"]), y_std=float(scaler_npz["y_std"]),
    )
    model = keras.models.load_model(MODEL_DIR / "model.keras", compile=False)
    return model, scaler, meta


def _predict_quantile(model, scaler, df, issue, *, mask_vre: bool = False):
    """Forecast a single delivery day.

    `mask_vre=True` simulates the production case where SMARD's VRE day-ahead
    module hasn't published yet — we zero out tso_vre_fc and its present flag
    BEFORE scaling so the model uses its degraded-mode path.
    """
    w = build_price_window(df, issue, include_weather=True)
    if np.isnan(w.X_enc).any() or np.isnan(w.X_dec).any():
        return None
    X_dec = w.X_dec.copy()
    if mask_vre:
        X_dec[..., VRE_FC_COL_IDX] = 0.0
        X_dec[..., VRE_PRESENT_COL_IDX] = 0.0
        X_dec[..., VRE_RATIO_COL_IDX] = 0.0
        X_dec[..., VRE_PCTILE_COL_IDX] = 0.0
    Xe, Xd = scaler.transform(w.X_enc[None, ...], X_dec[None, ...])
    raw = model.predict([Xe, Xd], verbose=0)
    y_norm = raw[0]
    y_quant = scaler.inverse_y(y_norm)  # (96, 3)
    return pd.DataFrame(
        {"p10": y_quant[:, 0], "p50": y_quant[:, 1], "p90": y_quant[:, 2]},
        index=w.target_idx,
    ), w.y_price


def _naive_lag(df, target_idx, days_back):
    """Yesterday (days_back=1) or last-week (days_back=7) prices, aligned to
    the delivery day's quarter-hour grid."""
    lagged_idx = target_idx - pd.Timedelta(days=days_back)
    return df[PRICE].reindex(lagged_idx).set_axis(target_idx)


def _run_holdout(df, model, scaler, holdout, *, mask_vre: bool):
    rows = []
    for d in holdout:
        issue = issue_time_for(d)
        result = _predict_quantile(model, scaler, df, issue, mask_vre=mask_vre)
        if result is None:
            continue
        forecast, y_true = result
        naive_1d = _naive_lag(df, forecast.index, 1).to_numpy()
        naive_7d = _naive_lag(df, forecast.index, 7).to_numpy()
        for ts, p10, p50, p90, n1, n7, y in zip(
            forecast.index, forecast["p10"], forecast["p50"], forecast["p90"],
            naive_1d, naive_7d, y_true, strict=True,
        ):
            rows.append({
                "issue_date": str(d),
                "target_ts": ts,
                "y_true": float(y),
                "p10": float(p10), "p50": float(p50), "p90": float(p90),
                "naive_1d": float(n1) if not np.isnan(n1) else np.nan,
                "naive_7d": float(n7) if not np.isnan(n7) else np.nan,
            })
    return pd.DataFrame(rows)


def _summarise(bt: pd.DataFrame, label: str) -> dict:
    bt["abs_err_model"] = (bt["y_true"] - bt["p50"]).abs()
    bt["abs_err_naive_1d"] = (bt["y_true"] - bt["naive_1d"]).abs()
    bt["abs_err_naive_7d"] = (bt["y_true"] - bt["naive_7d"]).abs()
    bt["inside_band"] = (bt["y_true"] >= bt["p10"]) & (bt["y_true"] <= bt["p90"])
    bt["band_width"] = bt["p90"] - bt["p10"]

    n_days = bt["issue_date"].nunique()
    mae_model = float(bt["abs_err_model"].mean())
    mae_n1 = float(bt["abs_err_naive_1d"].mean(skipna=True))
    mae_n7 = float(bt["abs_err_naive_7d"].mean(skipna=True))
    cov = float(bt["inside_band"].mean())
    width = float(bt["band_width"].mean())

    # Spread MAE — what battery dispatch cares about.
    daily_spread = bt.groupby("issue_date").agg(
        true_spread=("y_true", lambda s: s.max() - s.min()),
        model_spread=("p50", lambda s: s.max() - s.min()),
        naive_1d_spread=("naive_1d", lambda s: s.max() - s.min()),
    )
    spread_mae_model = float(
        (daily_spread["true_spread"] - daily_spread["model_spread"]).abs().mean()
    )
    spread_mae_naive = float(
        (daily_spread["true_spread"] - daily_spread["naive_1d_spread"]).abs().mean()
    )

    print()
    print("=" * 60)
    print(f"{label}  -  {n_days} days, {len(bt):,} 15-min slots")
    print("=" * 60)
    print(f"\nP50 point MAE (EUR/MWh):")
    print(f"  Model            : {mae_model:>7.2f}")
    print(f"  Naive yesterday  : {mae_n1:>7.2f}   (improvement: {(1-mae_model/mae_n1)*100:+.1f} %)")
    print(f"  Naive last-week  : {mae_n7:>7.2f}   (improvement: {(1-mae_model/mae_n7)*100:+.1f} %)")
    print(f"\nDaily price-spread MAE (EUR/MWh):")
    print(f"  Model            : {spread_mae_model:>7.2f}")
    print(f"  Naive yesterday  : {spread_mae_naive:>7.2f}   "
          f"(improvement: {(1-spread_mae_model/spread_mae_naive)*100:+.1f} %)")
    print(f"\n80 % band:")
    print(f"  Empirical coverage: {cov:.3%}")
    print(f"  Mean width:         {width:.1f} EUR/MWh")
    return {
        "n_days": n_days, "mae_model": mae_model, "mae_n1": mae_n1,
        "mae_n7": mae_n7, "cov": cov, "width": width,
        "spread_mae_model": spread_mae_model, "spread_mae_naive": spread_mae_naive,
    }


def main() -> None:
    print("Loading parquet + model...")
    df = load_smard_15min(PARQUET)
    model, scaler, meta = _load_model()
    print(f"  model: trained {meta['train_n']} days, "
          f"val P50 MAE = {meta['val_p50_mae_eur_mwh']:.2f} EUR/MWh")

    holdout = list(_drange(date(2026, 3, 1), date(2026, 4, 30)))
    print(f"\nBacktesting {len(holdout)} holdout days in BOTH modes...")

    # Full-feature mode (production reality after SMARD VRE publishes).
    bt_full = _run_holdout(df, model, scaler, holdout, mask_vre=False)
    s_full = _summarise(bt_full, "FULL FEATURES")

    # Degraded mode (production reality before SMARD publishes — typical
    # morning hours when the desk is shaping bids).
    bt_masked = _run_holdout(df, model, scaler, holdout, mask_vre=True)
    s_masked = _summarise(bt_masked, "VRE MASKED (degraded mode)")

    print()
    print("=" * 60)
    print("Mode comparison (degraded vs full)")
    print("=" * 60)
    print(f"  P50 MAE         : {s_masked['mae_model']:.2f}  vs  {s_full['mae_model']:.2f} EUR/MWh "
          f"(+{s_masked['mae_model'] - s_full['mae_model']:.2f}, "
          f"{(s_masked['mae_model']/s_full['mae_model']-1)*100:+.1f} %)")
    print(f"  Spread MAE      : {s_masked['spread_mae_model']:.2f}  vs  {s_full['spread_mae_model']:.2f} EUR/MWh")
    print(f"  Beats naive 1d  : full {(1-s_full['mae_model']/s_full['mae_n1'])*100:+.1f} %  |  "
          f"masked {(1-s_masked['mae_model']/s_masked['mae_n1'])*100:+.1f} %")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    bt_full.to_csv(OUT_CSV, index=False)
    bt_masked.to_csv(OUT_CSV_MASKED, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_CSV_MASKED}")


if __name__ == "__main__":
    main()
