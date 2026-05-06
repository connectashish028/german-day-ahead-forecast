"""Train the day-ahead price forecaster.

Mirrors train_lstm_quantile.py (the load model) but with:
  - target = raw price (not load residual)
  - encoder feature mix from price_dataset.py
  - training window starting 2024-01-01 (avoid the 2022 €500 outlier era)

Saves to model_checkpoints/price_quantile_v1/.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.dataset import FeatureScaler
from loadforecast.models.lstm_quantile import (
    QUANTILES,
    build_lstm_quantile,
    compile_lstm_quantile,
)
from loadforecast.models.price_dataset import (
    PRICE_DEC_FEATURE_NAMES,
    PRICE_ENC_FEATURE_NAMES,
    build_price_dataset,
)

PARQUET = "smard_merged_15min.parquet"
OUT_DIR = Path("model_checkpoints/price_quantile_v1")


def _drange(start: date, end: date, step: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def main() -> None:
    print("Loading parquet...")
    df = load_smard_15min(PARQUET)

    # Splits — see notebook 09 section 3 for justification.
    train_dates = [issue_time_for(d) for d in _drange(date(2024, 1, 8),  date(2025, 12, 31))]
    val_dates   = [issue_time_for(d) for d in _drange(date(2026, 1, 1),  date(2026, 2, 28))]

    print(f"\nBuilding price windows: {len(train_dates)} train, {len(val_dates)} val")
    Xe_tr, Xd_tr, Y_tr, kept_tr = build_price_dataset(
        df, train_dates, include_weather=True,
    )
    Xe_va, Xd_va, Y_va, kept_va = build_price_dataset(
        df, val_dates,   include_weather=True,
    )
    print(f"  kept: train={len(kept_tr)}, val={len(kept_va)}")
    print(f"  shapes: X_enc={Xe_tr.shape}  X_dec={Xd_tr.shape}  Y={Y_tr.shape}")
    print(f"  encoder features: {PRICE_ENC_FEATURE_NAMES} + 4 weather")
    print(f"  decoder features: {PRICE_DEC_FEATURE_NAMES} + 4 weather")
    print(f"  target stats: mean={Y_tr.mean():.1f} std={Y_tr.std():.1f} "
          f"min={Y_tr.min():.1f} max={Y_tr.max():.1f}")

    scaler = FeatureScaler.fit(Xe_tr, Xd_tr, Y_tr)
    Xe_tr_n, Xd_tr_n, Y_tr_n = scaler.transform(Xe_tr, Xd_tr, Y_tr)
    Xe_va_n, Xd_va_n, Y_va_n = scaler.transform(Xe_va, Xd_va, Y_va)

    print("\nBuilding model...")
    from tensorflow import keras
    model = compile_lstm_quantile(
        build_lstm_quantile(
            hidden=64,
            enc_features=Xe_tr.shape[-1],
            dec_features=Xd_tr.shape[-1],
        ),
        lr=1e-3,
    )
    model.summary(line_length=100)
    print(f"\nQuantiles: {QUANTILES}")

    print("\nTraining...")
    t0 = time.time()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5, verbose=1,
        ),
    ]
    history = model.fit(
        [Xe_tr_n, Xd_tr_n], Y_tr_n,
        validation_data=([Xe_va_n, Xd_va_n], Y_va_n),
        epochs=60, batch_size=32, callbacks=callbacks, verbose=2,
    )
    train_time = time.time() - t0
    print(f"\nTrained in {train_time:.0f}s ({len(history.epoch)} epochs)")

    # Validation diagnostics — same shape as the load quantile script.
    pred_va_n = model.predict([Xe_va_n, Xd_va_n], verbose=0)
    pred_va = scaler.inverse_y(pred_va_n)
    p10 = pred_va[..., 0]
    p50 = pred_va[..., 1]
    p90 = pred_va[..., 2]

    val_p50_mae = float(np.abs(Y_va - p50).mean())
    val_mean_abs_y = float(np.abs(Y_va).mean())
    inside = float(((Y_va >= p10) & (Y_va <= p90)).mean())
    avg_width = float((p90 - p10).mean())
    crossings = float(((p90 < p50) | (p50 < p10)).mean())

    print(f"\nValidation P50 MAE:                {val_p50_mae:>7.2f} €/MWh")
    print(f"Validation P50 / mean |y|:         {val_p50_mae / val_mean_abs_y * 100:>7.2f} %")
    print(f"\nInterval [P10, P90]:")
    print(f"  Empirical coverage:    {inside:.3%}   (target ~80%)")
    print(f"  Mean width:            {avg_width:.1f} €/MWh")
    print(f"  Quantile crossings:    {crossings:.3%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "model.keras")
    np.savez(
        OUT_DIR / "scaler.npz",
        enc_mean=scaler.enc_mean, enc_std=scaler.enc_std,
        dec_mean=scaler.dec_mean, dec_std=scaler.dec_std,
        y_mean=scaler.y_mean, y_std=scaler.y_std,
    )
    meta = {
        "model": "price_quantile",
        "target": "price__germany_luxembourg",
        "include_weather": True,
        "quantiles": list(QUANTILES),
        "hidden": 64,
        "enc_features": int(Xe_tr.shape[-1]),
        "dec_features": int(Xd_tr.shape[-1]),
        "epochs_run": len(history.epoch),
        "train_time_s": train_time,
        "train_n": int(len(kept_tr)),
        "val_n": int(len(kept_va)),
        "train_window": "2024-01-08 to 2025-12-31",
        "val_window": "2026-01-01 to 2026-02-28",
        "val_p50_mae_eur_mwh": val_p50_mae,
        "val_interval_coverage": inside,
        "val_interval_width_eur_mwh": avg_width,
        "val_quantile_crossings": crossings,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
