"""Train the seq2seq LSTM with weather features.

Identical splits, scaler, and training config as `train_lstm_plain.py`
so the comparison is apples-to-apples — only the input feature set
differs (6 -> 10 features in both encoder and decoder).

Saves to model_checkpoints/lstm_weather_v1/.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.dataset import (
    DEC_FEATURE_NAMES,
    ENC_FEATURE_NAMES,
    WEATHER_COLS,
    FeatureScaler,
    build_dataset,
)
from loadforecast.models.lstm_plain import build_lstm_plain, compile_lstm

PARQUET = "smard_merged_15min.parquet"
OUT_DIR = Path("model_checkpoints/lstm_weather_v1")


def _drange(start: date, end: date, step: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def main() -> None:
    print("Loading parquet...")
    df = load_smard_15min(PARQUET)
    print(f"  range: {df.index.min()}  ->  {df.index.max()}  ({len(df):,} rows)")

    # Verify weather columns exist before training.
    missing = [c for c in WEATHER_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing weather columns in parquet: {missing}. "
                         f"Run `python -m loadforecast.data.refresh --rebuild`.")

    train_dates = [issue_time_for(d) for d in _drange(date(2022, 1, 8),  date(2024, 12, 31))]
    val_dates   = [issue_time_for(d) for d in _drange(date(2025, 1, 1),  date(2025, 6, 30))]

    print(f"\nBuilding windows (with weather): {len(train_dates)} train, {len(val_dates)} val")
    Xe_tr, Xd_tr, Y_tr, kept_tr = build_dataset(df, train_dates, include_weather=True)
    Xe_va, Xd_va, Y_va, kept_va = build_dataset(df, val_dates,   include_weather=True)
    print(f"  kept: train={len(kept_tr)}, val={len(kept_va)}")
    print(f"  shapes: X_enc={Xe_tr.shape}  X_dec={Xd_tr.shape}  Y={Y_tr.shape}")
    print(f"  encoder features: {ENC_FEATURE_NAMES + WEATHER_COLS}")
    print(f"  decoder features: {DEC_FEATURE_NAMES + WEATHER_COLS}")

    scaler = FeatureScaler.fit(Xe_tr, Xd_tr, Y_tr)
    Xe_tr_n, Xd_tr_n, Y_tr_n = scaler.transform(Xe_tr, Xd_tr, Y_tr)
    Xe_va_n, Xd_va_n, Y_va_n = scaler.transform(Xe_va, Xd_va, Y_va)

    print("\nBuilding model...")
    from tensorflow import keras
    model = compile_lstm(
        build_lstm_plain(
            hidden=64,
            enc_features=Xe_tr.shape[-1],
            dec_features=Xd_tr.shape[-1],
        ),
        lr=1e-3,
    )
    model.summary(line_length=100)

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
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - t0
    print(f"\nTrained in {train_time:.0f}s ({len(history.epoch)} epochs)")

    pred_va_n = model.predict([Xe_va_n, Xd_va_n], verbose=0)
    pred_va = scaler.inverse_y(pred_va_n)
    val_mae = float(np.abs(Y_va - pred_va).mean())
    val_rmse = float(np.sqrt(((Y_va - pred_va) ** 2).mean()))
    val_implied_skill = float(1 - val_mae / np.abs(Y_va).mean())
    print(f"\nValidation residual MAE: {val_mae:.1f} MW")
    print(f"  Implied skill if generalised: {val_implied_skill:+.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "model.keras")
    np.savez(
        OUT_DIR / "scaler.npz",
        enc_mean=scaler.enc_mean, enc_std=scaler.enc_std,
        dec_mean=scaler.dec_mean, dec_std=scaler.dec_std,
        y_mean=scaler.y_mean, y_std=scaler.y_std,
    )
    meta = {
        "model": "lstm_weather",
        "include_weather": True,
        "hidden": 64,
        "enc_features": int(Xe_tr.shape[-1]),
        "dec_features": int(Xd_tr.shape[-1]),
        "epochs_run": len(history.epoch),
        "train_time_s": train_time,
        "train_n": int(len(kept_tr)),
        "val_n": int(len(kept_va)),
        "val_residual_mae_mw": val_mae,
        "val_residual_rmse_mw": val_rmse,
        "val_implied_skill": val_implied_skill,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
