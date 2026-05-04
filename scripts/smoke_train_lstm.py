"""Smoke-train the plain LSTM on a tiny slice.

Goal: verify the windowing -> scaling -> training -> inverse-transform
pipeline works end-to-end, NOT to produce a useful model. Trains 3
epochs on ~60 issue dates from 2024 H1, then prints the per-day MAE
on a few held-out 2024 H2 dates against the TSO baseline.

If MAE is finite and within an order of magnitude of TSO MAE (~500 MW),
the pipeline is wired correctly. Tighter performance requires the real
training run (scripts/train_lstm_plain.py — coming next).
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.backtest.baselines import tso_baseline_predict
from loadforecast.models.dataset import FeatureScaler, build_dataset
from loadforecast.models.lstm_plain import build_lstm_plain, compile_lstm

PARQUET = "smard_merged_15min.parquet"


def _drange(start: date, end: date, step: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step)


def main() -> None:
    print("Loading data …")
    df = load_smard_15min(PARQUET)

    train_dates = [issue_time_for(d) for d in _drange(date(2024, 1, 1), date(2024, 6, 30), step=2)]
    test_dates = [issue_time_for(d) for d in _drange(date(2024, 7, 1), date(2024, 7, 14))]

    print(f"Building windows: train={len(train_dates)} candidate, test={len(test_dates)} candidate")
    Xe_tr, Xd_tr, Y_tr, kept_tr = build_dataset(df, train_dates)
    Xe_te, Xd_te, Y_te, kept_te = build_dataset(df, test_dates)
    print(f"  -> kept train={len(kept_tr)}, test={len(kept_te)}")
    print(f"  shapes: X_enc={Xe_tr.shape}, X_dec={Xd_tr.shape}, Y={Y_tr.shape}")

    scaler = FeatureScaler.fit(Xe_tr, Xd_tr, Y_tr)
    Xe_tr_n, Xd_tr_n, Y_tr_n = scaler.transform(Xe_tr, Xd_tr, Y_tr)
    Xe_te_n, Xd_te_n = scaler.transform(Xe_te, Xd_te)

    print("Building model …")
    model = compile_lstm(build_lstm_plain(hidden=32))

    print("Training 3 epochs …")
    t0 = time.time()
    model.fit(
        [Xe_tr_n, Xd_tr_n], Y_tr_n,
        validation_split=0.2,
        epochs=3,
        batch_size=16,
        verbose=2,
    )
    print(f"  trained in {time.time() - t0:.1f}s")

    print("\nInference on test set:")
    pred_norm = model.predict([Xe_te_n, Xd_te_n], verbose=0)
    pred_resid = scaler.inverse_y(pred_norm)

    for i, t in enumerate(kept_te):
        delivery = (t.tz_convert("Europe/Berlin") + np.timedelta64(12, "h")).date()
        tso_pred = tso_baseline_predict(df, t)
        actual = df["actual_cons__grid_load"].reindex(tso_pred.index)
        # Final prediction = TSO + predicted residual
        lstm_pred = tso_pred.to_numpy() + pred_resid[i]
        mae_lstm = float(np.abs(actual - lstm_pred).mean())
        mae_tso = float(np.abs(actual - tso_pred).mean())
        print(
            f"  {delivery} | MAE  LSTM={mae_lstm:6.1f}  TSO={mae_tso:6.1f}  "
            f"skill={1 - mae_lstm / mae_tso:+.3f}"
        )


if __name__ == "__main__":
    main()
