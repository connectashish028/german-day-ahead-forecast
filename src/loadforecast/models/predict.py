"""Inference wrapper that turns a saved seq2seq LSTM into a `PredictFn`
compatible with the backtest harness.

A `PredictFn` is `Callable[[pd.DataFrame, pd.Timestamp], pd.Series]` —
takes the full parquet + an issue time, returns a 96-step grid load
forecast for the delivery day.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.baselines import tso_baseline_predict
from .dataset import FeatureScaler, build_window

DEFAULT_MODEL_DIR = Path("model_checkpoints/lstm_plain_v1")


@dataclass
class LoadedModel:
    keras_model: object
    scaler: FeatureScaler
    meta: dict
    model_dir: Path

    @classmethod
    def load(cls, model_dir: Path) -> LoadedModel:
        from tensorflow import keras

        meta = json.loads((model_dir / "meta.json").read_text())
        scaler_npz = np.load(model_dir / "scaler.npz")
        scaler = FeatureScaler(
            enc_mean=scaler_npz["enc_mean"],
            enc_std=scaler_npz["enc_std"],
            dec_mean=scaler_npz["dec_mean"],
            dec_std=scaler_npz["dec_std"],
            y_mean=float(scaler_npz["y_mean"]),
            y_std=float(scaler_npz["y_std"]),
        )
        keras_model = keras.models.load_model(model_dir / "model.keras", compile=False)
        return cls(keras_model=keras_model, scaler=scaler, meta=meta, model_dir=model_dir)


_CACHE: dict[str, LoadedModel] = {}


def _get(model_dir: Path) -> LoadedModel:
    key = str(model_dir.resolve())
    if key not in _CACHE:
        _CACHE[key] = LoadedModel.load(model_dir)
    return _CACHE[key]


def lstm_residual_predict(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
) -> pd.Series:
    """Predict the delivery-day grid load: TSO forecast + LSTM residual correction.

    Steps:
      1. Build the seq2seq window at `issue_time` (encoder=last 7d, decoder=delivery day).
      2. Standardise with the saved scaler (fit on training data only).
      3. Forward-pass to get predicted residual (96 normalised steps).
      4. Inverse-transform and add to the TSO published forecast for the day.

    If the encoder/decoder window contains any NaN (e.g. issue date too
    early in the dataset), fall back to the raw TSO forecast.
    """
    model_dir = Path(model_dir)
    bundle = _get(model_dir)
    w = build_window(df, issue_time)

    if (
        np.isnan(w.X_enc).any()
        or np.isnan(w.X_dec).any()
    ):
        return tso_baseline_predict(df, issue_time).rename("y_lstm_plain")

    Xe, Xd = bundle.scaler.transform(w.X_enc[None, ...], w.X_dec[None, ...])
    y_norm = bundle.keras_model.predict([Xe, Xd], verbose=0)[0]
    y_resid = bundle.scaler.inverse_y(y_norm)

    tso_fc = tso_baseline_predict(df, issue_time)
    pred = tso_fc.to_numpy() + y_resid
    return pd.Series(pred, index=tso_fc.index, name="y_lstm_plain")


__all__ = ["lstm_residual_predict", "LoadedModel", "DEFAULT_MODEL_DIR"]
