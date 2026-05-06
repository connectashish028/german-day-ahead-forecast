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
DEFAULT_ATTENTION_DIR = Path("model_checkpoints/lstm_attention_v1")
DEFAULT_WEATHER_DIR = Path("model_checkpoints/lstm_weather_v1")
DEFAULT_QUANTILE_DIR = Path("model_checkpoints/lstm_quantile_v1")
DEFAULT_PRICE_QUANTILE_DIR = Path("model_checkpoints/price_quantile_v4")


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


# Real-world data has occasional small gaps (an Open-Meteo hour drops out,
# a price publication is briefly delayed). A strict NaN check killed the
# whole forecast for ~0.2 % of cells; fill column-wise instead so the model
# still runs. If the gap is so large that ffill+bfill leaves NaN, only then
# fall back to TSO.
def _fill_small_gaps(arr: np.ndarray, max_nan_frac: float = 0.05) -> np.ndarray | None:
    """Forward-then-back-fill along the time axis. Returns None if too many
    NaNs to be plausibly imputable, or any column is all-NaN."""
    if not np.isnan(arr).any():
        return arr
    nan_frac = float(np.isnan(arr).mean())
    if nan_frac > max_nan_frac:
        return None
    filled = pd.DataFrame(arr).ffill().bfill().to_numpy()
    if np.isnan(filled).any():
        return None
    return filled


def lstm_residual_predict(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
) -> pd.Series:
    """Predict the delivery-day grid load: TSO forecast + LSTM residual correction.

    Reads `meta.json` to detect whether the trained model expects weather
    features, and builds the window accordingly. Falls back to the raw
    TSO forecast if the window contains NaN (e.g. for an issue date too
    early in the dataset).
    """
    model_dir = Path(model_dir)
    bundle = _get(model_dir)
    include_weather = bool(bundle.meta.get("include_weather", False))
    w = build_window(df, issue_time, include_weather=include_weather)

    enc = _fill_small_gaps(w.X_enc)
    dec = _fill_small_gaps(w.X_dec)
    if enc is None or dec is None:
        return tso_baseline_predict(df, issue_time).rename("y_lstm")

    Xe, Xd = bundle.scaler.transform(enc[None, ...], dec[None, ...])
    raw = bundle.keras_model.predict([Xe, Xd], verbose=0)
    # Plain model returns a single tensor; older multi-output trainings
    # may have returned a dict — handle both for forward compatibility.
    y_norm = raw["prediction"][0] if isinstance(raw, dict) else raw[0]
    y_resid = bundle.scaler.inverse_y(y_norm)

    tso_fc = tso_baseline_predict(df, issue_time)
    pred = tso_fc.to_numpy() + y_resid
    name = bundle.meta.get("model", "lstm")
    return pd.Series(pred, index=tso_fc.index, name=f"y_{name}")


def lstm_weather_predict(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    model_dir: Path | str = DEFAULT_WEATHER_DIR,
) -> pd.Series:
    """LSTM trained with NWP weather features; thin wrapper for the CLI."""
    return lstm_residual_predict(df, issue_time, model_dir=model_dir).rename("y_lstm_weather")


def lstm_quantile_predict_full(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    model_dir: Path | str = DEFAULT_QUANTILE_DIR,
) -> pd.DataFrame:
    """Probabilistic forecast: returns a DataFrame with columns p10, p50, p90.

    All three are *grid-load* values (TSO baseline + predicted residual quantile).
    Falls back to a degenerate frame (all three columns = TSO point forecast)
    if the encoder/decoder window has NaN.
    """
    model_dir = Path(model_dir)
    bundle = _get(model_dir)
    include_weather = bool(bundle.meta.get("include_weather", True))
    w = build_window(df, issue_time, include_weather=include_weather)
    tso_fc = tso_baseline_predict(df, issue_time)

    enc = _fill_small_gaps(w.X_enc)
    dec = _fill_small_gaps(w.X_dec)
    if enc is None or dec is None:
        v = tso_fc.to_numpy()
        return pd.DataFrame(
            {"p10": v, "p50": v, "p90": v}, index=tso_fc.index,
        )

    Xe, Xd = bundle.scaler.transform(enc[None, ...], dec[None, ...])
    raw = bundle.keras_model.predict([Xe, Xd], verbose=0)  # (1, 96, 3)
    y_resid = bundle.scaler.inverse_y(raw[0])              # (96, 3)
    base = tso_fc.to_numpy()
    out = pd.DataFrame(
        {
            "p10": base + y_resid[:, 0],
            "p50": base + y_resid[:, 1],
            "p90": base + y_resid[:, 2],
        },
        index=tso_fc.index,
    )
    out.index.name = "target_ts"
    return out


def lstm_quantile_predict(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    model_dir: Path | str = DEFAULT_QUANTILE_DIR,
) -> pd.Series:
    """Backtest-harness shim: return the median (p50) as the point forecast."""
    out = lstm_quantile_predict_full(df, issue_time, model_dir=model_dir)
    return out["p50"].rename("y_lstm_quantile")


def price_quantile_predict_full(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    model_dir: Path | str = DEFAULT_PRICE_QUANTILE_DIR,
) -> pd.DataFrame:
    """Probabilistic day-ahead price forecast: returns (96, 3) DataFrame
    with columns p10, p50, p90 in €/MWh.

    Unlike the load model, the price model targets the raw price directly
    (no TSO baseline to subtract — the day-ahead price is what we forecast).
    Returns an all-NaN frame indexed on the delivery day if the encoder/
    decoder window cannot be built (e.g. issue_time outside the parquet).
    """
    from .price_dataset import build_price_window

    model_dir = Path(model_dir)
    bundle = _get(model_dir)
    include_weather = bool(bundle.meta.get("include_weather", True))
    w = build_price_window(df, issue_time, include_weather=include_weather)

    enc = _fill_small_gaps(w.X_enc)
    dec = _fill_small_gaps(w.X_dec)
    if enc is None or dec is None:
        nan = np.full(96, np.nan)
        out = pd.DataFrame(
            {"p10": nan, "p50": nan, "p90": nan}, index=w.target_idx,
        )
        out.index.name = "target_ts"
        return out

    Xe, Xd = bundle.scaler.transform(enc[None, ...], dec[None, ...])
    raw = bundle.keras_model.predict([Xe, Xd], verbose=0)  # (1, 96, 3)
    y = bundle.scaler.inverse_y(raw[0])                    # (96, 3)
    out = pd.DataFrame(
        {"p10": y[:, 0], "p50": y[:, 1], "p90": y[:, 2]},
        index=w.target_idx,
    )
    out.index.name = "target_ts"
    return out


def lstm_attention_predict(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    model_dir: Path | str = DEFAULT_ATTENTION_DIR,
) -> pd.Series:
    """Same as `lstm_residual_predict` but uses the attention model.

    Defined as a thin wrapper so the harness CLI can dispatch by name
    without having to know about the multi-output dict.
    """
    s = lstm_residual_predict(df, issue_time, model_dir=model_dir)
    return s.rename("y_lstm_attention")


def lstm_attention_explain(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    model_dir: Path | str = DEFAULT_ATTENTION_DIR,
):
    """Run the attention model and return both the prediction and the (96,672)
    attention map. Used by notebook 05 for the per-day visualisation.

    Keras 3's Functional API doesn't cleanly let us rebuild a multi-output
    graph that exposes `return_attention_scores=True`, so we just call the
    trained layer instances directly. Their weights live with the Python
    objects regardless of which model they're "in."
    """
    bundle = _get(Path(model_dir))
    w = build_window(df, issue_time)
    if np.isnan(w.X_enc).any() or np.isnan(w.X_dec).any():
        return None, None
    Xe, Xd = bundle.scaler.transform(w.X_enc[None, ...], w.X_dec[None, ...])

    import tensorflow as tf
    enc_lstm = bundle.keras_model.get_layer("encoder_lstm")
    dec_lstm = bundle.keras_model.get_layer("decoder_lstm")
    attn = bundle.keras_model.get_layer("attention")
    concat = bundle.keras_model.get_layer("combine_dec_context")
    out_td = bundle.keras_model.get_layer("output_td")
    reshape = bundle.keras_model.get_layer("prediction")

    enc_t = tf.constant(Xe, dtype=tf.float32)
    dec_t = tf.constant(Xd, dtype=tf.float32)
    enc_seq, enc_h, enc_c = enc_lstm(enc_t)
    dec_seq = dec_lstm(dec_t, initial_state=[enc_h, enc_c])
    context, scores = attn([dec_seq, enc_seq], return_attention_scores=True)
    pred_norm = reshape(out_td(concat([dec_seq, context])))

    y_resid = bundle.scaler.inverse_y(pred_norm.numpy()[0])
    attn_map = scores.numpy()[0]  # (96, 672)
    tso_fc = tso_baseline_predict(df, issue_time)
    pred = pd.Series(
        tso_fc.to_numpy() + y_resid,
        index=tso_fc.index,
        name="y_lstm_attention",
    )
    return pred, attn_map


__all__ = [
    "DEFAULT_ATTENTION_DIR",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_PRICE_QUANTILE_DIR",
    "DEFAULT_QUANTILE_DIR",
    "DEFAULT_WEATHER_DIR",
    "LoadedModel",
    "lstm_attention_explain",
    "lstm_attention_predict",
    "lstm_quantile_predict",
    "lstm_quantile_predict_full",
    "lstm_residual_predict",
    "lstm_weather_predict",
    "price_quantile_predict_full",
]
