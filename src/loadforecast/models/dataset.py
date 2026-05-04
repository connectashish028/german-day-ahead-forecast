"""Windowing for the seq2seq LSTM.

For each delivery day D with issue time T = D-1 12:00 Berlin, build:

  - X_enc  shape (672, n_enc_features)
      The last 7 days of quarter-hourly history ending strictly before T.
      Features: load, residual=(load-TSO_fc), hour_sin/cos, dow_sin/cos.
      Future-leaking values are masked via M2's `usable_columns`.

  - X_dec  shape (96, n_dec_features)
      Future-known covariates for the delivery day [D 00:00, D+1 00:00).
      Features: tso_load_fc, hour_sin/cos, dow_sin/cos, is_holiday.
      The TSO forecast for D is fully published by D-2 ~10:00, so this
      block is leakage-safe at issue time T.

  - y_resid  shape (96,)
      Target: residual = actual_load - TSO_fc on the delivery day.
      Available only in training (requires post-issue ground truth).

The same windowing function is used for training (where y_resid is real)
and inference (where y_resid is unknown — we still build it so the harness
can score after the fact).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.loader import target_index_for
from ..features.availability import usable_columns
from ..features.calendar import calendar_features

LOOKBACK_DAYS = 7
QH_PER_DAY = 96
LOOKBACK_QH = LOOKBACK_DAYS * QH_PER_DAY  # 672

ACTUAL_LOAD = "actual_cons__grid_load"
TSO_FC = "fc_cons__grid_load"

ENC_FEATURE_NAMES = ("load", "residual", "hour_sin", "hour_cos", "dow_sin", "dow_cos")
DEC_FEATURE_NAMES = (
    "tso_load_fc",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_federal_holiday",
)

# Weather columns (optional, M5+). When `include_weather=True` is passed to
# build_window/build_dataset, these get appended to BOTH encoder and decoder.
WEATHER_COLS = (
    "weather__temperature_2m",
    "weather__shortwave_radiation",
    "weather__wind_speed_100m",
    "weather__cloud_cover",
)


@dataclass
class Window:
    issue_time: pd.Timestamp
    X_enc: np.ndarray  # (672, len(ENC_FEATURE_NAMES))
    X_dec: np.ndarray  # (96, len(DEC_FEATURE_NAMES))
    y_resid: np.ndarray  # (96,) — may contain NaN at inference time
    target_idx: pd.DatetimeIndex


def _encoder_index(issue_time: pd.Timestamp) -> pd.DatetimeIndex:
    """The 672-step history index ending strictly before `issue_time`."""
    end = issue_time
    start = end - pd.Timedelta(minutes=15 * LOOKBACK_QH)
    return pd.date_range(start=start, end=end, freq="15min", inclusive="left")


def _delivery_target_index(issue_time: pd.Timestamp) -> pd.DatetimeIndex:
    delivery_local = issue_time.tz_convert("Europe/Berlin").normalize() + pd.Timedelta(days=1)
    return target_index_for(delivery_local.date())


def build_window(
    df: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    include_weather: bool = False,
) -> Window:
    """Build encoder/decoder/target arrays for a single issue time.

    When `include_weather=True`, the four `weather__*` columns are
    appended to BOTH encoder and decoder feature stacks. The expected
    column count grows from 6 to 10 in each.
    """
    enc_idx = _encoder_index(issue_time)
    target_idx = _delivery_target_index(issue_time)

    # Mask future-leaking values per M2 availability rules.
    needed_cols = (ACTUAL_LOAD, TSO_FC)
    if include_weather:
        needed_cols = (*needed_cols, *WEATHER_COLS)
    masked = usable_columns(df, issue_time, include=needed_cols)

    # Encoder
    load = masked[ACTUAL_LOAD].reindex(enc_idx).to_numpy()
    tso_h = masked[TSO_FC].reindex(enc_idx).to_numpy()
    residual_hist = load - tso_h
    cal_enc = calendar_features(enc_idx)
    enc_stack = [
        load,
        residual_hist,
        cal_enc["hour_sin"].to_numpy(),
        cal_enc["hour_cos"].to_numpy(),
        cal_enc["dow_sin"].to_numpy(),
        cal_enc["dow_cos"].to_numpy(),
    ]
    if include_weather:
        for w in WEATHER_COLS:
            enc_stack.append(masked[w].reindex(enc_idx).to_numpy())
    X_enc = np.column_stack(enc_stack).astype(np.float32)

    # Decoder (future-known)
    tso_d = masked[TSO_FC].reindex(target_idx).to_numpy()
    cal_dec = calendar_features(target_idx)
    dec_stack = [
        tso_d,
        cal_dec["hour_sin"].to_numpy(),
        cal_dec["hour_cos"].to_numpy(),
        cal_dec["dow_sin"].to_numpy(),
        cal_dec["dow_cos"].to_numpy(),
        cal_dec["is_federal_holiday"].astype(float).to_numpy(),
    ]
    if include_weather:
        for w in WEATHER_COLS:
            dec_stack.append(masked[w].reindex(target_idx).to_numpy())
    X_dec = np.column_stack(dec_stack).astype(np.float32)

    # Target — uses the *raw* (non-masked) frame because at training time
    # we deliberately have ground truth from the future.
    y_resid = (df[ACTUAL_LOAD] - df[TSO_FC]).reindex(target_idx).to_numpy().astype(np.float32)

    return Window(issue_time=issue_time, X_enc=X_enc, X_dec=X_dec, y_resid=y_resid, target_idx=target_idx)


def build_dataset(
    df: pd.DataFrame,
    issue_times: Iterable[pd.Timestamp],
    *,
    drop_incomplete: bool = True,
    include_weather: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Stack many windows into training arrays.

    Returns (X_enc, X_dec, Y, kept_issue_times). Windows with any NaN in
    encoder, decoder, or target are dropped if `drop_incomplete=True`.
    """
    Xe, Xd, Y, kept = [], [], [], []
    for t in issue_times:
        w = build_window(df, t, include_weather=include_weather)
        if drop_incomplete and (
            np.isnan(w.X_enc).any()
            or np.isnan(w.X_dec).any()
            or np.isnan(w.y_resid).any()
        ):
            continue
        Xe.append(w.X_enc)
        Xd.append(w.X_dec)
        Y.append(w.y_resid)
        kept.append(t)
    if not Xe:
        raise RuntimeError("No complete windows found — check date range and data coverage.")
    return np.stack(Xe), np.stack(Xd), np.stack(Y), kept


@dataclass
class FeatureScaler:
    """Per-feature standardiser fit on training data only.

    Stored as plain numpy so we can pickle it next to the SavedModel.
    """

    enc_mean: np.ndarray
    enc_std: np.ndarray
    dec_mean: np.ndarray
    dec_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X_enc: np.ndarray, X_dec: np.ndarray, Y: np.ndarray) -> FeatureScaler:
        enc_flat = X_enc.reshape(-1, X_enc.shape[-1])
        dec_flat = X_dec.reshape(-1, X_dec.shape[-1])
        return cls(
            enc_mean=enc_flat.mean(axis=0),
            enc_std=enc_flat.std(axis=0).clip(min=1e-6),
            dec_mean=dec_flat.mean(axis=0),
            dec_std=dec_flat.std(axis=0).clip(min=1e-6),
            y_mean=float(Y.mean()),
            y_std=float(max(Y.std(), 1e-6)),
        )

    def transform(self, X_enc, X_dec, Y=None):
        Xe = (X_enc - self.enc_mean) / self.enc_std
        Xd = (X_dec - self.dec_mean) / self.dec_std
        if Y is None:
            return Xe.astype(np.float32), Xd.astype(np.float32)
        Yn = (Y - self.y_mean) / self.y_std
        return Xe.astype(np.float32), Xd.astype(np.float32), Yn.astype(np.float32)

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std + self.y_mean


__all__ = [
    "DEC_FEATURE_NAMES",
    "ENC_FEATURE_NAMES",
    "FeatureScaler",
    "LOOKBACK_DAYS",
    "LOOKBACK_QH",
    "QH_PER_DAY",
    "Window",
    "build_dataset",
    "build_window",
]
