"""Plain seq2seq LSTM (no attention).

Deliberately the smallest model that can do the job, so the educational
notebook 04 can walk a reader through every layer.

Shape lifecycle:

    enc_in   (batch, 672, 6)  ──► LSTM(64) ──► (h, c) state
                                                    │
    dec_in   (batch,  96, 6)  ──► LSTM(64, init=[h,c]) ──► (batch, 96, 64)
                                                    │
                                  Dense(1)          ▼
                                                    (batch, 96, 1)  ── residual prediction

The encoder reads 7 days of history and compresses it into a single
fixed-size state vector (h, c). The decoder takes that state plus the
future-known covariates for the delivery day and produces a residual
prediction at every quarter-hour.

This module exposes the *graph builder* only. Training lives in
`scripts/train_lstm_plain.py`; inference in
`loadforecast.models.predict.lstm_residual_predict` (PredictFn-shaped).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .dataset import DEC_FEATURE_NAMES, ENC_FEATURE_NAMES, LOOKBACK_QH, QH_PER_DAY


def build_lstm_plain(
    *,
    hidden: int = 64,
    enc_features: int = len(ENC_FEATURE_NAMES),
    dec_features: int = len(DEC_FEATURE_NAMES),
) -> keras.Model:
    """Build the seq2seq LSTM model graph (uncompiled)."""
    enc_in = keras.Input(shape=(LOOKBACK_QH, enc_features), name="encoder_in")
    # Single LSTM layer that returns its final (h, c) state.
    _, h, c = layers.LSTM(hidden, return_state=True, name="encoder_lstm")(enc_in)

    dec_in = keras.Input(shape=(QH_PER_DAY, dec_features), name="decoder_in")
    dec_out = layers.LSTM(
        hidden,
        return_sequences=True,
        name="decoder_lstm",
    )(dec_in, initial_state=[h, c])

    y = layers.TimeDistributed(layers.Dense(1, name="output_dense"), name="output_td")(dec_out)
    y = layers.Reshape((QH_PER_DAY,), name="squeeze")(y)

    model = keras.Model(inputs=[enc_in, dec_in], outputs=y, name="lstm_plain")
    return model


def compile_lstm(model: keras.Model, *, lr: float = 1e-3) -> keras.Model:
    """Default training config: AdamW + Huber loss."""
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


__all__ = ["build_lstm_plain", "compile_lstm"]


if __name__ == "__main__":
    # Print summary so a curious reader can run `python -m loadforecast.models.lstm_plain`.
    tf.get_logger().setLevel("ERROR")
    m = compile_lstm(build_lstm_plain())
    m.summary(line_length=100)
