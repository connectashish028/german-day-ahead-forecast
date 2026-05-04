"""Seq2seq LSTM with quantile output heads (P10 / P50 / P90).

Architecture is identical to `lstm_plain` (encoder LSTM(64) → state →
decoder LSTM(64)) but the final Dense layer projects to **3 outputs per
timestep** instead of 1, and the model is trained with **pinball loss**
on each quantile slice.

What pinball loss does
----------------------
For a target quantile `q` ∈ (0, 1), pinball loss penalises errors
asymmetrically:

    loss(y_true, y_pred; q) = max(q * (y_true - y_pred), (q - 1) * (y_true - y_pred))

For q = 0.5, this reduces to (1/2) * |y_true - y_pred| — i.e. minimising
mean absolute error → the *median*. For q = 0.1, the loss penalises
under-predictions 9× more than over-predictions, pushing the model's
output to the 10th percentile. Same idea symmetrically for q = 0.9.

Training the same network with a sum of three pinball losses
(q=0.1, 0.5, 0.9) makes each output column fit its own quantile of the
conditional distribution. We end up with a *probabilistic* day-ahead
forecast, not a point estimate.

Production use
--------------
- p50 (the median) is our headline point forecast. It will be close to
  but not identical to the Huber-trained model — pinball-0.5 minimises
  L1, Huber minimises smooth-L1.
- (p10, p90) bracket an 80% prediction interval. Empirical coverage on
  the holdout should land in [78%, 82%]. M8 (conformal calibration)
  tightens this with finite-sample guarantees.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .dataset import DEC_FEATURE_NAMES, ENC_FEATURE_NAMES, LOOKBACK_QH, QH_PER_DAY

QUANTILES = (0.1, 0.5, 0.9)


def pinball_loss(quantiles=QUANTILES):
    """Sum of pinball losses across the configured quantiles.

    Expects:
      y_true  shape (batch, 96)
      y_pred  shape (batch, 96, len(quantiles))
    """
    q = tf.constant(quantiles, dtype=tf.float32)

    def loss(y_true, y_pred):
        # broadcast y_true to (batch, 96, n_quantiles)
        y_true_b = tf.expand_dims(y_true, axis=-1)
        e = y_true_b - y_pred
        # pinball: max(q*e, (q-1)*e)  ==  e*q  if e>=0 else e*(q-1)
        per_q = tf.maximum(q * e, (q - 1.0) * e)
        return tf.reduce_mean(per_q)

    return loss


def _interval_coverage_metric(y_true, y_pred):
    """Fraction of (batch, 96) targets that fall inside [p10, p90]."""
    p10 = y_pred[..., 0]
    p90 = y_pred[..., 2]
    inside = tf.logical_and(y_true >= p10, y_true <= p90)
    return tf.reduce_mean(tf.cast(inside, tf.float32))


def _p50_mae_metric(y_true, y_pred):
    """MAE of the median output against the target."""
    p50 = y_pred[..., 1]
    return tf.reduce_mean(tf.abs(y_true - p50))


def build_lstm_quantile(
    *,
    hidden: int = 64,
    enc_features: int = len(ENC_FEATURE_NAMES),
    dec_features: int = len(DEC_FEATURE_NAMES),
    n_quantiles: int = len(QUANTILES),
) -> keras.Model:
    """Build the seq2seq LSTM-with-quantile-heads graph (uncompiled)."""
    enc_in = keras.Input(shape=(LOOKBACK_QH, enc_features), name="encoder_in")
    _, h, c = layers.LSTM(hidden, return_state=True, name="encoder_lstm")(enc_in)

    dec_in = keras.Input(shape=(QH_PER_DAY, dec_features), name="decoder_in")
    dec_seq = layers.LSTM(hidden, return_sequences=True, name="decoder_lstm")(
        dec_in, initial_state=[h, c]
    )

    # Project to n_quantiles outputs per timestep.
    y = layers.TimeDistributed(
        layers.Dense(n_quantiles, name="quantile_dense"), name="prediction"
    )(dec_seq)
    return keras.Model(inputs=[enc_in, dec_in], outputs=y, name="lstm_quantile")


def compile_lstm_quantile(model: keras.Model, *, lr: float = 1e-3) -> keras.Model:
    """AdamW + pinball loss + interval-coverage tracking metric."""
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss=pinball_loss(QUANTILES),
        metrics=[_p50_mae_metric, _interval_coverage_metric],
    )
    return model


__all__ = [
    "QUANTILES",
    "build_lstm_quantile",
    "compile_lstm_quantile",
    "pinball_loss",
]


if __name__ == "__main__":
    m = compile_lstm_quantile(build_lstm_quantile())
    m.summary(line_length=100)
