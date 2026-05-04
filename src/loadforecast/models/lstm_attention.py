"""Seq2seq LSTM with Bahdanau (additive) attention.

Same shape/inputs as `lstm_plain` but the encoder returns its full sequence
of outputs (not just the final state), and the decoder *attends over* the
encoder outputs at every step.

What attention buys us
----------------------
The plain LSTM compresses the entire 7-day history into a single 64-dim
state vector before the decoder ever sees it. Attention lets the decoder
look back at every encoder timestep at every decoder timestep. The model
learns *what to look at* rather than relying on a frozen summary.

We use Keras' built-in `AdditiveAttention` (Bahdanau et al. 2015).

Two models, shared weights
--------------------------
- `build_lstm_attention_train()` returns a graph with a single output
  (the residual prediction) — used for training.
- `build_lstm_attention_explain()` returns a graph that exposes the
  (96, 672) attention scores as a second output — used at inference for
  the visualisation notebook. Both share the same layer instances, so
  loading either keras file recovers the trained weights.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from .dataset import DEC_FEATURE_NAMES, ENC_FEATURE_NAMES, LOOKBACK_QH, QH_PER_DAY


def _build_layers(hidden: int, enc_features: int, dec_features: int):
    """Build the layer instances used by both training and explain graphs.

    Returning the inputs + layer references lets both graphs reuse the
    exact same weights.
    """
    enc_in = keras.Input(shape=(LOOKBACK_QH, enc_features), name="encoder_in")
    enc_lstm = layers.LSTM(
        hidden, return_sequences=True, return_state=True, name="encoder_lstm"
    )
    enc_seq, enc_h, enc_c = enc_lstm(enc_in)

    dec_in = keras.Input(shape=(QH_PER_DAY, dec_features), name="decoder_in")
    dec_lstm = layers.LSTM(hidden, return_sequences=True, name="decoder_lstm")
    dec_seq = dec_lstm(dec_in, initial_state=[enc_h, enc_c])

    attn = layers.AdditiveAttention(name="attention")
    concat = layers.Concatenate(name="combine_dec_context")
    out_td = layers.TimeDistributed(layers.Dense(1, name="output_dense"), name="output_td")
    reshape = layers.Reshape((QH_PER_DAY,), name="prediction")

    return enc_in, dec_in, enc_seq, dec_seq, attn, concat, out_td, reshape


def build_lstm_attention_train(
    *,
    hidden: int = 64,
    enc_features: int = len(ENC_FEATURE_NAMES),
    dec_features: int = len(DEC_FEATURE_NAMES),
) -> keras.Model:
    """Single-output training graph."""
    enc_in, dec_in, enc_seq, dec_seq, attn, concat, out_td, reshape = _build_layers(
        hidden, enc_features, dec_features
    )
    context = attn([dec_seq, enc_seq])  # (B, 96, 64)
    combined = concat([dec_seq, context])
    y = reshape(out_td(combined))
    return keras.Model(inputs=[enc_in, dec_in], outputs=y, name="lstm_attention")


def build_lstm_attention_explain(
    train_model: keras.Model,
) -> keras.Model:
    """Build an explain graph that shares weights with the training model.

    Creates fresh Input layers (the training model's inputs are bound to
    its own graph) and rebuilds the forward pass using the trained
    layers — which retain their weights as Python objects, even outside
    their original graph.
    """
    enc_lstm = train_model.get_layer("encoder_lstm")
    dec_lstm = train_model.get_layer("decoder_lstm")
    attn = train_model.get_layer("attention")
    concat = train_model.get_layer("combine_dec_context")
    out_td = train_model.get_layer("output_td")
    reshape = train_model.get_layer("prediction")

    # Read input shapes from the training model so the explain model
    # accepts identical inputs.
    enc_shape = train_model.get_layer("encoder_in").output.shape[1:]
    dec_shape = train_model.get_layer("decoder_in").output.shape[1:]
    enc_in = keras.Input(shape=enc_shape, name="encoder_in_explain")
    dec_in = keras.Input(shape=dec_shape, name="decoder_in_explain")

    enc_seq, enc_h, enc_c = enc_lstm(enc_in)
    dec_seq = dec_lstm(dec_in, initial_state=[enc_h, enc_c])
    context, attn_scores = attn([dec_seq, enc_seq], return_attention_scores=True)
    y = reshape(out_td(concat([dec_seq, context])))
    return keras.Model(
        inputs=[enc_in, dec_in],
        outputs=[y, attn_scores],
        name="lstm_attention_explain",
    )


def compile_lstm_attention(model: keras.Model, *, lr: float = 1e-3) -> keras.Model:
    """Train on residual MAE via Huber, same recipe as plain LSTM."""
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


__all__ = [
    "build_lstm_attention_explain",
    "build_lstm_attention_train",
    "compile_lstm_attention",
]


if __name__ == "__main__":
    m = compile_lstm_attention(build_lstm_attention_train())
    m.summary(line_length=110)
