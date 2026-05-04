"""Builds and executes notebooks/04_lstm_explained.ipynb.

Goal: a plain-language tour of the seq2seq LSTM, using the *trained*
model from model_checkpoints/lstm_plain_v1/ as a concrete reference.
The reader sees real shapes, real residual structure, and one worked
delivery day from input to forecast.

Audience: an enthusiast who knows pandas + a little ML but has never
built an LSTM. We avoid jargon that we don't immediately define.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NB = Path("notebooks/04_lstm_explained.ipynb")

CELLS = [
    nbf.v4.new_markdown_cell(
        "# M4 — How the LSTM works (plain-language tour)\n\n"
        "This notebook explains, in concrete terms, what our seq2seq LSTM does. "
        "Every shape and number you see is from the actual model trained in M4 — "
        "no toy code.\n\n"
        "**The job.** Predict the *residual* — actual grid load minus the "
        "TSO's published forecast — over each of tomorrow's 96 quarter-hours. "
        "We then add that residual to the TSO forecast to get our final number.\n\n"
        "**Why predict the residual, not the load directly?** "
        "The TSO already gets the easy 90% of load right — calendar, "
        "climatology, weekly cycle. The systematic *errors* in their forecast "
        "are smaller, structured, and learnable. Aiming at the residual means "
        "the model only has to learn the part the TSO can't already do."
    ),

    nbf.v4.new_code_cell(
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n"
        "from plotly.subplots import make_subplots\n"
        "from pathlib import Path\n"
        "\n"
        "from loadforecast.backtest import issue_time_for, load_smard_15min\n"
        "from loadforecast.backtest.baselines import tso_baseline_predict\n"
        "from loadforecast.models.dataset import (\n"
        "    ENC_FEATURE_NAMES, DEC_FEATURE_NAMES, build_window,\n"
        ")\n"
        "from loadforecast.models.predict import lstm_residual_predict, LoadedModel, DEFAULT_MODEL_DIR\n"
        "\n"
        "df = load_smard_15min('smard_merged_15min.parquet')\n"
        "bundle = LoadedModel.load(DEFAULT_MODEL_DIR)\n"
        "print(f'Trained model loaded from {DEFAULT_MODEL_DIR}')\n"
        "print(f'Encoder features ({len(ENC_FEATURE_NAMES)}):  {ENC_FEATURE_NAMES}')\n"
        "print(f'Decoder features ({len(DEC_FEATURE_NAMES)}):  {DEC_FEATURE_NAMES}')\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 1. The model in one picture\n\n"
        "```\n"
        "encoder input  (batch, 672 timesteps, 6 features)         <-- 7 days of past quarter-hours\n"
        "      |\n"
        "      v\n"
        "  LSTM(64)  --> final state (h, c) — a 64-dim summary of the past 7 days\n"
        "                                                  \n"
        "decoder input  (batch, 96 timesteps, 6 features)          <-- known features for delivery day\n"
        "      |                                                       (TSO forecast, calendar)\n"
        "      v\n"
        "  LSTM(64) initialised from encoder state\n"
        "      |\n"
        "      v\n"
        "  Dense(1)  --> predicted residual at each of 96 quarter-hours\n"
        "```\n\n"
        "Two LSTMs, sharing only a 64-dimensional state. The encoder's job is "
        "*'compress the relevant history into 64 numbers'*. The decoder's job "
        "is *'given those 64 numbers and the calendar, produce 96 residuals'*."
    ),

    nbf.v4.new_code_cell(
        "from loadforecast.models.lstm_plain import build_lstm_plain\n"
        "model = bundle.keras_model\n"
        "model.summary(line_length=100)\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 2. What an encoder LSTM actually does\n\n"
        "An LSTM reads its input one timestep at a time. After step `t` it "
        "has a 'hidden state' `h_t` that summarises everything it has seen so far. "
        "After step `t+1`, it updates `h_t` into `h_{t+1}` using a small set of "
        "trainable matrices. Crucially, those updates are *gated* — the model "
        "learns when to forget old info and when to keep it. After 672 steps "
        "(7 days), we keep just the final state.\n\n"
        "**You can think of `h_672` as a 64-number compressed summary** of the "
        "7-day history that the model believes is relevant to forecasting "
        "tomorrow's residual."
    ),

    nbf.v4.new_code_cell(
        "# Pick one delivery day from the holdout split (after training).\n"
        "delivery = pd.Timestamp('2026-04-26').date()\n"
        "issue = issue_time_for(delivery)\n"
        "w = build_window(df, issue)\n"
        "print(f'Delivery: {delivery}  (issue time: {issue})')\n"
        "print(f'  encoder window: {len(w.X_enc)} steps x {w.X_enc.shape[1]} features')\n"
        "print(f'  decoder window: {len(w.X_dec)} steps x {w.X_dec.shape[1]} features')\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 3. Encoder input visualised\n\n"
        "These are the 6 raw features the encoder sees over the past 7 days. "
        "Notice the strong daily cycles in `load` and the calendar features — "
        "those are what the LSTM compresses into its 64-dim summary."
    ),

    nbf.v4.new_code_cell(
        "fig = make_subplots(rows=3, cols=2, shared_xaxes=True,\n"
        "                    subplot_titles=ENC_FEATURE_NAMES, vertical_spacing=0.08)\n"
        "enc_idx = pd.date_range(end=issue, periods=672, freq='15min', inclusive='left')\n"
        "for i, fname in enumerate(ENC_FEATURE_NAMES):\n"
        "    r, c = (i // 2) + 1, (i % 2) + 1\n"
        "    fig.add_trace(go.Scatter(x=enc_idx, y=w.X_enc[:, i], mode='lines',\n"
        "                              name=fname, showlegend=False), row=r, col=c)\n"
        "fig.update_layout(height=600, template='plotly_white',\n"
        "                  title=f'Encoder input — 7 days ending at {issue}')\n"
        "fig.show()\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 4. Decoder: what does the model already know about tomorrow?\n\n"
        "The decoder sees only **future-known** features for the delivery day:\n"
        "- the TSO's published load forecast for tomorrow (the baseline we're correcting),\n"
        "- calendar features (hour, day-of-week as sin/cos),\n"
        "- the holiday flag.\n\n"
        "**Crucially, the decoder does *not* see tomorrow's actuals.** That would "
        "be cheating. The decoder's job is to figure out *the residual* — the "
        "delta the LSTM thinks it should add to the TSO forecast — using only "
        "what is genuinely available at issue time."
    ),

    nbf.v4.new_code_cell(
        "fig = make_subplots(rows=3, cols=2, shared_xaxes=True,\n"
        "                    subplot_titles=DEC_FEATURE_NAMES, vertical_spacing=0.08)\n"
        "dec_idx = pd.date_range(start=issue + pd.Timedelta(hours=12), periods=96, freq='15min')\n"
        "for i, fname in enumerate(DEC_FEATURE_NAMES):\n"
        "    r, c = (i // 2) + 1, (i % 2) + 1\n"
        "    fig.add_trace(go.Scatter(x=dec_idx, y=w.X_dec[:, i], mode='lines',\n"
        "                              showlegend=False), row=r, col=c)\n"
        "fig.update_layout(height=600, template='plotly_white',\n"
        "                  title=f'Decoder input — 96 quarter-hours of {delivery}')\n"
        "fig.show()\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 5. The forecast, end to end\n\n"
        "Now run the full pipeline: encode the past 7 days, decode the delivery "
        "day. We get a predicted residual at each quarter-hour. Add it to the "
        "TSO forecast → final load prediction. Compare to actual."
    ),

    nbf.v4.new_code_cell(
        "pred = lstm_residual_predict(df, issue)\n"
        "tso  = tso_baseline_predict(df, issue)\n"
        "actual = df['actual_cons__grid_load'].reindex(pred.index)\n"
        "\n"
        "fig = make_subplots(rows=2, cols=1, shared_xaxes=True,\n"
        "                    row_heights=[0.65, 0.35],\n"
        "                    subplot_titles=('Forecast vs actual',\n"
        "                                    'Residual (actual - TSO) and what the LSTM predicted'),\n"
        "                    vertical_spacing=0.08)\n"
        "fig.add_trace(go.Scatter(x=actual.index, y=actual, name='actual',\n"
        "                          line=dict(color='black', width=2)), row=1, col=1)\n"
        "fig.add_trace(go.Scatter(x=tso.index, y=tso, name='TSO baseline',\n"
        "                          line=dict(color='#2E86AB', dash='dot')), row=1, col=1)\n"
        "fig.add_trace(go.Scatter(x=pred.index, y=pred, name='LSTM',\n"
        "                          line=dict(color='#C73E1D', width=2)), row=1, col=1)\n"
        "fig.add_trace(go.Scatter(x=pred.index, y=actual - tso, name='actual residual',\n"
        "                          line=dict(color='black')), row=2, col=1)\n"
        "fig.add_trace(go.Scatter(x=pred.index, y=pred - tso, name='LSTM residual prediction',\n"
        "                          line=dict(color='#C73E1D', dash='dash')), row=2, col=1)\n"
        "fig.update_layout(height=620, template='plotly_white',\n"
        "                  title=f'Delivery day {delivery}', hovermode='x unified')\n"
        "fig.show()\n"
        "\n"
        "mae_lstm = float((actual - pred).abs().mean())\n"
        "mae_tso  = float((actual - tso).abs().mean())\n"
        "print(f'MAE  LSTM={mae_lstm:.1f}  TSO={mae_tso:.1f}  '\n"
        "      f'skill={1 - mae_lstm/mae_tso:+.3f}')\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 6. What did we actually train?\n\n"
        "Training metadata, loss curves, and what the model learned."
    ),

    nbf.v4.new_code_cell(
        "meta = json.loads(Path(DEFAULT_MODEL_DIR / 'meta.json').read_text())\n"
        "print('Training metadata:')\n"
        "for k in ('train_n', 'val_n', 'epochs_run', 'train_time_s',\n"
        "         'val_residual_mae_mw', 'val_residual_rmse_mw', 'val_implied_skill'):\n"
        "    print(f'  {k:<25s}  {meta[k]}')\n"
        "\n"
        "h = meta['history']\n"
        "fig = go.Figure()\n"
        "fig.add_trace(go.Scatter(x=list(range(1, len(h['loss'])+1)), y=h['loss'],\n"
        "                          name='train loss', line=dict(color='#2E86AB')))\n"
        "fig.add_trace(go.Scatter(x=list(range(1, len(h['val_loss'])+1)), y=h['val_loss'],\n"
        "                          name='val loss', line=dict(color='#C73E1D')))\n"
        "fig.update_layout(title='Loss curves (Huber, normalised residual)',\n"
        "                  xaxis_title='epoch', yaxis_title='loss',\n"
        "                  template='plotly_white', height=380)\n"
        "fig.show()\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 7. Closing thought\n\n"
        "**Why is this a sane first model?** Because we asked it to do exactly "
        "one thing: learn the systematic error in the TSO's forecast, given "
        "7 days of history. That's a much smaller learning problem than "
        "*'predict tomorrow's load from scratch'*. The hard part was already "
        "done by the TSO; the LSTM just adds the missing structure.\n\n"
        "**What's next?** Two upgrades, in order of expected impact:\n"
        "1. **Weather (M5)** — give the encoder access to NWP forecasts (PV "
        "  irradiance, temperature) for the major load centres. The current "
        "  model has no idea whether tomorrow is sunny.\n"
        "2. **Attention (M4 part 2)** — let the decoder *look back at the encoder*"
        "  at every step instead of just receiving its compressed final state. "
        "  Often a small lift on long-horizon forecasts; also gives us an "
        "  interpretable attention map.\n\n"
        "If neither beats Milestone 4 by a meaningful margin, we keep the "
        "simpler model — Occam's razor.\n"
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"display_name": "loadforecast", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB)
    print(f"Wrote {NB}; executing...")
    NotebookClient(nb, timeout=300, kernel_name="python3").execute()
    nbf.write(nb, NB)
    print(f"Executed and saved {NB}")


if __name__ == "__main__":
    main()
