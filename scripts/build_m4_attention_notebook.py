"""Builds and executes notebooks/05_attention_visualisation.ipynb.

What it shows:
  1. Headline: attention vs plain LSTM scoreboard.
  2. Three sample delivery days with the (96 forecast hours x 672 history hours)
     attention heatmap; each row is a forecast hour, each column a history hour,
     darker = more weight.
  3. The "average" attention pattern across many delivery days — what the
     decoder pays attention to in aggregate.
  4. A plain-language interpretation: at evening peak, decoder weights the
     same hour 7 days ago heavily; at solar noon, it weights the most-recent
     24h (weather-driven drift signal).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NB = Path("notebooks/05_attention_visualisation.ipynb")


CELLS = [
    nbf.v4.new_markdown_cell(
        "# M4 part 2 — Attention: a useful negative result\n\n"
        "**Spoiler.** Attention beat plain LSTM on the validation set "
        "(+0.27 vs +0.26 implied skill) — but on the **holdout** it **lost badly** "
        "(+0.07 vs +0.21 actual skill). We keep the plain LSTM as the production "
        "model and treat attention as an *interpretability artifact only*. This "
        "notebook walks through why.\n\n"
        "**Architecture difference.** The plain LSTM compresses the past 7 days "
        "into a single 64-dim state vector before the decoder ever sees it. "
        "Attention lets the decoder *look back* at every encoder timestep when "
        "producing each of tomorrow's 96 quarter-hours; the attention layer "
        "outputs a **(96 × 672) weighting matrix** we can read.\n\n"
        "**The lesson.** Adding model capacity to a 1016-sample training set is "
        "easy to do badly. Validation skill can flatter; holdout is the truth-teller. "
        "We follow Occam: simpler model, better generalisation, ship that one."
    ),

    nbf.v4.new_code_cell(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "# Resolve paths relative to the project root regardless of where the\n"
        "# notebook is launched from (notebooks/ or repo root).\n"
        "_here = Path.cwd()\n"
        "ROOT = _here if (_here / 'pyproject.toml').exists() else _here.parent\n"
        "os.chdir(ROOT)\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n"
        "from plotly.subplots import make_subplots\n"
        "\n"
        "from loadforecast.backtest import issue_time_for, load_smard_15min\n"
        "from loadforecast.backtest.baselines import tso_baseline_predict\n"
        "from loadforecast.models.predict import (\n"
        "    lstm_attention_explain, DEFAULT_MODEL_DIR, DEFAULT_ATTENTION_DIR,\n"
        ")\n"
        "\n"
        "df = load_smard_15min('smard_merged_15min.parquet')\n"
        "plain_meta = json.loads((DEFAULT_MODEL_DIR / 'meta.json').read_text())\n"
        "attn_meta  = json.loads((DEFAULT_ATTENTION_DIR / 'meta.json').read_text())\n"
    ),

    nbf.v4.new_markdown_cell("## 1. Scoreboard"),

    nbf.v4.new_code_cell(
        "rows = []\n"
        "for label, csv in [\n"
        "    ('TSO',             'backtest_results/tso_step7_v2.csv'),\n"
        "    ('seasonal_naive',  'backtest_results/naive_step7_v2.csv'),\n"
        "    ('SARIMAX',         'backtest_results/sarimax_step7_v2.csv'),\n"
        "    ('LSTM_plain',      'backtest_results/lstm_plain_step7.csv'),\n"
        "    ('LSTM_attention',  'backtest_results/lstm_attention_step7.csv'),\n"
        "]:\n"
        "    if not Path(csv).exists():\n"
        "        continue\n"
        "    bt = pd.read_csv(csv)\n"
        "    mae_model = (bt['y_true'] - bt['y_model']).abs().mean()\n"
        "    mae_tso   = (bt['y_true'] - bt['y_tso']).abs().mean()\n"
        "    mape = ((bt['y_true'] - bt['y_model']).abs() / bt['y_true'].abs()).mean() * 100\n"
        "    rows.append({\n"
        "        'predictor': label,\n"
        "        'MAE (MW)':  round(mae_model, 1),\n"
        "        'MAPE (%)':  round(mape, 3),\n"
        "        'skill':     round(1 - mae_model / mae_tso, 4),\n"
        "    })\n"
        "scoreboard = pd.DataFrame(rows).set_index('predictor')\n"
        "scoreboard\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 2. Attention heatmaps for three sample delivery days\n\n"
        "Pick a winter weekday, a sunny weekend, and a holiday — three\n"
        "regimes where the decoder's information needs differ.\n"
        "Each panel: rows are the 96 forecast quarter-hours (top = tomorrow 00:00,\n"
        "bottom = tomorrow 23:45). Columns are the 672 history quarter-hours\n"
        "(left = 7 days ago, right = issue time). Darker = more attention weight."
    ),

    nbf.v4.new_code_cell(
        "sample_days = [\n"
        "    pd.Timestamp('2025-12-15').date(),  # winter weekday\n"
        "    pd.Timestamp('2025-06-22').date(),  # sunny Sunday\n"
        "    pd.Timestamp('2025-12-25').date(),  # Christmas (federal holiday)\n"
        "]\n"
        "fig = make_subplots(rows=1, cols=len(sample_days),\n"
        "                    subplot_titles=[str(d) for d in sample_days],\n"
        "                    horizontal_spacing=0.05)\n"
        "for i, d in enumerate(sample_days, start=1):\n"
        "    issue = issue_time_for(d)\n"
        "    pred, attn = lstm_attention_explain(df, issue)\n"
        "    if attn is None:\n"
        "        continue\n"
        "    fig.add_trace(go.Heatmap(\n"
        "        z=attn, colorscale='Blues', showscale=(i == len(sample_days)),\n"
        "        zmin=0, zmax=float(attn.max()),\n"
        "    ), row=1, col=i)\n"
        "    fig.update_xaxes(title_text='history QH (0=7d ago, 671=now)', row=1, col=i)\n"
        "    if i == 1:\n"
        "        fig.update_yaxes(title_text='forecast QH (0=00:00, 95=23:45)', row=1, col=i)\n"
        "fig.update_layout(height=420, template='plotly_white',\n"
        "                  title='Decoder attention map: forecast QH x history QH')\n"
        "fig.show()\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 3. Attention averaged over the holdout\n\n"
        "What the model has learned to look at *in general*, averaged over many days."
    ),

    nbf.v4.new_code_cell(
        "from datetime import date, timedelta\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Sample 30 dates from the holdout, every 3 days starting 2025-07-01.\n"
        "sample_dates = [date(2025, 7, 1) + timedelta(days=i*3) for i in range(30)]\n"
        "all_attn = []\n"
        "for d in sample_dates:\n"
        "    issue = issue_time_for(d)\n"
        "    _, a = lstm_attention_explain(df, issue)\n"
        "    if a is not None:\n"
        "        all_attn.append(a)\n"
        "mean_attn = np.mean(np.stack(all_attn), axis=0)\n"
        "print(f'Averaged over {len(all_attn)} days; shape = {mean_attn.shape}')\n"
        "\n"
        "fig = go.Figure(data=go.Heatmap(z=mean_attn, colorscale='Blues',\n"
        "                                  zmin=0, zmax=float(mean_attn.max())))\n"
        "fig.update_layout(\n"
        "    title=f'Mean attention pattern (averaged over {len(all_attn)} delivery days)',\n"
        "    xaxis_title='history QH  (0 = 7 days before issue,  671 = issue time)',\n"
        "    yaxis_title='forecast QH  (0 = 00:00,  95 = 23:45)',\n"
        "    height=480, template='plotly_white',\n"
        ")\n"
        "fig.show()\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 4. Why did attention overfit?\n\n"
        "Compare validation vs holdout skill for both models:\n\n"
        "| Model | val skill | holdout skill | gap |\n"
        "|---|---|---|---|\n"
        "| LSTM plain | +0.2553 | **+0.2147** | −0.041 (small, healthy) |\n"
        "| LSTM attention | **+0.2709** | +0.0701 | **−0.201 (huge)** |\n\n"
        "Attention has *more* capacity (an extra dot-product over 672 timesteps) "
        "but only the same 1016 training windows to learn from. With enough "
        "capacity, the model can fit ~spurious~ patterns in train+val that don't "
        "hold on later dates — the holdout window includes the May 2026 "
        "negative-price events and the seasonal regime that wasn't in val.\n\n"
        "Attention also *amplifies* the residual signal that the decoder gets, "
        "which makes it more sensitive to small distribution shifts. Plain LSTM, "
        "with its lossy 64-dim state bottleneck, is forced to be more conservative.\n\n"
        "**The fix:** more data (M5 weather adds genuinely new signal, not "
        "more capacity), or regularisation (dropout, recurrent_dropout, weight "
        "decay), or a smaller hidden state. We don't pursue any of those here — "
        "negative result on attention noted, simpler model wins.\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 5. What the attention map *would* tell us if it generalised\n\n"
        "Even though we won't ship attention, the heatmap is interpretable:\n"
        "- Bright vertical bands at history-QH ≈ 96·k for k = 1..7 mean the model "
        "has learned **daily seasonality** — when forecasting hour *i* of "
        "tomorrow, it weights hour *i* of *each* prior day.\n"
        "- A strong band at history-QH ≈ 671 (very right edge) says recent "
        "observations matter most — the residual drift signal SARIMAX exploits.\n"
        "- Horizontal stripes at evening forecast-QHs (~70-80) but not at "
        "midday (~40-60) would mean evening peak follows weekly habits, midday "
        "is weather-dominated. The plain LSTM has implicitly learned the same "
        "thing, just without the visualisation handle.\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 6. Loss curves & training summary"
    ),

    nbf.v4.new_code_cell(
        "fig = make_subplots(rows=1, cols=2,\n"
        "                    subplot_titles=('LSTM plain', 'LSTM with attention'))\n"
        "for i, m in enumerate([plain_meta, attn_meta], start=1):\n"
        "    h = m['history']\n"
        "    fig.add_trace(go.Scatter(x=list(range(1, len(h['loss'])+1)), y=h['loss'],\n"
        "                              name='train', line=dict(color='#2E86AB'),\n"
        "                              showlegend=(i == 1)), row=1, col=i)\n"
        "    if 'val_loss' in h:\n"
        "        fig.add_trace(go.Scatter(x=list(range(1, len(h['val_loss'])+1)), y=h['val_loss'],\n"
        "                                  name='val', line=dict(color='#C73E1D'),\n"
        "                                  showlegend=(i == 1)), row=1, col=i)\n"
        "fig.update_layout(height=380, template='plotly_white')\n"
        "fig.show()\n"
        "\n"
        "summary = pd.DataFrame([{\n"
        "    'model': m['model'],\n"
        "    'train_n': m['train_n'], 'val_n': m['val_n'],\n"
        "    'epochs': m['epochs_run'], 'train_time_s': round(m['train_time_s'], 1),\n"
        "    'val_residual_mae_mw': round(m['val_residual_mae_mw'], 1),\n"
        "    'val_implied_skill': round(m['val_implied_skill'], 4),\n"
        "} for m in (plain_meta, attn_meta)]).set_index('model')\n"
        "summary\n"
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
    print(f"Wrote {NB}; executing…")
    NotebookClient(nb, timeout=600, kernel_name="python3").execute()
    nbf.write(nb, NB)
    print(f"Executed and saved {NB}")


if __name__ == "__main__":
    main()
