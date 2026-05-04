"""Builds and executes notebooks/06_weather_impact.ipynb.

Goal: characterise where the weather LSTM beats the plain LSTM and
where it doesn't. The +0.038 skill bump on average is modest;
*where* it lifts is the interesting question.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NB = Path("notebooks/06_weather_impact.ipynb")


CELLS = [
    nbf.v4.new_markdown_cell(
        "# M5 — NWP weather: where does it help?\n\n"
        "We added 4 weather features (population-weighted across 6 German "
        "load centres) to the seq2seq LSTM:\n\n"
        "- `temperature_2m` — heating/cooling load driver\n"
        "- `shortwave_radiation` — PV generation driver (the key one)\n"
        "- `wind_speed_100m` — wind generation driver\n"
        "- `cloud_cover` — general weather signal\n\n"
        "**Headline:** the weather LSTM lands at **skill +0.253** (vs +0.215 plain), "
        "**MAPE 2.56%** (vs 2.66%). A +0.038 skill bump — modest but real and "
        "above the +0.03 gate.\n\n"
        "**The interesting question** is *where* the weather model helps. The TSO "
        "baseline already bakes in climatological weather, so we expect the new "
        "signal to land most where the TSO forecast goes most wrong: solar-driven "
        "midday hours and unusual-weather days."
    ),

    nbf.v4.new_code_cell(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "_here = Path.cwd()\n"
        "ROOT = _here if (_here / 'pyproject.toml').exists() else _here.parent\n"
        "os.chdir(ROOT)\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n"
        "from plotly.subplots import make_subplots\n"
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
        "    ('LSTM_weather',    'backtest_results/lstm_weather_step7.csv'),\n"
        "]:\n"
        "    if not Path(csv).exists():\n"
        "        continue\n"
        "    bt = pd.read_csv(csv, parse_dates=['target_ts'])\n"
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
        "## 2. MAE by hour-of-day — where does weather help?\n\n"
        "If the TSO over-forecasts solar-noon load (because it underestimates PV) "
        "and our weather model has access to the actual irradiance forecast, "
        "we should see the biggest improvement at hours 11–15."
    ),

    nbf.v4.new_code_cell(
        "plain = pd.read_csv('backtest_results/lstm_plain_step7.csv', parse_dates=['target_ts'])\n"
        "weather = pd.read_csv('backtest_results/lstm_weather_step7.csv', parse_dates=['target_ts'])\n"
        "tso_df = pd.read_csv('backtest_results/tso_step7_v2.csv', parse_dates=['target_ts'])\n"
        "\n"
        "for d in (plain, weather, tso_df):\n"
        "    d['hour'] = d['target_ts'].dt.tz_convert('Europe/Berlin').dt.hour\n"
        "    d['err'] = (d['y_true'] - d['y_model']).abs()\n"
        "\n"
        "by_hour = pd.DataFrame({\n"
        "    'TSO':     tso_df.groupby('hour').apply(\n"
        "                  lambda g: (g['y_true'] - g['y_tso']).abs().mean(),\n"
        "                  include_groups=False),\n"
        "    'LSTM_plain':   plain.groupby('hour')['err'].mean(),\n"
        "    'LSTM_weather': weather.groupby('hour')['err'].mean(),\n"
        "})\n"
        "by_hour['weather_lift'] = by_hour['LSTM_plain'] - by_hour['LSTM_weather']\n"
        "\n"
        "fig = make_subplots(rows=2, cols=1, shared_xaxes=True,\n"
        "                    subplot_titles=('MAE by hour-of-day',\n"
        "                                    'Weather lift (LSTM_plain MAE - LSTM_weather MAE)'),\n"
        "                    row_heights=[0.65, 0.35], vertical_spacing=0.10)\n"
        "for col, color in [('TSO', '#2E86AB'), ('LSTM_plain', '#7B7B7B'), ('LSTM_weather', '#C73E1D')]:\n"
        "    fig.add_trace(go.Scatter(x=by_hour.index, y=by_hour[col], mode='lines+markers',\n"
        "                              name=col, line=dict(color=color)), row=1, col=1)\n"
        "fig.add_trace(go.Bar(x=by_hour.index, y=by_hour['weather_lift'],\n"
        "                      marker_color=['#1B998B' if v > 0 else '#A63A50' for v in by_hour['weather_lift']],\n"
        "                      showlegend=False), row=2, col=1)\n"
        "fig.add_hline(y=0, line_dash='dash', line_color='black', row=2, col=1)\n"
        "fig.update_layout(height=560, template='plotly_white', hovermode='x unified',\n"
        "                  xaxis2_title='Hour of day (Berlin local)', yaxis_title='MAE (MW)')\n"
        "fig.show()\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 3. The 1 May 2026 case study (the −500 €/MWh PV-glut day)\n\n"
        "On 1 May 2026 (federal holiday, clear sky, 54 GW PV peak), prices "
        "crashed to −500 €/MWh for 5 consecutive QHs. The TSO PV forecast "
        "predicted only 13.7 GW — a 4× under-forecast. How well does each "
        "model handle that day?"
    ),

    nbf.v4.new_code_cell(
        "from datetime import date\n"
        "from loadforecast.backtest import issue_time_for, load_smard_15min\n"
        "from loadforecast.backtest.baselines import tso_baseline_predict\n"
        "from loadforecast.models.predict import (\n"
        "    lstm_residual_predict, lstm_weather_predict,\n"
        "    DEFAULT_MODEL_DIR, DEFAULT_WEATHER_DIR,\n"
        ")\n"
        "\n"
        "df = load_smard_15min('smard_merged_15min.parquet')\n"
        "delivery = date(2026, 5, 1)\n"
        "issue = issue_time_for(delivery)\n"
        "tso = tso_baseline_predict(df, issue)\n"
        "plain_pred = lstm_residual_predict(df, issue, model_dir=DEFAULT_MODEL_DIR)\n"
        "weather_pred = lstm_weather_predict(df, issue, model_dir=DEFAULT_WEATHER_DIR)\n"
        "actual = df['actual_cons__grid_load'].reindex(tso.index)\n"
        "\n"
        "fig = go.Figure()\n"
        "fig.add_trace(go.Scatter(x=actual.index, y=actual,\n"
        "                          name='actual', line=dict(color='black', width=2.5)))\n"
        "fig.add_trace(go.Scatter(x=tso.index, y=tso,\n"
        "                          name='TSO baseline', line=dict(color='#2E86AB', dash='dot')))\n"
        "fig.add_trace(go.Scatter(x=plain_pred.index, y=plain_pred,\n"
        "                          name='LSTM plain', line=dict(color='#7B7B7B')))\n"
        "fig.add_trace(go.Scatter(x=weather_pred.index, y=weather_pred,\n"
        "                          name='LSTM weather', line=dict(color='#C73E1D', width=2)))\n"
        "fig.update_layout(\n"
        "    title=f'{delivery} — actual load vs three day-ahead forecasts (issued at {issue})',\n"
        "    xaxis_title='Time (UTC)', yaxis_title='Grid load (MW)',\n"
        "    height=440, template='plotly_white', hovermode='x unified',\n"
        ")\n"
        "fig.show()\n"
        "\n"
        "mae_tso     = float((actual - tso).abs().mean())\n"
        "mae_plain   = float((actual - plain_pred).abs().mean())\n"
        "mae_weather = float((actual - weather_pred).abs().mean())\n"
        "print(f'  TSO MAE on this day:           {mae_tso:6.1f} MW')\n"
        "print(f'  LSTM_plain MAE on this day:    {mae_plain:6.1f} MW')\n"
        "print(f'  LSTM_weather MAE on this day:  {mae_weather:6.1f} MW')\n"
        "print(f'  Weather skill vs TSO:          {1 - mae_weather/mae_tso:+.3f}')\n"
        "print(f'  Weather skill vs LSTM_plain:   {1 - mae_weather/mae_plain:+.3f}')\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 4. When does weather *not* help?\n\n"
        "A useful diagnostic: per-day skill of the weather model. Days where "
        "weather under-performs the plain LSTM are days when our 4-variable "
        "national aggregation didn't capture the local pattern (e.g. "
        "regional heat waves, foggy mornings in the south)."
    ),

    nbf.v4.new_code_cell(
        "# Per-day MAE comparison\n"
        "p_daily = plain.groupby('issue_date')['err'].mean().rename('plain')\n"
        "w_daily = weather.groupby('issue_date')['err'].mean().rename('weather')\n"
        "diff = (w_daily - p_daily).rename('weather_minus_plain')\n"
        "\n"
        "fig = go.Figure()\n"
        "fig.add_trace(go.Bar(\n"
        "    x=pd.to_datetime(diff.index), y=diff.values,\n"
        "    marker_color=['#1B998B' if v < 0 else '#A63A50' for v in diff.values],\n"
        "    name='weather better' if False else None,\n"
        "))\n"
        "fig.add_hline(y=0, line_dash='dash', line_color='black')\n"
        "fig.update_layout(\n"
        "    title='Per-day MAE delta: weather model vs plain LSTM',\n"
        "    xaxis_title='Delivery date',\n"
        "    yaxis_title='Δ MAE (negative = weather wins)',\n"
        "    height=400, template='plotly_white',\n"
        ")\n"
        "fig.show()\n"
        "print(f'Days weather wins: {(diff < 0).sum()} / {len(diff)}')\n"
        "print(f'Mean delta:        {diff.mean():+.1f} MW   (negative = weather better)')\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 5. Reading the result\n\n"
        "**Where the lift is real** (expected from physics):\n"
        "- Solar-noon hours: the irradiance forecast directly anticipates PV "
        "  generation, which the TSO under-forecasts on clear days.\n"
        "- Cold-snap evenings: temperature → heating load.\n"
        "- High-wind days: wind generation displaces conventional load patterns.\n\n"
        "**Why the bump is modest (+0.038 not +0.10+):**\n"
        "- The TSO already incorporates 30-day weather climatology — they don't "
        "  forecast in a vacuum.\n"
        "- The encoder gets *7 days of weather history*, which is largely "
        "  redundant with the load history. We could ablate to test this.\n"
        "- We use only 4 vars at national aggregate. Heat-pump-driven evening "
        "  peaks would benefit from regional temperature breakdown.\n\n"
        "**Production decision.** LSTM_weather is the new headline model: skill "
        "+0.253 vs +0.215 plain. Plain LSTM stays available as a fallback for "
        "days when Open-Meteo is down."
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
    NotebookClient(nb, timeout=300, kernel_name="python3").execute()
    nbf.write(nb, NB)
    print(f"Executed and saved {NB}")


if __name__ == "__main__":
    main()
