"""Builds and executes notebooks/09_price_forecast.ipynb.

This is a *living* notebook — we re-build it as the price-forecaster
project progresses. Each section that's done gets real data + charts;
each section that's pending shows a placeholder so the narrative arc
is visible end-to-end.

Sections (will fill in as we go):
  1. Why price forecasting
  2. Data audit (DONE)
  3. Design choices (DONE)
  4. Windowing (DONE)
  5. Training (PENDING)
  6. Backtest vs naive baselines (PENDING)
  7. Tomorrow's price preview (PENDING)
  8. Trading P&L on a 10MW/20MWh battery (PENDING)
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NB = Path("notebooks/09_price_forecast.ipynb")


CELLS = [
    nbf.v4.new_markdown_cell(
        "# 09 — Day-Ahead Price Forecasting\n\n"
        "Extending the load forecaster to predict the German day-ahead "
        "spot price. The architecture, leakage testing, refresh job, "
        "deployment, and dashboard all carry over verbatim — we only "
        "swap the target and the feature mix.\n\n"
        "**Why this exists.** Load forecasts are interesting to the TSO. "
        "*Price* forecasts are interesting to the **traders**. Battery "
        "operators, balancing-responsible parties, and intraday traders "
        "all anchor decisions on tomorrow's hourly clearing price — that's "
        "the signal that maps to € on a P&L statement.\n\n"
        "**Status:** this is a living notebook. Audit, design, and "
        "windowing are complete; training / backtest / P&L follow."
    ),

    nbf.v4.new_code_cell(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "# cwd-resolution preamble: notebooks live one level below repo root.\n"
        "_here = Path.cwd()\n"
        "ROOT = _here if (_here / 'pyproject.toml').exists() else _here.parent\n"
        "os.chdir(ROOT)\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n"
        "from plotly.subplots import make_subplots\n"
    ),

    # ----- 1. Why price forecasting -----
    nbf.v4.new_markdown_cell(
        "## 1. Why price forecasting\n\n"
        "The TSO's load forecast (which we beat by ~20 % in the load model) "
        "is the planning anchor for the grid. It does **not** tell anyone "
        "what to bid. The day-ahead spot price does.\n\n"
        "Concretely, a battery operator with 10 MW / 20 MWh of storage has a "
        "decision to make every day at 12:00 CET (the German day-ahead market "
        "gate): *which 15-minute slots tomorrow should I charge in, which should "
        "I discharge in?* The answer depends on tomorrow's price profile — "
        "specifically on the spread between the cheapest and most expensive "
        "slots. A better forecast = more profitable dispatch.\n\n"
        "This notebook builds that forecaster."
    ),

    # ----- 2. Data audit -----
    nbf.v4.new_markdown_cell(
        "## 2. Data audit\n\n"
        "Before designing the model, three things a senior DS would want to "
        "know about the target series."
    ),

    nbf.v4.new_code_cell(
        "PRICE = 'price__germany_luxembourg'\n"
        "df = pd.read_parquet('smard_merged_15min.parquet')\n"
        "px = df[PRICE].dropna()\n"
        "\n"
        "print(f'Coverage: {px.index.min()} -> {px.index.max()}  '\n"
        "      f'({len(px):,} rows, {df[PRICE].isna().sum():,} NaN)')\n"
        "\n"
        "# Year-over-year mean — non-stationarity check\n"
        "annual = px.groupby(px.index.year).agg(['mean', 'std'])\n"
        "annual.columns = ['mean €/MWh', 'std €/MWh']\n"
        "print('\\nAnnual stats:')\n"
        "print(annual.round(1).to_string())\n"
        "\n"
        "# Resolution check — pre-Oct-2025 was hourly, post is native 15-min\n"
        "h1 = px['2025-01':'2025-06']\n"
        "h2 = px['2025-11':'2026-04']\n"
        "u_h1 = h1.groupby(h1.index.floor('h')).nunique().mean()\n"
        "u_h2 = h2.groupby(h2.index.floor('h')).nunique().mean()\n"
        "print(f'\\nUnique prices per hour:')\n"
        "print(f'  2025 H1 (pre 15-min auction): {u_h1:.2f}')\n"
        "print(f'  2025 H2 (post 15-min auction): {u_h2:.2f}')\n"
        "\n"
        "# Negative-price frequency\n"
        "neg = px[px < 0]\n"
        "print(f'\\nNegative-price slots: {len(neg):,} / {len(px):,} = '\n"
        "      f'{len(neg)/len(px)*100:.1f} %')\n"
        "print(f'Most negative: {neg.idxmin()} = {neg.min():.0f} €/MWh')\n"
    ),

    nbf.v4.new_markdown_cell(
        "**Three findings that shape the design:**\n\n"
        "1. **Severe non-stationarity.** The 2022 mean (€235) sits in a totally "
        "different regime from 2024 onward (€78–€96). Training on the full "
        "history would teach the model patterns that no longer apply.\n\n"
        "2. **Structural break Oct 2025.** Day-ahead prices were hourly until "
        "EPEX migrated to a native 15-min auction. Pre-migration, all 4 quarter-hours "
        "of an hour share the same price (forward-filled); post-migration they vary. "
        "Sub-hour patterns are only learnable from late-2025 onward.\n\n"
        "3. **Negatives are common (4.2 %), not rare.** Rules out log-price as a "
        "target. Quantile loss handles the wide range natively, same recipe as the "
        "load model."
    ),

    # ----- 2b. Distribution chart -----
    nbf.v4.new_code_cell(
        "fig = make_subplots(\n"
        "    rows=1, cols=2,\n"
        "    subplot_titles=('Daily mean price by year', 'Distribution per year'),\n"
        "    column_widths=[0.55, 0.45],\n"
        ")\n"
        "\n"
        "daily_mean = px.resample('D').mean()\n"
        "fig.add_trace(\n"
        "    go.Scatter(x=daily_mean.index, y=daily_mean.values, mode='lines',\n"
        "               line=dict(color='#B8A1FF', width=1),\n"
        "               name='daily mean', showlegend=False),\n"
        "    row=1, col=1,\n"
        ")\n"
        "fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', dash='dot'),\n"
        "              row=1, col=1)\n"
        "\n"
        "for year, color in zip([2022, 2023, 2024, 2025, 2026],\n"
        "                       ['#888', '#aaa', '#3B82F6', '#B8A1FF', '#fff'], strict=True):\n"
        "    yr = px[px.index.year == year]\n"
        "    if len(yr):\n"
        "        fig.add_trace(\n"
        "            go.Box(y=yr.values, name=str(year), marker_color=color,\n"
        "                   showlegend=False, boxpoints=False),\n"
        "            row=1, col=2,\n"
        "        )\n"
        "\n"
        "fig.update_layout(\n"
        "    height=350, paper_bgcolor='#1f2228', plot_bgcolor='#1f2228',\n"
        "    font=dict(color='#fff'),\n"
        "    margin=dict(l=40, r=20, t=40, b=40),\n"
        ")\n"
        "fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', color='rgba(255,255,255,0.7)')\n"
        "fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', color='rgba(255,255,255,0.7)',\n"
        "                 title_text='€/MWh', row=1, col=1)\n"
        "fig.update_yaxes(title_text='€/MWh', row=1, col=2,\n"
        "                 gridcolor='rgba(255,255,255,0.1)', color='rgba(255,255,255,0.7)')\n"
        "fig.show()\n"
    ),

    # ----- 3. Design -----
    nbf.v4.new_markdown_cell(
        "## 3. Design choices\n\n"
        "| Choice | What | Why |\n"
        "|---|---|---|\n"
        "| **Target** | Raw price | Negatives rule out log-price; quantile loss handles wide range |\n"
        "| **Training window** | 2024-01-01 → 2025-12-31 | Captures post-crisis regime + most of the 15-min-native era |\n"
        "| **Validation** | 2026-01 → 2026-02 | ~60 days, recent regime, distinct from holdout |\n"
        "| **Holdout** | 2026-03 → 2026-04 | Same period as load model — apples-to-apples |\n"
        "| **Encoder features** | price history, load actuals, PV+wind actuals, calendar | What a trader sees at issue T |\n"
        "| **Decoder features** | TSO load forecast, weather forecast, calendar | All published before D-1 12:00 Berlin (gate closure) |\n"
        "| **Loss** | Pinball loss on (0.1, 0.5, 0.9) | Same recipe as load quantile model |\n"
        "| **Architecture** | seq2seq LSTM(64), 3 quantile heads | Reuse `lstm_quantile.py` verbatim |\n"
        "| **Naive baseline** | Yesterday-same-hour price | Standard electricity-price forecasting baseline |\n\n"
        "### Decoder feature: TSO renewable-generation forecast\n\n"
        "The single most important price driver in Germany is renewable supply. "
        "Sunny + windy day → cheap (or negative) prices. The TSO publishes its "
        "**day-ahead PV + wind generation forecast** at D-1 ~10:00 CET, comfortably "
        "before our issue time T = D-1 12:00. Adding this as a decoder feature "
        "(`fc_gen__photovoltaics_and_wind`, fetched via the SMARD downloadcenter "
        "API) let the model see exactly what the market is anticipating from "
        "renewables tomorrow.\n\n"
        "We also re-discovered the right SMARD module ID for this column "
        "(`2005097`) — confirmed it's a forecast (not actuals) by comparing to "
        "our existing `actual_gen__pv + wind_*` sum on a known past day: 3.4 % "
        "mean abs diff, the signature of a forecast.\n\n"
        "### Deliberately not included\n\n"
        "- **Cross-border prices** — published at the same auction as DE; not "
        "available at issue time. (Same reason we excluded them from the load model.)\n"
        "- **Gas / coal fuel prices** — major price driver but not in our parquet. "
        "Could add later."
    ),

    # ----- 4. Windowing smoke test -----
    nbf.v4.new_markdown_cell(
        "## 4. Windowing\n\n"
        "Build a single window for a known good delivery date in the holdout "
        "and check: shapes match the design, no NaN cells, value ranges sensible.\n\n"
        "Important fix from this work: the original price availability rule was "
        "too conservative (`ts < T - 12h`), confusing 'D-1 prices' (yesterday's "
        "delivery, known) with 'prices auctioned at D-1' (tomorrow's delivery, "
        "not yet known). Corrected to `ts < T + 12h` — at issue time T = D-1 12:00, "
        "all prices for delivery times before D 00:00 Berlin (= T + 12h) are "
        "published and known.\n\n"
        "All 24 leakage tests still pass with the corrected rule."
    ),

    nbf.v4.new_code_cell(
        "from datetime import date\n"
        "\n"
        "from loadforecast.backtest import issue_time_for, load_smard_15min\n"
        "from loadforecast.models.price_dataset import (\n"
        "    PRICE_ENC_FEATURE_NAMES,\n"
        "    build_price_window,\n"
        ")\n"
        "\n"
        "df = load_smard_15min('smard_merged_15min.parquet')\n"
        "issue = issue_time_for(date(2026, 4, 15))\n"
        "w = build_price_window(df, issue, include_weather=True)\n"
        "\n"
        "print(f'Issue time:        {issue}')\n"
        "print(f'X_enc shape:       {w.X_enc.shape}  '\n"
        "      f'(NaN: {int(np.isnan(w.X_enc).sum())})')\n"
        "print(f'X_dec shape:       {w.X_dec.shape}  '\n"
        "      f'(NaN: {int(np.isnan(w.X_dec).sum())})')\n"
        "print(f'y_price shape:     {w.y_price.shape}  '\n"
        "      f'(NaN: {int(np.isnan(w.y_price).sum())})')\n"
        "print(f'\\nDelivery price (€/MWh): mean={w.y_price.mean():.1f}, '\n"
        "      f'min={w.y_price.min():.1f}, max={w.y_price.max():.1f}')\n"
    ),

    nbf.v4.new_code_cell(
        "# Plot the encoder's price history + the day's actual price target,\n"
        "# all in Berlin local time.\n"
        "enc_idx = pd.date_range(\n"
        "    issue - pd.Timedelta(minutes=15 * 672), issue,\n"
        "    freq='15min', inclusive='left',\n"
        ").tz_convert('Europe/Berlin')\n"
        "tgt_idx = w.target_idx.tz_convert('Europe/Berlin')\n"
        "\n"
        "fig = go.Figure()\n"
        "fig.add_trace(go.Scatter(\n"
        "    x=enc_idx, y=w.X_enc[:, 0],  # column 0 = price\n"
        "    mode='lines', line=dict(color='#B8A1FF', width=1.4),\n"
        "    name='Encoder: price history (7 days)',\n"
        "))\n"
        "fig.add_trace(go.Scatter(\n"
        "    x=tgt_idx, y=w.y_price,\n"
        "    mode='lines', line=dict(color='#3B82F6', width=2.2),\n"
        "    name='Target: delivery-day price',\n"
        "))\n"
        "issue_berlin = issue.tz_convert('Europe/Berlin')\n"
        "fig.add_shape(type='line',\n"
        "              x0=issue_berlin, x1=issue_berlin,\n"
        "              y0=0, y1=1, yref='paper',\n"
        "              line=dict(color='rgba(255,255,255,0.4)', dash='dash'))\n"
        "fig.add_annotation(x=issue_berlin, y=1, yref='paper', yanchor='bottom',\n"
        "                   text='issue time T', showarrow=False,\n"
        "                   font=dict(color='rgba(255,255,255,0.6)', size=10))\n"
        "fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.25)', dash='dot'))\n"
        "fig.update_layout(\n"
        "    height=380, paper_bgcolor='#1f2228', plot_bgcolor='#1f2228',\n"
        "    font=dict(color='#fff'),\n"
        "    title='Single window: 7-day encoder + 1-day target',\n"
        "    xaxis=dict(title='Time (Berlin)', gridcolor='rgba(255,255,255,0.1)',\n"
        "               color='rgba(255,255,255,0.7)'),\n"
        "    yaxis=dict(title='€ / MWh', gridcolor='rgba(255,255,255,0.1)',\n"
        "               color='rgba(255,255,255,0.7)'),\n"
        "    margin=dict(l=50, r=20, t=50, b=50),\n"
        "    legend=dict(orientation='h', y=1.08, bgcolor='rgba(0,0,0,0)'),\n"
        ")\n"
        "fig.show()\n"
    ),

    # ----- 5. Training -----
    nbf.v4.new_markdown_cell(
        "## 5. Training\n\n"
        "Reused `lstm_quantile.py` verbatim — 64-unit encoder + 64-unit decoder + "
        "3 quantile heads. ~75 s on CPU, 19 epochs (early-stopped from 60). "
        "Saved to `model_checkpoints/price_quantile_v1/`.\n\n"
        "Validation results (Jan–Feb 2026):"
    ),

    nbf.v4.new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "meta = json.loads(Path('model_checkpoints/price_quantile_v1/meta.json').read_text())\n"
        "rows = [\n"
        "    ('Validation P50 MAE',     f\"{meta['val_p50_mae_eur_mwh']:.2f} €/MWh\"),\n"
        "    ('80 % band coverage',     f\"{meta['val_interval_coverage']*100:.1f} %  (target 80 %)\"),\n"
        "    ('Mean band width',        f\"{meta['val_interval_width_eur_mwh']:.1f} €/MWh\"),\n"
        "    ('Quantile crossings',     f\"{meta['val_quantile_crossings']*100:.1f} %\"),\n"
        "    ('Training days',          f\"{meta['train_n']}\"),\n"
        "    ('Validation days',        f\"{meta['val_n']}\"),\n"
        "    ('Training epochs',        f\"{meta['epochs_run']} (early-stopped)\"),\n"
        "    ('Training time',          f\"{meta['train_time_s']:.0f} s\"),\n"
        "]\n"
        "pd.DataFrame(rows, columns=['Metric', 'Value'])\n"
    ),

    # ----- 6. Backtest -----
    nbf.v4.new_markdown_cell(
        "## 6. Backtest on the holdout (Mar–Apr 2026) — the iteration story\n\n"
        "Compared the model's P50 against two naive baselines:\n\n"
        "- **Yesterday-same-quarter-hour** — the standard electricity-price-forecast "
        "baseline\n"
        "- **Last-week-same-quarter-hour** — captures weekly seasonality\n\n"
        "Two metrics — point MAE and **daily price-spread MAE** (`max − min` within "
        "the day). Battery dispatch makes money from spread, not point accuracy. "
        "A model with low point MAE but bad spread accuracy is **worse for trading** "
        "than a noisy model that captures the spread.\n\n"
        "### v1 → v2: what changed and why\n\n"
        "**v1 (initial)** used encoder = price + load + actual VRE history; decoder = "
        "TSO load forecast + weather. Got **+18 % point MAE vs naive** but **−65 % spread MAE** — "
        "the model was noticeably more accurate on average but compressed the daily "
        "extremes badly. Senior-DS read: the median objective drags the prediction toward "
        "the centre of the distribution; that's a known property of pinball-loss training.\n\n"
        "**Diagnosis**: the model didn't see the dominant price driver — *tomorrow's* "
        "renewable generation. We had weather (which is upstream of generation) but "
        "not the TSO's day-ahead PV + wind forecast itself. Without it, the model has "
        "to *infer* renewable supply from temperature / radiation / wind speed — "
        "noisier than reading the operator's published forecast directly.\n\n"
        "**v2** adds `fc_gen__photovoltaics_and_wind` as a decoder feature. Same "
        "architecture, same loss, same training recipe. The numbers below are v2."
    ),

    nbf.v4.new_code_cell(
        "bt = pd.read_csv('backtest_results/price_quantile_holdout.csv',\n"
        "                 parse_dates=['target_ts'])\n"
        "\n"
        "mae_model = (bt['y_true'] - bt['p50']).abs().mean()\n"
        "mae_n1 = (bt['y_true'] - bt['naive_1d']).abs().mean()\n"
        "mae_n7 = (bt['y_true'] - bt['naive_7d']).abs().mean()\n"
        "cov = ((bt['y_true'] >= bt['p10']) & (bt['y_true'] <= bt['p90'])).mean()\n"
        "width = (bt['p90'] - bt['p10']).mean()\n"
        "\n"
        "daily = bt.groupby('issue_date').agg(\n"
        "    true_spread=('y_true', lambda s: s.max() - s.min()),\n"
        "    model_spread=('p50', lambda s: s.max() - s.min()),\n"
        "    naive_1d_spread=('naive_1d', lambda s: s.max() - s.min()),\n"
        ")\n"
        "spread_mae_model = (daily['true_spread'] - daily['model_spread']).abs().mean()\n"
        "spread_mae_naive = (daily['true_spread'] - daily['naive_1d_spread']).abs().mean()\n"
        "\n"
        "scoreboard = pd.DataFrame([\n"
        "    {'Metric': 'P50 point MAE (€/MWh)',                 'Model': f'{mae_model:.2f}',\n"
        "     'Naive yesterday': f'{mae_n1:.2f}',\n"
        "     'Naive last-week': f'{mae_n7:.2f}',\n"
        "     'Δ vs best naive': f'{(1-mae_model/mae_n1)*100:+.1f} %'},\n"
        "    {'Metric': 'Daily spread MAE (€/MWh)',              'Model': f'{spread_mae_model:.2f}',\n"
        "     'Naive yesterday': f'{spread_mae_naive:.2f}',\n"
        "     'Naive last-week': '—',\n"
        "     'Δ vs best naive': f'{(1-spread_mae_model/spread_mae_naive)*100:+.1f} %'},\n"
        "])\n"
        "scoreboard\n"
    ),

    nbf.v4.new_markdown_cell(
        "### What this means (v2 read)\n\n"
        "**Point MAE: +34 % vs the best naive — a meaningful win.** Adding the "
        "TSO renewable forecast nearly doubled the lift over v1 (+18 % → +34 %). "
        "On its own this is a genuinely good price forecaster.\n\n"
        "**Spread MAE: gap closed from −65 % to −23 %, but still trails.** The "
        "median-collapse problem isn't fully solvable inside a pinball-loss "
        "framework — the P50 by construction is dragged toward the centre of the "
        "distribution. We've narrowed the gap a lot by giving the model the right "
        "feature, but a model with smooth median predictions will always under-shoot "
        "the daily extremes vs a noisy actual replay (which is what naive yesterday is).\n\n"
        "**Coverage stuck at ~52 %.** The 80 % bands actually contain the truth half "
        "the time on holdout. Validation showed 89 % — that's distribution shift "
        "between Jan-Feb 2026 (val) and Mar-Apr 2026 (holdout). **Conformal calibration** "
        "with a rolling window would fix this with theoretical guarantees; deferred for now.\n\n"
        "### The pivot for trading (Section 8)\n\n"
        "Rather than chase the spread metric with more architecture work, the "
        "right move for SUENA's actual question — **what's this forecast worth in € "
        "for a battery operator?** — is to **use the right quantile for each role**:\n\n"
        "- **P10 prices** → identify the cheapest 12 quarter-hours to charge\n"
        "- **P90 prices** → identify the most expensive 12 quarter-hours to discharge\n\n"
        "The wide bands exist because the model is uncertain at the extremes. P10 and P90 "
        "carry the volatility information that P50 throws away. Using them directly "
        "for dispatch sidesteps the spread issue entirely — and is the operationally "
        "correct framing anyway: a trader sizing a position cares about the worst case "
        "(P10) and the best case (P90), not the median."
    ),

    nbf.v4.new_code_cell(
        "# Daily MAE comparison chart — shows where the model wins and loses.\n"
        "import plotly.graph_objects as go\n"
        "\n"
        "daily_err = bt.groupby('issue_date').agg(\n"
        "    model=('y_true', lambda s: (s - bt.loc[s.index, 'p50']).abs().mean()),\n"
        "    naive_1d=('y_true', lambda s: (s - bt.loc[s.index, 'naive_1d']).abs().mean()),\n"
        ")\n"
        "daily_err.index = pd.to_datetime(daily_err.index)\n"
        "\n"
        "fig = go.Figure()\n"
        "fig.add_trace(go.Scatter(x=daily_err.index, y=daily_err['naive_1d'],\n"
        "    mode='lines', name='Naive yesterday',\n"
        "    line=dict(color='rgba(255,255,255,0.45)', width=1.4, dash='dash')))\n"
        "fig.add_trace(go.Scatter(x=daily_err.index, y=daily_err['model'],\n"
        "    mode='lines', name='Model (P50)',\n"
        "    line=dict(color='#B8A1FF', width=2.2)))\n"
        "fig.update_layout(\n"
        "    height=350, paper_bgcolor='#1f2228', plot_bgcolor='#1f2228',\n"
        "    font=dict(color='#fff'),\n"
        "    title='Daily P50 MAE on the holdout — model vs naive',\n"
        "    xaxis=dict(title='Delivery date', gridcolor='rgba(255,255,255,0.1)',\n"
        "               color='rgba(255,255,255,0.7)'),\n"
        "    yaxis=dict(title='MAE (€/MWh)', gridcolor='rgba(255,255,255,0.1)',\n"
        "               color='rgba(255,255,255,0.7)'),\n"
        "    margin=dict(l=50, r=20, t=50, b=50),\n"
        "    legend=dict(orientation='h', y=1.08, bgcolor='rgba(0,0,0,0)'),\n"
        ")\n"
        "fig.show()\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 7. Tomorrow's price preview\n\n"
        "*Pending.* Add a section to the deployed dashboard showing tomorrow's "
        "P10/P50/P90 price, alongside tomorrow's load forecast."
    ),

    nbf.v4.new_markdown_cell(
        "## 8. Trading P&L on a 10 MW / 20 MWh battery\n\n"
        "*Pending.* The single most important section for the SUENA application. "
        "Simulate a battery dispatching against:\n\n"
        "- Our forecast (baseline)\n"
        "- Naive forecast (yesterday-same-hour)\n"
        "- Perfect-foresight (oracle, the theoretical maximum)\n\n"
        "**Battery spec:** 10 MW / 20 MWh, 90 % round-trip efficiency, max 3 cycles/day. "
        "Strategy: greedy — at issue time, pick the cheapest 12 quarter-hours to "
        "charge in (3 × 4 = 12 slots × 0.25h × 10 MW = 30 MWh charge volume) and "
        "the most expensive 12 to discharge.\n\n"
        "Report: € per MWh of battery capacity per day, captured fraction of "
        "perfect-foresight P&L, € uplift over naive."
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    NB.parent.mkdir(parents=True, exist_ok=True)
    print(f"Executing {NB}...")
    NotebookClient(nb, timeout=300, kernel_name="python3").execute()
    nbf.write(nb, NB)
    print(f"Wrote {NB}")


if __name__ == "__main__":
    main()
