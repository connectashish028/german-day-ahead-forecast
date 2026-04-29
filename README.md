# ⚡ Day-Ahead German Load Forecasting — Beating the TSO Baseline

> **Headline (placeholder until M15):** TensorFlow-based day-ahead quarter-hourly forecast for German grid load, benchmarked directly against the published TSO forecast on SMARD.
>
> **Skill score vs TSO**: _to be filled in after Milestone 4_
>
> **Live demo**: _to be filled in after Milestone 12_

## Why this project

Every European TSO publishes a day-ahead grid-load forecast — in Germany that forecast lives on [SMARD](https://www.smard.de/) as `fc_cons__grid_load` and is the operational baseline that utilities, traders, and BRPs anchor on. This project trains a TensorFlow model on the same public data and measures itself directly against that published baseline using a **skill score** (`1 - MAE_model / MAE_TSO`).

## What's in the box

- **Data**: SMARD German grid data, 2022-01 → 2026-03, quarter-hourly. Actual + forecast consumption, generation by source, and day-ahead prices for DE/LU plus 14 neighbour zones.
- **Stack**: TensorFlow 2.16 / Keras (Functional API), `tf.data` pipelines, MLflow, FastAPI, Streamlit, GitHub Actions.
- **Approach**: residual learning on top of the TSO forecast, real NWP weather, leakage-safe feature pipeline, quantile outputs (P10/P50/P90), conformal calibration, weekly online retraining.

## Repo layout

```
src/loadforecast/
  data/        # SMARD ingestion, cleaning, alignment
  features/    # leakage-safe feature builders
  models/      # Keras models (seq2seq LSTM, Conv-LSTM, TFT)
  backtest/    # rolling-origin evaluator + TSO baseline
  calibrate/   # split-conformal calibration
  serve/       # FastAPI inference service
tests/         # pytest (leakage, harness, API)
notebooks/     # exploration + archived prior work
dashboards/    # Streamlit app
scripts/       # one-off utilities
.github/workflows/   # daily forecast + weekly retrain
```

## Quickstart

```bash
# 1. Create env (conda + uv for speed)
conda create -n loadforecast python=3.11 -y
conda activate loadforecast
pip install uv
uv pip install -e ".[dev]"

# 2. Verify install
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
pytest -q
ruff check src/

# 3. (After M1) Run the backtest against the TSO baseline
python -m loadforecast.backtest --start 2025-01-01 --end 2026-03-01
```

## Status

Built milestone-by-milestone with hard verification gates. Current milestone tracked in [PLAN.md](PLAN.md) (or `~/.claude/plans/i-have-this-folder-sleepy-dragonfly.md`).

| # | Milestone | Status |
|---|-----------|--------|
| 0 | Repo hygiene | 🔄 in progress |
| 1 | Backtest harness | ⏳ |
| 2 | Leakage-safe features | ⏳ |
| 3 | Classical baselines | ⏳ |
| 4 | Seq2seq LSTM (TF) | ⏳ |
| 5 | Weather features | ⏳ |
| 6 | Quantile outputs | ⏳ |
| 7 | TFT / Conv-LSTM upgrade | ⏳ |
| 8 | Conformal calibration | ⏳ |
| 9 | Ensemble | ⏳ |
| 10 | FastAPI service | ⏳ |
| 11 | Daily orchestration | ⏳ |
| 12 | Streamlit dashboard | ⏳ |
| 13 | Weekly retrain | ⏳ |
| 14 | Drift monitoring | ⏳ |
| 15 | README polish | ⏳ |

## Data source

- [SMARD — Bundesnetzagentur](https://www.smard.de/) (public German electricity market data)
- [Open-Meteo](https://open-meteo.com/) (numerical weather predictions, free tier)

## License

MIT
