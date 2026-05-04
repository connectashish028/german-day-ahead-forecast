# ⚡ Day-Ahead German Load Forecasting — Beating the TSO Baseline

> A TensorFlow seq2seq LSTM that day-ahead-forecasts the German grid load
> at 15-minute resolution and **beats the published TSO forecast by 21.5%**
> on a stratified 70-date holdout window.

| Predictor | MAE (MW) | MAPE (%) | Skill vs TSO |
|---|---|---|---|
| Seasonal-naive (D-7 actuals) | 618.3 | 4.21 | −0.256 |
| **TSO baseline** (`fc_cons__grid_load` on SMARD) | **492.4** | **3.36** | 0.000 |
| SARIMAX-on-residual | 441.7 | 3.00 | +0.103 |
| **Plain seq2seq LSTM (this work)** | **386.7** | **2.66** | **+0.215** |

*Backtest: rolling origin over 2025-01-01 → 2026-04-30, step 7 days, n = 70 delivery dates.
Skill = 1 − MAE_model / MAE_TSO. No weather inputs yet — weather lands in M5.*

## Why this project

Every European TSO publishes a day-ahead grid-load forecast — in Germany it lives on
[SMARD](https://www.smard.de/) as `fc_cons__grid_load` and is the operational baseline
that utilities, traders, and balancing-responsible parties anchor on. This project
trains a TensorFlow model on the same public data and measures itself directly against
that published baseline using a **skill score** (`1 − MAE_model / MAE_TSO`).

Most portfolio time-series projects compare a model to a naive baseline and stop there.
Comparing to a real, public, operational forecast — and beating it — is qualitatively
different.

## Approach

- **Residual learning.** The target is `actual_load − TSO_forecast`. The TSO already
  gets the easy 90% (calendar, climatology). The model only has to learn the
  systematic *errors*, which are large, structured, and stable — the TSO
  over-forecasts midday consumption by ~250 MW because it under-estimates PV.
- **Seq2seq LSTM** (encoder LSTM(64) → state → decoder LSTM(64) → Dense). 36k params.
  Trained in 3.2 minutes on CPU.
- **Leakage-safe feature pipeline** with a corrupt-future test: scramble every
  post-issue-time value in the source frame, rebuild features, assert bit-for-bit
  identical. Tested across 9 stratified delivery dates including DST transitions and
  holidays.
- **Multi-source data layer.** Energy-Charts (prices, generation), SMARD (load,
  forecasts), Open-Meteo (weather, M5). Idempotent refresh: one CLI command rebuilds
  the parquet from public APIs.

## Repo layout

```
src/loadforecast/
  data/        # multi-source ingestion (Energy-Charts, SMARD, Open-Meteo)
  features/    # leakage-safe feature builders (calendar, lags, availability)
  models/      # Keras models, dataset windowing, predict wrappers
  backtest/    # rolling-origin evaluator + TSO + SARIMAX baselines
  serve/       # FastAPI inference service (M10)
tests/         # pytest — 31 tests including 24 leakage tests
notebooks/     # 4 visualisation + explanation notebooks
scripts/       # training, refresh, exploration utilities
```

## Quickstart

```bash
# 1. Create env (conda + uv)
conda create -n loadforecast python=3.11 -y
conda activate loadforecast
pip install uv
uv pip install -e ".[dev]"

# 2. Verify install
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
pytest -q          # 31 tests
ruff check src/

# 3. Refresh the data parquet from public APIs
python -m loadforecast.data.refresh --rebuild --start 2022-01-01 --through 2026-05-04

# 4. Train the LSTM (~3 min on CPU)
python scripts/train_lstm_plain.py

# 5. Backtest against the TSO + classical baselines
python -m loadforecast.backtest --predictor lstm_plain \
    --start 2025-01-01 --end 2026-04-30 --step-days 7
```

## Notebooks

- **[01 – Backtest visualisation](notebooks/01_backtest_visualization.ipynb)** —
  TSO baseline characterised: 504 MW MAE, 3.73% MAPE; error by hour-of-day shows
  the persistent midday over-forecast.
- **[02 – Feature pipeline](notebooks/02_feature_pipeline_visualization.ipynb)** —
  availability rules, leakage-safe lags, rolling stats, top features by correlation
  with the residual.
- **[03 – Baseline shoot-out](notebooks/03_baselines_visualization.ipynb)** —
  TSO vs seasonal-naive vs SARIMAX. SARIMAX-on-residual beats the TSO by +11%
  skill — a no-weather AR model already extracts the persistent bias.
- **[04 – LSTM explained](notebooks/04_lstm_explained.ipynb)** —
  plain-language tour of the seq2seq architecture, walked through with one delivery
  day end-to-end. Loss curves, encoder/decoder inputs, predicted residual.

## Data sources

| Source | What | Auth |
|---|---|---|
| [Energy-Charts](https://api.energy-charts.info/) (Fraunhofer ISE) | Day-ahead prices for 15 bidding zones, actual generation by source | none |
| [SMARD](https://www.smard.de/) (Bundesnetzagentur) | Total grid load, residual load (API); TSO load + generation forecasts (CSV) | none |
| [ENTSO-E Transparency](https://transparency.entsoe.eu/) | Optional canonical fallback (skeleton ready) | token |
| [Open-Meteo](https://open-meteo.com/) | NWP forecasts for M5 | none |

All data CC-BY 4.0.

## What's next

Active development, building milestone-by-milestone with verification gates. Priority items:

- **NWP weather features** (Open-Meteo) — main remaining skill headroom; the model currently has no idea whether tomorrow is sunny.
- **Probabilistic outputs** — quantile heads (P10/P50/P90) with split-conformal calibration for guaranteed empirical coverage.
- **Production deployment** — FastAPI inference endpoint, GitHub Actions daily refresh, Streamlit dashboard with rolling skill-vs-TSO chart.


## License

MIT. Data: CC-BY 4.0 (SMARD / Bundesnetzagentur, ENTSO-E).
