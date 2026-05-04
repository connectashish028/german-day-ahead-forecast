# Day-Ahead Load Forecasting — Milestone Plan

Project goal: a TensorFlow day-ahead grid-load forecaster for Germany that
demonstrably and reproducibly beats the published TSO baseline on SMARD,
deployed as a daily-refreshing public service with calibrated uncertainty.

Each milestone has a **gate** — an automated check that must pass before
moving on. Gates make this a portfolio-grade pipeline rather than a notebook
collection: at any point in the timeline, the latest-passed milestone is a
working artifact you can show a recruiter.

## Headline so far

| Predictor | MAE (MW) | MAPE (%) | Skill |
|---|---|---|---|
| Seasonal-naive | 618.3 | 4.21 | −0.256 |
| **TSO baseline** | **492.4** | **3.36** | 0.000 |
| SARIMAX-on-residual | 441.7 | 3.00 | +0.103 |
| **Plain seq2seq LSTM** | **386.7** | **2.66** | **+0.215** |

(70 stratified delivery dates, 2025-01-01 → 2026-04-30, step 7 days.)

## Status

| # | Milestone | Status | Outcome |
|---|---|---|---|
| M0 | Repo hygiene | ✅ | `src/` layout, pinned deps, ruff + pytest, `.env`-gitignored |
| M1 | Backtest harness + TSO baseline | ✅ | Skill-score harness; TSO self-skill = 0 verified |
| M2 | Leakage-safe feature pipeline | ✅ | 17 leakage tests including corrupt-future on 9 stratified dates |
| M3 | Classical baselines (naive, SARIMAX) | ✅ | SARIMAX-on-residual hits **skill +0.103** |
| M3.5 | API-based data layer (unplanned but justified) | ✅ | Multi-source refresh: Energy-Charts + SMARD + ENTSO-E skeleton |
| M4 (part 1) | Plain seq2seq LSTM on residuals | ✅ | **Skill +0.215** — beats SARIMAX by 11pp |
| M4 (part 2) | Attention + interpretability | 📋 | Bahdanau attention; visualise attention map |
| M4 (part 3) | Feature-group ablation | 📋 | Attribute skill to calendar / lags / TSO fc / cross-border prices |
| M5 | NWP weather features (Open-Meteo) | 📋 | Population-weighted weather for 6 cities; gate +0.03 skill |
| M6 | Probabilistic outputs (P10/P50/P90) | 📋 | Quantile heads with pinball loss; coverage in [78%, 82%] |
| M7 | TFT / Conv-LSTM-Attention upgrade | 📋 | Modern architecture; gate +0.02 skill or document negative result |
| M8 | Split-conformal calibration | 📋 | 80% interval empirical coverage in [79%, 81%] |
| M9 | Ridge ensemble of model checkpoints | 📋 | If ensemble doesn't beat best single model, drop it |
| M10 | FastAPI inference service | 📋 | `POST /forecast` returns 96-step P10/P50/P90 in <30s |
| M11 | Daily forecast GitHub Action | 📋 | Cron runs at 13:00 CET, writes parquet to public storage |
| M12 | Streamlit dashboard | 📋 | Tomorrow's forecast + rolling 30-day skill chart |
| M13 | Weekly retrain GitHub Action | 📋 | Promote new model only if backtest skill ≥ current production |
| M14 | Drift monitoring (Evidently) | 📋 | Synthetic drift injection triggers an alert |
| M15 | README + portfolio polish | 📋 | Headline chart, 60-second pitch, peer + non-technical review |

## End-to-end "done" criteria

The project is portfolio-ready when all of these pass:

1. `pytest -q` — all tests green, including `test_no_leakage.py`
2. `python -m loadforecast.backtest --start 2025-01-01 --end 2026-04-30 --predictor lstm_attention` — prints **skill > 0.20** vs the TSO baseline
3. `python -m loadforecast.predict --issue-date <today>` — produces 96-step P10/P50/P90 in < 30s
4. Daily GitHub Action green for ≥ 7 consecutive days
5. Streamlit URL live; headline chart shows positive rolling 30-day skill
6. Calibrated 80% interval empirical coverage in [79%, 81%]
7. Weekly retrain has executed at least once and correctly chose whether to promote
