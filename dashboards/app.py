"""German Day-Ahead Load Forecast — explorative dashboard.

Run locally:
    streamlit run dashboards/app.py

Single scrollable page, xAI-inspired dark theme. Two accent colors only:
- lilac (#B8A1FF) — model prediction
- blue  (#3B82F6) — actual realised load (when known)
- TSO baseline rendered as dimmed white-dashed.

Sections, top → bottom:
1. Hero result — headline numbers, plain-English framing.
2. Forecast explorer — date picker + notable-days quick-pick + chart +
   per-day stats + signed-error chart.
3. Hour-of-day error profile — where the model wins, by hour.
4. Error reduction over time — 30-day rolling vs TSO.
5. Volatility quartiles — how the model holds up as price spread grows.
6. Feature ablation — where the error reduction comes from.
7. Methodology — short, links to repo and data source.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Resolve repo root regardless of cwd, then make both `loadforecast.*` and
# `dashboards.*` importable. Streamlit runs the script directly so neither
# is on sys.path by default.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "pyproject.toml").exists() else _HERE
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loadforecast.backtest import issue_time_for, load_smard_15min  # noqa: E402
from loadforecast.models.predict import lstm_quantile_predict_full  # noqa: E402
from dashboards import charts, styles  # noqa: E402

PARQUET = ROOT / "smard_merged_15min.parquet"
ACTUAL_COL = "actual_cons__grid_load"
TSO_COL = "fc_cons__grid_load"
ABLATION_CSV = ROOT / "backtest_results" / "ablation_summary.csv"
WEATHER_BACKTEST_CSV = ROOT / "backtest_results" / "lstm_weather_step7.csv"

# Curated case-study dates — the most narrative-rich days in the holdout.
NOTABLE_DAYS = [
    (date(2026, 5, 1),
     "1 May 2026 — PV record",
     "Federal holiday + 54 GW PV peak; prices crashed to −500 €/MWh. "
     "TSO badly under-forecast; weather model cut error in half."),
    (date(2025, 12, 25),
     "Christmas Day 2025",
     "Holiday demand pattern — TSO baselines tend to over-forecast "
     "industrial load on bank holidays."),
    (date(2025, 8, 12),
     "Heatwave 12 Aug 2025",
     "Peak summer demand; cooling load spikes that the climatological "
     "TSO baseline doesn't anticipate."),
]


# --- Page setup ---------------------------------------------------------

st.set_page_config(
    page_title="German Load Forecast",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="collapsed",
)
styles.inject(st)


# --- Data loading -------------------------------------------------------

@st.cache_resource(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    return load_smard_15min(str(PARQUET))


df = load_data()
data_min = df.index.min().tz_convert("Europe/Berlin").date()
data_max = df.index.max().tz_convert("Europe/Berlin").date()


# --- Hero bar -----------------------------------------------------------

st.markdown(
    f"""
    <div class="hero-bar">
        <div class="hero-brand">German Day-Ahead Load Forecast</div>
        <div class="hero-badge">25 % lower error than TSO</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "# A TensorFlow LSTM that beats the German TSO's published "
    "day-ahead load forecast."
)
st.markdown(
    f'<p style="color: rgba(255,255,255,0.5); font-family: \'JetBrains Mono\', monospace; '
    f'font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; '
    f'margin-top: 0.25rem;">Backtest 2025-01 → 2026-04 · n = 70 days · '
    f'data through {data_max.isoformat()} · model: lstm_quantile_v1</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    The German grid operator publishes a day-ahead forecast every afternoon
    for the following day's national electricity demand — that's the number
    every utility, trader, and balancing party anchors on. This model uses
    weather, calendar, and recent-error patterns to predict where the
    operator's forecast will be wrong, and corrects it. **Across the
    14-month holdout window, it cuts the average forecast error by 25 %.**
    """
)


# --- Headline stats grid ------------------------------------------------

st.markdown(
    """
    <div class="stat-grid">
        <div class="stat-cell">
            <div class="stat-label">Avg error reduction</div>
            <div class="stat-value">25.3<span class="stat-unit">%</span></div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Mean error</div>
            <div class="stat-value">368<span class="stat-unit">MW</span></div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Mean % error</div>
            <div class="stat-value">2.56<span class="stat-unit">%</span></div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">80 % band hit rate</div>
            <div class="stat-value">78.3<span class="stat-unit">%</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Forecast explorer --------------------------------------------------

st.markdown("## Forecast explorer")
st.markdown(
    "Pick any delivery day in the available range — or jump straight to "
    "one of the marquee case-study days below. Lilac line is the model's "
    "median forecast with the P10/P90 ribbon; blue is the realised load "
    "(when the day is past); dashed white is the TSO's published "
    "forecast. Issue time is D-1 12:00 Europe/Berlin."
)

default_day = min(data_max - timedelta(days=1), date(2026, 4, 30))
if "picked_date" not in st.session_state:
    st.session_state.picked_date = default_day

st.markdown(
    '<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; '
    'letter-spacing:0.12em; text-transform:uppercase; color:rgba(255,255,255,0.5); '
    'margin: 1rem 0 0.5rem 0;">Notable days</div>',
    unsafe_allow_html=True,
)
cols = st.columns(len(NOTABLE_DAYS))
for col, (d, label, tooltip) in zip(cols, NOTABLE_DAYS, strict=True):
    with col:
        if st.button(label, key=f"notable_{d}", help=tooltip,
                     use_container_width=True):
            if data_min + timedelta(days=8) <= d <= data_max:
                st.session_state.picked_date = d
            else:
                st.warning(f"{d} is outside the data window.")

col_date, col_actual, col_tso = st.columns([2, 1, 1])
with col_date:
    picked = st.date_input(
        "Delivery date",
        key="picked_date",
        min_value=data_min + timedelta(days=8),
        max_value=data_max,
        help="The day to forecast. Issue time is D-1 12:00 Berlin.",
    )
with col_actual:
    show_actual = st.checkbox("Show actual load", value=True)
with col_tso:
    show_tso = st.checkbox("Show TSO baseline", value=True)


@st.cache_data(show_spinner="Running model…")
def predict_day(delivery_date: date) -> pd.DataFrame | None:
    issue = issue_time_for(delivery_date)
    if issue > df.index.max():
        return None
    out = lstm_quantile_predict_full(df, issue)
    if out["p50"].isna().any():
        return None
    return out


forecast = predict_day(picked)

if forecast is None:
    st.warning(
        "Cannot build a leakage-safe window for that date — try a date with "
        "fuller feature coverage."
    )
else:
    target_idx = forecast.index
    actuals = (
        df[ACTUAL_COL].reindex(target_idx)
        if (show_actual and ACTUAL_COL in df.columns) else None
    )
    tso = (
        df[TSO_COL].reindex(target_idx)
        if (show_tso and TSO_COL in df.columns) else None
    )

    fig = charts.forecast_chart(forecast, actuals=actuals, tso=tso)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    if actuals is not None and actuals.notna().any():
        tso_full = df[TSO_COL].reindex(target_idx)
        model_mae = float(np.abs(actuals.values - forecast["p50"].values).mean())
        tso_mae = float(np.abs(actuals.values - tso_full.values).mean())
        improvement = (
            (1 - model_mae / tso_mae) * 100 if tso_mae > 0 else float("nan")
        )
        st.markdown(
            f"""
            <div class="stat-grid" style="grid-template-columns: repeat(3, 1fr);">
                <div class="stat-cell">
                    <div class="stat-label">Model error (this day)</div>
                    <div class="stat-value">{model_mae:.0f}<span class="stat-unit">MWh/QH</span></div>
                </div>
                <div class="stat-cell">
                    <div class="stat-label">TSO error (this day)</div>
                    <div class="stat-value">{tso_mae:.0f}<span class="stat-unit">MWh/QH</span></div>
                </div>
                <div class="stat-cell">
                    <div class="stat-label">Error reduction</div>
                    <div class="stat-value">{improvement:+.1f}<span class="stat-unit">%</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Forecast error over the day")
        st.markdown(
            "Signed error at each 15-minute step: **forecast minus actual**. "
            "Bars above zero mean the forecast over-predicts demand; below "
            "zero, under-predicts. The closer to the dotted line, the better."
        )
        st.plotly_chart(
            charts.error_chart(actuals, forecast, tso_full),
            use_container_width=True,
            config={"displaylogo": False},
        )


# --- Hour-of-day error profile ------------------------------------------

st.markdown("## Where the model wins, by hour of day")
st.markdown(
    "Average forecast error at each hour, aggregated across the entire "
    "70-day holdout. The dashed white line is the TSO baseline; the lilac "
    "line is the model. The shaded gap below the TSO line is the lift the "
    "model provides — concentrated in the morning ramp (5–9 h) and the "
    "afternoon peak (16–20 h), where the TSO baseline systematically "
    "mis-predicts."
)


@st.cache_data
def load_weather_backtest() -> pd.DataFrame | None:
    if not WEATHER_BACKTEST_CSV.exists():
        return None
    return pd.read_csv(WEATHER_BACKTEST_CSV)


bt = load_weather_backtest()
if bt is None or bt.empty:
    st.info(
        "Backtest CSV not found — run "
        "`python -m loadforecast.backtest --predictor lstm_weather "
        "--start 2025-01-01 --end 2026-04-30 --step-days 7 "
        "--out backtest_results/lstm_weather_step7.csv` to populate."
    )
else:
    st.plotly_chart(
        charts.hour_profile_chart(bt),
        use_container_width=True,
        config={"displaylogo": False},
    )


# --- Rolling skill chart ------------------------------------------------

st.markdown("## Error reduction over time")
st.markdown(
    "30-day rolling improvement of the model's average error vs. the "
    "TSO's published forecast. Above zero means the model is doing "
    "better that month. The lift is consistent across the holdout — "
    "this isn't a single-period fluke."
)


@st.cache_data
def rolling_skill() -> pd.DataFrame | None:
    if not WEATHER_BACKTEST_CSV.exists():
        return None
    df_bt = pd.read_csv(WEATHER_BACKTEST_CSV, parse_dates=["target_ts"])
    daily = (
        df_bt.assign(
            issue_date=pd.to_datetime(df_bt["issue_date"]),
            abs_err_model=lambda d: (d["y_true"] - d["y_model"]).abs(),
            abs_err_tso=lambda d: (d["y_true"] - d["y_tso"]).abs(),
        )
        .groupby("issue_date")[["abs_err_model", "abs_err_tso"]].mean()
        .sort_index()
    )
    daily["model_mae_30d"] = daily["abs_err_model"].rolling(30, min_periods=5).mean()
    daily["tso_mae_30d"] = daily["abs_err_tso"].rolling(30, min_periods=5).mean()
    daily["skill"] = 1 - daily["model_mae_30d"] / daily["tso_mae_30d"]
    return daily.dropna(subset=["skill"])


roll = rolling_skill()
if roll is not None and not roll.empty:
    st.plotly_chart(
        charts.skill_chart(roll), use_container_width=True,
        config={"displaylogo": False},
    )


# --- Volatility quartile bars -------------------------------------------

st.markdown("## How the model holds up as price volatility grows")
st.markdown(
    "Each delivery day is binned by its **intra-day price spread** "
    "(`max − min` of the day-ahead price within the day) — a proxy for "
    "how unusual that day's net-load shape was. Calm days are flat; "
    "extreme days have a wide spread, often driven by mid-day "
    "renewables gluts or scarcity peaks. Bars show the **mean daily "
    "forecast error** for each quartile."
)


@st.cache_data
def volatility_quartiles() -> pd.DataFrame | None:
    if not WEATHER_BACKTEST_CSV.exists():
        return None
    bt_local = pd.read_csv(WEATHER_BACKTEST_CSV, parse_dates=["target_ts"])
    bt_local["target_ts"] = pd.to_datetime(bt_local["target_ts"], utc=True)

    price_col = "price__germany_luxembourg"
    if price_col not in df.columns:
        return None
    bt_local["price"] = df[price_col].reindex(bt_local["target_ts"]).values

    daily = bt_local.groupby("issue_date").agg(
        mae_model=("y_true",
                   lambda s: float(np.abs(
                       s - bt_local.loc[s.index, "y_model"]).mean())),
        mae_tso=("y_true",
                 lambda s: float(np.abs(
                     s - bt_local.loc[s.index, "y_tso"]).mean())),
        price_max=("price", "max"),
        price_min=("price", "min"),
    ).reset_index()
    daily["price_spread"] = daily["price_max"] - daily["price_min"]
    daily = daily.dropna(subset=["price_spread"])
    if daily.empty:
        return None

    daily["bin"] = pd.qcut(
        daily["price_spread"], q=4,
        labels=["Calm", "Moderate", "High", "Extreme"],
    )
    grouped = (
        daily.groupby("bin", observed=True).agg(
            mae_model=("mae_model", "mean"),
            mae_tso=("mae_tso", "mean"),
            n_days=("issue_date", "count"),
            spread_lo=("price_spread", "min"),
            spread_hi=("price_spread", "max"),
        )
        .reset_index()
        .rename(columns={"bin": "label"})
    )
    grouped["range"] = grouped.apply(
        lambda r: f"{r.spread_lo:,.0f}–{r.spread_hi:,.0f} €", axis=1,
    )
    return grouped


vq = volatility_quartiles()
if vq is not None and not vq.empty:
    st.plotly_chart(
        charts.volatility_quartile_chart(vq),
        use_container_width=True,
        config={"displaylogo": False},
    )

    rows = []
    for _, r in vq.iterrows():
        improvement = (1 - r["mae_model"] / r["mae_tso"]) * 100
        rows.append(f"<b>{r['label']}</b>: {improvement:+.1f} %")
    st.markdown(
        f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                    color:rgba(255,255,255,0.7); margin-top:0.5rem;
                    display:flex; gap:1.5rem; flex-wrap:wrap;">
            <span style="color:rgba(255,255,255,0.5);">
                Error reduction by quartile —</span>
            {' · '.join(rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Feature ablation ---------------------------------------------------

st.markdown("## Where the error reduction comes from")
st.markdown(
    "Five LSTM variants, each adding one feature group on top of the "
    "previous, scored on the same 70-day holdout. The bars show the "
    "**marginal improvement** each group buys. Showing the model the "
    "recent `actual − TSO` error pattern alone accounts for roughly 57 % "
    "of the total improvement — a clean confirmation of the residual-"
    "learning design choice."
)


@st.cache_data
def load_ablation() -> pd.DataFrame | None:
    if not ABLATION_CSV.exists():
        return None
    return pd.read_csv(ABLATION_CSV)


abl = load_ablation()
if abl is not None and not abl.empty:
    st.plotly_chart(
        charts.ablation_chart(abl),
        use_container_width=True,
        config={"displaylogo": False},
    )


# --- Methodology footer --------------------------------------------------

st.markdown("---")
st.markdown("## Methodology")
st.markdown(
    """
    - **What we forecast:** German national grid load at 15-min resolution
      — 96 steps covering the full delivery day.
    - **When we forecast:** every prediction uses only data available by
      **D-1 12:00 Europe/Berlin** (the German day-ahead market closure). A
      corrupt-future leakage test scrambles every post-issue value and
      asserts the features come out bit-identical.
    - **How we beat the baseline:** instead of forecasting load directly,
      the model predicts the TSO's *error* — `actual − TSO_forecast` — and
      adds the correction. Most of the easy structure (calendar,
      climatology) is already in the TSO baseline; the model only has to
      learn the systematic remainder.
    - **Architecture:** sequence-to-sequence LSTM(64) encoder + LSTM(64)
      decoder → three quantile heads (P10/P50/P90) trained with pinball
      loss. ~36 k parameters. Trained on 2022–2024, validated on 2025-H1,
      held out on a 70-date stratified 2025-H2 + 2026-Q1 set.
    - **Features:** 7 days of load history, lagged forecast error,
      calendar (hour/day-of-week/federal holiday), and 4 numerical-weather
      variables (temperature, solar radiation, wind at 100 m, cloud cover)
      population-weighted across 6 German load centres.
    - **The "25 %" number:** in forecasting, this is conventionally
      reported as a *skill score*: `1 − MAE_model / MAE_TSO`. Score 0 = ties
      the baseline, score 1 = perfect. We land at +0.253 on the holdout.
    """
)
st.markdown(
    f"""
    <div style="display: flex; gap: 1rem; margin-top: 1.5rem;
                font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
                letter-spacing: 0.1em; text-transform: uppercase;">
        <a href="https://github.com/connectashish028/german-load-forecast"
           style="border: 1px solid {styles.BORDER_STRONG}; padding: 0.5rem 1rem;">
            GitHub repo →</a>
        <a href="https://www.smard.de/"
           style="border: 1px solid {styles.BORDER_STRONG}; padding: 0.5rem 1rem;">
            Data: SMARD →</a>
    </div>
    """,
    unsafe_allow_html=True,
)
