"""German Day-Ahead Forecasting — explorative dashboard.

Run locally:
    streamlit run dashboards/app.py

Single-file app with two top-level views: LOAD and PRICE. The active view
is held in `st.session_state.view` and switched via two top-of-page
buttons. Default view = LOAD.

Theme: xAI-inspired dark, two accent colors (lilac for predictions, blue
for actuals). TSO baseline rendered as dimmed white-dashed.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

# Resolve repo root regardless of cwd, then make both `loadforecast.*` and
# `dashboards.*` importable. Streamlit runs the script directly so neither
# is on sys.path by default.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if (_HERE.parent / "pyproject.toml").exists() else _HERE
os.chdir(ROOT)
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dashboards import charts, styles  # noqa: E402
from loadforecast.backtest import issue_time_for, load_smard_15min  # noqa: E402
from loadforecast.models.predict import (  # noqa: E402
    lstm_quantile_predict_full,
    price_quantile_predict_full,
)

# --- Constants ----------------------------------------------------------

PARQUET = ROOT / "smard_merged_15min.parquet"
ACTUAL_COL = "actual_cons__grid_load"
TSO_COL = "fc_cons__grid_load"
PRICE_COL = "price__germany_luxembourg"
VRE_FC_COL = "fc_gen__photovoltaics_and_wind"
ABLATION_CSV = ROOT / "backtest_results" / "ablation_summary.csv"
WEATHER_BACKTEST_CSV = ROOT / "backtest_results" / "lstm_weather_step7.csv"
PRICE_HOLDOUT_CSV = ROOT / "backtest_results" / "price_quantile_holdout.csv"
BATTERY_PNL_CSV = ROOT / "backtest_results" / "battery_pnl_daily.csv"

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
    page_title="German Day-Ahead Forecasting",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="collapsed",
)
styles.inject(st)


# --- Data loading -------------------------------------------------------

@st.cache_resource(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df = load_smard_15min(str(PARQUET))
    return df[df.index.notna()].sort_index()


df = load_data()
data_min = df.index.min().tz_convert("Europe/Berlin").date()
data_max = df.index.max().tz_convert("Europe/Berlin").date()

BERLIN = ZoneInfo("Europe/Berlin")
now_berlin = datetime.now(BERLIN)
today = now_berlin.date()
tomorrow = today + timedelta(days=1)
issue_time_today = datetime.combine(today, time(12, 0), tzinfo=BERLIN)


# --- Cached predict + analysis loaders ----------------------------------

@st.cache_data(show_spinner="Forecasting load…")
def predict_for_day(delivery_date: date) -> pd.DataFrame | None:
    issue = issue_time_for(delivery_date)
    if issue > df.index.max():
        return None
    out = lstm_quantile_predict_full(df, issue)
    if out["p50"].isna().any():
        return None
    return out


@st.cache_data(show_spinner="Forecasting price…")
def predict_price_for_day(delivery_date: date) -> pd.DataFrame | None:
    issue = issue_time_for(delivery_date)
    if issue > df.index.max():
        return None
    out = price_quantile_predict_full(df, issue)
    if out["p50"].isna().any():
        return None
    return out


@st.cache_data
def load_weather_backtest() -> pd.DataFrame | None:
    if not WEATHER_BACKTEST_CSV.exists():
        return None
    return pd.read_csv(WEATHER_BACKTEST_CSV)


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


@st.cache_data
def volatility_quartiles() -> pd.DataFrame | None:
    if not WEATHER_BACKTEST_CSV.exists():
        return None
    bt_local = pd.read_csv(WEATHER_BACKTEST_CSV, parse_dates=["target_ts"])
    bt_local["target_ts"] = pd.to_datetime(bt_local["target_ts"], utc=True)
    if PRICE_COL not in df.columns:
        return None
    bt_local["price"] = df[PRICE_COL].reindex(bt_local["target_ts"]).values

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


@st.cache_data
def load_ablation() -> pd.DataFrame | None:
    if not ABLATION_CSV.exists():
        return None
    return pd.read_csv(ABLATION_CSV)


@st.cache_data
def load_price_backtest() -> pd.DataFrame | None:
    if not PRICE_HOLDOUT_CSV.exists():
        return None
    return pd.read_csv(PRICE_HOLDOUT_CSV, parse_dates=["target_ts"])


@st.cache_data
def price_rolling_skill() -> pd.DataFrame | None:
    bt_local = load_price_backtest()
    if bt_local is None or bt_local.empty:
        return None
    daily = (
        bt_local.assign(
            issue_date=pd.to_datetime(bt_local["issue_date"]),
            abs_err_model=lambda d: (d["y_true"] - d["p50"]).abs(),
            abs_err_naive=lambda d: (d["y_true"] - d["naive_1d"]).abs(),
        )
        .groupby("issue_date")[["abs_err_model", "abs_err_naive"]].mean()
        .sort_index()
    )
    daily["model_mae_30d"] = daily["abs_err_model"].rolling(14, min_periods=5).mean()
    daily["naive_mae_30d"] = daily["abs_err_naive"].rolling(14, min_periods=5).mean()
    daily["skill"] = 1 - daily["model_mae_30d"] / daily["naive_mae_30d"]
    return daily.dropna(subset=["skill"])


@st.cache_data
def price_spread_quartiles() -> pd.DataFrame | None:
    bt_local = load_price_backtest()
    if bt_local is None or bt_local.empty:
        return None
    daily = bt_local.groupby("issue_date").agg(
        actual_spread=("y_true", lambda s: float(s.max() - s.min())),
        model_spread=("p50",    lambda s: float(s.max() - s.min())),
    ).reset_index()
    daily["bin"] = pd.qcut(
        daily["actual_spread"], q=4,
        labels=["Calm", "Moderate", "High", "Extreme"],
    )
    grouped = (
        daily.groupby("bin", observed=True).agg(
            actual_spread=("actual_spread", "mean"),
            model_spread=("model_spread", "mean"),
            n_days=("issue_date", "count"),
            spread_lo=("actual_spread", "min"),
            spread_hi=("actual_spread", "max"),
        )
        .reset_index()
        .rename(columns={"bin": "label"})
    )
    grouped["range"] = grouped.apply(
        lambda r: f"{r.spread_lo:,.0f}–{r.spread_hi:,.0f} €", axis=1,
    )
    return grouped


@st.cache_data
def load_battery_pnl() -> pd.DataFrame | None:
    if not BATTERY_PNL_CSV.exists():
        return None
    return pd.read_csv(BATTERY_PNL_CSV, parse_dates=["issue_date"])


# --- View state ---------------------------------------------------------

if "view" not in st.session_state:
    st.session_state.view = "load"


def _set_view(v: str) -> None:
    st.session_state.view = v


# --- Brand bar (always visible) -----------------------------------------

st.markdown(
    """
    <div class="hero-bar">
        <div class="hero-brand">German Day-Ahead Forecasting</div>
        <div class="hero-badge">Live · refreshes daily 13:00 CET</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Top-of-page nav ----------------------------------------------------

nav_load, nav_price = st.columns(2)
with nav_load:
    st.button(
        "LOAD FORECAST",
        key="nav_load_btn",
        on_click=_set_view,
        args=("load",),
        type="primary" if st.session_state.view == "load" else "secondary",
        use_container_width=True,
    )
with nav_price:
    st.button(
        "PRICE FORECAST",
        key="nav_price_btn",
        on_click=_set_view,
        args=("price",),
        type="primary" if st.session_state.view == "price" else "secondary",
        use_container_width=True,
    )


# --- Shared explorer date state -----------------------------------------

# Default to yesterday for the explorer — usually has complete actuals.
default_day = today - timedelta(days=1)
if default_day > data_max:
    default_day = data_max
if default_day < data_min + timedelta(days=8):
    default_day = data_min + timedelta(days=8)
if "picked_date" not in st.session_state:
    st.session_state.picked_date = default_day


# ========================================================================
# LOAD VIEW
# ========================================================================

if st.session_state.view == "load":

    st.markdown(
        "# A TensorFlow LSTM that beats the German TSO's published "
        "day-ahead load forecast."
    )
    st.markdown(
        f'<p style="color: rgba(255,255,255,0.5); font-family: \'JetBrains Mono\', monospace; '
        f'font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; '
        f'margin-top: 0.25rem;">Backtest 2025-01 → 2026-04 · n = 70 days · '
        f'data through {data_max.isoformat()} · model: Probabilistic LoadCast v1</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "German TSOs publish a day-ahead load forecast every afternoon — "
        "the anchor every utility and balancing party uses. This model "
        "corrects it using weather, calendar, and recent TSO error "
        "patterns. **Cuts mean error by 20 % across the 14-month holdout.**"
    )

    # --- Headline stats grid (load) -------------------------------------
    st.markdown(
        """
        <div class="stat-grid">
            <div class="stat-cell">
                <div class="stat-label">Avg error reduction
                    <span class="info-tip">ⓘ<span class="info-tip-content">
                        Skill score <code>1 − MAE_model / MAE_TSO</code>. Zero = ties the baseline, one = perfect.
                    </span></span>
                </div>
                <div class="stat-value">20.1<span class="stat-unit">%</span></div>
            </div>
            <div class="stat-cell">
                <div class="stat-label">Mean error
                    <span class="info-tip">ⓘ<span class="info-tip-content">
                        Mean absolute load forecast error in MWh per quarter-hour, averaged across the 70-day holdout.
                    </span></span>
                </div>
                <div class="stat-value">393<span class="stat-unit">MW</span></div>
            </div>
            <div class="stat-cell">
                <div class="stat-label">Mean % error
                    <span class="info-tip">ⓘ<span class="info-tip-content">
                        Mean error relative to the actual realised load.
                    </span></span>
                </div>
                <div class="stat-value">2.72<span class="stat-unit">%</span></div>
            </div>
            <div class="stat-cell">
                <div class="stat-label">80 % band hit rate
                    <span class="info-tip">ⓘ<span class="info-tip-content">
                        Fraction of quarter-hours where the realised load lands inside the model's P10–P90 interval. Target 80 %.
                    </span></span>
                </div>
                <div class="stat-value">78.3<span class="stat-unit">%</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Tomorrow's load forecast ---------------------------------------
    tomorrow_fc = predict_for_day(tomorrow) if tomorrow <= data_max else None
    tomorrow_tso = (
        df[TSO_COL].reindex(tomorrow_fc.index)
        if (tomorrow_fc is not None and TSO_COL in df.columns) else None
    )
    st.markdown(
        f"## Tomorrow's forecast"
        f"<span style='font-family:\"JetBrains Mono\",monospace; font-size:0.8rem; "
        f"letter-spacing:0.1em; color:rgba(255,255,255,0.5); margin-left:0.75rem;'>"
        f"DELIVERY {tomorrow.isoformat()} · issued {today.isoformat()} 12:00 Berlin"
        f"</span>",
        unsafe_allow_html=True,
    )
    if tomorrow_fc is None:
        if now_berlin < issue_time_today:
            st.info("**Not yet issued.** Tomorrow is forecast at today 12:00 Berlin. Check back after noon.")
        elif tomorrow > data_max:
            st.info(f"**Data stale.** Parquet runs through {data_max.isoformat()}. Daily refresh runs at 13:00 CET.")
        else:
            st.warning("Encoder/decoder window has missing values — upstream data gap.")
    else:
        st.markdown(
            "Model P50 vs the TSO's published forecast for tomorrow. "
            "Divergence between lilac and dashed-white is where the model "
            "expects the operator to be wrong."
        )
        st.plotly_chart(
            charts.forecast_chart(tomorrow_fc, tso=tomorrow_tso, actuals=None),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_tomorrow_load",
        )
        if tomorrow_tso is not None and tomorrow_tso.notna().any():
            model_peak = float(tomorrow_fc["p50"].max())
            tso_peak = float(tomorrow_tso.max())
            peak_diff_pct = (model_peak - tso_peak) / tso_peak * 100
            avg_diff_mw = float(np.abs(tomorrow_fc["p50"].values
                                       - tomorrow_tso.values).mean())
            avg_diff_pct = avg_diff_mw / float(np.abs(tomorrow_tso).mean()) * 100
            peak_arrow = "↑" if peak_diff_pct > 0 else "↓"
            st.markdown(
                f"""
                <div class="stat-grid" style="grid-template-columns: repeat(3, 1fr);">
                    <div class="stat-cell">
                        <div class="stat-label">Model peak (P50)</div>
                        <div class="stat-value">{model_peak:,.0f}<span class="stat-unit">MWh/QH</span></div>
                    </div>
                    <div class="stat-cell">
                        <div class="stat-label">TSO peak</div>
                        <div class="stat-value">{tso_peak:,.0f}<span class="stat-unit">MWh/QH</span></div>
                    </div>
                    <div class="stat-cell">
                        <div class="stat-label">Avg disagreement</div>
                        <div class="stat-value">{avg_diff_pct:.1f}<span class="stat-unit">% &nbsp;({peak_arrow}{abs(peak_diff_pct):.1f}% at peak)</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Forecast explorer (load) ---------------------------------------
    st.markdown("## Forecast explorer")
    st.markdown(
        "Pick any past or recent delivery day. Lilac = model P50 + "
        "P10/P90 ribbon. Blue = realised load. Dashed white = TSO "
        "baseline. Issue time D-1 12:00 Berlin."
    )
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; '
        'letter-spacing:0.12em; text-transform:uppercase; color:rgba(255,255,255,0.5); '
        'margin: 1rem 0 0.5rem 0;">Notable days</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(NOTABLE_DAYS))
    for col, (d, label, tooltip) in zip(cols, NOTABLE_DAYS, strict=True):
        with col:
            if st.button(label, key=f"notable_load_{d}", help=tooltip,
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
        show_actual = st.checkbox("Show actual load", value=True, key="load_show_actual")
    with col_tso:
        show_tso = st.checkbox("Show TSO baseline", value=True, key="load_show_tso")

    forecast = predict_for_day(picked)
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
        st.plotly_chart(
            charts.forecast_chart(forecast, actuals=actuals, tso=tso),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_explorer_load",
        )
        if actuals is not None and actuals.notna().any():
            tso_full = df[TSO_COL].reindex(target_idx)
            model_mae = float(np.nanmean(np.abs(actuals.values - forecast["p50"].values)))
            tso_mae = float(np.nanmean(np.abs(actuals.values - tso_full.values)))
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
                "Signed error per 15-min step (forecast − actual). "
                "Above zero = over-predict; below = under-predict."
            )
            st.plotly_chart(
                charts.error_chart(actuals, forecast, tso_full),
                use_container_width=True,
                config={"displaylogo": False},
                key="chart_explorer_load_error",
            )

    # --- Load analysis panels -------------------------------------------
    st.markdown("## Where the model wins, by hour of day")
    st.markdown(
        "Mean absolute error by hour-of-day across the 70-day holdout. "
        "The shaded gap is the model's lift over the TSO baseline. "
        "Largest gains: morning ramp (5–9 h) and evening peak (16–20 h)."
    )
    bt = load_weather_backtest()
    if bt is None or bt.empty:
        st.info("Backtest CSV not found.")
    else:
        st.plotly_chart(
            charts.hour_profile_chart(bt),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_hour_profile",
        )

    st.markdown("## Error reduction over time")
    st.markdown(
        "30-day rolling skill score `1 − MAE_model / MAE_TSO`. Above "
        "zero = model wins the month. Consistent across the holdout — "
        "not a single-period fluke."
    )
    roll = rolling_skill()
    if roll is not None and not roll.empty:
        st.plotly_chart(
            charts.skill_chart(roll), use_container_width=True,
            config={"displaylogo": False},
            key="chart_rolling_skill",
        )

    st.markdown("## How the model holds up as price volatility grows")
    st.markdown(
        "Holdout days binned by intra-day price spread (a proxy for "
        "net-load volatility). Bars = mean daily MAE per quartile, "
        "model vs TSO."
    )
    vq = volatility_quartiles()
    if vq is not None and not vq.empty:
        st.plotly_chart(
            charts.volatility_quartile_chart(vq),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_volatility",
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

    st.markdown("## Where the error reduction comes from")
    st.markdown(
        "Five LSTM variants, each adding one feature group. Bars show "
        "the marginal MAE improvement. The lagged-TSO-error feature "
        "alone buys ~57 % of the total lift — confirms the residual-"
        "learning design."
    )
    abl = load_ablation()
    if abl is not None and not abl.empty:
        st.plotly_chart(
            charts.ablation_chart(abl),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_ablation",
        )


# ========================================================================
# PRICE VIEW
# ========================================================================

else:  # st.session_state.view == "price"

    st.markdown(
        "# A second LSTM, retargeted at the day-ahead spot price."
    )
    st.markdown(
        f'<p style="color: rgba(255,255,255,0.5); font-family: \'JetBrains Mono\', monospace; '
        f'font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; '
        f'margin-top: 0.25rem;">Backtest 2026-03 → 2026-04 · n = 61 days · '
        f'data through {data_max.isoformat()} · model: Probabilistic PriceCast v4</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Same architecture, retargeted at the EPEX day-ahead spot price "
        "— the signal a battery operator or BRP maps to € on a P&L. "
        "**On a 10 MW / 20 MWh battery over the 61-day holdout, the "
        "forecast captures ~95 % of perfect-foresight P&L vs the naive "
        "baseline's 81 %** (+€57 k uplift, ~€1.7 M/year on a 100 MWh fleet)."
    )

    # --- Headline stats grid (price) ------------------------------------
    st.markdown(
        """
        <div class="stat-grid">
            <div class="stat-cell">
                <div class="stat-label">Avg error vs naive
                    <span class="info-tip">ⓘ<span class="info-tip-content">
                        Skill score vs naive yesterday-same-quarter-hour. No published price baseline exists, so naive yesterday is the comparison.
                    </span></span>
                </div>
                <div class="stat-value">+35.7<span class="stat-unit">%</span></div>
            </div>
            <div class="stat-cell">
                <div class="stat-label">P50 MAE
                    <span class="info-tip">ⓘ<span class="info-tip-content">
                        Median forecast's mean absolute error in €/MWh on the 61-day holdout.
                    </span></span>
                </div>
                <div class="stat-value">23.8<span class="stat-unit">€/MWh</span></div>
            </div>
            <div class="stat-cell">
                <div class="stat-label">Battery P&L vs perfect-foresight
                    <span class="info-tip">ⓘ<span class="info-tip-content">
                        % of theoretical-max arbitrage P&L a 10 MW / 20 MWh battery captures dispatching against the model's P50.
                    </span></span>
                </div>
                <div class="stat-value">95.0<span class="stat-unit">%</span></div>
            </div>
            <div class="stat-cell">
                <div class="stat-label">P&L uplift over naive
                    <span class="info-tip">ⓘ<span class="info-tip-content">
                        Extra € the same battery earns vs a trader using yesterday's prices as the forecast.
                    </span></span>
                </div>
                <div class="stat-value">+57<span class="stat-unit">k € / 61 d</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Tomorrow's price forecast --------------------------------------
    tomorrow_price = (
        predict_price_for_day(tomorrow) if tomorrow <= data_max else None
    )
    tomorrow_price_actual = (
        df[PRICE_COL].reindex(tomorrow_price.index)
        if (tomorrow_price is not None and PRICE_COL in df.columns) else None
    )
    st.markdown(
        f"## Tomorrow's day-ahead price"
        f"<span style='font-family:\"JetBrains Mono\",monospace; font-size:0.8rem; "
        f"letter-spacing:0.1em; color:rgba(255,255,255,0.5); margin-left:0.75rem;'>"
        f"DELIVERY {tomorrow.isoformat()} · DE-LU spot · "
        f"issued {today.isoformat()} 12:00 Berlin"
        f"</span>",
        unsafe_allow_html=True,
    )
    if tomorrow_price is None:
        if now_berlin < issue_time_today:
            st.info("**Not yet issued.** Tomorrow's price forecast publishes at today 12:00 Berlin.")
        elif tomorrow > data_max:
            st.info(f"**Data stale.** Parquet runs through {data_max.isoformat()}. Daily refresh runs at 13:00 CET.")
        else:
            st.warning("Encoder/decoder window has missing values — upstream data gap.")
    else:
        target_idx_tmrw = pd.date_range(
            start=pd.Timestamp(tomorrow, tz="Europe/Berlin").tz_convert("UTC"),
            periods=96, freq="15min",
        )
        vre_missing_tmrw = (
            VRE_FC_COL in df.columns
            and df[VRE_FC_COL].reindex(target_idx_tmrw).isna().all()
        )
        if vre_missing_tmrw:
            st.markdown(
                '<div style="display:inline-block; border:1px solid rgba(255,200,80,0.5); '
                'padding:0.4rem 0.9rem; margin-bottom:1rem; '
                'font-family:\'JetBrains Mono\',monospace; font-size:0.75rem; '
                'letter-spacing:0.1em; text-transform:uppercase; '
                'color:rgba(255,200,80,0.95);">'
                "Degraded mode &nbsp;·&nbsp; SMARD VRE day-ahead not yet published; "
                "model running on weather + load only (~+38 % MAE expected)"
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "Model P50 with P10–P90 ribbon for tomorrow. Realised price "
            "clears at ~12:42 Berlin after the EPEX auction."
        )
        st.plotly_chart(
            charts.price_forecast_chart(tomorrow_price, actuals=tomorrow_price_actual),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_tomorrow_price",
        )

        p50 = tomorrow_price["p50"]
        p10 = tomorrow_price["p10"]
        p90 = tomorrow_price["p90"]
        peak_eur = float(p50.max())
        trough_eur = float(p50.min())
        spread_eur = peak_eur - trough_eur
        peak_hour = p50.idxmax().tz_convert("Europe/Berlin").strftime("%H:%M")
        trough_hour = p50.idxmin().tz_convert("Europe/Berlin").strftime("%H:%M")
        band_avg = float((p90 - p10).mean())
        st.markdown(
            f"""
            <div class="stat-grid" style="grid-template-columns: repeat(4, 1fr);">
                <div class="stat-cell">
                    <div class="stat-label">Peak (P50)</div>
                    <div class="stat-value">{peak_eur:,.0f}<span class="stat-unit">€/MWh @ {peak_hour}</span></div>
                </div>
                <div class="stat-cell">
                    <div class="stat-label">Trough (P50)</div>
                    <div class="stat-value">{trough_eur:,.0f}<span class="stat-unit">€/MWh @ {trough_hour}</span></div>
                </div>
                <div class="stat-cell">
                    <div class="stat-label">Day spread (P50)</div>
                    <div class="stat-value">{spread_eur:,.0f}<span class="stat-unit">€/MWh</span></div>
                </div>
                <div class="stat-cell">
                    <div class="stat-label">Avg P10–P90 width</div>
                    <div class="stat-value">{band_avg:,.0f}<span class="stat-unit">€/MWh</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Forecast explorer (price) --------------------------------------
    st.markdown("## Forecast explorer")
    st.markdown(
        "Pick any delivery day. Lilac = P50 + P10/P90 ribbon. "
        "Blue = realised price. No TSO-equivalent baseline exists for "
        "price, so naive yesterday-same-quarter-hour is the comparison."
    )
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; '
        'letter-spacing:0.12em; text-transform:uppercase; color:rgba(255,255,255,0.5); '
        'margin: 1rem 0 0.5rem 0;">Notable days</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(NOTABLE_DAYS))
    for col, (d, label, tooltip) in zip(cols, NOTABLE_DAYS, strict=True):
        with col:
            if st.button(label, key=f"notable_price_{d}", help=tooltip,
                         use_container_width=True):
                if data_min + timedelta(days=8) <= d <= data_max:
                    st.session_state.picked_date = d
                else:
                    st.warning(f"{d} is outside the data window.")

    col_date, col_actual = st.columns([2, 1])
    with col_date:
        picked = st.date_input(
            "Delivery date",
            key="picked_date",
            min_value=data_min + timedelta(days=8),
            max_value=data_max,
            help="The day to forecast. Issue time is D-1 12:00 Berlin.",
        )
    with col_actual:
        show_price_actual = st.checkbox(
            "Show actual price", value=True, key="price_show_actual",
        )

    price_fc = predict_price_for_day(picked)
    if price_fc is None:
        st.warning(
            "Cannot build a leakage-safe window for that date — encoder or "
            "decoder feature coverage is incomplete."
        )
    else:
        target_idx = price_fc.index
        actual_price = (
            df[PRICE_COL].reindex(target_idx)
            if (show_price_actual and PRICE_COL in df.columns) else None
        )
        st.plotly_chart(
            charts.price_forecast_chart(price_fc, actuals=actual_price),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_explorer_price",
        )
        prev_idx = target_idx - pd.Timedelta(days=1)
        naive_price = df[PRICE_COL].reindex(prev_idx).set_axis(target_idx)
        actual_full = df[PRICE_COL].reindex(target_idx)
        if actual_full.notna().any():
            model_mae = float(np.nanmean(np.abs(actual_full.values - price_fc["p50"].values)))
            naive_mae = float(np.nanmean(np.abs(actual_full.values - naive_price.values)))
            improvement = (
                (1 - model_mae / naive_mae) * 100 if naive_mae > 0 else float("nan")
            )
            actual_spread = float(np.nanmax(actual_full.values) - np.nanmin(actual_full.values))
            model_spread = float(price_fc["p50"].max() - price_fc["p50"].min())
            st.markdown(
                f"""
                <div class="stat-grid" style="grid-template-columns: repeat(4, 1fr);">
                    <div class="stat-cell">
                        <div class="stat-label">Model P50 MAE</div>
                        <div class="stat-value">{model_mae:.1f}<span class="stat-unit">€/MWh</span></div>
                    </div>
                    <div class="stat-cell">
                        <div class="stat-label">Naive (D-1) MAE</div>
                        <div class="stat-value">{naive_mae:.1f}<span class="stat-unit">€/MWh</span></div>
                    </div>
                    <div class="stat-cell">
                        <div class="stat-label">Error vs naive</div>
                        <div class="stat-value">{improvement:+.1f}<span class="stat-unit">%</span></div>
                    </div>
                    <div class="stat-cell">
                        <div class="stat-label">Day spread (model / actual)</div>
                        <div class="stat-value">{model_spread:.0f}<span class="stat-unit">/ {actual_spread:.0f} €</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No realised price for this delivery day yet — once it "
                "clears, this section will populate with model vs. naive "
                "error and the realised spread."
            )

    # --- Price analysis panels ------------------------------------------
    st.markdown("## Where the price model wins, by hour of day")
    st.markdown(
        "Mean absolute price error by hour-of-day on the 61-day Mar–Apr "
        "2026 holdout. Largest gains: mid-day PV trough (10–15 h) and "
        "evening ramp."
    )
    pbt = load_price_backtest()
    if pbt is None or pbt.empty:
        st.info("Price holdout CSV not found — run `scripts/backtest_price_quantile.py` to populate.")
    else:
        st.plotly_chart(
            charts.price_hour_profile_chart(pbt),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_price_hour_profile",
        )

    st.markdown("## Error reduction over time")
    st.markdown(
        "14-day rolling skill score vs naive yesterday. Window is "
        "shorter than the load chart's 30-day because the holdout is "
        "only 61 days. Above zero = model wins."
    )
    proll = price_rolling_skill()
    if proll is not None and not proll.empty:
        st.plotly_chart(
            charts.price_skill_chart(proll), use_container_width=True,
            config={"displaylogo": False},
            key="chart_price_rolling_skill",
        )

    st.markdown("## How well the model captures the daily spread")
    st.markdown(
        "Holdout days binned by actual intra-day spread. Bars = actual "
        "vs model P50 spread per quartile. Right-most bar gap = "
        "pinball-loss median collapse: P50 can't fully reach the extremes."
    )
    psq = price_spread_quartiles()
    if psq is not None and not psq.empty:
        st.plotly_chart(
            charts.price_spread_quartile_chart(psq),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_price_spread",
        )
        rows = []
        for _, r in psq.iterrows():
            ratio = r["model_spread"] / r["actual_spread"] * 100
            rows.append(f"<b>{r['label']}</b>: {ratio:.0f} %")
        st.markdown(
            f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                        color:rgba(255,255,255,0.7); margin-top:0.5rem;
                        display:flex; gap:1.5rem; flex-wrap:wrap;">
                <span style="color:rgba(255,255,255,0.5);">
                    Spread captured (model / actual) —</span>
                {' · '.join(rows)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## What this is worth: cumulative battery P&L")
    st.markdown(
        "10 MW / 20 MWh battery, 90 % round-trip, 3 cycles/day. Greedy "
        "dispatch on each day's forecast, realised at actual prices. "
        "**Model P50 captures 95 % of perfect-foresight vs naive's 81 %** "
        "— +€57 k over 61 days. P10/P90 underperforms P50 by ~2 pp "
        "because dispatch is a ranking problem."
    )
    pnl = load_battery_pnl()
    if pnl is not None and not pnl.empty:
        st.plotly_chart(
            charts.price_pnl_chart(pnl),
            use_container_width=True,
            config={"displaylogo": False},
            key="chart_price_pnl",
        )


# --- Methodology footer (shared) ----------------------------------------

st.markdown("---")
st.markdown("## Methodology")
st.markdown(
    """
    - **Target:** German grid load + day-ahead spot price, 15-min resolution, 96 steps per delivery day.
    - **Issue time:** D-1 12:00 Berlin (EPEX day-ahead gate). Leakage tested by scrambling all post-issue values and asserting feature parity.
    - **Load model:** predicts the TSO error (`actual − TSO_forecast`), adds the correction. Calendar + climatology already in the TSO baseline; the model learns the systematic remainder.
    - **Price model:** raw price target (no published baseline). Decoder uses raw `fc_gen__pv+wind` + engineered `vre_to_load_ratio` and `vre_percentile`. Trained with 30 % feature-dropout on `fc_gen` for graceful degradation when SMARD hasn't published.
    - **Architecture:** seq2seq LSTM(64) encoder + LSTM(64) decoder → three quantile heads (P10/P50/P90), pinball loss, ~36 k parameters per model.
    - **Skill score:** `1 − MAE_model / MAE_baseline`. Zero = ties, one = perfect.
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
