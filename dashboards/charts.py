"""Plotly chart helpers — all charts share the xAI-style dark theme.

Two accent colors only: lilac for prediction, blue for actual. TSO baseline
is rendered as a dimmed white-dashed line so it stays visually subordinate.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .styles import (
    ACTUAL,
    BG,
    BORDER,
    PREDICTION,
    PREDICTION_FILL,
    TEXT,
    TEXT_30,
    TEXT_50,
    TEXT_70,
    TSO,
)

_AXIS = dict(
    showgrid=True, gridcolor=BORDER, gridwidth=1,
    zeroline=False, color=TEXT_70,
    tickfont=dict(family="JetBrains Mono, monospace", size=11, color=TEXT_70),
    title_font=dict(family="JetBrains Mono, monospace", size=11, color=TEXT_50),
)


def _base_layout(title: str | None = None, height: int = 420) -> dict:
    layout = dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color=TEXT),
        margin=dict(l=50, r=20, t=40 if title else 10, b=50),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(family="JetBrains Mono, monospace", size=10,
                      color=TEXT_70),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=_AXIS, yaxis=_AXIS,
        hoverlabel=dict(
            bgcolor="#2a2d35", bordercolor=BORDER,
            font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
        ),
    )
    if title:
        layout["title"] = dict(
            text=title, font=dict(size=14, color=TEXT_70),
            x=0, xanchor="left",
        )
    return layout


def forecast_chart(
    forecast: pd.DataFrame,
    actuals: pd.Series | None = None,
    tso: pd.Series | None = None,
    title: str | None = None,
) -> go.Figure:
    """Quantile-aware forecast chart.

    `forecast` must have columns p10, p50, p90 indexed by tz-aware timestamp.
    `actuals` (optional) is the realised load on that day.
    `tso` (optional) is the TSO baseline forecast.
    """
    fig = go.Figure()

    # P10 / P90 ribbon — pure visual; skipped from hover so the unified
    # tooltip only shows the lines users actually care about. Give them
    # explicit names to prevent Plotly's `tonexty` fill from injecting
    # an "undefined" legend entry.
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast["p90"], mode="lines",
        line=dict(color=PREDICTION, width=0.5, dash="dot"),
        name="p90_band", legendgroup="band",
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast["p10"], mode="lines",
        line=dict(color=PREDICTION, width=0.5, dash="dot"),
        fill="tonexty", fillcolor=PREDICTION_FILL,
        name="p10_band", legendgroup="band",
        hoverinfo="skip", showlegend=False,
    ))
    # Median (the headline line)
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast["p50"], mode="lines",
        line=dict(color=PREDICTION, width=2.2),
        name="Model forecast",
        hovertemplate="%{y:,.0f}<extra>Model</extra>",
    ))

    if tso is not None:
        fig.add_trace(go.Scatter(
            x=tso.index, y=tso.values, mode="lines",
            line=dict(color=TSO, width=1.4, dash="dash"),
            name="TSO baseline",
            hovertemplate="%{y:,.0f}<extra>TSO</extra>",
        ))

    if actuals is not None and actuals.notna().any():
        fig.add_trace(go.Scatter(
            x=actuals.index, y=actuals.values, mode="lines",
            line=dict(color=ACTUAL, width=2.2),
            name="Actual load",
            hovertemplate="%{y:,.0f}<extra>Actual</extra>",
        ))

    layout = _base_layout(title=title, height=440)
    layout["yaxis"] = {**_AXIS, "title": "Load (MWh / 15-min)"}
    layout["xaxis"] = {**_AXIS, "title": "Time (UTC)"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def skill_chart(rolling: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Rolling MAE skill score over time.

    `rolling` must have columns model_mae, tso_mae, skill indexed by date.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_30, width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling["skill"] * 100, mode="lines",
        line=dict(color=PREDICTION, width=2.2),
        fill="tozeroy", fillcolor=PREDICTION_FILL,
        name="30-day rolling improvement",
        hovertemplate="%{x|%Y-%m-%d}<br>error reduction: %{y:+.1f}%%<extra></extra>",
    ))
    layout = _base_layout(title=title, height=360)
    layout["yaxis"] = {**_AXIS, "tickformat": "+.0f",
                       "ticksuffix": " %",
                       "title": "Error reduction vs TSO (30-day rolling)"}
    layout["xaxis"] = {**_AXIS, "title": "Delivery date"}
    fig.update_layout(**layout)
    return fig


def error_chart(
    actuals: pd.Series,
    forecast: pd.DataFrame,
    tso: pd.Series,
    title: str | None = None,
) -> go.Figure:
    """Per-step forecast error — shows where the model out-performs TSO.

    Two traces: model error (lilac) and TSO error (dashed white). Errors
    are signed (positive = forecast too high), so under/over-forecasting
    is visible at a glance.
    """
    model_err = forecast["p50"].values - actuals.values
    tso_err = tso.values - actuals.values

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_30, width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=actuals.index, y=tso_err, mode="lines",
        line=dict(color=TSO, width=1.4, dash="dash"),
        name="TSO error",
        hovertemplate="%{y:+,.0f}<extra>TSO error</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=actuals.index, y=model_err, mode="lines",
        line=dict(color=PREDICTION, width=2.0),
        name="Model error",
        hovertemplate="%{y:+,.0f}<extra>Model error</extra>",
    ))

    layout = _base_layout(title=title, height=240)
    layout["yaxis"] = {**_AXIS, "title": "Error (forecast − actual)",
                       "tickformat": "+.0f"}
    layout["xaxis"] = {**_AXIS, "title": "Time (UTC)"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


__all__ = ["forecast_chart", "skill_chart", "error_chart"]
