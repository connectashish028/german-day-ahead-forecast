"""Domain-rule clip for the extreme-tail regime (Price M10).

A pinball-loss P50 head structurally can't reach −500 EUR/MWh on rare
holiday × top-1%-VRE days: the conditional median given features stays
around the conditional median, no matter how many extreme examples we
add to training. Instead of forcing more training data through the head,
we apply a small post-processing shift at exactly the regime that breaks.

Trigger (computed from columns the price model already uses):
  - delivery_day is a federal holiday OR weekend
  - tomorrow's TSO VRE forecast exceeds the 90-day rolling q90 by
    `vre_pctile_trigger` (default 1.2 → top ~5 % of recent days)

Action: at the N quarter-hours with the lowest P50 (i.e. the predicted
trough), subtract Δ from P50 and `p10_multiplier × Δ` from P10. The
trough mask is data-driven, not time-of-day-fixed, so it tracks
whatever hours the model thinks will be cheapest tomorrow.

Calibrated by `scripts/calibrate_extreme_clip.py` on out-of-holdout days.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import holidays as hols
import numpy as np
import pandas as pd

from .price_dataset import TSO_VRE_FC


@dataclass(frozen=True)
class ClipConfig:
    delta_eur_mwh: float
    vre_pctile_trigger: float
    n_trough_qh: int
    p10_multiplier: float

    @classmethod
    def from_file(cls, path: Path) -> ClipConfig:
        d = json.loads(path.read_text())
        return cls(
            delta_eur_mwh=float(d["delta_eur_mwh"]),
            vre_pctile_trigger=float(d["vre_pctile_trigger"]),
            n_trough_qh=int(d["n_trough_qh"]),
            p10_multiplier=float(d.get("p10_multiplier", 1.5)),
        )


def load_clip_config(model_dir: Path | str) -> ClipConfig | None:
    """Load extreme_clip.json next to the model checkpoint, or None."""
    p = Path(model_dir) / "extreme_clip.json"
    return ClipConfig.from_file(p) if p.exists() else None


def is_calendar_extreme(delivery_date: date,
                        country_holidays: set | None = None) -> bool:
    """Is the delivery day a federal holiday or weekend?"""
    if country_holidays is None:
        country_holidays = hols.country_holidays(
            "DE", years=range(2022, 2030),
        )
    return delivery_date.weekday() in (5, 6) or delivery_date in country_holidays


def vre_percentile_at_issue(df: pd.DataFrame, issue_time: pd.Timestamp,
                            target_idx: pd.DatetimeIndex) -> float:
    """Recompute the same vre_percentile signal that lives at decoder
    feature index 4. Returns max across the delivery day."""
    ref_window = df[TSO_VRE_FC].loc[
        issue_time - pd.Timedelta(days=90): issue_time
    ].dropna()
    if len(ref_window) <= 100:
        return 0.0
    q90 = float(ref_window.quantile(0.90))
    if q90 <= 0:
        return 0.0
    vre_d = df[TSO_VRE_FC].reindex(target_idx).to_numpy()
    vre_d = np.nan_to_num(vre_d, nan=0.0)
    return float(np.max(vre_d / q90))


def should_clip(df: pd.DataFrame, issue_time: pd.Timestamp,
                target_idx: pd.DatetimeIndex,
                cfg: ClipConfig) -> bool:
    """True iff calendar trigger AND VRE percentile trigger both fire."""
    delivery_local = (issue_time.tz_convert("Europe/Berlin").normalize()
                       + pd.Timedelta(days=1)).date()
    if not is_calendar_extreme(delivery_local):
        return False
    return vre_percentile_at_issue(df, issue_time, target_idx) >= cfg.vre_pctile_trigger


def apply_clip(forecast: pd.DataFrame, cfg: ClipConfig) -> pd.DataFrame:
    """Shift P50 by −Δ and P10 by −p10_multiplier·Δ at the N lowest-P50
    quarter-hours. Returns a copy; does not mutate input.

    The trough mask is data-driven: we sort P50 ascending and clip the
    bottom N. This way the rule tracks whatever hours the model already
    flagged as cheapest, rather than hard-coding "11:00–15:00".
    """
    out = forecast.copy()
    p50 = out["p50"].to_numpy()
    trough_idx = np.argsort(p50)[: cfg.n_trough_qh]
    out.iloc[trough_idx, out.columns.get_loc("p50")] -= cfg.delta_eur_mwh
    out.iloc[trough_idx, out.columns.get_loc("p10")] -= cfg.delta_eur_mwh * cfg.p10_multiplier
    return out


__all__ = [
    "ClipConfig",
    "apply_clip",
    "is_calendar_extreme",
    "load_clip_config",
    "should_clip",
    "vre_percentile_at_issue",
]
