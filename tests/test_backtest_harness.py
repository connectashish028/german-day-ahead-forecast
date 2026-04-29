"""Backtest harness gate tests (Milestone 1).

Invariants the harness must satisfy before any modelling work:

1. Evaluating the TSO baseline against itself yields a skill score of exactly 0
   (MAE_model == MAE_baseline by construction).

2. The seasonal-naive predictor produces finite, distinct, sensibly-scaled
   numbers — confirming the harness computes two independent forecast errors
   from the same ground truth. (Note: empirically the SMARD-published TSO
   day-ahead forecast on stable months can be matched or beaten by a trivial
   week-ago lookup. That's a real finding about headroom in the baseline, not
   a harness bug.)

3. The per-step output has the expected schema and shape (96 quarter-hours per
   delivery day).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from loadforecast.backtest import (
    load_smard_15min,
    run_backtest,
    seasonal_naive_predict,
    tso_baseline_predict,
)

PARQUET = Path("smard_merged_15min.parquet")
pytestmark = pytest.mark.skipif(
    not PARQUET.exists(), reason="SMARD parquet not present; run smard_client.py first."
)

START = date(2024, 6, 1)
END = date(2024, 6, 30)


@pytest.fixture(scope="module")
def df():
    return load_smard_15min(PARQUET)


def test_tso_self_skill_is_zero(df) -> None:
    res = run_backtest(tso_baseline_predict, df, START, END, progress=False, label="tso_self")
    assert res.overall["n_days"] >= 25, "should evaluate ~30 days in June 2024"
    assert abs(res.overall["mae_model"] - res.overall["mae_tso"]) < 1e-9
    assert abs(res.overall["skill_score"]) < 1e-9


def test_seasonal_naive_distinct_and_sensible(df) -> None:
    res = run_backtest(seasonal_naive_predict, df, START, END, progress=False, label="naive")
    mae_model = res.overall["mae_model"]
    mae_tso = res.overall["mae_tso"]
    assert pd.notna(mae_model) and mae_model > 0
    assert pd.notna(mae_tso) and mae_tso > 0
    # The two predictors are genuinely different — harness isn't collapsing them.
    assert abs(mae_model - mae_tso) > 1.0
    # Sanity band: German grid load is ~50 GW; a sane MAE for either predictor
    # is in single-percent-of-load (~50–5000 MW). If we're wildly outside,
    # something is broken (units, alignment, etc.).
    assert 50 < mae_model < 5000
    # Skill score must be finite even when the model is worse than baseline.
    assert pd.notna(res.overall["skill_score"])


def test_per_step_shape(df) -> None:
    res = run_backtest(
        tso_baseline_predict,
        df,
        date(2024, 6, 1),
        date(2024, 6, 7),
        progress=False,
        label="shape",
    )
    # 7 days × 96 quarter-hours
    assert res.per_step.shape[0] == 7 * 96
    expected_cols = {"issue_date", "target_ts", "horizon_qh", "y_true", "y_model", "y_tso"}
    assert expected_cols.issubset(res.per_step.columns)
