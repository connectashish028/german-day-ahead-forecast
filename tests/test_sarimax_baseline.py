"""Smoke tests for the SARIMAX-on-residual baseline.

Light tests: shape, no NaN, reasonable scale. Don't assert performance —
the *backtest* gate (M3) does that.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.backtest.sarimax_baseline import sarimax_residual_predict

PARQUET = Path("smard_merged_15min.parquet")
pytestmark = pytest.mark.skipif(
    not PARQUET.exists(), reason="SMARD parquet not present; run smard_client.py first."
)


@pytest.fixture(scope="module")
def df():
    return load_smard_15min(PARQUET)


def test_sarimax_returns_correct_shape(df):
    issue = issue_time_for(date(2025, 6, 15))
    pred = sarimax_residual_predict(df, issue)
    assert isinstance(pred, pd.Series)
    assert len(pred) == 96
    assert pred.index.tz is not None
    assert pred.notna().all(), "SARIMAX prediction must not contain NaN"


def test_sarimax_prediction_has_sane_scale(df):
    """Should be order-of-magnitude near actual German load (~50 GW)."""
    issue = issue_time_for(date(2025, 6, 15))
    pred = sarimax_residual_predict(df, issue)
    arr = pred.to_numpy()
    assert np.isfinite(arr).all()
    # SARIMAX residual + TSO can swing wide on bad fits — give it room.
    # German load floor is ~30 GW; predictions <5 GW or >100 GW indicate
    # a broken fit, but anything within is plausible from this baseline.
    assert arr.min() > 5_000, f"prediction min {arr.min()} suspiciously low"
    assert arr.max() < 100_000, f"prediction max {arr.max()} suspiciously high"


def test_sarimax_falls_back_when_history_missing(df):
    """If almost all pre-issue history is NaN, predictor must still return a valid
    96-step series (falling back to TSO forecast), not crash.
    """
    issue = issue_time_for(date(2025, 6, 15))
    blanked = df.copy()
    pre = blanked.index < issue
    blanked.loc[pre, "actual_cons__grid_load"] = float("nan")
    pred = sarimax_residual_predict(blanked, issue)
    assert len(pred) == 96
    assert pred.notna().all()
