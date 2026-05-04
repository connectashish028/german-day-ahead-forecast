"""Leakage tests for the feature pipeline (Milestone 2).

The single most important set of tests in the project. If a feature value
silently depends on data from after the issue time, every later evaluation
will look better than reality and the whole project's headline number is
worthless.

Two complementary approaches are used:

1. **Per-column rule check.** Every raw column is classified by
   `availability.classify_column` and the rule says when its values become
   knowable. We sample issue times across the dataset and verify the rule
   masks future values correctly.

2. **Corrupt-future test (the strong one).** Build features at issue time T.
   Then take the same DataFrame, scramble every actual value at index >= T
   (and every price value at index >= T - 12h), and rebuild features. The
   rebuilt features must match the originals bit-for-bit. If they don't,
   *some* future value is leaking into a feature.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.features import build_target_day_features
from loadforecast.features.availability import (
    ACTUAL_PREFIXES,
    PRICE_PREFIXES,
    RULES,
    classify_column,
)

PARQUET = Path("smard_merged_15min.parquet")
pytestmark = pytest.mark.skipif(
    not PARQUET.exists(), reason="SMARD parquet not present; run smard_client.py first."
)


@pytest.fixture(scope="module")
def df():
    return load_smard_15min(PARQUET)


# Sample issue dates spread across the dataset, deliberately including:
# - winter / summer (DST behaviour)
# - dates near holidays
# - dates close to dataset edges
SAMPLE_DATES = [
    date(2023, 1, 15),
    date(2023, 4, 9),    # Easter Sunday
    date(2023, 6, 20),
    date(2023, 10, 29),  # DST end
    date(2024, 3, 31),   # DST start
    date(2024, 5, 30),   # Corpus Christi (partial holiday)
    date(2024, 12, 25),  # Christmas
    date(2025, 7, 4),
    date(2026, 2, 14),
]


def test_classify_known_columns():
    assert classify_column("actual_cons__grid_load") == "actual"
    assert classify_column("actual_gen__wind_onshore") == "actual"
    assert classify_column("fc_cons__grid_load") == "forecast"
    assert classify_column("fc_gen__photovoltaics_and_wind") == "forecast"
    assert classify_column("price__germany_luxembourg") == "price"
    assert classify_column("price__france") == "price"
    assert classify_column("something_else") == "other"


def test_features_have_expected_shape(df):
    issue = issue_time_for(date(2024, 6, 15))
    feats = build_target_day_features(df, issue)
    assert feats.shape[0] == 96
    assert feats.index.tz is not None
    # No fully-NaN columns anywhere — that would silently mean a feature is broken
    all_nan = feats.columns[feats.isna().all()]
    assert len(all_nan) == 0, f"Fully-NaN feature columns: {list(all_nan)}"


@pytest.mark.parametrize("delivery", SAMPLE_DATES)
def test_corrupt_future_does_not_change_features(df, delivery):
    """The key test. Corrupting all post-issue actuals + post-(issue-12h) prices
    must not change any feature value built for the delivery day.
    """
    issue = issue_time_for(delivery)
    feats_clean = build_target_day_features(df, issue)

    # Build a corrupted copy.
    corrupt = df.copy()
    rng = np.random.default_rng(seed=0)
    for col in corrupt.columns:
        kind = classify_column(col)
        cutoff = issue + RULES[kind].max_age_offset
        mask = corrupt.index >= cutoff
        if mask.any():
            corrupt.loc[mask, col] = rng.normal(loc=1e9, scale=1e9, size=mask.sum())

    feats_corrupt = build_target_day_features(corrupt, issue)

    pd.testing.assert_frame_equal(
        feats_clean,
        feats_corrupt,
        check_exact=False,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("delivery", SAMPLE_DATES[:3])
def test_no_actual_value_at_or_after_issue_in_features(df, delivery):
    """Sanity: for any feature derived from an `actual_*` column, there should
    be no path that lets a value at timestamp >= issue_time enter the feature.
    We check this empirically: zero-out all pre-issue actuals and verify every
    actual-derived lag/rolling feature becomes NaN.
    """
    issue = issue_time_for(delivery)

    # Replace pre-issue actuals with NaN. Every lag/rolling feature derived from
    # them must therefore become NaN. If any such feature stays finite, it's
    # reading from a post-issue cell, which is a leak.
    blanked = df.copy()
    actual_cols = [c for c in blanked.columns if c.startswith(ACTUAL_PREFIXES)]
    pre_issue_mask = blanked.index < issue
    blanked.loc[pre_issue_mask, actual_cols] = float("nan")

    feats = build_target_day_features(blanked, issue)

    # Every load/residual lag and rolling feature must be all-NaN now.
    leak_suspects = [
        c
        for c in feats.columns
        if (c.startswith(("load__lag_", "residual__lag_", "load_roll_", "tso_residual_err__lag_")))
    ]
    assert leak_suspects, "expected at least one actual-derived feature to inspect"
    for col in leak_suspects:
        assert feats[col].isna().all(), f"Feature {col!r} stayed finite when pre-issue actuals were blanked — leak."


@pytest.mark.parametrize("delivery", SAMPLE_DATES[:3])
def test_no_post_minus12h_price_leakage(df, delivery):
    """Day-ahead prices for D-1 (issue date) clear at D-1 ~12:45, AFTER issue
    time. So no feature may depend on prices at timestamp >= issue - 12h.
    """
    issue = issue_time_for(delivery)
    cutoff = issue - pd.Timedelta(hours=12)

    blanked = df.copy()
    price_cols = [c for c in blanked.columns if c.startswith(PRICE_PREFIXES)]
    blanked.loc[blanked.index < cutoff, price_cols] = float("nan")

    feats = build_target_day_features(blanked, issue)

    price_features = [c for c in feats.columns if "price" in c.lower() or c.startswith("de_price")]
    assert price_features, "no price-derived features found"
    for col in price_features:
        assert feats[col].isna().all(), (
            f"Feature {col!r} stayed finite after blanking prices before {cutoff} — leak."
        )
