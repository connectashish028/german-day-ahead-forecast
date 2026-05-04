"""Leakage tests for the LSTM windowing.

Same logic as `test_no_leakage.py` but applied to the seq2seq windows:
build a window at issue time T, scramble every post-issue value in the
underlying frame (via the M2 availability rules), rebuild the window,
and assert that the encoder/decoder arrays are bit-for-bit identical.

The *target* y_resid is allowed to differ — it deliberately uses
post-issue ground truth (only available in training).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.features.availability import RULES, classify_column
from loadforecast.models.dataset import build_window

PARQUET = Path("smard_merged_15min.parquet")
pytestmark = pytest.mark.skipif(
    not PARQUET.exists(), reason="SMARD parquet not present."
)


@pytest.fixture(scope="module")
def df():
    return load_smard_15min(PARQUET)


SAMPLE_DATES = [
    date(2023, 6, 20),
    date(2024, 3, 31),  # DST start
    date(2024, 12, 25),  # holiday
    date(2025, 7, 4),
]


def test_window_has_expected_shape(df):
    issue = issue_time_for(date(2024, 6, 15))
    w = build_window(df, issue)
    assert w.X_enc.shape == (672, 6)
    assert w.X_dec.shape == (96, 6)
    assert w.y_resid.shape == (96,)


@pytest.mark.parametrize("delivery", SAMPLE_DATES)
def test_corrupt_future_does_not_change_encoder_or_decoder(df, delivery):
    """Encoder and decoder arrays must be bit-for-bit identical after
    scrambling every post-issue value in the source frame.
    """
    issue = issue_time_for(delivery)
    clean = build_window(df, issue)

    corrupt_df = df.copy()
    rng = np.random.default_rng(seed=0)
    for col in corrupt_df.columns:
        kind = classify_column(col)
        cutoff = issue + RULES[kind].max_age_offset
        mask = corrupt_df.index >= cutoff
        if mask.any():
            corrupt_df.loc[mask, col] = rng.normal(loc=1e9, scale=1e9, size=mask.sum())

    corrupt = build_window(corrupt_df, issue)

    np.testing.assert_array_equal(clean.X_enc, corrupt.X_enc)
    np.testing.assert_array_equal(clean.X_dec, corrupt.X_dec)
    # y_resid is allowed to differ — it's training-time-only ground truth.


def test_encoder_window_has_no_post_issue_data(df):
    issue = issue_time_for(date(2024, 6, 15))
    # Encoder index is internal to build_window — we test indirectly:
    # if we blank all *pre-issue* actuals, the encoder load column
    # must become entirely NaN, while the decoder TSO column stays finite
    # (TSO fc for delivery day is available at issue time).
    blanked = df.copy()
    pre_issue = blanked.index < issue
    blanked.loc[pre_issue, "actual_cons__grid_load"] = np.nan

    blanked_w = build_window(blanked, issue)
    # Encoder column 0 is `load` — must now be entirely NaN.
    assert np.isnan(blanked_w.X_enc[:, 0]).all(), "load encoder feature reads post-issue actuals"
    # Decoder TSO column 0 should stay finite.
    assert np.isfinite(blanked_w.X_dec[:, 0]).all(), "TSO forecast was unexpectedly NaN at delivery day"


def test_dataset_drops_incomplete_windows(df):
    from loadforecast.models.dataset import build_dataset

    # Pick one valid date and one impossibly early date that has no history.
    issue_good = issue_time_for(date(2024, 6, 15))
    issue_bad = pd.Timestamp("2022-01-02 11:00", tz="UTC")  # < 7d of data behind it
    Xe, Xd, Y, kept = build_dataset(df, [issue_bad, issue_good])
    assert len(kept) == 1
    assert kept[0] == issue_good
    assert Xe.shape == (1, 672, 6)
