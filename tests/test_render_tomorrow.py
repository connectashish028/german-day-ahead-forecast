"""Smoke test for scripts/render_tomorrow.py.

The daily GitHub Action calls this after `data.refresh`. We want any
breakage (missing column, model API change, matplotlib regression) to
fail in CI rather than producing a stale or broken PNG.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PARQUET = Path("smard_merged_15min.parquet")
SCRIPT = Path("scripts/render_tomorrow.py")
OUT_PNG = Path("docs/images/tomorrow.png")

pytestmark = pytest.mark.skipif(
    not PARQUET.exists(),
    reason="smard_merged_15min.parquet not present; run loadforecast.data.refresh first",
)


def _tomorrow_data_available() -> bool:
    """Skip-precondition: TSO forecast must be in the parquet for tomorrow.
    Real-world state — early in the morning before SMARD's publication
    cycle, this is legitimately unavailable and the test should skip,
    not fail."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    df = pd.read_parquet(PARQUET, columns=["fc_cons__grid_load"])
    tomorrow = datetime.now(ZoneInfo("Europe/Berlin")).date() + timedelta(days=1)
    dec_start = pd.Timestamp(tomorrow.isoformat(), tz="Europe/Berlin").tz_convert("UTC")
    dec_end = dec_start + pd.Timedelta(days=1)
    window = df.loc[dec_start:dec_end, "fc_cons__grid_load"]
    return len(window) >= 96 and window.notna().all()


def test_render_tomorrow_writes_a_valid_png(tmp_path, monkeypatch):
    """Run the script and assert it produced a non-trivial PNG.

    Skips when tomorrow's TSO forecast hasn't been published yet — that's
    a legitimate real-world state, not a script bug.
    """
    if not _tomorrow_data_available():
        pytest.skip("tomorrow's TSO forecast not in parquet yet (publication lag)")

    target = tmp_path / "tomorrow.png"
    monkeypatch.setenv("MPLBACKEND", "Agg")

    spec = importlib.util.spec_from_file_location("_render_tomorrow", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT_PATH", target)

    mod.main()

    assert target.exists(), "render_tomorrow.py did not write the expected PNG"
    assert target.stat().st_size > 5_000, "PNG suspiciously small"
    with target.open("rb") as f:
        head = f.read(8)
    assert head == b"\x89PNG\r\n\x1a\n", "output is not a valid PNG"
