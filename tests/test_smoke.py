"""Smoke test — the package imports and pytest discovers tests."""

import loadforecast


def test_version() -> None:
    assert loadforecast.__version__ == "0.1.0"
