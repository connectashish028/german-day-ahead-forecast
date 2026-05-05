"""Probe SMARD's downloadcenter JSON endpoint to find the right module IDs
for the TSO forecast columns we currently get from manual CSVs.

What we want:
  - fc_cons__grid_load        (forecasted consumption: total)
  - fc_cons__residual_load    (forecasted consumption: residual)
  - fc_gen__photovoltaics_and_wind  (forecasted generation: PV + wind)
  - fc_gen__total             (forecasted generation: total)

The downloadcenter form POSTs to:
  https://www.smard.de/nip-download-manager/nip/download/market-data

Each request specifies a list of `moduleIds`. The mapping below is based on
SMARD's public docs / inspection of the live form. We probe each candidate
ID with a 1-day window and show what comes back so we can confirm before
wiring it into refresh.

Run:
  python scripts/probe_smard_downloadcenter.py
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

URL = "https://www.smard.de/nip-download-manager/nip/download/market-data"

# Candidate module IDs (best guess from SMARD downloadcenter form fields).
# We confirm them by checking the returned column header.
CANDIDATES = [
    # (module_id, expected_label, our_target_column)
    # Confirmed via browser DevTools capture (forecast sub-category 6):
    (6000411, "Forecasted consumption: total or residual?", "fc_cons__grid_load"),
    (6004362, "Forecasted consumption: total or residual?", "fc_cons__residual_load"),
    # Probe nearby IDs in the 6XXXXXX range for forecasted generation:
    (6000412, "near-neighbour probe",                       "fc_gen__?"),
    (6000413, "near-neighbour probe",                       "fc_gen__?"),
    (6004363, "near-neighbour probe",                       "fc_gen__?"),
    (6000414, "near-neighbour probe",                       "fc_gen__?"),
    (6000415, "near-neighbour probe",                       "fc_gen__?"),
]


def _ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def probe(module_id: int, start: datetime, end: datetime) -> tuple[bool, str]:
    payload = {
        "request_form": [
            {
                "format": "CSV",
                "moduleIds": [module_id],
                "region": "DE-LU",
                "timestamp_from": _ms(start),
                "timestamp_to": _ms(end),
                "type": "discrete",
                "language": "en",
                "resolution": "quarterhour",
            }
        ]
    }
    try:
        r = requests.post(URL, json=payload, timeout=30)
    except requests.RequestException as e:
        return False, f"network error: {e}"

    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"

    body = r.text
    if not body.strip():
        return False, "empty body"

    try:
        df = pd.read_csv(io.StringIO(body), sep=";", decimal=",", thousands=".", nrows=5)
    except Exception as e:
        return False, f"csv parse failed: {e!r}; first 200 chars: {body[:200]!r}"

    cols = list(df.columns)
    n_rows = len(df)
    return True, f"OK | first-{n_rows}-rows-loaded | columns: {cols}"


def main() -> None:
    end = datetime(2026, 4, 1)
    start = end - timedelta(days=1)
    print(f"Probing window: {start} -> {end} (UTC)\n")

    for mod_id, label, target in CANDIDATES:
        ok, msg = probe(mod_id, start, end)
        flag = "✓" if ok else "✗"
        print(f"{flag} module_id={mod_id:>10} | {label}")
        print(f"   target col: {target}")
        print(f"   {msg}\n")


if __name__ == "__main__":
    main()
