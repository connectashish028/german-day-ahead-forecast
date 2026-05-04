"""Probe Open-Meteo APIs for what we need.

Three relevant endpoints:
  /forecast            real-time forecasts (today onward) — production
  /archive             historical *actual* observations    — risky for backtest
  /historical-forecast historical *forecasts at issue time* — what we want
"""

from __future__ import annotations

import json

import requests

CITIES = {
    "berlin":     (52.52, 13.41),
    "hamburg":    (53.55,  9.99),
    "munich":     (48.14, 11.58),
    "cologne":    (50.94,  6.96),
    "frankfurt":  (50.11,  8.68),
    "stuttgart":  (48.78,  9.18),
}

VARS = [
    "temperature_2m",
    "shortwave_radiation",
    "wind_speed_100m",
    "cloud_cover",
]


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def probe_forecast() -> None:
    section("/forecast (live forecast — what M11 production would use)")
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": 52.52, "longitude": 13.41,
            "hourly": ",".join(VARS),
            "forecast_days": 2,
            "timezone": "UTC",
        },
        timeout=30,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        keys = list(d.keys())
        print(f"top-level keys: {keys}")
        h = d.get("hourly", {})
        print(f"hourly keys: {list(h.keys())[:6]}...")
        print(f"first 4 timestamps: {h['time'][:4]}")
        print(f"first 4 temps: {h['temperature_2m'][:4]}")
        print(f"resolution: hourly")


def probe_archive() -> None:
    section("/archive (actual historical observations)")
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 52.52, "longitude": 13.41,
            "start_date": "2024-06-15", "end_date": "2024-06-16",
            "hourly": ",".join(VARS),
            "timezone": "UTC",
        },
        timeout=30,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        h = d.get("hourly", {})
        print(f"got {len(h.get('time', []))} hourly observations")
        print(f"first temp: {h['temperature_2m'][:3]}")


def probe_historical_forecast() -> None:
    section("/historical-forecast (NWP forecasts AS THEY WERE ISSUED — what we want)")
    # For 2024-06-15 delivery, issue time would be 2024-06-14 12:00 Berlin = 10:00 UTC.
    # We want the forecast that was issued just before that, valid through delivery.
    r = requests.get(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params={
            "latitude": 52.52, "longitude": 13.41,
            "start_date": "2024-06-14", "end_date": "2024-06-15",
            "hourly": ",".join(VARS),
            "models": "best_match",
            "timezone": "UTC",
        },
        timeout=30,
    )
    print(f"HTTP {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        print(f"top-level keys: {list(d.keys())}")
        h = d.get("hourly", {})
        print(f"hourly variables: {list(h.keys())}")
        print(f"timestamps {len(h.get('time', []))}, span: {h['time'][0]} -> {h['time'][-1]}")
        print(f"temperature sample: {h['temperature_2m'][:5]}")
        print(f"shortwave radiation sample: {h['shortwave_radiation'][:5]}")
    else:
        print(f"BODY: {r.text[:400]}")


if __name__ == "__main__":
    probe_forecast()
    probe_archive()
    probe_historical_forecast()
