"""Probe energy-charts.info API surface for what we need."""
from __future__ import annotations
import json
import requests
import pandas as pd

BASE = "https://api.energy-charts.info"

print("=== 1. Day-ahead prices: 1 May 2026 (the negative-price event) ===")
r = requests.get(f"{BASE}/price", params={"bzn": "DE-LU", "start": "2026-05-01", "end": "2026-05-02"}, timeout=30)
d = r.json()
ts = pd.to_datetime(d["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin")
prices = pd.Series(d["price"], index=ts)
print(f"Coverage: {ts.min()} -> {ts.max()}, {len(ts)} obs")
print(f"Min: {prices.min():.2f} EUR/MWh at {prices.idxmin()}")
print(f"Number of negative QHs: {(prices < 0).sum()}")
print(f"Number below -400: {(prices < -400).sum()}")
print(f"Hours <= -200 EUR/MWh:")
for tt, p in prices[prices <= -200].items():
    print(f"   {tt}  {p:.2f}")

print("\n=== 2. Bidding zones supported ===")
# Try a handful of known ENTSO-E zone codes
zones = ["DE-LU", "FR", "NL", "AT", "BE", "CH", "CZ", "PL", "DK1", "DK2", "NO2", "SE4", "IT-NORTH", "SI", "HU", "DE-AT-LU"]
for z in zones:
    r = requests.get(f"{BASE}/price", params={"bzn": z, "start": "2024-06-01", "end": "2024-06-02"}, timeout=15)
    if r.status_code == 200:
        d = r.json()
        n = len(d.get("price", []))
        print(f"   {z:<12s}  {n:>4d} obs")
    else:
        print(f"   {z:<12s}  HTTP {r.status_code}")

print("\n=== 3. Other endpoints we might need ===")
# Test public_power (generation), total_power, load, load_forecast etc.
for ep, params in [
    ("public_power", {"country": "de", "start": "2024-06-15", "end": "2024-06-16"}),
    ("total_power",  {"country": "de", "start": "2024-06-15", "end": "2024-06-16"}),
    ("load",         {"country": "de", "start": "2024-06-15", "end": "2024-06-16"}),
    ("load_forecast",{"country": "de", "start": "2024-06-15", "end": "2024-06-16"}),
    ("ren_share_forecast", {"country": "de", "start": "2024-06-15", "end": "2024-06-16"}),
    ("public_power_forecast", {"country": "de", "start": "2024-06-15", "end": "2024-06-16", "production_type": "solar"}),
    ("solar_wind_forecast", {"country": "de", "start": "2024-06-15", "end": "2024-06-16"}),
]:
    r = requests.get(f"{BASE}/{ep}", params=params, timeout=20)
    summary = ""
    if r.status_code == 200:
        try:
            j = r.json()
            keys = list(j.keys()) if isinstance(j, dict) else "list"
            summary = f"OK keys={keys}"
        except Exception:
            summary = "OK (non-JSON)"
    else:
        summary = f"HTTP {r.status_code}"
    print(f"   /{ep:<22s} {summary}")
