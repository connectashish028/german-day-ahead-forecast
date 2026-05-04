"""Quick check of the German holiday pipeline."""
from __future__ import annotations
import pandas as pd
import holidays as hols
from loadforecast.features.calendar import (
    BUNDESLAND_POPULATION,
    population_weighted_holiday_fraction,
    is_federal_holiday,
    is_bridge_day,
)


def list_year(year: int) -> None:
    print(f"\n=== Public holidays in Germany {year} ===")
    fed = hols.country_holidays("DE", years=year)  # federal subset
    rows = []
    # Build a per-Bundesland set of holiday dates
    by_state = {code: hols.country_holidays("DE", subdiv=code, years=year) for code in BUNDESLAND_POPULATION}

    # Union across all states
    all_dates = sorted({d for cal in by_state.values() for d in cal})
    total = sum(BUNDESLAND_POPULATION.values())
    for d in all_dates:
        observers = [c for c in BUNDESLAND_POPULATION if d in by_state[c]]
        weight = sum(BUNDESLAND_POPULATION[c] for c in observers) / total
        is_federal = d in fed
        rows.append((d, fed.get(d) or by_state[observers[0]].get(d), len(observers), weight, is_federal))

    print(f"{'Date':<12} {'Name':<40} {'States':>7} {'PopFrac':>8} {'Federal':>8}")
    for d, name, n, w, federal in rows:
        print(f"{d!s:<12} {name:<40} {n:>7d} {w*100:>7.1f}% {'YES' if federal else 'no':>8}")

    # Bridge days for the year
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz="Europe/Berlin").tz_convert("UTC")
    br = is_bridge_day(idx)
    bd = idx[br][~br[~br]].tz_convert("Europe/Berlin").date if False else [
        d.date() for d, flag in zip(idx.tz_convert("Europe/Berlin"), br.values) if flag
    ]
    print(f"\nBridge days {year}: {bd}")

    # A few sanity checks
    federal_count = sum(1 for r in rows if r[4])
    print(f"\nFederal holidays counted: {federal_count}  (expected 9 — Jan1, GoodFri, EasterMon, May1, Ascension, WhitMon, Oct3, Christmas, BoxingDay)")

    pop_check = population_weighted_holiday_fraction(idx).groupby(idx.tz_convert("Europe/Berlin").date).first()
    print(f"Days where pop-weighted hol fraction > 0: {(pop_check > 0).sum()}")
    print(f"Days at 100% (federal):                   {(pop_check == 1.0).sum()}")


if __name__ == "__main__":
    for y in (2024, 2025, 2026):
        list_year(y)
