"""Calendar features tailored to the German electricity load.

Beyond plain weekday/hour, German consumption has measurable structure from:
- Federal public holidays (16 states agree)
- Per-Bundesland public holidays (e.g. Corpus Christi only in 6/16 states)
- Bridge days — a Friday after a Thursday holiday is not legally a holiday
  but consumption drops substantially (the "Brueckentag effect").
- DST transitions — Sunday with 23 or 25 hours.

We compute a population-weighted "national holiday fraction" rather than a
boolean, since a holiday in NRW (~22% of population) bites consumption far
more than a holiday in Saarland (~1%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as hols
except ImportError as e:  # pragma: no cover
    raise ImportError("`holidays` package required. Install via `pip install holidays`.") from e


# Bundesland populations (millions, 2023 estimates from Destatis). Used as
# weights for the national-holiday-fraction feature.
BUNDESLAND_POPULATION = {
    "NW": 17.93,  # Nordrhein-Westfalen
    "BY": 13.18,  # Bayern
    "BW": 11.28,  # Baden-Württemberg
    "NI": 8.00,   # Niedersachsen
    "HE": 6.39,   # Hessen
    "RP": 4.16,   # Rheinland-Pfalz
    "SN": 4.07,   # Sachsen
    "BE": 3.76,   # Berlin
    "SH": 2.95,   # Schleswig-Holstein
    "BB": 2.57,   # Brandenburg
    "ST": 2.16,   # Sachsen-Anhalt
    "TH": 2.10,   # Thüringen
    "HH": 1.89,   # Hamburg
    "MV": 1.61,   # Mecklenburg-Vorpommern
    "SL": 0.99,   # Saarland
    "HB": 0.68,   # Bremen
}
TOTAL_POP = sum(BUNDESLAND_POPULATION.values())


def _build_holiday_calendars(years: range) -> dict[str, set]:
    """One holidays.HolidayBase per Bundesland, evaluated for the given years."""
    return {code: hols.country_holidays("DE", subdiv=code, years=years) for code in BUNDESLAND_POPULATION}


def population_weighted_holiday_fraction(idx: pd.DatetimeIndex) -> pd.Series:
    """Return a Series in [0, 1] giving the population-weighted fraction of Germany
    that observes a public holiday on each date in `idx`.

    `idx` may be sub-daily; the value is computed per calendar date in
    Europe/Berlin and broadcast to all timestamps falling on that date.
    """
    if idx.tz is None:
        raise ValueError("idx must be tz-aware")

    local_dates = pd.Series(idx.tz_convert("Europe/Berlin").date, index=idx)
    unique_dates = sorted(set(local_dates))
    if not unique_dates:
        return pd.Series(0.0, index=idx, name="hol_pop_frac")

    years = range(unique_dates[0].year, unique_dates[-1].year + 1)
    calendars = _build_holiday_calendars(years)

    frac_by_date: dict = {}
    for d in unique_dates:
        weight = sum(pop for code, pop in BUNDESLAND_POPULATION.items() if d in calendars[code])
        frac_by_date[d] = weight / TOTAL_POP

    return local_dates.map(frac_by_date).astype(float).rename("hol_pop_frac")


def is_federal_holiday(idx: pd.DatetimeIndex) -> pd.Series:
    """Boolean: is the date a federal-level public holiday (observed in all 16 states)?"""
    if idx.tz is None:
        raise ValueError("idx must be tz-aware")
    local_dates = pd.Series(idx.tz_convert("Europe/Berlin").date, index=idx)
    unique_dates = sorted(set(local_dates))
    if not unique_dates:
        return pd.Series(False, index=idx, name="is_federal_holiday")
    years = range(unique_dates[0].year, unique_dates[-1].year + 1)
    calendars = _build_holiday_calendars(years)
    federal_dates = {d for d in unique_dates if all(d in calendars[c] for c in BUNDESLAND_POPULATION)}
    return local_dates.isin(federal_dates).rename("is_federal_holiday")


def is_bridge_day(idx: pd.DatetimeIndex) -> pd.Series:
    """A 'bridge day' (Brueckentag) is a working day sandwiched between a public
    holiday and a weekend — Mondays after a Sunday-on-holiday don't count, but a
    Friday after a Thursday federal holiday does. We define it strictly as:

        weekday in {Mon, Fri} AND not itself a federal holiday AND
        (the adjacent Tuesday/Thursday is a federal holiday).
    """
    if idx.tz is None:
        raise ValueError("idx must be tz-aware")
    local_dates = pd.Series(idx.tz_convert("Europe/Berlin").date, index=idx)
    unique_dates = sorted(set(local_dates))
    if not unique_dates:
        return pd.Series(False, index=idx, name="is_bridge_day")
    years = range(unique_dates[0].year - 1, unique_dates[-1].year + 2)
    de = hols.country_holidays("DE", years=years)  # federal-level subset

    bridge_dates = set()
    for d in unique_dates:
        wd = d.weekday()  # Mon=0, Sun=6
        if d in de:
            continue
        if wd == 0 and (d + pd.Timedelta(days=-3).to_pytimedelta()) in de:
            # Monday after a Friday federal holiday — usually irrelevant but include for symmetry
            bridge_dates.add(d)
        if wd == 4 and (d + pd.Timedelta(days=3).to_pytimedelta()) in de:
            # Friday before a Monday federal holiday
            bridge_dates.add(d)
        if wd == 0 and (d + pd.Timedelta(days=1).to_pytimedelta()) in de:
            # Monday before a Tuesday federal holiday
            bridge_dates.add(d)
        if wd == 4 and (d + pd.Timedelta(days=-1).to_pytimedelta()) in de:
            # Friday after a Thursday federal holiday — the classic case
            bridge_dates.add(d)
    return local_dates.isin(bridge_dates).rename("is_bridge_day")


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """All calendar-derived features for a tz-aware UTC index. No data needed."""
    if idx.tz is None:
        raise ValueError("idx must be tz-aware UTC.")

    local = idx.tz_convert("Europe/Berlin")
    out = pd.DataFrame(index=idx)

    # Time-of-day / week / year
    out["hour"] = local.hour
    out["quarter_of_hour"] = local.minute // 15
    out["dow"] = local.dayofweek
    out["dom"] = local.day
    out["month"] = local.month
    out["week_of_year"] = local.isocalendar().week.astype(int).to_numpy()

    # Cyclical encodings — sine/cosine for each periodic feature
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)

    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_federal_holiday"] = is_federal_holiday(idx).astype(int).to_numpy()
    out["hol_pop_frac"] = population_weighted_holiday_fraction(idx).to_numpy()
    out["is_bridge_day"] = is_bridge_day(idx).astype(int).to_numpy()

    # DST flag — the local clock shifted in the last 24h
    offsets = local.to_series().apply(lambda t: t.utcoffset())
    offset_hours = offsets.dt.total_seconds() / 3600
    out["utc_offset_hours"] = offset_hours.to_numpy()

    return out
