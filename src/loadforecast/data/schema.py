"""Column registry — every output column declared once.

Every column we keep in the parquet has exactly one entry here. The
column name is a stable contract used across the project (M2 features
reference these names verbatim, the LSTM windowing uses two of them,
and so on).

Adding a new column: append a `Column(...)` to `COLUMNS` below.
- `name` — the parquet column name. Must be unique. Snake-case prefix
  encodes the kind: `actual_cons__`, `actual_gen__`, `fc_cons__`,
  `fc_gen__`, `price__`. The prefix is what `availability.classify_column`
  reads to decide leakage rules.
- `source` — which sources/<file>.py knows how to fetch it.
- `description` — one line for humans.

The schema is data-agnostic: it doesn't fetch anything itself. The
`refresh` orchestrator iterates `COLUMNS` and dispatches to the right
source module.
"""

from __future__ import annotations

from dataclasses import dataclass

# Source identifiers — must match the module name under sources/
SRC_ENERGY_CHARTS = "energy_charts"
SRC_ENTSOE = "entsoe"


# Bidding zones we pull day-ahead prices for, with the column suffix to use.
# Matches the existing parquet's `price__*` columns and the entsoe-py /
# energy-charts zone codes.
PRICE_ZONES: dict[str, dict[str, str]] = {
    "germany_luxembourg":   {"ec": "DE-LU",   "entsoe": "DE_LU"},
    "france":               {"ec": "FR",      "entsoe": "FR"},
    "netherlands":          {"ec": "NL",      "entsoe": "NL"},
    "austria":              {"ec": "AT",      "entsoe": "AT"},
    "belgium":              {"ec": "BE",      "entsoe": "BE"},
    "switzerland":          {"ec": "CH",      "entsoe": "CH"},
    "czech_republic":       {"ec": "CZ",      "entsoe": "CZ"},
    "poland":               {"ec": "PL",      "entsoe": "PL"},
    "denmark_1":            {"ec": "DK1",     "entsoe": "DK_1"},
    "denmark_2":            {"ec": "DK2",     "entsoe": "DK_2"},
    "norway_2":             {"ec": "NO2",     "entsoe": "NO_2"},
    "sweden_4":             {"ec": "SE4",     "entsoe": "SE_4"},
    "slovenia":             {"ec": "SI",      "entsoe": "SI"},
    "hungary":              {"ec": "HU",      "entsoe": "HU"},
    # IT-NORTH 400'd on energy-charts; ENTSO-E code is IT_NORD. Add later if needed.
}


# Energy-Charts production-type strings → our column suffix.
GEN_SOURCES: dict[str, str] = {
    "biomass":              "biomass",
    "hydro_run_of_river":   "hydropower",
    "wind_offshore":        "wind_offshore",
    "wind_onshore":         "wind_onshore",
    "solar":                "photovoltaics",
    "renewable_share_of_load": None,  # placeholder, ignored
    "fossil_brown_coal":    "lignite",
    "fossil_hard_coal":     "hard_coal",
    "fossil_gas":           "fossil_gas",
    "hydro_pumped_storage": "hydro_pumped_storage",
    "others":               "other_conventional",
    "renewables":           "other_renewable",  # rough mapping
}


@dataclass(frozen=True)
class Column:
    name: str           # exact parquet column name
    source: str         # SRC_ENERGY_CHARTS or SRC_ENTSOE
    description: str
    fetch_kwargs: dict = None  # source-specific fetch parameters


# Build the full column list programmatically.
def _build_columns() -> list[Column]:
    cols: list[Column] = []

    # 1. Day-ahead prices for every zone (energy-charts).
    for suffix, codes in PRICE_ZONES.items():
        cols.append(Column(
            name=f"price__{suffix}",
            source=SRC_ENERGY_CHARTS,
            description=f"Day-ahead spot price, bidding zone {codes['ec']}, EUR/MWh",
            fetch_kwargs={"endpoint": "price", "bzn": codes["ec"]},
        ))

    # 2. Actual generation by source (energy-charts /public_power).
    cols.append(Column(
        name="actual_gen__photovoltaics",
        source=SRC_ENERGY_CHARTS,
        description="Actual solar PV generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Solar"},
    ))
    cols.append(Column(
        name="actual_gen__wind_onshore",
        source=SRC_ENERGY_CHARTS,
        description="Actual onshore wind generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Wind onshore"},
    ))
    cols.append(Column(
        name="actual_gen__wind_offshore",
        source=SRC_ENERGY_CHARTS,
        description="Actual offshore wind generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Wind offshore"},
    ))
    cols.append(Column(
        name="actual_gen__biomass",
        source=SRC_ENERGY_CHARTS,
        description="Actual biomass generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Biomass"},
    ))
    cols.append(Column(
        name="actual_gen__hydropower",
        source=SRC_ENERGY_CHARTS,
        description="Actual hydropower (run-of-river) generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Hydro Run-of-River"},
    ))
    cols.append(Column(
        name="actual_gen__hydro_pumped_storage",
        source=SRC_ENERGY_CHARTS,
        description="Actual hydro pumped-storage generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Hydro pumped storage"},
    ))
    cols.append(Column(
        name="actual_gen__lignite",
        source=SRC_ENERGY_CHARTS,
        description="Actual lignite (brown coal) generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Fossil brown coal / lignite"},
    ))
    cols.append(Column(
        name="actual_gen__hard_coal",
        source=SRC_ENERGY_CHARTS,
        description="Actual hard coal generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Fossil hard coal"},
    ))
    cols.append(Column(
        name="actual_gen__fossil_gas",
        source=SRC_ENERGY_CHARTS,
        description="Actual fossil-gas generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Fossil gas"},
    ))
    cols.append(Column(
        name="actual_gen__other_renewable",
        source=SRC_ENERGY_CHARTS,
        description="Actual other-renewable generation (geothermal, etc.), MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Other renewable"},
    ))
    cols.append(Column(
        name="actual_gen__other_conventional",
        source=SRC_ENERGY_CHARTS,
        description="Actual other-conventional generation, MW",
        fetch_kwargs={"endpoint": "public_power", "production_type": "Others"},
    ))

    # 3. Actual + forecast load (ENTSO-E).
    cols.append(Column(
        name="actual_cons__grid_load",
        source=SRC_ENTSOE,
        description="Actual total grid load (national consumption), MW",
        fetch_kwargs={"method": "load", "country_code": "DE_LU"},
    ))
    cols.append(Column(
        name="fc_cons__grid_load",
        source=SRC_ENTSOE,
        description="Day-ahead total grid load forecast (TSO baseline), MW",
        fetch_kwargs={"method": "load_forecast", "country_code": "DE_LU"},
    ))

    # 4. TSO generation forecasts (ENTSO-E).
    cols.append(Column(
        name="fc_gen__total",
        source=SRC_ENTSOE,
        description="Day-ahead total generation forecast, MW",
        fetch_kwargs={"method": "generation_forecast", "country_code": "DE_LU"},
    ))
    cols.append(Column(
        name="fc_gen__photovoltaics_and_wind",
        source=SRC_ENTSOE,
        description="Day-ahead wind+solar forecast (sum of all VRE), MW",
        fetch_kwargs={"method": "wind_and_solar_forecast", "country_code": "DE_LU"},
    ))

    return cols


COLUMNS: list[Column] = _build_columns()
COLUMN_BY_NAME: dict[str, Column] = {c.name: c for c in COLUMNS}


def columns_by_source(source: str) -> list[Column]:
    return [c for c in COLUMNS if c.source == source]


__all__ = [
    "COLUMNS",
    "COLUMN_BY_NAME",
    "Column",
    "GEN_SOURCES",
    "PRICE_ZONES",
    "SRC_ENERGY_CHARTS",
    "SRC_ENTSOE",
    "columns_by_source",
]
