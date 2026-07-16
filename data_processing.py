"""
data_processing.py
==================
Data ingestion, cleaning, standardisation, and core building-analytics
routines for the HVAC Engineering Copilot research prototype.

This module is intentionally framework-agnostic (no Streamlit calls) so it
can be unit-tested or reused inside a future Digital Twin orchestration
layer. All functions operate on / return pandas DataFrames or plain dicts.

Author: Digital Twin Research Lab (prototype)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

# Canonical internal column names the rest of the pipeline relies on.
TIMESTAMP_COL = "timestamp"
INDOOR_TEMP_COL = "indoor_temp_c"
HUMIDITY_COL = "humidity_pct"
CO2_COL = "co2_ppm"
OUTDOOR_TEMP_COL = "outdoor_temp_c"
COOLING_LOAD_COL = "cooling_load_kw"          # optional
CHW_SUPPLY_COL = "chw_supply_temp_c"          # optional
CHW_RETURN_COL = "chw_return_temp_c"          # optional
ENERGY_COL = "energy_kwh"

REQUIRED_COLUMNS = [
    TIMESTAMP_COL, INDOOR_TEMP_COL, HUMIDITY_COL, CO2_COL,
    OUTDOOR_TEMP_COL, ENERGY_COL,
]
OPTIONAL_COLUMNS = [COOLING_LOAD_COL, CHW_SUPPLY_COL, CHW_RETURN_COL]

# A flexible alias map: many building management systems (BMS) export data
# with slightly different headers. We try to auto-map common variants onto
# our canonical schema so the tool "just works" on real-world exports.
COLUMN_ALIASES = {
    TIMESTAMP_COL: ["timestamp", "time", "datetime", "date_time", "date"],
    INDOOR_TEMP_COL: ["indoor temperature", "indoor_temperature", "indoor temp",
                       "room_temp", "zone_temp", "indoor_temp_c", "indoor temperature (c)"],
    HUMIDITY_COL: ["relative humidity", "relative_humidity", "humidity",
                   "rh", "humidity_pct", "relative humidity (%)"],
    CO2_COL: ["co2", "co2_ppm", "co₂", "carbon dioxide", "co2 (ppm)"],
    OUTDOOR_TEMP_COL: ["outdoor temperature", "outdoor_temperature", "outdoor temp",
                        "ambient_temp", "outdoor_temp_c", "oat"],
    COOLING_LOAD_COL: ["cooling load", "cooling_load", "cooling_load_kw", "load"],
    CHW_SUPPLY_COL: ["chilled water supply temperature", "chw_supply", "chw_supply_temp",
                      "chw supply temp", "supply_temp"],
    CHW_RETURN_COL: ["chilled water return temperature", "chw_return", "chw_return_temp",
                      "chw return temp", "return_temp"],
    ENERGY_COL: ["energy consumption", "energy_consumption", "energy", "energy_kwh",
                 "kwh", "power_kwh"],
}


@dataclass
class CleaningReport:
    """Structured summary of what happened during the cleaning pass."""
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    missing_values_filled: dict = field(default_factory=dict)
    sensor_errors_flagged: dict = field(default_factory=dict)
    timestamp_issues: int = 0
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading & column standardisation
# ---------------------------------------------------------------------------

def load_csv(file) -> pd.DataFrame:
    """Load a CSV (path or file-like buffer from Streamlit uploader)."""
    df = pd.read_csv(file)
    return standardize_columns(df)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename incoming columns to the canonical schema using alias matching.

    If more than one raw column matches the SAME canonical field (e.g. a CSV
    export that contains both "CO2" and "CO2 (ppm)"), we deliberately do NOT
    guess which one is correct — silently picking one would risk analysing
    the wrong sensor column without anyone noticing. Instead we leave all
    matching columns un-renamed and record the ambiguity on `df.attrs`, so
    `validate_schema()` correctly reports the field as missing and the
    caller (see `clean_data` / app.py) can show a clear, specific warning
    naming exactly which raw columns collided.
    """
    lower_map = {c: c.strip().lower() for c in df.columns}

    canonical_to_originals: dict[str, list] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        matches = [original for original, lowered in lower_map.items()
                   if lowered in aliases or lowered == canonical]
        if matches:
            canonical_to_originals[canonical] = matches

    rename_map = {}
    ambiguous = {}
    for canonical, originals in canonical_to_originals.items():
        if len(originals) == 1:
            rename_map[originals[0]] = canonical
        else:
            # Multiple raw columns collide on one canonical field — leave
            # them all un-renamed rather than guessing.
            ambiguous[canonical] = originals

    df = df.rename(columns=rename_map)
    df.attrs["column_mapping_ambiguities"] = ambiguous
    return df


def validate_schema(df: pd.DataFrame) -> list:
    """Return a list of missing required columns (empty list == OK)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def get_column_ambiguities(df: pd.DataFrame) -> dict:
    """
    Return the {canonical_field: [raw_column_names]} mapping ambiguities
    recorded by `standardize_columns`, if any (empty dict == no conflicts).
    Safe to call on any DataFrame, including ones that never passed through
    `load_csv` (returns {} in that case).
    """
    return df.attrs.get("column_mapping_ambiguities", {})


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Run the full cleaning pipeline:
      1. Standardise / parse timestamps
      2. Drop exact duplicate rows
      3. Detect and null-out physically implausible sensor readings
      4. Fill missing values via time-aware interpolation
      5. Sort chronologically and reset index

    Returns the cleaned DataFrame plus a CleaningReport describing every
    modification made (kept for transparency in the UI / audit trail).
    """
    report = CleaningReport(rows_in=len(df))
    df = df.copy()

    # --- 0. Surface any ambiguous column mappings from standardize_columns --
    ambiguities = get_column_ambiguities(df)
    for canonical, originals in ambiguities.items():
        report.notes.append(
            f"Ambiguous column mapping for '{canonical}': found {len(originals)} "
            f"matching raw columns {originals} — none were used automatically "
            f"to avoid guessing wrong. Please keep only one such column in "
            f"your CSV and remove/rename the others, then re-upload."
        )

    # --- 1. Timestamp standardisation -------------------------------------
    if TIMESTAMP_COL in df.columns:
        parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        bad_ts = parsed.isna().sum()
        report.timestamp_issues = int(bad_ts)
        df[TIMESTAMP_COL] = parsed
        df = df.dropna(subset=[TIMESTAMP_COL])
    else:
        report.notes.append("No timestamp column found — synthetic index used.")
        df[TIMESTAMP_COL] = pd.date_range("2024-01-01", periods=len(df), freq="15min")

    df = df.sort_values(TIMESTAMP_COL)

    # --- 2. Duplicate removal ----------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="first")
    df = df.drop_duplicates(keep="first")
    report.duplicates_removed = before - len(df)

    # --- 3. Sensor-error detection (physically implausible values) --------
    # Realistic bounds for a commercial building in a hot climate context.
    sensor_bounds = {
        INDOOR_TEMP_COL: (10.0, 40.0),
        HUMIDITY_COL: (0.0, 100.0),
        CO2_COL: (300.0, 5000.0),
        OUTDOOR_TEMP_COL: (-10.0, 55.0),
        ENERGY_COL: (0.0, None),
        COOLING_LOAD_COL: (0.0, None),
        CHW_SUPPLY_COL: (0.0, 25.0),
        CHW_RETURN_COL: (0.0, 30.0),
    }
    for col, (lo, hi) in sensor_bounds.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[col] < lo
        if hi is not None:
            mask |= df[col] > hi
        n_flagged = int(mask.sum())
        if n_flagged:
            report.sensor_errors_flagged[col] = n_flagged
            df.loc[mask, col] = np.nan  # invalidate; will be interpolated below

    # --- 4. Missing value handling ------------------------------------------
    numeric_cols = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
                    if c in df.columns and c != TIMESTAMP_COL]
    df = df.set_index(TIMESTAMP_COL)
    for col in numeric_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            # Time-aware linear interpolation, then forward/back fill edges.
            df[col] = df[col].interpolate(method="time", limit_direction="both")
            df[col] = df[col].ffill().bfill()
            report.missing_values_filled[col] = n_missing
    df = df.reset_index()

    report.rows_out = len(df)
    return df, report


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------

def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """High-level descriptive statistics used across the dashboard & LLM prompt."""
    stats = {}
    stats["period_start"] = df[TIMESTAMP_COL].min()
    stats["period_end"] = df[TIMESTAMP_COL].max()
    stats["n_records"] = len(df)

    stats["avg_indoor_temp"] = round(df[INDOOR_TEMP_COL].mean(), 2)
    stats["min_indoor_temp"] = round(df[INDOOR_TEMP_COL].min(), 2)
    stats["max_indoor_temp"] = round(df[INDOOR_TEMP_COL].max(), 2)

    stats["avg_humidity"] = round(df[HUMIDITY_COL].mean(), 2)
    stats["max_humidity"] = round(df[HUMIDITY_COL].max(), 2)

    stats["avg_co2"] = round(df[CO2_COL].mean(), 1)
    stats["max_co2"] = round(df[CO2_COL].max(), 1)
    stats["pct_time_co2_above_1000"] = round((df[CO2_COL] > 1000).mean() * 100, 1)

    stats["total_energy_kwh"] = round(df[ENERGY_COL].sum(), 1)
    stats["avg_energy_kwh"] = round(df[ENERGY_COL].mean(), 3)

    if COOLING_LOAD_COL in df.columns:
        stats["avg_cooling_load"] = round(df[COOLING_LOAD_COL].mean(), 2)
        stats["max_cooling_load"] = round(df[COOLING_LOAD_COL].max(), 2)

    if CHW_SUPPLY_COL in df.columns and CHW_RETURN_COL in df.columns:
        delta_t = df[CHW_RETURN_COL] - df[CHW_SUPPLY_COL]
        stats["avg_chw_delta_t"] = round(delta_t.mean(), 2)
        stats["min_chw_delta_t"] = round(delta_t.min(), 2)

    return stats


def compute_daily_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate energy consumption by calendar day."""
    daily = (
        df.set_index(TIMESTAMP_COL)[ENERGY_COL]
        .resample("D")
        .sum()
        .reset_index()
        .rename(columns={ENERGY_COL: "daily_energy_kwh"})
    )
    return daily


def compute_peak_cooling_periods(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Identify the top-N timestamps with the highest cooling demand.

    Uses cooling_load if available, otherwise falls back to energy consumption
    as a proxy for cooling demand.
    """
    metric_col = COOLING_LOAD_COL if COOLING_LOAD_COL in df.columns else ENERGY_COL
    peak = df.nlargest(top_n, metric_col)[[TIMESTAMP_COL, metric_col, INDOOR_TEMP_COL, OUTDOOR_TEMP_COL]]
    return peak.rename(columns={metric_col: "value"})


def estimate_occupancy(df: pd.DataFrame, outdoor_co2_baseline: float = 420.0) -> pd.DataFrame:
    """
    Rough occupancy estimation from CO2 build-up above outdoor baseline.

    This is a simplified single-zone mass-balance heuristic intended for
    research/demo purposes — NOT a substitute for a calibrated occupancy
    model. Roughly: each occupant steady-state contributes ~30-40 ppm above
    baseline in a moderately ventilated space; we use that as a rough divisor
    to produce a relative occupancy_index and a qualitative band.
    """
    out = df[[TIMESTAMP_COL, CO2_COL]].copy()
    out["co2_above_baseline"] = (out[CO2_COL] - outdoor_co2_baseline).clip(lower=0)
    out["occupancy_index"] = (out["co2_above_baseline"] / 35.0).round(1)  # heuristic divisor

    def band(v):
        if v < 2:
            return "Low"
        elif v < 8:
            return "Medium"
        else:
            return "High"

    out["occupancy_band"] = out["occupancy_index"].apply(band)
    return out


def compute_comfort_index(df: pd.DataFrame,
                           temp_setpoint: float = 23.0,
                           humidity_setpoint: float = 50.0) -> pd.DataFrame:
    """
    Compute a simplified 0-100 thermal-comfort index per record, combining
    temperature deviation, humidity deviation, and CO2 (air quality) penalty.

    This is a pedagogical approximation of comfort (inspired loosely by
    ASHRAE 55 concepts) — not a certified PMV/PPD calculation.
    """
    out = df[[TIMESTAMP_COL]].copy()

    temp_penalty = (df[INDOOR_TEMP_COL] - temp_setpoint).abs() * 6.0       # 6 pts per deg C
    humidity_penalty = (df[HUMIDITY_COL] - humidity_setpoint).abs() * 0.6   # 0.6 pts per %RH
    co2_penalty = ((df[CO2_COL] - 800).clip(lower=0) / 20.0)                # penalty above 800ppm

    raw_index = 100 - temp_penalty - humidity_penalty - co2_penalty
    out["comfort_index"] = raw_index.clip(lower=0, upper=100).round(1)

    def rating(v):
        if v >= 80:
            return "Excellent"
        elif v >= 60:
            return "Acceptable"
        elif v >= 40:
            return "Marginal"
        else:
            return "Poor"

    out["comfort_rating"] = out["comfort_index"].apply(rating)
    return out


def compute_humidity_risk_periods(df: pd.DataFrame, threshold: float = 65.0,
                                   min_duration_records: int = 3) -> pd.DataFrame:
    """
    Identify sustained periods where humidity exceeds a mould/condensation
    risk threshold for at least `min_duration_records` consecutive samples.
    """
    over = df[HUMIDITY_COL] > threshold
    groups = (over != over.shift()).cumsum()
    risk_periods = []
    for _, grp in df.groupby(groups):
        if over.loc[grp.index].all() and len(grp) >= min_duration_records:
            risk_periods.append({
                "start": grp[TIMESTAMP_COL].iloc[0],
                "end": grp[TIMESTAMP_COL].iloc[-1],
                "duration_records": len(grp),
                "max_humidity": round(grp[HUMIDITY_COL].max(), 1),
            })
    return pd.DataFrame(risk_periods)
