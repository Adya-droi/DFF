"""
anomaly_detection.py
=====================
AI/statistical anomaly detection layer for the HVAC Engineering Copilot.

Combines:
  1. An unsupervised ML model (scikit-learn IsolationForest) for multivariate
     "unusual operating state" detection across sensor channels.
  2. Domain-specific rule-based heuristics that encode HVAC engineering
     knowledge (humidity risk, CO2 spikes, ventilation failure signatures,
     cooling-loop inefficiency, and sensor faults such as flat-lining).

The rule-based layer is what makes the results *explainable* to a building
engineer — the ML layer is what catches things the rules didn't anticipate.

Author: Digital Twin Research Lab (prototype)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_processing import (
    TIMESTAMP_COL, INDOOR_TEMP_COL, HUMIDITY_COL, CO2_COL,
    OUTDOOR_TEMP_COL, ENERGY_COL, COOLING_LOAD_COL,
    CHW_SUPPLY_COL, CHW_RETURN_COL,
)


# ---------------------------------------------------------------------------
# 1. Machine-learning based multivariate anomaly detection
# ---------------------------------------------------------------------------

def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.03,
                          random_state: int = 42) -> pd.DataFrame:
    """
    Fit an IsolationForest across available numeric sensor channels to flag
    multivariate operating anomalies (points that are unusual jointly across
    several signals, even if each signal alone looks "in range").

    `contamination` is the assumed proportion of anomalous points and is a
    tunable research parameter — 3% is a reasonable default for HVAC time
    series without labelled ground truth.
    """
    candidate_features = [INDOOR_TEMP_COL, HUMIDITY_COL, CO2_COL,
                           OUTDOOR_TEMP_COL, ENERGY_COL, COOLING_LOAD_COL,
                           CHW_SUPPLY_COL, CHW_RETURN_COL]
    features = [f for f in candidate_features if f in df.columns]

    out = df.copy()
    if len(features) < 2 or len(df) < 20:
        # Not enough signal / data to train reliably.
        out["anomaly_score"] = 0.0
        out["is_anomaly"] = False
        return out

    X = out[features].fillna(out[features].mean())
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_scaled)

    # decision_function: higher = more normal. We invert & normalise to a
    # 0-1 "anomaly score" that's more intuitive for the dashboard.
    raw_scores = -model.decision_function(X_scaled)
    out["anomaly_score"] = np.round(
        (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9), 3
    )
    out["is_anomaly"] = model.predict(X_scaled) == -1
    return out


# ---------------------------------------------------------------------------
# 2. Rule-based / domain-knowledge checks
# ---------------------------------------------------------------------------

def detect_high_humidity_anomalies(df: pd.DataFrame, threshold: float = 65.0) -> pd.DataFrame:
    """Flag individual records where relative humidity exceeds a risk threshold."""
    flagged = df[df[HUMIDITY_COL] > threshold][[TIMESTAMP_COL, HUMIDITY_COL]].copy()
    flagged["issue"] = "High humidity"
    flagged["severity"] = np.where(flagged[HUMIDITY_COL] > 75, "High", "Medium")
    return flagged.rename(columns={HUMIDITY_COL: "value"})


def detect_co2_spikes(df: pd.DataFrame, threshold: float = 1000.0) -> pd.DataFrame:
    """Flag records where CO2 exceeds an indoor air quality threshold (ASHRAE-informed)."""
    flagged = df[df[CO2_COL] > threshold][[TIMESTAMP_COL, CO2_COL]].copy()
    flagged["issue"] = "CO2 spike"
    flagged["severity"] = np.where(flagged[CO2_COL] > 1500, "High", "Medium")
    return flagged.rename(columns={CO2_COL: "value"})


def detect_ventilation_failures(df: pd.DataFrame, co2_threshold: float = 1200.0,
                                 window: int = 4) -> pd.DataFrame:
    """
    Heuristic ventilation-failure signature: CO2 remains persistently high
    (above threshold) for a sustained rolling window while it is NOT falling,
    which suggests fresh-air intake is not keeping pace with occupancy —
    rather than a brief, self-correcting spike.
    """
    s = df[CO2_COL]
    rolling_min = s.rolling(window, min_periods=window).min()
    persistent_high = rolling_min > co2_threshold
    events = df.loc[persistent_high, [TIMESTAMP_COL, CO2_COL]].copy()
    events["issue"] = "Possible ventilation failure (sustained high CO2)"
    events["severity"] = "High"
    return events.rename(columns={CO2_COL: "value"})


def detect_cooling_inefficiency(df: pd.DataFrame, delta_t_threshold: float = 4.0) -> pd.DataFrame:
    """
    Flag periods of likely cooling-plant inefficiency using chilled-water
    supply/return delta-T (a classic chiller-plant health indicator — a
    "low delta-T syndrome" suggests reduced heat-transfer effectiveness,
    valve/control issues, or coil fouling), falling back to a cooling-load
    vs. energy consistency check when CHW temperatures aren't available.
    """
    events = pd.DataFrame()
    if CHW_SUPPLY_COL in df.columns and CHW_RETURN_COL in df.columns:
        delta_t = df[CHW_RETURN_COL] - df[CHW_SUPPLY_COL]
        mask = delta_t < delta_t_threshold
        events = df.loc[mask, [TIMESTAMP_COL]].copy()
        events["value"] = delta_t[mask].round(2)
        events["issue"] = "Low chilled-water delta-T (possible cooling inefficiency)"
        events["severity"] = "Medium"
    elif COOLING_LOAD_COL in df.columns:
        # Fallback heuristic: high energy draw without proportional cooling load
        # suggests the cooling plant is consuming power without delivering load.
        ratio = df[ENERGY_COL] / df[COOLING_LOAD_COL].replace(0, np.nan)
        threshold = ratio.mean() + 2 * ratio.std()
        mask = ratio > threshold
        events = df.loc[mask, [TIMESTAMP_COL]].copy()
        events["value"] = ratio[mask].round(3)
        events["issue"] = "High energy-to-cooling-load ratio (possible inefficiency)"
        events["severity"] = "Medium"
    return events


def detect_sensor_faults(df: pd.DataFrame, flatline_window: int = 6,
                          flatline_tolerance: float = 0.01) -> pd.DataFrame:
    """
    Detect likely sensor faults via flat-lining (a sensor reporting an
    essentially constant value for an implausibly long window — a classic
    stuck-sensor signature) across all monitored channels.
    """
    channels = [INDOOR_TEMP_COL, HUMIDITY_COL, CO2_COL, OUTDOOR_TEMP_COL, ENERGY_COL]
    channels = [c for c in channels if c in df.columns]

    events = []
    for col in channels:
        rolling_std = df[col].rolling(flatline_window, min_periods=flatline_window).std()
        flat_mask = rolling_std < flatline_tolerance
        for ts in df.loc[flat_mask, TIMESTAMP_COL]:
            events.append({
                "timestamp": ts,
                "value": None,
                "issue": f"Possible sensor fault: {col} flat-lined",
                "severity": "Medium",
                "channel": col,
            })
    return pd.DataFrame(events)


def run_all_rule_based_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run every rule-based detector and return a single unified anomaly log."""
    frames = [
        detect_high_humidity_anomalies(df),
        detect_co2_spikes(df),
        detect_ventilation_failures(df),
        detect_cooling_inefficiency(df),
        detect_sensor_faults(df),
    ]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=[TIMESTAMP_COL, "value", "issue", "severity"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return combined


def summarize_anomalies(rule_based_df: pd.DataFrame, ml_df: pd.DataFrame) -> dict:
    """Produce a compact anomaly summary dict suitable for the LLM prompt."""
    summary = {
        "total_rule_based_events": len(rule_based_df),
        "by_issue_type": rule_based_df["issue"].value_counts().to_dict() if not rule_based_df.empty else {},
        "ml_anomalies_detected": int(ml_df["is_anomaly"].sum()) if "is_anomaly" in ml_df.columns else 0,
        "ml_anomaly_rate_pct": round(ml_df["is_anomaly"].mean() * 100, 2) if "is_anomaly" in ml_df.columns else 0.0,
    }
    return summary
