"""
generate_sample_data.py
========================
Generates a synthetic building operational dataset for demoing the HVAC
Engineering Copilot without requiring a real BMS export. Simulates:

  - Diurnal outdoor temperature swings (hot climate / district cooling context)
  - Indoor temperature loosely tracking a cooling setpoint with noise
  - Humidity with occasional sustained high-humidity "risk" events
  - CO2 driven by an occupancy schedule (weekday office pattern) with an
    injected ventilation-failure event
  - Cooling load and chilled-water supply/return temperatures, including an
    injected "low delta-T" cooling-inefficiency period
  - Energy consumption correlated with cooling load
  - A handful of injected sensor faults (flat-lines, out-of-range spikes,
    missing values, duplicate rows) so the cleaning pipeline has something
    to do

Run with:  python generate_sample_data.py
Produces: sample_data/sample_building_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(7)

N_DAYS = 14
FREQ_MINUTES = 15
periods = int(N_DAYS * 24 * 60 / FREQ_MINUTES)

timestamps = pd.date_range("2024-06-01", periods=periods, freq=f"{FREQ_MINUTES}min")
hours = np.array(timestamps.hour + timestamps.minute / 60.0)
day_frac = np.array((timestamps - timestamps[0]).total_seconds() / 86400.0)
weekday = np.array(timestamps.weekday)  # 0=Mon .. 6=Sun

# --- Outdoor temperature: diurnal cycle around a hot-climate mean ----------
outdoor_temp = 34 + 6 * np.sin((hours - 15) / 24 * 2 * np.pi) + np.random.normal(0, 0.6, periods)

# --- Occupancy schedule: weekday 8am-6pm ramps -----------------------------
occ_schedule = np.where(
    (weekday < 5) & (hours >= 8) & (hours <= 18),
    1.0,
    np.where((weekday < 5) & ((hours >= 6) & (hours < 8) | (hours > 18) & (hours <= 20)), 0.4, 0.05)
)
occ_schedule += np.random.normal(0, 0.03, periods)
occ_schedule = np.clip(occ_schedule, 0, 1)

# --- Indoor temperature: cooling tries to hold ~23C, harder when hot out ---
cooling_effort = 0.6 + 0.4 * occ_schedule
indoor_temp = 23.5 - 1.0 * cooling_effort + 0.08 * (outdoor_temp - 34) + np.random.normal(0, 0.35, periods)

# --- Humidity: baseline ~55%, with two injected sustained high-humidity events
humidity = 55 + 8 * np.sin((hours - 10) / 24 * 2 * np.pi) + np.random.normal(0, 2.5, periods)
humidity = np.clip(humidity, 30, 95)
# Inject sustained high-humidity risk period (day 4, afternoon) -> dehumidification failure
mask_humid_event = (day_frac > 4.4) & (day_frac < 4.7)
humidity[mask_humid_event] = np.random.normal(78, 3, mask_humid_event.sum())

# --- CO2: baseline 420 outdoor + occupancy build-up ------------------------
co2 = 420 + occ_schedule * 550 + np.random.normal(0, 25, periods)
# Inject a ventilation-failure event: CO2 stays high on day 8 during occupied hours
mask_vent_fail = (day_frac > 8.0) & (day_frac < 8.5) & (occ_schedule > 0.3)
co2[mask_vent_fail] = np.random.normal(1400, 60, mask_vent_fail.sum())
co2 = np.clip(co2, 400, None)

# --- Cooling load & chilled water loop --------------------------------------
cooling_load = 40 + 120 * cooling_effort + 3 * (outdoor_temp - 30) + np.random.normal(0, 5, periods)
cooling_load = np.clip(cooling_load, 10, None)

chw_supply = 7.0 + np.random.normal(0, 0.2, periods)
chw_return = chw_supply + (7.5 - 0.015 * (cooling_load - 100)) + np.random.normal(0, 0.3, periods)
# Inject a "low delta-T syndrome" cooling-inefficiency period on day 11
mask_low_dt = (day_frac > 11.2) & (day_frac < 11.6)
chw_return[mask_low_dt] = chw_supply[mask_low_dt] + np.random.normal(2.5, 0.3, mask_low_dt.sum())

# --- Energy: correlated with cooling load plus baseline plug loads --------
energy = 8 + 0.09 * cooling_load + np.random.normal(0, 1.2, periods)
energy = np.clip(energy, 2, None)

df = pd.DataFrame({
    "Timestamp": timestamps,
    "Indoor Temperature": np.round(indoor_temp, 2),
    "Relative Humidity": np.round(humidity, 2),
    "CO2": np.round(co2, 1),
    "Outdoor Temperature": np.round(outdoor_temp, 2),
    "Cooling Load": np.round(cooling_load, 2),
    "Chilled Water Supply Temperature": np.round(chw_supply, 2),
    "Chilled Water Return Temperature": np.round(chw_return, 2),
    "Energy Consumption": np.round(energy, 3),
})

# --- Inject realistic messiness for the cleaning pipeline to handle --------
rng = np.random.default_rng(42)

# 1. Missing values scattered across a few columns
for col in ["Indoor Temperature", "Relative Humidity", "CO2", "Energy Consumption"]:
    idx = rng.choice(df.index, size=15, replace=False)
    df.loc[idx, col] = np.nan

# 2. Out-of-range sensor errors (implausible spikes)
idx = rng.choice(df.index, size=6, replace=False)
df.loc[idx, "Indoor Temperature"] = rng.uniform(60, 90, size=6)  # faulty sensor spike

idx = rng.choice(df.index, size=4, replace=False)
df.loc[idx, "Relative Humidity"] = rng.uniform(150, 200, size=4)  # impossible RH

# 3. Duplicate rows
dup_idx = rng.choice(df.index[1:-1], size=8, replace=False)
dup_rows = df.loc[dup_idx]
df = pd.concat([df, dup_rows], ignore_index=True)

# 4. A short flat-lined sensor stretch (stuck CO2 sensor)
flat_start = 200
df.loc[flat_start:flat_start + 8, "CO2"] = df.loc[flat_start, "CO2"]

# Shuffle isn't applied (keep mostly chronological but with duplicates mixed in,
# mirroring a real noisy BMS export) — sort at the end only partially to
# preserve some disorder for the cleaning demo, then re-sort duplicates in.
df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)

df.to_csv("sample_data/sample_building_data.csv", index=False)
print(f"Sample dataset written to sample_data/sample_building_data.csv ({len(df)} rows)")
