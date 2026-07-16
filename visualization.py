"""
visualization.py
=================
Plotly chart builders for the HVAC Engineering Copilot dashboard.

All functions return a `plotly.graph_objects.Figure` so app.py can render
them with `st.plotly_chart(fig, use_container_width=True)`. Kept separate
from app.py so charts can be reused/tested independently of Streamlit.

Author: Digital Twin Research Lab (prototype)
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_processing import (
    TIMESTAMP_COL, INDOOR_TEMP_COL, HUMIDITY_COL, CO2_COL,
    OUTDOOR_TEMP_COL, ENERGY_COL, COOLING_LOAD_COL,
)

# University research branding palette
BRAND_PRIMARY = "#0B3D91"      # deep research blue
BRAND_ACCENT = "#2CA6A4"       # teal accent
BRAND_WARNING = "#E4572E"      # anomaly / alert orange-red
BRAND_NEUTRAL = "#6C757D"
BRAND_BG = "#FFFFFF"

TEMPLATE = "plotly_white"


def _base_layout(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, font=dict(size=16, color=BRAND_PRIMARY)),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=380,
    )
    fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(title_text="Time")
    return fig


def plot_temperature(df) -> go.Figure:
    """Indoor vs. outdoor temperature over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[TIMESTAMP_COL], y=df[INDOOR_TEMP_COL],
                              name="Indoor Temp (°C)", line=dict(color=BRAND_PRIMARY, width=2)))
    fig.add_trace(go.Scatter(x=df[TIMESTAMP_COL], y=df[OUTDOOR_TEMP_COL],
                              name="Outdoor Temp (°C)", line=dict(color=BRAND_ACCENT, width=2, dash="dot")))
    return _base_layout(fig, "Indoor vs Outdoor Temperature", "Temperature (°C)")


def plot_humidity(df, threshold: float = 65.0) -> go.Figure:
    """Relative humidity with a risk-threshold reference line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[TIMESTAMP_COL], y=df[HUMIDITY_COL],
                              name="Relative Humidity (%)", line=dict(color=BRAND_ACCENT, width=2),
                              fill="tozeroy", fillcolor="rgba(44,166,164,0.12)"))
    fig.add_hline(y=threshold, line_dash="dash", line_color=BRAND_WARNING,
                  annotation_text=f"Risk threshold ({threshold}%)", annotation_position="top left")
    return _base_layout(fig, "Relative Humidity", "Humidity (%)")


def plot_co2(df, threshold: float = 1000.0) -> go.Figure:
    """CO2 concentration with an air-quality threshold reference line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[TIMESTAMP_COL], y=df[CO2_COL],
                              name="CO₂ (ppm)", line=dict(color="#7B5EA7", width=2)))
    fig.add_hline(y=threshold, line_dash="dash", line_color=BRAND_WARNING,
                  annotation_text=f"IAQ threshold ({int(threshold)} ppm)", annotation_position="top left")
    return _base_layout(fig, "CO₂ Concentration (Air Quality / Occupancy Proxy)", "CO₂ (ppm)")


def plot_energy(daily_energy_df) -> go.Figure:
    """Daily energy consumption bar chart."""
    fig = px.bar(daily_energy_df, x=TIMESTAMP_COL, y="daily_energy_kwh",
                  color_discrete_sequence=[BRAND_PRIMARY])
    fig.update_traces(marker_line_width=0)
    return _base_layout(fig, "Daily Energy Consumption", "Energy (kWh)")


def plot_anomalies_overlay(df, ml_df) -> go.Figure:
    """Energy/cooling-load series with ML-flagged anomalies highlighted."""
    metric_col = COOLING_LOAD_COL if COOLING_LOAD_COL in df.columns else ENERGY_COL
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[TIMESTAMP_COL], y=df[metric_col],
                              name=metric_col, line=dict(color=BRAND_NEUTRAL, width=1.5)))

    if "is_anomaly" in ml_df.columns:
        anomalies = ml_df[ml_df["is_anomaly"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies[TIMESTAMP_COL], y=anomalies[metric_col],
                mode="markers", name="ML-flagged anomaly",
                marker=dict(color=BRAND_WARNING, size=8, symbol="x"),
            ))
    return _base_layout(fig, "Anomaly Detection Overlay", metric_col)


def plot_comfort_index(comfort_df) -> go.Figure:
    """Comfort index trend over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comfort_df[TIMESTAMP_COL], y=comfort_df["comfort_index"],
                              name="Comfort Index", line=dict(color=BRAND_ACCENT, width=2),
                              fill="tozeroy", fillcolor="rgba(44,166,164,0.1)"))
    for y, label, color in [(80, "Excellent", "#2CA6A4"), (60, "Acceptable", "#F2C14E"),
                             (40, "Marginal", "#E4572E")]:
        fig.add_hline(y=y, line_dash="dot", line_color=color, opacity=0.6)
    return _base_layout(fig, "Thermal Comfort Index (0-100)", "Comfort Index")


def plot_correlation_heatmap(df, columns: list) -> go.Figure:
    """Correlation heatmap across numeric sensor channels."""
    corr = df[columns].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     template=TEMPLATE)
    fig.update_layout(title=dict(text="Sensor Correlation Matrix", font=dict(size=16, color=BRAND_PRIMARY)),
                       height=420, margin=dict(l=40, r=20, t=50, b=40))
    return fig
