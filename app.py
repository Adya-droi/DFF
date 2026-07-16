"""
app.py
======
HVAC Engineering Copilot — Streamlit front-end.

A research prototype developed for a university Digital Twin project on
district cooling and building energy optimisation. This app ties together:

    data_processing.py   -> cleaning + core analytics
    anomaly_detection.py -> ML (IsolationForest) + rule-based diagnostics
    llm_engine.py         -> rule-based / optional cloud-LLM engineering report
    visualization.py      -> Plotly dashboard charts
    report_generator.py   -> Markdown / PDF export

Run with:  streamlit run app.py

Author: Digital Twin Research Lab (prototype)
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime

import data_processing as dp
import anomaly_detection as ad
import llm_engine as llm
import visualization as viz
import report_generator as rg


# ---------------------------------------------------------------------------
# Page config & branding
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HVAC Engineering Copilot | Digital Twin Research Lab",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --brand-primary: #0B3D91;
    --brand-accent: #2CA6A4;
}
.main-header {
    padding: 1.2rem 1.6rem;
    background: linear-gradient(90deg, #0B3D91 0%, #2CA6A4 100%);
    border-radius: 10px;
    color: white;
    margin-bottom: 1.2rem;
}
.main-header h1 { margin: 0; font-size: 1.6rem; }
.main-header p { margin: 0.2rem 0 0 0; opacity: 0.92; font-size: 0.92rem; }
.metric-card {
    background: #F7F9FB;
    border: 1px solid #E3E8EE;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
.badge-research {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    margin-top: 6px;
}
.footer-note {
    font-size: 0.78rem;
    color: #6C757D;
    margin-top: 2rem;
    border-top: 1px solid #E3E8EE;
    padding-top: 0.8rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="main-header">
        <h1>🧊 HVAC Engineering Copilot</h1>
        <p>AI-assisted building operations analysis for district cooling &amp; energy optimisation</p>
        <span class="badge-research">DIGITAL TWIN RESEARCH LAB — PROTOTYPE v0.1</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — data upload & configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("📥 Data Input")
    building_name = st.text_input("Building / Dataset name", value="Demo Building A")
    uploaded_file = st.file_uploader("Upload building operational CSV", type=["csv"])

    st.markdown("---")
    st.subheader("⚙️ Analysis Settings")
    humidity_threshold = st.slider("Humidity risk threshold (%)", 50, 90, 65)
    co2_threshold = st.slider("CO₂ air-quality threshold (ppm)", 700, 2000, 1000, step=50)
    contamination = st.slider("ML anomaly sensitivity (contamination)", 0.01, 0.15, 0.03, step=0.01)

    st.markdown("---")
    st.subheader("🧠 AI Report Engine")
    provider_options = list(llm.PROVIDERS.keys())
    provider_labels = {k: v["label"] for k, v in llm.PROVIDERS.items()}
    provider_choice = st.selectbox(
        "Report generation engine",
        options=provider_options,
        format_func=lambda k: provider_labels[k],
        index=0,
    )

    provider_ready, provider_msg = llm.provider_status(provider_choice)
    if provider_ready:
        st.success(provider_msg)
    else:
        st.warning(provider_msg)
        cfg = llm.PROVIDERS[provider_choice]
        if cfg.get("signup_url"):
            st.caption(f"Get a free key: {cfg['signup_url']}")
        st.caption(
            "Add it under your app's **Settings → Secrets** on Streamlit "
            "Cloud (or in `.streamlit/secrets.toml` locally). The app works "
            "fully without this — the Rule-Based Engine needs no key."
        )

    model_override = None
    if provider_choice != "template":
        default_model = llm.get_effective_model(provider_choice)
        model_override = st.text_input(
            "Model name (optional override)",
            value="",
            placeholder=default_model,
            help=f"Leave blank to use the default: {default_model}. "
                 + (f"Full model list: {llm.PROVIDERS[provider_choice]['models_url']}"
                    if llm.PROVIDERS[provider_choice].get("models_url") else ""),
        ) or None

    st.markdown("---")
    st.caption(
        "This is a research prototype for a university-funded Digital Twin "
        "project, designed to run entirely in the browser via Streamlit "
        "Cloud. The default AI engine (Rule-Based) requires no API key. "
        "Cloud LLM providers are optional and opt-in via secrets."
    )


# ---------------------------------------------------------------------------
# Sample data helper (so reviewers can demo without their own CSV)
# ---------------------------------------------------------------------------

def _load_sample_data():
    return dp.load_csv("sample_data/sample_building_data.csv")


if uploaded_file is None:
    st.info(
        "👋 Upload a building operations CSV in the sidebar to begin, or "
        "explore the tool instantly with the bundled synthetic sample dataset below."
    )
    use_sample = st.button("▶ Load bundled sample dataset")
    if not use_sample:
        st.stop()
    raw_df = _load_sample_data()
else:
    raw_df = dp.load_csv(uploaded_file)

# --- Schema validation -------------------------------------------------
column_ambiguities = dp.get_column_ambiguities(raw_df)
required_ambiguities = {c: o for c, o in column_ambiguities.items() if c in dp.REQUIRED_COLUMNS}
if required_ambiguities:
    for canonical, originals in required_ambiguities.items():
        st.error(
            f"Your CSV has **{len(originals)} columns** that all look like "
            f"'{canonical.replace('_', ' ')}': {', '.join(repr(o) for o in originals)}. "
            "To avoid analysing the wrong sensor data, please keep only one "
            "of these columns (delete or rename the others) and re-upload."
        )
    st.stop()
# Ambiguities on optional fields (e.g. duplicate cooling-load columns) don't
# block the app — they're surfaced as warnings in the Data Cleaning tab
# instead, since that channel simply won't be used in the analysis.

missing_cols = dp.validate_schema(raw_df)
if missing_cols:
    st.error(
        f"The uploaded file is missing required columns: {', '.join(missing_cols)}. "
        f"Required columns are: {', '.join(dp.REQUIRED_COLUMNS)}."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Pipeline execution (cached per-session)
# ---------------------------------------------------------------------------

with st.spinner("Cleaning data and computing analytics..."):
    clean_df, cleaning_report = dp.clean_data(raw_df)
    stats = dp.compute_summary_statistics(clean_df)
    daily_energy = dp.compute_daily_energy(clean_df)
    peak_cooling = dp.compute_peak_cooling_periods(clean_df)
    occupancy_df = dp.estimate_occupancy(clean_df)
    comfort_df = dp.compute_comfort_index(clean_df)
    humidity_risk = dp.compute_humidity_risk_periods(clean_df, threshold=humidity_threshold)

with st.spinner("Running anomaly detection (IsolationForest + rule-based checks)..."):
    ml_df = ad.run_isolation_forest(clean_df, contamination=contamination)
    rule_events = ad.run_all_rule_based_checks(clean_df)
    anomaly_summary = ad.summarize_anomalies(rule_events, ml_df)

comfort_summary = {
    "avg_comfort_index": round(comfort_df["comfort_index"].mean(), 1),
    "pct_time_excellent": round((comfort_df["comfort_rating"] == "Excellent").mean() * 100, 1),
    "pct_time_poor": round((comfort_df["comfort_rating"] == "Poor").mean() * 100, 1),
}

st.session_state["pipeline_ready"] = True


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_cleaning, tab_analytics, tab_anomalies, tab_ai, tab_export = st.tabs(
    ["📊 Overview", "🧹 Data Cleaning", "📈 Analytics", "🚨 Anomalies", "🧠 AI Engineer Report", "📤 Export"]
)

# --- Overview -------------------------------------------------------------
with tab_overview:
    st.subheader(f"Dataset Overview — {building_name}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Indoor Temp", f"{stats['avg_indoor_temp']} °C")
    c2.metric("Avg Humidity", f"{stats['avg_humidity']} %")
    c3.metric("Avg CO₂", f"{stats['avg_co2']} ppm")
    c4.metric("Total Energy", f"{stats['total_energy_kwh']:,.0f} kWh")
    c5.metric("Avg Comfort Index", f"{comfort_summary['avg_comfort_index']}")

    st.markdown(f"**Monitoring period:** {stats['period_start']} → {stats['period_end']} "
                f"&nbsp;|&nbsp; **Records:** {stats['n_records']:,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_temperature(clean_df), use_container_width=True)
        st.plotly_chart(viz.plot_co2(clean_df, threshold=co2_threshold), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_humidity(clean_df, threshold=humidity_threshold), use_container_width=True)
        st.plotly_chart(viz.plot_comfort_index(comfort_df), use_container_width=True)

# --- Data Cleaning ----------------------------------------------------------
with tab_cleaning:
    st.subheader("Data Cleaning Report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows In", cleaning_report.rows_in)
    c2.metric("Rows Out", cleaning_report.rows_out)
    c3.metric("Duplicates Removed", cleaning_report.duplicates_removed)
    c4.metric("Timestamp Issues", cleaning_report.timestamp_issues)

    if cleaning_report.sensor_errors_flagged:
        st.markdown("**Sensor errors detected & nulled (then interpolated):**")
        st.table(pd.DataFrame(
            list(cleaning_report.sensor_errors_flagged.items()),
            columns=["Channel", "Flagged Readings"]
        ))
    else:
        st.success("No out-of-range sensor readings detected.")

    if cleaning_report.missing_values_filled:
        st.markdown("**Missing values filled via time-aware interpolation:**")
        st.table(pd.DataFrame(
            list(cleaning_report.missing_values_filled.items()),
            columns=["Channel", "Missing Values Filled"]
        ))
    else:
        st.success("No missing values detected.")

    if cleaning_report.notes:
        for note in cleaning_report.notes:
            st.warning(note)

    with st.expander("View cleaned data (first 200 rows)"):
        st.dataframe(clean_df.head(200), use_container_width=True)

# --- Analytics --------------------------------------------------------------
with tab_analytics:
    st.subheader("Building Performance Analytics")

    st.plotly_chart(viz.plot_energy(daily_energy), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Peak Cooling Demand Periods**")
        st.dataframe(peak_cooling, use_container_width=True)
    with col_b:
        st.markdown("**Sustained High-Humidity Risk Periods**")
        if humidity_risk.empty:
            st.success(f"No sustained periods above {humidity_threshold}% RH detected.")
        else:
            st.dataframe(humidity_risk, use_container_width=True)

    st.markdown("**Occupancy Estimation (CO₂-derived heuristic)**")
    occ_counts = occupancy_df["occupancy_band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    st.bar_chart(occ_counts)
    st.caption(
        "Occupancy is estimated from CO₂ build-up above an outdoor baseline "
        "(~420 ppm) using a simplified single-zone mass-balance heuristic. "
        "This is a research approximation, not a calibrated occupancy count."
    )

    numeric_cols = [c for c in [dp.INDOOR_TEMP_COL, dp.HUMIDITY_COL, dp.CO2_COL,
                                 dp.OUTDOOR_TEMP_COL, dp.ENERGY_COL, dp.COOLING_LOAD_COL]
                     if c in clean_df.columns]
    st.plotly_chart(viz.plot_correlation_heatmap(clean_df, numeric_cols), use_container_width=True)

    with st.expander("Full statistics (raw dict)"):
        st.json(stats)

# --- Anomalies ----------------------------------------------------------------
with tab_anomalies:
    st.subheader("AI Anomaly Detection")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rule-based events", anomaly_summary["total_rule_based_events"])
    c2.metric("ML-flagged anomalies", anomaly_summary["ml_anomalies_detected"])
    c3.metric("ML anomaly rate", f"{anomaly_summary['ml_anomaly_rate_pct']}%")

    st.plotly_chart(viz.plot_anomalies_overlay(clean_df, ml_df), use_container_width=True)

    st.markdown("**Rule-based diagnostic events**")
    if rule_events.empty:
        st.success("No rule-based anomalies detected in this dataset.")
    else:
        st.dataframe(rule_events, use_container_width=True, height=320)
        st.markdown("**Event breakdown by type**")
        st.bar_chart(rule_events["issue"].value_counts())

# --- AI Engineer Report -------------------------------------------------------
with tab_ai:
    st.subheader("AI HVAC Engineer — Narrative Report")

    if provider_choice == "template":
        st.caption(
            "Generated by the built-in Rule-Based Engine — runs entirely inside "
            "this app, no external calls, no API key needed."
        )
    else:
        st.caption(
            f"Generated via {llm.PROVIDERS[provider_choice]['label']}. Building "
            "statistics (not raw sensor data) are sent to this cloud API — avoid "
            "this mode for confidential datasets if that's a concern."
        )

    if not provider_ready:
        st.error(
            f"{provider_msg} Select a different engine in the sidebar, or add the "
            "required secret to enable this provider."
        )
    else:
        button_label = (
            "🧠 Generate Engineering Report"
            if provider_choice == "template"
            else f"🧠 Generate Report via {llm.PROVIDERS[provider_choice]['label']}"
        )
        generate_clicked = st.button(button_label, type="primary")

        if generate_clicked:
            events_preview = rule_events.head(15).to_dict(orient="records") if not rule_events.empty else []
            humidity_risk_preview = humidity_risk.to_dict(orient="records") if not humidity_risk.empty else []

            spinner_msg = (
                "Running rule-based analysis..."
                if provider_choice == "template"
                else f"Querying {llm.PROVIDERS[provider_choice]['label']}..."
            )
            with st.spinner(spinner_msg):
                try:
                    report = llm.generate_report(
                        building_stats=stats,
                        anomaly_summary=anomaly_summary,
                        rule_based_events_preview=events_preview,
                        comfort_summary=comfort_summary,
                        humidity_risk_periods=humidity_risk_preview,
                        provider=provider_choice,
                        model_override=model_override,
                    )
                    st.session_state["ai_report"] = report
                except RuntimeError as e:
                    st.error(str(e))

        if "ai_report" in st.session_state:
            report = st.session_state["ai_report"]
            sections = report["sections"]

            health = sections.get("Building Health Assessment", "")
            if "critical" in health.lower():
                st.error(f"**Building Health:** {health.splitlines()[0]}")
            elif "needs attention" in health.lower():
                st.warning(f"**Building Health:** {health.splitlines()[0]}")
            else:
                st.success(f"**Building Health:** {health.splitlines()[0] if health else 'N/A'}")

            for section_title in llm.REPORT_SECTIONS:
                with st.expander(f"**{section_title}**", expanded=(section_title == "Executive Summary")):
                    st.markdown(sections.get(section_title, "_Not provided._"))

            with st.expander("View raw model output"):
                st.text(report["raw_text"])

# --- Export --------------------------------------------------------------------
with tab_export:
    st.subheader("Export Report")

    if "ai_report" not in st.session_state:
        st.info("Generate the AI Engineer Report in the previous tab before exporting.")
    else:
        report = st.session_state["ai_report"]
        sections = report["sections"]
        model_name = report["model"]

        md_text = rg.generate_markdown_report(stats, sections, model_name, building_name)
        pdf_bytes = rg.generate_pdf_report(stats, sections, model_name, building_name)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
        safe_name = "".join(c if c.isalnum() else "_" for c in building_name.lower())

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇ Download Markdown Report",
                data=md_text,
                file_name=f"hvac_report_{safe_name}_{timestamp_str}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇ Download PDF Report",
                data=pdf_bytes,
                file_name=f"hvac_report_{safe_name}_{timestamp_str}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with st.expander("Preview Markdown report"):
            st.markdown(md_text)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="footer-note">
        HVAC Engineering Copilot — Digital Twin Research Lab research prototype.
        Built with Python, Streamlit, Pandas, Plotly, and scikit-learn. Runs entirely
        in the browser via Streamlit Cloud — the default report engine is a built-in
        rule-based system (no API key needed); optional cloud LLM providers (Groq,
        Hugging Face, or any OpenAI-compatible API) can be enabled via secrets.
        For research and demonstration purposes only — not a certified engineering tool.
    </div>
    """,
    unsafe_allow_html=True,
)
