"""
llm_engine.py  (Streamlit Cloud edition)
=========================================
Report-generation layer for the HVAC Engineering Copilot.

IMPORTANT — this version does NOT use a local Ollama/Llama server, because
Streamlit Community Cloud cannot run background local LLM processes. Instead
it offers a choice of "providers":

  1. "template"  — a built-in, deterministic, rule-based report generator.
                   Pure Python, zero dependencies, zero API keys, zero cost,
                   and works instantly on Streamlit Cloud with no setup.
                   This is the DEFAULT provider, so the app is always fully
                   functional even if nobody configures any secrets.

  2. "groq"       — optional cloud LLM via Groq's free-tier API (OpenAI-
                   compatible), serving open-weight models such as Llama 3.1.
  3. "huggingface"— optional cloud LLM via Hugging Face's Inference Providers
                   router (OpenAI-compatible), also serving open models.
  4. "custom"     — optional: point at ANY OpenAI-compatible chat-completions
                   endpoint (OpenRouter, Together AI, a self-hosted vLLM
                   server, etc.) by supplying a base URL + API key + model
                   name via Streamlit secrets.

All cloud providers are opt-in via Streamlit secrets / environment variables
— nothing is called unless the person running the app has configured a key.

Author: Digital Twin Research Lab (prototype)
"""

from __future__ import annotations

import json
import os
import re
import requests

# ---------------------------------------------------------------------------
# Section schema (shared by template + LLM providers + report_generator.py)
# ---------------------------------------------------------------------------

REPORT_SECTIONS = [
    "Executive Summary",
    "Building Health Assessment",
    "Possible Causes",
    "Engineering Recommendations",
    "Suggested Maintenance Actions",
    "Suggested Additional Sensors",
    "Confidence Level",
]

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Each cloud provider is described declaratively so the Streamlit sidebar can
# render status/setup instructions without hardcoding provider-specific logic.

PROVIDERS = {
    "template": {
        "label": "Rule-Based Engine (offline, no key needed)",
        "requires_key": False,
    },
    "groq": {
        "label": "Groq Cloud (fast, free-tier, open-weight models)",
        "requires_key": True,
        "secret_names": ["GROQ_API_KEY"],
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.1-8b-instant",
        "signup_url": "https://console.groq.com/keys",
        "models_url": "https://console.groq.com/docs/models",
    },
    "huggingface": {
        "label": "Hugging Face Inference Providers (free tier)",
        "requires_key": True,
        "secret_names": ["HF_TOKEN"],
        "base_url": "https://router.huggingface.co/v1/chat/completions",
        "default_model": "meta-llama/Llama-3.1-8B-Instruct",
        "signup_url": "https://huggingface.co/settings/tokens",
        "models_url": "https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation",
    },
    "custom": {
        "label": "Custom OpenAI-compatible API (OpenRouter, Together AI, etc.)",
        "requires_key": True,
        "secret_names": ["CUSTOM_API_BASE_URL", "CUSTOM_API_KEY", "CUSTOM_MODEL"],
        "signup_url": None,
        "models_url": None,
    },
}


# ---------------------------------------------------------------------------
# Secrets / config helpers
# ---------------------------------------------------------------------------

def get_secret(key: str) -> str | None:
    """
    Look up a config value from Streamlit secrets first, then environment
    variables. Safe to call even when no secrets.toml exists at all (e.g.
    a first-time Streamlit Cloud deployment with nothing configured yet).
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


def provider_status(provider_key: str) -> tuple[bool, str]:
    """
    Return (is_ready, message) for a given provider — whether the required
    secrets are present, so the UI can show a clear ready/not-ready state.
    """
    provider = PROVIDERS[provider_key]
    if not provider["requires_key"]:
        return True, "Ready — runs locally in this app, no API key required."

    missing = [name for name in provider["secret_names"] if not get_secret(name)]
    if missing:
        return False, f"Missing secret(s): {', '.join(missing)}."
    return True, "API key detected — ready to generate reports."


def get_effective_model(provider_key: str, override: str | None = None) -> str:
    """Resolve which model name to use: explicit override > secret > default."""
    if override:
        return override
    provider = PROVIDERS[provider_key]
    if provider_key == "custom":
        return get_secret("CUSTOM_MODEL") or "gpt-3.5-turbo"
    return provider.get("default_model", "")


# ---------------------------------------------------------------------------
# Prompt construction (shared across all cloud providers)
# ---------------------------------------------------------------------------

def build_prompt(building_stats: dict, anomaly_summary: dict,
                  rule_based_events_preview: list, comfort_summary: dict,
                  humidity_risk_periods: list) -> str:
    """
    Construct the analyst prompt sent to the LLM. We hand the model a
    condensed, already-computed statistical summary (NOT the raw time
    series) so it reasons over engineering context rather than trying to
    do arithmetic itself — the numbers below come straight from our
    pandas/sklearn pipeline, and the LLM's job is interpretation, not
    calculation.
    """
    prompt = f"""You are a senior HVAC and building-energy engineer supporting a university
Digital Twin research project on district cooling systems. You have been given
pre-computed statistics from a building monitoring dataset. Your job is to
interpret these numbers as an experienced engineer would, and produce a
structured engineering report.

=== BUILDING OPERATIONAL STATISTICS ===
{json.dumps(building_stats, indent=2, default=str)}

=== COMFORT INDEX SUMMARY ===
{json.dumps(comfort_summary, indent=2, default=str)}

=== ANOMALY DETECTION SUMMARY (rule-based + ML) ===
{json.dumps(anomaly_summary, indent=2, default=str)}

=== SAMPLE OF FLAGGED EVENTS (up to 15 shown) ===
{json.dumps(rule_based_events_preview, indent=2, default=str)}

=== SUSTAINED HIGH-HUMIDITY RISK PERIODS ===
{json.dumps(humidity_risk_periods, indent=2, default=str)}

Using ONLY the information above, produce a report with EXACTLY these
Markdown section headers, in this order, each starting with "## ":

## Executive Summary
(3-5 sentences, plain engineering language suitable for a facilities manager)

## Building Health Assessment
(overall verdict: Good / Fair / Needs Attention / Critical, with justification)

## Possible Causes
(bulleted list of likely root causes for the anomalies observed)

## Engineering Recommendations
(bulleted, actionable, prioritized recommendations)

## Suggested Maintenance Actions
(bulleted, concrete maintenance tasks a facilities team could schedule)

## Suggested Additional Sensors
(bulleted list of sensors/instrumentation that would improve diagnostic confidence)

## Confidence Level
(state Low / Medium / High and briefly justify, considering data completeness and volume)

Be specific, concise, and avoid generic filler. Do not invent data that was not
provided above. Do not include any text before "## Executive Summary" or after
the Confidence Level section.
"""
    return prompt


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible chat-completions caller
# ---------------------------------------------------------------------------
# Groq, Hugging Face Inference Providers, OpenRouter, Together AI, and many
# others all implement the same request/response shape as OpenAI's chat
# completions endpoint. One function covers all of them — only the base URL,
# API key, and model name differ.

def _call_openai_compatible_api(base_url: str, api_key: str, model: str,
                                 prompt: str, temperature: float = 0.3,
                                 timeout: int = 60) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    try:
        resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Could not connect to {base_url}. Check the base URL and your network.")
    except requests.exceptions.Timeout:
        raise RuntimeError("The API request timed out. Try again, or choose a smaller/faster model.")

    if resp.status_code == 401:
        raise RuntimeError("Authentication failed (401). Check that your API key is correct and active.")
    if resp.status_code == 404:
        raise RuntimeError(
            f"Model or endpoint not found (404) for model='{model}'. "
            "It may have been renamed/retired — check the provider's current model list."
        )
    if resp.status_code == 429:
        raise RuntimeError("Rate limit reached (429). Wait a moment and try again, or switch providers.")
    if resp.status_code >= 400:
        raise RuntimeError(f"API returned an error ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response shape: {e}. Raw response: {json.dumps(data)[:300]}")


# ---------------------------------------------------------------------------
# Rule-based / template report generator (default, always available)
# ---------------------------------------------------------------------------

def _generate_template_report(building_stats: dict, anomaly_summary: dict,
                               rule_based_events_preview: list, comfort_summary: dict,
                               humidity_risk_periods: list) -> str:
    """
    Deterministic, explainable engineering report built entirely from the
    statistics our own pipeline already computed — no external model call.
    This is intentionally conservative: every statement is traceable back to
    a specific number or detected event, which makes it well-suited as the
    zero-config default for a research demo.
    """
    issue_types = list(anomaly_summary.get("by_issue_type", {}).keys())
    total_events = anomaly_summary.get("total_rule_based_events", 0)
    ml_rate = anomaly_summary.get("ml_anomaly_rate_pct", 0.0)
    avg_comfort = comfort_summary.get("avg_comfort_index", 100)
    pct_poor_comfort = comfort_summary.get("pct_time_poor", 0.0)
    n_humidity_risk_periods = len(humidity_risk_periods)

    has_humidity_issue = any("humidity" in i.lower() for i in issue_types)
    has_co2_issue = any("co2" in i.lower() for i in issue_types)
    has_ventilation_issue = any("ventilation" in i.lower() for i in issue_types)
    has_cooling_inefficiency = any("inefficiency" in i.lower() or "delta-t" in i.lower() for i in issue_types)
    has_sensor_fault = any("sensor fault" in i.lower() for i in issue_types)

    # --- Health verdict -----------------------------------------------------
    if avg_comfort < 40 or ml_rate > 15 or has_ventilation_issue and total_events > 50:
        health = "Critical"
        health_reason = (
            f"average comfort index of {avg_comfort} and a {ml_rate}% multivariate anomaly "
            "rate indicate the building is frequently operating well outside normal bounds."
        )
    elif avg_comfort < 60 or total_events > 30 or n_humidity_risk_periods > 0 or has_ventilation_issue:
        health = "Needs Attention"
        health_reason = (
            f"recurring diagnostic events ({total_events} rule-based flags, {ml_rate}% ML anomaly rate) "
            "point to at least one system that is not performing as designed."
        )
    elif avg_comfort < 80 or total_events > 5:
        health = "Fair"
        health_reason = (
            f"comfort index averaging {avg_comfort} and a modest number of flagged events "
            f"({total_events}) suggest generally acceptable but not optimal operation."
        )
    else:
        health = "Good"
        health_reason = (
            f"comfort index averaging {avg_comfort} with very few flagged events ({total_events}) "
            "indicates the building is operating within expected engineering bounds."
        )

    # --- Executive summary ---------------------------------------------------
    exec_summary = (
        f"Over the monitored period ({building_stats.get('n_records', 'N/A')} records from "
        f"{building_stats.get('period_start', 'N/A')} to {building_stats.get('period_end', 'N/A')}), "
        f"the building averaged {building_stats.get('avg_indoor_temp', 'N/A')}°C indoor temperature and "
        f"{building_stats.get('avg_humidity', 'N/A')}% relative humidity, consuming "
        f"{building_stats.get('total_energy_kwh', 'N/A'):,} kWh in total. "
        f"The diagnostic layer flagged {total_events} rule-based events and an ML-based anomaly rate of "
        f"{ml_rate}%. Overall building health is assessed as **{health}**. "
    )
    if has_cooling_inefficiency:
        exec_summary += "A recurring low chilled-water delta-T signature suggests reduced cooling-plant effectiveness during at least part of the period. "
    if has_ventilation_issue:
        exec_summary += "Sustained high-CO2 periods suggest fresh-air ventilation may not be keeping pace with occupancy at times. "
    if n_humidity_risk_periods > 0:
        exec_summary += f"{n_humidity_risk_periods} sustained high-humidity period(s) were identified, which carries condensation/mould risk if left unaddressed."

    # --- Possible causes ------------------------------------------------------
    causes = []
    if has_cooling_inefficiency:
        causes.append("Reduced chilled-water coil heat-transfer effectiveness (fouling, airflow restriction, or coil bypass)")
        causes.append("Chilled-water control valve not fully modulating, or a stuck/failing actuator")
    if has_ventilation_issue or has_co2_issue:
        causes.append("Fresh-air damper undersized, stuck closed, or economizer/DCV control not responding to occupancy")
        causes.append("AHU fan speed or schedule mismatched to actual occupancy patterns")
    if has_humidity_issue:
        causes.append("Insufficient dehumidification capacity relative to latent load, especially during humid outdoor conditions")
        causes.append("Cooling coil oversized for sensible load, causing short cycling with limited moisture removal")
    if has_sensor_fault:
        causes.append("One or more sensors reporting stuck/flat-lined values, indicating possible wiring, calibration, or transmitter failure")
    if not causes:
        causes.append("No significant anomaly patterns were detected; observed variation is consistent with normal operational cycling")

    # --- Recommendations --------------------------------------------------------
    recommendations = []
    if has_cooling_inefficiency:
        recommendations.append("Inspect and clean chilled-water coils; verify strainers are not fouled")
        recommendations.append("Verify chilled-water control valve calibration and full-stroke operation")
    if has_ventilation_issue or has_co2_issue:
        recommendations.append("Audit fresh-air damper position and economizer/DCV (demand-controlled ventilation) setpoints against current occupancy")
        recommendations.append("Verify AHU fan operating schedule aligns with the building's actual occupancy pattern")
    if has_humidity_issue:
        recommendations.append("Review dehumidification strategy — consider reheat, a dedicated outdoor air system (DOAS), or coil bypass factor adjustments")
    if has_sensor_fault:
        recommendations.append("Physically inspect flagged sensors for wiring faults, and recalibrate or replace as needed")
    if not recommendations:
        recommendations.append("Continue routine monitoring; no urgent corrective action indicated by current data")
    recommendations.append("Re-run this analysis after any corrective action to confirm the anomaly signature has cleared")

    # --- Maintenance actions -------------------------------------------------
    maintenance = []
    if has_cooling_inefficiency:
        maintenance.append("Schedule chilled-water coil cleaning within the next maintenance cycle")
        maintenance.append("Add chilled-water delta-T to the routine weekly BMS trend review")
    if has_ventilation_issue or has_co2_issue:
        maintenance.append("Schedule damper actuator inspection and lubrication/replacement as needed")
    if has_humidity_issue:
        maintenance.append("Inspect condensate drain and dehumidification components for scale/blockage")
    if has_sensor_fault:
        maintenance.append("Add flagged sensors to the next calibration/PM (preventive maintenance) round")
    if not maintenance:
        maintenance.append("Maintain current preventive maintenance schedule; no additional actions triggered by this dataset")

    # --- Suggested additional sensors --------------------------------------
    additional_sensors = []
    if "cooling_load_kw" not in str(building_stats):
        pass  # placeholder — real check happens via column presence upstream
    if has_cooling_inefficiency:
        additional_sensors.append("Dedicated chilled-water flow meter (enables direct BTU/cooling-capacity calculation, not just delta-T)")
    if has_ventilation_issue or has_co2_issue:
        additional_sensors.append("Outdoor/supply air CFM (airflow) sensors at the AHU to confirm ventilation rate independent of CO2 inference")
    if has_humidity_issue:
        additional_sensors.append("Dedicated dew-point or absolute-humidity sensor to separate latent load issues from sensible cooling issues")
    additional_sensors.append("Zone-level occupancy counters (e.g. PIR or camera-based) to validate the CO2-derived occupancy estimate in this report")

    # --- Confidence level -----------------------------------------------------
    n_records = building_stats.get("n_records", 0)
    if n_records >= 1000:
        confidence = f"High — the assessment is based on {n_records} records, giving good statistical coverage of the monitored period."
    elif n_records >= 200:
        confidence = f"Medium — {n_records} records were available, sufficient for directional conclusions but a longer monitoring window would improve confidence."
    else:
        confidence = f"Low — only {n_records} records were available, which limits confidence in the patterns identified."

    def bullets(items):
        return "\n".join(f"- {item}" for item in items)

    report = f"""## Executive Summary
{exec_summary}

## Building Health Assessment
**{health}** — {health_reason}

## Possible Causes
{bullets(causes)}

## Engineering Recommendations
{bullets(recommendations)}

## Suggested Maintenance Actions
{bullets(maintenance)}

## Suggested Additional Sensors
{bullets(additional_sensors)}

## Confidence Level
{confidence}
"""
    return report


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(building_stats: dict, anomaly_summary: dict,
                     rule_based_events_preview: list, comfort_summary: dict,
                     humidity_risk_periods: list,
                     provider: str = "template",
                     model_override: str | None = None,
                     temperature: float = 0.3) -> dict:
    """
    Generate the structured engineering report using the selected provider.
    Returns {"raw_text", "sections", "provider", "model"}.

    Raises RuntimeError with a user-friendly message on any failure so the
    Streamlit UI layer can display it gracefully.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid options: {list(PROVIDERS.keys())}")

    if provider == "template":
        raw_text = _generate_template_report(
            building_stats, anomaly_summary, rule_based_events_preview,
            comfort_summary, humidity_risk_periods,
        )
        model_name = "rule-based-engine-v1"
    else:
        ready, message = provider_status(provider)
        if not ready:
            raise RuntimeError(message)

        prompt = build_prompt(building_stats, anomaly_summary,
                               rule_based_events_preview, comfort_summary,
                               humidity_risk_periods)

        if provider == "custom":
            base_url = get_secret("CUSTOM_API_BASE_URL")
            api_key = get_secret("CUSTOM_API_KEY")
            model_name = get_effective_model("custom", model_override)
            if not base_url:
                raise RuntimeError("CUSTOM_API_BASE_URL secret is not set.")
        else:
            provider_cfg = PROVIDERS[provider]
            base_url = provider_cfg["base_url"]
            api_key = get_secret(provider_cfg["secret_names"][0])
            model_name = get_effective_model(provider, model_override)

        raw_text = _call_openai_compatible_api(
            base_url=base_url, api_key=api_key, model=model_name,
            prompt=prompt, temperature=temperature,
        )

    sections = parse_report_sections(raw_text)
    return {"raw_text": raw_text, "sections": sections, "provider": provider, "model": model_name}


def parse_report_sections(raw_text: str) -> dict:
    """
    Split Markdown output into a {section_title: body_text} dict based on
    the '## Header' pattern used by both the template engine and the prompt
    given to cloud LLMs. Falls back gracefully if a model didn't follow the
    format perfectly.
    """
    pattern = r"^##\s+(.+?)\s*$"
    lines = raw_text.splitlines()

    sections = {}
    current_title = None
    buffer = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            if current_title is not None:
                sections[current_title] = "\n".join(buffer).strip()
            current_title = match.group(1).strip()
            buffer = []
        else:
            buffer.append(line)
    if current_title is not None:
        sections[current_title] = "\n".join(buffer).strip()

    for expected in REPORT_SECTIONS:
        sections.setdefault(expected, "_Not provided by model._")

    return sections
