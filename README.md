# INDIRECT EVAPORATIVE COOLING AND DIGITAL TWINS FOR BUILDING EFFICIENCY 

**A browser-based research prototype for a university-funded Digital Twin
project on district cooling and building energy optimisation.**

This tool ingests building operational data (indoor temperature, humidity,
CO₂, outdoor temperature, cooling load, chilled-water temperatures, and
energy consumption), cleans it, computes engineering analytics, runs AI-based
anomaly detection, and generates a structured engineering report — executive
summary, building health assessment, probable causes, recommendations,
maintenance actions, suggested additional sensors, and a confidence level.

> **Status:** Early-stage research prototype, designed to demonstrate the
> feasibility of an AI-assisted diagnostic layer that could later be
> integrated into a full Digital Twin platform. Not a certified engineering
> or safety tool — outputs should always be reviewed by a qualified engineer
> before any operational action is taken.

---

## What changed from the local (Ollama) version?

This edition is built specifically to run on **Streamlit Community Cloud**
(`share.streamlit.io`), which cannot host a local LLM server such as Ollama.
The AI report layer now works like this:

| Report Engine | Requires an API key? | Runs where? |
|---|---|---|
| **Rule-Based Engine** *(default)* | ❌ No | Entirely inside the Streamlit app — zero setup |
| **Groq Cloud** *(optional)* | ✅ Yes (free tier) | Groq's cloud API, serving open-weight models |
| **Hugging Face Inference Providers** *(optional)* | ✅ Yes (free tier) | HF's cloud router, serving open-weight models |
| **Custom OpenAI-compatible API** *(optional)* | ✅ Yes | Any endpoint you point it at (OpenRouter, Together AI, self-hosted, etc.) |

**The app is fully functional the moment you deploy it — with zero secrets
configured.** The Rule-Based Engine produces a genuine, statistics-grounded
engineering report (health verdict, causes, recommendations, maintenance
actions, sensor suggestions, confidence level) using deterministic logic
built on top of the same analytics pipeline — no external model call at all.
Adding a cloud LLM provider is entirely optional, for richer/more
free-form narrative writing.

All other functionality — CSV upload & cleaning, analytics, Plotly dashboards,
IsolationForest + rule-based anomaly detection, and Markdown/PDF export — is
unchanged and requires no configuration.

---

## Project structure

```
hvac_copilot/
├── app.py                          # Streamlit front-end / orchestration
├── data_processing.py              # CSV loading, cleaning, core analytics
├── anomaly_detection.py            # IsolationForest + rule-based diagnostics
├── llm_engine.py                    # Rule-based engine + optional cloud LLM providers
├── visualization.py                 # Plotly chart builders
├── report_generator.py              # Markdown & PDF report export
├── generate_sample_data.py          # Synthetic demo dataset generator
├── sample_data/
│   └── sample_building_data.csv     # Bundled demo dataset (14 days, 15-min intervals)
├── .streamlit/
│   ├── config.toml                  # Theming
│   └── secrets.toml.example         # Template for optional cloud LLM keys
├── requirements.txt
└── README.md
```

---

## Deploy to Streamlit Community Cloud (share.streamlit.io)

### Step 1 — Push this project to a GitHub repository

```bash
cd hvac_copilot
git init
git add .
git commit -m "Initial commit: HVAC Engineering Copilot (Streamlit Cloud edition)"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> ⚠️ Do **not** commit a real `.streamlit/secrets.toml` file with live API
> keys. Only `.streamlit/secrets.toml.example` (which contains placeholder
> text, not real keys) should go into the repository. Add a `.gitignore`
> entry for `.streamlit/secrets.toml` if you create one locally.

### Step 2 — Create the app on Streamlit Cloud

1. Go to **<https://share.streamlit.io>** and sign in (GitHub login is the
   easiest option).
2. Click **"New app"**.
3. Choose:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy"**.

That's it — Streamlit Cloud will install everything in `requirements.txt`
and launch the app. The **Rule-Based Engine** works immediately with no
further configuration. Give the build 1–3 minutes on first deploy.

### Step 3 — (Optional) Enable a cloud LLM provider

If you'd like AI-narrative reports written by an actual LLM instead of the
rule-based engine:

1. On your app's page in Streamlit Cloud, click **⋮ (menu) → Settings → Secrets**.
2. Paste in **one** of the blocks below (see `.streamlit/secrets.toml.example`
   for the full annotated version), then click **Save**. The app will
   automatically restart with the new secrets available.

**Option A — Groq Cloud** (fast, generous free tier):
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
Get a free key at <https://console.groq.com/keys>. Current model list:
<https://console.groq.com/docs/models> (the app defaults to
`llama-3.1-8b-instant`; you can override the model name in the app's sidebar
if Groq renames or retires a model).

**Option B — Hugging Face Inference Providers** (free tier):
```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
Get a free token at <https://huggingface.co/settings/tokens> (enable the
"Make calls to Inference Providers" permission on the token).

**Option C — Any OpenAI-compatible API** (OpenRouter, Together AI, your own
hosted endpoint, etc.):
```toml
CUSTOM_API_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
CUSTOM_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
CUSTOM_MODEL = "meta-llama/llama-3.1-8b-instruct"
```

3. Reload the app, open the sidebar, and switch **"Report generation
   engine"** to your chosen provider. The sidebar will show a green
   confirmation once the secret is detected.

You can configure more than one provider at once and switch between them
live from the sidebar dropdown — useful for comparing rule-based vs.
LLM-generated narrative style during a research demo.

---

## Running locally (optional)

You don't need Ollama or any local LLM server for this edition. To run it
on your own machine before deploying:

```bash
cd hvac_copilot
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

To use a cloud LLM provider locally, copy `.streamlit/secrets.toml.example`
to `.streamlit/secrets.toml` and fill in real values — Streamlit will pick
it up automatically.

---

## Using the app

1. **Upload data** — In the sidebar, upload a CSV with the required columns
   (see schema below), or click **"Load bundled sample dataset"** on the
   main screen to explore immediately with synthetic data.
2. **Overview tab** — Key metrics and headline charts (temperature,
   humidity, CO₂, comfort index).
3. **Data Cleaning tab** — Transparency into what was cleaned: duplicates
   removed, sensor errors nulled & interpolated, missing values filled.
4. **Analytics tab** — Daily energy consumption, peak cooling periods,
   sustained humidity-risk windows, CO₂-derived occupancy estimation, and a
   sensor correlation matrix.
5. **Anomalies tab** — Combined view of ML (IsolationForest) and
   rule-based anomaly detections (high humidity, CO₂ spikes, ventilation
   failure signatures, cooling inefficiency, sensor faults).
6. **AI Engineer Report tab** — Click **"Generate Engineering Report"** to
   produce a structured narrative report using whichever engine is selected
   in the sidebar.
7. **Export tab** — Download the report as Markdown or PDF.

---

## Expected CSV schema

| Column                              | Required? | Notes                                   |
|--------------------------------------|-----------|------------------------------------------|
| Timestamp                            | ✅ Yes    | Any common date/time format              |
| Indoor Temperature                   | ✅ Yes    | °C                                       |
| Relative Humidity                    | ✅ Yes    | %                                        |
| CO2                                   | ✅ Yes    | ppm                                       |
| Outdoor Temperature                   | ✅ Yes    | °C                                       |
| Cooling Load                          | Optional  | kW                                        |
| Chilled Water Supply Temperature      | Optional  | °C                                       |
| Chilled Water Return Temperature      | Optional  | °C                                       |
| Energy Consumption                    | ✅ Yes    | kWh                                       |

Column names are matched flexibly (case-insensitive, common synonyms
recognised — see `COLUMN_ALIASES` in `data_processing.py`), so exports from
typical Building Management Systems (BMS) will usually work without manual
renaming.

---

## Methodology notes (for research transparency)

- **Comfort index** is a simplified 0–100 heuristic combining deviation
  from a temperature setpoint, humidity setpoint, and a CO₂ penalty. It is
  *inspired by* ASHRAE 55 comfort concepts but is **not** a certified
  PMV/PPD calculation.
- **Occupancy estimation** uses a single-zone CO₂ mass-balance heuristic
  (build-up above an outdoor baseline of ~420 ppm). This is a relative
  indicator, not a calibrated headcount.
- **Anomaly detection** combines an unsupervised `IsolationForest` (catches
  multivariate "unusual state" patterns) with explainable, engineering
  rule-based checks (humidity thresholds, CO₂ thresholds, sustained
  ventilation-failure signatures, chilled-water delta-T "low delta-T
  syndrome" for cooling-plant inefficiency, and sensor flat-line detection).
- **Rule-Based Engine report**: every statement is generated from
  conditional logic tied directly to a specific computed statistic or
  detected event — fully traceable, fully deterministic, and free to run at
  any scale.
- **Cloud LLM reports** (optional): the model is given *pre-computed
  statistics*, never raw time series, so its role is engineering
  interpretation rather than arithmetic — reducing the risk of numerical
  hallucination. Exact wording will vary between providers/models.

---

## Known limitations (research-prototype scope)

- Single-building / single-zone analysis only; no multi-building or
  district-level aggregation yet.
- No persistence layer — each session re-processes the uploaded CSV from
  scratch (a natural next step for a full Digital Twin integration would be
  a time-series database backend).
- Optional cloud LLM providers depend on third-party model availability and
  free-tier rate limits, which can change; the sidebar lets you override the
  model name if a provider renames or retires a model.
- Occupancy and comfort-index calculations are simplified research
  heuristics, not certified building-physics models.
- On Streamlit Community Cloud's free tier, apps that are idle for a while
  go to sleep and take a few seconds to wake up on the next visit — this is
  normal platform behaviour, not an app bug.

---

## License & attribution

This is an academic research prototype built for demonstration purposes.
Core dependencies (Streamlit, pandas, NumPy, Plotly, scikit-learn, ReportLab)
are open-source. Cloud LLM providers, if enabled, are third-party services
subject to their own terms and free-tier limits.
