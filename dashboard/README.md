---
title: OC P8 Monitoring
emoji: 📊
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# OC P8 — Monitoring Dashboard

Streamlit dashboard for the Credit Scoring API. Reads logs from Neon
(`predictions_log`) and an Evidently HTML report embedded in `static/`.

## Run locally

```powershell
$env:DATABASE_URL = "postgresql://..."
uv run streamlit run dashboard/app.py
```

## Build & test the container locally

```powershell
docker build -t oc-p8-monitoring -f dashboard/Dockerfile dashboard
docker run -p 7860:7860 -e DATABASE_URL="postgresql://..." oc-p8-monitoring
```

## Deploy to HF Space `KLEB38/OC_P8_monitoring`

```powershell
$env:HF_TOKEN = "hf_..."
uv run python scripts/deploy_dashboard.py
```

`DATABASE_URL` must be added as a Space secret (Settings → Variables and
secrets → New secret).

## Regenerate the drift report

```powershell
uv run python scripts/generate_drift_report.py --days 30
```

The resulting HTML is written to `dashboard/static/drift_report.html` and
picked up by the dashboard automatically.
