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

`HF_RESTART_TOKEN` (optional Space secret) powers the sidebar's *Wake up the API*
button, which any visitor can press when the API Space is asleep. Use a
fine-grained token with write access to `KLEB38/OC_P8` only.

## Try the API from the dashboard

The sidebar's *Try the API* buttons send a real `/predict` call (low-risk
profile, high-risk profile, or a random client) so visitors can watch it land
in the tabs. One call per 10 s per session. Payloads come from
`static/demo_profiles.json`: the two curated profiles from `examples/` plus
200 known `SK_ID_CURR`. No raw application row is exported; the random client
reuses a curated profile and re-draws its main drivers (`demo.py`).

```powershell
uv run python scripts/build_demo_profiles.py   # rebuild the artefact
```

## Regenerate the drift report

```powershell
uv run python scripts/generate_drift_report.py --days 30
```

The resulting HTML is written to `dashboard/static/drift_report.html` and
picked up by the dashboard automatically.
