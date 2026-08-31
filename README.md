[![Python][python-badge]][python-url]
[![FastAPI][fastapi-badge]][fastapi-url]
[![CI][ci-badge]][ci-url]
[![uv][uv-badge]][uv-url]
[![License: Internal][license-badge]](#license)

<br />
<div align="center">
  <h1 align="center">OC P8 — Credit Scoring API</h1>
  <p align="center">
    Production-grade FastAPI wrapper around the LightGBM credit scoring model
    trained in OC_P6. Built for a consumer-credit lender's express-loan
    department: real-time default risk prediction for loan officers.
    <br />
    <br />
    <a href="https://kleb38-oc-p8.hf.space/docs"><strong>Live API — Swagger UI »</strong></a>
    ·
    <a href="https://huggingface.co/spaces/KLEB38/OC_P8_monitoring"><strong>Monitoring dashboard »</strong></a>
    <br />
    <br />
    <a href="#try-it-in-30-seconds">Try it</a>
    ·
    <a href="#architecture">Architecture</a>
    ·
    <a href="#monitoring--data-drift">Monitoring</a>
    ·
    <a href="#cicd-pipeline">CI/CD</a>
    ·
    <a href="#running-it-locally">Run locally</a>
  </p>
</div>

---

## Table of Contents

- [Try it in 30 seconds](#try-it-in-30-seconds)
- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Architecture](#architecture)
- [Monitoring & Data Drift](#monitoring--data-drift)
- [Latency Optimisation](#latency-optimisation)
- [CI/CD Pipeline](#cicd-pipeline)
- [Running it locally](#running-it-locally)
- [Project Layout](#project-layout)
- [License](#license)
- [Links](#links)
- [Acknowledgments](#acknowledgments)

---

## Try it in 30 seconds

The API is deployed and live — nothing to install, nothing to clone.

| What | Where |
|------|-------|
| Interactive Swagger UI | [kleb38-oc-p8.hf.space/docs](https://kleb38-oc-p8.hf.space/docs) |
| Health check | [/health](https://kleb38-oc-p8.hf.space/health) |
| Model info | [/model/info](https://kleb38-oc-p8.hf.space/model/info) |
| Monitoring dashboard | [KLEB38/OC_P8_monitoring](https://huggingface.co/spaces/KLEB38/OC_P8_monitoring) |

Two ready-to-send payloads sit in [`examples/`](examples/) — 121 fields each,
valid against `api/schemas.py`:

```bash
# Low-risk applicant -> GRANTED
curl -X POST https://kleb38-oc-p8.hf.space/predict \
  -H "Content-Type: application/json" \
  -d @examples/low_risk.json
```

```json
{"sk_id_curr":282751,"probability_default":0.0193,"decision":"GRANTED",
 "threshold":0.33,"model_version":"2","client_known":true}
```

```bash
# High-risk applicant -> REFUSED
curl -X POST https://kleb38-oc-p8.hf.space/predict \
  -H "Content-Type: application/json" \
  -d @examples/high_risk.json
```

```json
{"sk_id_curr":244757,"probability_default":0.9337,"decision":"REFUSED",
 "threshold":0.33,"model_version":"2","client_known":true}
```

> **Cold start.** The Space sleeps after a period of inactivity. The first
> request may take ~30 s while the container wakes up; subsequent calls are in
> the low tens of milliseconds.

Three further profiles — a medium-risk applicant and two *unknown* clients that
exercise the `no_history_template` path — are documented in
[`examples/PROFILES.md`](examples/PROFILES.md).

---

## About The Project

<br />

The **Credit Scoring API** exposes a single `POST /predict` endpoint. Given a loan application (`SK_ID_CURR` + 120 raw `application_train` fields), it returns:

- `probability_default` — model score between 0 and 1
- `decision` — `"REFUSED"` if `proba ≥ 0.33`, `"GRANTED"` otherwise
- `threshold`, `model_version`, `client_known` — explainability metadata

The threshold **0.33** is optimised for an asymmetric cost function (10 × false negatives + false positives), meaning the model is intentionally conservative: missing a bad borrower costs 10× more than wrongly refusing a good one.

Beyond the endpoint itself, the project covers what happens once a model stops being a notebook: a CI/CD pipeline that ships it, a logging layer that records every decision, a drift report that watches the input distribution, and a profiling pass that took the request path from LightGBM to ONNX Runtime.

---

## Built With

[![Python][python-badge]][python-url]
[![FastAPI][fastapi-badge]][fastapi-url]
[![LightGBM][lightgbm-badge]][lightgbm-url]
[![ONNX][onnx-badge]][onnx-url]
[![uv][uv-badge]][uv-url]
[![Docker][docker-badge]][docker-url]
[![GitHub Actions][gha-badge]][gha-url]

---

## Architecture

```
JSON {SK_ID_CURR + 120 raw application_train fields}
        ▼
   Pydantic validation (121 ranged fields = SK_ID_CURR + 120 raw)
        ▼
   ┌─────────────────────────┐
   │ Known SK_ID_CURR ?      │
   └────┬────────────────┬───┘
   yes ▼                no ▼
  feature_store     no_history_template
  parquet lookup    (counts=0, NaN)
        │                  │
        └────────┬─────────┘
                 ▼
   transform app_train inputs (factorize + one-hot
   with training categories) + 5 derived ratios
                 ▼
   reindex to feature_names → 768 cols (float32 ndarray)
                 ▼
   ONNX Runtime InferenceSession (single-threaded)
                 ▼
   decision = proba ≥ 0.33  (business threshold optimised
                              for 10*FN + FP cost)
                 ▼
   {sk_id_curr, probability_default, decision,
    threshold, model_version, client_known}
                 ▼
   Supabase log via BackgroundTask (deferred — does not
   block the HTTP response on the success path)
```

**Two-case inference flow:**

| Case | Trigger | Aggregate source |
|------|---------|-----------------|
| **Known client** | `SK_ID_CURR` found in `features_store.parquet` | Pre-computed bureau / prev / POS / CC / install |
| **Unknown client** | `SK_ID_CURR` not found | `no_history_template.json` (counts=0, rest NaN) |

The unknown-client path preserves LightGBM's training-time NaN signal ("no historical data") rather than imputing fictitious medians.

### Data layer — code/data separation

The 235 MB `features_store.parquet` is **not bundled** in the Docker image. It lives in a companion HF Dataset repo (`KLEB38/oc-p8-features`) and is fetched at the API's first cold start via `huggingface_hub.hf_hub_download`, then cached on disk. This follows HF's recommended pattern: Spaces hold code, Datasets hold data.

| Layer | Repo | Content |
|-------|------|---------|
| Code + small artefacts | `KLEB38/OC_P8` (Space, Docker) | `api/`, `models/*.json`, `models/model.onnx` (served at runtime), `models/model.joblib` (kept for benchmark / drift checks) |
| Large data | `KLEB38/oc-p8-features` (Dataset) | `features_store.parquet` (235 MB, LFS) |

The local path (`data/features_store.parquet`) takes precedence — the HF download only fires when the file is absent (Space cold start). Configurable via `OC_P8_HF_DATASET_REPO_ID` and `OC_P8_HF_DATASET_FILENAME`.

---

## Monitoring & Data Drift

Every `/predict` call is logged to a Supabase PostgreSQL table (`predictions_log`),
and a Streamlit dashboard turns that table into production observability.

**Live dashboard: [KLEB38/OC_P8_monitoring](https://huggingface.co/spaces/KLEB38/OC_P8_monitoring)**

<div align="center">
  <img src="docs/monitoring.gif" width="900"
       alt="Monitoring dashboard: operational metrics, latency breakdown, drift report and business tabs" />
  <br />
  <em>The dashboard, recorded live — the four tabs below in motion.</em>
</div>

### The four tabs

| Tab | What it answers |
|-----|-----------------|
| **Operational** | Traffic volume, error rate, latency p50/p95 by hour, `probability_default` histogram split by decision |
| **Data Drift Report** | Embedded Evidently report. Watch *"Share of Drifted Features"* — above ~30 % typically warrants retraining or a threshold revision |
| **Business** | GRANTED / REFUSED ratio, known / unknown client mix, last 50 raw calls |
| **Advanced Data Drift** | Per-feature drill-down: which columns drifted, which test was used, current vs. reference distribution |

The drift tab reads a static `dashboard/static/drift_report.html` committed to the
repo, so it renders even when the database is unreachable. The other three query
Supabase live.

### Stack

| Component | Purpose | Location |
|-----------|---------|----------|
| Supabase Postgres | Storage for prediction logs | `database/` |
| `api/logger.py` | Deferred, best-effort insert — see [Deferred DB logging](#deferred-db-logging) | `api/` |
| Evidently | Generates the feature-drift HTML report | `scripts/generate_drift_report.py` |
| Streamlit dashboard | Reads Supabase + embeds the Evidently HTML | `dashboard/`, deployed at `KLEB38/OC_P8_monitoring` |

### Schema (`predictions_log`)

One row per request. Metadata in proper columns; the 121 raw inputs and
the 768 engineered features are kept in JSONB to absorb PostgreSQL's
63-char identifier limit and stay flexible if the feature pipeline evolves.

```
id (uuid) | timestamp (tz) | sk_id_curr | client_known | latency_ms
status_code | error_message | raw_input (jsonb) | features (jsonb)
probability_default | decision | threshold | model_version
top_shap (jsonb, nullable) | ground_truth (nullable)
```

Per-step timings (`feature_assembly_ms`, `inference_ms`, `inference_cpu_ms`,
`plumbing_ms`) are persisted on every row so the *Operational* tab can break
down latency by sub-step.

### Initial setup (once)

```powershell
# .env file: DATABASE_URL=postgresql://...   (gitignored, never commit)
uv run python -m database.setup --create
```

A separate table `predictions_log_test` is created in the same database for
CI/integration tests — production data is never polluted.

### Regenerate the drift report

```powershell
# 1. Build the frozen reference (10k stratified rows from training)
uv run python scripts/build_reference_dataset.py --upload

# 2. Compare last 30 days of prod vs reference
uv run python scripts/generate_drift_report.py --days 30
# -> dashboard/static/drift_report.html
```

### Run and deploy the dashboard

```powershell
# Locally
$env:DATABASE_URL = "postgresql://..."
cd dashboard && uv run streamlit run app.py   # http://localhost:8501

# Deploy to its own Space (the API pipeline deliberately does not ship dashboard/)
$env:HF_TOKEN = "hf_..."
uv run python scripts/deploy_dashboard.py
```

`DATABASE_URL` must be set as a Space secret on `KLEB38/OC_P8_monitoring`.
A read-only Supabase role is recommended.

---

## Latency Optimisation

Profiling showed that a `/predict` call was dominated by two hotspots: synchronous
Supabase logging on the request path, and a LightGBM single-row `predict_proba`
whose Python overhead dwarfed the actual tree traversal. Three independent
changes, each landed as its own PR.

### Deferred DB logging

The Supabase round-trip used to block the HTTP response. It now runs in a
FastAPI `BackgroundTask` on the success path, so the client gets the
`PredictionResponse` before the row is persisted. Failures still log
**synchronously** — `BackgroundTasks` are attached to the route's Response,
and the exception-handler chain builds its own Response and silently drops
pending tasks. Trading one-shot latency on failing requests for full error
observability is the right call.

See [`api/main.py`](api/main.py) (the `predict` handler's `finally:` block) and
[`api/logger.py`](api/logger.py).

### Tighter feature assembler

[`api/inference_assembler.py`](api/inference_assembler.py) was rewritten to
avoid redundant DataFrame allocations and column reindexing on the hot
path. The known/unknown branching now produces a single (1, 768) frame in
the canonical `feature_names` order without intermediate copies — the slow
part is no longer the join with the feature store but the upstream
`inputs_transform` pass, profiled via
[`scripts/profile_transform_lines.py`](scripts/profile_transform_lines.py).

### LightGBM to ONNX Runtime

[`scripts/export_to_onnx.py`](scripts/export_to_onnx.py) converts the
LightGBM `model.joblib` into `models/model.onnx` (with `zipmap=False` so
the second output is a plain `(n, 2)` probability matrix). At runtime,
[`api/predictor.py`](api/predictor.py) loads an `ort.InferenceSession`
once at lifespan and calls it on every request, replacing
`predict_proba`.

**Thread pinning fix.** ONNX Runtime defaults to
`intra_op_num_threads = num_cpus`, which on a shared HF Space VM contends
with pandas during feature assembly and *increases* end-to-end latency on
1-row inputs. The session is now built with
`intra_op_num_threads = inter_op_num_threads = 1` — single-threaded ONNX
on a single row is already in the microsecond range and leaves the rest
of the CPU budget for the assembler.

### Numerical drift caveat

ONNX runs in float32 while LightGBM runs in float64, so tree split
thresholds diverge marginally. Benchmark on 1 000 reference rows:

- `max |delta_proba|` ≈ **3.3e-03**
- 6 rows / 1 000 with `|delta| > 1e-5`

For most clients this is irrelevant, but **borderline clients with
`proba ∈ [0.325, 0.335]` may flip GRANTED ↔ REFUSED** vs. the original
`model.joblib`. The dashboard's *Business* tab is the right place to monitor
this — filter the proba band post-deploy and watch the GRANTED share.

### Benchmarks & drift checks

```powershell
# Latency + numerical equivalence vs LightGBM (writes a JSON report)
uv run python scripts/benchmark_onnx.py --n 1000 --out profiling/benchmark_onnx.json

# Quick proba-drift check on a fixed batch
uv run python scripts/check_onnx_drift.py

# End-to-end pipeline profiling (cProfile + pstats)
uv run python scripts/profile_predict.py
```

---

## CI/CD Pipeline

Defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (workflow name
`CI_CD`), running on **GitHub Actions**. Two sequential jobs.

```
push to main  /  pull request to main  /  manual dispatch
        │
        ▼
  ┌──────────────────┐
  │  build_and_test  │  lint + tests + coverage gate
  └────────┬─────────┘
           │ success  AND  event == push  AND  ref == main
           ▼
  ┌──────────────────┐
  │      deploy      │  push the repo to the HF Space
  └──────────────────┘
```

| Job | Runs on | Purpose |
|-----|---------|---------|
| `build_and_test` | Every push and PR to `main`, plus manual dispatch | Lint, tests, coverage gate |
| `deploy` | Only after `build_and_test` succeeds **on a push to `main`** | Ship the Space |

A pull request therefore gets the full test suite but never deploys, and a manual
`workflow_dispatch` run tests without shipping. Only a merge to `main` deploys.

### Job: `build_and_test`

1. **Checkout** with Git LFS (`actions/checkout@v6`)
2. **Set up Python 3.12** (`actions/setup-python@v6`)
3. **Install uv** pinned to `0.5.4` (`astral-sh/setup-uv@v8.1.0`)
4. **Install dependencies** — `uv sync --frozen` (lockfile respected, no version drift)
5. **Lint** — `ruff check api database feature_engineering tests`
6. **Ensure the Supabase test table exists** — `database.setup --create-test`.
   Skipped when `TEST_DATABASE_URL` is absent, which is the case on Dependabot
   PRs since they do not receive repository secrets.
7. **Run tests with coverage** — `pytest --cov=api --cov-fail-under=80`.
   The **80 %** gate fails the pipeline below that threshold.
8. **Truncate the test table** — always runs, even after a failure, so the shared
   database never accumulates test rows.
9. **Upload `coverage.xml`** as an artifact (30-day retention, always runs).

Any failing step blocks `deploy`.

> **Note.** Steps 6 and 8 talk to the same Supabase instance as production
> (different table). A free-tier database paused for inactivity will therefore
> fail the pipeline on the next push to `main`.

### Job: `deploy`

Checks out with full history and calls `HfApi.upload_folder()` to push the repo
to the `KLEB38/OC_P8` Space, ignoring `data/`, `dashboard/`, `.github/`, `.venv/`
and caches. Hugging Face rebuilds the Docker container automatically once the
Space repo is updated. The dashboard has its own deploy path
(`scripts/deploy_dashboard.py`) precisely because it is excluded here.

This README is excluded too. A Space is configured through the YAML
front-matter of its `README.md`, which GitHub renders as a metadata table — so
the card lives in `.hf/README.md` and a follow-up `upload_file()` pushes it to
the Space as `README.md`. Editing the card means editing `.hf/README.md`.

### Required secrets

Configure in **GitHub → Settings → Secrets and variables → Actions**:

| Secret | Used by | Description |
|--------|---------|-------------|
| `HF_TOKEN` | `deploy` | Hugging Face write token ([settings/tokens](https://huggingface.co/settings/tokens)), used by `upload_folder()` |
| `TEST_DATABASE_URL` | `build_and_test` | Supabase connection string for the **`predictions_log_test`** table |
| `DATABASE_URL` | Dashboard Space runtime | Read-only Supabase string, set as a **Space secret** on `KLEB38/OC_P8_monitoring` — not in GitHub |

---

## Running it locally

> **Heads-up.** Rebuilding this project from scratch requires the
> [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
> dataset (~5 GB), laid out as the sibling project `OC_P6`. If you only want to
> see the API work, use the [live deployment](#try-it-in-30-seconds) instead — it
> needs nothing.

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12 | [python.org](https://www.python.org/downloads/) |
| uv | latest | `pip install uv` |
| Docker | any | [docs.docker.com](https://docs.docker.com/get-docker/) |
| OC_P6 data | — | `~/OC_P6/data/` — Kaggle Home Credit CSVs |

### One-time offline setup

Generate all runtime artefacts (feature store parquet + metadata JSONs):

```powershell
uv sync
uv run python scripts/build_feature_store.py
uv run python scripts/build_no_history_template.py
```

This creates:

| Artefact | Size | Description |
|----------|------|-------------|
| `data/features_store.parquet` | ~200 MB | Pre-computed bureau / prev / POS / CC / install aggregates |
| `models/feature_names.json` | ~30 KB | Canonical 768-column order |
| `models/app_train_columns.json` | ~50 KB | Spec for the 122 Kaggle CSV columns (SK_ID_CURR + TARGET + 120 features) |
| `models/app_train_categories.json` | ~5 KB | Categorical vocabulary for one-hot encoding |
| `models/app_train_binary_mappings.json` | <1 KB | Factorize codes for binary columns |
| `models/no_history_template.json` | ~30 KB | Default values for unknown clients |
| `models/model.onnx` | ~2 MB | ONNX export of the LightGBM model (served at runtime) |

Re-export the ONNX model whenever `model.joblib` is refreshed:

```powershell
uv run python scripts/export_to_onnx.py
uv run python scripts/benchmark_onnx.py --n 1000   # sanity-check drift & latency
```

### Run the API

```powershell
uv run uvicorn api.main:app --reload
```

| Endpoint | URL |
|----------|-----|
| Swagger UI | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |
| Model info | http://127.0.0.1:8000/model/info |

The `examples/` payloads work against the local server too — only the host changes:

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d "@examples/low_risk.json"
```

### Docker

```powershell
docker build -t oc-p8-api .
docker run -p 7860:7860 oc-p8-api
curl http://127.0.0.1:7860/health
```

The image bundles only the code and the small JSON/joblib artefacts under
`models/`; the `Dockerfile` fails fast at build time if any is missing. The
235 MB `features_store.parquet` is **not** in the image — it is downloaded at
startup from the companion HF Dataset (see
[Data layer](#data-layer--codedata-separation)).

### Tests

```powershell
uv run pytest --cov=api --cov-report=term-missing
```

**98 % coverage across 45 tests**, with the CI gate at 80 %. Every test runs on
synthetic fixtures — no 5 GB training data required. The full breakdown (fixture
strategy, per-module unit tests, integration tests) is in
[`docs/TESTING.md`](docs/TESTING.md).

---

## Project Layout

```
api/                      # Runtime — bundled in Docker image
  main.py                 # FastAPI app + lifespan model loading + BackgroundTask logging
  predictor.py            # ONNX Runtime InferenceSession + threshold wrapper
  schemas.py              # Pydantic — 121 hand-crafted fields with ranges (SK_ID_CURR + 120 raw)
  inputs_transform.py     # Single-row app_train transform (one-hot fix)
  ratios.py               # 5 derived ratio formulas
  inference_assembler.py  # Branch known/unknown + reindex to 768 cols (optimised path)
  logger.py               # Best-effort Supabase insert with per-step latencies
  db.py                   # SQLAlchemy engine init/reset (lifespan-managed)
  settings.py             # Paths resolved from env vars with defaults

dashboard/                # Streamlit monitoring app — deployed to its own Space
  app.py                  # Four tabs: Operational, Drift, Business, Advanced Drift
  static/drift_report.html  # Committed Evidently report (renders without a DB)

examples/                 # Ready-to-send /predict payloads
  low_risk.json           # Known client, GRANTED
  high_risk.json          # Known client, REFUSED
  PROFILES.md             # Five documented profiles, known and unknown clients

feature_engineering/      # Offline ONLY — not imported by the API
  aggregations.py         # 5 aggregation funcs (bureau, prev, POS, CC, install)
  orchestrator.py         # merge_files() — full training dataframe build

scripts/                  # Offline maintenance scripts
  build_feature_store.py
  build_no_history_template.py
  export_model.py         # Imports model.joblib from OC_P6 MLflow registry
  export_to_onnx.py       # Converts LightGBM .joblib to ONNX (zipmap=False, float32 graph)
  benchmark_onnx.py       # p50/p95/p99 latency + numerical equivalence vs LightGBM
  check_onnx_drift.py     # Quick proba-drift check between LightGBM and ONNX
  profile_predict.py      # End-to-end profiling of the /predict pipeline
  profile_transform_lines.py  # Line-level profiling of inputs_transform / assembler
  build_reference_dataset.py  # Frozen 10k stratified drift reference
  generate_drift_report.py    # Evidently prod-vs-reference report
  deploy_dashboard.py     # Pushes dashboard/ to the monitoring Space
  smoke_test_model.py
  check_registry.py
  upload_data_to_hf.py    # One-shot upload of features_store.parquet to HF Dataset

tests/
  conftest.py             # Synthetic fixtures — no real data needed
  unit/                   # Per-module unit tests
  integration/            # FastAPI TestClient end-to-end tests

docs/
  TESTING.md              # Full test strategy and per-test breakdown

database/                 # Supabase schema setup (predictions_log + test table)
models/                   # model.joblib + JSON metadata (committed to git)
data/                     # features_store.parquet (gitignored — fetched from HF Dataset at runtime)
.github/workflows/
  ci.yml                  # CI_CD — build_and_test + deploy
.hf/
  README.md               # HF Space card — pushed as the Space's README.md by CI
Dockerfile
pyproject.toml
```

---

## License

Internal project — MLOps coursework.

---

## Links

| | |
|---|---|
| Source | [github.com/KL38/OC_P8_MLOPS_Credit-scoring-API](https://github.com/KL38/OC_P8_MLOPS_Credit-scoring-API) |
| Live API | [huggingface.co/spaces/KLEB38/OC_P8](https://huggingface.co/spaces/KLEB38/OC_P8) |
| Monitoring | [huggingface.co/spaces/KLEB38/OC_P8_monitoring](https://huggingface.co/spaces/KLEB38/OC_P8_monitoring) |
| Feature store | [huggingface.co/datasets/KLEB38/oc-p8-features](https://huggingface.co/datasets/KLEB38/oc-p8-features) |

---

## Acknowledgments

- [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk) — source of the training data
- [LightGBM](https://lightgbm.readthedocs.io/) · [ONNX Runtime](https://onnxruntime.ai/) · [FastAPI](https://fastapi.tiangolo.com/) · [Evidently](https://www.evidentlyai.com/) · [Supabase](https://supabase.com/) · [Hugging Face Spaces](https://huggingface.co/spaces)
- README structure inspired by [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template)

---

<!-- BADGE LINKS -->
[python-badge]: https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[fastapi-badge]: https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white
[fastapi-url]: https://fastapi.tiangolo.com/
[lightgbm-badge]: https://img.shields.io/badge/LightGBM-4.x-2E8B57?style=for-the-badge
[lightgbm-url]: https://lightgbm.readthedocs.io/
[onnx-badge]: https://img.shields.io/badge/ONNX%20Runtime-1.x-005CED?style=for-the-badge&logo=onnx&logoColor=white
[onnx-url]: https://onnxruntime.ai/
[uv-badge]: https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge
[uv-url]: https://docs.astral.sh/uv/
[docker-badge]: https://img.shields.io/badge/Docker-container-2496ED?style=for-the-badge&logo=docker&logoColor=white
[docker-url]: https://www.docker.com/
[gha-badge]: https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white
[gha-url]: https://github.com/features/actions
[ci-badge]: https://img.shields.io/github/actions/workflow/status/KL38/OC_P8_MLOPS_Credit-scoring-API/ci.yml?branch=main&style=for-the-badge&label=CI
[ci-url]: https://github.com/KL38/OC_P8_MLOPS_Credit-scoring-API/actions/workflows/ci.yml
[license-badge]: https://img.shields.io/badge/license-internal-lightgrey?style=for-the-badge
