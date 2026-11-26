# Test strategy

Back to the [README](../README.md).

```powershell
# All tests + coverage report
uv run pytest --cov=api --cov-report=term-missing

# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# A specific module
uv run pytest tests/unit/test_ratios.py -v
```

Current coverage: **98 %** across 45 tests. Coverage gate enforced at **80 %** in CI.

---

## No real data required

All tests run **without the real 5 GB training data**. The `conftest.py` fixture
`synthetic_artefacts_dir` generates a complete but minimal artefact set in
`tmp_path` at test time:

- A 2-row `features_store.parquet` (clients 100002 and 100003)
- Minimal JSON vocabularies (13 categorical columns, 3 binary columns)
- A `FakeModel` whose probability is deterministically driven by
  `AMT_INCOME_TOTAL / AMT_CREDIT`, so tests can exercise both `GRANTED` and
  `REFUSED` branches reliably
- A realistic 121-field `VALID_PAYLOAD` dict (SK_ID_CURR + 120 raw inputs)
  reused across all test modules

The `patched_settings` fixture redirects all `api.settings` paths to `tmp_path`
via env vars + `importlib.reload()`, so no monkey-patching of internal state is
needed.

---

## Unit tests

| File | Module tested | What is verified |
|------|--------------|-----------------|
| `test_ratios.py` | `api/ratios.py` | 5 ratio formulas (known values, division-by-zero → NaN, NaN propagation, input immutability) |
| `test_inputs_transform.py` | `api/inputs_transform.py` | One-hot fix (all training categories emitted on a single row), binary factorization, `DAYS_EMPLOYED` sentinel → NaN, unknown category → all-zero |
| `test_predictor.py` | `api/predictor.py` | Threshold loaded from `model_info.json`, fallback to default, GRANTED/REFUSED boundary at exactly `threshold`, 1D and 3D prediction shape handling |
| `test_inference_assembler.py` | `api/inference_assembler.py` | Known client pulls parquet aggregates, unknown client uses template (counts=0, NaN), column order matches `feature_names`, inf values scrubbed to NaN |
| `test_schemas.py` | `api/schemas.py` | Pydantic range guards: negative income rejected, age < 18 and > 70 rejected, unknown contract type rejected, extra fields rejected, optional fields accept `null` |

---

## Integration tests

`tests/integration/test_api.py` boots the full FastAPI app via `TestClient` with
synthetic artefacts:

| Test | What is verified |
|------|-----------------|
| `test_health_endpoint` | `GET /health` returns 200 with `model_version` |
| `test_swagger_docs_available` | `GET /docs` returns 200 (brief requirement) |
| `test_openapi_schema` | `/openapi.json` exposes the `/predict` path |
| `test_model_info_endpoint` | `GET /model/info` returns threshold, version, n_features |
| `test_predict_known_client` | Full predict flow for a client in the feature store: `client_known=true`, decision in `{GRANTED, REFUSED}`, proba ∈ [0, 1] |
| `test_predict_unknown_client` | Unknown `SK_ID_CURR` → `client_known=false`, no crash |
| `test_predict_rejects_negative_income` | HTTP 422 on invalid input (Pydantic guard) |
| `test_predict_rejects_missing_required_field` | HTTP 422 when `DAYS_BIRTH` is absent |
| `test_predict_rejects_extra_field` | HTTP 422 on field injection attempt |
| `test_predict_rejects_unknown_contract_type` | HTTP 422 on out-of-vocabulary enum value |

---

## Tests and the database

The CI pipeline creates and truncates a dedicated `predictions_log_test` table in
the same Neon database as production. Those two steps are skipped when
`TEST_DATABASE_URL` is unset — which is the case on Dependabot PRs, since they do
not receive repository secrets. See the
[CI/CD Pipeline](../README.md#cicd-pipeline) section for the full job breakdown.
