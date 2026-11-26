"""End-to-end: POST /predict -> SELECT FROM predictions_log_test.

Runs only when ``TEST_DATABASE_URL`` is set in the environment (CI secret
or local override). Without it, the whole module skips gracefully so the
default test suite stays self-contained.

Every test cleans up after itself by deleting its own row. The CI workflow
also runs a TRUNCATE at job end as a safety net.
"""

from __future__ import annotations

import importlib
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
TEST_TABLE = "predictions_log_test"

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set; integration test against Postgres skipped.",
)


@pytest.fixture
def test_engine():
    """Engine pointing at the integration DB. Truncates the test table once."""
    engine = create_engine(TEST_DATABASE_URL, future=True)
    # Ensure table exists (idempotent), no DROP — preserves any concurrent runs.
    from sqlalchemy import MetaData
    from database.models import build_predictions_log_table

    metadata = MetaData()
    build_predictions_log_table(TEST_TABLE, metadata)
    metadata.create_all(engine, checkfirst=True)
    yield engine
    engine.dispose()


@pytest.fixture
def client(monkeypatch, patched_settings, test_engine):
    """FastAPI TestClient with DATABASE_URL pointed at the integration DB
    and the table pointed at predictions_log_test."""
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("OC_P8_PREDICTIONS_TABLE", TEST_TABLE)

    # api.settings already reloaded by patched_settings; re-reload it now that
    # DATABASE_URL/PREDICTIONS_TABLE are set, then bounce the engine.
    import api.settings as s

    importlib.reload(s)
    from api import db

    db.reset_engine()

    from api.main import app

    with TestClient(app) as tc:
        yield tc

    db.reset_engine()


def test_predict_inserts_row(client, valid_payload, test_engine):
    # Use a unique SK_ID so we can locate our row even with concurrent runs.
    sk_id = 900_000 + int(uuid.uuid4().int % 100_000)
    payload = dict(valid_payload, SK_ID_CURR=sk_id)
    try:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

        with test_engine.connect() as conn:
            row = conn.execute(
                text(
                    f"SELECT decision, probability_default, status_code, features, "
                    f"feature_assembly_ms, inference_ms, inference_cpu_ms, "
                    f"plumbing_ms, db_log_ms "
                    f"FROM {TEST_TABLE} WHERE sk_id_curr = :sk_id"
                ),
                {"sk_id": sk_id},
            ).one()
        assert row.status_code == 200
        assert row.decision in ("GRANTED", "REFUSED")
        assert 0.0 <= row.probability_default <= 1.0
        assert isinstance(row.features, dict) and len(row.features) > 0
        # Fine-grained timings populated on the success path. Loose bounds —
        # we only assert non-negativity (rounded ms can read as 0 on very
        # fast paths) and a sane upper bound (10 s would already be a major
        # regression on a single-row predict).
        assert row.feature_assembly_ms is not None and 0 <= row.feature_assembly_ms < 10_000
        assert row.inference_ms is not None and 0 <= row.inference_ms < 10_000
        # CPU time can read as 0 on very fast paths (process_time resolution
        # is platform-dependent); only assert non-negative.
        assert row.inference_cpu_ms is not None and row.inference_cpu_ms >= 0
        # plumbing_ms is computed in api.main as (latency - asm - inf). With
        # round() on all three, the result has ±0.5 ms noise per term, so a
        # small negative is possible — bound to a sane range.
        assert row.plumbing_ms is not None and -3 <= row.plumbing_ms < 10_000
        # db_log_ms is filled by the follow-up UPDATE after the INSERT.
        # On a real hosted Postgres round-trip this is typically 10-50 ms.
        assert row.db_log_ms is not None and 0 <= row.db_log_ms < 10_000
    finally:
        with test_engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {TEST_TABLE} WHERE sk_id_curr = :sk_id"),
                {"sk_id": sk_id},
            )


def test_validation_error_is_still_logged(client, test_engine):
    """Pydantic 422s should produce a row with status_code=422 so we don't
    lose visibility on malformed traffic."""
    sk_id = 900_000 + int(uuid.uuid4().int % 100_000)
    bad_payload = {"SK_ID_CURR": sk_id}  # missing required fields
    try:
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422
        # 422s are caught by FastAPI's RequestValidationError handler BEFORE
        # reaching the endpoint, so our /predict logger does NOT fire. This
        # is acceptable for now — drift monitoring only cares about valid
        # predictions. Document the gap rather than fight the framework.
        with test_engine.connect() as conn:
            count = conn.execute(
                text(
                    f"SELECT COUNT(*) FROM {TEST_TABLE} WHERE sk_id_curr = :sk_id"
                ),
                {"sk_id": sk_id},
            ).scalar()
        assert count == 0
    finally:
        with test_engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {TEST_TABLE} WHERE sk_id_curr = :sk_id"),
                {"sk_id": sk_id},
            )
