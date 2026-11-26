"""Postgres read-only helpers + static monitoring artefacts loaders.

DB queries hit ``predictions_log`` (production data) and never touch the
test table. Static artefacts (proba_reference, feature_importance,
drift_report JSON snapshot) live under ``dashboard/static/`` and are
loaded once per session.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import Engine, create_engine, text

PROD_TABLE = "predictions_log"
STATIC_DIR = Path(__file__).parent / "static"

# Cap on rows returned by fetch_recent — the Business tab only renders the
# 50 most recent ones, and the proba histogram in the Operational tab is
# representative beyond a few thousand. Without a LIMIT, a 7-day window at
# production QPS would materialise tens of thousands of rows in memory.
RECENT_ROW_LIMIT = 5000


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        # Local dev convenience: try the repo's database/.env file. Silent
        # no-op on the HF Space where python-dotenv may not be installed and
        # the URL is configured via Space secrets.
        try:
            from dotenv import load_dotenv

            env_path = Path(__file__).resolve().parents[1] / "database" / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                url = os.getenv("DATABASE_URL")
        except ImportError:
            pass
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure it as a Space secret "
            "(read-only role recommended) or in database/.env for local dev."
        )
    # Name the driver explicitly: since SQLAlchemy 2.1 a bare postgresql:// URL
    # resolves to psycopg (v3), but this image only ships psycopg2.
    url = re.sub(r"^postgres(ql)?://", "postgresql+psycopg2://", url)
    return create_engine(
        url, pool_size=2, max_overflow=2, pool_pre_ping=True, future=True
    )


def _since(hours: int | None) -> datetime:
    """Lower bound of the time window; ``None`` means all time."""
    if hours is None:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc) - timedelta(hours=hours)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_recent(hours: int | None) -> pd.DataFrame:
    """Wide DataFrame of recent rows (capped at ``RECENT_ROW_LIMIT``).

    JSONB ``features`` stays as a Python dict. The cap prevents large
    windows on a busy day from materialising tens of thousands of rows in
    Streamlit memory — Business / Operational tabs only consume the latest
    few hundred anyway.
    """
    since = _since(hours)
    sql = text(
        f"""
        SELECT timestamp, sk_id_curr, client_known, latency_ms, status_code,
               error_message, probability_default, decision, threshold,
               model_version, features
        FROM {PROD_TABLE}
        WHERE timestamp >= :since
        ORDER BY timestamp DESC
        LIMIT :lim
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"since": since, "lim": RECENT_ROW_LIMIT})
    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_summary(hours: int | None) -> dict:
    """Aggregate KPIs computed in SQL to keep the dashboard responsive.

    Returns p50/p95 of total ``latency_ms`` plus the étape-4 breakdown
    (``feature_assembly_ms``, ``inference_ms``, ``inference_cpu_ms``). The
    breakdown columns are NULL on legacy rows logged before the
    instrumentation was added — PERCENTILE_CONT ignores NULLs natively.
    """
    since = _since(hours)
    sql = text(
        f"""
        SELECT
            COUNT(*)                                                AS total,
            SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END)      AS ok,
            SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END)     AS errors,
            SUM(CASE WHEN decision = 'GRANTED' THEN 1 ELSE 0 END)   AS granted,
            SUM(CASE WHEN decision = 'REFUSED' THEN 1 ELSE 0 END)   AS refused,
            SUM(CASE WHEN client_known = false THEN 1 ELSE 0 END)   AS unknowns,
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY feature_assembly_ms) AS asm_p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY feature_assembly_ms) AS asm_p95,
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY inference_ms)        AS inf_p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY inference_ms)        AS inf_p95,
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY inference_cpu_ms)    AS inf_cpu_p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY inference_cpu_ms)    AS inf_cpu_p95,
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY db_log_ms)           AS db_log_p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY db_log_ms)           AS db_log_p95,
            -- plumbing_ms is computed at log time in the API as
            -- (latency_ms - feature_assembly_ms - inference_ms). We read it
            -- directly here — no arithmetic in the dashboard SQL.
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY plumbing_ms)         AS plumbing_p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY plumbing_ms)         AS plumbing_p95,
            AVG(probability_default)                                AS avg_proba
        FROM {PROD_TABLE}
        WHERE timestamp >= :since
        """
    )
    with get_engine().connect() as conn:
        row = conn.execute(sql, {"since": since}).mappings().one()
    return dict(row)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_volume_by_hour(hours: int | None) -> pd.DataFrame:
    since = _since(hours)
    sql = text(
        f"""
        SELECT date_trunc('hour', timestamp) AS hour,
               COUNT(*)                       AS total,
               SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS errors,
               PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS p50,
               PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95
        FROM {PROD_TABLE}
        WHERE timestamp >= :since
        GROUP BY 1
        ORDER BY 1
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"since": since})


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latency_breakdown(hours: int | None) -> pd.DataFrame:
    """Hourly p50 of each timing component (étape 4).

    Returns columns: hour, total_p50, feature_assembly_p50, inference_p50,
    inference_cpu_p50, db_log_p50. NULL columns from legacy rows are
    skipped by PERCENTILE_CONT, so hours predating the instrumentation
    surface as NaN.
    """
    since = _since(hours)
    sql = text(
        f"""
        SELECT date_trunc('hour', timestamp) AS hour,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms)          AS total_p50,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY feature_assembly_ms) AS feature_assembly_p50,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY inference_ms)        AS inference_p50,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY inference_cpu_ms)    AS inference_cpu_p50,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY db_log_ms)           AS db_log_p50
        FROM {PROD_TABLE}
        WHERE timestamp >= :since AND status_code = 200
        GROUP BY 1
        ORDER BY 1
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"since": since})


@st.cache_data(ttl=60, show_spinner=False)
def fetch_proba_distribution(limit: int) -> list[float]:
    """Return the last ``limit`` successful prediction probabilities (most recent first)."""
    sql = text(
        f"""
        SELECT probability_default FROM {PROD_TABLE}
        WHERE status_code = 200 AND probability_default IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT :lim
        """
    )
    with get_engine().connect() as conn:
        rows = conn.execute(sql, {"lim": limit}).all()
    return [float(r.probability_default) for r in rows]


# ---------------------------------------------------------------------------
# Static artefacts (built offline by scripts/build_monitoring_artefacts.py
# and scripts/generate_drift_report.py). Cached for the lifetime of the
# Streamlit process — refresh by restarting the Space.
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_proba_reference() -> dict | None:
    """Histogram + raw samples of training-time probability_default."""
    return _load_json(STATIC_DIR / "proba_reference.json")


@st.cache_resource(show_spinner=False)
def load_feature_importance() -> dict | None:
    """Top-K features by SHAP mean(|value|) on the reference dataset."""
    return _load_json(STATIC_DIR / "feature_importance.json")


@st.cache_resource(show_spinner=False)
def load_drift_report_json() -> dict | None:
    """JSON snapshot saved alongside the HTML by generate_drift_report.py."""
    return _load_json(STATIC_DIR / "drift_report.json")


def parse_drift_results(report_json: dict | None) -> dict[str, dict[str, Any]]:
    """Parse the Evidently 0.7+ JSON snapshot.

    Schema observed in 0.7.x::

        {
          "metrics": [
            {
              "metric_name": "ValueDrift(column=AMT_INCOME_TOTAL,...)",
              "config": {"column": "AMT_INCOME_TOTAL", "method": "K-S p_value",
                          "threshold": 0.05, "type": "evidently:metric_v2:ValueDrift"},
              "value": 7.06e-07
            },
            {
              "metric_name": "DriftedColumnsCount(...)",
              "value": {"count": 40.0, "share": 0.052}
            },
            ...
          ],
          "tests": [...]
        }

    For each ValueDrift entry: drift detected when ``value < threshold``
    (the value is a p-value, lower = more drift).

    Returns: ``{feature_name: {"detected": bool, "score": float, "stattest": str}}``.
    """
    results: dict[str, dict[str, Any]] = {}
    if not report_json:
        return results

    for metric in report_json.get("metrics", []) or []:
        config = metric.get("config", {}) or {}
        # Only ValueDrift entries are per-column; skip the aggregate
        # DriftedColumnsCount and any other metric type.
        if config.get("type", "").endswith(":ValueDrift") is False:
            continue
        column = config.get("column")
        if not isinstance(column, str):
            continue
        threshold = float(config.get("threshold", 0.05))
        method = config.get("method") or "—"
        raw_value = metric.get("value")
        score: float | None
        detected: bool | None
        if isinstance(raw_value, (int, float)):
            score = float(raw_value)
            detected = score < threshold
        else:
            score = None
            detected = None
        results[column] = {
            "detected": detected,
            "score": score,
            "stattest": str(method),
        }
    return results


@st.cache_data(show_spinner=False)
def load_drift_summary(_report_json: dict | None) -> dict | None:
    """Extract the Evidently dataset-level drift verdict from the JSON.

    Reads the ``DriftedColumnsCount`` metric in the snapshot. Returns
    ``None`` if absent (older Evidently versions or empty report).
    """
    if not _report_json:
        return None
    for metric in _report_json.get("metrics", []) or []:
        config = metric.get("config", {}) or {}
        if config.get("type", "").endswith(":DriftedColumnsCount"):
            value = metric.get("value") or {}
            return {
                "count": int(value.get("count", 0)),
                "share": float(value.get("share", 0.0)),
                "threshold": float(config.get("drift_share", 0.5)),
            }
    return None
