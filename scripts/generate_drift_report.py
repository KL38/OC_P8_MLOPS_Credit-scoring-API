"""Generate the Evidently data-drift HTML report.

Reads:
- Reference: ``data/reference_dataset.parquet`` (training sample, frozen)
- Current: last N production predictions from Neon, ordered by most
  recent (ignores history beyond that window)

The "last N" approach makes the pipeline replayable: re-run seed_traffic
and re-run this script — the report reflects the most recent batch
without needing to truncate the DB.

Writes an HTML report consumable by the Streamlit dashboard.

Usage:
    uv run python scripts/generate_drift_report.py                  # last 100
    uv run python scripts/generate_drift_report.py --limit 500
    uv run python scripts/generate_drift_report.py --output dashboard/static/drift_report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger("scripts.generate_drift_report")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_REFERENCE = Path("data/reference_dataset.parquet")
DEFAULT_OUTPUT = Path("dashboard/static/drift_report.html")
DEFAULT_FEATURE_NAMES = Path("models/feature_names.json")

# A scheduled job replays examples/low_risk.json every two days so the free-tier
# database is never paused for inactivity. Those rows are all the same applicant:
# counted as production traffic they would manufacture drift that says nothing
# about real usage, so they are excluded from the current window.
KEEP_ALIVE_SK_ID_CURR = 282751


def _load_reference(path: Path, feature_names: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    keep = [c for c in feature_names if c in df.columns]
    missing = set(feature_names) - set(df.columns)
    if missing:
        logger.warning(
            "%d feature(s) absent from reference dataset (drift skipped on those)",
            len(missing),
        )
    return df[keep].copy()


def _load_current(
    database_url: str, limit: int, feature_names: list[str]
) -> pd.DataFrame:
    """Pull the last ``limit`` successful predictions, ordered by most recent.

    Returns the JSONB feature payloads expanded into a wide DataFrame for
    Evidently. Newer rows come first in SQL, but order does not matter for
    drift (the test is distribution-based, not temporal).
    """
    engine = create_engine(database_url, future=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT features FROM predictions_log "
                "WHERE status_code = 200 "
                "AND sk_id_curr <> :keep_alive "
                "ORDER BY timestamp DESC "
                "LIMIT :lim"
            ),
            {"lim": limit, "keep_alive": KEEP_ALIVE_SK_ID_CURR},
        ).all()
    if not rows:
        raise SystemExit(
            "No successful prediction rows in predictions_log — generate some "
            "traffic first (scripts/seed_traffic.py)."
        )
    flat = pd.json_normalize([r.features for r in rows])
    keep = [c for c in feature_names if c in flat.columns]
    logger.info(
        "Loaded %d production rows (last %d requested), %d features available",
        len(flat),
        limit,
        len(keep),
    )
    return flat[keep].copy()


def _build_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Import Evidently lazily so the script still works when only inspecting.

    Returns the evaluation object produced by ``Report.run``. The 0.7+ API
    no longer mutates ``report`` in place — ``.run()`` returns a separate
    snapshot that carries ``save_html``.
    """
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise SystemExit(
            "evidently not installed. Add it to dashboard/requirements.txt "
            "or `uv add --group dev 'evidently>=0.7'`."
        ) from exc

    report = Report([DataDriftPreset()])
    # Evidently emits dozens of "divide by zero" RuntimeWarnings from scipy
    # when statistical tests run on near-constant columns. These are benign
    # (the resulting NaN/inf is interpreted as "skip this feature" downstream)
    # and they drown out real log output. Silence them only inside this call.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        return report.run(reference, current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Compare against the last N successful predictions (default %(default)s).",
    )
    parser.add_argument("--feature-names", type=Path, default=DEFAULT_FEATURE_NAMES)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()

    if not args.database_url:
        # Allow loading from database/.env for local runs.
        try:
            from dotenv import load_dotenv

            load_dotenv(Path("database/.env"))
            args.database_url = os.getenv("DATABASE_URL")
        except ImportError:
            pass
    if not args.database_url:
        raise SystemExit("DATABASE_URL is required (env var or database/.env).")

    feature_names = json.loads(args.feature_names.read_text())
    reference = _load_reference(args.reference, feature_names)
    current = _load_current(args.database_url, args.limit, feature_names)

    # Align columns: drop any column not in BOTH frames (Evidently expects parity)
    common = [c for c in reference.columns if c in current.columns]
    reference = reference[common]
    current = current[common]
    logger.info("Aligned on %d common features", len(common))

    # Evidently rejects all-NaN columns ("empty column ... was provided for drift
    # calculation"). Drop any column that is fully null on either side.
    empty_ref = {c for c in reference.columns if reference[c].isna().all()}
    empty_cur = {c for c in current.columns if current[c].isna().all()}
    empty = empty_ref | empty_cur
    if empty:
        sample = ", ".join(sorted(empty)[:5])
        logger.warning(
            "Dropping %d all-NaN column(s) (empty in ref=%d, in current=%d). Sample: %s%s",
            len(empty),
            len(empty_ref),
            len(empty_cur),
            sample,
            " ..." if len(empty) > 5 else "",
        )
        reference = reference.drop(columns=list(empty))
        current = current.drop(columns=list(empty))
    logger.info("Comparing on %d features after NaN filter", len(reference.columns))

    evaluation = _build_report(reference, current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluation.save_html(str(args.output))
    logger.info("Saved drift report to %s", args.output)

    # Also dump a JSON snapshot of the run so the dashboard can parse
    # per-feature drift scores without scraping the HTML.
    json_path = args.output.with_suffix(".json")
    try:
        json_payload = evaluation.json()
        if not isinstance(json_payload, str):
            import json as _json

            json_payload = _json.dumps(json_payload)
        json_path.write_text(json_payload, encoding="utf-8")
        logger.info("Saved drift report JSON to %s", json_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save JSON snapshot of the drift report: %s", exc)


if __name__ == "__main__":
    main()
