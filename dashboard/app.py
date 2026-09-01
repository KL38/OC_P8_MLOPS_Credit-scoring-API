"""Credit Scoring Monitoring Dashboard.

Four tabs:
- Operational: volume, latency p50/p95, error rate, score distribution
- Drift: embedded Evidently HTML + summary
- Business: GRANTED vs REFUSED, top-driver features
- Advanced: output drift, critical features, weighted drift score

Reads from Supabase (predictions_log) — never touches the test table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats

from queries import (
    fetch_latency_breakdown,
    fetch_proba_distribution,
    fetch_recent,
    fetch_summary,
    fetch_volume_by_hour,
    load_drift_report_json,
    load_feature_importance,
    load_proba_reference,
    parse_drift_results,
)

logger = logging.getLogger(__name__)

DRIFT_REPORT_PATH = Path(__file__).parent / "static" / "drift_report.html"

st.set_page_config(
    page_title="OC P8 Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Credit Scoring — Monitoring")
st.caption("Prêt à Dépenser · prod observability + data drift")

with st.sidebar:
    st.header("Filters")
    hours = st.slider(
        "Window (hours)",
        min_value=1,
        max_value=168,
        value=168,
        help="Time range for every metric. 24h = 1 day, 168h = 7 days.",
    )
    st.markdown("---")
    st.markdown(
        "**Sources**\n\n"
        "- Logs: Supabase `predictions_log`\n"
        "- Drift: `static/drift_report.html`\n"
        "- Regenerate the report: `uv run python scripts/generate_drift_report.py`"
    )


# Fetched once and reused across the Operational and Business tabs. The
# @st.cache_data decorator on fetch_recent already deduplicates the DB
# round-trip, but computing the boolean mask twice would still cost two
# DataFrame allocations.
try:
    _recent_df = fetch_recent(hours)
    _ok_df = _recent_df[_recent_df["status_code"] == 200]
    DB_AVAILABLE = True
except Exception:
    # The connection string, host and driver internals belong in the Space
    # logs, never on screen: this page is public.
    logger.exception("predictions_log unreachable — rendering in degraded mode")
    _recent_df = pd.DataFrame()
    _ok_df = pd.DataFrame()
    DB_AVAILABLE = False

if not DB_AVAILABLE:
    st.warning(
        "**Degraded mode — log database unreachable.** The demo instance is most "
        "likely asleep. The *Data Drift Report* and *Advanced Data Drift* tabs "
        "remain available: they read frozen artefacts versioned with the "
        "application."
    )

tab_ops, tab_drift, tab_business, tab_advanced = st.tabs(
    ["⚙️ Operational", "🌊 Data Drift Report", "💼 Business", "🧠 Advanced Data Drift"]
)


# ------------------------------------------------------------------ Drift --
with tab_drift:
    st.subheader("Data Drift Report (Evidently)")
    if DRIFT_REPORT_PATH.exists():
        st.caption(f"Source: {DRIFT_REPORT_PATH.name}")
        html = DRIFT_REPORT_PATH.read_text(encoding="utf-8")
        st.components.v1.html(html, height=900, scrolling=True)
    else:
        st.info(
            "No Evidently report available. Generate it with:\n\n"
            "`uv run python scripts/generate_drift_report.py --days 30`\n\n"
            "Then redeploy the Space or copy the HTML into `dashboard/static/`."
        )


# --------------------------------------------------------------- Business --
with tab_business:
    if _recent_df.empty:
        st.warning("No data for this window.")
    else:
        ok = _ok_df
        c1, c2 = st.columns(2)
        with c1:
            decision_counts = ok["decision"].value_counts().reset_index()
            decision_counts.columns = ["decision", "count"]
            st.plotly_chart(
                px.pie(
                    decision_counts,
                    names="decision",
                    values="count",
                    title="Decisions",
                ),
                use_container_width=True,
            )
        with c2:
            known = (
                ok["client_known"]
                .value_counts()
                .rename({True: "Known", False: "Unknown"})
            )
            st.plotly_chart(
                px.pie(
                    pd.DataFrame({"type": known.index, "count": known.values}),
                    names="type",
                    values="count",
                    title="Known vs unknown clients",
                ),
                use_container_width=True,
            )

        st.subheader("Latest calls")
        st.dataframe(
            ok[
                [
                    "timestamp",
                    "sk_id_curr",
                    "client_known",
                    "probability_default",
                    "decision",
                    "latency_ms",
                    "model_version",
                ]
            ].head(50),
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------------- Advanced KPIs --
with tab_advanced:
    st.caption(
        "Advanced indicators beyond per-feature drift: model output drift, "
        "critical-feature tracking, and a drift score weighted by SHAP "
        "importance."
    )

    proba_ref = load_proba_reference()
    importance = load_feature_importance()
    drift_json = load_drift_report_json()
    drift_results = parse_drift_results(drift_json)

    # ---------------------------------------------------- Output drift --
    st.subheader("1. Output drift — probability_default distribution")
    if proba_ref is None:
        st.info(
            "`dashboard/static/proba_reference.json` not found. "
            "Generate it with `uv run python scripts/build_monitoring_artefacts.py`."
        )
    else:
        try:
            current_proba = fetch_proba_distribution(limit=500)
        except Exception:
            logger.exception("fetch_proba_distribution failed")
            st.info(
                "Production distribution unavailable (log database unreachable). "
                "The frozen reference is still shown below."
            )
            current_proba = []

        if not current_proba:
            st.warning("No logged prediction to compute the production distribution.")
        else:
            ref_values = np.array(proba_ref.get("values", []))
            cur_values = np.array(current_proba)

            # K-S test on raw samples — robust comparison of distributions.
            # scipy returns a KstestResult NamedTuple (statistic, pvalue); the
            # type stubs are weak, hence the ignore comment.
            ks_result = scipy_stats.ks_2samp(ref_values, cur_values)
            ks_p = float(ks_result.pvalue)  # type: ignore[attr-defined]
            detected = ks_p < 0.05

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reference mean", f"{ref_values.mean():.3f}")
            c2.metric(
                "Current mean",
                f"{cur_values.mean():.3f}",
                delta=f"{(cur_values.mean() - ref_values.mean()):+.3f}",
            )
            c3.metric("K-S p-value", f"{ks_p:.2e}")
            c4.metric(
                "Output drift",
                "✓ detected" if detected else "✗ stable",
                delta_color="inverse" if detected else "normal",
            )

            # Overlay histogram.
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=ref_values,
                    name="Reference (training)",
                    opacity=0.55,
                    nbinsx=40,
                    histnorm="probability",
                    marker_color="#888",
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=cur_values,
                    name=f"Current (last {len(cur_values)})",
                    opacity=0.7,
                    nbinsx=40,
                    histnorm="probability",
                    marker_color="#e74c3c",
                )
            )
            fig.update_layout(
                barmode="overlay",
                xaxis_title="probability_default",
                yaxis_title="density",
                title="Default probability distribution — reference vs current",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "The K-S test compares the shape of the two sample "
                "distributions. Model output drift is the most direct signal of "
                "altered model behaviour in production — it aggregates the "
                "effect of every input drift at once."
            )

    # ------------------------------------------------- Critical features --
    st.subheader("2. Critical features (top 10 SHAP)")
    if importance is None:
        st.info(
            "`dashboard/static/feature_importance.json` not found. "
            "Generate it with `uv run python scripts/build_monitoring_artefacts.py`."
        )
    elif not drift_results:
        st.info(
            "`dashboard/static/drift_report.json` not found. "
            "Regenerate the drift report with `uv run python scripts/generate_drift_report.py`."
        )
    else:
        top_n = 10
        rows = []
        for entry in importance["top"][:top_n]:
            feat = entry["feature"]
            imp = entry["importance"]
            result = drift_results.get(feat, {})
            detected = result.get("detected")
            score = result.get("score")
            stattest = result.get("stattest") or "—"
            rows.append(
                {
                    "Rank": entry["rank"],
                    "Feature": feat,
                    "SHAP importance": round(imp, 4),
                    "Drift": "🔴 Detected"
                    if detected
                    else ("🟢 Stable" if detected is False else "—"),
                    "Drift score": (f"{score:.4f}" if score is not None else "—"),
                    "Stat test": stattest,
                }
            )
        df_critical = pd.DataFrame(rows)

        n_drifted = sum(1 for r in rows if "Detected" in r["Drift"])
        c1, c2 = st.columns([1, 3])
        c1.metric(
            f"Drifted in top {top_n}",
            f"{n_drifted}/{top_n}",
            delta_color="inverse",
        )
        c2.caption(
            f"Method: {importance['method']} over {importance['sample_size']} "
            "reference rows. How many critical features have drifted is the most "
            "actionable indicator — drift on a top feature calls for retraining "
            "as a priority."
        )

        st.dataframe(df_critical, use_container_width=True, hide_index=True)

    # -------------------------------------------------- Weighted drift --
    st.subheader("3. Importance-weighted drift score")
    if importance is None or not drift_results:
        st.info(
            "Indicator unavailable until both `feature_importance.json` and "
            "`drift_report.json` are present."
        )
    else:
        total_importance = 0.0
        drifted_importance = 0.0
        n_features_seen = 0
        for entry in importance["top"]:
            feat = entry["feature"]
            imp = float(entry["importance"])
            total_importance += imp
            result = drift_results.get(feat)
            if result is None:
                continue
            n_features_seen += 1
            if result.get("detected"):
                drifted_importance += imp

        weighted_ratio = (
            (drifted_importance / total_importance) if total_importance > 0 else 0.0
        )
        threshold = 0.30

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Weighted drift",
            f"{weighted_ratio:.1%}",
            delta=f"threshold {threshold:.0%}",
            delta_color="inverse" if weighted_ratio >= threshold else "normal",
        )
        c2.metric(
            "Importance covered",
            f"{n_features_seen} / {len(importance['top'])} features",
        )
        c3.metric(
            "Verdict",
            "🔴 Alert" if weighted_ratio >= threshold else "🟢 OK",
        )

        st.caption(
            "**Formula**: Σ(importance × drift_detected) / Σ(importance) over the "
            f"top-{len(importance['top'])} SHAP features. Weights Evidently's "
            "binary verdict by how much each feature actually drives the model. "
            "Threshold: 30% of total importance drifting → alert. A finer signal "
            "than the raw ratio Evidently shows in the Data Drift tab."
        )


# -------------------------------------------------------------------- Ops --
# Rendered last on purpose: it is the only tab that cannot show anything
# without the database, and st.stop() halts the whole script run. Filling the
# file-backed tabs first keeps a database outage local to this one.
with tab_ops:
    try:
        summary = fetch_summary(hours)
    except Exception:
        logger.exception("fetch_summary failed")
        st.info(
            "Operational metrics unavailable — see the banner at the top of the "
            "page. The drift tabs remain available."
        )
        st.stop()

    if not summary["total"]:
        st.warning(f"No prediction logged over the last {hours} hours.")
        st.stop()

    # Headline: total server-side wall-clock = handler + DB log. The detail
    # decomposition lives in the dedicated section below.
    _total_p50_top = int(round(float(summary["p50"] or 0))) + int(
        round(float(summary["db_log_p50"] or 0))
    )
    _total_p95_top = int(round(float(summary["p95"] or 0))) + int(
        round(float(summary["db_log_p95"] or 0))
    )

    cols = st.columns(6)
    cols[0].metric("Total requests", f"{summary['total']:,}")
    cols[1].metric(
        "Errors",
        f"{summary['errors']:,}",
        delta=f"{(summary['errors'] / summary['total']) * 100:.1f} %",
        delta_color="inverse",
    )
    cols[2].metric(
        "Total p50",
        f"{_total_p50_top} ms",
        help="Full server wall-clock = handler (`latency_ms`) + DB log (`db_log_ms`). Details in the *Latency breakdown* section below.",
    )
    cols[3].metric(
        "Total p95",
        f"{_total_p95_top} ms",
        help="Server wall-clock p95 = handler p95 + DB log p95.",
    )
    cols[4].metric(
        "% REFUSED",
        f"{(summary['refused'] / max(summary['total'], 1)) * 100:.1f} %",
    )
    cols[5].metric(
        "% New clients",
        f"{(summary['unknowns'] / max(summary['total'], 1)) * 100:.1f} %",
        help="Share of clients with no entry in the feature store (no_history_template).",
    )

    st.subheader("Volume & latency per hour")
    hourly = fetch_volume_by_hour(hours)
    if not hourly.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(hourly, x="hour", y="total", title="Requests / hour"),
                use_container_width=True,
            )
        with c2:
            fig = px.line(
                hourly.melt(id_vars="hour", value_vars=["p50", "p95"]),
                x="hour",
                y="value",
                color="variable",
                title="Latency (ms)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latency breakdown")

    def _ms(v) -> int:
        """Format helper — round to int ms, default 0 when SQL returns NULL."""
        return 0 if v is None else int(round(float(v)))

    handler_p50 = _ms(summary["p50"])
    handler_p95 = _ms(summary["p95"])
    asm_p50 = _ms(summary["asm_p50"])
    asm_p95 = _ms(summary["asm_p95"])
    inf_p50 = _ms(summary["inf_p50"])
    inf_p95 = _ms(summary["inf_p95"])
    inf_cpu_p50 = _ms(summary["inf_cpu_p50"])
    inf_cpu_p95 = _ms(summary["inf_cpu_p95"])
    db_log_p50 = _ms(summary["db_log_p50"])
    db_log_p95 = _ms(summary["db_log_p95"])
    plumb_p50 = _ms(summary["plumbing_p50"])
    plumb_p95 = _ms(summary["plumbing_p95"])
    total_p50 = handler_p50 + db_log_p50
    total_p95 = handler_p95 + db_log_p95

    st.caption(
        f"**Client-perceived latency ≈ handler ({handler_p50} ms p50).** "
        f"The **DB log** ({db_log_p50} ms p50) runs in a `BackgroundTask` after "
        "the response is sent — it no longer affects the client (step 4).  \n"
        "The **handler** (`latency_ms`) covers assembly + inference + response "
        "building. The **DB log** (`db_log_ms`) is measured separately in "
        "`api/logger.py` around the Supabase INSERT, and is still shown as a "
        "server health metric. The **plumbing Δ** = `latency_ms - assembly - "
        "inference` isolates the Python residual between sub-measurements "
        "(variable init, return statement, entering the `finally`) — typically "
        "< 1 ms."
    )

    # Seven metrics sharing a 1/7 column each do not fit Streamlit's default
    # 2.25rem metric value: "3066 / 3066 ms" renders as "3066 / 3066…" (the
    # Metric component truncates rather than wraps). The token is not exposed
    # in config.toml, so the font size is overridden here — scoped to this row
    # through the container key, to leave the KPI row above at full size.
    st.html(
        """
        <style>
          .st-key-latency_perf_row [data-testid="stMetricValue"] {
            font-size: 1.7rem;
          }
        </style>
        """
    )
    # The columns are created inside the keyed container so the CSS above
    # matches them; the metrics below stay attached to those columns wherever
    # they are declared from.
    with st.container(key="latency_perf_row"):
        cols_perf = st.columns(7)

    cols_perf[0].metric(
        "Total p50 / p95",
        f"{total_p50} / {total_p95} ms",
        help="Full server wall-clock = `latency_ms` (handler) + `db_log_ms` (INSERT). The real time spent server-side on a request.",
    )
    cols_perf[1].metric(
        "Handler p50 / p95",
        f"{handler_p50} / {handler_p95} ms",
        help="`latency_ms` = assembly + inference + plumbing. **Excludes** the DB log.",
    )
    cols_perf[2].metric(
        "Feature assembly p50 / p95",
        f"{asm_p50} / {asm_p95} ms",
        help="Feature store lookup + transforms + ratios + reindex.",
    )
    cols_perf[3].metric(
        "Inference wall p50 / p95",
        f"{inf_p50} / {inf_p95} ms",
        help="`model.predict_proba` (wall-clock).",
    )
    cols_perf[4].metric(
        "Inference CPU p50 / p95",
        f"{inf_cpu_p50} / {inf_cpu_p95} ms",
        help="CPU time consumed during inference (can read 0 on very fast paths — `time.process_time` resolution).",
    )
    cols_perf[5].metric(
        "DB log p50 / p95",
        f"{db_log_p50} / {db_log_p95} ms",
        help="Supabase INSERT measured around `conn.execute(insert(...))` in `api/logger.py`. Usually dominates the total overhead.",
    )
    cols_perf[6].metric(
        "Plumbing Δ p50 / p95",
        f"{plumb_p50} / {plumb_p95} ms",
        help="`latency_ms - feature_assembly_ms - inference_ms`. Python residual between sub-measurements (typically < 1 ms).",
    )

    breakdown = fetch_latency_breakdown(hours)
    if not breakdown.empty:
        breakdown = breakdown.copy()
        # Plumbing per hour = handler - assembly - inference, clamped at 0 to
        # absorb sub-ms rounding artefacts. We then stack 4 components whose
        # total equals handler + db_log = full server wall-clock.
        breakdown["plumbing_p50"] = (
            breakdown["total_p50"].fillna(0)
            - breakdown["feature_assembly_p50"].fillna(0)
            - breakdown["inference_p50"].fillna(0)
        ).clip(lower=0)
        long_df = breakdown.melt(
            id_vars="hour",
            value_vars=[
                "feature_assembly_p50",
                "inference_p50",
                "plumbing_p50",
                "db_log_p50",
            ],
            var_name="component",
            value_name="ms",
        )
        long_df["component"] = long_df["component"].map(
            {
                "feature_assembly_p50": "Feature assembly",
                "inference_p50": "Model inference",
                "plumbing_p50": "Python plumbing (residual)",
                "db_log_p50": "DB log (Supabase INSERT)",
            }
        )
        fig_breakdown = px.area(
            long_df,
            x="hour",
            y="ms",
            color="component",
            title="p50 breakdown per hour (stacked = server wall-clock)",
        )
        fig_breakdown.update_layout(yaxis_title="p50 latency (ms)")
        st.plotly_chart(fig_breakdown, use_container_width=True)
    else:
        st.info(
            "No instrumented data on this window yet. "
            "Send traffic with `scripts/seed_traffic.py` after deploying the step-4 API."
        )

    st.subheader("Probability distribution")
    if not _ok_df.empty:
        st.plotly_chart(
            px.histogram(
                _ok_df,
                x="probability_default",
                nbins=40,
                color="decision",
                title="probability_default — split by decision",
            ),
            use_container_width=True,
        )
