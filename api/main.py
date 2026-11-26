"""FastAPI application entrypoint.

Loads the model and feature artefacts once at startup (lifespan), then
serves them to every request without ever reloading — per the brief's
critical guideline.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from api import db, settings
from api.inference_assembler import InferenceArtefacts, assemble
from api.logger import log_prediction
from api.predictor import CreditScoringPredictor
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


def _resolve_feature_store_path() -> Path:
    """Return a usable path for the parquet, downloading it on demand.

    Local file (offline dev, tests, prebuilt image) takes precedence. When
    absent — typically on the HF Space first cold start — fetch it from the
    companion Dataset repo via huggingface_hub. The downloaded path is
    cached locally by huggingface_hub for subsequent boots.
    """
    if settings.FEATURE_STORE_PATH.exists():
        return settings.FEATURE_STORE_PATH

    from huggingface_hub import hf_hub_download

    logger.info(
        "Feature store missing locally — downloading %s from dataset %s",
        settings.HF_DATASET_FILENAME,
        settings.HF_DATASET_REPO_ID,
    )
    cached_path = hf_hub_download(
        repo_id=settings.HF_DATASET_REPO_ID,
        filename=settings.HF_DATASET_FILENAME,
        repo_type="dataset",
    )
    return Path(cached_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy artefacts once and attach them to app.state."""
    logger.info("Loading model from %s", settings.MODEL_PATH)
    app.state.predictor = CreditScoringPredictor.load(
        model_path=settings.MODEL_PATH,
        model_info_path=settings.MODEL_INFO_PATH,
        default_threshold=settings.DEFAULT_THRESHOLD,
    )
    logger.info(
        "Predictor ready: version=%s threshold=%.4f",
        app.state.predictor.model_version,
        app.state.predictor.threshold,
    )

    feature_store_path = _resolve_feature_store_path()
    logger.info("Loading inference artefacts (feature_store=%s)...", feature_store_path)
    app.state.artefacts = InferenceArtefacts.load(
        feature_names_path=settings.FEATURE_NAMES_PATH,
        categories_path=settings.APP_TRAIN_CATEGORIES_PATH,
        binary_mappings_path=settings.APP_TRAIN_BINARY_MAPPINGS_PATH,
        no_history_template_path=settings.NO_HISTORY_TEMPLATE_PATH,
        feature_store_path=feature_store_path,
    )
    logger.info(
        "Artefacts ready: %d feature_names, feature_store=%d clients",
        len(app.state.artefacts.feature_names),
        len(app.state.artefacts.feature_store),
    )

    # Cache model_info for the /model/info route.
    app.state.model_info = json.loads(settings.MODEL_INFO_PATH.read_text())

    # Best-effort: initialise the prediction logger. If DATABASE_URL is unset
    # or the database is unreachable, the API still serves predictions but
    # without persistence.
    db.init_engine()

    yield

    db.reset_engine()


app = FastAPI(
    title="Credit Scoring API",
    description=(
        "Real-time credit default prediction for Prêt à Dépenser. "
        "Wraps a LightGBM model with business threshold 10*FN + FP."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """JSON error envelope for unexpected failures, structured for log shipping."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "type": exc.__class__.__name__},
    )


@app.get("/", include_in_schema=False)
async def read_root() -> RedirectResponse:
    """Send the root to the interactive docs.

    The Hugging Face Space page embeds the app in an iframe pointing at ``/``
    and offers no address bar, so whatever ``/`` returns is the entire demo a
    visitor gets. Landing them on Swagger makes the Space self-explanatory.
    Machine callers should use ``/health``.
    """
    return RedirectResponse(url="/docs", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok", model_version=request.app.state.predictor.model_version
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
async def model_info(request: Request) -> ModelInfoResponse:
    info = request.app.state.model_info
    return ModelInfoResponse(
        model_name=info.get("model_name", "lgbm_credit_scoring"),
        version=str(info.get("version", "unknown")),
        threshold=request.app.state.predictor.threshold,
        n_features_expected=len(request.app.state.artefacts.feature_names),
        metrics=info.get("metrics", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
async def predict(
    payload: PredictionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    raw_inputs = payload.model_dump()
    sk_id = raw_inputs.pop("SK_ID_CURR")

    artefacts = request.app.state.artefacts
    predictor: CreditScoringPredictor = request.app.state.predictor

    started = time.perf_counter()
    features = None
    client_known = False
    proba: float | None = None
    decision: str | None = None
    status_code = status.HTTP_200_OK
    error_message: str | None = None
    # Fine-grained timings (étape 4). Stored as int milliseconds (via round()
    # to avoid the systematic downward bias of int() on small values). Left
    # as None when the corresponding sub-step did not complete.
    feature_assembly_ms: int | None = None
    inference_ms: int | None = None
    inference_cpu_ms: int | None = None

    try:
        t_asm = time.perf_counter()
        features, client_known = assemble(
            raw_inputs, sk_id_curr=sk_id, artefacts=artefacts
        )
        feature_assembly_ms = round((time.perf_counter() - t_asm) * 1000.0)

        t_inf_wall = time.perf_counter()
        t_inf_cpu = time.process_time()
        proba, decision = predictor.predict(features)
        inference_ms = round((time.perf_counter() - t_inf_wall) * 1000.0)
        inference_cpu_ms = round((time.process_time() - t_inf_cpu) * 1000.0)

        return PredictionResponse(
            sk_id_curr=sk_id,
            probability_default=proba,
            decision=decision,
            threshold=predictor.threshold,
            model_version=predictor.model_version,
            client_known=client_known,
        )
    except HTTPException as http_exc:
        status_code = http_exc.status_code
        error_message = str(http_exc.detail)
        raise
    except Exception as exc:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_message = f"{exc.__class__.__name__}: {exc}"
        logger.exception("Failed to predict for sk_id=%s", sk_id)
        raise HTTPException(
            status_code=status_code,
            detail=f"Prediction failed: {exc.__class__.__name__}",
        ) from exc
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000.0)
        # Plumbing = handler time not accounted for by the two timed sub-steps.
        # Computed here (not in SQL) so every row carries a self-describing
        # value. Only meaningful on the success path where both sub-steps
        # ran. Clamped at 0 so the rounding noise from three independent
        # round() calls (each ±0.5 ms) can never produce a stored negative.
        plumbing_ms: int | None
        if feature_assembly_ms is not None and inference_ms is not None:
            plumbing_ms = max(0, latency_ms - feature_assembly_ms - inference_ms)
        else:
            plumbing_ms = None
        log_kwargs: dict[str, object | None] = dict(
            sk_id_curr=sk_id,
            client_known=client_known,
            raw_input=raw_inputs,
            features=features,
            probability_default=proba,
            decision=decision,
            threshold=predictor.threshold,
            model_version=predictor.model_version,
            latency_ms=latency_ms,
            feature_assembly_ms=feature_assembly_ms,
            inference_ms=inference_ms,
            inference_cpu_ms=inference_cpu_ms,
            plumbing_ms=plumbing_ms,
            status_code=status_code,
            error_message=error_message,
        )
        # Success: defer the DB round-trip until AFTER the response
        # is sent, so the client never waits on ~350 ms of DB log overhead.
        # Failure: log synchronously — BackgroundTasks scheduled on a route
        # are attached to its Response, and the exception-handler chain
        # builds its own Response, silently dropping pending tasks. Losing
        # observability on errors would be a worse trade-off than the
        # one-shot latency hit paid by failing requests.
        if status_code < 400:
            background_tasks.add_task(log_prediction, **log_kwargs)
        else:
            log_prediction(**log_kwargs)
