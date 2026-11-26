"""API runtime configuration. Paths and constants resolved at import time."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(os.getenv("OC_P8_MODEL_PATH", ROOT / "models" / "model.onnx"))
MODEL_INFO_PATH = Path(
    os.getenv("OC_P8_MODEL_INFO_PATH", ROOT / "models" / "model_info.json")
)
FEATURE_NAMES_PATH = Path(
    os.getenv("OC_P8_FEATURE_NAMES_PATH", ROOT / "models" / "feature_names.json")
)
APP_TRAIN_CATEGORIES_PATH = Path(
    os.getenv(
        "OC_P8_APP_TRAIN_CATEGORIES_PATH",
        ROOT / "models" / "app_train_categories.json",
    )
)
APP_TRAIN_BINARY_MAPPINGS_PATH = Path(
    os.getenv(
        "OC_P8_APP_TRAIN_BINARY_MAPPINGS_PATH",
        ROOT / "models" / "app_train_binary_mappings.json",
    )
)
NO_HISTORY_TEMPLATE_PATH = Path(
    os.getenv("OC_P8_NO_HISTORY_TEMPLATE_PATH", ROOT / "models" / "no_history_template.json")
)
FEATURE_STORE_PATH = Path(
    os.getenv("OC_P8_FEATURE_STORE_PATH", ROOT / "data" / "features_store.parquet")
)

# When the local parquet is missing (HF Space cold start), fall back to
# downloading from a companion Dataset repo. Decoupling code (Space) from
# data (Dataset) is the pattern HF officially recommends for files >10 MB.
HF_DATASET_REPO_ID = os.getenv("OC_P8_HF_DATASET_REPO_ID", "KLEB38/oc-p8-features")
HF_DATASET_FILENAME = os.getenv("OC_P8_HF_DATASET_FILENAME", "features_store.parquet")

# Default fallback if model_info.json does not expose the optimised threshold.
# 0.33 minimises the business cost function 10*FN + FP from OC_P6 — re-run the
# threshold search if the model is retrained.
DEFAULT_THRESHOLD = 0.33

# Neon / PostgreSQL connection. When unset, the prediction logger is
# disabled gracefully (the API keeps serving, just without persistence).
# The logger writes to PREDICTIONS_TABLE — overridable so integration tests
# can target predictions_log_test via env without code changes.
DATABASE_URL = os.getenv("DATABASE_URL")
PREDICTIONS_TABLE = os.getenv("OC_P8_PREDICTIONS_TABLE", "predictions_log")
