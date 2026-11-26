"""Live demo calls from the dashboard to the scoring API.

Lets a visitor send a real ``/predict`` request and watch it land in the
monitoring tabs. Payloads come from ``static/demo_profiles.json`` (built by
``scripts/build_demo_profiles.py``).
"""

from __future__ import annotations

import copy
import json
import random
import time
import urllib.request
from pathlib import Path
from typing import Any

PREDICT_URL = "https://kleb38-oc-p8.hf.space/predict"
PROFILES_PATH = Path(__file__).parent / "static" / "demo_profiles.json"


def load_profiles() -> dict[str, Any]:
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def random_client(profiles: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Known client id + a curated profile whose main drivers are re-drawn.

    Values stay inside the bounds of ``api/schemas.PredictionRequest``. One
    latent "quality" draw drives the three EXT_SOURCE scores together, so the
    decisions spread across GRANTED and REFUSED instead of clustering.
    """
    p = copy.deepcopy(profiles[rng.choice(["low_risk", "high_risk"])])
    p["SK_ID_CURR"] = rng.choice(profiles["known_ids"])

    quality = rng.random()
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        p[col] = round(min(max(quality + rng.gauss(0, 0.08), 0.01), 0.95), 3)

    income = rng.randrange(60_000, 400_001, 1_000)
    credit = round(income * rng.uniform(1.5, 6.0), -3)
    p["AMT_INCOME_TOTAL"] = float(income)
    p["AMT_CREDIT"] = credit
    p["AMT_GOODS_PRICE"] = round(credit * 0.9, -3)
    p["AMT_ANNUITY"] = round(credit * rng.uniform(0.035, 0.06), 1)

    age_days = rng.randint(21 * 365, 65 * 365)
    p["DAYS_BIRTH"] = -age_days
    p["DAYS_EMPLOYED"] = -rng.randint(30, min(age_days - 18 * 365, 15_000))

    adults = p["CNT_FAM_MEMBERS"] - p["CNT_CHILDREN"]
    p["CNT_CHILDREN"] = rng.randint(0, 3)
    p["CNT_FAM_MEMBERS"] = adults + p["CNT_CHILDREN"]
    p["AMT_REQ_CREDIT_BUREAU_YEAR"] = rng.randint(0, 10)
    return p


def call_predict(payload: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
    """POST the payload; returns the API response plus the round-trip in ms."""
    request = urllib.request.Request(
        PREDICT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    body["round_trip_ms"] = round((time.perf_counter() - t0) * 1000)
    return body
