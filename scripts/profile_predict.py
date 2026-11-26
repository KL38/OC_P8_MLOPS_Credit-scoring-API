"""Profile /predict with cProfile (étape 4 — bottleneck identification).

Runs N in-process /predict calls via FastAPI's TestClient and dumps a
``.prof`` file consumable by snakeviz. By default DATABASE_URL is unset
so the profile shows ONLY what remains on the critical path after the
BackgroundTask migration (handler = assembly + inference + response).

Usage:
    uv run python scripts/profile_predict.py
    uv run python scripts/profile_predict.py --n 200 --warmup 5
    uv run snakeviz profiling/profile_predict.prof   # interactive view
"""

from __future__ import annotations

import argparse
import cProfile
import logging
import os
import pstats
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("scripts.profile_predict")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100,
                        help="number of /predict calls to profile (default: 100)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="warm-up calls executed BEFORE profiling starts (default: 3)")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "profiling" / "profile_predict.prof",
                        help="output .prof path (default: ./profiling/profile_predict.prof)")
    parser.add_argument("--seed", type=int, default=42,
                        help="payload sampling seed (default: 42)")
    parser.add_argument("--top", type=int, default=25,
                        help="number of rows to print from the cumulative-time table")
    args = parser.parse_args()

    # Profile the handler ONLY. Unset DATABASE_URL so the BackgroundTask
    # short-circuits — we want to see what the client actually waits on
    # post-migration, not the deferred DB round-trip.
    os.environ.pop("DATABASE_URL", None)

    # Build payloads with the same logic used by seed_traffic — real rows
    # from application_train.csv so the profile reflects realistic input
    # distributions (mix of NaN, ints, floats, categorical strings).
    from scripts.seed_traffic import APP_TRAIN_PATH, build_payloads
    rng = random.Random(args.seed)
    payloads = build_payloads(
        APP_TRAIN_PATH, n_known=args.n + args.warmup, n_unknown=0, rng=rng
    )

    # Lazy import so DATABASE_URL change above is honoured by api.settings.
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        # Warm-up: pay one-time costs (lazy imports, JIT-like caches, pandas
        # categorical materialisation) so they don't pollute the profile.
        for payload in payloads[: args.warmup]:
            resp = client.post("/predict", json=payload)
            if resp.status_code != 200:
                raise SystemExit(
                    f"Warm-up failed: status={resp.status_code} body={resp.text}"
                )
        logger.info("Warm-up: %d calls OK", args.warmup)

        # Profiled window
        profiler = cProfile.Profile()
        profiler.enable()
        for payload in payloads[args.warmup:]:
            resp = client.post("/predict", json=payload)
            if resp.status_code != 200:
                logger.warning(
                    "Unexpected status %s during profiling: %s",
                    resp.status_code, resp.text[:200],
                )
        profiler.disable()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(args.output))
    logger.info("Profile saved to %s", args.output)

    # Top-N by cumulative time — quick console view before snakeviz.
    logger.info("\nTop %d functions by cumulative time:\n", args.top)
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(args.top)

    logger.info("\nVisualise interactively:\n  uv run snakeviz %s\n", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
