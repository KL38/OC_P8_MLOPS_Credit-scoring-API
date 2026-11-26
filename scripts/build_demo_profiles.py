"""Build the payload artefact behind the dashboard's "Try the API" buttons.

Writes ``dashboard/static/demo_profiles.json`` with:
- the two curated profiles from ``examples/`` (low risk, high risk);
- a sample of SK_ID_CURR values present in the feature store, so the
  "random client" button hits known clients with a real credit history.

No raw application row is exported: application_train.csv holds personal
data and stays local. The random client reuses a curated profile and only
varies its main drivers (see ``dashboard/demo.py``).

Usage:
    uv run python scripts/build_demo_profiles.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE = ROOT / "data" / "features_store.parquet"
OUTPUT = ROOT / "dashboard" / "static" / "demo_profiles.json"
N_KNOWN_IDS = 200
SEED = 42


def main() -> None:
    ids = pq.read_table(FEATURE_STORE, columns=["SK_ID_CURR"]).column(0).to_pylist()
    known_ids = sorted(random.Random(SEED).sample(ids, N_KNOWN_IDS))
    profiles = {
        name: json.loads(
            (ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8")
        )
        for name in ("low_risk", "high_risk")
    }
    OUTPUT.write_text(
        json.dumps({**profiles, "known_ids": known_ids}, indent=1), encoding="utf-8"
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(known_ids)} known ids)")


if __name__ == "__main__":
    main()
