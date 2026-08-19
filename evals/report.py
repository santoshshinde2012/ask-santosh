"""Publish the measured retrieval metrics as shields.io endpoint JSON.

Run after the gate passes. Emits one small file per badge plus a raw summary,
into a directory that CI pushes to the `gh-pages` branch. Because the numbers
come from `metrics.compute()` — the same call the gate asserts on — the badge
cannot drift from what was actually measured.

    python evals/report.py out/
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from metrics import Metrics, compute

# Mirrors the thresholds asserted in test_retrieval.py.
THRESHOLDS = {"recall@1": 0.80, "recall@3": 1.00, "mrr": 0.90}


def _colour(value: float, floor: float) -> str:
    if value >= floor:
        return "brightgreen"
    if value >= floor - 0.10:
        return "orange"
    return "red"


def _badge(label: str, value: float, floor: float) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "label": label,
        "message": f"{value:.2f}",
        "color": _colour(value, floor),
    }


def write(metrics: Metrics, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    stamp = os.getenv("RUN_DATE_ISO") or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    files = {
        "recall1.json": _badge("recall@1", metrics.recall_at_1, THRESHOLDS["recall@1"]),
        "recall3.json": _badge("recall@3", metrics.recall_at_3, THRESHOLDS["recall@3"]),
        "mrr.json": _badge("MRR", metrics.mrr, THRESHOLDS["mrr"]),
        "updated.json": {
            "schemaVersion": 1,
            "label": "evals",
            "message": stamp,
            "color": "blue",
        },
        # The underlying numbers, for anyone who wants to check the badges.
        "summary.json": {
            "measured_at": stamp,
            "queries": metrics.queries,
            "chunks": metrics.chunks,
            "thresholds": THRESHOLDS,
            "metrics": {
                "recall@1": round(metrics.recall_at_1, 4),
                "recall@3": round(metrics.recall_at_3, 4),
                "mrr": round(metrics.mrr, 4),
            },
            "misses": list(metrics.misses),
        },
    }

    written = []
    for name, payload in files.items():
        path = out / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        written.append(path)
    return written


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    metrics = compute()
    for path in write(metrics, out):
        print(f"wrote {path}")
    print(
        f"recall@1={metrics.recall_at_1:.3f} "
        f"recall@3={metrics.recall_at_3:.3f} "
        f"MRR={metrics.mrr:.3f} over {metrics.queries} queries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
