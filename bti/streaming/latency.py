# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Inline latency benchmark: the live decision path, end to end.

It never touches the operational database or store. It builds a scratch
SQLite database and a Redis key prefix of its own, seeds them with history up
to a cut-off, then sends the next N transactions (in time order) through
`score_and_decide`, the same function the API and the stream consumer call.

Each transaction goes through the whole path:
- features
- champion score, and reason codes unless deferred
- decision
- governed rules
- the challenger's shadow score
- score-log and case writes

It is timed under three configurations:
- `database`: history from the database, explanations inline (the Phase 6
  path)
- `store_inline`: history from the feature store, explanations inline
- `store_async`: feature store, explanations on the worker pool. This mode also
  reports how long reasons take to arrive.

The target is p99 under 50 ms. Results go to outputs/streaming/latency_report.json.

The figures come from a single process on the development container, with
SQLite and a local Redis. Production numbers depend on the hardware, the
network round trip to Redis and Postgres, and concurrency. Rerun this on the
target platform before relying on them.

    python -m bti.streaming.latency --n 1000
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

OUT = Path("outputs/streaming/latency_report.json")
TARGET_P99_MS = 50.0


def _pct(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"n": int(len(a)), "p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "max": round(float(a.max()), 2)}


def run(n: int = 1000, data_path: str = None, redis_url: str = None, warmup: int = 50) -> Dict:
    workdir = Path(tempfile.mkdtemp(prefix="bti-latency-"))
    os.environ["BTI_DATABASE_URL"] = f"sqlite:///{workdir / 'bench.db'}"
    from bti.config import get_settings
    get_settings.cache_clear()
    settings = get_settings()
    redis_url = redis_url or settings.feature_store_url or "redis://localhost:6379/0"

    from bti.database.init_db import create_tables, seed_from_csv
    from bti.modeling.features import event_timestamps
    from bti.modeling.train import default_data_path
    from bti.streaming import feature_store
    from bti.streaming.feature_store import FeatureStore, events_from_frame

    df = pd.read_csv(data_path or default_data_path(), low_memory=False)
    df = df.assign(_ts=event_timestamps(df)).sort_values("_ts", kind="stable").reset_index(drop=True)
    history, live = df.iloc[:-(n + warmup)], df.iloc[-(n + warmup):]
    seed_csv = workdir / "history.csv"
    history.drop(columns="_ts").to_csv(seed_csv, index=False)

    from bti.database import connection
    create_tables()
    seed_from_csv(str(seed_csv))
    store = FeatureStore(redis_url, prefix=f"bti:bench:{uuid.uuid4().hex[:8]}:")
    if not store.ping():
        raise SystemExit(f"Redis not reachable at {redis_url}")
    store.write_events(events_from_frame(history.drop(columns="_ts")))

    from bti.operations import explanations
    from bti.operations.scoring_service import score_and_decide, warm_up
    warm_up()
    txns = live.drop(columns="_ts").to_dict("records")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "transactions": n,
              "warmup": warmup, "history_rows": int(len(history)), "target_p99_ms": TARGET_P99_MS,
              "platform": {"python": platform.python_version(), "machine": platform.machine(),
                           "cpus": os.cpu_count(), "database": "sqlite (scratch)", "store": "redis (local)"},
              "configurations": {}}
    configs = (("database", "", "inline"), ("store_inline", redis_url, "inline"), ("store_async", redis_url, "async"))
    try:
        for name, url, mode in configs:
            settings.feature_store_url, settings.explain_mode = url, mode
            feature_store._client = store if url else None
            db = connection.SessionLocal()
            e2e, scorer_ms, paths, submitted = [], [], {}, {}
            try:
                for i, t in enumerate(txns):
                    t0 = time.perf_counter()
                    sd = score_and_decide(dict(t), db, explain=True)
                    dt = (time.perf_counter() - t0) * 1000
                    if i < warmup:
                        continue
                    e2e.append(dt)
                    scorer_ms.append(sd.live.latency_ms)
                    paths[sd.live.feature_path] = paths.get(sd.live.feature_path, 0) + 1
                    if sd.explanation == "pending":
                        submitted[sd.live.transaction_id] = datetime.utcnow()
            finally:
                db.close()
            entry = {"end_to_end_ms": _pct(e2e), "scorer_ms": _pct(scorer_ms), "feature_path": paths,
                     "meets_target": _pct(e2e)["p99"] < TARGET_P99_MS}
            if submitted:
                explanations.drain(300)
                lags, ready = [], 0
                for tid, sent in submitted.items():
                    got = explanations.get(tid)
                    if got.get("status") == "ready":
                        ready += 1
                        lags.append((datetime.fromisoformat(got["explained_at"]) - sent).total_seconds() * 1000)
                entry["explanations"] = {"submitted": len(submitted), "ready": ready,
                                         "ready_after_ms": _pct(lags) if lags else None}
            report["configurations"][name] = entry
    finally:
        store.flush()
        feature_store._client = None
    report["store_vs_database_p99_speedup"] = round(
        report["configurations"]["database"]["end_to_end_ms"]["p99"]
        / max(report["configurations"]["store_inline"]["end_to_end_ms"]["p99"], 1e-9), 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="End-to-end inline latency benchmark")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--data", default=None)
    parser.add_argument("--redis", default=None, help="Redis URL (default: the configured store, else localhost)")
    args = parser.parse_args()
    r = run(args.n, args.data, args.redis)
    for name, c in r["configurations"].items():
        print(f"{name:13s} e2e p50 {c['end_to_end_ms']['p50']:6.1f}  p95 {c['end_to_end_ms']['p95']:6.1f}  "
              f"p99 {c['end_to_end_ms']['p99']:6.1f} ms  paths {c['feature_path']}  "
              f"{'meets' if c['meets_target'] else 'MISSES'} p99<{TARGET_P99_MS:.0f}")


if __name__ == "__main__":
    main()
