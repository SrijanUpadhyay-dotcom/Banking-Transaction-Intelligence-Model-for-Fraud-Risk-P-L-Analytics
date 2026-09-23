"""
In-process request metrics: counts, 5xx errors and latency samples per
endpoint group since this worker started. Per-worker by design; a production
deployment would export these to Prometheus or the bank's APM.
"""

import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict

import numpy as np

STARTED_AT = datetime.now(timezone.utc)
_START_MONOTONIC = time.monotonic()
_SAMPLES = 5000
_lock = threading.Lock()
_requests: Dict[str, int] = defaultdict(int)
_errors: Dict[str, int] = defaultdict(int)
_latency: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_SAMPLES))


def route_group(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "api":
        return "/".join(parts[2:4]) if parts[2] in ("sas", "v3", "score") and len(parts) > 3 else parts[2]
    return parts[0] if parts else "root"


def record(path: str, status: int, ms: float) -> None:
    group = route_group(path)
    with _lock:
        _requests[group] += 1
        if status >= 500:
            _errors[group] += 1
        _latency[group].append(ms)


def uptime_seconds() -> float:
    return round(time.monotonic() - _START_MONOTONIC, 1)


def snapshot() -> Dict:
    with _lock:
        groups = {}
        for g, n in sorted(_requests.items()):
            lat = np.asarray(_latency[g], dtype=float)
            groups[g] = {
                "requests": n,
                "errors_5xx": _errors[g],
                "error_rate": round(_errors[g] / n, 5) if n else 0.0,
                "latency_ms": {q: round(float(np.percentile(lat, v)), 1) for q, v in
                               (("p50", 50), ("p95", 95), ("p99", 99))} if len(lat) else None,
            }
    return {"started_at": STARTED_AT.isoformat(), "uptime_seconds": uptime_seconds(),
            "sample_window": f"last {_SAMPLES} requests per group", "http": groups}


def reset() -> None:
    with _lock:
        _requests.clear()
        _errors.clear()
        _latency.clear()
