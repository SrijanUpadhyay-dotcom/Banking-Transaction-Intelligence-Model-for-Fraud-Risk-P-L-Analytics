# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Readiness probe: whether this instance should receive live traffic.

`/health` (liveness) only says the process is up. `/readyz` says it can make
correct decisions now. A load balancer or Kubernetes stops routing to an
instance that is not ready.

**Critical checks** (any failure means not ready, HTTP 503):
- the database answers
- a scoring model resolves and its artifact loads
- data residency is compliant when it is enforced

**Degraded checks** (still ready, reported for operators):
- the feature store is unreachable. Live scoring falls back to the database
  path: correct, but about 7× slower at p99.
- the graph snapshot is missing for a model that uses graph features
- the asynchronous explanation backlog is high
- residency violations in warn mode
- no tokenisation key, so card and account messages cannot be accepted
- access control: the shared legacy key is enabled, or no principals exist
"""

from __future__ import annotations

import time
from typing import Dict

from bti.config import get_settings


def readiness() -> Dict:
    t0 = time.perf_counter()
    checks: Dict[str, Dict] = {}
    critical_fail, degraded = [], []

    try:
        from sqlalchemy import text
        from bti.database.connection import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok"}
    except Exception as exc:
        checks["database"] = {"status": "fail", "error": type(exc).__name__}
        critical_fail.append("database")

    art = None
    try:
        from bti.modeling import registry
        from bti.modeling.scorer import scorer
        model_id, role, provisional = scorer.resolve("champion")
        art = registry.load_artifact(model_id)
        checks["model"] = {"status": "ok", "model_id": model_id, "role": role, "provisional": provisional,
                           "feature_version": art.get("feature_version", 1)}
    except Exception as exc:
        checks["model"] = {"status": "fail", "error": str(exc)[:200]}
        critical_fail.append("model")

    s = get_settings()
    if s.feature_store_url:
        try:
            from bti.streaming.feature_store import get_store
            ok = get_store().ping()
        except Exception:
            ok = False
        checks["feature_store"] = {"status": "ok" if ok else "degraded",
                                   **({} if ok else {"impact": "live scoring uses the database path (slower)"})}
        if not ok:
            degraded.append("feature_store")
        if art is not None and checks["model"]["status"] == "ok":
            from bti.streaming.parity import online_allowed
            checks["feature_store"]["online_path_allowed"] = online_allowed(checks["model"]["model_id"], art)
    else:
        checks["feature_store"] = {"status": "not_configured"}

    if art is not None and (art.get("graph") or {}).get("uses_graph"):
        from bti.graph.snapshot import status as graph_status
        g = graph_status()
        checks["graph_snapshot"] = g
        if g.get("status") != "ok":
            degraded.append("graph_snapshot")

    from bti.operations import explanations
    backlog = explanations.backlog()
    checks["explanations"] = {"mode": s.explain_mode, "backlog": backlog}
    if backlog > 1000:
        degraded.append("explanations")

    from bti.operations.residency import check
    r = check()
    checks["residency"] = {"jurisdiction": r["jurisdiction"], "mode": r["mode"], "compliant": r["compliant"],
                           "violations": [f"{v['name']} ({v['host']})" for v in r["violations"]]}
    if not r["compliant"]:
        (critical_fail if r["mode"] == "enforce" else degraded).append("residency")

    checks["tokenisation"] = {"status": "ok" if s.token_key else
                              ("development_key" if s.environment.lower() in ("development", "dev", "test")
                               else "missing")}
    if checks["tokenisation"]["status"] == "missing":
        degraded.append("tokenisation")

    from bti.security.principals import load as load_principals
    active = sum(1 for pr in load_principals().values() if pr.active)
    checks["access_control"] = {"active_principals": active, "legacy_shared_key": s.allow_legacy_api_key}
    if s.allow_legacy_api_key or active == 0:
        degraded.append("access_control")

    return {"ready": not critical_fail, "failed": critical_fail, "degraded": degraded, "checks": checks,
            "checked_in_ms": round((time.perf_counter() - t0) * 1000, 1)}
