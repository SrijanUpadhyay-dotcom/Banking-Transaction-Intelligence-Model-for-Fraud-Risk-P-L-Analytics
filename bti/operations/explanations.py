# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Asynchronous explanations.

With `scoring.explain_mode: async`, the live decision returns as soon as the
probability and decision are known. SHAP reason codes are then computed on a
small worker pool. When they are ready they are written to:
- the champion's score-log row (`reason_codes`, which reads null until then)
- any case opened for the transaction (analyst reasons)
- the audit log, as an EXPLANATION_ADDED event

`GET /api/v1/v3/explanations/{transaction_id}` returns the full reasons, or
`pending` until they are ready.

The decision never depends on the explanation: rules, step-up and cost
overrides read features and probability only. A customer notice that needs
reasons (adverse action, GDPR Art. 22) is produced from the case, after the
reasons exist.

For the current model inline SHAP costs about 2 ms, so the default is inline.
Async is for heavier models and for high-throughput streaming.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session, sessionmaker

from bti.config import get_settings
from bti.logging_config import get_logger

log = get_logger("operations.explanations")

CACHE_SIZE = 20_000
_pool: Optional[ThreadPoolExecutor] = None
_lock = threading.Lock()
_cache: "OrderedDict[str, Dict]" = OrderedDict()
_pending: Dict[str, Future] = {}


def async_enabled() -> bool:
    return get_settings().explain_mode == "async"


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=max(1, get_settings().explain_workers),
                                       thread_name_prefix="bti-explain")
        return _pool


def _remember(transaction_id: str, entry: Dict) -> None:
    with _lock:
        _cache[transaction_id] = entry
        _cache.move_to_end(transaction_id)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


def submit(transaction_id: str, model_id: str, model_input, values: Dict[str, object],
           db: Optional[Session] = None) -> Future:
    """Queue the explanation for a scored transaction; persists it when `db` is given."""
    bind = db.get_bind() if db is not None else None
    _remember(transaction_id, {"transaction_id": transaction_id, "model_id": model_id, "status": "pending"})

    def job():
        from bti.modeling import registry
        from bti.modeling.scorer import scorer
        try:
            art = registry.load_artifact(model_id)
            contributions, reasons, base = scorer.explain(model_id, art, model_input, values)
            entry = {"transaction_id": transaction_id, "model_id": model_id, "status": "ready",
                     "reason_codes": reasons, "contributions": contributions, "base_probability": base,
                     "explained_at": datetime.utcnow().isoformat(timespec="milliseconds")}
            _remember(transaction_id, entry)
            if bind is not None:
                _persist(bind, transaction_id, model_id, reasons)
            return entry
        except Exception as exc:
            log.error("Explanation failed", extra={"transaction_id": transaction_id, "error": str(exc)})
            _remember(transaction_id, {"transaction_id": transaction_id, "model_id": model_id, "status": "failed",
                                       "error": str(exc)})
            raise
        finally:
            with _lock:
                _pending.pop(transaction_id, None)

    future = _executor().submit(job)
    with _lock:
        _pending[transaction_id] = future
    return future


def _persist(bind, transaction_id: str, model_id: str, reasons: List[dict]) -> None:
    from bti.database.models import AuditLog, FraudCase, ScoreLog
    session = sessionmaker(bind=bind)()
    try:
        codes = [{k: r[k] for k in ("code", "rank", "share_of_risk")} for r in reasons]
        (session.query(ScoreLog)
         .filter(ScoreLog.transaction_id == transaction_id, ScoreLog.model_id == model_id,
                 ScoreLog.is_shadow.is_(False), ScoreLog.reason_codes.is_(None))
         .update({ScoreLog.reason_codes: codes}, synchronize_session=False))
        for case in session.query(FraudCase).filter(FraudCase.transaction_id == transaction_id).all():
            if not case.reason_codes:
                case.reason_codes = [{k: r[k] for k in ("code", "analyst_text")} for r in reasons]
        session.add(AuditLog(ts=datetime.utcnow(), event_type="EXPLANATION_ADDED", transaction_id=transaction_id,
                             payload={"model_id": model_id, "reason_codes": codes}))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get(transaction_id: str, db: Optional[Session] = None) -> Dict:
    """The explanation for a transaction: from memory, else the score log."""
    with _lock:
        entry = _cache.get(transaction_id)
    if entry is not None:
        return entry
    if db is not None:
        from bti.database.models import ScoreLog
        row = (db.query(ScoreLog).filter(ScoreLog.transaction_id == transaction_id, ScoreLog.is_shadow.is_(False))
               .order_by(ScoreLog.id.desc()).first())
        if row is not None:
            if row.reason_codes is None:
                return {"transaction_id": transaction_id, "model_id": row.model_id, "status": "pending"}
            from bti.governance.reason_codes import REASON_CODES
            return {"transaction_id": transaction_id, "model_id": row.model_id, "status": "ready",
                    "reason_codes": [{**r, "analyst_text": REASON_CODES.get(r["code"], {}).get("analyst"),
                                      "customer_text": REASON_CODES.get(r["code"], {}).get("customer")}
                                     for r in row.reason_codes]}
    return {"transaction_id": transaction_id, "status": "unknown"}


def drain(timeout: float = 30.0) -> int:
    """Wait for queued explanations (tests, graceful shutdown). Returns how many were waited on."""
    with _lock:
        futures = list(_pending.values())
    wait(futures, timeout=timeout)
    return len(futures)


def backlog() -> int:
    with _lock:
        return len(_pending)
