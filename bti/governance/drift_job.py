"""
Scheduled population-stability check.

Compares the last `window_days` of logged traffic with the training baseline
for the scoring model (and the shadow challenger, if one is running), records
the outcome in the audit log, and alerts through the webhook / email channels
when any score or feature reaches "investigate" (PSI ≥ 0.10) or "escalate"
(PSI ≥ 0.25).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import AuditLog
from bti.logging_config import get_logger
from bti.modeling import registry
from bti.operations.kpis import live_drift

log = get_logger("governance.drift_job")

EVENT_TYPE = "DRIFT_CHECK"
_RANK = {"no_model": -1, "insufficient_data": 0, "stable": 0, "investigate": 1, "escalate": 2}


def _summarise(report: Dict) -> Dict:
    return {
        "model_id": report.get("model_id"),
        "status": report.get("status"),
        "n": report.get("n"),
        "score_psi": report.get("score_psi"),
        "score_status": report.get("score_status"),
        "features_flagged": [f for f in report.get("features", []) if f["status"] != "stable"][:10],
        "detail": report.get("detail"),
    }


def run_drift_check(db: Session, window_days: Optional[int] = None, now: Optional[datetime] = None,
                    notify: bool = True) -> Dict:
    window_days = window_days or get_settings().drift_window_days
    end = now or datetime.utcnow()
    start = end - timedelta(days=window_days)
    champion = registry.model_for_role("champion")
    challenger = registry.model_for_role("challenger")

    checks: List[Dict] = []
    live_model = champion or challenger
    if live_model:
        checks.append({"traffic": "live", **_summarise(live_drift(db, start, end, model_id=live_model, shadow=False))})
    if champion and challenger:
        checks.append({"traffic": "shadow", **_summarise(live_drift(db, start, end, model_id=challenger, shadow=True))})

    worst = max((c["status"] for c in checks), key=lambda s: _RANK.get(s, 0), default="no_model")
    result = {"checked_at": end.isoformat(), "window": {"from": start.isoformat(), "to": end.isoformat()},
              "status": worst, "checks": checks, "alert": None}

    if notify and worst in ("investigate", "escalate"):
        from bti.alerts import AlertDispatcher
        result["alert"] = AlertDispatcher().dispatch_event(
            "MODEL_DRIFT", "CRITICAL" if worst == "escalate" else "WARNING", result)

    db.add(AuditLog(ts=end, event_type=EVENT_TYPE, payload=result))
    db.commit()
    log.info("Drift check complete", extra={"status": worst, "window_days": window_days})
    return result


def drift_history(db: Session, limit: int = 20) -> List[Dict]:
    rows = (db.query(AuditLog.ts, AuditLog.payload).filter(AuditLog.event_type == EVENT_TYPE)
            .order_by(AuditLog.ts.desc(), AuditLog.id.desc()).limit(limit).all())
    return [r.payload for r in rows]
