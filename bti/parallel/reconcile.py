"""
Daily reconciliation of the two decision logs.

For one UTC day it compares what BTI's router recorded with what the incumbent's
feed recorded, and flags the breaks that make a parallel run untrustworthy or
leave a customer with the wrong outcome:

- `routed_without_incumbent_record`: BTI routed a transaction the incumbent's
  feed never reported (a feed gap, or traffic the incumbent never saw).
- `incumbent_without_bti`: the incumbent decided a transaction BTI never scored
  (BTI is missing traffic).
- `duplicate_routing`: one transaction routed more than once with different
  effective decisions.
- `incumbent_decision_mismatch`: the decision the switch sent to BTI with the
  request differs from the one in the incumbent's own log.
- `enforcement_mismatch`: the decision the bank actually executed differs from
  the effective decision the router returned. This is the most serious break,
  because a customer got an outcome no system chose. It needs `executed_decision`
  in the feed.

Fallbacks (BTI timeout or error in the BTI arm) are reported with their rate.
The result is written to the audit log, and breaks above the threshold alert.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import AuditLog, IncumbentDecision, RoutedDecision, ScoreLog
from bti.logging_config import get_logger
from bti.parallel.incumbent import CHUNK, latest_incumbent

log = get_logger("parallel.reconcile")

EVENT_TYPE = "RECONCILIATION"
SAMPLE = 20


def _day_bounds(day: date):
    start = datetime(day.year, day.month, day.day)
    return start, start + timedelta(days=1)


def reconcile(db: Session, day: Optional[date] = None, notify: bool = True) -> Dict:
    day = day or (datetime.utcnow() - timedelta(days=1)).date()
    start, end = _day_bounds(day)
    routed = pd.DataFrame(
        db.query(RoutedDecision.transaction_id, RoutedDecision.arm, RoutedDecision.effective_decision,
                 RoutedDecision.incumbent_decision, RoutedDecision.decided_by, RoutedDecision.fallback_reason,
                 RoutedDecision.routed_at)
        .filter(RoutedDecision.routed_at >= start, RoutedDecision.routed_at < end).all(),
        columns=["transaction_id", "arm", "effective_decision", "sent_incumbent_decision", "decided_by",
                 "fallback_reason", "routed_at"])
    inc_day = pd.DataFrame(
        db.query(IncumbentDecision.transaction_id)
        .filter(IncumbentDecision.decided_at >= start, IncumbentDecision.decided_at < end).distinct().all(),
        columns=["transaction_id"])
    routed_ids = routed["transaction_id"].unique().tolist()
    inc_for_routed = latest_incumbent(db, transaction_ids=routed_ids) if routed_ids else pd.DataFrame(
        columns=["transaction_id", "decision", "executed_decision"])

    breaks: Dict[str, Dict] = {}

    def add(name: str, ids: List[str], detail: str):
        breaks[name] = {"count": len(ids), "sample": sorted(ids)[:SAMPLE], "detail": detail}

    last = routed.sort_values("routed_at", kind="stable").drop_duplicates("transaction_id", keep="last") if not routed.empty \
        else routed
    add("routed_without_incumbent_record",
        sorted(set(routed_ids) - set(inc_for_routed["transaction_id"])),
        "Routed by BTI, absent from the incumbent's feed")

    day_ids = inc_day["transaction_id"].tolist()
    scored = set()
    for i in range(0, len(day_ids), CHUNK):
        scored |= {r[0] for r in db.query(ScoreLog.transaction_id)
                   .filter(ScoreLog.transaction_id.in_(day_ids[i:i + CHUNK])).distinct().all()}
    add("incumbent_without_bti", sorted(set(inc_day["transaction_id"]) - set(routed_ids) - scored),
        "Decided by the incumbent, never scored by BTI")

    dup = routed.groupby("transaction_id")["effective_decision"].nunique(dropna=False)
    add("duplicate_routing", dup[dup > 1].index.tolist(), "Routed more than once with different decisions")

    joined = last.merge(inc_for_routed.rename(columns={"decision": "logged_incumbent_decision"}),
                        on="transaction_id", how="inner") if not last.empty else pd.DataFrame()
    if not joined.empty:
        every = routed.merge(inc_for_routed.rename(columns={"decision": "logged_incumbent_decision"}),
                             on="transaction_id", how="inner")
        sent = every["sent_incumbent_decision"].notna()
        add("incumbent_decision_mismatch",
            sorted(set(every.loc[sent & (every["sent_incumbent_decision"] != every["logged_incumbent_decision"]),
                                 "transaction_id"])),
            "The decision sent with a request differs from the incumbent's own log")
        executed = joined["executed_decision"].notna() & joined["effective_decision"].notna()
        add("enforcement_mismatch",
            joined.loc[executed & (joined["executed_decision"] != joined["effective_decision"]),
                       "transaction_id"].tolist(),
            "The bank executed a different decision from the router's effective decision")
        executed_known = int(executed.sum())
    else:
        add("incumbent_decision_mismatch", [], "The decision sent with the request differs from the incumbent's log")
        add("enforcement_mismatch", [], "The bank executed a different decision from the router's")
        executed_known = 0

    bti_arm = last[last["arm"] == "bti"] if not last.empty else last
    total = max(len(set(routed_ids) | set(inc_day["transaction_id"])), 1)
    break_count = sum(b["count"] for b in breaks.values())
    threshold = get_settings().reconciliation_break_rate_alert
    severity = None
    if breaks["enforcement_mismatch"]["count"]:
        severity = "CRITICAL"
    elif break_count / total > threshold:
        severity = "WARNING"
    result = {
        "day": day.isoformat(),
        "routed_transactions": len(routed_ids),
        "incumbent_transactions": int(len(inc_day)),
        "executed_decisions_known": executed_known,
        "breaks": breaks,
        "break_rate": round(break_count / total, 6),
        "fallbacks": {"bti_arm_transactions": int(len(bti_arm)),
                      "by_reason": bti_arm["fallback_reason"].value_counts().to_dict() if len(bti_arm) else {},
                      "rate": round(float(bti_arm["fallback_reason"].notna().mean()), 5) if len(bti_arm) else None},
        "status": "clean" if not break_count else "breaks",
        "alert": None,
    }
    if notify and severity:
        from bti.alerts import AlertDispatcher
        result["alert"] = AlertDispatcher().dispatch_event(
            "RECONCILIATION_BREAKS", severity,
            {"day": result["day"], "break_rate": result["break_rate"],
             "breaks": {k: v["count"] for k, v in breaks.items()}})
    db.add(AuditLog(ts=datetime.utcnow(), event_type=EVENT_TYPE, payload=result))
    db.commit()
    log.info("Reconciliation complete", extra={"day": result["day"], "status": result["status"]})
    return result


def reconciliation_history(db: Session, limit: int = 30) -> List[Dict]:
    rows = (db.query(AuditLog.payload).filter(AuditLog.event_type == EVENT_TYPE)
            .order_by(AuditLog.ts.desc(), AuditLog.id.desc()).limit(limit).all())
    return [r[0] for r in rows]
