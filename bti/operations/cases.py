# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Case management that feeds the label loop.

Every live REVIEW or DECLINE decision opens a case (once per transaction).
Declines need a disposition too: the money never moves, so no chargeback will
ever label them, and without one the model would learn nothing from its own
declines. Cases also open from manual referrals, such as a customer asking for
human review of an automated decline.

**Queues** (`cases.queues`). A case goes to the first queue it qualifies for:
- `urgent`: probability ≥ 0.8, or expected loss ≥ $1,000; 60-minute SLA.
- `high_value`: amount ≥ $10,000; 120-minute SLA.
- `standard`: everything else; 8-hour SLA.

Within a queue, cases are worked in expected-loss order (probability ×
amount), oldest first on ties.

**Dispositions become labels.**
- `confirmed_fraud` records an INVESTIGATOR_CONFIRMED label.
- `confirmed_genuine` records an INVESTIGATOR_CLEARED label.
- `inconclusive` / `customer_unreachable` / `duplicate` write no label; the
  90-day maturity rule decides later.

A later chargeback still overrides, because the latest label wins.

**Maker-checker.** Clearing a case of $10,000 or more as genuine needs a second
reviewer (`checked_by`) other than the analyst.

SLA breaches alert once per case through the monitoring channels.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import AuditLog, FraudCase
from bti.logging_config import get_logger
from bti.operations.feedback import record_labels

log = get_logger("operations.cases")

DISPOSITIONS = {"confirmed_fraud": ("INVESTIGATOR_CONFIRMED", 1), "confirmed_genuine": ("INVESTIGATOR_CLEARED", 0),
                "inconclusive": None, "customer_unreachable": None, "duplicate": None}
OPEN_STATES = ("open", "assigned", "pending_customer")


class CaseError(ValueError):
    pass


def _row(c: FraudCase, now: Optional[datetime] = None) -> Dict:
    now = now or datetime.utcnow()
    out = {k.name: (getattr(c, k.name).isoformat() if isinstance(getattr(c, k.name), datetime) else getattr(c, k.name))
           for k in FraudCase.__table__.columns}
    out["sla_breached"] = (c.closed_at or now) > c.sla_due_at
    return out


def choose_queue(probability: Optional[float], amount_usd: Optional[float]) -> str:
    p, amt = float(probability or 0), float(amount_usd or 0)
    for name, rule in get_settings().case_queues.items():
        checks = []
        if "min_probability" in rule:
            checks.append(p >= rule["min_probability"])
        if "min_expected_loss_usd" in rule:
            checks.append(p * amt >= rule["min_expected_loss_usd"])
        if "min_amount_usd" in rule:
            checks.append(amt >= rule["min_amount_usd"])
        if not checks or any(checks):
            return name
    return list(get_settings().case_queues)[-1]


def open_case(db: Session, transaction_id: str, source: str, customer_id: Optional[str] = None,
              model_id: Optional[str] = None, fraud_probability: Optional[float] = None,
              amount_usd: Optional[float] = None, decision: Optional[str] = None,
              reason_codes: Optional[List] = None, queue: Optional[str] = None, commit: bool = True) -> Dict:
    """Open a case unless one is already open for the transaction (idempotent)."""
    existing = (db.query(FraudCase).filter(FraudCase.transaction_id == str(transaction_id),
                                           FraudCase.status.in_(OPEN_STATES)).first())
    if existing is not None:
        return _row(existing)
    queues = get_settings().case_queues
    queue = queue or choose_queue(fraud_probability, amount_usd)
    if queue not in queues:
        raise CaseError(f"queue must be one of {list(queues)}")
    now = datetime.utcnow()
    case = FraudCase(transaction_id=str(transaction_id), customer_id=customer_id, queue=queue,
                     priority_score=round(float(fraud_probability or 0) * float(amount_usd or 0), 2), source=source,
                     model_id=model_id, fraud_probability=fraud_probability, amount_usd=amount_usd, decision=decision,
                     reason_codes=reason_codes, status="open", created_at=now,
                     sla_due_at=now + timedelta(minutes=queues[queue]["sla_minutes"]))
    db.add(case)
    db.flush()
    db.add(AuditLog(ts=now, event_type="CASE_OPENED", transaction_id=case.transaction_id,
                    payload={"case_id": case.id, "queue": queue, "source": source}))
    if commit:
        db.commit()
    return _row(case)


def assign_next(db: Session, analyst: str, queue: Optional[str] = None) -> Optional[Dict]:
    """Give the analyst the most urgent unassigned case: queue order, then expected loss, then age."""
    if not analyst or not analyst.strip():
        raise CaseError("analyst is required")
    order = {name: i for i, name in enumerate(get_settings().case_queues)}
    q = db.query(FraudCase).filter(FraudCase.status == "open")
    if queue:
        q = q.filter(FraudCase.queue == queue)
    candidates = q.all()
    if not candidates:
        return None
    case = min(candidates, key=lambda c: (order.get(c.queue, 99), -c.priority_score, c.created_at))
    return assign(db, case.id, analyst)


def assign(db: Session, case_id: int, analyst: str) -> Dict:
    case = db.get(FraudCase, case_id)
    if case is None:
        raise CaseError(f"No case {case_id}")
    if case.status not in OPEN_STATES:
        raise CaseError(f"Case {case_id} is {case.status}")
    now = datetime.utcnow()
    case.assigned_to, case.status = analyst.strip(), "assigned"
    case.assigned_at = case.assigned_at or now
    db.add(AuditLog(ts=now, event_type="CASE_ASSIGNED", transaction_id=case.transaction_id,
                    payload={"case_id": case.id, "analyst": case.assigned_to}))
    db.commit()
    return _row(case)


def dispose(db: Session, case_id: int, disposition: str, analyst: str, notes: Optional[str] = None,
            checked_by: Optional[str] = None, fraud_type: Optional[str] = None,
            loss_amount: Optional[float] = None) -> Dict:
    case = db.get(FraudCase, case_id)
    if case is None:
        raise CaseError(f"No case {case_id}")
    if case.status not in OPEN_STATES:
        raise CaseError(f"Case {case_id} is already {case.status}")
    if disposition not in DISPOSITIONS:
        raise CaseError(f"disposition must be one of {list(DISPOSITIONS)}")
    if not analyst or not analyst.strip():
        raise CaseError("analyst is required")
    threshold = get_settings().case_checker_threshold_usd
    if disposition == "confirmed_genuine" and (case.amount_usd or 0) >= threshold:
        if not checked_by or checked_by.strip().lower() == analyst.strip().lower():
            raise CaseError(f"Maker-checker: clearing a case of ${threshold:,.0f} or more needs a second reviewer "
                            f"other than the analyst (checked_by)")
    now = datetime.utcnow()
    case.status, case.closed_at, case.disposition = "closed", now, disposition
    case.disposition_by, case.checked_by, case.disposition_notes = analyst.strip(), checked_by, notes
    case.fraud_type, case.loss_amount = fraud_type, loss_amount
    case.assigned_to = case.assigned_to or analyst.strip()
    case.assigned_at = case.assigned_at or now
    label = DISPOSITIONS[disposition]
    db.add(AuditLog(ts=now, event_type="CASE_CLOSED", transaction_id=case.transaction_id,
                    payload={"case_id": case.id, "disposition": disposition, "analyst": analyst,
                             "checked_by": checked_by, "sla_breached": now > case.sla_due_at}))
    db.commit()
    if label is not None:
        source, value = label
        record_labels(db, [{"transaction_id": case.transaction_id, "label": value, "label_source": source,
                            "fraud_type": fraud_type, "event_at": now, "loss_amount": loss_amount,
                            "currency": "USD" if loss_amount is not None else None,
                            "reported_by": analyst, "notes": f"Case {case.id}: {notes or disposition}"}])
        case.label_written = True
        db.commit()
    return _row(case)


def record_check(db: Session, case_id: int, checker: str, notes: Optional[str] = None) -> Dict:
    """
    The second reviewer's own confirmation, made with their own credentials (Phase 10). Clearing a high-value
    case then accepts only a checker recorded here, so the maker cannot name a checker who never looked.
    """
    case = db.get(FraudCase, case_id)
    if case is None or case.status not in OPEN_STATES:
        raise CaseError(f"Case {case_id} is not open")
    if not checker or not checker.strip():
        raise CaseError("checker is required")
    if case.assigned_to and case.assigned_to.strip().lower() == checker.strip().lower():
        raise CaseError("Maker-checker: the assigned analyst cannot also be the checker")
    case.checked_by = checker.strip()
    db.add(AuditLog(ts=datetime.utcnow(), event_type="CASE_CHECKED", transaction_id=case.transaction_id,
                    payload={"case_id": case.id, "checker": checker.strip(), "notes": notes}))
    db.commit()
    return _row(case)


def set_pending_customer(db: Session, case_id: int, analyst: str) -> Dict:
    case = db.get(FraudCase, case_id)
    if case is None or case.status not in OPEN_STATES:
        raise CaseError(f"Case {case_id} is not open")
    case.status = "pending_customer"
    db.add(AuditLog(ts=datetime.utcnow(), event_type="CASE_PENDING_CUSTOMER", transaction_id=case.transaction_id,
                    payload={"case_id": case.id, "analyst": analyst}))
    db.commit()
    return _row(case)


def list_cases(db: Session, status: Optional[str] = None, queue: Optional[str] = None,
               assigned_to: Optional[str] = None, limit: int = 200) -> List[Dict]:
    q = db.query(FraudCase)
    if status:
        q = q.filter(FraudCase.status == status)
    if queue:
        q = q.filter(FraudCase.queue == queue)
    if assigned_to:
        q = q.filter(FraudCase.assigned_to == assigned_to)
    now = datetime.utcnow()
    return [_row(c, now) for c in q.order_by(FraudCase.sla_due_at).limit(limit).all()]


def case_context(db: Session, case_id: int, history: int = 20) -> Dict:
    """Everything an analyst needs beside the case: the scored decision, the customer's recent activity,
    their earlier cases and confirmed outcomes."""
    from bti.database.models import FraudLabel, ScoreLog
    case = db.get(FraudCase, case_id)
    if case is None:
        raise CaseError(f"No case {case_id}")
    score = (db.query(ScoreLog).filter(ScoreLog.transaction_id == case.transaction_id,
                                       ScoreLog.is_shadow.is_(False))
             .order_by(ScoreLog.scored_at.desc()).first())
    recent, earlier, outcomes = [], [], []
    if case.customer_id:
        rows = (db.query(ScoreLog).filter(ScoreLog.customer_id == case.customer_id, ScoreLog.is_shadow.is_(False))
                .order_by(ScoreLog.scored_at.desc()).limit(history).all())
        recent = [{"transaction_id": r.transaction_id, "scored_at": r.scored_at.isoformat(),
                   "amount_usd": r.amount_usd, "fraud_probability": r.fraud_probability, "decision": r.decision}
                  for r in rows]
        earlier = [_row(c) for c in db.query(FraudCase).filter(FraudCase.customer_id == case.customer_id,
                                                               FraudCase.id != case.id)
                   .order_by(FraudCase.created_at.desc()).limit(10).all()]
        ids = [r["transaction_id"] for r in recent]
        if ids:
            outcomes = [{"transaction_id": l.transaction_id, "label": l.label, "source": l.label_source,
                         "at": l.event_at.isoformat()}
                        for l in db.query(FraudLabel).filter(FraudLabel.transaction_id.in_(ids))
                        .order_by(FraudLabel.event_at.desc()).all()]
    return {
        "case": _row(case),
        "score": None if score is None else {
            "model_id": score.model_id, "model_role": score.model_role, "fraud_probability": score.fraud_probability,
            "decision": score.decision, "jurisdiction": score.jurisdiction, "amount_usd": score.amount_usd,
            "guardrails": score.guardrails or [], "features": score.features or {},
            "scored_at": score.scored_at.isoformat()},
        "customer_recent": recent, "customer_earlier_cases": earlier, "customer_outcomes": outcomes,
        "checker_threshold_usd": get_settings().case_checker_threshold_usd,
        "dispositions": list(DISPOSITIONS),
    }


def queue_status(db: Session, days: int = 7, now: Optional[datetime] = None) -> Dict:
    now = now or datetime.utcnow()
    rows = db.query(FraudCase).filter((FraudCase.status.in_(OPEN_STATES)) |
                                      (FraudCase.closed_at >= now - timedelta(days=days))).all()
    frame = pd.DataFrame([_row(c, now) for c in rows])
    out = {"as_of": now.isoformat(), "window_days": days, "queues": {}}
    for name, rule in get_settings().case_queues.items():
        g = frame[frame["queue"] == name] if not frame.empty else frame
        open_ = g[g["status"].isin(OPEN_STATES)] if not g.empty else g
        closed = g[g["status"] == "closed"] if not g.empty else g
        minutes = ((pd.to_datetime(closed["closed_at"]) - pd.to_datetime(closed["created_at"]))
                   .dt.total_seconds() / 60) if not closed.empty else pd.Series(dtype=float)
        decided = closed[closed["disposition"].isin(["confirmed_fraud", "confirmed_genuine"])] if not closed.empty else closed
        out["queues"][name] = {
            "sla_minutes": rule["sla_minutes"],
            "open": int(len(open_)),
            "unassigned": int((open_["status"] == "open").sum()) if not open_.empty else 0,
            "open_breached": int(open_["sla_breached"].sum()) if not open_.empty else 0,
            "oldest_open_minutes": round(float((now - pd.to_datetime(open_["created_at"]).min()).total_seconds() / 60), 1)
            if not open_.empty else None,
            "closed": int(len(closed)),
            "sla_attainment": round(float((~closed["sla_breached"].astype(bool)).mean()), 4) if not closed.empty else None,
            "median_minutes_to_close": round(float(np.median(minutes)), 1) if len(minutes) else None,
            "dispositions": closed["disposition"].value_counts().to_dict() if not closed.empty else {},
            "fraud_confirmation_rate": round(float((decided["disposition"] == "confirmed_fraud").mean()), 4)
            if not decided.empty else None,
        }
    return out


def check_sla(db: Session, notify: bool = True, now: Optional[datetime] = None) -> Dict:
    """Alert once for each open case past its SLA."""
    now = now or datetime.utcnow()
    breached = (db.query(FraudCase).filter(FraudCase.status.in_(OPEN_STATES), FraudCase.sla_due_at < now,
                                           FraudCase.breach_alerted_at.is_(None)).all())
    result = {"checked_at": now.isoformat(), "new_breaches": [c.id for c in breached], "alert": None}
    if breached:
        for c in breached:
            c.breach_alerted_at = now
        if notify:
            from bti.alerts import AlertDispatcher
            result["alert"] = AlertDispatcher().dispatch_event(
                "CASE_SLA_BREACH", "WARNING",
                {"cases": [{"id": c.id, "queue": c.queue, "transaction_id": c.transaction_id,
                            "minutes_over": round((now - c.sla_due_at).total_seconds() / 60, 1)} for c in breached]})
        db.add(AuditLog(ts=now, event_type="CASE_SLA_BREACH", payload={"cases": result["new_breaches"]}))
        db.commit()
    return result
