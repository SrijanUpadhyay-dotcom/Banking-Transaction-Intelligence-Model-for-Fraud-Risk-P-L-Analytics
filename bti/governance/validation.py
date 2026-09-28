"""
Validation workflow (SR 11-7 / PRA SS1/23): model inventory, findings tracker,
independent sign-offs, periodic review schedule, and the champion gate that
ties them together.

**Separation of roles.**
- The model developer cannot sign off their own model.
- The person who owns a finding's fix cannot close it or accept its risk.

**Champion gate.** A model becomes champion only when all of these hold:
1. It passed every automated validation gate.
2. An in-date approving sign-off exists from a validator other than the
   developer.
3. It has no open high-severity finding.
4. The approver differs from the developer (checked in the registry).

**Periodic review.** A sign-off is valid for one year (Tier 1). The weekly
governance check alerts on findings past their due date and on reviews due
within 30 days or overdue.

Every action is written to the hash-chained audit log.

Usage (seed the developer's known limitations for a model):
  python -m bti.governance.validation seed --model bti-v3-lgbm-rn-20260925011359 --raised-by "Srijan Upadhyay"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from bti.database.models import AuditLog, ModelInventoryEntry, ValidationFinding, ValidationSignoff
from bti.logging_config import get_logger
from bti.modeling import registry

log = get_logger("governance.validation")

SEVERITIES = {"high": 30, "medium": 90, "low": 180}          # default days to remediate
CATEGORIES = ("data", "methodology", "performance", "fairness", "robustness", "implementation", "governance",
              "monitoring")
SOURCES = ("independent_validation", "development", "monitoring", "internal_audit", "regulator")
STATUSES = ("open", "remediating", "closed", "risk_accepted")
DECISIONS = ("approve", "approve_with_conditions", "reject")
APPROVING = ("approve", "approve_with_conditions")
SIGNOFF_VALID_DAYS = 365
REVIEW_WARNING_DAYS = 30


class ValidationError(ValueError):
    pass


def _audit(db: Session, event: str, payload: Dict) -> None:
    db.add(AuditLog(ts=datetime.utcnow(), event_type=event, payload=payload))


def _required(**values) -> None:
    missing = [k for k, v in values.items() if v is None or (isinstance(v, str) and not v.strip())]
    if missing:
        raise ValidationError(f"Required: {', '.join(missing)}")


def _row(obj) -> Dict:
    return {c.name: (getattr(obj, c.name).isoformat() if isinstance(getattr(obj, c.name), datetime)
                     else getattr(obj, c.name)) for c in obj.__table__.columns}


def _developer(model_id: str) -> Optional[str]:
    return registry.load_card(model_id).get("ownership", {}).get("developer")


def _same(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b and a.strip().lower() == b.strip().lower())


# ── Inventory ────────────────────────────────────────────────────────────────

def sync_inventory(db: Session) -> List[Dict]:
    index = registry.read_index()
    roles = {index.get("champion"): "champion", index.get("challenger"): "challenger"}
    for m in index["models"]:
        model_id = m["model_id"]
        card = registry.load_card(model_id)
        role = roles.get(model_id, "none")
        lifecycle = {"champion": "in_use", "challenger": "shadow_challenger"}.get(role, "registered")
        if role == "none" and any(h.get("previous") == model_id for h in index.get("history", [])):
            lifecycle = "superseded"
        latest = latest_signoff(db, model_id)
        entry = db.get(ModelInventoryEntry, model_id) or ModelInventoryEntry(model_id=model_id)
        entry.model_family = card.get("model_family")
        entry.algorithm = card.get("methodology", {}).get("algorithm_key", "hgb")
        entry.feature_set = card.get("features", {}).get("feature_set", "core")
        entry.developer = card.get("ownership", {}).get("developer")
        entry.business_owner = card.get("ownership", {}).get("business_owner")
        entry.role, entry.lifecycle = role, lifecycle
        entry.validation_status = card.get("validation", {}).get("status")
        entry.registered_at = datetime.fromisoformat(m["registered_at"]).replace(tzinfo=None)
        if latest:
            entry.last_signoff, entry.last_validator = latest.decision, latest.validator
            entry.last_validated_at = latest.signed_at
            entry.next_review_due = latest.valid_until if latest.decision in APPROVING else None
        entry.updated_at = datetime.utcnow()
        db.merge(entry)
    db.commit()
    return [_row(e) for e in db.query(ModelInventoryEntry).order_by(ModelInventoryEntry.registered_at.desc()).all()]


# ── Findings ─────────────────────────────────────────────────────────────────

def raise_finding(db: Session, model_id: str, title: str, severity: str, category: str, source: str,
                  description: str, raised_by: str, owner: str, due_date: Optional[datetime] = None) -> Dict:
    _required(model_id=model_id, title=title, description=description, raised_by=raised_by, owner=owner)
    registry.load_card(model_id)                                  # unknown model → RegistryError
    if severity not in SEVERITIES:
        raise ValidationError(f"severity must be one of {list(SEVERITIES)}")
    if category not in CATEGORIES or source not in SOURCES:
        raise ValidationError(f"category must be one of {CATEGORIES}; source one of {SOURCES}")
    now = datetime.utcnow()
    f = ValidationFinding(model_id=model_id, title=title.strip(), severity=severity, category=category, source=source,
                          description=description.strip(), raised_by=raised_by.strip(), raised_at=now,
                          owner=owner.strip(), due_date=due_date or now + timedelta(days=SEVERITIES[severity]),
                          status="open", updated_at=now)
    db.add(f)
    db.flush()
    _audit(db, "FINDING_RAISED", _row(f))
    db.commit()
    return _row(f)


def update_finding(db: Session, finding_id: int, status: str, actor: str, resolution: Optional[str] = None,
                   evidence: Optional[Dict] = None, accepted_until: Optional[datetime] = None) -> Dict:
    f = db.get(ValidationFinding, finding_id)
    if f is None:
        raise ValidationError(f"No finding {finding_id}")
    if status not in STATUSES:
        raise ValidationError(f"status must be one of {STATUSES}")
    if f.status in ("closed", "risk_accepted"):
        raise ValidationError(f"Finding {finding_id} is already {f.status}; raise a new finding to reopen the issue")
    _required(actor=actor)
    now = datetime.utcnow()
    if status == "closed":
        _required(resolution=resolution)
        if not evidence:
            raise ValidationError("Closing a finding needs evidence (test results, document references, commit)")
        if _same(actor, f.owner):
            raise ValidationError("Four-eyes: the finding's owner cannot close it; an independent reviewer must")
        f.closed_by, f.closed_at, f.resolution, f.evidence = actor.strip(), now, resolution.strip(), evidence
    elif status == "risk_accepted":
        _required(resolution=resolution, accepted_until=accepted_until)
        if _same(actor, f.owner) or _same(actor, _developer(f.model_id)):
            raise ValidationError("Risk acceptance must come from someone other than the owner and the developer")
        if f.severity == "high" and accepted_until > now + timedelta(days=180):
            raise ValidationError("High-severity risk acceptance is limited to 180 days")
        f.accepted_by, f.accepted_until, f.resolution = actor.strip(), accepted_until, resolution.strip()
        f.evidence = evidence
    f.status, f.updated_at = status, now
    _audit(db, "FINDING_UPDATED", {"id": f.id, "status": status, "actor": actor, "resolution": resolution})
    db.commit()
    return _row(f)


def list_findings(db: Session, model_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
    q = db.query(ValidationFinding)
    if model_id:
        q = q.filter(ValidationFinding.model_id == model_id)
    if status:
        q = q.filter(ValidationFinding.status == status)
    now = datetime.utcnow()
    out = []
    for f in q.order_by(ValidationFinding.id).all():
        row = _row(f)
        row["overdue"] = f.status in ("open", "remediating") and f.due_date < now
        out.append(row)
    return out


def _open_high(db: Session, model_id: str) -> List[ValidationFinding]:
    return (db.query(ValidationFinding).filter(ValidationFinding.model_id == model_id,
                                               ValidationFinding.severity == "high",
                                               ValidationFinding.status.in_(("open", "remediating"))).all())


# ── Sign-offs ────────────────────────────────────────────────────────────────

def record_signoff(db: Session, model_id: str, validator: str, decision: str, scope: str,
                   validator_role: Optional[str] = None, conditions: Optional[str] = None,
                   evidence: Optional[Dict] = None) -> Dict:
    _required(model_id=model_id, validator=validator, scope=scope)
    developer = _developer(model_id)
    if decision not in DECISIONS:
        raise ValidationError(f"decision must be one of {DECISIONS}")
    if _same(validator, developer):
        raise ValidationError("Independence: the model developer cannot validate their own model")
    if decision == "approve_with_conditions":
        _required(conditions=conditions)
    if decision in APPROVING:
        blocking = _open_high(db, model_id)
        if blocking:
            raise ValidationError(f"Cannot approve with open high-severity findings: "
                                  f"{[(f.id, f.title) for f in blocking]}")
        if registry.load_card(model_id).get("validation", {}).get("status") != "passed":
            raise ValidationError("Cannot approve a model that failed its automated validation gates")
    now = datetime.utcnow()
    s = ValidationSignoff(model_id=model_id, validator=validator.strip(), validator_role=validator_role,
                          decision=decision, scope=scope.strip(), conditions=conditions, evidence=evidence,
                          signed_at=now, valid_until=now + timedelta(days=SIGNOFF_VALID_DAYS))
    db.add(s)
    db.flush()
    _audit(db, "VALIDATION_SIGNOFF", _row(s))
    db.commit()
    return _row(s)


def latest_signoff(db: Session, model_id: str) -> Optional[ValidationSignoff]:
    return (db.query(ValidationSignoff).filter(ValidationSignoff.model_id == model_id)
            .order_by(ValidationSignoff.signed_at.desc(), ValidationSignoff.id.desc()).first())


def list_signoffs(db: Session, model_id: Optional[str] = None) -> List[Dict]:
    q = db.query(ValidationSignoff)
    if model_id:
        q = q.filter(ValidationSignoff.model_id == model_id)
    return [_row(s) for s in q.order_by(ValidationSignoff.id).all()]


# ── Champion gate and review schedule ────────────────────────────────────────

def champion_readiness(db: Session, model_id: str) -> Dict:
    card = registry.load_card(model_id)
    latest = latest_signoff(db, model_id)
    now = datetime.utcnow()
    developer = card.get("ownership", {}).get("developer")
    checks = [
        {"check": "automated_gates", "passed": card.get("validation", {}).get("status") == "passed",
         "detail": f"validation status {card.get('validation', {}).get('status')}"},
        {"check": "independent_signoff",
         "passed": bool(latest and latest.decision in APPROVING and latest.valid_until > now
                        and not _same(latest.validator, developer)),
         "detail": (f"latest sign-off: {latest.decision} by {latest.validator}, valid until "
                    f"{latest.valid_until:%Y-%m-%d}") if latest else "no validation sign-off recorded"},
        {"check": "no_open_high_findings", "passed": not _open_high(db, model_id),
         "detail": f"{len(_open_high(db, model_id))} open high-severity findings"},
    ]
    conditions = latest.conditions if latest and latest.decision == "approve_with_conditions" else None
    return {"model_id": model_id, "ready": all(c["passed"] for c in checks), "checks": checks,
            "conditions": conditions, "developer": developer}


def assert_ready_for_champion(db: Session, model_id: str) -> None:
    r = champion_readiness(db, model_id)
    if not r["ready"]:
        failed = "; ".join(f"{c['check']}: {c['detail']}" for c in r["checks"] if not c["passed"])
        raise ValidationError(f"{model_id} is not ready to become champion — {failed}")


def review_schedule(db: Session, now: Optional[datetime] = None) -> Dict:
    now = now or datetime.utcnow()
    sync_inventory(db)
    rows = []
    for e in db.query(ModelInventoryEntry).filter(ModelInventoryEntry.role.in_(("champion", "challenger"))).all():
        if e.next_review_due is None:
            state = "never_validated"
        elif e.next_review_due < now:
            state = "overdue"
        elif e.next_review_due < now + timedelta(days=REVIEW_WARNING_DAYS):
            state = "due_soon"
        else:
            state = "current"
        rows.append({"model_id": e.model_id, "role": e.role, "risk_tier": e.risk_tier,
                     "last_validated_at": e.last_validated_at.isoformat() if e.last_validated_at else None,
                     "next_review_due": e.next_review_due.isoformat() if e.next_review_due else None,
                     "state": state})
    overdue_findings = [f for f in list_findings(db) if f["overdue"]]
    return {"as_of": now.isoformat(), "models": rows, "overdue_findings": overdue_findings}


def run_governance_check(db: Session, notify: bool = True) -> Dict:
    schedule = review_schedule(db)
    issues = [f"{m['model_id']} ({m['role']}): review {m['state'].replace('_', ' ')}" for m in schedule["models"]
              if m["state"] in ("overdue", "due_soon") or (m["role"] == "champion" and m["state"] == "never_validated")]
    issues += [f"Finding {f['id']} ({f['severity']}) on {f['model_id']} is past its due date: {f['title']}"
               for f in schedule["overdue_findings"]]
    result = {"checked_at": datetime.utcnow().isoformat(), "issues": issues, "schedule": schedule, "alert": None}
    if notify and issues:
        from bti.alerts import AlertDispatcher
        result["alert"] = AlertDispatcher().dispatch_event("MODEL_GOVERNANCE", "WARNING", {"issues": issues})
    _audit(db, "GOVERNANCE_CHECK", {"issues": issues})
    db.commit()
    return result


# ── Known limitations raised by the developer ────────────────────────────────

KNOWN_LIMITATIONS = [
    ("Performance established on synthetic data only", "high", "data",
     "All discrimination, calibration and fairness evidence comes from a synthetic generator whose fraud is driven "
     "by a few simple signals. Re-establish every figure on the bank's own labelled history before production use."),
    ("First-ever balance-draining transaction not covered by a balance-ratio feature", "medium", "methodology",
     "amount_to_balance was removed to fix the Private Banking disparity. A new customer's first transaction that "
     "drains most of the balance is no longer flagged by that feature. Re-test account takeover of new customers on "
     "bank data; core-relative-own is the registered alternative."),
    ("Private Banking residual in one window", "low", "fairness",
     "Legitimate Private Banking customers are flagged 1.35x the overall rate in the calibration window alone at "
     "the 10% stress budget (pooled 1.13x). Re-test on matured production labels."),
    ("German customers need German data before a German deployment", "medium", "fairness",
     "The German finding was a multiple-testing artefact, but no real German customer data has been tested. "
     "Required before any deployment in Germany."),
    ("Feed-dependent signals unvalidated", "low", "data",
     "Impossible travel, payee velocity and security-event features are built and tested for plumbing only; "
     "their predictive value is unmeasured until the bank supplies the feeds."),
    ("Cost-model and capacity figures are illustrative", "medium", "methodology",
     "Decision costs (loss given fraud, friction, analyst cost) and the 2% review / 5% step-up budgets are "
     "defaults. Calibrate them with the bank's loss, operations and customer data."),
]


def seed_known_limitations(db: Session, model_id: str, raised_by: str, owner: Optional[str] = None,
                           include_benchmark: bool = True) -> List[Dict]:
    """Raise the developer's known limitations plus the findings the latest benchmark report supports."""
    existing = {f.title for f in db.query(ValidationFinding).filter(ValidationFinding.model_id == model_id).all()}
    items = list(KNOWN_LIMITATIONS)
    if include_benchmark:
        from bti.governance.benchmarking import latest_report
        report = latest_report(model_id)
        if report:
            items += [(f["title"], f["severity"], f["category"], f["description"])
                      for f in report.get("proposed_findings", [])]
    return [raise_finding(db, model_id, title, severity, category, "development", description, raised_by,
                          owner or raised_by)
            for title, severity, category, description in items if title not in existing]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation workflow utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    seed = sub.add_parser("seed", help="Raise the developer's known limitations as findings")
    seed.add_argument("--model", required=True)
    seed.add_argument("--raised-by", required=True)
    sub.add_parser("schedule", help="Show the review schedule and overdue findings")
    args = parser.parse_args()
    from bti.database.connection import SessionLocal
    from bti.database.init_db import create_tables
    create_tables()
    db = SessionLocal()
    try:
        if args.cmd == "seed":
            for f in seed_known_limitations(db, args.model, args.raised_by):
                print(f"#{f['id']} [{f['severity']}] {f['title']} — due {f['due_date'][:10]}")
        else:
            import json
            print(json.dumps(review_schedule(db), indent=2, default=str))
    finally:
        db.close()


if __name__ == "__main__":
    main()
