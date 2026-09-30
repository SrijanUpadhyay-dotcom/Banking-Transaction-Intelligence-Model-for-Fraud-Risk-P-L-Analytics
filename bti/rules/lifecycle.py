# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Rule lifecycle: author → simulate → four-eyes approval (active or shadow) → retire.

- **Versions are immutable.** Editing a rule creates a new version as a draft.
  At most one active (champion) and one shadow (challenger) version of a rule
  are live at a time. Approving a version replaces the earlier version in that
  mode.
- **Simulation.** A rule is run on the out-of-time history against the scoring
  model's own capacity-constrained decisions. The simulation reports:
  - hits and daily volume
  - precision
  - fraud caught, and fraud caught that the model alone would have approved
    (the rule's incremental value)
  - genuine customers disturbed
  - fairness of the rule's hits, using the same corrected false-positive-rate
    test as the model

  Approval needs a simulation no older than 30 days.
- **Approval rules.**
  - The approver must differ from the author.
  - A DECLINE rule whose historical precision is under 50% cannot be approved:
    it would decline more genuine customers than fraudsters, the same line the
    model's guardrail draws.
  - A rule with a fairness finding may run only in shadow.
- **Live effect.** An active rule sets a floor on the decision: it can raise
  APPROVE to STEP_UP, REVIEW or DECLINE, never lower one. STEP_UP falls back to
  REVIEW where the channel cannot challenge. A shadow rule is evaluated and
  logged but never changes the outcome.
- **Record.** Every hit is recorded, and matured labels give each version its
  live precision and incremental value.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from bti.database.models import AuditLog, RuleHit, RuleVersion
from bti.logging_config import get_logger
from bti.rules.language import RuleError, describe, evaluate, validate

log = get_logger("rules.lifecycle")

ACTIONS = ("STEP_UP", "REVIEW", "DECLINE")
SEVERITY = {"APPROVE": 0, "STEP_UP": 1, "REVIEW": 2, "DECLINE": 3}
SIMULATION_MAX_AGE_DAYS = 30
MIN_DECLINE_PRECISION = 0.5
_cache = {"at": 0.0, "rules": []}
_cache_lock = threading.Lock()
CACHE_SECONDS = 15


def _audit(db: Session, event: str, payload: Dict) -> None:
    db.add(AuditLog(ts=datetime.utcnow(), event_type=event, payload=payload))


def _row(r: RuleVersion) -> Dict:
    out = {c.name: (getattr(r, c.name).isoformat() if isinstance(getattr(r, c.name), datetime) else getattr(r, c.name))
           for c in RuleVersion.__table__.columns}
    out["readable"] = describe(r.condition)
    return out


def _check(name, description, condition, action, author) -> List[str]:
    for label, v in (("name", name), ("description", description), ("author", author)):
        if not v or not str(v).strip():
            raise RuleError(f"{label} is required")
    if action not in ACTIONS:
        raise RuleError(f"action must be one of {ACTIONS} (rules raise decisions; they never approve)")
    return validate(condition)


def create_rule(db: Session, name: str, description: str, condition: Dict, action: str, author: str,
                expires_at: Optional[datetime] = None) -> Dict:
    fields = _check(name, description, condition, action, author)
    n = db.query(func.count(func.distinct(RuleVersion.rule_id))).scalar() or 0
    rule_id = f"R-{n + 1:04d}"
    while db.query(RuleVersion).filter(RuleVersion.rule_id == rule_id).first():
        n += 1
        rule_id = f"R-{n + 1:04d}"
    r = RuleVersion(rule_id=rule_id, version=1, name=name.strip(), description=description.strip(),
                    condition=condition, action=action, author=author.strip(), expires_at=expires_at, status="draft")
    db.add(r)
    db.flush()
    _audit(db, "RULE_CREATED", {**_row(r), "fields": fields})
    db.commit()
    return _row(r)


def new_version(db: Session, rule_id: str, author: str, condition: Optional[Dict] = None, action: Optional[str] = None,
                name: Optional[str] = None, description: Optional[str] = None,
                expires_at: Optional[datetime] = None) -> Dict:
    latest = (db.query(RuleVersion).filter(RuleVersion.rule_id == rule_id)
              .order_by(RuleVersion.version.desc()).first())
    if latest is None:
        raise RuleError(f"No rule {rule_id}")
    name, description = name or latest.name, description or latest.description
    condition, action = condition or latest.condition, action or latest.action
    _check(name, description, condition, action, author)
    r = RuleVersion(rule_id=rule_id, version=latest.version + 1, name=name, description=description,
                    condition=condition, action=action, author=author.strip(), expires_at=expires_at, status="draft")
    db.add(r)
    db.flush()
    _audit(db, "RULE_VERSION_CREATED", _row(r))
    db.commit()
    return _row(r)


# ── Simulation ───────────────────────────────────────────────────────────────

_history_cache: Dict = {}


def _history(model_id: str) -> pd.DataFrame:
    """Out-of-time history with features, model probability and the model's own decisions (cached)."""
    from bti.jurisdiction.policies import policy_for
    from bti.modeling import registry
    from bti.modeling.reassess import model_probabilities
    from bti.modeling.train import prepare
    from bti.operations.capacity import capacity_overrides
    from bti.operations.decisioning import decide

    if model_id in _history_cache:
        return _history_cache[model_id]
    version = registry.load_artifact(model_id).get("feature_version", 1)
    data = prepare(feature_version=version)
    p = model_probabilities(model_id, data)
    te = data.te
    df, feats = data.df.loc[te].reset_index(drop=True), data.features.loc[te].reset_index(drop=True)
    frame = feats.copy()
    frame["fraud_probability"] = p[te]
    for col in ("currency", "merchant_name", "payee_id", "device_id", "ip_location"):
        frame[col] = df[col] if col in df.columns else None
    capacity = capacity_overrides(model_id)
    frame["model_decision"] = [decide(float(pi), float(a or 0), policy_for(c), ch, tt,
                                      provisional_model=registry.model_for_role("champion") != model_id,
                                      cost_overrides=capacity).action
                               for pi, a, c, ch, tt in zip(frame["fraud_probability"], frame["amount_usd"],
                                                           df["country"], df["channel"], df["transaction_type"])]
    frame["label"] = data.y[te]
    for c in ("customer_segment", "customer_age_band", "country", "transaction_id"):
        frame[f"_{c}"] = df[c].to_numpy()
    frame["_ts"] = df["_ts"].to_numpy()
    _history_cache[model_id] = frame
    return frame


def simulate(db: Session, version_id: int, actor: str) -> Dict:
    from bti.governance.fairness import fairness_report
    from bti.modeling import registry

    r = db.get(RuleVersion, version_id)
    if r is None:
        raise RuleError(f"No rule version {version_id}")
    if r.status == "retired":
        raise RuleError("A retired version cannot be simulated")
    model_id = registry.model_for_role("champion") or registry.model_for_role("challenger")
    h = _history(model_id)
    hit = evaluate(r.condition, h)
    y = h["label"].to_numpy(int)
    amt = pd.to_numeric(h["amount_usd"], errors="coerce").fillna(0).to_numpy(float)
    approved = (h["model_decision"] == "APPROVE").to_numpy()
    raised = hit & np.array([SEVERITY[a] < SEVERITY[r.action] for a in h["model_decision"]])
    days = max((h["_ts"].max() - h["_ts"].min()).days, 1)
    fraud, genuine = y == 1, y == 0
    fair = fairness_report(y, hit, {c: h[f"_{c}"] for c in ("customer_segment", "customer_age_band", "country")})
    sim = {
        "model_id": model_id, "window": "out-of-time history", "transactions": int(len(h)), "days": int(days),
        "hits": int(hit.sum()), "hit_rate": round(float(hit.mean()), 5), "hits_per_day": round(float(hit.sum() / days), 2),
        "precision": round(float(y[hit].mean()), 4) if hit.any() else None,
        "fraud_caught": int((hit & fraud).sum()),
        "fraud_value_caught_usd": round(float(amt[hit & fraud].sum()), 2),
        "decisions_raised": int(raised.sum()),
        "incremental_fraud_caught": int((hit & approved & fraud).sum()),
        "incremental_fraud_value_usd": round(float(amt[hit & approved & fraud].sum()), 2),
        "incremental_genuine_disturbed": int((hit & approved & genuine).sum()),
        "incremental_volume_per_day": round(float((hit & approved).sum() / days), 2),
        "fairness": {"status": fair["status"],
                     "findings": [{k: f[k] for k in ("attribute", "group", "fpr_ratio", "fpr_q_value")}
                                  for f in fair["findings"]]},
    }
    r.simulation, r.simulated_at = sim, datetime.utcnow()
    if r.status == "draft":
        r.status = "simulated"
    _audit(db, "RULE_SIMULATED", {"rule_version_id": r.id, "rule_id": r.rule_id, "version": r.version,
                                  "actor": actor, "summary": {k: sim[k] for k in ("hits", "precision",
                                                                                  "incremental_fraud_caught")}})
    db.commit()
    return _row(r)


# ── Approval and retirement ──────────────────────────────────────────────────

def approve(db: Session, version_id: int, approver: str, mode: str = "active") -> Dict:
    r = db.get(RuleVersion, version_id)
    if r is None:
        raise RuleError(f"No rule version {version_id}")
    if mode not in ("active", "shadow"):
        raise RuleError("mode must be 'active' (enforced) or 'shadow' (logged only)")
    if not approver or not approver.strip():
        raise RuleError("An approver is required")
    if approver.strip().lower() == r.author.strip().lower():
        raise RuleError("Four-eyes: the rule's author cannot approve it")
    if r.status not in ("simulated", "shadow"):
        raise RuleError(f"Version is {r.status}; simulate it on history before approval")
    if not r.simulated_at or datetime.utcnow() - r.simulated_at > timedelta(days=SIMULATION_MAX_AGE_DAYS):
        raise RuleError(f"The simulation is older than {SIMULATION_MAX_AGE_DAYS} days; re-run it")
    sim = r.simulation or {}
    if mode == "active" and r.action == "DECLINE" and (sim.get("precision") or 0) < MIN_DECLINE_PRECISION:
        raise RuleError(f"A DECLINE rule needs at least {MIN_DECLINE_PRECISION:.0%} historical precision "
                        f"(simulated {sim.get('precision')}); use REVIEW or STEP_UP instead")
    if mode == "active" and sim.get("fairness", {}).get("status") == "review_required":
        raise RuleError(f"The rule's hits fail the fairness test {sim['fairness']['findings']}; it can run in shadow "
                        f"only until that is resolved")
    now = datetime.utcnow()
    for other in db.query(RuleVersion).filter(RuleVersion.rule_id == r.rule_id, RuleVersion.status == mode,
                                              RuleVersion.id != r.id).all():
        other.status, other.retired_at, other.retired_by = "retired", now, approver.strip()
        other.retire_reason = f"Replaced by version {r.version}"
    r.status, r.approved_by, r.approved_at = mode, approver.strip(), now
    _audit(db, "RULE_APPROVED", {"rule_version_id": r.id, "rule_id": r.rule_id, "version": r.version, "mode": mode,
                                 "approver": approver, "author": r.author, "condition": describe(r.condition),
                                 "action": r.action})
    db.commit()
    invalidate()
    return _row(r)


def retire(db: Session, version_id: int, actor: str, reason: str) -> Dict:
    r = db.get(RuleVersion, version_id)
    if r is None:
        raise RuleError(f"No rule version {version_id}")
    if r.status == "retired":
        raise RuleError("Already retired")
    if not actor or not reason or len(reason.strip()) < 5:
        raise RuleError("actor and a reason are required")
    r.status, r.retired_by, r.retired_at, r.retire_reason = "retired", actor.strip(), datetime.utcnow(), reason.strip()
    _audit(db, "RULE_RETIRED", {"rule_version_id": r.id, "rule_id": r.rule_id, "version": r.version,
                                "actor": actor, "reason": reason})
    db.commit()
    invalidate()
    return _row(r)


def list_rules(db: Session, include_retired: bool = False) -> List[Dict]:
    q = db.query(RuleVersion)
    if not include_retired:
        q = q.filter(RuleVersion.status != "retired")
    return [_row(r) for r in q.order_by(RuleVersion.rule_id, RuleVersion.version).all()]


def versions(db: Session, rule_id: str) -> List[Dict]:
    return [_row(r) for r in db.query(RuleVersion).filter(RuleVersion.rule_id == rule_id)
            .order_by(RuleVersion.version).all()]


# ── Live evaluation ──────────────────────────────────────────────────────────

def invalidate() -> None:
    with _cache_lock:
        _cache["at"] = 0.0


def live_rules(db: Session) -> List[RuleVersion]:
    with _cache_lock:
        if time.time() - _cache["at"] < CACHE_SECONDS:
            return _cache["rules"]
    now = datetime.utcnow()
    rules = [r for r in db.query(RuleVersion).filter(RuleVersion.status.in_(("active", "shadow"))).all()
             if r.expires_at is None or r.expires_at > now]
    for r in rules:
        db.expunge(r)
    with _cache_lock:
        _cache.update(at=time.time(), rules=rules)
    return rules


def apply_rules(db: Session, values: Dict, decision, can_step_up: bool) -> List[Dict]:
    """Evaluate live rules on one transaction; raise `decision` in place for active hits. Returns the hits."""
    rules = live_rules(db) if db is not None else []
    if not rules:
        return []
    frame = pd.DataFrame([values])
    hits = []
    before = decision.action
    for r in rules:
        if not evaluate(r.condition, frame)[0]:
            continue
        action = r.action if (r.action != "STEP_UP" or can_step_up) else "REVIEW"
        enforced = r.status == "active" and SEVERITY[action] > SEVERITY[decision.action]
        if enforced:
            decision.guardrails_applied.append(
                f"Rule {r.rule_id} v{r.version} '{r.name}' raised the decision from {decision.action} to {action}")
            decision.action = action
        hits.append({"rule_version_id": r.id, "rule_id": r.rule_id, "version": r.version, "name": r.name,
                     "mode": r.status, "action": action, "enforced": enforced})
    for h in hits:
        h["decision_before"], h["decision_after"] = before, decision.action
    return hits


def record_hits(db: Session, transaction_id: str, hits: List[Dict]) -> None:
    for h in hits:
        db.add(RuleHit(transaction_id=transaction_id, rule_version_id=h["rule_version_id"], rule_id=h["rule_id"],
                       version=h["version"], mode=h["mode"], action=h["action"], enforced=h["enforced"],
                       decision_before=h["decision_before"], decision_after=h["decision_after"],
                       at=datetime.utcnow()))


def performance(db: Session, rule_id: str, days: int = 90, maturity_days: int = 90,
                now: Optional[datetime] = None) -> Dict:
    """Live hits per version (champion and challenger) with precision on matured labels."""
    from bti.operations.feedback import latest_labels
    now = now or datetime.utcnow()
    rows = db.query(RuleHit).filter(RuleHit.rule_id == rule_id, RuleHit.at >= now - timedelta(days=days)).all()
    frame = pd.DataFrame([{"transaction_id": h.transaction_id, "version": h.version, "mode": h.mode,
                           "enforced": h.enforced, "decision_before": h.decision_before, "at": h.at} for h in rows])
    out = {"rule_id": rule_id, "window_days": days, "versions": []}
    if frame.empty:
        return out
    labels = latest_labels(db, frame["transaction_id"].unique().tolist())
    frame = frame.merge(labels[["transaction_id", "label"]] if not labels.empty
                        else pd.DataFrame(columns=["transaction_id", "label"]), on="transaction_id", how="left")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    matured = frame["at"] < now - timedelta(days=maturity_days)
    frame.loc[frame["label"].isna() & matured, "label"] = 0.0
    for (version, mode), g in frame.groupby(["version", "mode"]):
        known = g[g["label"].notna()]
        out["versions"].append({
            "version": int(version), "mode": mode, "hits": int(len(g)), "enforced": int(g["enforced"].sum()),
            "hits_on_model_approvals": int((g["decision_before"] == "APPROVE").sum()),
            "labelled": int(len(known)),
            "precision": round(float(known["label"].mean()), 4) if len(known) else None,
            "incremental_fraud_caught": int(((known["decision_before"] == "APPROVE") & (known["label"] == 1)).sum()),
        })
    return out
