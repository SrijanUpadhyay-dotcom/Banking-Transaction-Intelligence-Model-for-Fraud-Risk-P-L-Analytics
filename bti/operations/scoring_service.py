# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
One scoring path for every API: v3 score → expected-cost decision →
shadow challenger → score log.

/v3/score, /score, /score/explain, /score/upload and the SAS enrichment API
all call `score_and_decide`, so every decision is made by the same governed
model and lands in the same audit log that drift monitoring and
champion/challenger reporting read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import null
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import ScoreLog
from bti.jurisdiction.policies import JurisdictionPolicy, policy_for
from bti.logging_config import get_logger
from bti.modeling import registry
from bti.modeling.fx import to_usd
from bti.modeling.scorer import V3Score, scorer
from bti.operations.capacity import capacity_overrides
from bti.operations.decisioning import Decision, decide, step_up_available

log = get_logger("operations.scoring_service")

# Recorded with each score for fairness monitoring on matured outcomes; excluded from the model by lineage.
MONITORING_ATTRIBUTES = ("customer_segment", "customer_age_band", "country")


@dataclass
class ScoredDecision:
    live: V3Score
    decision: Decision
    policy: Optional[JurisdictionPolicy]
    amount_usd: float
    shadow: Optional[dict]
    rules: list = field(default_factory=list)
    step_up: Optional[dict] = None
    explanation: str = "inline"          # inline | pending (async, see bti.operations.explanations) | none
    scam: Optional[dict] = None          # APP-scam overlay for outbound payments (Phase 9)


def model_available() -> bool:
    try:
        scorer.resolve("champion")
        return True
    except registry.RegistryError:
        return False


def warm_up() -> Dict:
    """Load the scoring models and run one score each, so the first live request is not a cold start."""
    import time
    out = {}
    for role in ("champion", "challenger"):
        model_id = registry.model_for_role(role)
        if not model_id:
            continue
        t0 = time.perf_counter()
        txn = prepare_transaction({"transaction_id": "WARM-UP", "customer_id": "WARM-UP", "transaction_amount": 10.0,
                                   "currency": "USD", "channel": "Mobile Banking"})
        scorer.score(txn, history=None, role=role, explain=True)
        capacity_overrides(model_id)
        out[model_id] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def prepare_transaction(txn: dict) -> dict:
    """Fill fields the v3 feature builder needs; missing date means 'now' (real-time scoring)."""
    out = dict(txn)
    if not out.get("transaction_date"):
        out["transaction_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    time = str(out.get("transaction_time") or "12:00:00")
    out["transaction_time"] = time if time.count(":") == 2 else f"{time}:00"
    out["currency"] = (out.get("currency") or "USD").upper()
    if not out.get("customer_id"):
        # never pool unidentified customers into one shared history
        out["customer_id"] = f"UNIDENTIFIED-{out.get('transaction_id', 'UNKNOWN')}"
    return out


def _log(db: Session, s: V3Score, txn: dict, decision: Decision, jurisdiction: Optional[str],
         amount_usd: float, shadow: bool, pending: bool = False, scam: Optional[dict] = None) -> None:
    """Score-log row; `pending` leaves reason_codes null until the asynchronous explanation fills it."""
    db.add(ScoreLog(
        transaction_id=s.transaction_id, customer_id=txn.get("customer_id"), model_id=s.model_id,
        model_role=s.model_role, is_shadow=shadow, fraud_probability=s.fraud_probability, score=s.score,
        decision=decision.action, jurisdiction=jurisdiction, amount_usd=amount_usd,
        reason_codes=(null() if pending else [{k: r[k] for k in ("code", "rank", "share_of_risk")}
                                            for r in s.reason_codes]),
        features=s.features, guardrails=decision.guardrails_applied, latency_ms=s.latency_ms,
        monitoring_attributes={k: txn[k] for k in MONITORING_ATTRIBUTES if txn.get(k)} or None,
        model_probability=s.model_probability, calibration_overlay=s.calibration_overlay,
        scam_model_id=(scam or {}).get("model_id"), scam_probability=(scam or {}).get("probability"),
        scam_exposure_gbp=(scam or {}).get("exposure_gbp"), scam_action=(scam or {}).get("action"),
        scored_at=datetime.utcnow(),
    ))


def score_and_decide(txn: dict, db: Optional[Session] = None, explain: bool = True,
                     log_scores: bool = True) -> ScoredDecision:
    txn = prepare_transaction(txn)
    policy = policy_for(txn.get("country"))
    amount_usd = to_usd(float(txn.get("transaction_amount") or 0), txn["currency"])
    from bti.operations import explanations
    deferred = explain and explanations.async_enabled()
    live = scorer.score(txn, db_session=db, explain=explain and not deferred, record=log_scores and db is not None)
    capacity = capacity_overrides(live.model_id)
    from bti.operations.cost_model import overrides_for
    decision = decide(live.fraud_probability, amount_usd, policy, txn.get("channel"), txn.get("transaction_type"),
                      provisional_model=live.provisional, cost_overrides=overrides_for(db, txn, amount_usd, capacity))
    if capacity is None:
        decision.guardrails_applied.append("No capacity policy fitted for this model: review and step-up volumes "
                                           "are unconstrained (python -m bti.operations.capacity)")
    iso = policy.iso2 if policy else None

    scam = None
    from bti.scams.overlay import apply as apply_scam, assess as assess_scam, is_payment
    if is_payment(txn):
        try:
            from bti.modeling.scorer import fetch_history
            history = fetch_history(db, txn, 365) if db is not None else pd.DataFrame()
            scam = assess_scam(txn, history, db, explain=explain)
            apply_scam(decision, scam)
        except Exception as exc:                       # the overlay must never break a payment decision
            log.error("Scam overlay failed", extra={"transaction_id": txn.get("transaction_id"), "error": str(exc)})

    rule_hits = []
    if db is not None:
        from bti.rules.lifecycle import apply_rules
        values = {**live.features, "fraud_probability": live.fraud_probability, "currency": txn.get("currency"),
                  **{k: txn.get(k) for k in ("merchant_name", "payee_id", "device_id", "ip_location")}}
        rule_hits = apply_rules(db, values, decision, step_up_available(txn.get("channel"), txn.get("transaction_type")))
        if decision.action == "DECLINE" and policy is not None and policy.decline_requires_human_review_route \
                and not decision.human_review_route:
            decision.human_review_route = True
            decision.guardrails_applied.append("GDPR Art. 22 — decline notice must offer human review")

    shadow = None
    challenger_id = registry.model_for_role("challenger")
    if challenger_id and challenger_id != live.model_id:
        sh = scorer.score(txn, db_session=db, role="challenger", explain=False)
        sh_decision = decide(sh.fraud_probability, amount_usd, policy, txn.get("channel"),
                             txn.get("transaction_type"), provisional_model=True,
                             cost_overrides=capacity_overrides(sh.model_id))
        shadow = {"model_id": sh.model_id, "fraud_probability": sh.fraud_probability, "decision": sh_decision.action}
        if db is not None and log_scores:
            _log(db, sh, txn, sh_decision, iso, amount_usd, shadow=True)

    step_up = None
    if db is not None and log_scores:
        try:
            _log(db, live, txn, decision, iso, amount_usd, shadow=False, pending=deferred, scam=scam)
            if rule_hits:
                from bti.rules.lifecycle import record_hits
                record_hits(db, live.transaction_id, rule_hits)
            if decision.action in ("REVIEW", "DECLINE"):
                # Declines need a disposition too: no money moves, so no chargeback will ever label them.
                from bti.operations.cases import open_case
                open_case(db, live.transaction_id, f"live_{decision.action.lower()}", customer_id=txn.get("customer_id"),
                          model_id=live.model_id, fraud_probability=live.fraud_probability, amount_usd=amount_usd,
                          decision=decision.action,
                          reason_codes=(None if deferred else
                                        [{k: r[k] for k in ("code", "analyst_text")} for r in live.reason_codes]),
                          commit=False)
            db.commit()
        except Exception as exc:
            db.rollback()
            log.error("Score log write failed", extra={"transaction_id": live.transaction_id, "error": str(exc)})
        if decision.action == "STEP_UP" and get_settings().stepup_auto_issue:
            from bti.operations.stepup import StepUpError, issue
            try:
                step_up = issue(db, live.transaction_id, txn.get("customer_id"), txn.get("channel"),
                                txn.get("transaction_type"), amount_usd, txn.get("currency") or "USD")
            except StepUpError as exc:
                step_up = {"status": "not_issued", "detail": str(exc)}
            if step_up.get("status") in ("send_failed", "not_issued"):
                decision.guardrails_applied.append("Step-up could not be issued; route the transaction to REVIEW")

    explanation = "inline" if explain else "none"
    if deferred:
        explanations.submit(live.transaction_id, live.model_id, live.model_input, live.features,
                            db if (db is not None and log_scores) else None)
        explanation = "pending"
    return ScoredDecision(live=live, decision=decision, policy=policy, amount_usd=amount_usd, shadow=shadow,
                          rules=rule_hits, step_up=step_up, explanation=explanation, scam=scam)
