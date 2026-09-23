"""
Legacy response contract over the v3 model.

/score, /score/explain, /score/upload and the SAS enrichment API keep their
response shape but are scored by the registered v3 model and the expected-cost
decision engine. The 19-rule engine still runs and is reported to analysts,
but no longer contributes to the score; rules R06, R12 and R17 are evaluated
with risk_score withheld because that field is label-derived in the training
data. The legacy ML fields (ml_lr_proba, ml_rf_proba, ml_iso_score) are null.
"""

from __future__ import annotations

import time
from typing import List, Optional

from bti.modeling.features import FEATURE_BY_NAME
from bti.operations.scoring_service import ScoredDecision, prepare_transaction, score_and_decide
from bti.scoring.realtime import (
    RULE_WEIGHTS, TOTAL_WEIGHT, ScoreResult, _apply_db_rules, _apply_standalone_rules,
)

TIER_BANDS = ((0.90, "CRITICAL"), (0.70, "VERY HIGH"), (0.50, "HIGH"), (0.10, "MEDIUM"))
RULES_NEEDING_RISK_SCORE = ("R06_geo_mismatch", "R12_device_ip_mismatch", "R17_cross_border")

_RECOMMENDATION = {
    "DECLINE": "BLOCK — Decline and escalate to Fraud Operations",
    "REVIEW": "REVIEW — Hold for analyst review before processing",
    "STEP_UP": "HOLD — Step-up authentication required before processing",
}
_LEGACY_ACTION = {"DECLINE": "BLOCK", "REVIEW": "HOLD", "STEP_UP": "HOLD"}


def tier_for(p: float) -> str:
    for cut, tier in TIER_BANDS:
        if p >= cut:
            return tier
    return "LOW"


def recommendation(decision: str, p: float) -> str:
    if decision in _RECOMMENDATION:
        return _RECOMMENDATION[decision]
    if p >= 0.10:
        return "MONITOR — Allow with enhanced monitoring"
    return "ALLOW — Transaction within normal risk parameters"


def legacy_action(decision: str, p: float) -> str:
    """Map the v3 decision onto the SAS-facing BLOCK / HOLD / MONITOR / ALLOW vocabulary."""
    return _LEGACY_ACTION.get(decision) or ("MONITOR" if p >= 0.10 else "ALLOW")


def top_drivers(contributions: dict, values: dict, n: int = 8) -> List[dict]:
    total = sum(abs(v) for v in contributions.values()) or 1.0
    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:n]
    return [{
        "rank": i,
        "feature": f,
        "label": FEATURE_BY_NAME[f].description if f in FEATURE_BY_NAME else f,
        "value": values.get(f),
        "shap_value": round(v, 6),
        "direction": "increases_risk" if v > 0 else "decreases_risk",
        "impact_pct": round(abs(v) / total * 100, 1),
    } for i, (f, v) in enumerate(ranked, start=1)]


def score_legacy_contract(txn: dict, db_session=None, log_scores: bool = True) -> ScoreResult:
    t0 = time.perf_counter()
    txn = prepare_transaction(txn)
    rules_txn = {**txn, "risk_score": 0}
    flags = {**_apply_standalone_rules(rules_txn), **_apply_db_rules(rules_txn, db_session)}
    fired = [r for r, v in flags.items() if v == 1]
    rule_score = round(sum(RULE_WEIGHTS[r] * v for r, v in flags.items() if r in RULE_WEIGHTS) / TOTAL_WEIGHT * 100, 2)

    sd: ScoredDecision = score_and_decide(txn, db_session, explain=True, log_scores=log_scores)
    p = sd.live.fraud_probability
    return ScoreResult(
        transaction_id=sd.live.transaction_id,
        fraud_rule_score=rule_score,
        rules_triggered=len(fired),
        rules_fired=fired,
        ml_iso_score=None,
        ml_lr_proba=None,
        ml_rf_proba=None,
        ml_anomaly_score=round(p * 100, 2),
        final_risk_score=round(p * 100, 2),
        final_alert_tier=tier_for(p),
        is_suspicious=sd.decision.action != "APPROVE",
        processing_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        risk_score_input=txn.get("risk_score"),
        model_version=sd.live.model_id,
        db_context_used=db_session is not None,
        fraud_probability=p,
        decision=sd.decision.action,
        guardrails=sd.decision.guardrails_applied,
        reason_codes=sd.live.reason_codes,
        contributions=sd.live.contributions,
        feature_values=sd.live.features,
        base_probability=sd.live.base_probability,
        provisional=sd.live.provisional,
        jurisdiction=sd.policy.iso2 if sd.policy else None,
        notes=sd.live.notes,
    )
