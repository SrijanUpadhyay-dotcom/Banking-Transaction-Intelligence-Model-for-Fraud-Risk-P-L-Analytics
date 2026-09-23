"""
BTI → SAS Enrichment Engine

Orchestrates BTI's intelligence stack — the governed v3 model, expected-cost
decisioning, exact SHAP explanations and graph ring detection — into a single
enrichment result that SAS (or any external fraud platform) can consume as a
REST enrichment call.

SAS calls /api/v1/sas/enrich with a transaction; BTI returns:
  - Second-opinion calibrated fraud probability (reported 0–100)
  - Top SHAP drivers and reason codes (why BTI scored it that way)
  - Ring membership (is this transaction part of a coordinated fraud ring?)
  - Plain-English investigator note
  - Recommended action (BLOCK / HOLD / MONITOR / ALLOW)

This is the "complement SAS, don't compete" architecture: SAS keeps its own
score; BTI adds network intelligence and explainability that SAS cannot do.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bti.graph.ring_detector import detect_fraud_rings
from bti.logging_config import get_logger
from bti.scoring.realtime import RealTimeScorer, ScoreResult
from bti.scoring.v3_adapter import legacy_action, top_drivers

log = get_logger("sas.enricher")

_scorer = RealTimeScorer()


# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclass
class ShapDriver:
    feature:       str
    label:         str
    direction:     str    # "increases_risk" | "decreases_risk"
    impact_pct:    float  # contribution as % of total SHAP mass


@dataclass
class RingInfo:
    in_ring:           bool
    ring_id:           Optional[str]      = None
    ring_alert_tier:   Optional[str]      = None
    ring_risk_score:   Optional[float]    = None
    ring_size:         Optional[int]      = None
    ring_total_exposure: Optional[float]  = None
    ring_shared_entities: Optional[List[str]] = None


@dataclass
class EnrichmentResult:
    enrichment_id:       str
    transaction_id:      str
    bti_risk_score:      float
    bti_alert_tier:      str
    bti_recommended_action: str
    top_drivers:         List[ShapDriver]
    ring:                RingInfo
    notes_for_sas:       str          # plain-English investigator note (no LLM)
    model_version:       str
    rules_fired:         List[str]
    processing_time_ms:  float
    enriched_at:         str
    bti_fraud_probability: Optional[float] = None
    bti_decision:          Optional[str] = None     # APPROVE / STEP_UP / REVIEW / DECLINE
    bti_model_provisional: bool = False
    reason_codes:          List[dict] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _recommended_action(sr: ScoreResult) -> str:
    return legacy_action(sr.decision, sr.fraud_probability)


def _normalise(txn: dict) -> dict:
    """Accept IEEE-CIS style keys (TransactionID, TransactionAmt) alongside BTI keys."""
    out = dict(txn)
    if not out.get("transaction_id") and out.get("TransactionID") is not None:
        out["transaction_id"] = str(out["TransactionID"])
    if not out.get("transaction_amount") and out.get("TransactionAmt") is not None:
        out["transaction_amount"] = float(out["TransactionAmt"])
    out["transaction_id"] = str(out.get("transaction_id") or "UNKNOWN")
    return out


def _generate_note(score_result: ScoreResult, ring: RingInfo) -> str:
    """
    Build a short plain-English investigator note from structured data alone —
    no LLM call, so latency stays low. For deeper analysis use /copilot/ask.
    """
    tier = score_result.final_alert_tier
    action = _recommended_action(score_result)
    reasons = ", ".join(r["analyst_text"].lower() for r in score_result.reason_codes[:2])
    note = (
        f"BTI rates this transaction {tier} ({score_result.fraud_probability:.1%} fraud probability). "
        f"Recommended action: {action}."
        f"{' Main drivers: ' + reasons + '.' if reasons else ''}"
    )
    if score_result.provisional:
        note += " BTI model is provisional pending champion approval."
    if ring.in_ring:
        note += (
            f" RING ALERT: transaction is part of fraud ring {ring.ring_id} "
            f"({ring.ring_size} linked txns, total exposure "
            f"${ring.ring_total_exposure:,.0f}, ring tier {ring.ring_alert_tier}). "
            "Escalate network-level review."
        )
    return note


def _extract_top_drivers(score_result: ScoreResult) -> List[ShapDriver]:
    """Top five exact SHAP drivers from the v3 score."""
    return [ShapDriver(feature=d["feature"], label=d["label"], direction=d["direction"],
                       impact_pct=d["impact_pct"])
            for d in top_drivers(score_result.contributions, score_result.feature_values, n=5)]


def _result(sr: ScoreResult, ring: RingInfo, elapsed_ms: float) -> EnrichmentResult:
    return EnrichmentResult(
        enrichment_id          = f"ENR-{uuid.uuid4().hex[:8].upper()}",
        transaction_id         = sr.transaction_id,
        bti_risk_score         = sr.final_risk_score,
        bti_alert_tier         = sr.final_alert_tier,
        bti_recommended_action = _recommended_action(sr),
        top_drivers            = _extract_top_drivers(sr),
        ring                   = ring,
        notes_for_sas          = _generate_note(sr, ring),
        model_version          = sr.model_version,
        rules_fired            = sr.rules_fired,
        processing_time_ms     = elapsed_ms,
        enriched_at            = datetime.now(timezone.utc).isoformat(),
        bti_fraud_probability  = sr.fraud_probability,
        bti_decision           = sr.decision,
        bti_model_provisional  = sr.provisional,
        reason_codes           = sr.reason_codes,
    )


# ── Main enrichment functions ─────────────────────────────────────────────────

def enrich_single(txn: dict, db_session=None) -> EnrichmentResult:
    """
    Enrich a single transaction with BTI intelligence.
    Ring detection is skipped (requires at least 2 transactions for a graph).
    For ring intelligence, use enrich_batch or the /graph/analyze endpoint.
    """
    t0 = time.time()
    score_result = _scorer.score(_normalise(txn), db_session=db_session)
    return _result(score_result, RingInfo(in_ring=False), round((time.time() - t0) * 1000, 1))


def enrich_batch(
    transactions: List[dict],
    db_session=None,
) -> List[EnrichmentResult]:
    """
    Enrich a batch of transactions with BTI intelligence, including ring detection.

    Ring detection runs across the full batch — any two transactions sharing a
    device, IP, card token, or address are flagged as ring members.
    """
    t_batch_start = time.time()
    transactions = [_normalise(t) for t in transactions]

    # Score all transactions
    scored: Dict[str, ScoreResult] = {}
    for txn in transactions:
        result = _scorer.score(txn, db_session=db_session)
        scored[result.transaction_id] = result

    # Add BTI scores back to transaction dicts so ring scorer can read them
    enriched_txns = []
    for txn in transactions:
        sr = scored.get(txn["transaction_id"])
        if sr:
            txn_copy = dict(txn)
            txn_copy["final_risk_score"] = sr.final_risk_score
            txn_copy["final_alert_tier"] = sr.final_alert_tier
            enriched_txns.append(txn_copy)
        else:
            enriched_txns.append(txn)

    # Run ring detection across the batch
    ring_result = detect_fraud_rings(enriched_txns, min_ring_size=2)

    # Build a lookup: transaction_id → ring it belongs to
    txn_to_ring: Dict[str, Any] = {}
    for ring in ring_result.rings:
        for member in ring.members:
            txn_to_ring[member.transaction_id] = ring

    # Build enrichment results
    results: List[EnrichmentResult] = []
    for txn in transactions:
        t0 = time.time()
        tid = txn["transaction_id"]
        sr = scored.get(tid)

        if sr is None:
            continue

        detected_ring = txn_to_ring.get(tid)
        if detected_ring:
            ring_info = RingInfo(
                in_ring             = True,
                ring_id             = detected_ring.ring_id,
                ring_alert_tier     = detected_ring.ring_alert_tier,
                ring_risk_score     = detected_ring.ring_risk_score,
                ring_size           = detected_ring.size,
                ring_total_exposure = detected_ring.total_amount,
                ring_shared_entities = detected_ring.shared_entities[:10],
            )
        else:
            ring_info = RingInfo(in_ring=False)

        results.append(_result(sr, ring_info, round((time.time() - t0) * 1000, 1)))

    log.info(
        "Batch enrichment complete",
        extra={
            "transactions": len(transactions),
            "rings_detected": ring_result.rings_detected,
            "in_rings": ring_result.transactions_in_rings,
            "total_ms": round((time.time() - t_batch_start) * 1000, 1),
        },
    )
    return results
