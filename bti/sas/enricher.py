"""
BTI → SAS Enrichment Engine
Phase 4 of BTI v2 (Fraud Intelligence Layer)

Orchestrates BTI's full intelligence stack — ML scoring, SHAP explanations,
and graph ring detection — into a single enrichment result that SAS (or any
external fraud platform) can consume as a REST enrichment call.

SAS calls /api/v1/sas/enrich with a transaction; BTI returns:
  - Second opinion risk score (ML ensemble, 0–100)
  - Top SHAP drivers (why BTI flagged it)
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

import numpy as np

from bti.graph.ring_detector import detect_fraud_rings
from bti.logging_config import get_logger
from bti.scoring.explainer import explain as shap_explain
from bti.scoring.realtime import RealTimeScorer, ScoreResult, _build_feature_vector

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _recommended_action(tier: str) -> str:
    return {
        "CRITICAL": "BLOCK",
        "VERY HIGH": "BLOCK",
        "HIGH":      "HOLD",
        "MEDIUM":    "MONITOR",
        "LOW":       "ALLOW",
    }.get(tier, "MONITOR")


def _generate_note(score_result: ScoreResult, ring: RingInfo) -> str:
    """
    Build a short plain-English investigator note from structured data alone —
    no LLM call, so latency stays low. For deeper analysis use /copilot/ask.
    """
    tier = score_result.final_alert_tier
    action = _recommended_action(tier)
    note = (
        f"BTI rates this transaction {tier} ({score_result.final_risk_score:.0f}/100). "
        f"Recommended action: {action}. "
        f"{score_result.rules_triggered} fraud rule(s) fired"
        f"{': ' + ', '.join(score_result.rules_fired[:3]) if score_result.rules_fired else ''}."
    )
    if ring.in_ring:
        note += (
            f" RING ALERT: transaction is part of fraud ring {ring.ring_id} "
            f"({ring.ring_size} linked txns, total exposure "
            f"${ring.ring_total_exposure:,.0f}, ring tier {ring.ring_alert_tier}). "
            "Escalate network-level review."
        )
    return note


def _extract_top_drivers(
    score_result: ScoreResult,
    txn: dict,
) -> List[ShapDriver]:
    """Run SHAP and return top 5 drivers. Falls back to empty list on error."""
    try:
        bundle = _scorer.bundle
        if bundle is None:
            return []
        X = _build_feature_vector(txn, bundle.feature_cols, bundle.label_encoders)
        X = X.fillna(0).replace([float("inf"), float("-inf")], 0)
        explanation = shap_explain(
            rf_model=bundle.rf_model,
            feature_vector=X.values,
            feature_cols=bundle.feature_cols,
            transaction_id=score_result.transaction_id,
            fraud_prob=score_result.ml_rf_proba,
            risk_tier=score_result.final_alert_tier,
            top_n=5,
        )
        return [
            ShapDriver(
                feature=d.feature,
                label=d.label,
                direction=d.direction,
                impact_pct=round(abs(d.impact_pct), 1),
            )
            for d in explanation.top_drivers
        ]
    except Exception as e:
        log.warning("SHAP extraction failed, returning empty drivers", extra={"error": str(e)})
        return []


# ── Main enrichment functions ─────────────────────────────────────────────────

def enrich_single(txn: dict, db_session=None) -> EnrichmentResult:
    """
    Enrich a single transaction with BTI intelligence.
    Ring detection is skipped (requires at least 2 transactions for a graph).
    For ring intelligence, use enrich_batch or the /graph/analyze endpoint.
    """
    t0 = time.time()

    score_result = _scorer.score(txn, db_session=db_session)
    top_drivers = _extract_top_drivers(score_result, txn)
    ring = RingInfo(in_ring=False)

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    return EnrichmentResult(
        enrichment_id      = f"ENR-{uuid.uuid4().hex[:8].upper()}",
        transaction_id     = score_result.transaction_id,
        bti_risk_score     = score_result.final_risk_score,
        bti_alert_tier     = score_result.final_alert_tier,
        bti_recommended_action = _recommended_action(score_result.final_alert_tier),
        top_drivers        = top_drivers,
        ring               = ring,
        notes_for_sas      = _generate_note(score_result, ring),
        model_version      = score_result.model_version,
        rules_fired        = score_result.rules_fired,
        processing_time_ms = elapsed_ms,
        enriched_at        = datetime.now(timezone.utc).isoformat(),
    )


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

    # Score all transactions
    scored: Dict[str, ScoreResult] = {}
    for txn in transactions:
        result = _scorer.score(txn, db_session=db_session)
        scored[result.transaction_id] = result

    # Add BTI scores back to transaction dicts so ring scorer can read them
    enriched_txns = []
    for txn in transactions:
        tid = str(txn.get("transaction_id") or txn.get("TransactionID") or "UNKNOWN")
        sr = scored.get(tid)
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
        tid = str(txn.get("transaction_id") or txn.get("TransactionID") or "UNKNOWN")
        sr = scored.get(tid)

        if sr is None:
            continue

        top_drivers = _extract_top_drivers(sr, txn)

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

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        results.append(EnrichmentResult(
            enrichment_id          = f"ENR-{uuid.uuid4().hex[:8].upper()}",
            transaction_id         = sr.transaction_id,
            bti_risk_score         = sr.final_risk_score,
            bti_alert_tier         = sr.final_alert_tier,
            bti_recommended_action = _recommended_action(sr.final_alert_tier),
            top_drivers            = top_drivers,
            ring                   = ring_info,
            notes_for_sas          = _generate_note(sr, ring_info),
            model_version          = sr.model_version,
            rules_fired            = sr.rules_fired,
            processing_time_ms     = elapsed_ms,
            enriched_at            = datetime.now(timezone.utc).isoformat(),
        ))

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
