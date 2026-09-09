"""
/api/v1/sas — SAS Integration Enrichment API
Phase 4 of BTI v2 (Fraud Intelligence Layer)

Packages BTI intelligence as a REST enrichment endpoint that SAS (or any
external fraud platform) can call with a transaction and receive back:

  - BTI risk score (0–100) and alert tier
  - Top SHAP feature contributions (why BTI flagged it)
  - Fraud ring membership (coordinated network-level fraud detection)
  - Plain-English investigator note
  - Recommended action: BLOCK / HOLD / MONITOR / ALLOW

Authentication: optional API key header `X-BTI-Api-Key`.
Set BTI_SAS_API_KEY env var to enforce it; omit for open dev access.

Architecture: "complement SAS, don't compete" — SAS keeps its own score;
BTI adds network intelligence and explainability that SAS cannot provide.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Security, UploadFile
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from bti.database import get_db
from bti.sas.enricher import enrich_single, enrich_batch, EnrichmentResult
from bti.logging_config import get_logger

import io
import pandas as pd

router = APIRouter(prefix="/sas", tags=["SAS Integration — Enrichment API"])
log = get_logger("api.sas")

_API_KEY_HEADER = APIKeyHeader(name="X-BTI-Api-Key", auto_error=False)


# ── Auth dependency ───────────────────────────────────────────────────────────

def _require_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)):
    """
    Enforce X-BTI-Api-Key when BTI_SAS_API_KEY env var is configured.
    If the env var is unset, the endpoint is open (dev mode).
    """
    expected = os.environ.get("BTI_SAS_API_KEY")
    if not expected:
        return  # no key configured — open access for development
    if api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-BTI-Api-Key header. "
                   "Contact your BTI administrator for an API key.",
        )


# ── Request / Response schemas ────────────────────────────────────────────────

class TransactionInput(BaseModel):
    """A single transaction in SAS or BTI format."""
    transaction_id:      Optional[str]   = None
    customer_id:         Optional[str]   = None
    transaction_amount:  Optional[float] = Field(default=0.0, ge=0)
    transaction_time:    Optional[str]   = None
    merchant_category:   Optional[str]   = None
    merchant_name:       Optional[str]   = None
    device_id:           Optional[str]   = None
    ip_location:         Optional[str]   = None
    card1:               Optional[str]   = None
    card2:               Optional[str]   = None
    addr1:               Optional[str]   = None
    p_emaildomain:       Optional[str]   = None
    # Pass-through of any SAS pre-scored fields
    sas_risk_score:      Optional[float] = Field(default=None, description="SAS own risk score (passed through, not used in BTI scoring)")
    sas_alert_flag:      Optional[str]   = Field(default=None, description="SAS alert flag (passed through, not used in BTI scoring)")

    model_config = {"extra": "allow"}


class ShapDriverOut(BaseModel):
    feature:    str
    label:      str
    direction:  str
    impact_pct: float


class RingInfoOut(BaseModel):
    in_ring:              bool
    ring_id:              Optional[str]   = None
    ring_alert_tier:      Optional[str]   = None
    ring_risk_score:      Optional[float] = None
    ring_size:            Optional[int]   = None
    ring_total_exposure:  Optional[float] = None
    ring_shared_entities: Optional[List[str]] = None


class EnrichmentOut(BaseModel):
    enrichment_id:          str
    transaction_id:         str
    bti_risk_score:         float = Field(description="BTI composite fraud score 0–100")
    bti_alert_tier:         str   = Field(description="LOW | MEDIUM | HIGH | VERY HIGH | CRITICAL")
    bti_recommended_action: str   = Field(description="BLOCK | HOLD | MONITOR | ALLOW")
    top_drivers:            List[ShapDriverOut]
    ring:                   RingInfoOut
    notes_for_sas:          str   = Field(description="Plain-English summary for the SAS analyst queue")
    model_version:          str
    rules_fired:            List[str]
    processing_time_ms:     float
    enriched_at:            str


class EnrichRequest(BaseModel):
    transaction: TransactionInput
    model_config = {"json_schema_extra": {"example": {
        "transaction": {
            "transaction_id": "TXN-20260909-001",
            "customer_id":    "CUST-7823",
            "transaction_amount": 4750.00,
            "merchant_category":  "Crypto Exchanges",
            "device_id":  "DEV-X99",
            "ip_location": "198.51.100.1",
            "transaction_time": "02:17",
            "sas_risk_score": 72,
            "sas_alert_flag": "REVIEW",
        }
    }}}


class BatchEnrichRequest(BaseModel):
    transactions: List[TransactionInput] = Field(
        ..., min_length=1, max_length=500,
        description="Batch of transactions to enrich (1–500)"
    )


class BatchEnrichResponse(BaseModel):
    enriched_at:        str
    total_transactions: int
    rings_detected:     int
    in_rings:           int
    results:            List[EnrichmentOut]


# ── Serialisers ───────────────────────────────────────────────────────────────

def _to_out(r: EnrichmentResult) -> EnrichmentOut:
    return EnrichmentOut(
        enrichment_id          = r.enrichment_id,
        transaction_id         = r.transaction_id,
        bti_risk_score         = r.bti_risk_score,
        bti_alert_tier         = r.bti_alert_tier,
        bti_recommended_action = r.bti_recommended_action,
        top_drivers=[
            ShapDriverOut(
                feature    = d.feature,
                label      = d.label,
                direction  = d.direction,
                impact_pct = d.impact_pct,
            )
            for d in r.top_drivers
        ],
        ring=RingInfoOut(
            in_ring              = r.ring.in_ring,
            ring_id              = r.ring.ring_id,
            ring_alert_tier      = r.ring.ring_alert_tier,
            ring_risk_score      = r.ring.ring_risk_score,
            ring_size            = r.ring.ring_size,
            ring_total_exposure  = r.ring.ring_total_exposure,
            ring_shared_entities = r.ring.ring_shared_entities,
        ),
        notes_for_sas      = r.notes_for_sas,
        model_version      = r.model_version,
        rules_fired        = r.rules_fired,
        processing_time_ms = r.processing_time_ms,
        enriched_at        = r.enriched_at,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/enrich",
             response_model=EnrichmentOut,
             summary="Enrich a single transaction with BTI intelligence")
def enrich_transaction(
    body: EnrichRequest,
    db: Session = Depends(get_db),
    _: None = Depends(_require_api_key),
):
    """
    Submit a single transaction and receive BTI's full intelligence enrichment.

    **What BTI adds on top of SAS:**
    - A second-opinion ML risk score from a Random Forest trained on 590K
      real IEEE-CIS fraud transactions (ROC-AUC 0.908)
    - Top SHAP feature contributions — *why* BTI scored it that way, in
      investigator-readable language
    - Ring membership: is this transaction part of a coordinated fraud network?
      (for network analysis across a batch, use `/sas/enrich/batch`)
    - Plain-English `notes_for_sas` ready for the analyst exception queue
    - Recommended action: BLOCK / HOLD / MONITOR / ALLOW

    **SAS fields are passed through, not used in BTI scoring.**
    `sas_risk_score` and `sas_alert_flag` appear in the request for context
    but do not influence BTI's assessment — this ensures an independent
    second opinion.

    **Authentication:** include `X-BTI-Api-Key` header if configured.
    """
    txn_dict = body.transaction.model_dump()
    try:
        result = enrich_single(txn_dict, db_session=db)
    except Exception as e:
        log.exception("Enrichment error", extra={"txn": txn_dict.get("transaction_id")})
        raise HTTPException(status_code=500, detail=f"Enrichment failed: {e}")

    return _to_out(result)


@router.post("/enrich/batch",
             response_model=BatchEnrichResponse,
             summary="Enrich a batch of transactions — includes fraud ring detection")
def enrich_batch_endpoint(
    body: BatchEnrichRequest,
    db: Session = Depends(get_db),
    _: None = Depends(_require_api_key),
):
    """
    Submit 1–500 transactions and receive BTI intelligence for each, plus
    **fraud ring detection** across the entire batch.

    Ring detection builds a bipartite transaction graph where any two transactions
    sharing a device ID, IP address, card token, billing address, or email domain
    are connected. Connected components reveal coordinated fraud rings that
    individual-transaction scoring cannot detect.

    Each result includes `ring.in_ring` — if `true`, the transaction is part of
    a fraud ring identified in this batch, and `ring.ring_id`, `ring_size`,
    `ring_total_exposure`, and `ring_alert_tier` give full ring details.

    **Ideal SAS workflow:**
    1. Flush your alert queue to `/sas/enrich/batch`
    2. Filter results where `ring.in_ring == true` → priority network-level cases
    3. Sort remaining by `bti_risk_score` descending → individual risk queue
    4. Use `/copilot/ask` with a ring as context for investigator Q&A

    **Authentication:** include `X-BTI-Api-Key` header if configured.
    """
    txn_dicts = [t.model_dump() for t in body.transactions]
    try:
        results = enrich_batch(txn_dicts, db_session=db)
    except Exception as e:
        log.exception("Batch enrichment error")
        raise HTTPException(status_code=500, detail=f"Batch enrichment failed: {e}")

    in_rings = sum(1 for r in results if r.ring.in_ring)
    ring_ids  = {r.ring.ring_id for r in results if r.ring.in_ring}

    return BatchEnrichResponse(
        enriched_at        = datetime.now(timezone.utc).isoformat(),
        total_transactions = len(results),
        rings_detected     = len(ring_ids),
        in_rings           = in_rings,
        results            = [_to_out(r) for r in results],
    )


@router.post("/enrich/upload",
             response_model=BatchEnrichResponse,
             summary="Enrich a CSV/Excel transaction file — includes fraud ring detection")
async def enrich_upload(
    file: UploadFile = File(..., description="CSV or Excel file of transactions"),
    _: None = Depends(_require_api_key),
):
    """
    Upload a CSV or Excel file of transactions and receive full BTI enrichment
    including fraud ring detection across all rows.

    **Minimum required columns:** transaction_id (auto-generated if absent)

    **Entity columns for ring detection (include as many as available):**
    device_id, ip_location, card1, card2, addr1, p_emaildomain
    """
    filename = file.filename or "upload"
    content  = await file.read()

    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File contains no rows.")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [f"ROW-{i:06d}" for i in range(len(df))]

    txn_dicts = df.to_dict(orient="records")
    try:
        results = enrich_batch(txn_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrichment failed: {e}")

    in_rings = sum(1 for r in results if r.ring.in_ring)
    ring_ids  = {r.ring.ring_id for r in results if r.ring.in_ring}

    return BatchEnrichResponse(
        enriched_at        = datetime.now(timezone.utc).isoformat(),
        total_transactions = len(results),
        rings_detected     = len(ring_ids),
        in_rings           = in_rings,
        results            = [_to_out(r) for r in results],
    )


@router.get("/schema",
            summary="SAS field mapping schema for BTI enrichment response")
def enrichment_schema():
    """
    Returns documentation on every field in the BTI enrichment response —
    field name, type, range, and description — so SAS administrators can
    map BTI fields into SAS alert attributes.

    Recommended SAS attribute mappings:
    - `bti_risk_score` → BTI_RISK_SCORE (numeric, 0–100)
    - `bti_alert_tier` → BTI_ALERT_TIER (text)
    - `bti_recommended_action` → BTI_ACTION (text)
    - `ring.in_ring` → BTI_IN_RING (boolean)
    - `ring.ring_id` → BTI_RING_ID (text)
    - `ring.ring_size` → BTI_RING_SIZE (integer)
    - `notes_for_sas` → BTI_NOTES (text, max ~500 chars)
    """
    return {
        "version": "4.0.0",
        "endpoint": "POST /api/v1/sas/enrich",
        "auth_header": "X-BTI-Api-Key",
        "fields": {
            "enrichment_id":          {"type": "string",  "example": "ENR-A1B2C3D4", "desc": "Unique enrichment call ID for audit trail"},
            "transaction_id":         {"type": "string",  "desc": "Echo of transaction_id from the request"},
            "bti_risk_score":         {"type": "float",   "range": "0–100", "desc": "BTI composite fraud risk score"},
            "bti_alert_tier":         {"type": "string",  "values": ["LOW","MEDIUM","HIGH","VERY HIGH","CRITICAL"]},
            "bti_recommended_action": {"type": "string",  "values": ["ALLOW","MONITOR","HOLD","BLOCK"]},
            "top_drivers": {
                "type": "array",
                "max_items": 5,
                "item_fields": {
                    "feature":    "internal feature name",
                    "label":      "human-readable feature name",
                    "direction":  "increases_risk | decreases_risk",
                    "impact_pct": "contribution as % of total SHAP mass (float)",
                }
            },
            "ring.in_ring":              {"type": "boolean", "desc": "True if transaction is part of a detected fraud ring"},
            "ring.ring_id":              {"type": "string",  "desc": "Ring identifier (null if not in ring)"},
            "ring.ring_alert_tier":      {"type": "string",  "desc": "Ring-level alert tier"},
            "ring.ring_risk_score":      {"type": "float",   "range": "0–100"},
            "ring.ring_size":            {"type": "integer", "desc": "Number of transactions in the ring"},
            "ring.ring_total_exposure":  {"type": "float",   "desc": "Total USD at risk across the ring"},
            "ring.ring_shared_entities": {"type": "array",   "desc": "Entity nodes (device/IP/card/address) linking the ring"},
            "notes_for_sas":             {"type": "string",  "desc": "Plain-English investigator note, ~100–300 chars"},
            "model_version":             {"type": "string",  "desc": "BTI model version string"},
            "rules_fired":               {"type": "array",   "desc": "List of fraud rule IDs that triggered"},
            "processing_time_ms":        {"type": "float",   "desc": "BTI enrichment latency for this transaction (ms)"},
            "enriched_at":               {"type": "string",  "format": "ISO-8601 UTC"},
        },
        "recommended_sas_workflow": [
            "1. POST alert queue (1–500 txns) to /api/v1/sas/enrich/batch",
            "2. Filter ring.in_ring == true → priority network-level fraud queue",
            "3. Sort remaining by bti_risk_score descending → individual risk queue",
            "4. Pass notes_for_sas to analyst exception queue comment field",
            "5. Use /api/v1/copilot/ask with ring context for AI-assisted investigation",
        ],
    }


@router.get("/health",
            summary="SAS Integration API health check")
def sas_health():
    """Confirms the SAS Integration module is loaded and ready."""
    api_key_enforced = bool(os.environ.get("BTI_SAS_API_KEY"))
    scorer_ready = _scorer.bundle is not None
    return {
        "status":           "ok" if scorer_ready else "degraded",
        "module":           "BTI SAS Integration Enrichment API",
        "version":          "4.0.0",
        "scorer_ready":     scorer_ready,
        "api_key_enforced": api_key_enforced,
        "capabilities": [
            "single-transaction enrichment (< 200ms)",
            "batch enrichment with fraud ring detection (1–500 transactions)",
            "CSV/Excel file upload enrichment",
            "SHAP feature contribution explanations",
            "independent second-opinion risk scoring",
            "SAS field mapping schema (/sas/schema)",
        ],
        "note": (
            "Set BTI_SAS_API_KEY env var to enforce X-BTI-Api-Key header auth."
            if not api_key_enforced else
            "API key authentication is active."
        ),
    }
