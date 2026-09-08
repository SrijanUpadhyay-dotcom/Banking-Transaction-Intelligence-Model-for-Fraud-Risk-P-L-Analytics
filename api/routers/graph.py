"""
/api/v1/graph — Graph / Network Analytics for Fraud Ring Detection
Phase 2 of BTI v2 (Fraud Intelligence Layer)

Accepts a batch of transactions and returns a network-level fraud analysis:
which transactions are connected via shared devices, IPs, cards, or addresses,
and how those clusters score as coordinated fraud rings.
"""

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from bti.database import get_db
from bti.graph.ring_detector import detect_fraud_rings, FraudRing, RingMember
from bti.logging_config import get_logger

router = APIRouter(prefix="/graph", tags=["Graph Analytics — Fraud Rings"])
log = get_logger("api.graph")


# ── Request / Response schemas ────────────────────────────────────────────────

class TransactionNode(BaseModel):
    transaction_id:   str
    customer_id:      Optional[str] = None
    transaction_amount: Optional[float] = Field(default=0.0, ge=0)
    transaction_time: Optional[str] = None
    merchant_category: Optional[str] = None
    merchant_name:    Optional[str] = None
    device_id:        Optional[str] = None
    ip_location:      Optional[str] = None
    card1:            Optional[str] = None
    card2:            Optional[str] = None
    addr1:            Optional[str] = None
    p_emaildomain:    Optional[str] = None
    # Pass-through of any pre-scored BTI fields
    final_risk_score: Optional[float] = Field(default=None, ge=0, le=100)
    final_alert_tier: Optional[str]   = None
    risk_score:       Optional[float] = Field(default=None, ge=0, le=100)

    model_config = {"extra": "allow"}   # accept any additional fields silently


class RingMemberOut(BaseModel):
    transaction_id:  str
    customer_id:     str
    amount:          float
    risk_score:      float
    alert_tier:      str
    shared_entities: List[str]


class FraudRingOut(BaseModel):
    ring_id:          str
    size:             int
    ring_risk_score:  float
    ring_alert_tier:  str
    entity_types:     List[str]
    shared_entities:  List[str]
    avg_amount:       float
    total_amount:     float
    high_risk_count:  int
    velocity_flag:    bool
    narrative:        str
    recommendation:   str
    members:          List[RingMemberOut]


class GraphAnalyzeRequest(BaseModel):
    transactions: List[TransactionNode] = Field(
        ..., min_length=2,
        description="Batch of transactions to analyse for network-level fraud patterns"
    )
    min_ring_size: int = Field(
        default=2, ge=2, le=20,
        description="Minimum number of linked transactions to qualify as a ring"
    )

    model_config = {"json_schema_extra": {"example": {
        "min_ring_size": 2,
        "transactions": [
            {"transaction_id": "T001", "customer_id": "C001", "transaction_amount": 500,
             "device_id": "DEV-X99", "ip_location": "198.51.100.1",
             "merchant_category": "Crypto Exchanges"},
            {"transaction_id": "T002", "customer_id": "C002", "transaction_amount": 480,
             "device_id": "DEV-X99", "ip_location": "198.51.100.1",
             "merchant_category": "Crypto Exchanges"},
            {"transaction_id": "T003", "customer_id": "C003", "transaction_amount": 520,
             "device_id": "DEV-X99", "merchant_category": "Crypto Exchanges"},
            {"transaction_id": "T004", "customer_id": "C004", "transaction_amount": 42,
             "device_id": "DEV-SAFE", "ip_location": "192.168.1.1"},
        ]
    }}}


def _ring_to_out(ring: FraudRing) -> FraudRingOut:
    return FraudRingOut(
        ring_id         = ring.ring_id,
        size            = ring.size,
        ring_risk_score = ring.ring_risk_score,
        ring_alert_tier = ring.ring_alert_tier,
        entity_types    = ring.entity_types,
        shared_entities = ring.shared_entities,
        avg_amount      = ring.avg_amount,
        total_amount    = ring.total_amount,
        high_risk_count = ring.high_risk_count,
        velocity_flag   = ring.velocity_flag,
        narrative       = ring.narrative,
        recommendation  = ring.recommendation,
        members=[
            RingMemberOut(
                transaction_id  = m.transaction_id,
                customer_id     = m.customer_id,
                amount          = m.amount,
                risk_score      = m.risk_score,
                alert_tier      = m.alert_tier,
                shared_entities = m.shared_entities,
            )
            for m in ring.members
        ],
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/analyze",
             summary="Detect fraud rings in a batch of transactions")
def analyze_graph(body: GraphAnalyzeRequest):
    """
    Submit a batch of transactions and receive a network-level fraud ring analysis.

    The engine builds a transaction graph where nodes are connected by shared
    attributes — same device ID, IP address, card token, billing address, or
    email domain. Connected components in this graph reveal **fraud rings**:
    groups of transactions linked by common actors.

    Each detected ring includes:
    - **ring_risk_score** (0–100): composite score based on ring size, shared
      entity types, member risk scores, velocity, and merchant categories
    - **ring_alert_tier**: CRITICAL / HIGH / MEDIUM / LOW
    - **shared_entities**: the exact nodes (device IDs, IPs, etc.) that connect
      the ring members
    - **narrative**: plain-English description for a fraud investigator
    - **recommendation**: block / hold / monitor action covering all ring members

    This capability complements individual transaction scoring (e.g. SAS) by
    surfacing *network-level* fraud patterns that single-transaction models
    cannot detect.
    """
    txn_dicts = [t.model_dump() for t in body.transactions]
    result    = detect_fraud_rings(txn_dicts, min_ring_size=body.min_ring_size)

    return {
        "analysed_at":           datetime.utcnow().isoformat() + "Z",
        "total_transactions":    result.total_transactions,
        "transactions_in_rings": result.transactions_in_rings,
        "isolated_transactions": result.isolated_count,
        "rings_detected":        result.rings_detected,
        "rings_critical":        result.rings_critical,
        "rings_high":            result.rings_high,
        "summary":               result.summary_narrative,
        "rings":                 [_ring_to_out(r) for r in result.rings],
    }


@router.post("/analyze/upload",
             summary="Detect fraud rings from a CSV or Excel file")
async def analyze_graph_upload(
    file: UploadFile = File(..., description="CSV or Excel file of transactions"),
    min_ring_size: int = 2,
):
    """
    Upload a CSV or Excel file and run fraud ring detection across all rows.

    **Minimum required columns:** transaction_id, transaction_amount

    **Entity columns used for graph edges (include as many as available):**
    device_id, ip_location, card1, card2, addr1, p_emaildomain, merchant_name

    The more entity columns you provide, the richer the graph and the more
    accurate the ring detection.
    """
    filename = file.filename or "upload"
    content  = await file.read()

    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File contains no rows.")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "transaction_id" not in df.columns:
        df["transaction_id"] = [f"ROW-{i:06d}" for i in range(len(df))]

    txn_dicts = df.to_dict(orient="records")
    result    = detect_fraud_rings(txn_dicts, min_ring_size=min_ring_size)

    return {
        "file_name":             filename,
        "analysed_at":           datetime.utcnow().isoformat() + "Z",
        "total_transactions":    result.total_transactions,
        "transactions_in_rings": result.transactions_in_rings,
        "isolated_transactions": result.isolated_count,
        "rings_detected":        result.rings_detected,
        "rings_critical":        result.rings_critical,
        "rings_high":            result.rings_high,
        "summary":               result.summary_narrative,
        "rings":                 [_ring_to_out(r) for r in result.rings],
    }


@router.get("/health",
            summary="Graph analytics module health check")
def graph_health():
    """Confirms the graph analytics module is loaded and ready."""
    return {
        "status":  "ok",
        "module":  "BTI Graph Analytics — Fraud Ring Detector",
        "version": "2.0.0",
        "engine":  "NetworkX",
        "capabilities": [
            "shared-device detection",
            "shared-IP detection",
            "shared-card detection",
            "shared-address detection",
            "shared-email-domain detection",
            "connected-component ring clustering",
            "ring risk scoring (0-100)",
            "investigator narrative generation",
        ],
    }
