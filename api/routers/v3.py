"""
/api/v1/v3 — leak-free, calibrated, explainable scoring with expected-cost decisions.

Every call is scored by the champion (or, until one is approved, by the
challenger flagged as provisional), decided under the transaction's
jurisdiction policy, shadow-scored by the challenger, and written to the
score log for audit, monitoring and champion/challenger comparison.
"""

from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from bti.database import get_db
from bti.database.models import SecurityEvent, Transaction
from bti.logging_config import get_logger
from bti.modeling import registry
from bti.modeling.features import FEEDS, SECURITY_EVENT_TYPES
from bti.modeling.fx import supported_currencies
from bti.modeling.scorer import scorer
from bti.operations.scoring_service import score_and_decide

router = APIRouter(prefix="/v3", tags=["BTI v3 Scoring"])
log = get_logger("api.v3")

MAX_BATCH = 500


class V3Transaction(BaseModel):
    transaction_id: str
    customer_id: str
    transaction_date: str = Field(..., description="YYYY-MM-DD")
    transaction_time: str = Field("12:00:00", description="HH:MM:SS, local time of the account")
    transaction_amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    country: Optional[str] = Field(None, description="Account country or ISO-2 code; selects jurisdiction policy")
    channel: Optional[str] = None
    transaction_type: Optional[str] = None
    merchant_category: Optional[str] = None
    merchant_name: Optional[str] = None
    authorization_method: Optional[str] = None
    debit_credit_flag: Optional[str] = "Debit"
    device_id: Optional[str] = None
    ip_location: Optional[str] = None
    historical_average_transaction_amount: Optional[float] = None
    account_balance_before: Optional[float] = None
    failed_attempt_count: Optional[int] = 0
    login_attempts: Optional[int] = 1
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Location feed: where the transaction "
                                                                        "happened (terminal, device GPS or IP geo)")
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    payee_id: Optional[str] = Field(None, description="Payee feed: beneficiary account key (tokenised is fine)")

    model_config = {"extra": "ignore"}


class SecurityEventIn(BaseModel):
    customer_id: str
    event_type: str = Field(..., description=f"One of {', '.join(SECURITY_EVENT_TYPES)}")
    event_time: datetime = Field(..., description="When the event happened (ISO 8601, account local time)")
    source: Optional[str] = Field(None, description="Originating system, e.g. identity platform, MNO API")
    detail: Optional[dict] = None


def _validate_currency(txn: V3Transaction) -> None:
    if txn.currency.upper() not in supported_currencies():
        raise HTTPException(status_code=422, detail=f"Unsupported currency {txn.currency}. Supported: "
                                                    f"{supported_currencies()}. Add a rate under fx.rates_to_usd.")


def _score_one(txn_model: V3Transaction, db: Session, explain: bool = True) -> dict:
    _validate_currency(txn_model)
    try:
        sd = score_and_decide(txn_model.model_dump(), db, explain=explain)
    except registry.RegistryError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    live, decision, policy = sd.live, sd.decision, sd.policy
    return {
        "transaction_id": live.transaction_id,
        "model": {"model_id": live.model_id, "role": live.model_role, "provisional": live.provisional},
        "fraud_probability": live.fraud_probability,
        "score": live.score,
        "risk_band": live.risk_band,
        "amount_usd": round(sd.amount_usd, 2),
        "decision": asdict(decision),
        "reason_codes": live.reason_codes,
        "jurisdiction": {"iso2": policy.iso2 if policy else None, "country": policy.country if policy else None,
                         "loss_given_fraud": decision.cost_model["loss_given_fraud"],
                         "decline_requires_human_review_route": bool(policy and
                                                                     policy.decline_requires_human_review_route)},
        "shadow": sd.shadow,
        "history_rows_used": live.history_rows_used,
        "latency_ms": live.latency_ms,
        "notes": live.notes + ([] if policy else ["Unknown or missing country — default cost model applied."]),
    }


@router.post("/score")
def score_v3(txn: V3Transaction, db: Session = Depends(get_db)):
    """Score one transaction: calibrated probability, decision, reason codes and shadow challenger score."""
    return _score_one(txn, db)


@router.post("/score/batch")
def score_v3_batch(txns: List[V3Transaction], explain: bool = Query(False), db: Session = Depends(get_db)):
    """Score up to 500 transactions, each against its own database history."""
    if not 1 <= len(txns) <= MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Batch size must be between 1 and {MAX_BATCH}")
    results = [_score_one(t, db, explain=explain) for t in txns]
    actions = pd.Series([r["decision"]["action"] for r in results]).value_counts().to_dict()
    return {"count": len(results), "action_mix": actions, "results": results}


@router.get("/model")
def v3_model():
    """The model currently answering /v3/score, with its headline validation figures."""
    try:
        model_id, role, provisional = scorer.resolve("champion")
    except registry.RegistryError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    card = registry.load_card(model_id)
    return {
        "model_id": model_id, "role": role, "provisional": provisional,
        "validation_status": card["validation"]["status"],
        "out_of_time": card["performance"]["metrics"]["out_of_time"],
        "reference_operating_point": card["performance"]["reference_operating_point"],
        "features": [f["name"] for f in card["features"]["model_features"]],
        "supported_currencies": supported_currencies(),
    }


@router.post("/security-events", status_code=201, dependencies=[Depends(require_api_key)])
def ingest_security_events(events: List[SecurityEventIn], db: Session = Depends(get_db)):
    """
    Security-event feed: password resets, SIM swaps / number ports, contact-detail changes and device
    enrolments. Append-only. Models trained with the security-event signals read these point-in-time.
    """
    if not 1 <= len(events) <= 5000:
        raise HTTPException(status_code=422, detail="Send between 1 and 5000 events per call")
    unknown = sorted({e.event_type for e in events} - set(SECURITY_EVENT_TYPES))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown event_type {unknown}; use {list(SECURITY_EVENT_TYPES)}")
    db.add_all([SecurityEvent(customer_id=e.customer_id, event_type=e.event_type,
                              event_time=e.event_time.replace(tzinfo=None), source=e.source, detail=e.detail)
                for e in events])
    db.commit()
    return {"accepted": len(events)}


@router.get("/feeds")
def feed_status(db: Session = Depends(get_db)):
    """Which optional data feeds are populated, and whether the scoring model uses them."""
    from sqlalchemy import func
    total = db.query(func.count(Transaction.transaction_id)).scalar() or 0
    located = db.query(func.count(Transaction.transaction_id)).filter(Transaction.latitude.isnot(None),
                                                                     Transaction.longitude.isnot(None)).scalar() or 0
    with_payee = db.query(func.count(Transaction.transaction_id)).filter(Transaction.payee_id.isnot(None)).scalar() or 0
    events = db.query(func.count(SecurityEvent.id)).scalar() or 0
    try:
        model_id, _, _ = scorer.resolve("champion")
        used = registry.load_artifact(model_id).get("feeds", [])
    except registry.RegistryError:
        model_id, used = None, []
    stored = {"location": {"transactions_with_location": located},
              "payee": {"transactions_with_payee": with_payee},
              "security_events": {"events_stored": events}}
    share = {"location": located / total if total else 0.0, "payee": with_payee / total if total else 0.0,
             "security_events": None}
    return {
        "transactions": total,
        "scoring_model": model_id,
        "feeds": {name: {**stored[name], "share_of_transactions": None if share[name] is None else
                         round(share[name], 4), "used_by_scoring_model": name in used,
                         "features": spec["features"], "fields": spec["fields"], "source": spec["source"]}
                  for name, spec in FEEDS.items()},
        "note": "Feed-dependent features stay out of every model until the feed covers each window of the "
                "training history; train with --feature-sets signals-relative once it does.",
    }
