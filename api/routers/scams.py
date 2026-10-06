# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
/api/v1/scams — APP-scam and mule models (Phase 9).

Assessments and reports are read-only. Mule-alert scans and reviews need the API key.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from bti.database import get_db
from bti.database.models import AuditLog, MuleAlert

router = APIRouter(prefix="/scams", tags=["Scams and mules"])


def _summary(family: str) -> dict:
    from bti.modeling import registry
    out = {}
    for role in ("champion", "challenger"):
        model_id = registry.model_for_role(role, family=family)
        if model_id:
            card = registry.load_card(model_id, family=family)
            out[role] = {"model_id": model_id, "validation_status": card["validation"]["status"],
                         "gates": card["validation"]["gates"], "notes": registry.notes_for(model_id, family=family)}
        else:
            out[role] = None
    return out


@router.get("/models")
def models():
    """Scam and mule model roles, validation gates and post-registration notes."""
    from bti.config import get_settings
    return {"scam": _summary("scam"), "mule": _summary("mule"), "scam_mode": get_settings().scam_mode}


class Payment(BaseModel):
    transaction_id: str
    customer_id: str
    transaction_date: str
    transaction_time: str = "12:00:00"
    transaction_amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    country: Optional[str] = None
    transaction_type: str = "Transfer"
    payee_id: str
    cop_result: Optional[str] = Field(None, pattern="^(match|close_match|no_match|unavailable)$")
    payee_account_opened_date: Optional[str] = None
    payee_customer_id: Optional[str] = None
    historical_average_transaction_amount: Optional[float] = None
    account_balance_before: Optional[float] = None
    is_vulnerable: Optional[bool] = None


@router.post("/assess")
def assess(payment: Payment, db: Session = Depends(get_db)):
    """Scam probability, reimbursement exposure and the intervention the overlay would choose. Changes nothing."""
    from bti.modeling.scorer import fetch_history
    from bti.scams.overlay import assess as assess_scam
    txn = {**payment.model_dump(), "debit_credit_flag": "Debit"}
    result = assess_scam(txn, fetch_history(db, txn, 365), db)
    if result is None:
        raise HTTPException(status_code=503, detail="No scam model registered (python -m bti.scams.app_model)")
    return result


def _report(name: str):
    path = Path("outputs/scams") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No report yet: {name}")
    return json.loads(path.read_text())


@router.get("/reports/app-model")
def app_model_report():
    return _report("app_model_evaluation.json")


@router.get("/reports/reimbursement-exposure")
def exposure_report():
    return _report("reimbursement_exposure.json")


@router.get("/reports/mule-model")
def mule_model_report():
    return _report("mule_model_evaluation.json")


@router.get("/mule-alerts")
def mule_alerts(status: str = Query("open", pattern="^(open|confirmed_mule|cleared|all)$"),
                db: Session = Depends(get_db)):
    q = db.query(MuleAlert)
    if status != "all":
        q = q.filter(MuleAlert.status == status)
    rows = q.order_by(MuleAlert.score.desc()).limit(500).all()
    return [{c.name: (getattr(r, c.name).isoformat() if isinstance(getattr(r, c.name), datetime) else getattr(r, c.name))
             for c in MuleAlert.__table__.columns} for r in rows]


@router.post("/mule-alerts/run", dependencies=[Depends(require_api_key)])
def run_mule_scan(db: Session = Depends(get_db)):
    from bti.scams.mule import run_scan
    return run_scan(db)


class MuleReview(BaseModel):
    status: str = Field(..., pattern="^(confirmed_mule|cleared)$")
    reviewed_by: str = Field(..., min_length=2)
    note: str = Field(..., min_length=10, description="Evidence for the decision (always required)")


@router.post("/mule-alerts/{alert_id}/review", dependencies=[Depends(require_api_key)])
def review_mule_alert(alert_id: int, body: MuleReview, db: Session = Depends(get_db)):
    row = db.get(MuleAlert, alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such alert")
    if row.status != "open":
        raise HTTPException(status_code=409, detail=f"Alert already {row.status}")
    row.status, row.reviewed_by, row.reviewed_at, row.note = body.status, body.reviewed_by.strip(), datetime.utcnow(), body.note
    db.add(AuditLog(ts=datetime.utcnow(), event_type="MULE_ALERT_REVIEWED",
                    payload={"id": alert_id, "customer_id": row.customer_id, "status": body.status,
                             "by": body.reviewed_by, "note": body.note}))
    db.commit()
    return {"id": alert_id, "status": row.status}
