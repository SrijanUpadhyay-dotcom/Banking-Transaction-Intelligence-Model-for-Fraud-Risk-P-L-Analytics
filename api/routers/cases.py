# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
/api/v1/cases — investigation queues with SLAs. Live REVIEW decisions open cases automatically; dispositions
become confirmed labels for monitoring and retraining.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from bti.database import get_db
from bti.operations import cases

router = APIRouter(prefix="/cases", tags=["Case management"])


class CaseIn(BaseModel):
    transaction_id: str
    source: str = Field("manual_referral", description="e.g. manual_referral, customer_review_request")
    customer_id: Optional[str] = None
    fraud_probability: Optional[float] = Field(None, ge=0, le=1)
    amount_usd: Optional[float] = Field(None, ge=0)
    queue: Optional[str] = None
    notes: Optional[List[str]] = None


class AssignIn(BaseModel):
    analyst: str
    queue: Optional[str] = None


class DispositionIn(BaseModel):
    disposition: str = Field(..., description="confirmed_fraud | confirmed_genuine | inconclusive | "
                                              "customer_unreachable | duplicate")
    analyst: str
    notes: Optional[str] = None
    checked_by: Optional[str] = Field(None, description="Second reviewer; required to clear high-value cases")
    fraud_type: Optional[str] = None
    loss_amount: Optional[float] = None


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except cases.CaseError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("")
def list_cases(status: Optional[str] = None, queue: Optional[str] = None, assigned_to: Optional[str] = None,
               limit: int = Query(200, ge=1, le=2000), db: Session = Depends(get_db)):
    return {"cases": cases.list_cases(db, status, queue, assigned_to, limit)}


@router.get("/queues")
def queues(days: int = Query(7, ge=1, le=90), db: Session = Depends(get_db)):
    """Per queue: open, unassigned, breached, oldest; closed, SLA attainment, time to close, confirmation rate."""
    return cases.queue_status(db, days)


@router.get("/{case_id}/context")
def case_context(case_id: int, db: Session = Depends(get_db)):
    """The case, the scored decision behind it, and the customer's recent activity, cases and outcomes."""
    try:
        return cases.case_context(db, case_id)
    except cases.CaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("", dependencies=[Depends(require_api_key)], status_code=201)
def open_case(body: CaseIn, db: Session = Depends(get_db)):
    """Manual referral, e.g. a customer asking for human review of an automated decline."""
    return _call(cases.open_case, db, body.transaction_id, body.source, customer_id=body.customer_id,
                 fraud_probability=body.fraud_probability, amount_usd=body.amount_usd, queue=body.queue)


@router.post("/next", dependencies=[Depends(require_api_key)])
def next_case(body: AssignIn, db: Session = Depends(get_db)):
    """Assign the most urgent unassigned case to the analyst."""
    case = _call(cases.assign_next, db, body.analyst, body.queue)
    if case is None:
        return {"case": None, "detail": "No unassigned cases"}
    return {"case": case}


@router.post("/{case_id}/assign", dependencies=[Depends(require_api_key)])
def assign(case_id: int, body: AssignIn, db: Session = Depends(get_db)):
    return _call(cases.assign, db, case_id, body.analyst)


@router.post("/{case_id}/pending-customer", dependencies=[Depends(require_api_key)])
def pending_customer(case_id: int, body: AssignIn, db: Session = Depends(get_db)):
    return _call(cases.set_pending_customer, db, case_id, body.analyst)


@router.post("/{case_id}/disposition", dependencies=[Depends(require_api_key)])
def disposition(case_id: int, body: DispositionIn, db: Session = Depends(get_db)):
    """Close the case. Confirmed fraud / confirmed genuine are written as labels automatically."""
    return _call(cases.dispose, db, case_id, body.disposition, body.analyst, body.notes, body.checked_by,
                 body.fraud_type, body.loss_amount)


@router.post("/sla-check", dependencies=[Depends(require_api_key)])
def sla_check(notify: bool = True, db: Session = Depends(get_db)):
    return cases.check_sla(db, notify=notify)
