"""
/api/v1/stepup — customer challenges for STEP_UP decisions: SMS one-time code, push approval, 3-D Secure.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from bti.database import get_db
from bti.operations import stepup

router = APIRouter(prefix="/stepup", tags=["Step-up orchestration"])


class ChallengeIn(BaseModel):
    transaction_id: str
    customer_id: Optional[str] = None
    channel: Optional[str] = None
    transaction_type: Optional[str] = None
    amount_usd: Optional[float] = Field(None, ge=0)
    currency: str = "USD"
    method: Optional[str] = Field(None, description="sms_otp | push | 3ds; chosen from the channel if omitted")
    callback_url: Optional[str] = None


class CodeIn(BaseModel):
    code: str = Field(..., min_length=4, max_length=10)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except stepup.StepUpError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/challenges", dependencies=[Depends(require_api_key)], status_code=201)
def issue(body: ChallengeIn, db: Session = Depends(get_db)):
    """Issue a challenge. A `send_failed` status means the caller should route the transaction to REVIEW."""
    return _call(stepup.issue, db, body.transaction_id, body.customer_id, body.channel, body.transaction_type,
                 body.amount_usd, body.currency, body.method, body.callback_url)


@router.get("/challenges/{challenge_id}")
def status(challenge_id: str, db: Session = Depends(get_db)):
    try:
        return stepup.get(db, challenge_id)
    except stepup.StepUpError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/challenges/{challenge_id}/verify", dependencies=[Depends(require_api_key)])
def verify(challenge_id: str, body: CodeIn, db: Session = Depends(get_db)):
    """Check an SMS one-time code (3 attempts, 5-minute expiry)."""
    return _call(stepup.verify_code, db, challenge_id, body.code)


@router.post("/challenges/{challenge_id}/callback")
async def challenge_callback(challenge_id: str, request: Request, db: Session = Depends(get_db),
                             x_bti_signature: Optional[str] = Header(default=None)):
    """Push or 3-D Secure result from the bank's systems, signed with HMAC-SHA256 of the raw body."""
    raw = await request.body()
    try:
        return stepup.callback(db, challenge_id, raw, x_bti_signature)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except stepup.StepUpError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/expire", dependencies=[Depends(require_api_key)])
def expire(db: Session = Depends(get_db)):
    return stepup.expire(db)


@router.get("/stats")
def stats(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)):
    """Pass, failure and abandonment rates per method; catch rate on matured labels."""
    return stepup.stepup_stats(db, days)
