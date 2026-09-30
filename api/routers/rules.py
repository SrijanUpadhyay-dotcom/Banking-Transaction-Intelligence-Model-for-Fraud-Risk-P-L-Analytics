# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
/api/v1/rules — governed analyst rules.

Rules are authored as JSON conditions over lineage-checked fields and
simulated on history. A different person approves each rule, either as active
(it can raise decisions) or shadow (logged only). Performance is tracked on
matured labels.
"""

from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from bti.database import get_db
from bti.rules import lifecycle
from bti.rules.language import RuleError, allowed_fields

router = APIRouter(prefix="/rules", tags=["Analyst rules"])


class RuleIn(BaseModel):
    name: str = Field(..., min_length=3, max_length=120)
    description: str = Field(..., min_length=10)
    condition: dict
    action: Literal["STEP_UP", "REVIEW", "DECLINE"]
    author: str
    expires_at: Optional[datetime] = None


class VersionIn(BaseModel):
    author: str
    condition: Optional[dict] = None
    action: Optional[Literal["STEP_UP", "REVIEW", "DECLINE"]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    expires_at: Optional[datetime] = None


class Actor(BaseModel):
    actor: str


class ApproveIn(BaseModel):
    approver: str
    mode: Literal["active", "shadow"] = "active"


class RetireIn(BaseModel):
    actor: str
    reason: str = Field(..., min_length=5)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuleError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/fields")
def fields():
    """Fields a rule may use. Protected, post-event and label-derived fields are refused."""
    return {"fields": allowed_fields(),
            "operators": ["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "is_null", "not_null"],
            "combinators": ["all", "any", "not"], "actions": list(lifecycle.ACTIONS)}


@router.get("")
def list_rules(include_retired: bool = False, db: Session = Depends(get_db)):
    return {"rules": lifecycle.list_rules(db, include_retired)}


@router.post("", dependencies=[Depends(require_api_key)], status_code=201)
def create(body: RuleIn, db: Session = Depends(get_db)):
    return _call(lifecycle.create_rule, db, body.name, body.description, body.condition, body.action, body.author,
                 body.expires_at)


@router.get("/{rule_id}")
def rule_versions(rule_id: str, db: Session = Depends(get_db)):
    rows = lifecycle.versions(db, rule_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No rule {rule_id}")
    return {"rule_id": rule_id, "versions": rows}


@router.post("/{rule_id}/versions", dependencies=[Depends(require_api_key)], status_code=201)
def new_version(rule_id: str, body: VersionIn, db: Session = Depends(get_db)):
    return _call(lifecycle.new_version, db, rule_id, body.author, body.condition, body.action, body.name,
                 body.description, body.expires_at)


@router.post("/versions/{version_id}/simulate", dependencies=[Depends(require_api_key)])
def simulate(version_id: int, body: Actor, db: Session = Depends(get_db)):
    """Replay on out-of-time history against the model's own decisions: hits, precision, incremental value,
    fairness of the rule's hits."""
    return _call(lifecycle.simulate, db, version_id, body.actor)


@router.post("/versions/{version_id}/approve", dependencies=[Depends(require_api_key)])
def approve(version_id: int, body: ApproveIn, db: Session = Depends(get_db)):
    return _call(lifecycle.approve, db, version_id, body.approver, body.mode)


@router.post("/versions/{version_id}/retire", dependencies=[Depends(require_api_key)])
def retire(version_id: int, body: RetireIn, db: Session = Depends(get_db)):
    return _call(lifecycle.retire, db, version_id, body.actor, body.reason)


@router.get("/{rule_id}/performance")
def performance(rule_id: str, days: int = Query(90, ge=1, le=365), maturity_days: int = Query(90, ge=1, le=365),
                db: Session = Depends(get_db)):
    """Live hits per version (active and shadow) with precision and incremental catches on matured labels."""
    return lifecycle.performance(db, rule_id, days, maturity_days)
