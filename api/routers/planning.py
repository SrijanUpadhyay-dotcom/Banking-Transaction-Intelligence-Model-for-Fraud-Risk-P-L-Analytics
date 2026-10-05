# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
/api/v1/planning — fraud-loss and staffing forecasts, attack early warning, policy what-if.

Forecasts and what-ifs are read-only. Early-warning reviews and runs need the API key.
"""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from bti.database import get_db
from bti.database.models import AuditLog, EarlyWarning

router = APIRouter(prefix="/planning", tags=["Planning and early warning"])


def _history(source: str, db: Session):
    from bti.planning import history
    h = history.from_database(db) if source == "db" else history.load()
    if h.empty or (h["date"].max() - h["date"].min()).days < 120:
        raise HTTPException(status_code=409, detail="At least 120 days of transaction history are needed")
    return h


@router.get("/forecast/loss")
def loss_forecast(source: str = Query("db", pattern="^(db|extract)$"), backtest: bool = Query(False),
                  db: Session = Depends(get_db)):
    """30/60/90-day fraud-loss forecast with intervals, by country and channel (optionally with a backtest)."""
    from bti.planning import forecast
    h = _history(source, db)
    span = (h["date"].max() - h["date"].min()).days
    out = {"forecast": forecast.forecast(h, fit_days=min(365, span))}
    if backtest:
        out["backtest"] = forecast.backtest(h)
    return out


@router.get("/forecast/staffing")
def staffing_forecast(source: str = Query("db", pattern="^(db|extract)$"),
                      analysts: Optional[float] = Query(None, gt=0), scale: float = Query(1.0, gt=0),
                      db: Session = Depends(get_db)):
    """Case-volume and analyst forecast; with `analysts`, a proposed review-capacity setting (not applied)."""
    from bti.planning import staffing
    h = _history(source, db)
    span = (h["date"].max() - h["date"].min()).days
    return staffing.forecast(h, staffing.case_rates(), staffing.handling_minutes(db), fit_days=min(365, span),
                             analysts_fte=analysts, volume_scale=scale)


@router.get("/early-warning")
def early_warnings(status: Optional[str] = Query("open", pattern="^(open|acknowledged|closed|all)$"),
                   db: Session = Depends(get_db)):
    q = db.query(EarlyWarning)
    if status != "all":
        q = q.filter(EarlyWarning.status == status)
    rows = q.order_by(EarlyWarning.alarmed_on.desc()).limit(500).all()
    return [{c.name: (getattr(r, c.name).isoformat() if isinstance(getattr(r, c.name), datetime) else getattr(r, c.name))
             for c in EarlyWarning.__table__.columns} for r in rows]


@router.post("/early-warning/run", dependencies=[Depends(require_api_key)])
def run_early_warning(db: Session = Depends(get_db)):
    from bti.planning.early_warning import run_daily
    return run_daily(db)


class Review(BaseModel):
    status: str = Field(..., pattern="^(acknowledged|closed)$")
    reviewed_by: str = Field(..., min_length=2)
    note: Optional[str] = None


@router.post("/early-warning/{alarm_id}/review", dependencies=[Depends(require_api_key)])
def review_early_warning(alarm_id: int, body: Review, db: Session = Depends(get_db)):
    row = db.get(EarlyWarning, alarm_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such alarm")
    if body.status == "closed" and not (body.note or "").strip():
        raise HTTPException(status_code=422, detail="Closing an alarm needs a note (cause, or why it was benign)")
    row.status, row.reviewed_by, row.reviewed_at, row.note = body.status, body.reviewed_by.strip(), datetime.utcnow(), body.note
    db.add(AuditLog(ts=datetime.utcnow(), event_type="EARLY_WARNING_REVIEWED",
                    payload={"id": alarm_id, "status": body.status, "by": body.reviewed_by, "note": body.note}))
    db.commit()
    return {"id": alarm_id, "status": row.status}


@router.post("/whatif")
def whatif(proposal: Dict = Body(..., examples=[{"max_review_rate": 0.03, "label": "50% more analysts"}])):
    """Replay a proposed decision policy against the live one on labelled history. Changes nothing."""
    from bti.planning.whatif import WhatIfError, simulate
    try:
        return simulate(proposal)
    except WhatIfError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/whatif/{report_id}")
def whatif_report(report_id: str):
    import json
    from bti.planning.whatif import OUT_DIR
    if not report_id.isalnum():
        raise HTTPException(status_code=422, detail="bad id")
    path = OUT_DIR / f"{report_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No such what-if report")
    return json.loads(path.read_text())
