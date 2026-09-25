"""
/api/v1/parallel — run BTI alongside the incumbent fraud platform (e.g. SAS).

Four parts:

- **Incumbent feed.** The incumbent's scores and decisions for the same
  transactions come in as JSON batches or file uploads.
- **Parallel-run report.** BTI is compared with the incumbent at equal
  intervention rate on matured labels, weekly and on demand.
- **Randomised traffic split.** A four-eyes, capped experiment that routes a
  share of live decisions to BTI.
- **Router and reconciliation.** `POST /parallel/decide` returns the effective
  decision, falls back to the incumbent on a BTI timeout or error, and the two
  decision logs are reconciled daily.
"""

import io
import json
from dataclasses import asdict
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.routers.v3 import V3Transaction
from api.security import require_api_key
from bti.database import get_db
from bti.operations.feedback import DEFAULT_MATURITY_DAYS
from bti.parallel import incumbent, reconcile, report, traffic

router = APIRouter(prefix="/parallel", tags=["Parallel run alongside the incumbent"])
MAX_ROWS = 50_000


class IncumbentRecord(BaseModel):
    transaction_id: str
    decision: str = Field(..., description="The incumbent's action code (mapped to APPROVE/STEP_UP/REVIEW/DECLINE)")
    score: Optional[float] = None
    decided_at: Optional[datetime] = None
    executed_decision: Optional[str] = Field(None, description="What the bank's switch actually did, if known")
    latency_ms: Optional[float] = None
    customer_id: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    rule_ids: Optional[List[str]] = None


class IncumbentBatch(BaseModel):
    system: Optional[str] = Field(None, description="Defaults to parallel_run.incumbent_system (SAS)")
    records: List[IncumbentRecord]


class ExperimentProposal(BaseModel):
    name: str = Field(..., min_length=3, max_length=80)
    bti_share: float = Field(..., gt=0, le=1)
    unit: str = "customer"
    proposed_by: str
    rationale: str


class ExperimentApproval(BaseModel):
    approver: str


class ExperimentStop(BaseModel):
    stopped_by: str
    reason: str = Field(..., min_length=5)


class DecideRequest(BaseModel):
    transaction: V3Transaction
    incumbent_decision: Optional[str] = Field(None, description="The incumbent's decision for this transaction, "
                                                                "sent so BTI can fall back to it")
    timeout_ms: Optional[float] = Field(None, gt=0, le=5000)


def _bad(exc: Exception, code: int = 422):
    raise HTTPException(status_code=code, detail=str(exc))


# ── Incumbent feed ───────────────────────────────────────────────────────────

@router.post("/incumbent/decisions", dependencies=[Depends(require_api_key)])
def ingest_incumbent(body: IncumbentBatch, db: Session = Depends(get_db)):
    """Incumbent scores and decisions (JSON). Valid rows are stored; rejected rows come back with reasons."""
    if not 1 <= len(body.records) <= MAX_ROWS:
        raise HTTPException(status_code=422, detail=f"Send between 1 and {MAX_ROWS} records")
    try:
        return incumbent.ingest(db, [r.model_dump() for r in body.records], system=body.system)
    except ValueError as exc:
        _bad(exc)


@router.post("/incumbent/upload", dependencies=[Depends(require_api_key)])
async def upload_incumbent(file: UploadFile = File(..., description="CSV, Excel or Parquet decision export"),
                           system: Optional[str] = Form(None),
                           column_map: str = Form("{}", description='JSON, e.g. {"TXN_ID": "transaction_id"}'),
                           db: Session = Depends(get_db)):
    """Incumbent decision export as a file. `column_map` renames the vendor's columns to BTI's field names."""
    content, name = await file.read(), (file.filename or "upload").lower()
    try:
        mapping = json.loads(column_map or "{}")
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content), dtype=str)
        elif name.endswith((".parquet", ".pq")):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content), dtype=str)
        rows = incumbent.frame_to_rows(df, mapping)
    except (ValueError, json.JSONDecodeError) as exc:
        _bad(exc, 400)
    if not 1 <= len(rows) <= MAX_ROWS:
        raise HTTPException(status_code=400, detail=f"The file must hold between 1 and {MAX_ROWS} rows")
    try:
        return incumbent.ingest(db, rows, system=system)
    except ValueError as exc:
        _bad(exc)


@router.get("/incumbent/status")
def incumbent_status(date_from: Optional[datetime] = Query(None, alias="from"),
                     date_to: Optional[datetime] = Query(None, alias="to"), db: Session = Depends(get_db)):
    """How well the incumbent's and BTI's decision streams line up."""
    return incumbent.ingestion_status(db, date_from, date_to)


@router.get("/incumbent/decision-map")
def incumbent_decision_map():
    """Vendor decision codes and the BTI action each maps to (extend under parallel_run.decision_map)."""
    return incumbent.decision_map()


# ── Parallel-run report ──────────────────────────────────────────────────────

@router.get("/report")
def parallel_report(date_from: Optional[datetime] = Query(None, alias="from"),
                    date_to: Optional[datetime] = Query(None, alias="to"),
                    maturity_days: int = Query(DEFAULT_MATURITY_DAYS, ge=1, le=365),
                    db: Session = Depends(get_db)):
    """BTI vs the incumbent on the same transactions: agreement, equal-intervention-rate detection, significance."""
    return report.parallel_run_report(db, date_from, date_to, maturity_days)


@router.post("/report/run", dependencies=[Depends(require_api_key)])
def parallel_report_run(notify: bool = True, db: Session = Depends(get_db)):
    """Run the weekly report now (last 7 days, the cohort that matured, and to date), store it, alert on issues."""
    return report.run_weekly_report(db, notify=notify)


@router.get("/reports")
def parallel_reports(limit: int = Query(12, ge=1, le=100), db: Session = Depends(get_db)):
    return {"reports": report.report_history(db, limit)}


# ── Randomised traffic split ─────────────────────────────────────────────────

@router.get("/experiments")
def experiments(db: Session = Depends(get_db)):
    return {"experiments": traffic.list_experiments(db)}


@router.post("/experiments", dependencies=[Depends(require_api_key)], status_code=201)
def propose_experiment(body: ExperimentProposal, db: Session = Depends(get_db)):
    """Propose a split. It does not start until a different person approves it."""
    try:
        return traffic.propose(db, body.name, body.bti_share, body.proposed_by, body.rationale, body.unit)
    except traffic.ExperimentError as exc:
        _bad(exc)


@router.post("/experiments/{experiment_id}/approve", dependencies=[Depends(require_api_key)])
def approve_experiment(experiment_id: int, body: ExperimentApproval, db: Session = Depends(get_db)):
    """Start a proposed split. Needs a different approver and an approved champion model."""
    try:
        return traffic.approve(db, experiment_id, body.approver)
    except traffic.ExperimentError as exc:
        _bad(exc, 409)


@router.post("/experiments/{experiment_id}/stop", dependencies=[Depends(require_api_key)])
def stop_experiment(experiment_id: int, body: ExperimentStop, db: Session = Depends(get_db)):
    """Stop immediately; all traffic returns to the incumbent."""
    try:
        return traffic.stop(db, experiment_id, body.stopped_by, body.reason)
    except traffic.ExperimentError as exc:
        _bad(exc, 409)


@router.get("/experiments/{experiment_id}/report")
def experiment_report(experiment_id: int, maturity_days: int = Query(DEFAULT_MATURITY_DAYS, ge=1, le=365),
                      db: Session = Depends(get_db)):
    """Arms compared on matured outcomes with customer-level bootstrap intervals and a sample-ratio check."""
    try:
        return traffic.experiment_report(db, experiment_id, maturity_days)
    except traffic.ExperimentError as exc:
        _bad(exc, 404)


# ── Router with fallback ─────────────────────────────────────────────────────

@router.post("/decide", dependencies=[Depends(require_api_key)])
def decide(body: DecideRequest, db: Session = Depends(get_db)):
    """
    The decision to apply for one transaction. BTI scores every call. In the BTI arm of a running experiment,
    BTI's decision applies; otherwise, or when BTI errors or exceeds the time budget, the incumbent's does.
    """
    try:
        result = traffic.route(db, body.transaction.model_dump(), body.incumbent_decision, body.timeout_ms)
    except traffic.ExperimentError as exc:
        _bad(exc)
    return asdict(result)


# ── Reconciliation ───────────────────────────────────────────────────────────

@router.post("/reconcile/run", dependencies=[Depends(require_api_key)])
def reconcile_run(day: Optional[date] = None, notify: bool = True, db: Session = Depends(get_db)):
    """Reconcile one UTC day (default: yesterday) of routed decisions against the incumbent's log."""
    return reconcile.reconcile(db, day, notify=notify)


@router.get("/reconcile/history")
def reconcile_history(limit: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)):
    return {"runs": reconcile.reconciliation_history(db, limit)}
