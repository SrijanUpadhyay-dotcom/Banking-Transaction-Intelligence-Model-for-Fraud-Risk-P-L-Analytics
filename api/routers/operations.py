"""
/api/v1/operations — confirmed-label feedback, fraud-ops KPIs,
champion/challenger comparison, policy backtesting and model-agnostic decisions.
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from bti.database import get_db
from bti.jurisdiction.policies import policy_for
from bti.modeling import registry
from bti.modeling.features import event_timestamps
from bti.modeling.scorer import scorer
from bti.modeling.train import LABEL, default_data_path
from bti.operations.decisioning import backtest_policy, cost_model_for, decide, fit_capacity
from bti.operations.feedback import (
    DEFAULT_MATURITY_DAYS, LABEL_SOURCES, LabelError, label_status, record_labels,
)
from bti.operations.kpis import champion_challenger, kpi_report, scoring_latency
from bti.config import get_settings
from api import metrics

router = APIRouter(prefix="/operations", tags=["Fraud Operations"])

MAX_LABELS = 5000


class LabelIn(BaseModel):
    transaction_id: str
    label_source: str = Field(..., description=f"One of {sorted(LABEL_SOURCES)}")
    label: Optional[int] = Field(None, ge=0, le=1, description="Optional; must agree with label_source")
    event_at: Optional[datetime] = None
    fraud_type: Optional[str] = None
    loss_amount: Optional[float] = None
    recovered_amount: Optional[float] = None
    currency: Optional[str] = None
    reported_by: Optional[str] = None
    notes: Optional[str] = None


class LabelBatch(BaseModel):
    labels: List[LabelIn]


class DecideRequest(BaseModel):
    fraud_probability: float = Field(..., ge=0, le=1, description="Calibrated probability from any model, e.g. SAS")
    amount_usd: float = Field(..., ge=0)
    country: Optional[str] = None
    channel: Optional[str] = None
    transaction_type: Optional[str] = None
    provisional_model: bool = False
    cost_overrides: Optional[Dict[str, float]] = None


class CapacityFitRequest(BaseModel):
    fitted_by: str
    model_id: Optional[str] = None
    max_review_rate: Optional[float] = Field(None, gt=0, le=0.5, description="Analyst capacity, share of traffic")
    max_step_up_rate: Optional[float] = Field(None, gt=0, le=0.5, description="Customer-challenge budget")


class BacktestRequest(BaseModel):
    max_review_rate: Optional[float] = Field(0.01, gt=0, le=1, description="Analyst capacity as share of traffic")
    max_step_up_rate: Optional[float] = Field(0.05, gt=0, le=1, description="Customer challenge budget")
    country: Optional[str] = None
    cost_overrides: Optional[Dict[str, float]] = None


@router.post("/labels", dependencies=[Depends(require_api_key)])
def ingest_labels(body: LabelBatch, db: Session = Depends(get_db)):
    """Record confirmed outcomes (chargebacks, investigator decisions, customer reports)."""
    if not 1 <= len(body.labels) <= MAX_LABELS:
        raise HTTPException(status_code=422, detail=f"Send between 1 and {MAX_LABELS} labels")
    try:
        return record_labels(db, [l.model_dump(exclude_none=True) for l in body.labels])
    except LabelError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/labels/status")
def labels_status(maturity_days: int = Query(DEFAULT_MATURITY_DAYS, ge=1, le=365), db: Session = Depends(get_db)):
    return label_status(db, maturity_days)


@router.get("/kpis")
def kpis(date_from: Optional[datetime] = Query(None, alias="from"), date_to: Optional[datetime] = Query(None, alias="to"),
         maturity_days: int = Query(DEFAULT_MATURITY_DAYS, ge=1, le=365), db: Session = Depends(get_db)):
    """Intervention rates, TDR, VDR, hit rate, false-positive ratio, false-decline rate, fraud loss in bps."""
    return kpi_report(db, date_from, date_to, maturity_days)


@router.get("/champion-challenger")
def champ_chall(date_from: Optional[datetime] = Query(None, alias="from"),
                date_to: Optional[datetime] = Query(None, alias="to"),
                maturity_days: int = Query(DEFAULT_MATURITY_DAYS, ge=1, le=365), db: Session = Depends(get_db)):
    """Live champion vs shadow challenger on the same transactions."""
    return champion_challenger(db, date_from, date_to, maturity_days)


@router.get("/service-metrics")
def service_metrics(hours: int = Query(24, ge=1, le=24 * 90), db: Session = Depends(get_db)):
    """
    Uptime, request volume, 5xx error rate and end-to-end latency per endpoint
    group for this worker, plus model scoring latency from the score log
    against the configured SLA.
    """
    sla = get_settings().scoring_latency_sla_ms
    return {**metrics.snapshot(), "scoring": scoring_latency(db, hours, sla)}


@router.post("/decide")
def decide_endpoint(body: DecideRequest):
    """
    Expected-cost decision for a probability from any model — lets BTI act as the
    decision layer over SAS or another incumbent score.
    """
    policy = policy_for(body.country)
    d = decide(body.fraud_probability, body.amount_usd, policy, body.channel, body.transaction_type,
               provisional_model=body.provisional_model, cost_overrides=body.cost_overrides)
    return {**asdict(d), "jurisdiction": policy.iso2 if policy else None}


@router.post("/policy/backtest")
def policy_backtest(body: BacktestRequest):
    """
    Replay the expected-cost policy on the scoring model's out-of-time window and
    compare it with approve-all and a single-threshold decline rule. Review
    capacity is fitted on the calibration window, never on the test window.
    """
    try:
        model_id, role, _ = scorer.resolve("champion")
    except registry.RegistryError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    card = registry.load_card(model_id)
    path = Path(card["data"]["path"])
    if not path.exists():
        path = Path(default_data_path())
    if not path.exists():
        raise HTTPException(status_code=503, detail="Training data for the backtest is not available")
    df = pd.read_csv(path, low_memory=False)
    scored = scorer.score_frame(df, role=role)
    ts = event_timestamps(df)
    te = (ts >= pd.Timestamp(card["data"]["split"]["out_of_time_test"]["from"])).to_numpy()
    ca = ((ts >= pd.Timestamp(card["data"]["split"]["calibration"]["from"])) & ~te).to_numpy()
    cm = cost_model_for(policy_for(body.country), body.cost_overrides)
    cm = fit_capacity(scored["fraud_probability"][ca], scored["amount_usd"][ca], df["channel"][ca],
                      df["transaction_type"][ca], cm, body.max_review_rate, body.max_step_up_rate)
    result = backtest_policy(df[LABEL][te], scored["fraud_probability"][te], scored["amount_usd"][te],
                             df["channel"][te], df["transaction_type"][te], cm,
                             reference_threshold=card["methodology"]["reference_threshold"])
    return {"model_id": model_id, "window": "out_of_time_test", "max_review_rate": body.max_review_rate,
            "max_step_up_rate": body.max_step_up_rate, **result}


@router.get("/policy/capacity")
def capacity_policy(model_id: Optional[str] = None):
    """The capacity policy (review and step-up shadow prices) the scoring model decides with."""
    from bti.modeling import registry
    from bti.operations.capacity import load_policy
    model_id = model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
    policy = load_policy(model_id) if model_id else None
    if policy is None:
        raise HTTPException(status_code=404, detail=f"No capacity policy fitted for {model_id}")
    return policy


@router.post("/policy/capacity/fit", dependencies=[Depends(require_api_key)])
def capacity_fit(body: CapacityFitRequest):
    """Refit the live capacity policy on the model's calibration window."""
    from bti.operations.capacity import fit_live_policy
    try:
        return fit_live_policy(body.fitted_by, body.model_id, body.max_review_rate, body.max_step_up_rate)
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))



# ── Continuous learning ──────────────────────────────────────────────────────

class RecalibrationRollback(BaseModel):
    model_id: str
    actor: str
    reason: str = Field(..., min_length=10)


@router.get("/retraining/triggers")
def retraining_triggers(db: Session = Depends(get_db)):
    """Why (or why not) retraining would run now: schedule, drift, performance findings, new labels."""
    from bti.modeling.retrain import retraining_triggers as triggers
    return triggers(db)


@router.post("/retraining/run", dependencies=[Depends(require_api_key)])
def retraining_run(force: bool = False, notify: bool = True, db: Session = Depends(get_db)):
    """Retrain on the latest extract with confirmed labels applied. A winner becomes a shadow challenger only."""
    from bti.modeling.retrain import run_retraining
    return run_retraining(db, force=force, notify=notify)


@router.get("/recalibration")
def recalibration_state(model_id: Optional[str] = None):
    """The active recalibration overlay (and history) for a model."""
    from bti.modeling.recalibration import load_overlay
    model_id = model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
    return load_overlay(model_id) or {"model_id": model_id, "current": None, "history": []}


@router.post("/recalibration/run", dependencies=[Depends(require_api_key)])
def recalibration_run(model_id: Optional[str] = None, apply: bool = True, window_days: int = Query(90, ge=14, le=365),
                      db: Session = Depends(get_db)):
    """Fit a bounded, rank-preserving overlay on matured labels; applied only if it passes every limit."""
    from bti.modeling.recalibration import recalibrate
    return recalibrate(db, model_id, window_days=window_days, apply=apply)


@router.post("/recalibration/rollback", dependencies=[Depends(require_api_key)])
def recalibration_rollback(body: RecalibrationRollback, db: Session = Depends(get_db)):
    from bti.modeling.recalibration import rollback
    try:
        return rollback(db, body.model_id, body.actor, body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))



# ── Cost model ───────────────────────────────────────────────────────────────

@router.get("/cost-model")
def cost_model_state(db: Session = Depends(get_db)):
    """Which cost model decides, its bounds, and the step-up rates it would use (measured or prior)."""
    from bti.database.models import CustomerValue
    from bti.operations.cost_model import measured_stepup
    from sqlalchemy import func
    s = get_settings()
    return {"version": s.cost_model_version, "customer_value_bounds_usd": s.cost_v2_value_bounds,
            "customers_valued": db.query(func.count(CustomerValue.customer_id)).scalar() or 0,
            "step_up_rates": measured_stepup(db),
            "note": "v2 uses per-customer value (bounded) and measured step-up outcomes; segment is never priced."}


@router.post("/cost-model/customer-values/refresh", dependencies=[Depends(require_api_key)])
def refresh_values(db: Session = Depends(get_db)):
    from bti.operations.cost_model import refresh_customer_values
    return refresh_customer_values(db)


@router.post("/cost-model/backtest", dependencies=[Depends(require_api_key)])
def cost_backtest(model_id: Optional[str] = None):
    """Replay v1 and v2 on out-of-time history: realised cost, action mix, raw and amount-standardised
    decision fairness, and a recommendation."""
    from bti.operations.cost_model import backtest
    return backtest(model_id)
