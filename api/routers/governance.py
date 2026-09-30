# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
/api/v1/governance — model inventory, documentation, promotion, drift,
leakage screening and jurisdiction policy.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from bti.database import get_db
from bti.governance import scheduler as monitoring_scheduler
from bti.governance.drift_job import drift_history, run_drift_check
from bti.governance.model_card import render_model_card
from bti.jurisdiction.policies import DISCLAIMER, POLICIES, policy_dict, policy_for, tra_eligibility
from bti.modeling import registry
from bti.modeling.features import leakage_audit
from bti.modeling.train import LABEL, default_data_path
from bti.operations.kpis import live_drift

router = APIRouter(prefix="/governance", tags=["Model Governance"])


class PromotionRequest(BaseModel):
    role: Literal["champion", "challenger"]
    approver: str = Field(..., min_length=2)
    rationale: str = Field(..., min_length=10)


class TRARequest(BaseModel):
    fraud_value_eur: float = Field(..., ge=0)
    total_value_eur: float = Field(..., gt=0)
    payment_type: Literal["remote_card", "credit_transfer"] = "remote_card"


def _card_or_404(model_id: str) -> dict:
    try:
        return registry.load_card(model_id)
    except registry.RegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/models")
def list_models():
    """Model inventory: every registered model, current roles and the promotion history."""
    return registry.read_index()


@router.get("/models/{model_id}")
def model_card(model_id: str, include_baseline: bool = Query(False)):
    """Full model card (validation, fairness, lineage). The monitoring baseline is omitted unless requested."""
    card = _card_or_404(model_id)
    if not include_baseline:
        card = {**card, "monitoring": {k: v for k, v in card["monitoring"].items() if k != "baseline"}}
    return card


@router.get("/models/{model_id}/documentation", response_class=PlainTextResponse)
def model_documentation(model_id: str, db: Session = Depends(get_db)):
    """Model documentation pack in Markdown, structured for SR 11-7 / PRA SS1/23 review, with the validation
    evidence: benchmarking and stress tests, outcomes analysis, sign-offs, findings and champion readiness."""
    from bti.governance.benchmarking import latest_report
    from bti.governance.outcomes import outcomes_history
    from bti.governance.validation import champion_readiness, list_findings, list_signoffs
    card = _card_or_404(model_id)
    outcome = next((dict(m, window=f"{run['window']['from']} to {run['window']['to']}")
                    for run in outcomes_history(db, 20) for m in run.get("models", []) if m["model_id"] == model_id),
                   None)
    validation = {"benchmark": latest_report(model_id), "outcomes": outcome,
                  "readiness": champion_readiness(db, model_id), "signoffs": list_signoffs(db, model_id),
                  "findings": list_findings(db, model_id)}
    return PlainTextResponse(render_model_card(card, registry.read_index(), validation), media_type="text/markdown")


@router.post("/models/{model_id}/promote", dependencies=[Depends(require_api_key)])
def promote_model(model_id: str, body: PromotionRequest, db: Session = Depends(get_db)):
    """
    Assign a model to champion or challenger. Champion requires passed automated gates, an in-date approving
    sign-off from a validator other than the developer, no open high-severity findings, and an approver who is
    not the model developer (four-eyes).
    """
    from bti.governance.validation import ValidationError, assert_ready_for_champion
    _card_or_404(model_id)
    try:
        if body.role == "champion":
            registry.check_four_eyes(model_id, body.approver)
            assert_ready_for_champion(db, model_id)
        event = registry.assign_role(model_id, body.role, body.approver, body.rationale)
    except (registry.RegistryError, ValidationError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    from bti.database.models import AuditLog
    db.add(AuditLog(ts=datetime.utcnow(), event_type="MODEL_ROLE_ASSIGNED", payload=event))
    db.commit()
    return {"status": "ok", "event": event}


@router.get("/drift")
def drift(date_from: Optional[datetime] = Query(None, alias="from"),
          date_to: Optional[datetime] = Query(None, alias="to"),
          shadow: bool = False, model_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Score PSI and per-feature CSI for logged traffic against the model's training baseline."""
    return live_drift(db, date_from, date_to, model_id=model_id, shadow=shadow)


@router.post("/drift/run", dependencies=[Depends(require_api_key)])
def drift_run(window_days: Optional[int] = Query(None, ge=1, le=365), notify: bool = True,
              db: Session = Depends(get_db)):
    """Run the population-stability check now (the scheduler runs it weekly); alerts if drift is found."""
    return run_drift_check(db, window_days=window_days, notify=notify)


@router.get("/drift/history")
def drift_runs(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    """Previous drift checks, newest first, from the audit log."""
    return {"runs": drift_history(db, limit)}


@router.get("/monitoring/schedule")
def monitoring_schedule():
    return monitoring_scheduler.status()


@router.get("/leakage-audit")
def leakage_screen(threshold: float = Query(0.97, ge=0.5, le=1.0)):
    """Single-feature separability screen over the training source data; flags label leakage."""
    path = Path(default_data_path())
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Training data not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    findings = leakage_audit(df, LABEL, threshold=threshold)
    return {"source": str(path), "threshold": threshold,
            "suspected_leaks": [f for f in findings if f["suspected_leak"]],
            "all": findings}


@router.get("/jurisdictions")
def jurisdictions():
    return {"disclaimer": DISCLAIMER, "jurisdictions": [policy_dict(p) for p in POLICIES.values()]}


@router.get("/jurisdictions/{code}")
def jurisdiction(code: str):
    policy = policy_for(code)
    if policy is None:
        raise HTTPException(status_code=404, detail=f"No policy for '{code}'. Known: {sorted(POLICIES)}")
    return policy_dict(policy)


@router.post("/psd2/tra-eligibility")
def psd2_tra(body: TRARequest):
    """Highest PSD2 transaction-risk-analysis exemption threshold available at a given fraud rate."""
    return tra_eligibility(body.fraud_value_eur, body.total_value_eur, body.payment_type)


# ── Tamper-evident audit storage ─────────────────────────────────────────────

@router.get("/audit/verify")
def audit_verify(db: Session = Depends(get_db)):
    """Walk the audit hash chain and report the first altered, removed or reordered row, if any."""
    from bti.governance.audit_chain import verify_chain
    return verify_chain(db)


@router.post("/audit/archive", dependencies=[Depends(require_api_key)])
def audit_archive(day: Optional[date] = None, db: Session = Depends(get_db)):
    """Seal one UTC day (default: yesterday) of audit rows into a read-only segment with a manifest."""
    from bti.governance.audit_chain import archive_day
    return archive_day(db, day)


@router.get("/audit/archive/{day}/verify")
def audit_archive_verify(day: date, db: Session = Depends(get_db)):
    from bti.governance.audit_chain import verify_archive
    return verify_archive(db, day)


@router.get("/audit/retention")
def audit_retention():
    """Retention period and archive segments past it (reported only; disposal is a controlled procedure)."""
    from bti.governance.audit_chain import retention_report
    return retention_report()


# ── Validation workflow ──────────────────────────────────────────────────────

class FindingIn(BaseModel):
    model_id: str
    title: str = Field(..., min_length=5, max_length=200)
    severity: Literal["high", "medium", "low"]
    category: str
    source: str
    description: str = Field(..., min_length=10)
    raised_by: str
    owner: str
    due_date: Optional[datetime] = None


class FindingUpdate(BaseModel):
    status: Literal["open", "remediating", "closed", "risk_accepted"]
    actor: str
    resolution: Optional[str] = None
    evidence: Optional[dict] = None
    accepted_until: Optional[datetime] = None


class SignoffIn(BaseModel):
    model_id: str
    validator: str
    validator_role: Optional[str] = None
    decision: Literal["approve", "approve_with_conditions", "reject"]
    scope: str = Field(..., min_length=10)
    conditions: Optional[str] = None
    evidence: Optional[dict] = None


def _workflow(fn, *args, **kwargs):
    from bti.governance.validation import ValidationError
    try:
        return fn(*args, **kwargs)
    except ValidationError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except registry.RegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/inventory")
def inventory(db: Session = Depends(get_db)):
    """Model inventory: every registered model with role, lifecycle, last validation and next review."""
    from bti.governance.validation import sync_inventory
    return {"models": sync_inventory(db)}


@router.get("/findings")
def findings(model_id: Optional[str] = None, status: Optional[str] = None, db: Session = Depends(get_db)):
    from bti.governance.validation import list_findings
    return {"findings": list_findings(db, model_id, status)}


@router.post("/findings", dependencies=[Depends(require_api_key)], status_code=201)
def raise_finding(body: FindingIn, db: Session = Depends(get_db)):
    from bti.governance import validation
    return _workflow(validation.raise_finding, db, **body.model_dump())


@router.patch("/findings/{finding_id}", dependencies=[Depends(require_api_key)])
def update_finding(finding_id: int, body: FindingUpdate, db: Session = Depends(get_db)):
    """Move a finding on. Closing needs evidence and someone other than the owner; risk acceptance needs someone
    other than the owner and the developer, time-limited."""
    from bti.governance import validation
    return _workflow(validation.update_finding, db, finding_id, **body.model_dump())


@router.get("/signoffs")
def signoffs(model_id: Optional[str] = None, db: Session = Depends(get_db)):
    from bti.governance.validation import list_signoffs
    return {"signoffs": list_signoffs(db, model_id)}


@router.post("/signoffs", dependencies=[Depends(require_api_key)], status_code=201)
def record_signoff(body: SignoffIn, db: Session = Depends(get_db)):
    """Independent validation sign-off (validator ≠ developer; approval blocked by open high findings)."""
    from bti.governance import validation
    return _workflow(validation.record_signoff, db, **body.model_dump())


@router.get("/models/{model_id}/readiness")
def readiness(model_id: str, db: Session = Depends(get_db)):
    """What stands between this model and the champion role."""
    from bti.governance.validation import champion_readiness
    return _workflow(champion_readiness, db, model_id)


@router.get("/review-schedule")
def schedule(db: Session = Depends(get_db)):
    """Periodic review dates for models in use, and findings past their due date."""
    from bti.governance.validation import review_schedule
    return review_schedule(db)


# ── Benchmarking, sensitivity and stress testing ─────────────────────────────

@router.post("/models/{model_id}/benchmark", dependencies=[Depends(require_api_key)])
def run_benchmark(model_id: str):
    """Benchmark against logistic regression, alternatives and rules; sensitivity; monotonicity; stress tests."""
    from bti.governance.benchmarking import run_and_store
    _card_or_404(model_id)
    try:
        return run_and_store(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/models/{model_id}/benchmark")
def get_benchmark(model_id: str):
    from bti.governance.benchmarking import latest_report
    _card_or_404(model_id)
    report = latest_report(model_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No benchmark report for {model_id}; POST to run one")
    return report



# ── Outcomes analysis and governance checks ──────────────────────────────────

@router.get("/outcomes")
def outcomes(date_from: Optional[datetime] = Query(None, alias="from"),
             date_to: Optional[datetime] = Query(None, alias="to"),
             maturity_days: int = Query(90, ge=1, le=365), db: Session = Depends(get_db)):
    """Live discrimination, calibration and fairness on matured labels against development (no findings raised)."""
    from bti.governance.outcomes import outcomes_analysis
    return outcomes_analysis(db, date_from, date_to, maturity_days)


@router.post("/outcomes/run", dependencies=[Depends(require_api_key)])
def outcomes_run(notify: bool = True, db: Session = Depends(get_db)):
    """Run the quarterly outcomes analysis now; degradation raises findings in the tracker."""
    from bti.governance.outcomes import run_quarterly
    return run_quarterly(db, notify=notify)


@router.get("/outcomes/history")
def outcomes_history(limit: int = Query(8, ge=1, le=40), db: Session = Depends(get_db)):
    from bti.governance.outcomes import outcomes_history as history
    return {"runs": history(db, limit)}


@router.post("/check", dependencies=[Depends(require_api_key)])
def governance_check(notify: bool = True, db: Session = Depends(get_db)):
    """Reviews due or overdue and findings past their due date (runs weekly on the scheduler)."""
    from bti.governance.validation import run_governance_check
    return run_governance_check(db, notify=notify)
