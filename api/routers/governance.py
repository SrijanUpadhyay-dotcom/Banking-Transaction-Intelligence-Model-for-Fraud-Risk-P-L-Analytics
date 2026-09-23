"""
/api/v1/governance — model inventory, documentation, promotion, drift,
leakage screening and jurisdiction policy.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import require_api_key
from bti.database import get_db
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
def model_documentation(model_id: str):
    """Model documentation pack in Markdown, structured for SR 11-7 / PRA SS1/23 review."""
    card = _card_or_404(model_id)
    return PlainTextResponse(render_model_card(card, registry.read_index()), media_type="text/markdown")


@router.post("/models/{model_id}/promote", dependencies=[Depends(require_api_key)])
def promote_model(model_id: str, body: PromotionRequest):
    """
    Assign a model to champion or challenger. Champion requires passed validation
    and an approver who is not the model developer (four-eyes).
    """
    _card_or_404(model_id)
    try:
        event = registry.assign_role(model_id, body.role, body.approver, body.rationale)
    except registry.RegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "ok", "event": event}


@router.get("/drift")
def drift(date_from: Optional[datetime] = Query(None, alias="from"),
          date_to: Optional[datetime] = Query(None, alias="to"),
          shadow: bool = False, model_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Score PSI and per-feature CSI for logged traffic against the model's training baseline."""
    return live_drift(db, date_from, date_to, model_id=model_id, shadow=shadow)


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
