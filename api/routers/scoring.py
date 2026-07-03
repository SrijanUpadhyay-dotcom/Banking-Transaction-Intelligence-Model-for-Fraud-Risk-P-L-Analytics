"""
/api/v1/score — real-time single-transaction fraud scoring.

This is the core production endpoint: submit one transaction, get a full
fraud risk assessment back in <100ms.

No pipeline needed — uses pre-trained models loaded into memory at startup.
"""

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from bti.database import get_db
from bti.scoring.realtime import RealTimeScorer, ScoreResult
from bti.scoring.model_loader import models_available
from bti.logging_config import get_logger, AuditLogger
from bti.config import get_settings

router = APIRouter(prefix="/score", tags=["Real-Time Scoring"])
log = get_logger("api.scoring")
settings = get_settings()
_scorer = RealTimeScorer()


# ── Request schema ─────────────────────────────────────────────────────────────

class TransactionScoreRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id:    str = Field(..., description="Customer identifier")
    account_id:     Optional[str] = None
    transaction_date: Optional[str] = Field(
        default=None, description="ISO date string (YYYY-MM-DD)"
    )
    transaction_time: Optional[str] = Field(
        default="12:00:00", description="HH:MM:SS local time"
    )
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in base currency")
    transaction_type:   Optional[str] = None
    debit_credit_flag:  Optional[str] = Field(default="Debit", pattern="^(Debit|Credit)$")
    channel:            Optional[str] = None
    merchant_category:  Optional[str] = None
    merchant_name:      Optional[str] = None
    customer_segment:   Optional[str] = None
    account_balance_before: Optional[float] = None
    account_balance_after:  Optional[float] = None
    risk_score:             Optional[float] = Field(default=50, ge=0, le=100)
    historical_average_transaction_amount: Optional[float] = Field(default=500.0, gt=0)
    failed_attempt_count:  Optional[int] = Field(default=0, ge=0)
    login_attempts:        Optional[int] = Field(default=1, ge=0)
    refund_flag:           Optional[int] = Field(default=0, ge=0, le=1)
    chargeback_flag:       Optional[int] = Field(default=0, ge=0, le=1)
    reversal_flag:         Optional[int] = Field(default=0, ge=0, le=1)
    authorization_method:  Optional[str] = None
    device_id:             Optional[str] = None
    ip_location:           Optional[str] = None
    currency:              Optional[str] = Field(default="USD")

    # ML feature extras (optional — fall back to 0 if not provided)
    monthly_customer_transaction_count: Optional[int] = None
    fee_income:        Optional[float] = None
    interchange_income: Optional[float] = None
    processing_cost:   Optional[float] = None
    chargeback_loss:   Optional[float] = None
    refund_loss:       Optional[float] = None
    fraud_loss:        Optional[float] = None
    net_revenue:       Optional[float] = None
    net_pnl_impact:    Optional[float] = None

    model_config = {"json_schema_extra": {
        "example": {
            "transaction_id": "TXN-20240715-00001",
            "customer_id": "CUST-4821",
            "transaction_date": "2024-07-15",
            "transaction_time": "02:47:00",
            "transaction_amount": 12500.00,
            "channel": "API/Open Banking",
            "merchant_category": "Crypto Exchanges",
            "merchant_name": "CryptoFX Ltd",
            "customer_segment": "Mass Market",
            "risk_score": 72,
            "historical_average_transaction_amount": 450.0,
            "failed_attempt_count": 4,
            "login_attempts": 6,
            "account_balance_before": 13000.00,
            "account_balance_after": 500.00,
            "debit_credit_flag": "Debit",
            "device_id": "DEV-X9921-UNKNOWN",
            "ip_location": "203.0.113.45",
            "refund_flag": 0,
            "chargeback_flag": 0,
        }
    }}


class RuleDetail(BaseModel):
    rule_name: str
    fired:     bool
    weight:    int


class TransactionScoreResponse(BaseModel):
    transaction_id:     str
    scored_at:          str
    fraud_rule_score:   float = Field(..., description="Rule-based score 0–100")
    rules_triggered:    int
    rules_fired:        List[str]
    ml_lr_proba:        float = Field(..., description="Logistic regression fraud probability 0–1")
    ml_rf_proba:        float = Field(..., description="Random forest fraud probability 0–1")
    ml_iso_score:       float = Field(..., description="Isolation forest anomaly score")
    ml_anomaly_score:   float = Field(..., description="Combined ML score 0–100")
    final_risk_score:   float = Field(..., description="Composite risk score 0–100")
    final_alert_tier:   str   = Field(..., description="LOW / MEDIUM / HIGH / VERY HIGH / CRITICAL")
    is_suspicious:      bool
    processing_time_ms: float
    model_version:      str
    db_context_used:    bool
    recommendation:     str


def _recommendation(result: ScoreResult) -> str:
    tier = result.final_alert_tier
    if tier == "CRITICAL":
        return "BLOCK — Immediately decline and escalate to Fraud Operations"
    elif tier == "VERY HIGH":
        return "HOLD — Step-up authentication required before processing"
    elif tier == "HIGH":
        return "REVIEW — Flag for analyst review within 1 business hour"
    elif tier == "MEDIUM":
        return "MONITOR — Log for batch review; allow with enhanced monitoring"
    else:
        return "ALLOW — Transaction within normal risk parameters"


@router.post("/", response_model=TransactionScoreResponse,
             summary="Score a single transaction in real-time")
def score_transaction_endpoint(
    body: TransactionScoreRequest,
    db: Session = Depends(get_db),
):
    """
    Submit a transaction and receive a full fraud risk assessment in <100ms.

    The response includes:
    - Rule-based fraud score (19 rules, 0–100)
    - ML ensemble score (Isolation Forest + Logistic Regression + Random Forest)
    - Composite final risk score (0–100)
    - Alert tier (LOW → CRITICAL)
    - Actionable recommendation (ALLOW / MONITOR / REVIEW / HOLD / BLOCK)
    """
    if not models_available():
        raise HTTPException(
            status_code=503,
            detail="ML models not yet trained. Run: python main.py pipeline --force"
        )

    txn_dict = body.model_dump()
    result = _scorer.score(txn_dict, db_session=db)

    # Async audit log
    AuditLogger(settings.audit_log_path).record(
        "REALTIME_SCORE", result.transaction_id,
        {
            "final_risk_score": result.final_risk_score,
            "alert_tier": result.final_alert_tier,
            "rules_triggered": result.rules_triggered,
        }
    )

    if result.final_alert_tier in ("CRITICAL", "VERY HIGH"):
        log.warning(
            "High-risk transaction scored",
            extra={
                "transaction_id": result.transaction_id,
                "tier": result.final_alert_tier,
                "score": result.final_risk_score,
                "rules": result.rules_triggered,
                "ms": result.processing_time_ms,
            }
        )

    return TransactionScoreResponse(
        transaction_id=result.transaction_id,
        scored_at=datetime.utcnow().isoformat() + "Z",
        fraud_rule_score=result.fraud_rule_score,
        rules_triggered=result.rules_triggered,
        rules_fired=result.rules_fired,
        ml_lr_proba=result.ml_lr_proba,
        ml_rf_proba=result.ml_rf_proba,
        ml_iso_score=result.ml_iso_score,
        ml_anomaly_score=result.ml_anomaly_score,
        final_risk_score=result.final_risk_score,
        final_alert_tier=result.final_alert_tier,
        is_suspicious=result.is_suspicious,
        processing_time_ms=result.processing_time_ms,
        model_version=result.model_version,
        db_context_used=result.db_context_used,
        recommendation=_recommendation(result),
    )


@router.get("/model-info", summary="Model version and performance metrics")
def get_model_info():
    """Returns current model version, training date, and validation metrics."""
    if not models_available():
        raise HTTPException(status_code=503, detail="Models not trained yet")
    bundle = _scorer.bundle
    return {
        "trained_at": bundle.trained_at,
        "feature_count": len(bundle.feature_cols),
        "logistic_regression": {"roc_auc": bundle.lr_roc_auc},
        "random_forest": {"roc_auc": bundle.rf_roc_auc, "f1": bundle.rf_f1},
    }


@router.post("/reload-models",
             summary="Reload ML models from disk (call after retraining)")
def reload_models():
    """Forces the scorer to reload all model artifacts from disk."""
    _scorer.reload()
    return {"status": "reloaded", "trained_at": _scorer.bundle.trained_at}
