"""
/api/v1/score — real-time single-transaction fraud scoring.

This is the core production endpoint: submit one transaction, get a full
fraud risk assessment back in <100ms.

No pipeline needed — uses pre-trained models loaded into memory at startup.
"""

import io
from collections import Counter
from datetime import datetime
from typing import Optional, List

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
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


# ── Bulk CSV / Excel Upload ────────────────────────────────────────────────────

_COL_ALIASES = {
    "txn_id": "transaction_id",
    "id": "transaction_id",
    "amount": "transaction_amount",
    "cust_id": "customer_id",
    "date": "transaction_date",
    "time": "transaction_time",
}


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.rename(columns=_COL_ALIASES, inplace=True)
    return df


def _row_to_txn(row: dict, idx: int) -> dict:
    txn: dict = {}
    txn["transaction_id"] = str(row.get("transaction_id") or f"UPLOAD-{idx:06d}")
    txn["customer_id"]    = str(row.get("customer_id")    or f"CUST-UNKNOWN-{idx:06d}")
    txn["transaction_amount"] = float(row.get("transaction_amount") or 0)
    txn["transaction_date"]   = str(row.get("transaction_date") or "2024-01-01")
    txn["transaction_time"]   = str(row.get("transaction_time") or "12:00:00")

    for field in ["channel", "merchant_category", "merchant_name", "customer_segment",
                  "debit_credit_flag", "device_id", "ip_location", "currency",
                  "authorization_method", "transaction_type", "account_id"]:
        val = row.get(field)
        if val is not None and str(val) not in ("", "nan", "NaN", "None"):
            txn[field] = str(val)

    for field in ["risk_score", "historical_average_transaction_amount",
                  "account_balance_before", "account_balance_after"]:
        val = row.get(field)
        if val is not None:
            try:
                txn[field] = float(val)
            except (ValueError, TypeError):
                pass

    for field in ["failed_attempt_count", "login_attempts", "refund_flag",
                  "chargeback_flag", "reversal_flag", "monthly_customer_transaction_count"]:
        val = row.get(field)
        if val is not None:
            try:
                txn[field] = int(float(val))
            except (ValueError, TypeError):
                pass

    return txn


_TEMPLATE_CSV = (
    "transaction_id,customer_id,transaction_amount,transaction_date,transaction_time,"
    "channel,merchant_category,merchant_name,customer_segment,debit_credit_flag,"
    "device_id,ip_location,risk_score,historical_average_transaction_amount,"
    "failed_attempt_count,login_attempts,account_balance_before,account_balance_after,"
    "refund_flag,chargeback_flag\n"
    "TXN-SAMPLE-001,CUST-0001,12500.00,2024-07-15,02:47:00,"
    "API/Open Banking,Crypto Exchanges,CryptoFX Ltd,Mass Market,Debit,"
    "DEV-X9921-UNKNOWN,203.0.113.45,72,450.0,4,6,13000.00,500.00,0,0\n"
    "TXN-SAMPLE-002,CUST-0002,42.50,2024-07-15,14:30:00,"
    "Mobile Banking,Groceries,Tesco,Premium,Debit,"
    "DEV-IPHONE-001,192.168.1.1,12,55.0,0,1,3200.00,3157.50,0,0\n"
)


@router.get("/upload/template",
            summary="Download CSV template for bulk upload")
def download_upload_template():
    """
    Returns a ready-to-fill CSV template with all supported columns and two
    example rows (one HIGH-risk, one LOW-risk). Fill it with your transactions
    and POST to /score/upload.
    """
    return StreamingResponse(
        io.StringIO(_TEMPLATE_CSV),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=bti_upload_template.csv"},
    )


@router.post("/upload",
             summary="Batch score transactions from a CSV or Excel file")
async def batch_score_upload(
    file: UploadFile = File(..., description="CSV (.csv) or Excel (.xlsx / .xls) file"),
    db: Session = Depends(get_db),
):
    """
    Upload a CSV or Excel file of transactions. Every row is scored independently
    through the full 19-rule fraud engine and ML ensemble (Isolation Forest +
    Logistic Regression + Random Forest).

    **Required columns:** transaction_id, customer_id, transaction_amount

    All other columns are optional — the system applies sensible defaults for
    anything missing. Download the template from **GET /score/upload/template**
    to see the full column list with two example rows.

    Returns a batch summary (tier breakdown) plus the full fraud risk assessment
    for every row.
    """
    if not models_available():
        raise HTTPException(
            status_code=503,
            detail="ML models not trained. Run POST /pipeline/run/sync first.",
        )

    filename = file.filename or "upload"
    content  = await file.read()

    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no rows.")

    df = _normalise_df(df)

    missing_required = [c for c in ("transaction_id", "customer_id", "transaction_amount")
                        if c not in df.columns]
    if missing_required:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Missing required columns: {missing_required}. "
                "Download the template at GET /score/upload/template"
            ),
        )

    results, errors, tier_counts = [], [], Counter()

    for idx, row in df.iterrows():
        row_num = int(idx) + 2  # 1-indexed, header = row 1
        try:
            txn = _row_to_txn(row.to_dict(), int(idx))
            if txn["transaction_amount"] <= 0:
                errors.append({"row": row_num, "error": "transaction_amount must be > 0"})
                continue

            result = _scorer.score(txn, db_session=db)
            tier_counts[result.final_alert_tier] += 1

            results.append({
                "row":                 row_num,
                "transaction_id":      result.transaction_id,
                "customer_id":         txn.get("customer_id"),
                "transaction_amount":  txn["transaction_amount"],
                "final_risk_score":    round(result.final_risk_score, 2),
                "final_alert_tier":    result.final_alert_tier,
                "fraud_rule_score":    round(result.fraud_rule_score, 2),
                "rules_triggered":     result.rules_triggered,
                "rules_fired":         result.rules_fired,
                "ml_lr_proba":         round(result.ml_lr_proba, 4),
                "ml_rf_proba":         round(result.ml_rf_proba, 4),
                "ml_anomaly_score":    round(result.ml_anomaly_score, 2),
                "is_suspicious":       result.is_suspicious,
                "recommendation":      _recommendation(result),
                "processing_time_ms":  round(result.processing_time_ms, 1),
            })
        except Exception as exc:
            errors.append({"row": row_num, "error": str(exc)})

    log.info("Batch upload scored",
             extra={"file": filename, "rows": len(df),
                    "scored": len(results), "errors": len(errors)})

    return {
        "file_name":  filename,
        "total_rows": len(df),
        "scored":     len(results),
        "failed":     len(errors),
        "summary": {
            "CRITICAL":  tier_counts.get("CRITICAL",  0),
            "VERY HIGH": tier_counts.get("VERY HIGH", 0),
            "HIGH":      tier_counts.get("HIGH",      0),
            "MEDIUM":    tier_counts.get("MEDIUM",    0),
            "LOW":       tier_counts.get("LOW",       0),
        },
        "results": results,
        "errors":  errors or None,
    }
