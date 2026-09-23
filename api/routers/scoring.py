"""
/api/v1/score — real-time single-transaction fraud scoring.

Scored by the governed v3 model and the expected-cost decision engine (the
same path as /api/v1/v3/score), returned in the original response shape.
The legacy ML fields ml_lr_proba, ml_rf_proba and ml_iso_score are null
since the migration: the legacy ensemble read label-derived inputs.
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
from bti.modeling import registry
from bti.modeling.scorer import scorer as _v3_scorer
from bti.operations.scoring_service import model_available
from bti.scoring.realtime import RealTimeScorer, ScoreResult
from bti.scoring.v3_adapter import recommendation, top_drivers
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
    country:            Optional[str] = Field(default=None, description="Account country or ISO-2 code; selects "
                                                                        "the jurisdiction policy")
    account_balance_before: Optional[float] = None
    account_balance_after:  Optional[float] = None
    risk_score:             Optional[float] = Field(default=None, ge=0, le=100,
                                                    description="Ignored — label-derived in the training data; "
                                                                "accepted for backward compatibility")
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

    # Accepted for backward compatibility and ignored: known only after settlement or dispute
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
            "country": "GB",
            "currency": "GBP",
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
    ml_lr_proba:        Optional[float] = Field(None, description="Deprecated — always null since the v3 migration")
    ml_rf_proba:        Optional[float] = Field(None, description="Deprecated — always null since the v3 migration")
    ml_iso_score:       Optional[float] = Field(None, description="Deprecated — always null since the v3 migration")
    ml_anomaly_score:   float = Field(..., description="v3 fraud probability × 100")
    final_risk_score:   float = Field(..., description="v3 calibrated fraud probability × 100")
    final_alert_tier:   str   = Field(..., description="LOW / MEDIUM / HIGH / VERY HIGH / CRITICAL")
    is_suspicious:      bool  = Field(..., description="True when the decision is anything other than APPROVE")
    processing_time_ms: float
    model_version:      str   = Field(..., description="Registered v3 model id")
    db_context_used:    bool
    recommendation:     str
    fraud_probability:  float = Field(..., description="Calibrated probability of fraud, 0–1")
    decision:           str   = Field(..., description="APPROVE / STEP_UP / REVIEW / DECLINE")
    guardrails:         List[str]
    reason_codes:       List[dict]
    model_provisional:  bool  = Field(..., description="True until a champion is approved; blocks auto-decline")
    jurisdiction:       Optional[str] = None
    notes:              List[str]


def _recommendation(result: ScoreResult) -> str:
    return recommendation(result.decision, result.fraud_probability)


def _require_model() -> None:
    if not model_available():
        raise HTTPException(status_code=503,
                            detail="No v3 model is registered. Run: python -m bti.modeling.train")


@router.post("/", response_model=TransactionScoreResponse,
             summary="Score a single transaction in real-time")
def score_transaction_endpoint(
    body: TransactionScoreRequest,
    db: Session = Depends(get_db),
):
    """
    Submit a transaction and receive a fraud risk assessment.

    The response includes:
    - Calibrated fraud probability from the governed v3 model
    - Expected-cost decision (APPROVE / STEP_UP / REVIEW / DECLINE) and guardrails
    - Reason codes with analyst and customer wording
    - Alert tier (LOW → CRITICAL) and recommendation, as before
    - The 19-rule engine output, reported for analysts (not part of the score)
    """
    _require_model()

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
        fraud_probability=result.fraud_probability,
        decision=result.decision,
        guardrails=result.guardrails,
        reason_codes=result.reason_codes,
        model_provisional=result.provisional,
        jurisdiction=result.jurisdiction,
        notes=result.notes,
    )


@router.post("/explain",
             summary="Score + SHAP explanation — why was this transaction flagged?")
def explain_transaction_endpoint(
    body: TransactionScoreRequest,
    db: Session = Depends(get_db),
):
    """
    Score a transaction and explain it with exact SHAP contributions from the
    v3 model.

    - **top_drivers**: the eight features that moved the probability most, up
      or down, with the feature value, SHAP value (log-odds) and share of impact.
    - **reason_codes**: up to four principal reasons with analyst and
      customer-facing wording.
    - **narrative**: a plain-English summary for a case note.
    """
    _require_model()
    result = _scorer.score(body.model_dump(), db_session=db)
    drivers = top_drivers(result.contributions, result.feature_values)
    reasons = "; ".join(r["analyst_text"].lower() for r in result.reason_codes[:3]) or "no risk-raising factors"
    narrative = (f"BTI rates this transaction {result.final_alert_tier} with a {result.fraud_probability:.1%} "
                 f"probability of fraud. Main drivers: {reasons}. Decision: {result.decision}."
                 + (" The model is provisional until a champion is approved." if result.provisional else ""))
    return {
        "transaction_id":    result.transaction_id,
        "scored_at":         datetime.utcnow().isoformat() + "Z",
        "final_risk_score":  result.final_risk_score,
        "final_alert_tier":  result.final_alert_tier,
        "fraud_probability": result.fraud_probability,
        "decision":          result.decision,
        "guardrails":        result.guardrails,
        "fraud_rule_score":  result.fraud_rule_score,
        "rules_triggered":   result.rules_triggered,
        "rules_fired":       result.rules_fired,
        "is_suspicious":     result.is_suspicious,
        "recommendation":    _recommendation(result),
        "model_version":     result.model_version,
        "model_provisional": result.provisional,
        "processing_time_ms": result.processing_time_ms,
        "explanation": {
            "fraud_probability": result.fraud_probability,
            "base_probability":  result.base_probability,
            "narrative":         narrative,
            "reason_codes":      result.reason_codes,
            "top_drivers":       drivers,
        },
    }


@router.get("/model-info", summary="Model version and performance metrics")
def get_model_info():
    """The model answering /score, with its out-of-time validation figures."""
    _require_model()
    model_id, role, provisional = _v3_scorer.resolve("champion")
    card = registry.load_card(model_id)
    return {
        "model_id": model_id,
        "role": role,
        "provisional": provisional,
        "trained_at": card["created_at"],
        "validation_status": card["validation"]["status"],
        "feature_count": len(card["features"]["model_features"]),
        "out_of_time": card["performance"]["metrics"]["out_of_time"],
        "documentation": f"/api/v1/governance/models/{model_id}/documentation",
    }


@router.post("/reload-models",
             summary="Reload the registered model (call after a promotion or retrain)")
def reload_models():
    _scorer.reload()
    model_id, role, provisional = _v3_scorer.resolve("champion")
    return {"status": "reloaded", "model_id": model_id, "role": role, "provisional": provisional}


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
                  "debit_credit_flag", "device_id", "ip_location", "currency", "country",
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
    "device_id,ip_location,currency,country,historical_average_transaction_amount,"
    "failed_attempt_count,login_attempts,account_balance_before,account_balance_after\n"
    "TXN-SAMPLE-001,CUST-0001,12500.00,2024-07-15,02:47:00,"
    "API/Open Banking,Crypto Exchanges,CryptoFX Ltd,Mass Market,Debit,"
    "DEV-X9921-UNKNOWN,203.0.113.45,GBP,GB,450.0,4,6,13000.00,500.00\n"
    "TXN-SAMPLE-002,CUST-0002,42.50,2024-07-15,14:30:00,"
    "Mobile Banking,Groceries,Tesco,Premium,Debit,"
    "DEV-IPHONE-001,192.168.1.1,USD,US,55.0,0,1,3200.00,3157.50\n"
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
    Upload a CSV or Excel file of transactions. Every row is scored by the v3
    model and decision engine against its own database history.

    **Required columns:** transaction_id, customer_id, transaction_amount

    All other columns are optional — the system applies sensible defaults for
    anything missing. Download the template from **GET /score/upload/template**
    to see the full column list with two example rows.

    Returns a batch summary (tier breakdown) plus the full fraud risk assessment
    for every row.
    """
    _require_model()

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
                "fraud_probability":   result.fraud_probability,
                "decision":            result.decision,
                "reason_codes":        [r["code"] for r in result.reason_codes],
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
