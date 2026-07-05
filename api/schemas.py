"""
Pydantic request/response schemas for the BTI REST API.
Separate from the ORM models to allow independent versioning.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ── Transaction ────────────────────────────────────────────────────────────────

class TransactionBase(BaseModel):
    transaction_id: str
    customer_id: str
    account_id: str
    transaction_date: datetime
    transaction_amount: float
    transaction_type: Optional[str] = None
    debit_credit_flag: Optional[str] = None
    channel: Optional[str] = None
    merchant_category: Optional[str] = None
    merchant_name: Optional[str] = None
    customer_segment: Optional[str] = None
    geography: Optional[str] = None
    currency: Optional[str] = "USD"
    risk_score: Optional[float] = None
    fraud_flag: Optional[int] = 0


class TransactionScored(TransactionBase):
    fraud_rule_score: Optional[float] = None
    rules_triggered: Optional[int] = None
    is_suspicious: Optional[int] = None
    alert_tier: Optional[str] = None
    final_risk_score: Optional[float] = None
    final_alert_tier: Optional[str] = None
    ml_fraud_prediction: Optional[int] = None
    ml_anomaly_score_norm: Optional[float] = None
    lr_fraud_proba: Optional[float] = None
    rf_fraud_proba: Optional[float] = None
    net_pnl_impact: Optional[float] = None

    model_config = {"from_attributes": True}


class TransactionPage(BaseModel):
    items: List[TransactionScored]
    total: int
    page: int
    page_size: int
    pages: int


# ── Fraud Alerts ───────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    id: int
    transaction_id: str
    alert_tier: str
    final_risk_score: float
    fraud_rule_score: Optional[float] = None
    ml_anomaly_score_norm: Optional[float] = None
    rules_triggered: Optional[int] = None
    status: str
    assigned_to: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertUpdateRequest(BaseModel):
    status: Optional[str] = None  # OPEN / REVIEWING / CLOSED / ESCALATED
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None
    notes: Optional[str] = None


# ── P&L ───────────────────────────────────────────────────────────────────────

class PnLPeriod(BaseModel):
    period_month: str
    total_transactions: Optional[int] = None
    total_volume: Optional[float] = None
    net_revenue: Optional[float] = None
    net_pnl_impact: Optional[float] = None
    fraud_count: Optional[int] = None
    fraud_loss_rate: Optional[float] = None
    chargeback_ratio: Optional[float] = None
    cost_to_income: Optional[float] = None
    mom_pnl_variance_pct: Optional[float] = None

    model_config = {"from_attributes": True}


class PnLKPIs(BaseModel):
    total_transactions: int
    total_volume: float
    net_revenue: float
    net_pnl_impact: float
    fraud_count: int
    fraud_loss_rate: float
    chargeback_ratio: float
    cost_to_income: float
    risk_adj_revenue: float
    revenue_leakage_count: int


# ── Risk Score ─────────────────────────────────────────────────────────────────

class CustomerRiskProfile(BaseModel):
    customer_id: str
    avg_final_risk_score: float
    max_final_risk_score: float
    total_transactions: int
    fraud_count: int
    suspicious_count: int
    total_volume: float
    avg_amount: float
    primary_segment: Optional[str] = None
    primary_channel: Optional[str] = None
    chargeback_count: int


# ── Pipeline ───────────────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    force: bool = Field(default=False,
                        description="If true, re-runs all stages even if outputs exist")


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    stages: dict
    elapsed_seconds: Optional[float] = None
    db_rows_inserted: Optional[int] = None
    error: Optional[str] = None


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    database: str
    uptime_seconds: float
