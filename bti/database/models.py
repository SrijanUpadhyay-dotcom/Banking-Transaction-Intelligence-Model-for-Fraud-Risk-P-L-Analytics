"""
SQLAlchemy ORM models. Single source of truth for the database schema.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, Index,
    ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    pass


class AlertTierEnum(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY HIGH"
    CRITICAL = "CRITICAL"


class Transaction(Base):
    """
    Core fact table — mirrors the 42-field synthetic dataset plus all
    computed fraud/risk/ML columns appended by the pipeline.
    """
    __tablename__ = "transactions"

    # ── Raw fields ────────────────────────────────────────────────────────────
    transaction_id     = Column(String(50), primary_key=True, index=True)
    customer_id        = Column(String(50), nullable=False, index=True)
    account_id         = Column(String(50), nullable=False, index=True)
    transaction_date   = Column(DateTime, nullable=False, index=True)
    transaction_time   = Column(String(8))
    transaction_amount = Column(Float, nullable=False)
    transaction_type   = Column(String(50))
    debit_credit_flag  = Column(String(10))
    channel            = Column(String(50), index=True)
    branch_or_digital  = Column(String(20))
    merchant_category  = Column(String(100), index=True)
    merchant_name      = Column(String(200))
    customer_segment   = Column(String(50), index=True)
    customer_age_band  = Column(String(20))
    geography          = Column(String(50))
    country            = Column(String(50))
    city               = Column(String(100))
    currency           = Column(String(5))
    account_balance_before = Column(Float)
    account_balance_after  = Column(Float)
    transaction_status = Column(String(30))
    failed_attempt_count = Column(Integer, default=0)
    reversal_flag      = Column(Integer, default=0)
    refund_flag        = Column(Integer, default=0)
    chargeback_flag    = Column(Integer, default=0)
    fraud_flag         = Column(Integer, default=0, index=True)
    fraud_type         = Column(String(100))
    risk_score         = Column(Float)
    authorization_method = Column(String(50))
    device_id          = Column(String(100))
    ip_location        = Column(String(50))
    login_attempts     = Column(Integer, default=1)
    historical_average_transaction_amount = Column(Float)
    monthly_customer_transaction_count    = Column(Integer)
    fee_income         = Column(Float, default=0)
    interchange_income = Column(Float, default=0)
    processing_cost    = Column(Float, default=0)
    chargeback_loss    = Column(Float, default=0)
    refund_loss        = Column(Float, default=0)
    fraud_loss         = Column(Float, default=0)
    net_revenue        = Column(Float, default=0)
    net_pnl_impact     = Column(Float, default=0)

    # ── Enrichment (from pipeline) ────────────────────────────────────────────
    transaction_year    = Column(Integer)
    transaction_month   = Column(Integer)
    transaction_quarter = Column(Integer)
    day_of_week         = Column(String(10))
    is_weekend          = Column(Boolean, default=False)
    is_off_hours        = Column(Integer, default=0)
    amount_band         = Column(String(30))
    amount_vs_hist_avg_ratio = Column(Float)
    is_high_risk        = Column(Integer, default=0, index=True)
    is_revenue_leakage  = Column(Integer, default=0)
    month_year          = Column(String(8))

    # ── Rule engine output ────────────────────────────────────────────────────
    fraud_rule_score = Column(Float)
    rules_triggered  = Column(Integer)
    is_suspicious    = Column(Integer, default=0, index=True)
    alert_tier       = Column(String(20))

    # ── ML output ─────────────────────────────────────────────────────────────
    iso_forest_flag  = Column(Integer)
    iso_forest_score = Column(Float)
    z_score_flag     = Column(Integer)
    lr_fraud_proba   = Column(Float)
    rf_fraud_proba   = Column(Float)
    ml_anomaly_score = Column(Float)
    ml_anomaly_score_norm = Column(Float)
    ml_fraud_prediction   = Column(Integer, index=True)
    final_risk_score      = Column(Float, index=True)
    final_alert_tier      = Column(String(20), index=True)

    # ── Metadata ──────────────────────────────────────────────────────────────
    pipeline_run_id = Column(String(50))
    ingested_at     = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_txn_date_segment", "transaction_date", "customer_segment"),
        Index("ix_txn_fraud_risk", "fraud_flag", "final_risk_score"),
    )


class FraudAlert(Base):
    """Persisted alerts for the exception queue — used by analysts."""
    __tablename__ = "fraud_alerts"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id  = Column(String(50), ForeignKey("transactions.transaction_id"), index=True)
    alert_tier      = Column(String(20), nullable=False, index=True)
    final_risk_score = Column(Float, nullable=False)
    fraud_rule_score = Column(Float)
    ml_anomaly_score_norm = Column(Float)
    rules_triggered = Column(Integer)
    status          = Column(String(20), default="OPEN", index=True)  # OPEN / REVIEWING / CLOSED / ESCALATED
    assigned_to     = Column(String(100))
    resolution      = Column(Text)
    notes           = Column(Text)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at     = Column(DateTime)

    transaction = relationship("Transaction", foreign_keys=[transaction_id])


class PnLSummary(Base):
    """Materialised P&L aggregation by month — avoids re-scanning the full table."""
    __tablename__ = "pnl_summary"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    period_month      = Column(String(8), nullable=False, index=True)  # "2024-01"
    total_transactions = Column(Integer)
    total_volume       = Column(Float)
    total_fee_income   = Column(Float)
    total_interchange  = Column(Float)
    total_processing_cost = Column(Float)
    total_chargeback_loss = Column(Float)
    total_refund_loss  = Column(Float)
    total_fraud_loss   = Column(Float)
    net_revenue        = Column(Float)
    net_pnl_impact     = Column(Float)
    fraud_count        = Column(Integer)
    fraud_loss_rate    = Column(Float)
    chargeback_ratio   = Column(Float)
    cost_to_income     = Column(Float)
    risk_adj_revenue   = Column(Float)
    mom_pnl_variance_pct = Column(Float)
    computed_at        = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """
    Immutable audit trail — every system decision about a transaction is logged.
    Required for AML/BSA/FCA/OCC compliance.
    """
    __tablename__ = "audit_logs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    ts             = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_type     = Column(String(50), nullable=False, index=True)
    transaction_id = Column(String(50), index=True)
    analyst_id     = Column(String(100))
    pipeline_run_id = Column(String(50))
    payload        = Column(JSON)
    source_ip      = Column(String(50))
    user_agent     = Column(String(200))


class ModelRegistry(Base):
    """Versioned ML model metadata — tracks which model version scored a transaction."""
    __tablename__ = "model_registry"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    model_name    = Column(String(100), nullable=False)
    version       = Column(String(20), nullable=False)
    trained_at    = Column(DateTime, nullable=False)
    roc_auc       = Column(Float)
    f1_score      = Column(Float)
    precision_score = Column(Float)
    recall_score  = Column(Float)
    training_rows = Column(Integer)
    feature_list  = Column(JSON)
    artifact_path = Column(String(500))
    is_active     = Column(Boolean, default=False, index=True)
    notes         = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow)
