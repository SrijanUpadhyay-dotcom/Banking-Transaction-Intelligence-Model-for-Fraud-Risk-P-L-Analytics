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
    # Optional feeds: transaction location and beneficiary key (null when the source system lacks them)
    latitude           = Column(Float)
    longitude          = Column(Float)
    payee_id           = Column(String(100), index=True)
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
    # Tamper evidence (bti.governance.audit_chain): each row hashes the previous row's hash.
    seq            = Column(Integer, index=True)
    prev_hash      = Column(String(64))
    row_hash       = Column(String(64))


class AuditChainHead(Base):
    """Single row holding the latest sequence number and hash of the audit chain."""
    __tablename__ = "audit_chain_head"

    id         = Column(Integer, primary_key=True)
    seq        = Column(Integer, nullable=False, default=0)
    head_hash  = Column(String(64), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ScoreLog(Base):
    """
    Every v3 score, champion and shadow challenger alike. The source for live
    drift monitoring, champion/challenger comparison, operational KPIs and the
    decision audit trail.
    """
    __tablename__ = "score_log"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id    = Column(String(50), nullable=False, index=True)
    customer_id       = Column(String(50), index=True)
    model_id          = Column(String(64), nullable=False, index=True)
    model_role        = Column(String(20), nullable=False)
    is_shadow         = Column(Boolean, default=False, index=True)
    fraud_probability = Column(Float, nullable=False)
    score             = Column(Integer)
    decision          = Column(String(20), index=True)
    jurisdiction      = Column(String(2), index=True)
    amount_usd        = Column(Float)
    reason_codes      = Column(JSON)
    features          = Column(JSON)
    guardrails        = Column(JSON)
    latency_ms        = Column(Float)
    model_probability = Column(Float)        # before any recalibration overlay (bti.modeling.recalibration)
    calibration_overlay = Column(Integer)     # overlay version applied, if any
    # Segment, age band and country for fairness monitoring on matured outcomes — never model inputs.
    monitoring_attributes = Column(JSON)
    scored_at         = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (Index("ix_score_log_model_time", "model_id", "scored_at"),)


class FraudLabel(Base):
    """
    Confirmed outcomes fed back from disputes, chargebacks and investigations.
    Append-only: the latest label per transaction wins, earlier ones stay for audit.
    """
    __tablename__ = "fraud_labels"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id   = Column(String(50), nullable=False, index=True)
    label            = Column(Integer, nullable=False)
    label_source     = Column(String(40), nullable=False)
    fraud_type       = Column(String(100))
    event_at         = Column(DateTime, nullable=False, index=True)
    loss_amount      = Column(Float)
    recovered_amount = Column(Float)
    currency         = Column(String(5))
    reported_by      = Column(String(100))
    notes            = Column(Text)
    created_at       = Column(DateTime, default=datetime.utcnow, nullable=False)


class SecurityEvent(Base):
    """
    The bank's security-event log: password resets, SIM swaps / number ports,
    contact-detail changes, device enrolments. Append-only; read point-in-time
    by the scoring model (only events logged before a transaction count).
    """
    __tablename__ = "security_events"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), nullable=False, index=True)
    event_type  = Column(String(40), nullable=False)
    event_time  = Column(DateTime, nullable=False, index=True)
    source      = Column(String(60))
    detail      = Column(JSON)
    received_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_security_event_customer_time", "customer_id", "event_time"),)


class IncumbentDecision(Base):
    """
    Scores and decisions from the incumbent fraud platform (e.g. SAS) for the same
    transactions BTI sees. Append-only; the latest record per transaction wins.
    `decision` is normalised to APPROVE / STEP_UP / REVIEW / DECLINE; the vendor's
    own code is kept in `raw_decision`. `executed_decision` is what the bank's
    switch actually did, when the feed carries it (used by reconciliation).
    """
    __tablename__ = "incumbent_decisions"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id    = Column(String(50), nullable=False, index=True)
    system            = Column(String(30), nullable=False, default="SAS")
    customer_id       = Column(String(50), index=True)
    score             = Column(Float)
    decision          = Column(String(20), nullable=False, index=True)
    raw_decision      = Column(String(60))
    executed_decision = Column(String(20))
    decided_at        = Column(DateTime, nullable=False, index=True)
    latency_ms        = Column(Float)
    amount            = Column(Float)
    currency          = Column(String(5))
    rule_ids          = Column(JSON)
    batch_id          = Column(String(40), index=True)
    received_at       = Column(DateTime, default=datetime.utcnow, nullable=False)


class TrafficExperiment(Base):
    """
    Randomised split of live decisions between the incumbent and BTI. Proposed by
    one person, started by a different approver (four-eyes), capped in share.
    """
    __tablename__ = "traffic_experiments"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(80), nullable=False, unique=True)
    bti_share   = Column(Float, nullable=False)
    unit        = Column(String(20), nullable=False, default="customer")
    salt        = Column(String(40), nullable=False)
    status      = Column(String(20), nullable=False, default="proposed", index=True)
    model_id    = Column(String(64))
    proposed_by = Column(String(100), nullable=False)
    rationale   = Column(Text)
    approved_by = Column(String(100))
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at  = Column(DateTime)
    stopped_at  = Column(DateTime)
    stopped_by  = Column(String(100))
    stop_reason = Column(Text)


class RoutedDecision(Base):
    """Every decision that passed through the parallel-run router, with both systems' answers."""
    __tablename__ = "routed_decisions"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id      = Column(Integer, index=True)
    transaction_id     = Column(String(50), nullable=False, index=True)
    customer_id        = Column(String(50), index=True)
    arm                = Column(String(10), nullable=False, index=True)     # bti | control | none
    bti_model_id       = Column(String(64))
    bti_probability    = Column(Float)
    bti_decision       = Column(String(20))
    bti_latency_ms     = Column(Float)
    incumbent_decision = Column(String(20))
    effective_decision = Column(String(20))
    decided_by         = Column(String(20), nullable=False)                 # BTI | INCUMBENT | NONE
    fallback_reason    = Column(String(40))
    amount_usd         = Column(Float)
    routed_at          = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class ModelInventoryEntry(Base):
    """Model inventory (SR 11-7): one row per registered model, synced from the registry."""
    __tablename__ = "model_inventory"

    model_id          = Column(String(64), primary_key=True)
    model_family      = Column(String(100))
    algorithm         = Column(String(30))
    feature_set       = Column(String(40))
    developer         = Column(String(100))
    business_owner    = Column(String(100))
    risk_tier         = Column(String(10), nullable=False, default="Tier 1")
    role              = Column(String(20))                   # champion | challenger | none
    lifecycle         = Column(String(30), nullable=False)   # in_use | shadow_challenger | registered | superseded
    validation_status = Column(String(20))                   # automated gates: passed | failed
    last_signoff      = Column(String(40))
    last_validator    = Column(String(100))
    last_validated_at = Column(DateTime)
    next_review_due   = Column(DateTime)
    registered_at     = Column(DateTime)
    updated_at        = Column(DateTime, default=datetime.utcnow, nullable=False)


class ValidationFinding(Base):
    """Findings tracker: issues raised by validation, development, monitoring or audit, to closure."""
    __tablename__ = "validation_findings"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    model_id     = Column(String(64), nullable=False, index=True)
    title        = Column(String(200), nullable=False)
    severity     = Column(String(10), nullable=False, index=True)    # high | medium | low
    category     = Column(String(30), nullable=False)
    source       = Column(String(30), nullable=False)
    description  = Column(Text, nullable=False)
    raised_by    = Column(String(100), nullable=False)
    raised_at    = Column(DateTime, default=datetime.utcnow, nullable=False)
    owner        = Column(String(100), nullable=False)
    due_date     = Column(DateTime, nullable=False)
    status       = Column(String(20), nullable=False, default="open", index=True)
    resolution   = Column(Text)
    evidence     = Column(JSON)
    closed_by    = Column(String(100))
    closed_at    = Column(DateTime)
    accepted_by  = Column(String(100))
    accepted_until = Column(DateTime)
    updated_at   = Column(DateTime, default=datetime.utcnow, nullable=False)


class ValidationSignoff(Base):
    """Independent validation sign-off. Append-only; the latest in-date sign-off governs promotion."""
    __tablename__ = "validation_signoffs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    model_id       = Column(String(64), nullable=False, index=True)
    validator      = Column(String(100), nullable=False)
    validator_role = Column(String(100))
    decision       = Column(String(30), nullable=False)          # approve | approve_with_conditions | reject
    scope          = Column(Text, nullable=False)
    conditions     = Column(Text)
    evidence       = Column(JSON)
    signed_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    valid_until    = Column(DateTime, nullable=False)


class FraudCase(Base):
    """
    Investigation case for a live decision that needs a human (REVIEW), or a
    manual referral. Queued with an SLA; the analyst's disposition becomes a
    confirmed label automatically.
    """
    __tablename__ = "fraud_cases"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id    = Column(String(50), nullable=False, index=True)
    customer_id       = Column(String(50), index=True)
    queue             = Column(String(30), nullable=False, index=True)
    priority_score    = Column(Float, nullable=False, default=0.0)     # expected loss, USD
    source            = Column(String(40), nullable=False)
    model_id          = Column(String(64))
    fraud_probability = Column(Float)
    amount_usd        = Column(Float)
    decision          = Column(String(20))
    reason_codes      = Column(JSON)
    status            = Column(String(20), nullable=False, default="open", index=True)
    assigned_to       = Column(String(100), index=True)
    created_at        = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    sla_due_at        = Column(DateTime, nullable=False, index=True)
    assigned_at       = Column(DateTime)
    closed_at         = Column(DateTime)
    disposition       = Column(String(30))
    disposition_by    = Column(String(100))
    checked_by        = Column(String(100))
    disposition_notes = Column(Text)
    fraud_type        = Column(String(100))
    loss_amount       = Column(Float)
    label_written     = Column(Boolean, default=False)
    breach_alerted_at = Column(DateTime)


class RuleVersion(Base):
    """One immutable version of an analyst rule. Editing a rule creates a new version."""
    __tablename__ = "rule_versions"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    rule_id      = Column(String(40), nullable=False, index=True)
    version      = Column(Integer, nullable=False)
    name         = Column(String(120), nullable=False)
    description  = Column(Text, nullable=False)
    condition    = Column(JSON, nullable=False)
    action       = Column(String(20), nullable=False)          # STEP_UP | REVIEW | DECLINE (a floor)
    author       = Column(String(100), nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
    status       = Column(String(20), nullable=False, default="draft", index=True)  # draft|simulated|active|shadow|retired
    simulation   = Column(JSON)
    simulated_at = Column(DateTime)
    approved_by  = Column(String(100))
    approved_at  = Column(DateTime)
    expires_at   = Column(DateTime)
    retired_by   = Column(String(100))
    retired_at   = Column(DateTime)
    retire_reason = Column(Text)

    __table_args__ = (Index("ux_rule_version", "rule_id", "version", unique=True),)


class RuleHit(Base):
    """Every live rule match, enforced (active) or logged only (shadow)."""
    __tablename__ = "rule_hits"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id  = Column(String(50), nullable=False, index=True)
    rule_version_id = Column(Integer, nullable=False, index=True)
    rule_id         = Column(String(40), nullable=False, index=True)
    version         = Column(Integer, nullable=False)
    mode            = Column(String(10), nullable=False)
    action          = Column(String(20), nullable=False)
    decision_before = Column(String(20))
    decision_after  = Column(String(20))
    enforced        = Column(Boolean, nullable=False, default=False)
    at              = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class StepUpChallenge(Base):
    """A customer challenge (SMS one-time code, push approval, 3-D Secure) issued for a STEP_UP decision."""
    __tablename__ = "stepup_challenges"

    id             = Column(String(32), primary_key=True)             # unguessable token
    transaction_id = Column(String(50), nullable=False, index=True)
    customer_id    = Column(String(50), index=True)
    method         = Column(String(20), nullable=False, index=True)    # sms_otp | push | 3ds
    channel        = Column(String(50))
    amount_usd     = Column(Float)
    status         = Column(String(20), nullable=False, default="pending", index=True)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    expires_at     = Column(DateTime, nullable=False)
    attempts       = Column(Integer, nullable=False, default=0)
    max_attempts   = Column(Integer, nullable=False, default=3)
    secret_hash    = Column(String(64))
    salt           = Column(String(32))
    provider       = Column(String(20))
    provider_ref   = Column(String(100))
    completed_at   = Column(DateTime)
    outcome_detail = Column(JSON)


class CustomerValue(Base):
    """Annual net revenue per customer from the bank's P&L data (cost model v2). Refreshed daily."""
    __tablename__ = "customer_values"

    customer_id      = Column(String(50), primary_key=True)
    annual_value_usd = Column(Float, nullable=False)
    months_observed  = Column(Float, nullable=False)
    transactions     = Column(Integer, nullable=False)
    computed_at      = Column(DateTime, default=datetime.utcnow, nullable=False)


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


# Hash-chain every audit row on insert (the listener lives with the rest of the audit controls).
import bti.governance.audit_chain  # noqa: E402,F401
