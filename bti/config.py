# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Centralised configuration — reads settings.yaml then overrides with env vars.
All application code imports from here; never use hardcoded paths or values.
"""

import os
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config" / "settings.yaml"


def _load_yaml() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml()


class Settings(BaseSettings):
    # App
    app_name: str = _yaml.get("app", {}).get("name", "BTI Model")
    app_version: str = _yaml.get("app", {}).get("version", "2.0.0")
    environment: str = Field(default=_yaml.get("app", {}).get("environment", "production"),
                              alias="BTI_ENVIRONMENT")
    log_level: str = Field(default=_yaml.get("app", {}).get("log_level", "INFO"),
                            alias="BTI_LOG_LEVEL")

    # Database
    database_url: str = Field(
        default=_yaml.get("database", {}).get("url", f"sqlite:///{BASE_DIR}/data/bti.db"),
        alias="BTI_DATABASE_URL"
    )
    db_pool_size: int = _yaml.get("database", {}).get("pool_size", 10)
    db_echo: bool = _yaml.get("database", {}).get("echo", False)

    # Security
    secret_key: str = Field(default="CHANGE_ME_IN_PRODUCTION", alias="BTI_SECRET_KEY")
    api_key: str = Field(default="CHANGE_ME_API_KEY", alias="BTI_API_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Pipeline
    raw_data_path: str = str(BASE_DIR / _yaml.get("pipeline", {}).get("raw_data_path",
                                                                        "data/raw/banking_transactions_raw.csv"))
    processed_data_dir: str = str(BASE_DIR / "data" / "processed")
    outputs_dir: str = str(BASE_DIR / "outputs")
    models_dir: str = str(BASE_DIR / "models")
    synthetic_rows: int = _yaml.get("pipeline", {}).get("synthetic_rows", 50000)
    fraud_rate: float = _yaml.get("pipeline", {}).get("fraud_rate", 0.05)

    # Fraud rules
    velocity_spike_threshold: int = _yaml.get("fraud_rules", {}).get("velocity_spike_threshold", 10)
    failed_auth_threshold: int = _yaml.get("fraud_rules", {}).get("failed_auth_threshold", 3)
    login_attempts_threshold: int = _yaml.get("fraud_rules", {}).get("login_attempts_threshold", 4)
    off_hours_end: int = _yaml.get("fraud_rules", {}).get("off_hours_end", 6)
    high_value_ratio: float = _yaml.get("fraud_rules", {}).get("high_value_ratio", 5.0)
    high_risk_score: int = _yaml.get("fraud_rules", {}).get("high_risk_score", 60)
    amount_outlier_zscore: float = _yaml.get("fraud_rules", {}).get("amount_outlier_zscore", 3.5)
    refund_ratio_threshold: float = _yaml.get("fraud_rules", {}).get("refund_ratio_threshold", 0.30)
    chargeback_ratio_threshold: float = _yaml.get("fraud_rules", {}).get("chargeback_ratio_threshold", 0.10)
    fraud_prob_threshold: float = _yaml.get("ml_models", {}).get("fraud_prob_threshold", 0.40)

    # Alerts
    alerts_enabled: bool = Field(default=_yaml.get("alerts", {}).get("enabled", True),
                                  alias="BTI_ALERTS_ENABLED")
    alert_critical_threshold: float = _yaml.get("alerts", {}).get("critical_threshold", 85)
    alert_high_threshold: float = _yaml.get("alerts", {}).get("high_threshold", 70)
    smtp_host: str = Field(default=_yaml.get("alerts", {}).get("email", {}).get("smtp_host", ""),
                            alias="BTI_SMTP_HOST")
    smtp_port: int = Field(default=_yaml.get("alerts", {}).get("email", {}).get("smtp_port", 587),
                            alias="BTI_SMTP_PORT")
    smtp_user: str = Field(default="", alias="BTI_SMTP_USER")
    smtp_password: str = Field(default="", alias="BTI_SMTP_PASSWORD")
    webhook_url: str = Field(default="", alias="BTI_WEBHOOK_URL")
    webhook_secret: str = Field(default="", alias="BTI_WEBHOOK_SECRET")

    # API
    api_host: str = _yaml.get("api", {}).get("host", "0.0.0.0")
    api_port: int = _yaml.get("api", {}).get("port", 8000)
    cors_origins: list = _yaml.get("api", {}).get("cors_origins", ["http://localhost:8501"])

    # v3 model training inside the pipeline
    model_developer: str = Field(default=_yaml.get("modeling", {}).get("developer", ""), alias="BTI_MODEL_DEVELOPER")
    pipeline_algorithms: list = _yaml.get("modeling", {}).get("algorithms", ["hgb", "lightgbm", "xgboost"])
    pipeline_feature_sets: list = _yaml.get("modeling", {}).get("feature_sets", ["core", "extended"])

    # Model monitoring — run the scheduler on exactly one worker per deployment
    monitoring_scheduler_enabled: bool = Field(
        default=_yaml.get("monitoring", {}).get("scheduler_enabled", True),
        alias="BTI_MONITORING_SCHEDULER_ENABLED")
    drift_check_cron: str = Field(default=_yaml.get("monitoring", {}).get("drift_check_cron", "0 6 * * mon"),
                                  alias="BTI_DRIFT_CHECK_CRON")
    drift_window_days: int = _yaml.get("monitoring", {}).get("drift_window_days", 7)
    outcomes_cron: str = _yaml.get("monitoring", {}).get("outcomes_cron", "0 8 1 1,4,7,10 *")
    governance_check_cron: str = _yaml.get("monitoring", {}).get("governance_check_cron", "0 8 * * mon")
    scoring_latency_sla_ms: float = _yaml.get("monitoring", {}).get("scoring_latency_sla_ms", 100.0)

    # Live decision capacity (fitted per model by bti.operations.capacity)
    max_review_rate: float = _yaml.get("decisioning", {}).get("max_review_rate", 0.02)
    max_step_up_rate: float = _yaml.get("decisioning", {}).get("max_step_up_rate", 0.05)

    # Continuous retraining and recalibration (Phase 4)
    retrain_interval_days: int = _yaml.get("retraining", {}).get("interval_days", 30)
    retrain_min_interval_days: int = _yaml.get("retraining", {}).get("min_interval_days", 7)
    retrain_min_new_labels: int = _yaml.get("retraining", {}).get("min_new_labels", 500)
    retrain_check_cron: str = _yaml.get("retraining", {}).get("check_cron", "0 4 * * *")
    retrain_label_maturity_days: int = _yaml.get("retraining", {}).get("label_maturity_days", 90)
    recalibration_cron: str = _yaml.get("retraining", {}).get("recalibration_cron", "0 5 * * *")

    # Cost model (Phase 5): "v1" flat economics, "v2" per-customer value and measured step-up outcomes
    cost_model_version: str = Field(default=_yaml.get("decisioning", {}).get("cost_model", "v1"),
                                    alias="BTI_COST_MODEL")
    cost_v2_value_bounds: list = _yaml.get("decisioning", {}).get("customer_value_bounds_usd", [75.0, 1200.0])
    cost_v2_min_challenges: int = _yaml.get("decisioning", {}).get("min_challenges_for_measured_rates", 200)
    cost_v2_abandonment_prior: float = _yaml.get("decisioning", {}).get("step_up_abandonment_prior", 0.08)
    cost_v2_message_cost_usd: dict = _yaml.get("decisioning", {}).get("step_up_message_cost_usd",
                                                                    {"sms_otp": 0.05, "push": 0.01, "3ds": 0.10})
    cost_v2_margin_rate: float = _yaml.get("decisioning", {}).get("transaction_margin_rate", 0.01)
    customer_value_cron: str = _yaml.get("decisioning", {}).get("customer_value_cron", "30 1 * * *")

    # Online feature store (Phase 7): Redis URL; empty means live scoring reads history from the database
    feature_store_url: str = Field(default=_yaml.get("feature_store", {}).get("url", ""), alias="BTI_FEATURE_STORE_URL")
    # Streaming (Phase 7): Kafka ingestion
    streaming_bootstrap_servers: str = Field(default=_yaml.get("streaming", {}).get("bootstrap_servers",
                                                                                    "localhost:9092"),
                                             alias="BTI_KAFKA_BOOTSTRAP_SERVERS")
    streaming_group_id: str = _yaml.get("streaming", {}).get("group_id", "bti-scoring")
    streaming_topics: dict = _yaml.get("streaming", {}).get("topics", {
        "bti.transactions.json": "json", "bti.transactions.iso8583": "iso8583",
        "bti.transactions.iso20022": "iso20022"})
    streaming_decisions_topic: str = _yaml.get("streaming", {}).get("decisions_topic", "bti.decisions")
    streaming_dlq_topic: str = _yaml.get("streaming", {}).get("dlq_topic", "bti.dlq")
    streaming_partitions: int = _yaml.get("streaming", {}).get("partitions", 6)
    streaming_default_country: Optional[str] = _yaml.get("streaming", {}).get("default_country")
    streaming_max_retries: int = _yaml.get("streaming", {}).get("max_retries", 3)
    streaming_metrics_port: int = _yaml.get("streaming", {}).get("metrics_port", 9308)
    streaming_dlq_include_json_payload: bool = _yaml.get("streaming", {}).get("dlq_include_json_payload", False)
    streaming_security_protocol: str = Field(default=_yaml.get("streaming", {}).get("security_protocol", "PLAINTEXT"),
                                             alias="BTI_KAFKA_SECURITY_PROTOCOL")

    # Data residency (Phase 7): every endpoint that receives customer data must be in-country.
    residency_jurisdiction: Optional[str] = Field(default=_yaml.get("residency", {}).get("jurisdiction"),
                                                  alias="BTI_RESIDENCY_JURISDICTION")
    residency_mode: str = Field(default=_yaml.get("residency", {}).get("mode", "warn"), alias="BTI_RESIDENCY_MODE")
    residency_in_country_hosts: list = _yaml.get("residency", {}).get("in_country_hosts", [
        "localhost", "127.0.0.0/8", "::1", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "*.internal"])

    # Tokenisation key for card numbers and account identifiers from payment messages (HMAC-SHA256).
    # Keep it in the bank's secret store / HSM; rotating it changes every token, so plan rotation as a re-key.
    token_key: str = Field(default="", alias="BTI_TOKEN_KEY")
    # Explanations: "inline" computes SHAP reason codes before responding (~2 ms for the current model);
    # "async" responds first and fills reason codes on the score log and case from a worker pool.
    explain_mode: str = Field(default=_yaml.get("scoring", {}).get("explain_mode", "inline"), alias="BTI_EXPLAIN_MODE")
    explain_workers: int = _yaml.get("scoring", {}).get("explain_workers", 2)

    # Graph intelligence (Phase 6): assumed fraud-confirmation delay when the data has no confirmation time
    graph_label_delay_days: int = _yaml.get("graph", {}).get("label_delay_days", 30)
    graph_snapshot_cron: str = _yaml.get("graph", {}).get("snapshot_cron", "15 0 * * *")
    graph_snapshot_path: str = Field(default=str(BASE_DIR / _yaml.get("graph", {}).get(
        "snapshot_path", "data/graph/snapshot.joblib")), alias="BTI_GRAPH_SNAPSHOT_PATH")

    # Step-up orchestration (Phase 5)
    stepup_provider: str = Field(default=_yaml.get("stepup", {}).get("provider", "log"), alias="BTI_STEPUP_PROVIDER")
    stepup_webhook_url: str = Field(default=_yaml.get("stepup", {}).get("webhook_url", ""), alias="BTI_STEPUP_WEBHOOK_URL")
    stepup_webhook_secret: str = Field(default="", alias="BTI_STEPUP_WEBHOOK_SECRET")
    stepup_callback_secret: str = Field(default="", alias="BTI_STEPUP_CALLBACK_SECRET")
    stepup_ttl_seconds: int = _yaml.get("stepup", {}).get("ttl_seconds", 300)
    stepup_max_attempts: int = _yaml.get("stepup", {}).get("max_attempts", 3)
    stepup_auto_issue: bool = _yaml.get("stepup", {}).get("auto_issue", False)
    stepup_methods: dict = _yaml.get("stepup", {}).get("methods", {
        "Card Not Present": "3ds", "Mobile Banking": "push", "Internet Banking": "sms_otp",
        "API/Open Banking": "sms_otp"})
    stepup_expiry_cron: str = _yaml.get("stepup", {}).get("expiry_cron", "*/5 * * * *")

    # Case management (Phase 4): queues, SLAs in minutes, maker-checker threshold for clearing high-value cases
    case_queues: dict = _yaml.get("cases", {}).get("queues", {
        "urgent": {"sla_minutes": 60, "min_probability": 0.8, "min_expected_loss_usd": 1000},
        "high_value": {"sla_minutes": 120, "min_amount_usd": 10000},
        "standard": {"sla_minutes": 480},
    })
    case_checker_threshold_usd: float = _yaml.get("cases", {}).get("checker_threshold_usd", 10000.0)
    case_sla_check_cron: str = _yaml.get("cases", {}).get("sla_check_cron", "*/15 * * * *")

    # Parallel run alongside the incumbent platform (Phase 2)
    incumbent_system: str = _yaml.get("parallel_run", {}).get("incumbent_system", "SAS")
    incumbent_decision_map: dict = _yaml.get("parallel_run", {}).get("decision_map", {})
    parallel_max_bti_share: float = _yaml.get("parallel_run", {}).get("max_bti_share", 0.10)
    parallel_bti_timeout_ms: float = Field(default=_yaml.get("parallel_run", {}).get("bti_timeout_ms", 150.0),
                                           alias="BTI_PARALLEL_TIMEOUT_MS")
    parallel_report_cron: str = _yaml.get("parallel_run", {}).get("report_cron", "0 7 * * mon")
    reconciliation_cron: str = _yaml.get("parallel_run", {}).get("reconciliation_cron", "30 2 * * *")
    reconciliation_break_rate_alert: float = _yaml.get("parallel_run", {}).get("break_rate_alert", 0.001)

    # Audit
    audit_enabled: bool = Field(default=True, alias="BTI_AUDIT_ENABLED")
    audit_log_path: str = str(BASE_DIR / "logs" / "audit.jsonl")
    audit_archive_dir: str = Field(default=str(BASE_DIR / _yaml.get("audit", {}).get("archive_dir", "audit_archive")),
                                   alias="BTI_AUDIT_ARCHIVE_DIR")
    audit_retention_days: int = _yaml.get("audit", {}).get("retention_days", 2555)
    audit_archive_cron: str = _yaml.get("audit", {}).get("archive_cron", "0 3 * * *")

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
