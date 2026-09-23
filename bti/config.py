"""
Centralised configuration — reads settings.yaml then overrides with env vars.
All application code imports from here; never use hardcoded paths or values.
"""

import os
import yaml
from pathlib import Path
from functools import lru_cache
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
    scoring_latency_sla_ms: float = _yaml.get("monitoring", {}).get("scoring_latency_sla_ms", 100.0)

    # Audit
    audit_enabled: bool = Field(default=True, alias="BTI_AUDIT_ENABLED")
    audit_log_path: str = str(BASE_DIR / "logs" / "audit.jsonl")

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
