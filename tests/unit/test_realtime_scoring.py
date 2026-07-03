"""
Unit tests for the real-time scoring engine.
Uses the trained model artifacts — requires the pipeline to have run at least once.
Tests are marked to skip gracefully if models aren't available.
"""

import pytest
from pathlib import Path


MODELS_AVAILABLE = (Path("models") / "manifest.json").exists()
requires_models = pytest.mark.skipif(
    not MODELS_AVAILABLE,
    reason="Models not trained yet — run: python main.py pipeline --force"
)


def _suspicious_txn(**overrides) -> dict:
    base = {
        "transaction_id": "TEST-SUSPICIOUS-001",
        "customer_id": "CUST-TEST",
        "transaction_amount": 12500.0,
        "transaction_time": "02:47:00",
        "channel": "API/Open Banking",
        "merchant_category": "Crypto Exchanges",
        "merchant_name": "CryptoFX Ltd",
        "customer_segment": "Mass Market",
        "risk_score": 72,
        "historical_average_transaction_amount": 450.0,
        "failed_attempt_count": 4,
        "login_attempts": 6,
        "account_balance_after": 500.0,
        "debit_credit_flag": "Debit",
        "device_id": "DEV-X9921-UNKNOWN",
        "ip_location": "203.0.113.45",
        "refund_flag": 0,
        "chargeback_flag": 0,
    }
    base.update(overrides)
    return base


def _normal_txn(**overrides) -> dict:
    base = {
        "transaction_id": "TEST-NORMAL-001",
        "customer_id": "CUST-SAFE",
        "transaction_amount": 45.0,
        "transaction_time": "14:30:00",
        "channel": "Mobile App",
        "merchant_category": "Grocery & Supermarkets",
        "merchant_name": "Tesco",
        "customer_segment": "Mass Market",
        "risk_score": 15,
        "historical_average_transaction_amount": 52.0,
        "failed_attempt_count": 0,
        "login_attempts": 1,
        "account_balance_after": 2500.0,
        "debit_credit_flag": "Debit",
        "device_id": "DEV-IPHONE14-HOME",
        "ip_location": "192.168.1.1",
        "refund_flag": 0,
        "chargeback_flag": 0,
    }
    base.update(overrides)
    return base


@requires_models
class TestModelLoader:
    def test_load_models_returns_bundle(self):
        from bti.scoring.model_loader import load_models
        bundle = load_models()
        assert bundle is not None
        assert bundle.iso_forest is not None
        assert bundle.rf_model is not None
        assert bundle.lr_model is not None

    def test_feature_cols_nonempty(self):
        from bti.scoring.model_loader import load_models
        bundle = load_models()
        assert len(bundle.feature_cols) > 0

    def test_trained_at_is_iso_string(self):
        from bti.scoring.model_loader import load_models
        bundle = load_models()
        assert "T" in bundle.trained_at  # ISO 8601 format

    def test_roc_auc_above_threshold(self):
        from bti.scoring.model_loader import load_models
        bundle = load_models()
        assert bundle.rf_roc_auc >= 0.90, f"RF ROC-AUC too low: {bundle.rf_roc_auc}"

    def test_models_cached_on_second_call(self):
        from bti.scoring.model_loader import load_models
        b1 = load_models()
        b2 = load_models()
        assert b1 is b2  # same object — cached


@requires_models
class TestScoringEngine:
    def setup_method(self):
        from bti.scoring.model_loader import load_models
        self.bundle = load_models()

    def test_suspicious_txn_scores_higher_than_normal(self):
        from bti.scoring.realtime import score_transaction
        r_sus = score_transaction(_suspicious_txn(), self.bundle)
        r_nrm = score_transaction(_normal_txn(), self.bundle)
        assert r_sus.final_risk_score > r_nrm.final_risk_score

    def test_scores_are_in_valid_range(self):
        from bti.scoring.realtime import score_transaction
        result = score_transaction(_suspicious_txn(), self.bundle)
        assert 0 <= result.fraud_rule_score <= 100
        assert 0 <= result.ml_anomaly_score <= 100
        assert 0 <= result.final_risk_score <= 100
        assert 0 <= result.ml_lr_proba <= 1
        assert 0 <= result.ml_rf_proba <= 1

    def test_normal_txn_fires_zero_rules(self):
        from bti.scoring.realtime import score_transaction
        result = score_transaction(_normal_txn(), self.bundle)
        assert result.rules_triggered == 0
        assert result.rules_fired == []

    def test_off_hours_rule_fires(self):
        from bti.scoring.realtime import score_transaction
        result = score_transaction(_suspicious_txn(transaction_time="03:00:00"), self.bundle)
        assert "R05_off_hours" in result.rules_fired

    def test_off_hours_does_not_fire_during_day(self):
        from bti.scoring.realtime import score_transaction
        result = score_transaction(_normal_txn(transaction_time="09:00:00"), self.bundle)
        assert "R05_off_hours" not in result.rules_fired

    def test_high_value_vs_avg_fires_at_5x(self):
        from bti.scoring.realtime import score_transaction
        txn = _normal_txn(transaction_amount=2600.0,
                          historical_average_transaction_amount=500.0)
        result = score_transaction(txn, self.bundle)
        assert "R01_high_value_vs_avg" in result.rules_fired

    def test_failed_auth_threshold(self):
        from bti.scoring.realtime import score_transaction
        # 2 failures — should NOT fire
        r_below = score_transaction(_normal_txn(failed_attempt_count=2), self.bundle)
        assert "R03_failed_auth" not in r_below.rules_fired
        # 3 failures — should fire
        r_at = score_transaction(_normal_txn(failed_attempt_count=3), self.bundle)
        assert "R03_failed_auth" in r_at.rules_fired

    def test_alert_tier_is_valid(self):
        from bti.scoring.realtime import score_transaction
        result = score_transaction(_suspicious_txn(), self.bundle)
        valid_tiers = {"LOW", "MEDIUM", "HIGH", "VERY HIGH", "CRITICAL"}
        assert result.final_alert_tier in valid_tiers

    def test_processing_time_under_500ms(self):
        from bti.scoring.realtime import score_transaction
        result = score_transaction(_suspicious_txn(), self.bundle)
        # Subsequent calls (warm cache) must be fast
        result2 = score_transaction(_normal_txn(), self.bundle)
        assert result2.processing_time_ms < 500

    def test_balance_inconsistency_rule(self):
        from bti.scoring.realtime import score_transaction
        txn = _normal_txn(debit_credit_flag="Debit", account_balance_after=-100.0)
        result = score_transaction(txn, self.bundle)
        assert "R11_balance_inconsistency" in result.rules_fired

    def test_returns_transaction_id(self):
        from bti.scoring.realtime import score_transaction
        result = score_transaction(_suspicious_txn(transaction_id="UNIQUE-ABC"), self.bundle)
        assert result.transaction_id == "UNIQUE-ABC"
