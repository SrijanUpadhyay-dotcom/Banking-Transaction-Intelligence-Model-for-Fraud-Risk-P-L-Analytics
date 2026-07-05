"""
Unit tests for the fraud rules engine.
Each rule is tested in isolation with minimal synthetic data.
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Allow importing src scripts directly
SRC_DIR = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR.parent))


def _load_rules_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rules", SRC_DIR / "03_fraud_rules_engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rules = _load_rules_module()


def _base_row(**overrides) -> dict:
    """Minimal transaction row — override only what each test needs."""
    row = {
        "transaction_id": "T001",
        "customer_id": "C001",
        "transaction_date": pd.Timestamp("2024-01-15"),
        "transaction_amount": 500.0,
        "historical_average_transaction_amount": 200.0,
        "failed_attempt_count": 0,
        "login_attempts": 1,
        "transaction_time": "10:00:00",
        "ip_location": "203.0.113.1",
        "risk_score": 30,
        "refund_flag": 0,
        "chargeback_flag": 0,
        "merchant_name": "Amazon",
        "merchant_category": "Online Marketplaces",
        "customer_segment": "Mass Market",
        "debit_credit_flag": "Debit",
        "account_balance_after": 5000.0,
        "channel": "Mobile App",
        "device_id": "DEV-ABC123",
    }
    row.update(overrides)
    return row


def _df(*rows, **kwargs) -> pd.DataFrame:
    """Build a DataFrame. Pass dicts as positional args for multi-row, or kwargs for single-row."""
    if rows:
        return pd.DataFrame([_base_row(**r) for r in rows])
    return pd.DataFrame([_base_row(**kwargs)])


# ── R01: High value vs average ────────────────────────────────────────────────

class TestR01HighValueVsAvg:
    def test_triggers_when_ratio_gte_5(self):
        df = _df(transaction_amount=1001.0, historical_average_transaction_amount=200.0)
        flag, _ = rules.rule_high_value_vs_avg(df)
        assert flag.iloc[0] == 1

    def test_no_trigger_when_ratio_below_5(self):
        df = _df(transaction_amount=999.0, historical_average_transaction_amount=200.0)
        flag, _ = rules.rule_high_value_vs_avg(df)
        assert flag.iloc[0] == 0

    def test_zero_historical_average_no_crash(self):
        df = _df(transaction_amount=500.0, historical_average_transaction_amount=0.0)
        flag, _ = rules.rule_high_value_vs_avg(df)
        # Zero historical average → ratio is NaN → no flag
        assert flag.iloc[0] == 0


# ── R03: Failed authentication ─────────────────────────────────────────────────

class TestR03FailedAuth:
    def test_triggers_at_threshold(self):
        df = _df(failed_attempt_count=3)
        flag, _ = rules.rule_failed_auth_attempts(df)
        assert flag.iloc[0] == 1

    def test_no_trigger_below_threshold(self):
        df = _df(failed_attempt_count=2)
        flag, _ = rules.rule_failed_auth_attempts(df)
        assert flag.iloc[0] == 0

    def test_triggers_well_above_threshold(self):
        df = _df(failed_attempt_count=10)
        flag, _ = rules.rule_failed_auth_attempts(df)
        assert flag.iloc[0] == 1


# ── R05: Off-hours transaction ─────────────────────────────────────────────────

class TestR05OffHours:
    def test_midnight_triggers(self):
        df = _df(transaction_time="00:15:00")
        flag, _ = rules.rule_off_hours_transaction(df)
        assert flag.iloc[0] == 1

    def test_5am_triggers(self):
        df = _df(transaction_time="05:59:00")
        flag, _ = rules.rule_off_hours_transaction(df)
        assert flag.iloc[0] == 1

    def test_6am_no_trigger(self):
        df = _df(transaction_time="06:00:00")
        flag, _ = rules.rule_off_hours_transaction(df)
        assert flag.iloc[0] == 0

    def test_midday_no_trigger(self):
        df = _df(transaction_time="12:30:00")
        flag, _ = rules.rule_off_hours_transaction(df)
        assert flag.iloc[0] == 0


# ── R11: Balance inconsistency ─────────────────────────────────────────────────

class TestR11BalanceInconsistency:
    def test_negative_balance_after_debit_triggers(self):
        df = _df(debit_credit_flag="Debit", account_balance_after=-50.0)
        flag, _ = rules.rule_balance_inconsistency(df)
        assert flag.iloc[0] == 1

    def test_positive_balance_no_trigger(self):
        df = _df(debit_credit_flag="Debit", account_balance_after=100.0)
        flag, _ = rules.rule_balance_inconsistency(df)
        assert flag.iloc[0] == 0

    def test_credit_with_negative_balance_no_trigger(self):
        # Credit shouldn't trigger the debit balance rule
        df = _df(debit_credit_flag="Credit", account_balance_after=-50.0)
        flag, _ = rules.rule_balance_inconsistency(df)
        assert flag.iloc[0] == 0


# ── R15: Duplicate transaction ─────────────────────────────────────────────────

class TestR15Duplicate:
    def test_exact_duplicate_triggers(self):
        rows = [
            {"transaction_id": "T001", "customer_id": "C001",
             "transaction_amount": 100.0, "merchant_name": "Amazon",
             "transaction_date": pd.Timestamp("2024-01-15")},
            {"transaction_id": "T002", "customer_id": "C001",
             "transaction_amount": 100.0, "merchant_name": "Amazon",
             "transaction_date": pd.Timestamp("2024-01-15")},
        ]
        df = pd.DataFrame([_base_row(**r) for r in rows])
        flag, _ = rules.rule_duplicate_transaction(df)
        assert flag.sum() == 2

    def test_different_amounts_no_flag(self):
        rows = [
            {"transaction_id": "T001", "transaction_amount": 100.0},
            {"transaction_id": "T002", "transaction_amount": 200.0},
        ]
        df = pd.DataFrame([_base_row(**r) for r in rows])
        flag, _ = rules.rule_duplicate_transaction(df)
        assert flag.sum() == 0


# ── Composite scoring ──────────────────────────────────────────────────────────

class TestCompositeScoring:
    def test_score_is_between_0_and_100(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cleaning", SRC_DIR / "02_data_cleaning.py")
        clean_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clean_mod)

        # Build a minimal clean-ish DataFrame
        rows = [_base_row(
            failed_attempt_count=5,
            transaction_amount=2000.0,
            historical_average_transaction_amount=100.0,
            risk_score=80,
            transaction_time="02:00:00",
        )]
        df = pd.DataFrame(rows)
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        result = rules.apply_fraud_rules(df)
        assert 0 <= result["fraud_rule_score"].iloc[0] <= 100

    def test_alert_tier_categories(self):
        df = pd.DataFrame([_base_row()])
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        result = rules.apply_fraud_rules(df)
        valid_tiers = {"Green", "Yellow", "Orange", "Red", "Critical"}
        assert set(result["alert_tier"].dropna().unique()).issubset(valid_tiers)
