# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 9: registry families, payee-risk features, UK reimbursement exposure, scam overlay and mule detection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from bti.config import get_settings
from bti.modeling import registry
from bti.scams import mule, reimbursement
from bti.scams.features import payment_rows, scam_features


def _pay(rows):
    cols = ("transaction_id", "customer_id", "transaction_date", "transaction_time", "transaction_amount", "payee_id")
    base = {"currency": "GBP", "transaction_type": "Transfer", "debit_credit_flag": "Debit", "country": "United Kingdom",
            "historical_average_transaction_amount": 100.0, "account_balance_before": 5000.0, "cop_result": "match",
            "payee_account_opened_date": None, "payee_customer_id": None, "fraud_flag": 0, "label_confirmed_at": None,
            "device_id": "D", "ip_location": "I"}
    return pd.DataFrame([{**base, **dict(zip(cols, r[:6])), **(r[6] if len(r) > 6 else {})} for r in rows])


def test_registry_families_are_isolated_and_keep_four_eyes(tmp_path, monkeypatch):
    monkeypatch.setenv("BTI_MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    registry.clear_cache()
    card = {"ownership": {"developer": "dev.one"}, "validation": {"status": "passed"}}
    registry.save_model("scam-1", {"stub": True}, card, family="scam")
    assert registry.read_index()["models"] == []                                     # fraud family untouched
    assert [m["model_id"] for m in registry.read_index("scam")["models"]] == ["scam-1"]
    assert registry.registry_dir("scam") == tmp_path / "registry_scam"
    with pytest.raises(registry.RegistryError, match="Four-eyes"):
        registry.assign_role("scam-1", "champion", "dev.one", "self approval must fail", family="scam")
    registry.assign_role("scam-1", "champion", "risk.officer", "independent validation done", family="scam")
    assert registry.model_for_role("champion", family="scam") == "scam-1" and registry.model_for_role("champion") is None
    with pytest.raises(registry.RegistryError):
        registry.registry_dir("unknown")
    registry.clear_cache()


def test_payee_features_are_point_in_time():
    df = _pay([
        ("T1", "A", "2024-01-01", "10:00:00", 100, "P1"),
        ("T2", "B", "2024-01-01", "10:00:00", 100, "P1"),                          # same second: invisible to T1
        ("T3", "C", "2024-01-02", "09:00:00", 500, "P1", {"cop_result": "no_match",
                                                          "payee_account_opened_date": "2023-12-20"}),
        ("T4", "C", "2024-01-03", "09:00:00", 900, "P1"),
        ("T5", "C", "2024-01-03", "10:00:00", 50, "P9", {"cop_result": None}),
    ])
    f = scam_features(df, graph=False).set_index(df["transaction_id"])
    assert f.loc["T1", "payee_distinct_senders_30d"] == 0 and f.loc["T2", "payee_new_to_bank"] == 1
    assert f.loc["T3", "payee_distinct_senders_30d"] == 2 and f.loc["T3", "payee_new_for_customer"] == 1
    assert f.loc["T3", "payee_account_age_days"] == 13 and f.loc["T3", "cop_no_match"] == 1
    assert f.loc["T4", "cust_payments_to_payee_30d"] == 1 and f.loc["T4", "amount_vs_cust_max_payment"] == pytest.approx(1.8)
    assert f.loc["T5", "cust_new_payees_7d"] == 1 and np.isnan(f.loc["T5", "cop_no_match"])
    later = pd.concat([df, _pay([("T9", "Z", "2024-02-01", "10:00:00", 10, "P1")])], ignore_index=True)
    pd.testing.assert_frame_equal(scam_features(later, graph=False).iloc[:5].reset_index(drop=True),
                                  f.reset_index(drop=True), check_names=False)        # future rows change nothing


def test_uk_reimbursement_exposure_rules(monkeypatch):
    r = dict(get_settings().scam_reimbursement, excess_gbp=100)
    monkeypatch.setattr(get_settings(), "scam_reimbursement", r)
    df = _pay([
        ("A", "C1", "2024-01-01", "10:00:00", 1000, "P1"),
        ("B", "C1", "2024-01-01", "10:00:00", 200000, "P2"),                       # capped at £85k
        ("C", "C2", "2024-01-01", "10:00:00", 1000, "P3", {"payee_customer_id": "C9"}),   # on-us: 100%
        ("D", "C3", "2024-01-01", "10:00:00", 1000, "P4", {"country": "Germany"}),        # out of scope
        ("E", "C4", "2024-01-01", "10:00:00", 1000, "P5", {"transaction_type": "Wire Transfer"}),
        ("F", "C5", "2024-01-01", "10:00:00", 1000, "P6", {"is_vulnerable": 1}),          # no excess
    ])
    df["is_vulnerable"] = df.get("is_vulnerable", 0)
    e = reimbursement.exposure_gbp(df.fillna({"is_vulnerable": 0}))
    assert list(e) == [450.0, 42450.0, 900.0, 0.0, 0.0, 500.0]
    claims = reimbursement.claim_exposure(_pay([("A", "C1", "2024-01-01", "10:00:00", 60000, "P1"),
                                                ("B", "C1", "2024-01-05", "10:00:00", 60000, "P1")]))
    assert claims["claims"] == 1 and claims["claims_at_cap"] == 1 and claims["bank_exposure_gbp"] == pytest.approx(42450)


def test_intervention_choice_scales_with_risk_and_exposure():
    assert reimbursement.choose_intervention(0.001, 50, 0)["action"] == "none"
    assert reimbursement.choose_intervention(0.6, 5000, 2000)["action"] == "hold_and_call"
    mid = reimbursement.choose_intervention(0.05, 300, 100)
    assert mid["action"] in ("warning", "none") and set(mid["expected_cost_gbp"]) == {"none", "warning", "hold_and_call"}


@dataclass
class _Decision:
    action: str
    guardrails_applied: List[str] = field(default_factory=list)


def test_overlay_acts_only_when_active_and_approved_and_never_lowers():
    from bti.scams.overlay import apply
    scam = {"acting": False, "action": "hold_and_call"}
    d = _Decision("APPROVE")
    apply(d, scam)
    assert d.action == "APPROVE"                                                   # shadow / provisional: no effect
    d = _Decision("APPROVE")
    apply(d, {**scam, "acting": True})
    assert d.action == "REVIEW" and "APP-scam" in d.guardrails_applied[0]
    d = _Decision("DECLINE")
    apply(d, {**scam, "acting": True})
    assert d.action == "DECLINE"                                                    # never lowered
    d = _Decision("APPROVE")
    apply(d, {"acting": True, "action": "warning"})
    assert d.action == "APPROVE" and "warning" in d.guardrails_applied[0]


def test_mule_labels_exclude_ambiguous_snapshots():
    snaps = pd.DataFrame({"customer_id": ["M", "M", "M", "G"],
                          "snapshot_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-05-01", "2024-02-01"])})
    labels = pd.DataFrame({"customer_id": ["M"], "activated_at": [pd.Timestamp("2024-01-15")],
                           "uncovered_at": [pd.Timestamp("2024-03-01")]})
    y = mule.label(snaps, labels)
    assert np.isnan(y[0]) and y[1] == 1 and np.isnan(y[2]) and y[3] == 0


def test_mule_snapshot_flow_features_and_reported_fraud_timing():
    rows = []
    for i in range(6):                                                              # 6 senders, money out next day
        rows.append({"transaction_id": f"I{i}", "customer_id": "M", "transaction_date": f"2024-03-{2 + i:02d}",
                     "transaction_time": "10:00:00", "transaction_amount": 100.0, "currency": "GBP",
                     "debit_credit_flag": "Credit", "transaction_type": "Transfer", "counterparty_id": f"S{i}",
                     "payee_id": None, "channel": "Mobile Banking", "merchant_category": None})
        rows.append({"transaction_id": f"O{i}", "customer_id": "M", "transaction_date": f"2024-03-{3 + i:02d}",
                     "transaction_time": "09:00:00", "transaction_amount": 90.0, "currency": "GBP",
                     "debit_credit_flag": "Debit", "transaction_type": "Withdrawal", "counterparty_id": None,
                     "payee_id": None, "channel": "ATM", "merchant_category": None})
    rows.append({"transaction_id": "V", "customer_id": "VICTIM", "transaction_date": "2024-03-05",
                 "transaction_time": "11:00:00", "transaction_amount": 100.0, "currency": "GBP",
                 "debit_credit_flag": "Debit", "transaction_type": "Transfer", "payee_id": "PAYEE-M",
                 "payee_customer_id": "M", "fraud_flag": 1, "label_confirmed_at": "2024-03-12",
                 "channel": "Mobile Banking", "merchant_category": None})
    df = pd.DataFrame(rows)
    df["account_opened_date"] = "2024-02-20"
    for c in ("device_id", "ip_location", "label_confirmed_at", "fraud_flag", "payee_customer_id"):
        if c not in df:
            df[c] = None
    s = mule.snapshots(df, dates=[pd.Timestamp("2024-03-10"), pd.Timestamp("2024-03-14")]).set_index(
        ["customer_id", "snapshot_date"])
    first = s.loc[("M", pd.Timestamp("2024-03-10"))]
    assert first["in_distinct_senders_30d"] == 6 and first["account_age_days"] == 19
    assert first["fast_out_share"] == pytest.approx(0.9, abs=0.01) and first["cashout_share"] == 1.0
    assert first["inbound_reported_fraud_30d"] == 0                               # confirmed on the 12th
    assert s.loc[("M", pd.Timestamp("2024-03-14")), "inbound_reported_fraud_30d"] == 1


def test_registered_scam_model_scores_a_risky_payment_above_a_biller():
    from bti.scams.overlay import assess
    if not registry.model_for_role("challenger", family="scam"):
        pytest.skip("no scam model registered")
    base = dict(customer_id="C-1", transaction_date="2024-11-02", transaction_time="14:10:00", currency="GBP",
                country="United Kingdom", transaction_type="Transfer", debit_credit_flag="Debit",
                historical_average_transaction_amount=120.0, account_balance_before=9000.0)
    risky = assess({**base, "transaction_id": "R", "transaction_amount": 4800.0, "payee_id": "NEW",
                    "cop_result": "no_match", "payee_account_opened_date": "2024-10-20"}, pd.DataFrame())
    safe = assess({**base, "transaction_id": "S", "transaction_amount": 80.0, "payee_id": "BILLER",
                   "cop_result": "match", "payee_account_opened_date": "2009-01-01"}, pd.DataFrame())
    assert risky["probability"] > 10 * safe["probability"] and risky["exposure_gbp"] == 2400.0
    assert risky["mode"] == "shadow" and not risky["acting"] and risky["reason_codes"]
    assert assess({**base, "transaction_id": "X", "transaction_amount": 10.0, "transaction_type": "Purchase"},
                  pd.DataFrame()) is None                                           # not a payment
