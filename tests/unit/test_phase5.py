"""Phase 5: governed rules, step-up orchestration, cost model v2 and decision-level fairness."""

import hashlib
import hmac
import json
import re
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from bti.database.models import Base, FraudLabel, RuleHit, StepUpChallenge
from bti.rules import lifecycle
from bti.rules.language import RuleError, evaluate, validate


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    lifecycle.invalidate()
    yield s
    s.close()
    lifecycle.invalidate()


# ── Rule language ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("condition,message", [
    ({"field": "customer_segment", "op": "==", "value": "Student"}, "protected"),
    ({"field": "risk_score", "op": ">", "value": 60}, "label_derived"),
    ({"field": "chargeback_flag", "op": "==", "value": 1}, "post_event"),
    ({"field": "channel", "op": ">", "value": 3}, "not meaningful"),
    ({"field": "amount_usd", "op": "between", "value": 3}, "Unknown operator"),
    ({"all": []}, "non-empty list"),
    ({"field": "device_id", "op": "in", "value": []}, "list of"),
])
def test_rule_language_refuses_ungoverned_conditions(condition, message):
    with pytest.raises(RuleError, match=message):
        validate(condition)


def test_rule_language_evaluates_with_null_safe_semantics():
    frame = pd.DataFrame({"amount_usd": [50, 5000, np.nan, 9000], "channel": ["ATM", "Branch", "ATM", None],
                          "merchant_name": ["A", "B", "BAD", "BAD"]})
    cond = {"any": [{"all": [{"field": "amount_usd", "op": ">=", "value": 1000},
                             {"not": {"field": "channel", "op": "==", "value": "Branch"}}]},
                    {"field": "merchant_name", "op": "in", "value": ["BAD"]}]}
    assert validate(cond) == ["amount_usd", "channel", "merchant_name"]
    assert evaluate(cond, frame).tolist() == [False, False, True, True]
    assert evaluate({"field": "amount_usd", "op": "is_null"}, frame).tolist() == [False, False, True, False]


# ── Rule lifecycle ───────────────────────────────────────────────────────────

@pytest.fixture
def history(monkeypatch):
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.05).astype(int)
    frame = pd.DataFrame({
        "amount_usd": rng.gamma(2, 300, n), "failed_attempt_count": np.where(y == 1, 4, rng.integers(0, 2, n)),
        "login_attempts": rng.integers(1, 3, n), "fraud_probability": rng.random(n) * 0.1,
        "model_decision": np.where(rng.random(n) < 0.9, "APPROVE", "REVIEW"), "label": y,
        "_customer_segment": rng.choice(["Retail", "Premium", "SME"], n), "_customer_age_band": "26–35",
        "_country": rng.choice(["United Kingdom", "Germany"], n), "_transaction_id": [f"H{i}" for i in range(n)],
        "_ts": pd.date_range("2024-07-01", periods=n, freq="h"), "currency": "GBP", "merchant_name": "M",
        "payee_id": None, "device_id": None, "ip_location": None})
    monkeypatch.setattr(lifecycle, "_history", lambda model_id: frame)
    from bti.modeling import registry
    monkeypatch.setattr(registry, "model_for_role", lambda role: "m-test" if role == "challenger" else None)
    return frame


def test_rules_need_simulation_and_four_eyes(db, history):
    r = lifecycle.create_rule(db, "Failed auth", "Three or more failed attempts", {"field": "failed_attempt_count",
                              "op": ">=", "value": 3}, "REVIEW", "ana")
    with pytest.raises(RuleError, match="simulate"):
        lifecycle.approve(db, r["id"], "ben")
    sim = lifecycle.simulate(db, r["id"], "ana")["simulation"]
    assert sim["hits"] == int((history["failed_attempt_count"] >= 3).sum()) and sim["precision"] == 1.0
    assert sim["incremental_fraud_caught"] > 0 and sim["fairness"]["status"] == "pass"
    with pytest.raises(RuleError, match="Four-eyes"):
        lifecycle.approve(db, r["id"], "ANA")
    assert lifecycle.approve(db, r["id"], "ben")["status"] == "active"
    v2 = lifecycle.new_version(db, r["rule_id"], "ana", condition={"field": "failed_attempt_count", "op": ">=",
                                                                   "value": 4})
    lifecycle.simulate(db, v2["id"], "ana")
    assert lifecycle.approve(db, v2["id"], "ben", mode="shadow")["status"] == "shadow"       # challenger version
    statuses = {v["version"]: v["status"] for v in lifecycle.versions(db, r["rule_id"])}
    assert statuses == {1: "active", 2: "shadow"}
    lifecycle.simulate(db, v2["id"], "ana")
    lifecycle.approve(db, v2["id"], "ben", mode="active")                                    # promote challenger
    assert {v["version"]: v["status"] for v in lifecycle.versions(db, r["rule_id"])} == {1: "retired", 2: "active"}


def test_low_precision_decline_rules_run_in_shadow_only(db, history):
    r = lifecycle.create_rule(db, "Big payments", "Decline anything over $200", {"field": "amount_usd", "op": ">",
                              "value": 200}, "DECLINE", "ana")
    lifecycle.simulate(db, r["id"], "ana")
    with pytest.raises(RuleError, match="precision"):
        lifecycle.approve(db, r["id"], "ben")
    assert lifecycle.approve(db, r["id"], "ben", mode="shadow")["status"] == "shadow"


def test_live_rules_raise_decisions_and_shadow_rules_only_log(db, history):
    active = lifecycle.create_rule(db, "Watchlist merchant", "Known compromised merchant",
                                   {"field": "merchant_name", "op": "in", "value": ["BAD MERCHANT"]}, "STEP_UP", "ana")
    shadow = lifecycle.create_rule(db, "Night logins", "Many logins", {"field": "login_attempts", "op": ">=",
                                   "value": 5}, "DECLINE", "ana")
    for r, mode in ((active, "active"), (shadow, "shadow")):
        lifecycle.simulate(db, r["id"], "ana")
        lifecycle.approve(db, r["id"], "ben", mode=mode)
    decision = SimpleNamespace(action="APPROVE", guardrails_applied=[])
    hits = lifecycle.apply_rules(db, {"merchant_name": "BAD MERCHANT", "login_attempts": 7}, decision,
                                 can_step_up=False)
    assert decision.action == "REVIEW"                                      # STEP_UP impossible here → REVIEW
    assert {(h["rule_id"], h["enforced"]) for h in hits} == {(active["rule_id"], True), (shadow["rule_id"], False)}
    high = SimpleNamespace(action="DECLINE", guardrails_applied=[])
    lifecycle.apply_rules(db, {"merchant_name": "BAD MERCHANT"}, high, can_step_up=True)
    assert high.action == "DECLINE"                                         # rules never lower a decision
    lifecycle.record_hits(db, "T1", hits)
    db.add(FraudLabel(transaction_id="T1", label=1, label_source="CHARGEBACK", event_at=datetime.utcnow()))
    db.commit()
    perf = lifecycle.performance(db, active["rule_id"])
    assert perf["versions"][0]["hits"] == 1 and perf["versions"][0]["precision"] == 1.0
    assert db.query(RuleHit).count() == 2


# ── Step-up ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dev(monkeypatch):
    from bti.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "environment", "development")
    monkeypatch.setattr(s, "stepup_provider", "log")
    monkeypatch.setattr(s, "stepup_callback_secret", "cb-secret")
    return s


def _capture(monkeypatch):
    from bti.operations import stepup
    sent = []
    monkeypatch.setattr(stepup, "_send", lambda c, message: sent.append(message) or "ref-1")
    return sent


def test_sms_code_is_hashed_limited_and_expires(db, dev, monkeypatch):
    from bti.operations import stepup
    sent = _capture(monkeypatch)
    c = stepup.issue(db, "T1", "C1", "Internet Banking", "Transfer", 250)
    code = re.search(r"\b(\d{6})\b", sent[0]["text"]).group(1)
    row = db.get(StepUpChallenge, c["challenge_id"])
    assert c["method"] == "sms_otp" and code not in (row.secret_hash or "") and len(c["challenge_id"]) == 32
    assert "2 attempt" in stepup.verify_code(db, c["challenge_id"], "000000" if code != "000000" else "111111")["detail"]
    assert stepup.verify_code(db, c["challenge_id"], code)["status"] == "passed"
    d = stepup.issue(db, "T2", "C1", "Internet Banking", "Transfer", 250)
    for _ in range(3):
        wrong = "999999" if re.search(r"\b(\d{6})\b", sent[1]["text"]).group(1) != "999999" else "888888"
        last = stepup.verify_code(db, d["challenge_id"], wrong)
    assert last["status"] == "failed"
    e = stepup.issue(db, "T3", "C1", "Internet Banking", "Transfer", 250)
    assert stepup.expire(db, now=datetime.utcnow() + timedelta(minutes=10))["abandoned"] == 1
    with pytest.raises(stepup.StepUpError, match="abandoned"):
        stepup.verify_code(db, e["challenge_id"], "123456")
    stats = stepup.stepup_stats(db)["methods"]["sms_otp"]
    assert (stats["passed"], stats["failed"], stats["abandoned"]) == (1, 1, 1)


def test_push_and_3ds_results_need_a_valid_signature(db, dev, monkeypatch):
    from bti.operations import stepup
    _capture(monkeypatch)
    sign = lambda body: hmac.new(b"cb-secret", body, hashlib.sha256).hexdigest()
    push = stepup.issue(db, "T4", "C2", "Mobile Banking", "Transfer", 90)
    body = json.dumps({"result": "denied"}).encode()
    with pytest.raises(PermissionError):
        stepup.callback(db, push["challenge_id"], body, "forged")
    assert stepup.callback(db, push["challenge_id"], body, sign(body))["status"] == "failed"
    tds = stepup.issue(db, "T5", "C2", "Web", "Card Not Present", 60)
    assert tds["method"] == "3ds"
    wait = json.dumps({"transStatus": "C"}).encode()
    assert stepup.callback(db, tds["challenge_id"], wait, sign(wait))["status"] == "pending"
    ok = json.dumps({"transStatus": "Y"}).encode()
    assert stepup.callback(db, tds["challenge_id"], ok, sign(ok))["status"] == "passed"


def test_log_provider_is_refused_in_production(db, monkeypatch):
    from bti.config import get_settings
    from bti.operations import stepup
    monkeypatch.setattr(get_settings(), "environment", "production")
    monkeypatch.setattr(get_settings(), "stepup_provider", "log")
    c = stepup.issue(db, "T6", "C3", "Internet Banking", "Transfer", 10)
    assert c["status"] == "send_failed"


# ── Cost model v2 ────────────────────────────────────────────────────────────

def test_customer_value_is_point_in_time_annualised_and_bounded():
    from bti.operations.cost_model import bounded_value, customer_values_from_pnl
    df = pd.DataFrame({"customer_id": ["A", "A", "A", "B"], "currency": "USD",
                       "transaction_date": ["2024-01-10", "2024-03-10", "2024-12-01", "2024-02-01"],
                       "net_revenue": [100.0, 50.0, 9999.0, 5.0]})
    v = customer_values_from_pnl(df, as_of=pd.Timestamp("2024-07-01")).set_index("customer_id")
    assert v.loc["A", "annual_value_usd"] == pytest.approx(150 * 12 / v.loc["A", "months_observed"], rel=1e-3)  # no Dec row
    assert bounded_value(10) == 75 and bounded_value(50_000) == 1200 and bounded_value(None) == 300


def test_v2_overrides_keep_capacity_prices_as_floors(monkeypatch):
    from bti.config import get_settings
    from bti.operations.cost_model import overrides_for, v2_overrides
    rates = {"push": {"abandonment_rate": 0.2, "catch_rate": 0.7}}
    ov = v2_overrides(1000, 900, "push", rates, capacity={"review_cost_usd": 290.0, "step_up_friction_usd": 26.7})
    assert ov["customer_annual_value_usd"] == 900 and ov["step_up_catch_rate"] == 0.7
    assert ov["step_up_friction_usd"] == 26.7 and ov["review_cost_usd"] == 290.0   # capacity floor wins here
    cheap = v2_overrides(100, 300, "push", rates)
    assert cheap["step_up_friction_usd"] == pytest.approx(0.01 + 0.2 * (100 * 0.01 + 0.02 * 300))
    monkeypatch.setattr(get_settings(), "cost_model_version", "v1")
    assert overrides_for(None, {}, 100, {"review_cost_usd": 1.0}) == {"review_cost_usd": 1.0}


def test_decision_fairness_separates_amount_from_group():
    from bti.governance.fairness import decision_fairness
    rng = np.random.default_rng(1)
    n = 9000
    group = np.repeat(["Wealthy", "Typical", "Singled"], n // 3)
    amount = np.where(group == "Wealthy", rng.gamma(2, 5000, n), rng.gamma(2, 300, n))
    iv = rng.random(n) < np.clip(amount / 20000, 0, 0.9)                  # intervention driven by amount only
    singled = group == "Singled"
    iv[singled] = iv[singled] | (rng.random(singled.sum()) < 0.1)        # plus a group effect
    r = decision_fairness(np.zeros(n, dtype=int), iv, {"segment": group}, amount)
    rows = {g["group"]: g for g in r["attributes"][0]["groups"]}
    assert rows["Wealthy"]["raw_ratio"] > 2 and rows["Wealthy"]["amount_standardised_ratio"] < 1.25
    assert [f["group"] for f in r["findings"]] == ["Singled"]
