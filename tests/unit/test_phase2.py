"""Phase 2: incumbent ingestion, parallel-run report, traffic split with fallback, reconciliation."""

import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from bti.database.models import AuditLog, Base, FraudLabel, IncumbentDecision, RoutedDecision, ScoreLog
from bti.parallel import incumbent, reconcile, report, traffic


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


# ── Ingestion ────────────────────────────────────────────────────────────────

def test_vendor_codes_are_normalised_and_unknown_codes_rejected(db, monkeypatch):
    from bti.config import get_settings
    monkeypatch.setattr(get_settings(), "incumbent_decision_map", {"R7": "REVIEW"})
    result = incumbent.ingest(db, [
        {"transaction_id": "T1", "decision": "accept", "score": "120"},
        {"transaction_id": "T2", "decision": "Refer", "executed_decision": "DECLINE"},
        {"transaction_id": "T3", "decision": "R7"},                          # bank-specific code
        {"transaction_id": "T4", "decision": "MAYBE"},                       # unknown: rejected, not guessed
        {"transaction_id": "", "decision": "DECLINE"},
    ], system="SAS")
    assert result["accepted"] == 3 and result["rejected"] == 2
    assert {r["reason"].split(" ")[0] for r in result["rejections"]} == {"unknown", "missing"}
    latest = incumbent.latest_incumbent(db).set_index("transaction_id")
    assert latest.loc["T1", "decision"] == "APPROVE" and latest.loc["T1", "score"] == 120
    assert latest.loc["T2", "executed_decision"] == "DECLINE" and latest.loc["T3", "decision"] == "REVIEW"


def test_latest_incumbent_record_wins_and_columns_can_be_mapped(db):
    rows = incumbent.frame_to_rows(pd.DataFrame({"TXN": ["T1"], "ACTION_CD": ["DECLINE"], "SAS_SCORE": ["910"]}),
                                   {"TXN": "transaction_id", "ACTION_CD": "decision", "SAS_SCORE": "score"})
    incumbent.ingest(db, rows)
    incumbent.ingest(db, [{"transaction_id": "T1", "decision": "APPROVE"}])       # correction arrives later
    latest = incumbent.latest_incumbent(db).set_index("transaction_id").loc["T1"]
    assert latest["decision"] == "APPROVE" and pd.isna(latest["score"])       # whole record replaced, not merged
    with pytest.raises(ValueError, match="transaction_id and decision"):
        incumbent.frame_to_rows(pd.DataFrame({"id": [1]}))


# ── Parallel-run comparison ──────────────────────────────────────────────────

def _frame(n=6000, seed=0, bti_quality=2.5, inc_quality=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(float)
    bti_p = 1 / (1 + np.exp(-(bti_quality * (y - 0.5) * 2 + rng.normal(0, 1, n) - 3)))
    inc_s = inc_quality * y * 2 + rng.normal(0, 1, n)
    inc = np.where(inc_s > np.quantile(inc_s, 0.92), "REVIEW", "APPROVE")
    bti = np.where(bti_p > np.quantile(bti_p, 0.92), "REVIEW", "APPROVE")
    return pd.DataFrame({"transaction_id": [f"T{i}" for i in range(n)], "bti_probability": bti_p,
                         "bti_decision": bti, "incumbent_decision": inc, "amount_usd": rng.gamma(2, 200, n),
                         "label": y})


def test_equal_intervention_rate_comparison_detects_a_better_system():
    r = report.compare_systems(_frame(), "SAS", bootstrap_reps=200)
    eq = r["at_equal_intervention_rate"]
    assert r["verdict"] == "bti_detects_more"
    assert eq["bti"]["interventions"] == eq["incumbent"]["interventions"]      # same number of interventions
    assert eq["tdr_difference"] > 0 and eq["tdr_difference_ci95"][0] > 0
    counts = r["operational"]["agreement_matrix"]["counts"]
    assert sum(sum(row.values()) for row in counts.values()) == r["paired_transactions"]


def test_equal_systems_are_not_declared_different():
    f = _frame(bti_quality=1.0, inc_quality=1.0, seed=3)
    f["incumbent_decision"] = f["bti_decision"]                               # identical decisions
    r = report.compare_systems(f, "SAS", bootstrap_reps=100)
    assert r["verdict"] == "no_significant_difference"
    assert r["fraud_overlap"]["bti_only"] == r["fraud_overlap"]["sas_only"] == 0


def test_outcomes_wait_for_matured_labels():
    f = _frame(n=2000)
    f.loc[f.index[:1900], "label"] = np.nan
    r = report.compare_systems(f, "SAS")
    assert r["status"] == "insufficient_labels" and "operational" in r


def test_weekly_report_joins_the_logs_and_flags_feed_gaps(db):
    now = datetime(2025, 6, 30, 12)
    for i in range(40):
        db.add(ScoreLog(transaction_id=f"T{i}", model_id="m", model_role="champion", is_shadow=False,
                        fraud_probability=0.9 if i < 5 else 0.01, decision="REVIEW" if i < 5 else "APPROVE",
                        amount_usd=100.0, latency_ms=20.0, scored_at=now - timedelta(days=1)))
    db.commit()
    incumbent.ingest(db, [{"transaction_id": f"T{i}", "decision": "APPROVE", "decided_at": now - timedelta(days=1)}
                          for i in range(20)])                                # only half the feed arrived
    result = report.run_weekly_report(db, now=now, notify=False)
    last = result["views"]["last_7_days"]
    assert last["coverage"] == {"bti_decisions": 40, "with_incumbent_decision": 20, "pairing_rate": 0.5}
    assert any("incumbent decision" in i["issue"] for i in result["issues"])
    assert db.query(AuditLog).filter(AuditLog.event_type == report.EVENT_TYPE).count() == 1


# ── Traffic split ────────────────────────────────────────────────────────────

def test_assignment_is_deterministic_and_matches_the_share():
    e = type("E", (), {"salt": "abc123", "bti_share": 0.10, "unit": "customer"})()
    arms = [traffic.assign_arm(e, f"C{i}", f"T{i}") for i in range(20000)]
    assert 0.09 < arms.count("bti") / len(arms) < 0.11
    assert all(traffic.assign_arm(e, "C7", f"T{i}") == traffic.assign_arm(e, "C7", "X") for i in range(50))


@pytest.fixture
def champion(monkeypatch):
    from bti.modeling import registry
    monkeypatch.setattr(registry, "model_for_role", lambda role: "bti-model-x" if role == "champion" else None)


def test_experiment_needs_cap_four_eyes_and_an_approved_champion(db, monkeypatch):
    from bti.modeling import registry
    with pytest.raises(traffic.ExperimentError, match="at most"):
        traffic.propose(db, "too-big", 0.5, "alice", "Pilot routing half of all traffic")
    e = traffic.propose(db, "pilot-1", 0.05, "alice", "Five percent customer-level pilot")
    with pytest.raises(traffic.ExperimentError, match="Four-eyes"):
        traffic.approve(db, e["id"], "Alice")
    monkeypatch.setattr(registry, "model_for_role", lambda role: None)
    with pytest.raises(traffic.ExperimentError, match="champion"):
        traffic.approve(db, e["id"], "bob")
    monkeypatch.setattr(registry, "model_for_role", lambda role: "bti-model-x" if role == "champion" else None)
    started = traffic.approve(db, e["id"], "bob")
    assert started["status"] == "running" and started["model_id"] == "bti-model-x"
    other = traffic.propose(db, "pilot-2", 0.05, "carol", "Second pilot proposed in parallel")
    with pytest.raises(traffic.ExperimentError, match="running"):
        traffic.approve(db, other["id"], "dave")
    assert traffic.stop(db, e["id"], "bob", "Pilot ended")["status"] == "stopped"


def _fake_bti(decision="DECLINE", delay=0.0, fail=False):
    def score(bind, txn):
        if delay:
            time.sleep(delay)
        if fail:
            raise RuntimeError("model unavailable")
        return {"decision": decision, "probability": 0.97, "model_id": "bti-model-x", "amount_usd": 50.0}
    return score


def _running(db, share=1.0):
    from bti.config import get_settings
    get_settings().parallel_max_bti_share = 1.0
    e = traffic.propose(db, f"split-{share}", share, "alice", "Routing test experiment")
    return traffic.approve(db, e["id"], "bob")


def test_router_uses_bti_in_its_arm_and_falls_back_on_timeout_or_error(db, champion, monkeypatch):
    txn = {"transaction_id": "T1", "customer_id": "C1", "transaction_amount": 50, "currency": "USD"}
    monkeypatch.setattr(traffic, "_score_in_own_session", _fake_bti())
    none = traffic.route(db, txn, "APPROVE")
    assert (none.arm, none.effective_decision, none.decided_by, none.bti_decision) == \
           ("none", "APPROVE", "INCUMBENT", "DECLINE")                     # no experiment: incumbent decides
    _running(db, share=1.0)
    bti = traffic.route(db, {**txn, "transaction_id": "T2"}, "APPROVE")
    assert (bti.arm, bti.effective_decision, bti.decided_by) == ("bti", "DECLINE", "BTI")
    monkeypatch.setattr(traffic, "_score_in_own_session", _fake_bti(delay=0.5))
    slow = traffic.route(db, {**txn, "transaction_id": "T3"}, "REFER", timeout_ms=50)
    assert (slow.effective_decision, slow.decided_by, slow.fallback_reason) == ("REVIEW", "INCUMBENT", "timeout")
    monkeypatch.setattr(traffic, "_score_in_own_session", _fake_bti(fail=True))
    err = traffic.route(db, {**txn, "transaction_id": "T4"})
    assert (err.effective_decision, err.fallback_reason, err.instruction) == \
           (None, "error", "apply the incumbent's decision")
    assert db.query(RoutedDecision).count() == 4
    from bti.config import get_settings
    get_settings().parallel_max_bti_share = 0.10


def test_experiment_report_checks_the_split_and_compares_arms(db, champion):
    from bti.config import get_settings
    get_settings().parallel_max_bti_share = 1.0
    e = traffic.approve(db, traffic.propose(db, "arms", 0.5, "alice", "Half and half for the test")["id"], "bob")
    get_settings().parallel_max_bti_share = 0.10
    rng = np.random.default_rng(1)
    old = datetime.utcnow() - timedelta(days=200)
    for i in range(4000):
        cust = f"C{i}"
        arm = traffic.assign_arm(type("E", (), {"salt": e["salt"], "bti_share": 0.5, "unit": "customer"})(), cust, "")
        fraud = rng.random() < 0.08
        caught = rng.random() < (0.9 if arm == "bti" else 0.6)
        decision = "REVIEW" if fraud and caught else "APPROVE"
        db.add(RoutedDecision(experiment_id=e["id"], transaction_id=f"T{i}", customer_id=cust, arm=arm,
                              effective_decision=decision, decided_by="BTI" if arm == "bti" else "INCUMBENT",
                              amount_usd=100.0, routed_at=old))
        if fraud:
            db.add(FraudLabel(transaction_id=f"T{i}", label=1, label_source="CHARGEBACK", event_at=old))
    db.commit()
    r = traffic.experiment_report(db, e["id"], bootstrap_reps=200)
    assert r["status"] == "ok" and not r["split"]["sample_ratio_mismatch"]
    assert r["arms"]["bti"]["tdr"] > r["arms"]["control"]["tdr"]
    assert r["verdict"] == "bti_lower_loss" and r["bti_minus_control_ci95"]["fraud_loss_bps"][1] < 0


# ── Reconciliation ───────────────────────────────────────────────────────────

def test_reconciliation_finds_each_kind_of_break(db):
    day = date(2025, 6, 1)
    at = datetime(2025, 6, 1, 10)
    routed = [("T1", "APPROVE", "APPROVE"), ("T2", "DECLINE", "APPROVE"), ("T3", "APPROVE", "APPROVE"),
              ("T4", "REVIEW", "REVIEW")]
    for tid, effective, sent in routed:
        db.add(RoutedDecision(transaction_id=tid, arm="bti", effective_decision=effective,
                              incumbent_decision=sent, decided_by="BTI", routed_at=at))
    db.add(RoutedDecision(transaction_id="T4", arm="bti", effective_decision="APPROVE", decided_by="BTI",
                          fallback_reason="timeout", routed_at=at))
    db.commit()
    incumbent.ingest(db, [
        {"transaction_id": "T1", "decision": "APPROVE", "executed_decision": "APPROVE", "decided_at": at},
        {"transaction_id": "T2", "decision": "APPROVE", "executed_decision": "APPROVE", "decided_at": at},  # bank ignored BTI
        {"transaction_id": "T4", "decision": "DECLINE", "decided_at": at},                                 # sent ≠ logged
        {"transaction_id": "T9", "decision": "APPROVE", "decided_at": at},                                 # BTI never saw
    ])
    r = reconcile.reconcile(db, day, notify=False)
    b = {k: v["sample"] for k, v in r["breaks"].items()}
    assert b["routed_without_incumbent_record"] == ["T3"]
    assert b["incumbent_without_bti"] == ["T9"]
    assert b["duplicate_routing"] == ["T4"]
    assert b["enforcement_mismatch"] == ["T2"]
    assert b["incumbent_decision_mismatch"] == ["T4"]       # any routing whose sent decision ≠ the log
    assert r["status"] == "breaks" and r["fallbacks"]["by_reason"] == {"timeout": 1}


# ── Capacity-constrained live decisions ──────────────────────────────────────

def test_live_decisions_respect_the_fitted_capacity_policy(tmp_path, monkeypatch):
    from bti.operations import capacity
    from bti.operations.decisioning import decide
    monkeypatch.setattr(capacity, "policy_dir", lambda: tmp_path)
    capacity.clear_cache()
    assert capacity.capacity_overrides("m1") is None
    (tmp_path / "m1.json").write_text('{"model_id": "m1", "history": [], "current": {"overrides": '
                                      '{"review_cost_usd": 300.0, "step_up_friction_usd": 30.0}}}')
    capacity.clear_cache()
    overrides = capacity.capacity_overrides("m1")
    # p=0.03 on a $2,000 transfer: worth an $8 review unconstrained, not at a $300 capacity shadow price
    assert decide(0.03, 2000, None, "Branch", "Transfer").action == "REVIEW"
    assert decide(0.03, 2000, None, "Branch", "Transfer", cost_overrides=overrides).action == "APPROVE"
    assert decide(0.95, 2000, None, "Branch", "Transfer", cost_overrides=overrides).action == "DECLINE"
    capacity.clear_cache()
