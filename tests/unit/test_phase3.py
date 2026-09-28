"""Phase 3: tamper-evident audit storage, validation workflow, benchmark findings, outcomes analysis."""

import os
import stat
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from bti.database.models import AuditLog, Base, FraudLabel, ScoreLog
from bti.governance import audit_chain, validation
from bti.modeling import registry


@pytest.fixture
def engine():
    e = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(e)
    return e


@pytest.fixture
def db(engine):
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _events(db, n=5, day=datetime(2026, 9, 1, 12)):
    for i in range(n):
        db.add(AuditLog(ts=day + timedelta(minutes=i), event_type="TEST", payload={"i": i, "v": [1.5, "x"]}))
    db.commit()


# ── Audit chain ──────────────────────────────────────────────────────────────

def test_chain_detects_alteration_removal_and_truncation(db, engine):
    _events(db, 6)
    assert audit_chain.verify_chain(db) == {**audit_chain.verify_chain(db), "status": "intact", "rows_checked": 6}
    with engine.begin() as c:
        c.execute(text("UPDATE audit_logs SET payload = '{\"i\": 99}' WHERE seq = 3"))
    db.expire_all()
    assert "altered" in audit_chain.verify_chain(db)["first_break"]["problem"]


def test_chain_detects_a_removed_row_and_a_truncated_tail(db, engine):
    _events(db, 6)
    with engine.begin() as c:
        c.execute(text("DELETE FROM audit_logs WHERE seq = 4"))
    db.expire_all()
    assert "sequence gap" in audit_chain.verify_chain(db)["first_break"]["problem"]


def test_truncating_the_tail_is_detected(db, engine):
    _events(db, 6)
    with engine.begin() as c:
        c.execute(text("DELETE FROM audit_logs WHERE seq = 6"))
    db.expire_all()
    r = audit_chain.verify_chain(db)
    assert r["status"] == "broken" and "chain head" in r["first_break"]["problem"]


def test_guards_make_sealed_rows_append_only_and_legacy_rows_can_be_sealed_once(db, engine):
    with engine.begin() as c:                                        # a row written before the chain existed
        c.execute(text("INSERT INTO audit_logs (ts, event_type, payload) VALUES ('2026-01-01 00:00:00', 'OLD', '{}')"))
    assert audit_chain.seal_legacy(db) == 1
    _events(db, 2)
    assert audit_chain.install_guards(engine).startswith("append-only guards installed")
    for sql in ("UPDATE audit_logs SET event_type = 'X' WHERE seq = 1", "DELETE FROM audit_logs WHERE seq = 2"):
        with pytest.raises(Exception, match="append-only"):
            with engine.begin() as c:
                c.execute(text(sql))
    assert audit_chain.verify_chain(db)["status"] == "intact"


def test_daily_archive_is_sealed_verified_and_never_overwritten(db, tmp_path, monkeypatch):
    from bti.config import get_settings
    monkeypatch.setattr(get_settings(), "audit_archive_dir", str(tmp_path))
    monkeypatch.setattr(get_settings(), "audit_retention_days", 1)            # retention is fixed when sealed
    _events(db, 4)
    first = audit_chain.archive_day(db, date(2026, 9, 1))
    assert first["status"] == "archived" and first["rows"] == 4
    data_file = next(tmp_path.rglob("*.jsonl"))
    assert not os.stat(data_file).st_mode & stat.S_IWUSR                     # read-only
    assert audit_chain.archive_day(db, date(2026, 9, 1))["status"] == "already_archived"
    assert audit_chain.verify_archive(db, date(2026, 9, 1))["status"] == "intact"
    os.chmod(data_file, stat.S_IWUSR | stat.S_IRUSR)
    data_file.write_text(data_file.read_text().replace('"TEST"', '"EDIT"', 1))
    assert audit_chain.verify_archive(db, date(2026, 9, 1))["status"] == "broken"
    assert audit_chain.retention_report(date(2026, 9, 10))["eligible_for_disposal"] == ["2026-09-01"]


# ── Validation workflow ──────────────────────────────────────────────────────

@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setenv("BTI_MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    registry.clear_cache()
    for mid, status in (("m-pass", "passed"), ("m-fail", "failed")):
        registry.save_model(mid, {"stub": True}, {"model_id": mid, "ownership": {"developer": "dev.one"},
                                                 "validation": {"status": status}, "model_family": "BTI v3",
                                                 "methodology": {}, "features": {}})
    registry.assign_role("m-pass", "challenger", "dev.one", "Challenger for the workflow test")
    yield
    registry.clear_cache()


def test_findings_need_evidence_and_independent_closure(db, models):
    f = validation.raise_finding(db, "m-pass", "Calibration drift", "high", "monitoring", "independent_validation",
                                 "ECE above tolerance on matured labels", "val.one", "dev.one")
    assert f["due_date"][:10] == (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")
    with pytest.raises(validation.ValidationError, match="evidence"):
        validation.update_finding(db, f["id"], "closed", "val.one", resolution="Recalibrated")
    with pytest.raises(validation.ValidationError, match="owner cannot close"):
        validation.update_finding(db, f["id"], "closed", "dev.one", resolution="Recalibrated", evidence={"ref": "x"})
    with pytest.raises(validation.ValidationError, match="other than the owner and the developer"):
        validation.update_finding(db, f["id"], "risk_accepted", "dev.one", resolution="Accept",
                                  accepted_until=datetime.utcnow() + timedelta(days=30))
    closed = validation.update_finding(db, f["id"], "closed", "val.one", resolution="Recalibrated on Q3 labels",
                                       evidence={"report": "outcomes-2026Q3"})
    assert closed["status"] == "closed"
    with pytest.raises(validation.ValidationError, match="already closed"):
        validation.update_finding(db, f["id"], "open", "val.one")
    assert db.query(AuditLog).filter(AuditLog.event_type.in_(["FINDING_RAISED", "FINDING_UPDATED"])).count() == 2


def test_signoff_is_independent_and_blocked_by_open_high_findings(db, models):
    with pytest.raises(validation.ValidationError, match="Independence"):
        validation.record_signoff(db, "m-pass", "Dev.One", "approve", "Self validation")
    with pytest.raises(validation.ValidationError, match="failed its automated"):
        validation.record_signoff(db, "m-fail", "val.one", "approve", "Validation of a failing model")
    high = validation.raise_finding(db, "m-pass", "Synthetic data only", "high", "data", "development",
                                    "Performance not established on bank data", "dev.one", "dev.one")
    with pytest.raises(validation.ValidationError, match="open high-severity"):
        validation.record_signoff(db, "m-pass", "val.one", "approve", "Full validation")
    assert not validation.champion_readiness(db, "m-pass")["ready"]
    validation.update_finding(db, high["id"], "risk_accepted", "committee.chair", resolution="Accepted for a "
                              "limited pilot", accepted_until=datetime.utcnow() + timedelta(days=90))
    with pytest.raises(validation.ValidationError, match="conditions"):
        validation.record_signoff(db, "m-pass", "val.one", "approve_with_conditions", "Full validation")
    validation.record_signoff(db, "m-pass", "val.one", "approve_with_conditions", "Full validation of m-pass",
                              validator_role="Model Risk", conditions="Pilot limited to 5% of UK traffic")
    r = validation.champion_readiness(db, "m-pass")
    assert r["ready"] and r["conditions"] == "Pilot limited to 5% of UK traffic"
    validation.assert_ready_for_champion(db, "m-pass")


def test_inventory_and_review_schedule(db, models):
    inv = {m["model_id"]: m for m in validation.sync_inventory(db)}
    assert inv["m-pass"]["lifecycle"] == "shadow_challenger" and inv["m-fail"]["lifecycle"] == "registered"
    sched = validation.review_schedule(db)
    assert [m["state"] for m in sched["models"]] == ["never_validated"]
    validation.record_signoff(db, "m-pass", "val.one", "approve", "Annual validation of m-pass")
    later = validation.review_schedule(db, now=datetime.utcnow() + timedelta(days=350))
    assert later["models"][0]["state"] == "due_soon"
    assert validation.review_schedule(db, now=datetime.utcnow() + timedelta(days=400))["models"][0]["state"] == "overdue"
    check = validation.run_governance_check(db, notify=False)
    assert check["issues"] == []


# ── Benchmark findings ───────────────────────────────────────────────────────

def test_benchmark_report_proposes_findings_from_its_evidence():
    from bti.governance.benchmarking import proposed_findings
    flat = {"plus_1sd": {"mean_abs_change": 0.0, "flips_at_10pct_threshold": 0.0},
            "minus_1sd": {"mean_abs_change": 0.0, "flips_at_10pct_threshold": 0.0}, "max_flip_rate": 0.0}
    report = {
        "benchmarks": {"model": {"pr_auc": 0.88, "pr_auc_ci95": [0.86, 0.90]},
                       "logistic_regression": {"pr_auc": 0.875}},
        "sensitivity": {"features": [{"feature": "amount_vs_hist_avg", **flat, "max_flip_rate": 0.78},
                                     {"feature": "cust_txn_count_1h", **flat}]},
        "stress": [{"scenario": "missing_profile_data", "description": "30% missing", "flags": ["detection 44% → 30%"]},
                   {"scenario": "fraud_prior_doubled", "observed_fraud_rate": 0.095, "mean_probability_weighted": 0.086}],
    }
    titles = [f["title"] for f in proposed_findings(report)]
    assert titles == ["Gain over the logistic-regression benchmark is within noise",
                      "Decisions concentrated on one input: amount_vs_hist_avg", "Velocity features carry no weight",
                      "Stress scenario degrades performance: missing_profile_data",
                      "Probabilities do not follow a shift in the fraud rate"]


# ── Outcomes analysis ────────────────────────────────────────────────────────

def _live_scores(db, model_id, n=6000, miscalibrate=1.0, biased_segment=None, seed=2):
    rng = np.random.default_rng(seed)
    start = datetime.utcnow() - timedelta(days=200)
    p = np.clip(rng.beta(0.5, 9, n), 0.0005, 0.999)
    y = rng.random(n) < np.clip(p * miscalibrate, 0, 1)
    for i in range(n):
        seg = ["Retail", "Premium", "Student"][i % 3]
        prob = float(p[i])
        if biased_segment and seg == biased_segment and not y[i] and rng.random() < 0.08:
            prob = 0.6                                                  # genuine customers in one segment pushed up
        db.add(ScoreLog(transaction_id=f"L{i}", model_id=model_id, model_role="champion", is_shadow=False,
                        fraud_probability=prob, decision="APPROVE", amount_usd=100.0,
                        monitoring_attributes={"customer_segment": seg, "country": "United Kingdom"},
                        scored_at=start + timedelta(minutes=20 * i)))
        if y[i]:
            db.add(FraudLabel(transaction_id=f"L{i}", label=1, label_source="CHARGEBACK", event_at=start))
    db.commit()


@pytest.fixture
def live_model():
    model_id = registry.model_for_role("challenger")
    if not model_id:
        pytest.skip("no registered model")
    return model_id


def test_outcomes_analysis_passes_a_calibrated_model(db, live_model):
    from bti.governance.outcomes import outcomes_analysis
    _live_scores(db, live_model)
    r = outcomes_analysis(db, maturity_days=90)
    m = r["models"][0]
    assert m["status"] == "ok" and m["fairness"]["status"] == "pass"
    assert not [f for f in m["flags"] if "Calibration" in f["title"]]
    assert len(m["calibration"]["table"]) == 10


def test_outcomes_analysis_raises_findings_for_drift_and_live_unfairness(db, live_model):
    from bti.governance.outcomes import run_quarterly
    from bti.database.models import ValidationFinding
    _live_scores(db, live_model, miscalibrate=2.5, biased_segment="Student")
    r = run_quarterly(db, now=datetime.utcnow() + timedelta(days=30), notify=False)
    titles = [f["title"] for f in r["models"][0]["flags"]]
    assert any("Calibration drift" in t for t in titles)
    assert any("customer_segment = Student" in t for t in titles)
    assert db.query(ValidationFinding).filter(ValidationFinding.source == "monitoring").count() == len(titles)
    again = run_quarterly(db, now=datetime.utcnow() + timedelta(days=30), notify=False)
    assert again["findings_raised"] == []                                   # no duplicate open findings
