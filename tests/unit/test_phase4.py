# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 4: case management feeding labels, bounded recalibration, continuous retraining."""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from bti.database.models import AuditLog, Base, FraudCase, FraudLabel, ScoreLog
from bti.modeling import registry
from bti.operations import cases
from bti.operations.feedback import latest_labels

CLEAN = Path("data/processed/banking_transactions_clean.csv")


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


@pytest.fixture
def temp_registry(tmp_path, monkeypatch):
    if not (Path("models/registry") / "index.json").exists():
        pytest.skip("no registered model")
    reg = tmp_path / "registry"
    shutil.copytree("models/registry", reg)
    monkeypatch.setenv("BTI_MODEL_REGISTRY_DIR", str(reg))
    registry.clear_cache()
    from bti.modeling import recalibration
    recalibration.clear_cache()
    yield reg
    registry.clear_cache()
    recalibration.clear_cache()


# ── Case management ──────────────────────────────────────────────────────────

def test_queue_rules():
    assert cases.choose_queue(0.9, 50) == "urgent"
    assert cases.choose_queue(0.2, 6000) == "urgent"                       # expected loss $1,200
    assert cases.choose_queue(0.05, 12000) == "high_value"
    assert cases.choose_queue(0.05, 300) == "standard"


def test_cases_are_idempotent_prioritised_and_dispositions_become_labels(db):
    a = cases.open_case(db, "T1", "live_review", fraud_probability=0.3, amount_usd=200)      # standard, $60
    assert cases.open_case(db, "T1", "live_review", fraud_probability=0.3, amount_usd=200)["id"] == a["id"]
    cases.open_case(db, "T2", "live_review", fraud_probability=0.2, amount_usd=900)           # standard, $180
    urgent = cases.open_case(db, "T3", "live_review", fraud_probability=0.95, amount_usd=40)
    assert [cases.assign_next(db, "ana")["transaction_id"] for _ in range(3)] == ["T3", "T2", "T1"]
    assert cases.assign_next(db, "ana") is None

    cases.dispose(db, urgent["id"], "confirmed_fraud", "ana", fraud_type="Account Takeover", loss_amount=40.0)
    cases.dispose(db, a["id"], "inconclusive", "ana")
    labels = latest_labels(db).set_index("transaction_id")
    assert labels.loc["T3", "label"] == 1 and labels.loc["T3", "label_source"] == "INVESTIGATOR_CONFIRMED"
    assert "T1" not in labels.index                                          # inconclusive: no label
    with pytest.raises(cases.CaseError, match="already closed"):
        cases.dispose(db, urgent["id"], "confirmed_genuine", "ana")


def test_clearing_a_high_value_case_needs_a_second_reviewer(db):
    big = cases.open_case(db, "T9", "live_review", fraud_probability=0.05, amount_usd=25000)
    assert big["queue"] == "urgent"                                          # expected loss $1,250
    with pytest.raises(cases.CaseError, match="Maker-checker"):
        cases.dispose(db, big["id"], "confirmed_genuine", "ana")
    with pytest.raises(cases.CaseError, match="Maker-checker"):
        cases.dispose(db, big["id"], "confirmed_genuine", "ana", checked_by="ANA")
    done = cases.dispose(db, big["id"], "confirmed_genuine", "ana", checked_by="ben", notes="Customer confirmed")
    assert done["label_written"] is True
    assert latest_labels(db).set_index("transaction_id").loc["T9", "label"] == 0


def test_sla_breaches_alert_once_and_queue_metrics(db):
    c = cases.open_case(db, "T5", "live_review", fraud_probability=0.9, amount_usd=100)
    later = datetime.utcnow() + timedelta(minutes=90)
    assert cases.check_sla(db, notify=False, now=later)["new_breaches"] == [c["id"]]
    assert cases.check_sla(db, notify=False, now=later)["new_breaches"] == []
    status = cases.queue_status(db, now=later)["queues"]["urgent"]
    assert status["open"] == 1 and status["open_breached"] == 1


# ── Bounded recalibration ────────────────────────────────────────────────────

def _matured_scores(db, model_id, factor, n=8000, seed=0):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(0.6, 12, n), 0.0005, 0.6)
    y = rng.random(n) < np.clip(p * factor, 0, 1)
    start = datetime.utcnow() - timedelta(days=175)
    for i in range(n):
        db.add(ScoreLog(transaction_id=f"R{i}", model_id=model_id, model_role="challenger", is_shadow=False,
                        fraud_probability=float(p[i]), model_probability=float(p[i]), decision="APPROVE",
                        amount_usd=100.0, scored_at=start + timedelta(minutes=10 * i)))
        if y[i]:
            db.add(FraudLabel(transaction_id=f"R{i}", label=1, label_source="CHARGEBACK", event_at=start))
    db.commit()


def test_recalibration_accepts_a_bounded_shift_and_the_scorer_applies_it(db, temp_registry):
    from bti.modeling import recalibration
    model_id = registry.model_for_role("challenger")
    _matured_scores(db, model_id, factor=1.4)
    r = recalibration.recalibrate(db, model_id, window_days=90)
    assert r["status"] == "accepted" and r["applied"] and r["version"] == 1, r
    assert r["holdout_ece"]["candidate"] < r["holdout_ece"]["no_overlay"]
    assert 0.1 < r["candidate"]["beta"] < 1.0
    assert recalibration.apply_overlay(model_id, 0.02) > 0.02                # probabilities raised, order kept
    assert registry.notes_for(model_id)[-1]["subject"] == "Minor change: recalibration"
    assert db.query(AuditLog).filter(AuditLog.event_type == "MODEL_RECALIBRATED").count() == 1
    back = recalibration.rollback(db, model_id, "risk.officer", "Rolled back for the test")
    assert back["active_version"] is None and recalibration.apply_overlay(model_id, 0.02) == 0.02


def test_recalibration_refuses_shifts_that_need_retraining_or_lack_labels(db, temp_registry):
    from bti.modeling import recalibration
    model_id = registry.model_for_role("challenger")
    assert recalibration.recalibrate(db, model_id)["status"] == "insufficient_labels"
    _matured_scores(db, model_id, factor=6.0, seed=1)
    r = recalibration.recalibrate(db, model_id, window_days=90)
    assert r["status"] == "rejected" and not r["applied"]
    assert any("needs retraining" in reason for reason in r["reasons"])
    assert recalibration.load_overlay(model_id) is None


# ── Continuous retraining ────────────────────────────────────────────────────

def test_retraining_triggers(db):
    from bti.modeling.retrain import retraining_triggers
    first = retraining_triggers(db)
    assert first["should_run"] and first["reasons"] == ["no previous retraining"]
    db.add(AuditLog(ts=datetime.utcnow() - timedelta(days=3), event_type="RETRAIN_RUN", payload={}))
    db.add(AuditLog(ts=datetime.utcnow() - timedelta(days=1), event_type="DRIFT_CHECK", payload={"status": "escalate"}))
    db.commit()
    soon = retraining_triggers(db)
    assert soon["too_soon"] and not soon["should_run"] and any("drift" in r for r in soon["reasons"])
    later = retraining_triggers(db, now=datetime.utcnow() + timedelta(days=5))
    assert later["should_run"]


def test_training_extract_applies_labels_and_drops_immature_rows(db, tmp_path):
    if not CLEAN.exists():
        pytest.skip("processed data not available")
    from bti.modeling.retrain import build_training_extract
    df = pd.read_csv(CLEAN, low_memory=False).sample(4000, random_state=2)
    src = tmp_path / "extract.csv"
    df.to_csv(src, index=False)
    genuine = df[df["fraud_flag"] == 0]["transaction_id"].iloc[0]
    fraud = df[df["fraud_flag"] == 1]["transaction_id"].iloc[0]
    db.add_all([FraudLabel(transaction_id=str(genuine), label=1, label_source="CHARGEBACK", event_at=datetime.utcnow()),
                FraudLabel(transaction_id=str(fraud), label=0, label_source="INVESTIGATOR_CLEARED",
                           event_at=datetime.utcnow())])
    db.commit()
    out = build_training_extract(db, src, maturity_days=30, out_dir=tmp_path)
    assert out["labels_applied"] == 2 and out["labels_changed_flag"] == 2
    assert 0 < out["dropped_immature"] < 4000
    extract = pd.read_csv(out["path"]).set_index("transaction_id")
    assert extract.loc[str(genuine), "fraud_flag"] == 1


def test_retraining_produces_a_shadow_challenger_at_most(db, temp_registry, tmp_path):
    if not CLEAN.exists():
        pytest.skip("processed data not available")
    from bti.config import get_settings
    from bti.modeling.retrain import run_retraining
    src = tmp_path / "sample.csv"
    pd.read_csv(CLEAN, low_memory=False).sample(15000, random_state=5).to_csv(src, index=False)
    before = registry.read_index()
    r = run_retraining(db, force=True, notify=False, source_path=src, algorithms=["hgb"],
                       feature_sets=["core-relative-nb"], extract_dir=tmp_path)
    assert r["status"] == "retrained" and r["decision"]["outcome"] in (
        "new_challenger", "incumbent_retained", "no_eligible_candidate")
    after = registry.read_index()
    assert after["champion"] == before["champion"]                         # never promotes
    import json
    report = json.loads(Path(r["tournament_report"]).read_text())
    assert any(e.get("re_evaluated_on_new_data") for e in report["entrants"] if e["incumbent"])
    assert db.query(AuditLog).filter(AuditLog.event_type == "RETRAIN_RUN").count() == 1
    assert get_settings().retrain_min_interval_days == 7
