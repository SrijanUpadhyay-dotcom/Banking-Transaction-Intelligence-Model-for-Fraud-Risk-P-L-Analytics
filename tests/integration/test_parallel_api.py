"""
Parallel run through the API with the real scoring model: incumbent feed → router → report → reconciliation.
Runs against a temporary copy of the model registry and an in-memory database.
"""

import io
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("BTI_API_KEY", "test-api-key")

from bti.database.connection import get_db
from bti.database.models import Base, RoutedDecision, ScoreLog
from bti.modeling import registry

SOURCE_REGISTRY = Path("models/registry")
pytestmark = pytest.mark.skipif(not (SOURCE_REGISTRY / "index.json").exists(), reason="No v3 model registered")

ENGINE = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
Session = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
KEY = {"X-API-Key": os.environ["BTI_API_KEY"]}
TODAY = datetime.utcnow().strftime("%Y-%m-%d")


def _db():
    db = Session()
    try:
        yield db
    finally:
        db.close()


def _txn(i):
    return {"transaction_id": f"P-{i:04d}", "customer_id": f"CUST-{i % 7}", "transaction_date": TODAY,
            "transaction_time": "11:00:00", "transaction_amount": 80.0 + i, "currency": "GBP",
            "country": "United Kingdom", "channel": "Mobile Banking", "transaction_type": "Transfer",
            "merchant_category": "Retail / General Merchandise", "merchant_name": "Tesco",
            "authorization_method": "Biometric", "device_id": "DEV-1", "ip_location": "81.2.69.160"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    reg = tmp_path_factory.mktemp("registry") / "registry"
    shutil.copytree(SOURCE_REGISTRY, reg)
    old = os.environ.get("BTI_MODEL_REGISTRY_DIR")
    os.environ["BTI_MODEL_REGISTRY_DIR"] = str(reg)
    registry.clear_cache()
    from api.main import app
    app.dependency_overrides[get_db] = _db
    Base.metadata.create_all(bind=ENGINE)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=ENGINE)
    if old is None:
        os.environ.pop("BTI_MODEL_REGISTRY_DIR", None)
    else:
        os.environ["BTI_MODEL_REGISTRY_DIR"] = old
    registry.clear_cache()


def test_incumbent_feed_accepts_json_and_files(client):
    assert client.post("/api/v1/parallel/incumbent/decisions",
                       json={"records": [{"transaction_id": "X", "decision": "ACCEPT"}]}).status_code == 401
    r = client.post("/api/v1/parallel/incumbent/decisions", headers=KEY, json={"system": "SAS", "records": [
        {"transaction_id": f"P-{i:04d}", "decision": "ACCEPT", "score": 100 + i, "latency_ms": 35}
        for i in range(10)] + [{"transaction_id": "P-9999", "decision": "WHATEVER"}]})
    assert r.status_code == 200 and r.json()["accepted"] == 10 and r.json()["rejected"] == 1
    csv = "TXN_ID,ACTION_CD,SAS_SCORE\nP-0010,REFER,640\nP-0011,DECLINE,910\n"
    up = client.post("/api/v1/parallel/incumbent/upload", headers=KEY,
                     files={"file": ("sas.csv", io.BytesIO(csv.encode()), "text/csv")},
                     data={"column_map": '{"TXN_ID": "transaction_id", "ACTION_CD": "decision", "SAS_SCORE": "score"}'})
    assert up.status_code == 200 and up.json()["decisions"] == {"REVIEW": 1, "DECLINE": 1}


def test_router_scores_with_bti_and_keeps_the_incumbent_in_charge_without_an_experiment(client):
    for i in range(12):
        r = client.post("/api/v1/parallel/decide", headers=KEY,
                        json={"transaction": _txn(i), "incumbent_decision": "ACCEPT", "timeout_ms": 5000})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["arm"] == "none" and body["decided_by"] == "INCUMBENT" and body["effective_decision"] == "APPROVE"
        assert body["bti_decision"] in ("APPROVE", "STEP_UP", "REVIEW", "DECLINE") and body["fallback_reason"] is None
    db = Session()
    assert db.query(RoutedDecision).count() == 12
    assert db.query(ScoreLog).filter(ScoreLog.is_shadow.is_(False)).count() == 12     # BTI logged every score
    db.close()


def test_experiment_cannot_start_without_an_approved_champion(client):
    p = client.post("/api/v1/parallel/experiments", headers=KEY, json={
        "name": "uk-pilot", "bti_share": 0.05, "proposed_by": "Srijan Upadhyay",
        "rationale": "Five percent customer-level pilot in the UK"})
    assert p.status_code == 201
    assert client.post("/api/v1/parallel/experiments", headers=KEY, json={
        "name": "greedy", "bti_share": 0.5, "proposed_by": "x", "rationale": "Half of all traffic"}).status_code == 422
    a = client.post(f"/api/v1/parallel/experiments/{p.json()['id']}/approve", headers=KEY,
                    json={"approver": "Head of Fraud Strategy"})
    assert a.status_code == 409 and "champion" in a.json()["detail"]


def test_report_status_and_reconciliation(client):
    status = client.get("/api/v1/parallel/incumbent/status").json()
    assert status["matched"] >= 10
    rep = client.get("/api/v1/parallel/report", params={"maturity_days": 90}).json()
    assert rep["paired_transactions"] >= 10 and rep["status"] == "insufficient_labels"
    assert rep["latency"]["incumbent"]["p50_ms"] == 35.0
    run = client.post("/api/v1/parallel/report/run", headers=KEY, params={"notify": False})
    assert run.status_code == 200 and set(run.json()["views"]) == {"last_7_days", "matured_cohort", "to_date"}
    rec = client.post("/api/v1/parallel/reconcile/run", headers=KEY, params={"day": TODAY, "notify": False}).json()
    assert rec["routed_transactions"] == 12
    assert rec["breaks"]["routed_without_incumbent_record"]["count"] == 0
    assert client.get("/api/v1/parallel/reconcile/history").json()["runs"][0]["day"] == TODAY
