"""
Phase 0 guarantees:
  * /score, /score/explain, /score/upload and /sas/enrich run on the v3 model
  * label-derived inputs (risk_score, fraud_loss) can no longer move a score
  * a provisional model never auto-declines, on any endpoint
  * every endpoint writes the score log that monitoring reads
  * scheduled drift checks run, alert, and are recorded
  * service metrics report uptime, errors and latency
"""

import io
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("BTI_API_KEY", "test-api-key")

from bti.database.connection import get_db
from bti.database.models import Base, ScoreLog, Transaction
from bti.modeling import registry

SOURCE_REGISTRY = Path("models/registry")
pytestmark = pytest.mark.skipif(not (SOURCE_REGISTRY / "index.json").exists(),
                                reason="No v3 model registered — run python -m bti.modeling.train")

ENGINE = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
Session = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
KEY = {"X-API-Key": os.environ["BTI_API_KEY"]}


def _db():
    db = Session()
    try:
        yield db
    finally:
        db.close()


def _legacy_txn(i, **kw):
    base = {"transaction_id": f"P0-{i:04d}", "customer_id": "CUST-P0", "transaction_date": "2024-09-20",
            "transaction_time": "14:00:00", "transaction_amount": 120.0, "currency": "USD", "country": "US",
            "channel": "Mobile Banking", "transaction_type": "Transfer", "merchant_name": "Target",
            "merchant_category": "Retail / General Merchandise", "device_id": "DEV-P0", "ip_location": "8.8.4.4",
            "historical_average_transaction_amount": 150.0, "account_balance_before": 4000.0,
            "failed_attempt_count": 0, "login_attempts": 1}
    return {**base, **kw}


ATTACK = dict(transaction_amount=9000.0, device_id="DEV-UNSEEN", ip_location="45.9.9.9", login_attempts=8,
              transaction_time="03:05:00")


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
    db = Session()
    db.add_all([Transaction(transaction_id=f"P0-HIST-{d}", customer_id="CUST-P0", account_id="ACC-P0",
                            transaction_date=datetime(2024, 9, d), transaction_time="12:00:00",
                            transaction_amount=140.0, currency="USD", device_id="DEV-P0", ip_location="8.8.4.4",
                            merchant_name="Target", channel="Mobile Banking") for d in range(1, 15)])
    db.commit()
    db.close()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=ENGINE)
    if old is None:
        os.environ.pop("BTI_MODEL_REGISTRY_DIR", None)
    else:
        os.environ["BTI_MODEL_REGISTRY_DIR"] = old
    registry.clear_cache()


def _v3_model_id():
    return registry.model_for_role("challenger")


def _live_log_count():
    db = Session()
    try:
        return db.query(ScoreLog).filter(ScoreLog.is_shadow.is_(False)).count()
    finally:
        db.close()


class TestLegacyScoreEndpoints:
    def test_score_runs_on_v3(self, client):
        before = _live_log_count()
        body = client.post("/api/v1/score/", json=_legacy_txn(1)).json()
        assert body["model_version"] == _v3_model_id()
        assert body["ml_lr_proba"] is None and body["ml_rf_proba"] is None and body["ml_iso_score"] is None
        assert body["decision"] == "APPROVE" and body["model_provisional"] is True
        assert body["final_risk_score"] == pytest.approx(body["fraud_probability"] * 100, abs=0.01)
        assert body["jurisdiction"] == "US"
        assert _live_log_count() == before + 1

    def test_label_derived_inputs_cannot_move_the_score(self, client):
        low = client.post("/api/v1/score/", json=_legacy_txn(2, risk_score=5, fraud_loss=0)).json()
        high = client.post("/api/v1/score/", json=_legacy_txn(2, risk_score=95, fraud_loss=5000,
                                                               chargeback_loss=5000)).json()
        assert low["fraud_probability"] == high["fraud_probability"]
        assert low["decision"] == high["decision"]
        assert low["rules_fired"] == high["rules_fired"]

    def test_provisional_model_never_blocks(self, client):
        body = client.post("/api/v1/score/", json=_legacy_txn(3, **ATTACK)).json()
        assert body["fraud_probability"] > 0.5
        assert body["decision"] != "DECLINE"
        assert not body["recommendation"].startswith("BLOCK")
        assert body["reason_codes"]

    def test_explain_matches_score_and_reports_drivers(self, client):
        txn = _legacy_txn(4, **ATTACK)
        scored = client.post("/api/v1/score/", json=txn).json()
        explained = client.post("/api/v1/score/explain", json=txn).json()
        assert explained["fraud_probability"] == scored["fraud_probability"]
        drivers = explained["explanation"]["top_drivers"]
        assert len(drivers) == 8
        assert sum(d["impact_pct"] for d in drivers) <= 100.5
        assert 0 < explained["explanation"]["base_probability"] < 1
        assert "probability of fraud" in explained["explanation"]["narrative"]

    def test_upload_scores_every_row(self, client):
        csv = ("transaction_id,customer_id,transaction_amount,transaction_date,currency,country,device_id\n"
               "UP-1,CUST-P0,120,2024-09-20,USD,US,DEV-P0\nUP-2,CUST-NEW,9000,2024-09-20,GBP,GB,DEV-X\n")
        body = client.post("/api/v1/score/upload", files={"file": ("t.csv", io.BytesIO(csv.encode()), "text/csv")}).json()
        assert body["scored"] == 2 and body["failed"] == 0
        assert {"decision", "fraud_probability", "reason_codes"} <= set(body["results"][0])

    def test_model_info_reports_v3(self, client):
        info = client.get("/api/v1/score/model-info").json()
        assert info["model_id"] == _v3_model_id() and info["validation_status"] == "passed"


class TestSasEnrichment:
    def test_enrich_runs_on_v3_with_decision(self, client):
        before = _live_log_count()
        r = client.post("/api/v1/sas/enrich", json={"transaction": {**_legacy_txn(10, **ATTACK),
                                                                    "sas_risk_score": 12}})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_version"] == _v3_model_id()
        assert body["bti_decision"] in ("STEP_UP", "REVIEW")
        assert body["bti_recommended_action"] == "HOLD"
        assert body["bti_model_provisional"] is True
        assert 0 < len(body["top_drivers"]) <= 5 and body["reason_codes"]
        assert "provisional" in body["notes_for_sas"]
        assert _live_log_count() == before + 1

    def test_sas_score_is_not_used(self, client):
        a = client.post("/api/v1/sas/enrich", json={"transaction": {**_legacy_txn(11), "sas_risk_score": 1}}).json()
        b = client.post("/api/v1/sas/enrich", json={"transaction": {**_legacy_txn(11), "sas_risk_score": 99}}).json()
        assert a["bti_fraud_probability"] == b["bti_fraud_probability"]

    def test_ieee_style_batch_is_not_dropped(self, client):
        txns = [{"TransactionID": 9001 + i, "TransactionAmt": 50.0 + i, "device_id": "DEV-SHARED",
                 "customer_id": f"C-{i}"} for i in range(3)]
        body = client.post("/api/v1/sas/enrich/batch", json={"transactions": txns}).json()
        assert body["total_transactions"] == 3
        assert {r["transaction_id"] for r in body["results"]} == {"9001", "9002", "9003"}

    def test_health_and_schema(self, client):
        health = client.get("/api/v1/sas/health").json()
        assert health["scorer_ready"] is True and health["model"]["model_id"] == _v3_model_id()
        schema = client.get("/api/v1/sas/schema").json()
        assert "bti_decision" in schema["fields"] and schema["version"] == "4.1.0"


class TestMonitoring:
    def test_drift_run_requires_key_and_is_recorded(self, client):
        assert client.post("/api/v1/governance/drift/run").status_code == 401
        run = client.post("/api/v1/governance/drift/run", headers=KEY).json()
        assert run["status"] in ("insufficient_data", "stable", "investigate", "escalate")
        history = client.get("/api/v1/governance/drift/history").json()["runs"]
        assert history and history[0]["checked_at"] == run["checked_at"]

    def test_shifted_traffic_escalates_and_alerts(self, client):
        model_id = _v3_model_id()
        db = Session()
        now = datetime.utcnow()
        db.add_all([ScoreLog(transaction_id=f"DRIFT-{i}", model_id=model_id, model_role="challenger",
                             is_shadow=False, fraud_probability=0.97, score=970, decision="REVIEW",
                             amount_usd=9e6, features={"amount_usd": 9e6, "login_attempts": 10.0,
                                                       "amount_vs_hist_avg": 50.0, "channel": "USSD"},
                             latency_ms=70.0, scored_at=now - timedelta(hours=1)) for i in range(150)])
        db.commit()
        db.close()
        run = client.post("/api/v1/governance/drift/run?window_days=1", headers=KEY).json()
        assert run["status"] == "escalate"
        live = next(c for c in run["checks"] if c["traffic"] == "live")
        assert live["score_status"] == "escalate" and live["features_flagged"]
        assert run["alert"] is not None          # dispatched (channels empty when none configured)

    def test_schedule_status(self, client):
        status = client.get("/api/v1/governance/monitoring/schedule").json()
        assert status["cron_utc"] and "next_run" in status
        if status["enabled"]:
            assert status["running"] and status["next_run"]

    def test_service_metrics(self, client):
        body = client.get("/api/v1/operations/service-metrics?hours=24").json()
        assert body["uptime_seconds"] > 0
        assert body["http"]["score"]["requests"] >= 1
        assert body["http"]["score"]["latency_ms"]["p50"] > 0
        models = {m["model_id"]: m for m in body["scoring"]["models"] if not m["shadow"]}
        assert models[_v3_model_id()]["scored"] >= 1
        assert body["scoring"]["sla_ms"] > 0
