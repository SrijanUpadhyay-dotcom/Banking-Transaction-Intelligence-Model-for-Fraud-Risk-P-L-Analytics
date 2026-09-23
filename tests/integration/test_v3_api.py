"""
End-to-end tests of the governed v3 lifecycle through the API:
score → shadow → labels → KPIs → promotion (four-eyes) → champion/challenger.
Runs against a temporary copy of the model registry and an in-memory database.
"""

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


def _txn(i, **kw):
    base = {"transaction_id": f"V3-{i:04d}", "customer_id": "CUST-V3", "transaction_date": "2024-09-20",
            "transaction_time": "14:00:00", "transaction_amount": 120.0, "currency": "GBP", "country": "United Kingdom",
            "channel": "Mobile Banking", "transaction_type": "Transfer", "merchant_category": "Retail / General Merchandise",
            "merchant_name": "Tesco", "authorization_method": "Biometric", "device_id": "DEV-HOME",
            "ip_location": "81.2.69.160", "historical_average_transaction_amount": 150.0,
            "account_balance_before": 4000.0, "failed_attempt_count": 0, "login_attempts": 1}
    return {**base, **kw}


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
    db.add_all([Transaction(transaction_id=f"HIST-{d}", customer_id="CUST-V3", account_id="ACC-V3",
                            transaction_date=datetime(2024, 9, d), transaction_time="12:00:00",
                            transaction_amount=140.0, currency="GBP", device_id="DEV-HOME", ip_location="81.2.69.160",
                            merchant_name="Tesco", channel="Mobile Banking") for d in range(1, 15)])
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


def _passed_and_failed():
    passed = registry.model_for_role("challenger") or registry.model_for_role("champion")
    failed = next(m["model_id"] for m in registry.read_index()["models"] if m["validation_status"] == "failed")
    return passed, failed


class TestScoring:
    def test_typical_transaction_is_low_risk_with_history(self, client):
        r = client.post("/api/v1/v3/score", json=_txn(1))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"]["provisional"] is True          # no champion approved yet
        assert body["history_rows_used"] >= 10
        assert body["jurisdiction"]["iso2"] == "GB"
        assert body["decision"]["action"] == "APPROVE"

    def test_account_takeover_pattern_is_flagged_and_explained(self, client):
        r = client.post("/api/v1/v3/score", json=_txn(2, transaction_amount=2400.0, device_id="DEV-NEW-9",
                                                         ip_location="45.1.1.1", login_attempts=7,
                                                         transaction_time="03:10:00"))
        body = r.json()
        assert body["fraud_probability"] > 0.5
        assert body["decision"]["action"] != "DECLINE"      # provisional model may not auto-decline
        codes = {rc["code"] for rc in body["reason_codes"]}
        assert codes & {"AMT_VS_HISTORY", "NEW_DEVICE", "LOGIN_ANOMALY"}
        assert all(rc["customer_text"] for rc in body["reason_codes"])

    def test_unsupported_currency_rejected(self, client):
        assert client.post("/api/v1/v3/score", json=_txn(3, currency="XYZ")).status_code == 422

    def test_batch_and_score_log(self, client):
        r = client.post("/api/v1/v3/score/batch", json=[_txn(10 + i) for i in range(3)])
        assert r.status_code == 200 and r.json()["count"] == 3
        db = Session()
        assert db.query(ScoreLog).filter(ScoreLog.is_shadow.is_(False)).count() >= 4
        db.close()

    def test_model_endpoint(self, client):
        body = client.get("/api/v1/v3/model").json()
        assert body["role"] == "challenger" and body["validation_status"] == "passed"
        assert body["out_of_time"]["roc_auc"] > 0.9


class TestGovernance:
    def test_inventory_and_documentation(self, client):
        assert len(client.get("/api/v1/governance/models").json()["models"]) >= 2
        passed, _ = _passed_and_failed()
        doc = client.get(f"/api/v1/governance/models/{passed}/documentation")
        assert doc.status_code == 200
        assert "Validation gates" in doc.text and "Excluded source fields" in doc.text
        card = client.get(f"/api/v1/governance/models/{passed}").json()
        assert "baseline" not in card["monitoring"]

    def test_promotion_requires_api_key(self, client):
        passed, _ = _passed_and_failed()
        r = client.post(f"/api/v1/governance/models/{passed}/promote",
                        json={"role": "champion", "approver": "risk.officer", "rationale": "Independent validation"})
        assert r.status_code == 401

    def test_failed_model_cannot_be_champion(self, client):
        _, failed = _passed_and_failed()
        r = client.post(f"/api/v1/governance/models/{failed}/promote", headers=KEY,
                        json={"role": "champion", "approver": "risk.officer", "rationale": "Should be refused"})
        assert r.status_code == 409

    def test_four_eyes_then_promotion_and_shadow(self, client):
        passed, failed = _passed_and_failed()
        developer = registry.load_card(passed)["ownership"]["developer"]
        self_approved = client.post(f"/api/v1/governance/models/{passed}/promote", headers=KEY,
                                    json={"role": "champion", "approver": developer,
                                          "rationale": "Developer approving own model"})
        assert self_approved.status_code == 409
        ok = client.post(f"/api/v1/governance/models/{passed}/promote", headers=KEY,
                         json={"role": "champion", "approver": "risk.officer",
                               "rationale": "Independent validation completed"})
        assert ok.status_code == 200
        shadow = client.post(f"/api/v1/governance/models/{failed}/promote", headers=KEY,
                             json={"role": "challenger", "approver": "risk.officer",
                                   "rationale": "Run superseded model in shadow for comparison"})
        assert shadow.status_code == 200
        body = client.post("/api/v1/v3/score", json=_txn(20)).json()
        assert body["model"] == {"model_id": passed, "role": "champion", "provisional": False}
        assert body["shadow"]["model_id"] == failed

    def test_jurisdictions_and_tra(self, client):
        assert len(client.get("/api/v1/governance/jurisdictions").json()["jurisdictions"]) == 8
        assert client.get("/api/v1/governance/jurisdictions/india").json()["iso2"] == "IN"
        assert client.get("/api/v1/governance/jurisdictions/atlantis").status_code == 404
        tra = client.post("/api/v1/governance/psd2/tra-eligibility",
                          json={"fraud_value_eur": 5, "total_value_eur": 100_000}).json()
        assert tra["max_exemption_threshold_eur"] == 500
        tra = client.post("/api/v1/governance/psd2/tra-eligibility",
                          json={"fraud_value_eur": 200, "total_value_eur": 100_000}).json()
        assert tra["max_exemption_threshold_eur"] == 0

    def test_drift_needs_minimum_sample(self, client):
        assert client.get("/api/v1/governance/drift").json()["status"] == "insufficient_data"

    def test_leakage_audit_finds_risk_score(self, client):
        leaks = {r["column"] for r in client.get("/api/v1/governance/leakage-audit").json()["suspected_leaks"]}
        assert "risk_score" in leaks


class TestOperations:
    def test_label_validation(self, client):
        bad_source = {"labels": [{"transaction_id": "V3-0002", "label_source": "HUNCH"}]}
        assert client.post("/api/v1/operations/labels", headers=KEY, json=bad_source).status_code == 422
        contradiction = {"labels": [{"transaction_id": "V3-0002", "label_source": "CHARGEBACK", "label": 0}]}
        assert client.post("/api/v1/operations/labels", headers=KEY, json=contradiction).status_code == 422
        assert client.post("/api/v1/operations/labels", json={"labels": []}).status_code == 401

    def test_feedback_loop_feeds_kpis(self, client):
        r = client.post("/api/v1/operations/labels", headers=KEY, json={"labels": [
            {"transaction_id": "V3-0002", "label_source": "INVESTIGATOR_CONFIRMED", "fraud_type": "Account Takeover"},
            {"transaction_id": "V3-0001", "label_source": "CUSTOMER_CONFIRMED_GENUINE"},
        ]})
        assert r.status_code == 200 and r.json() == {"recorded": 2, "fraud": 1, "genuine": 1}
        kpis = client.get("/api/v1/operations/kpis").json()["overall"]
        assert kpis["labelled"] >= 2 and kpis["fraud_count"] == 1
        assert kpis["tdr"] == 1.0
        status = client.get("/api/v1/operations/labels/status").json()
        assert status["transactions_with_explicit_label"] == 2

    def test_champion_challenger_pairs_shadow_scores(self, client):
        cc = client.get("/api/v1/operations/champion-challenger").json()
        assert cc["status"] == "ok" and cc["paired_transactions"] >= 1

    def test_model_agnostic_decision(self, client):
        d = client.post("/api/v1/operations/decide", json={"fraud_probability": 0.95, "amount_usd": 40_000,
                                                           "country": "DE", "channel": "Branch"}).json()
        assert d["action"] == "DECLINE" and d["human_review_route"] is True
        low = client.post("/api/v1/operations/decide", json={"fraud_probability": 0.3, "amount_usd": 40_000,
                                                             "country": "US", "channel": "Branch"}).json()
        assert low["action"] != "DECLINE"

    def test_policy_backtest_beats_single_threshold(self, client):
        r = client.post("/api/v1/operations/policy/backtest", json={"max_review_rate": 0.01})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["saving_vs_single_threshold_usd"] > 0
        assert body["expected_cost_policy"]["false_decline_rate"] < 0.01
        assert body["expected_cost_policy"]["cases_for_review"] <= 0.015 * body["n"]
