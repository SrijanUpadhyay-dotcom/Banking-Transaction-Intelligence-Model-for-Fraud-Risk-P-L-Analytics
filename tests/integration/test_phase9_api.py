# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 9 scams API: mule-alert review workflow, model summary, and the read-only scam assessment."""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("BTI_API_KEY", "test-api-key")

from bti.database.connection import get_db
from bti.database.models import AuditLog, Base, MuleAlert

ENGINE = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
Session = sessionmaker(bind=ENGINE)
KEY = {"X-API-Key": os.environ["BTI_API_KEY"]}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    Base.metadata.create_all(ENGINE)

    def _db():
        db = Session()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = _db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


def test_mule_alert_review_needs_key_and_evidence_and_is_final(client):
    db = Session()
    db.add(MuleAlert(customer_id="CUST-X", snapshot_date="2026-10-05", model_id="m", score=0.97, rank=1,
                     reasons=[{"feature": "fast_out_share", "value": 0.92}]))
    db.commit()
    alert = db.query(MuleAlert).first().id
    assert [a["customer_id"] for a in client.get("/api/v1/scams/mule-alerts").json()] == ["CUST-X"]
    url = f"/api/v1/scams/mule-alerts/{alert}/review"
    body = {"status": "confirmed_mule", "reviewed_by": "A. Analyst", "note": "Inbound from 14 senders, all out by ATM"}
    assert client.post(url, json=body).status_code in (401, 403)
    assert client.post(url, headers=KEY, json={**body, "note": "short"}).status_code == 422
    assert client.post(url, headers=KEY, json=body).json()["status"] == "confirmed_mule"
    assert client.post(url, headers=KEY, json=body).status_code == 409                # decided once
    assert Session().query(AuditLog).filter_by(event_type="MULE_ALERT_REVIEWED").count() == 1


def test_models_summary_and_assessment(client):
    summary = client.get("/api/v1/scams/models").json()
    assert summary["scam_mode"] in ("shadow", "active") and "scam" in summary and "mule" in summary
    if not (summary["scam"]["challenger"] or summary["scam"]["champion"]):
        pytest.skip("no scam model registered")
    r = client.post("/api/v1/scams/assess", json={
        "transaction_id": "API-S1", "customer_id": "C-9", "transaction_date": "2024-11-02", "transaction_amount": 4800,
        "currency": "GBP", "country": "United Kingdom", "payee_id": "NEW", "cop_result": "no_match",
        "payee_account_opened_date": "2024-10-20", "historical_average_transaction_amount": 120,
        "account_balance_before": 9000})
    assert r.status_code == 200 and r.json()["exposure_gbp"] == 2400.0
    assert client.post("/api/v1/scams/assess", json={"transaction_id": "x", "customer_id": "c",
                                                      "transaction_date": "2024-11-02", "transaction_amount": 1,
                                                      "currency": "GBP", "payee_id": "P",
                                                      "cop_result": "maybe"}).status_code == 422
