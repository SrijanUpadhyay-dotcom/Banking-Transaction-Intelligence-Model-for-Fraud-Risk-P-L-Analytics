"""
Integration tests for the FastAPI REST endpoints.
Uses FastAPI's dependency override to inject an in-memory SQLite DB.
No side effects on the real database.
"""

import os
import pytest
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set env vars before any bti imports
os.environ.setdefault("BTI_SECRET_KEY", "test-secret-key-minimum-32-chars-long")
os.environ.setdefault("BTI_API_KEY", "test-api-key")

from bti.database.models import Base, Transaction, FraudAlert
from bti.database.connection import get_db

# ── In-memory test DB (shared across all tests in this module) ────────────────

TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # single connection — all sessions see the same in-memory DB
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
def client():
    from api.main import app
    app.dependency_overrides[get_db] = override_get_db

    # Create tables in test DB
    Base.metadata.create_all(bind=TEST_ENGINE)

    # Seed minimal test data
    db = TestingSessionLocal()
    try:
        sample_txns = [
            Transaction(
                transaction_id=f"TEST{i:04d}",
                customer_id=f"CUST{i % 5:04d}",
                account_id=f"ACC{i:04d}",
                transaction_date=datetime(2024, 1, i % 28 + 1),
                transaction_amount=float(100 * (i + 1)),
                channel="Mobile App" if i % 2 == 0 else "Web",
                customer_segment="Mass Market",
                fraud_flag=1 if i < 3 else 0,
                is_suspicious=1 if i < 5 else 0,
                final_risk_score=float(80 if i < 3 else 20),
                final_alert_tier="CRITICAL" if i < 3 else "LOW",
                fraud_rule_score=float(70 if i < 3 else 10),
                net_pnl_impact=float(50 - i * 5),
                fee_income=2.0,
                interchange_income=1.0,
                processing_cost=0.5,
                fraud_loss=float(100 * (i + 1) if i < 3 else 0),
                chargeback_loss=0.0,
                refund_loss=0.0,
                month_year="2024-01",
            )
            for i in range(10)
        ]
        db.bulk_save_objects(sample_txns)
        db.commit()
    finally:
        db.close()

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=TEST_ENGINE)


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_includes_version(self, client):
        r = client.get("/health")
        assert "version" in r.json()

    def test_health_db_status(self, client):
        r = client.get("/health")
        assert r.json()["database"] == "ok"


class TestTransactionsAPI:
    def test_list_transactions_returns_200(self, client):
        r = client.get("/api/v1/transactions/")
        assert r.status_code == 200

    def test_list_transactions_pagination(self, client):
        r = client.get("/api/v1/transactions/?page=1&page_size=5")
        body = r.json()
        assert body["page"] == 1
        assert len(body["items"]) <= 5
        assert body["total"] >= 10

    def test_get_single_transaction(self, client):
        r = client.get("/api/v1/transactions/TEST0000")
        assert r.status_code == 200
        assert r.json()["transaction_id"] == "TEST0000"

    def test_get_missing_transaction_404(self, client):
        r = client.get("/api/v1/transactions/NONEXISTENT")
        assert r.status_code == 404

    def test_filter_by_fraud_only(self, client):
        r = client.get("/api/v1/transactions/?fraud_only=true")
        body = r.json()
        assert body["total"] == 3
        assert all(t["fraud_flag"] == 1 for t in body["items"])

    def test_filter_by_channel(self, client):
        r = client.get("/api/v1/transactions/?channel=Mobile+App")
        body = r.json()
        assert all(t["channel"] == "Mobile App" for t in body["items"])


class TestAnalyticsAPI:
    def test_pnl_kpis_returns_200(self, client):
        r = client.get("/api/v1/analytics/pnl/kpis")
        assert r.status_code == 200
        body = r.json()
        assert "total_transactions" in body
        assert body["total_transactions"] == 10

    def test_pnl_kpis_contains_expected_fields(self, client):
        r = client.get("/api/v1/analytics/pnl/kpis")
        body = r.json()
        for field in ("net_revenue", "fraud_count", "fraud_loss_rate", "cost_to_income"):
            assert field in body, f"Missing field: {field}"

    def test_customer_risk_profile(self, client):
        r = client.get("/api/v1/analytics/risk/customer/CUST0000")
        assert r.status_code == 200
        body = r.json()
        assert body["customer_id"] == "CUST0000"
        assert body["total_transactions"] > 0

    def test_customer_not_found_404(self, client):
        r = client.get("/api/v1/analytics/risk/customer/NOTEXIST")
        assert r.status_code == 404

    def test_top_risk_customers(self, client):
        r = client.get("/api/v1/analytics/risk/top-customers?limit=5")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestAlertsAPI:
    def test_list_alerts_empty_initially(self, client):
        r = client.get("/api/v1/alerts/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_refresh_exception_queue(self, client):
        r = client.post("/api/v1/alerts/refresh")
        assert r.status_code == 200
        body = r.json()
        assert "created" in body
        assert body["created"] == 3  # 3 CRITICAL transactions seeded

    def test_list_alerts_after_refresh(self, client):
        client.post("/api/v1/alerts/refresh")
        r = client.get("/api/v1/alerts/?status=OPEN")
        assert r.status_code == 200
        assert len(r.json()) >= 3

    def test_update_alert_status(self, client):
        # First refresh to get alerts
        client.post("/api/v1/alerts/refresh")
        alerts = client.get("/api/v1/alerts/?status=OPEN").json()
        assert len(alerts) >= 1
        alert_id = alerts[0]["id"]

        r = client.patch(f"/api/v1/alerts/{alert_id}",
                         json={"status": "REVIEWING", "assigned_to": "analyst01"})
        assert r.status_code == 200
        assert r.json()["status"] == "REVIEWING"

    def test_update_nonexistent_alert_404(self, client):
        r = client.patch("/api/v1/alerts/99999", json={"status": "CLOSED"})
        assert r.status_code == 404


class TestPipelineAPI:
    def test_status_returns_before_run(self, client):
        r = client.get("/api/v1/pipeline/status")
        assert r.status_code == 200

    def test_trigger_requires_api_key(self, client):
        r = client.post("/api/v1/pipeline/run", json={"force": False})
        assert r.status_code == 401

    def test_trigger_with_api_key_accepted(self, client):
        from bti.config import get_settings
        api_key = get_settings().api_key
        r = client.post(
            "/api/v1/pipeline/run",
            json={"force": False},
            headers={"X-API-Key": api_key},
        )
        assert r.status_code == 200
        body = r.json()
        assert "run_id" in body
        assert body["status"] == "TRIGGERED"
