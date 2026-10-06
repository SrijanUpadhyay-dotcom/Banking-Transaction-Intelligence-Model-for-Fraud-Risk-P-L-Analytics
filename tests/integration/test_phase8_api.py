# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 8 planning API: early-warning review workflow, what-if validation and report retrieval."""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("BTI_API_KEY", "test-api-key")

from bti.database.connection import get_db
from bti.database.models import AuditLog, Base, EarlyWarning

NOKEY = {"X-API-Key": ""}      # explicitly anonymous (the client sends the test key by default)

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
    with TestClient(app, headers={"X-API-Key": os.environ["BTI_API_KEY"]}) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


def test_early_warning_review_needs_the_key_and_a_note_to_close(client):
    db = Session()
    db.add(EarlyWarning(family="merchant", segment="M1", signal="alerts", started_on="2026-10-01",
                        alarmed_on="2026-10-03", observed=12, expected=3.1, ratio=3.87, cusum=6.2, threshold=5.1))
    db.commit()
    alarm_id = db.query(EarlyWarning).first().id
    listed = client.get("/api/v1/planning/early-warning").json()
    assert [a["segment"] for a in listed] == ["M1"]
    url = f"/api/v1/planning/early-warning/{alarm_id}/review"
    assert client.post(url, headers=NOKEY, json={"status": "acknowledged", "reviewed_by": "A. Analyst"}).status_code in (401, 403)
    assert client.post(url, headers=KEY, json={"status": "closed", "reviewed_by": "A. Analyst"}).status_code == 422
    ok = client.post(url, headers=KEY, json={"status": "closed", "reviewed_by": "A. Analyst",
                                             "note": "Merchant confirmed a tokenisation outage; benign"})
    assert ok.status_code == 200 and ok.json()["status"] == "closed"
    assert client.get("/api/v1/planning/early-warning").json() == []
    assert Session().query(AuditLog).filter_by(event_type="EARLY_WARNING_REVIEWED").count() == 1


def test_whatif_rejects_bad_proposals_and_unknown_reports(client):
    assert client.post("/api/v1/planning/whatif", json={"threshold": 0.5}).status_code == 422
    assert client.get("/api/v1/planning/whatif/doesnotexist").status_code == 404
    assert client.get("/api/v1/planning/whatif/..%2Fsecrets").status_code in (404, 422)


def test_forecasts_need_enough_history(client):
    assert client.get("/api/v1/planning/forecast/loss").status_code == 409     # empty database
