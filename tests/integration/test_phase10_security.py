# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 10: authentication, role-based access, identity binding and hardening, with the legacy key switched off."""

import logging
import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from bti.config import get_settings
from bti.database.connection import get_db
from bti.database.models import Base
from bti.security import access, principals
from bti.security.hardening import RedactionFilter, redact

ENGINE = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
Session = sessionmaker(bind=ENGINE)


@pytest.fixture
def secure(tmp_path, monkeypatch):
    """Production-like: legacy key off, principals from a temporary file."""
    monkeypatch.setenv("BTI_PRINCIPALS_FILE", str(tmp_path / "principals.json"))
    monkeypatch.setattr(get_settings(), "allow_legacy_api_key", False)
    keys = {}
    for pid, roles in (("ana.lyst", ["analyst"]), ("mr.one", ["model_risk"]), ("ops.one", ["operations"]),
                       ("aud.it", ["auditor"]), ("svc.score", ["scoring"])):
        _, keys[pid] = principals.add(pid, pid, roles)
    return keys


@pytest.fixture
def client(secure):
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


def _routes():
    from api.main import app
    for path, ops in app.openapi()["paths"].items():
        for m in ops:
            yield m.upper(), path, re.sub(r"\{[^}]+\}", "X1", path)


def test_every_route_has_an_explicit_access_rule():
    missing = [(m, p) for m, p, c in _routes() if access.rule_for(m, c) is None]
    assert missing == []


def test_every_protected_route_refuses_anonymous_callers(client):
    leaks = []
    for method, path, concrete in _routes():
        if access.is_public(method, concrete):
            continue
        r = client.request(method, concrete, json={} if method != "GET" else None)
        if r.status_code != 401:
            leaks.append((method, path, r.status_code))
    assert leaks == []


def test_roles_are_enforced(client, secure):
    def h(pid):
        return {"X-API-Key": secure[pid]}
    assert client.get("/api/v1/transactions/", headers=h("ana.lyst")).status_code == 200
    assert client.get("/api/v1/transactions/", headers=h("ops.one")).status_code == 403      # no customer data
    assert client.get("/api/v1/governance/models", headers=h("ops.one")).status_code == 200   # aggregate report
    assert client.get("/api/v1/governance/models", headers=h("svc.score")).status_code == 403
    assert client.post("/api/v1/governance/findings", headers=h("ana.lyst"), json={}).status_code == 403
    assert client.post("/api/v1/governance/findings", headers=h("aud.it"), json={}).status_code == 403  # read only
    assert client.post("/api/v1/governance/findings", headers=h("mr.one"), json={}).status_code == 422   # allowed
    assert client.get("/api/v1/transactions/", headers={"X-API-Key": "test-api-key"}).status_code == 401  # legacy off


def test_acting_identity_is_bound_to_the_authenticated_principal(client, secure):
    body = {"model_id": "m", "validator": "someone.else", "decision": "approve", "scope": "full",
            "rationale": "independent validation complete"}
    r = client.post("/api/v1/governance/signoffs", headers={"X-API-Key": secure["mr.one"]}, json=body)
    assert r.status_code == 403 and "validator" in r.json()["detail"]
    r = client.post("/api/v1/governance/signoffs", headers={"X-API-Key": secure["mr.one"]},
                    json={**body, "validator": "mr.one"})
    assert r.status_code != 403                                     # passes access control; the handler decides
    assert access.identity_mismatches({"assigned_to": "someone.else", "approver": "MR.ONE"}, "mr.one") == []


def test_deactivated_principals_lose_access_immediately(client, secure):
    h = {"X-API-Key": secure["ana.lyst"]}
    assert client.get("/api/v1/cases", headers=h).status_code == 200
    principals.deactivate("ana.lyst")
    assert client.get("/api/v1/cases", headers=h).status_code == 401


def test_keys_are_stored_only_as_hashes(secure):
    text = principals.principals_file().read_text()
    assert all(k not in text for k in secure.values()) and "key_sha256" in text


def test_public_endpoints_reveal_nothing_sensitive(client, secure):
    anon = client.get("/readyz").json()
    assert set(anon) == {"ready"}
    full = client.get("/readyz", headers={"X-API-Key": secure["aud.it"]}).json()
    assert "checks" in full
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_security_headers_size_limit_and_rate_limit(client, secure, monkeypatch):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
    big = client.post("/api/v1/v3/score", headers={"X-API-Key": secure["svc.score"], "Content-Length": str(10**9)},
                      content=b"{}")
    assert big.status_code == 413
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", {"default": 3, "scoring": 3, "anonymous": 3})
    from api import security
    security._rate.clear()
    codes = [client.get("/api/v1/cases", headers={"X-API-Key": secure["ana.lyst"]}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[-1] == 429
    security._rate.clear()


def test_card_numbers_and_ibans_are_redacted_from_logs():
    assert redact("card 4111 1111 1111 1111 declined") == "card [PAN ****1111] declined"
    assert redact("order 1234567890123 shipped") == "order 1234567890123 shipped"            # not Luhn-valid
    assert redact("to GB29NWBK60161331926819 today") == "to [IBAN GB****] today"
    record = logging.LogRecord("x", logging.INFO, "f", 1, "pan=%s", ("4111111111111111",), None)
    record.extra_pan = "5500005555555559"
    RedactionFilter().filter(record)
    assert "4111111111111111" not in record.getMessage() and record.extra_pan == "[PAN ****5559]"


def test_production_refuses_the_default_shared_key(monkeypatch):
    from bti.security.hardening import StartupSecurityError, enforce_secrets_at_startup
    s = get_settings()
    monkeypatch.setattr(s, "environment", "production")
    monkeypatch.setattr(s, "allow_legacy_api_key", True)
    monkeypatch.setattr(s, "api_key", "CHANGE_ME_API_KEY")
    with pytest.raises(StartupSecurityError):
        enforce_secrets_at_startup()


def test_high_value_clear_needs_the_checker_to_confirm_with_their_own_key(client, secure, tmp_path):
    from datetime import datetime, timedelta
    from bti.database.models import FraudCase
    _, ben = principals.add("ben.checker", "Ben", ["analyst"])
    db = Session()
    case = FraudCase(transaction_id="HV-1", queue="high_value", priority_score=1.0, source="test", amount_usd=25000.0,
                     status="assigned", assigned_to="ana.lyst", created_at=datetime.utcnow(),
                     sla_due_at=datetime.utcnow() + timedelta(hours=2))
    db.add(case)
    db.commit()
    cid = case.id
    ana = {"X-API-Key": secure["ana.lyst"]}
    clear = {"disposition": "confirmed_genuine", "analyst": "ana.lyst", "checked_by": "ben.checker"}
    assert client.post(f"/api/v1/cases/{cid}/disposition", headers=ana, json=clear).status_code == 409  # unconfirmed
    assert client.post(f"/api/v1/cases/{cid}/check", headers=ana, json={"checker": "ben.checker"}).status_code == 403
    assert client.post(f"/api/v1/cases/{cid}/check", headers=ana, json={"checker": "ana.lyst"}).status_code == 409
    assert client.post(f"/api/v1/cases/{cid}/check", headers={"X-API-Key": ben},
                       json={"checker": "ben.checker", "notes": "Customer verified by call-back"}).status_code == 200
    done = client.post(f"/api/v1/cases/{cid}/disposition", headers=ana, json=clear)
    assert done.status_code == 200 and done.json()["checked_by"] == "ben.checker"


def test_tampered_model_artifact_is_refused(tmp_path, monkeypatch):
    from bti.modeling import registry
    monkeypatch.setenv("BTI_MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    registry.clear_cache()
    registry.save_model("m-int", {"weights": [1, 2, 3]}, {"ownership": {"developer": "d"}, "validation": {"status": "passed"}})
    assert registry.load_artifact("m-int")["weights"] == [1, 2, 3]
    registry.clear_cache()
    import joblib
    joblib.dump({"weights": "swapped"}, tmp_path / "registry" / "m-int" / "model.joblib")
    with pytest.raises(registry.RegistryError, match="integrity"):
        registry.load_artifact("m-int")
    registry.clear_cache()


def test_evidence_pack_is_hashed_anchored_and_detects_changes(tmp_path, monkeypatch, secure):
    from bti.governance.audit_chain import install_guards
    from bti.security import evidence
    monkeypatch.setattr(evidence, "ROOT", tmp_path / "evidence")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    install_guards(engine)
    db = sessionmaker(bind=engine)()
    r = evidence.build(db, run_assessment=False, sbom=False)
    assert {"access_matrix.json", "principals_review.json", "audit_chain.json", "change_history.json",
            "config_snapshot.json"} <= set(r["files"])
    pack = tmp_path / "evidence" / r["folder"].split("/")[-1]
    review = (pack / "principals_review.json").read_text()
    assert "ana.lyst" in review and "key_sha256" not in review and all(k not in review for k in secure.values())
    snapshot = (pack / "config_snapshot.json").read_text()
    assert "test-api-key" not in snapshot
    assert evidence.verify(str(pack), db)["intact"]
    (pack / "access_matrix.json").write_text("{}")
    v = evidence.verify(str(pack), db)
    assert not v["intact"] and v["files_changed"] == ["access_matrix.json"]


def test_pan_discovery_finds_cards_but_not_hashes_or_decimals(tmp_path):
    from bti.security.pan_scan import find_pans, scan_files
    assert find_pans("card 4111 1111 1111 1111 and amex 378282246310005") == ["411111******1111", "378282*****0005"]
    assert find_pans("score 0.4111111111111111 hash 6e35428048955969c test 9999991234567893") == []
    (tmp_path / "out.json").write_text('{"note": "pan 5555555555554444"}')
    (tmp_path / "ok.csv").write_text("a,b\n0.5212121212121212,3\n")
    hits = scan_files([tmp_path])
    assert [h["location"].split("/")[-1] for h in hits] == ["out.json"] and hits[0]["examples"] == ["555555******4444"]
