# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 7.5: data-residency guard and readiness probe."""

import pytest

from bti.config import get_settings
from bti.operations import residency


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "residency_jurisdiction", "IN")
    monkeypatch.setattr(s, "residency_in_country_hosts", ["localhost", "10.0.0.0/8", "*.ap-south-1.rds.amazonaws.com",
                                                          "*.bank.in.internal"])
    monkeypatch.setattr(s, "database_url", "postgresql://bti@bti-prod.abc.ap-south-1.rds.amazonaws.com/bti")
    monkeypatch.setattr(s, "feature_store_url", "redis://10.2.3.4:6379/0")
    monkeypatch.setattr(s, "streaming_bootstrap_servers", "kafka-1.bank.in.internal:9093,kafka-2.bank.in.internal:9093")
    monkeypatch.setattr(s, "smtp_host", "")
    monkeypatch.setattr(s, "webhook_url", "")
    monkeypatch.setattr(s, "stepup_webhook_url", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return s


def test_in_country_matching_by_name_suffix_and_cidr():
    allowed = ["localhost", "10.0.0.0/8", "*.ap-south-1.amazonaws.com"]
    assert residency.in_country("10.9.8.7", allowed) and residency.in_country("x.ap-south-1.amazonaws.com", allowed)
    assert not residency.in_country("x.eu-west-1.amazonaws.com", allowed)
    assert not residency.in_country("8.8.8.8", allowed) and residency.in_country("local-disk", allowed)


def test_compliant_inventory_passes(settings):
    r = residency.check()
    assert r["compliant"] and r["jurisdiction"] == "IN"
    assert {e["name"] for e in r["endpoints"]} == {"database", "feature_store", "kafka_broker_1", "kafka_broker_2"}


def test_out_of_country_llm_is_flagged_and_blocked_in_enforce_mode(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r = residency.check()
    assert not r["compliant"] and [v["name"] for v in r["violations"]] == ["copilot_llm"]
    monkeypatch.setattr(settings, "residency_mode", "warn")
    residency.enforce_at_startup("test")                                   # warns only
    monkeypatch.setattr(settings, "residency_mode", "enforce")
    with pytest.raises(residency.ResidencyViolation, match="copilot_llm"):
        residency.enforce_at_startup("test")
    with pytest.raises(residency.ResidencyViolation):
        residency.allow_endpoint("copilot_llm")
    residency.allow_endpoint("database")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm-gateway.bank.in.internal")   # in-region gateway
    assert residency.check()["compliant"]


def test_guard_is_off_without_a_jurisdiction(settings, monkeypatch):
    monkeypatch.setattr(settings, "residency_jurisdiction", None)
    monkeypatch.setattr(settings, "database_url", "postgresql://db.eu-west-1.rds.amazonaws.com/bti")
    assert residency.check()["compliant"]


def test_readiness_reports_store_outage_as_degraded_and_residency_as_critical(settings, monkeypatch):
    from bti.operations import readiness
    from bti.streaming import feature_store

    class Down:
        def ping(self):
            return False
    monkeypatch.setattr(feature_store, "get_store", lambda: Down())
    r = readiness.readiness()
    assert r["checks"]["feature_store"]["status"] == "degraded" and "feature_store" in r["degraded"]
    monkeypatch.setattr(settings, "residency_mode", "enforce")
    monkeypatch.setattr(settings, "feature_store_url", "redis://cache.eu-west-1.example.com:6379/0")
    r = readiness.readiness()
    assert not r["ready"] and "residency" in r["failed"]
