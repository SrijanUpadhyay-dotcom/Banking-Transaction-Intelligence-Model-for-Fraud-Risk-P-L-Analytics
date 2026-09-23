"""Unit tests for Phase 0 helpers: legacy mapping, transaction preparation, metrics grouping, promote CLI."""

import sys

import pytest

from api.metrics import route_group
from bti.modeling import promote, registry
from bti.operations.scoring_service import prepare_transaction
from bti.scoring.v3_adapter import legacy_action, recommendation, tier_for, top_drivers


@pytest.mark.parametrize("p,tier", [(0.95, "CRITICAL"), (0.75, "VERY HIGH"), (0.55, "HIGH"), (0.2, "MEDIUM"),
                                    (0.01, "LOW")])
def test_tier_bands(p, tier):
    assert tier_for(p) == tier


def test_legacy_action_follows_decision_not_tier():
    assert legacy_action("DECLINE", 0.9) == "BLOCK"
    assert legacy_action("REVIEW", 0.95) == "HOLD"        # a provisional CRITICAL score is held, never blocked
    assert legacy_action("STEP_UP", 0.3) == "HOLD"
    assert legacy_action("APPROVE", 0.2) == "MONITOR"
    assert legacy_action("APPROVE", 0.01) == "ALLOW"
    assert recommendation("REVIEW", 0.95).startswith("REVIEW")


def test_top_drivers_rank_by_absolute_impact():
    drivers = top_drivers({"login_attempts": 2.0, "txn_hour": -3.0, "channel": 0.5}, {"login_attempts": 8}, n=2)
    assert [d["feature"] for d in drivers] == ["txn_hour", "login_attempts"]
    assert drivers[0]["direction"] == "decreases_risk" and drivers[1]["value"] == 8


def test_prepare_transaction_defaults():
    out = prepare_transaction({"transaction_id": "T1", "transaction_amount": 5, "transaction_time": "02:17",
                               "currency": "gbp"})
    assert out["transaction_time"] == "02:17:00" and out["currency"] == "GBP"
    assert len(out["transaction_date"]) == 10
    assert out["customer_id"] == "UNIDENTIFIED-T1"      # unidentified customers never share history


@pytest.mark.parametrize("path,group", [("/api/v1/score/", "score"), ("/api/v1/score/explain", "score/explain"),
                                        ("/api/v1/sas/enrich/batch", "sas/enrich"), ("/api/v1/v3/score", "v3/score"),
                                        ("/api/v1/transactions/T1", "transactions"), ("/health", "health")])
def test_route_groups(path, group):
    assert route_group(path) == group


def test_promote_cli_refuses_self_approval(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BTI_MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    registry.clear_cache()
    registry.save_model("m-ok", {"stub": True}, {"ownership": {"developer": "dev.one"},
                                                 "validation": {"status": "passed"}})
    argv = ["promote", "--model", "m-ok", "--role", "champion", "--approver", "dev.one",
            "--rationale", "Self approval attempt"]
    monkeypatch.setattr(sys, "argv", argv)
    assert promote.main() == 1
    assert "Four-eyes" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", argv[:6] + ["risk.officer", "--rationale", "Independent validation done"])
    assert promote.main() == 0
    assert registry.model_for_role("champion") == "m-ok"
    registry.clear_cache()
