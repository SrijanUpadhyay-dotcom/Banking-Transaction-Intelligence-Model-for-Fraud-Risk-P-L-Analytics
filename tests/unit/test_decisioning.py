"""Unit tests for expected-cost decisioning guardrails and jurisdiction policy."""

import numpy as np
import pytest

from bti.jurisdiction.policies import POLICIES, policy_for, tra_eligibility
from bti.operations.decisioning import CostModel, backtest_policy, decide, fit_action_budget


def test_larger_amount_escalates_at_same_probability():
    small = decide(0.55, 5, policy_for("US"), channel="Branch")
    large = decide(0.55, 20_000, policy_for("US"), channel="Branch")
    assert small.action == "APPROVE"
    assert large.action == "DECLINE"


def test_never_declines_customer_more_likely_genuine():
    d = decide(0.45, 1_000_000, policy_for("US"), channel="Branch")
    assert d.action == "REVIEW"
    assert any("Decline withheld" in g for g in d.guardrails_applied)


def test_provisional_model_cannot_auto_decline():
    d = decide(0.99, 50_000, policy_for("US"), channel="Branch", provisional_model=True)
    assert d.action == "REVIEW"
    assert any("Provisional" in g for g in d.guardrails_applied)


def test_gdpr_decline_carries_human_review_route():
    assert decide(0.99, 50_000, policy_for("DE"), channel="Branch").human_review_route is True
    assert decide(0.99, 50_000, policy_for("US"), channel="Branch").human_review_route is False


def test_step_up_only_on_capable_channels():
    assert decide(0.05, 300, None, channel="ATM").action == "APPROVE"
    assert decide(0.05, 300, None, channel="Mobile Banking").action == "STEP_UP"
    assert decide(0.05, 300, None, channel="POS Terminal", transaction_type="Card Not Present").action == "STEP_UP"


def test_jurisdiction_sets_loss_given_fraud():
    assert decide(0.5, 100, policy_for("GB")).cost_model["loss_given_fraud"] == POLICIES["GB"].loss_given_fraud
    assert decide(0.5, 100, None, cost_overrides={"loss_given_fraud": 0.4}).cost_model["loss_given_fraud"] == 0.4


def test_action_budget_is_respected():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.beta(0.5, 20, n)
    amt = rng.lognormal(6, 1.5, n)
    channels = ["Mobile Banking"] * n
    types = ["Transfer"] * n
    cm = fit_action_budget(p, amt, channels, types, CostModel(), "STEP_UP", 0.03)
    r = backtest_policy(np.zeros(n), p, amt, channels, types, cm)
    assert r["expected_cost_policy"]["action_mix"]["STEP_UP"] / n <= 0.031


def test_policy_lookup_aliases():
    assert policy_for("United Kingdom").iso2 == "GB"
    assert policy_for("uae").iso2 == "AE"
    assert policy_for("in").iso2 == "IN"
    assert policy_for(None) is None and policy_for("Atlantis") is None
    assert len(POLICIES) == 8


@pytest.mark.parametrize("fraud,total,expected", [(5, 100_000, 500), (50, 100_000, 250), (100, 100_000, 100),
                                                  (500, 100_000, 0)])
def test_psd2_tra_bands(fraud, total, expected):
    assert tra_eligibility(fraud, total, "remote_card")["max_exemption_threshold_eur"] == expected


def test_tra_rejects_bad_input():
    with pytest.raises(ValueError):
        tra_eligibility(1, 0)
    with pytest.raises(ValueError):
        tra_eligibility(1, 100, "cheque")
