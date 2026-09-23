"""Unit tests for the BTI v3 model: feature governance, point-in-time features, metrics, registry, scoring."""

import numpy as np
import pandas as pd
import pytest

from bti.governance.fairness import fairness_report
from bti.governance.monitoring import build_baseline, drift_report, psi
from bti.governance.reason_codes import principal_reasons
from bti.modeling import fx, registry
from bti.modeling.calibration import PlattCalibrator
from bti.modeling.features import (
    FEATURE_NAMES, FeatureGovernanceError, FeatureSpec, Kind, assert_feature_lineage, build_features,
    fit_category_encodings, leakage_audit, out_of_fold_category_encoding, to_model_matrix,
)
from bti.modeling.metrics import (
    alert_budget_table, classification_metrics, expected_calibration_error, lift_table, operating_point,
)


def _txns(rows):
    base = {"currency": "USD", "device_id": "D1", "ip_location": "1.1.1.1", "merchant_name": "M",
            "historical_average_transaction_amount": 100.0, "account_balance_before": 1000.0,
            "failed_attempt_count": 0, "login_attempts": 1, "debit_credit_flag": "Debit", "channel": "ATM",
            "authorization_method": "PIN", "merchant_category": "Retail", "transaction_type": "Purchase"}
    return pd.DataFrame([{**base, **r} for r in rows])


# ── Feature governance ────────────────────────────────────────────────────────

def test_model_features_pass_lineage():
    assert_feature_lineage()


@pytest.mark.parametrize("forbidden", ["fraud_loss", "risk_score", "chargeback_flag", "customer_age_band",
                                       "transaction_status"])
def test_lineage_blocks_forbidden_sources(forbidden):
    bad = FeatureSpec("leaky", Kind.NUMERIC, (forbidden,), "AMT_HIGH", "test")
    with pytest.raises(FeatureGovernanceError):
        assert_feature_lineage([bad])


def test_lineage_blocks_uncatalogued_source():
    with pytest.raises(FeatureGovernanceError):
        assert_feature_lineage([FeatureSpec("x", Kind.NUMERIC, ("mystery_column",), "AMT_HIGH", "t")])


def test_leakage_audit_flags_label_copy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    df = pd.DataFrame({"fraud_flag": y, "leak": y * 60 + rng.integers(0, 3, 500), "noise": rng.random(500)})
    findings = {r["column"]: r for r in leakage_audit(df)}
    assert findings["leak"]["suspected_leak"] is True
    assert findings["noise"]["suspected_leak"] is False


# ── Point-in-time features ────────────────────────────────────────────────────

def test_velocity_counts_only_strictly_prior_events():
    df = _txns([
        {"transaction_id": "a", "customer_id": "C1", "transaction_date": "2024-01-01", "transaction_time": "10:00:00",
         "transaction_amount": 100},
        {"transaction_id": "b", "customer_id": "C1", "transaction_date": "2024-01-01", "transaction_time": "10:30:00",
         "transaction_amount": 200},
        {"transaction_id": "c", "customer_id": "C1", "transaction_date": "2024-01-01", "transaction_time": "10:30:00",
         "transaction_amount": 300},
        {"transaction_id": "d", "customer_id": "C1", "transaction_date": "2024-01-09", "transaction_time": "10:00:00",
         "transaction_amount": 10},
    ])
    f = build_features(df)
    assert f["cust_txn_count_1h"].tolist() == [0, 1, 1, 0]          # same-timestamp events excluded
    assert f["cust_amount_usd_24h"].tolist() == [0.0, 100.0, 100.0, 0.0]
    assert f["cust_txn_count_7d"].iloc[3] == 0
    assert f["secs_since_last_txn"].iloc[1] == 1800
    assert np.isnan(f["secs_since_last_txn"].iloc[0])


def test_future_events_never_change_past_features():
    rows = [{"transaction_id": "a", "customer_id": "C1", "transaction_date": "2024-01-01",
             "transaction_time": "10:00:00", "transaction_amount": 100}]
    before = build_features(_txns(rows)).iloc[0]
    later = rows + [{"transaction_id": "z", "customer_id": "C1", "transaction_date": "2024-01-01",
                     "transaction_time": "11:00:00", "transaction_amount": 9999, "device_id": "D9"}]
    after = build_features(_txns(later)).iloc[0]
    pd.testing.assert_series_equal(before, after, check_names=False)


def test_device_novelty_and_sharing():
    df = _txns([
        {"transaction_id": "a", "customer_id": "C1", "transaction_date": "2024-01-01", "transaction_time": "10:00:00",
         "transaction_amount": 1, "device_id": "D1"},
        {"transaction_id": "b", "customer_id": "C2", "transaction_date": "2024-01-02", "transaction_time": "10:00:00",
         "transaction_amount": 1, "device_id": "D1"},
        {"transaction_id": "c", "customer_id": "C1", "transaction_date": "2024-01-03", "transaction_time": "10:00:00",
         "transaction_amount": 1, "device_id": "D1"},
    ])
    f = build_features(df)
    assert f["device_new_for_customer"].tolist() == [1.0, 1.0, 0.0]
    assert f["device_other_customer_txns"].tolist() == [0.0, 1.0, 1.0]


def test_fx_normalisation_and_unknown_currency():
    assert fx.to_usd(1000, "INR") == pytest.approx(1000 * fx.RATES_TO_USD["INR"])
    assert fx.to_usd(50, "usd") == 50
    assert np.isnan(fx.to_usd(50, "XYZ"))
    df = _txns([{"transaction_id": "a", "customer_id": "C1", "transaction_date": "2024-01-01",
                 "transaction_time": "10:00:00", "transaction_amount": 1000, "currency": "INR"}])
    f = build_features(df)
    assert f["amount_usd"].iloc[0] == pytest.approx(1000 * fx.RATES_TO_USD["INR"])
    assert f["amount_vs_hist_avg"].iloc[0] == pytest.approx(10.0)   # currency-invariant ratio


def test_category_encoding_is_out_of_fold_and_handles_unseen():
    feats = pd.DataFrame({c: ["A"] * 50 + ["B"] * 50 for c in ["channel", "authorization_method",
                                                               "merchant_category", "transaction_type"]})
    y = np.array([1] * 50 + [0] * 50)
    enc = fit_category_encodings(feats, y, smoothing=10)
    assert enc["channel"]["rates"]["A"] > enc["channel"]["rates"]["B"]
    oof = out_of_fold_category_encoding(feats, y, smoothing=10)
    assert oof.notna().all().all()
    unseen = pd.DataFrame({c: ["NEVER_SEEN"] for c in FEATURE_NAMES})
    X = to_model_matrix(unseen, enc)
    assert X["channel"].iloc[0] == pytest.approx(enc["channel"]["prior"])


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_operating_point_and_value_detection():
    y = np.array([1, 1, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    amt = np.array([100, 50, 10, 10, 10, 850])
    op = operating_point(y, p, 0.75, amounts=amt, accounts=["A", "A", "B", "C", "D", "E"])
    assert op["alerts"] == 2 and op["precision"] == 1.0
    assert op["tdr"] == pytest.approx(2 / 3, abs=1e-4)
    assert op["vdr"] == pytest.approx(150 / 1000)       # counts dollars, not transactions
    assert op["adr"] == pytest.approx(0.5)


def test_lift_table_and_budgets():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    p = y * 0.6 + rng.random(1000) * 0.5
    lt = lift_table(y, p)
    captured = [r["cum_fraud_captured"] for r in lt]
    assert captured == sorted(captured) and captured[-1] == pytest.approx(1.0)
    budgets = alert_budget_table(y, p, budgets=(0.1,))
    assert budgets[0]["alert_rate"] == pytest.approx(0.1, abs=0.01)


def test_calibration_metrics():
    rng = np.random.default_rng(2)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    m = classification_metrics(y, p)
    assert 0.5 < m["roc_auc"] < 1 and m["gini"] == pytest.approx(2 * m["roc_auc"] - 1, abs=1e-3)


def test_platt_calibration_preserves_ranking():
    rng = np.random.default_rng(3)
    raw = rng.random(2000)
    y = (rng.random(2000) < raw ** 2).astype(int)
    cal = PlattCalibrator().fit(raw, y).predict(raw)
    assert (np.argsort(raw) == np.argsort(cal)).all()


# ── Monitoring & fairness ─────────────────────────────────────────────────────

def test_psi_stable_and_shifted():
    rng = np.random.default_rng(4)
    base = pd.DataFrame({"x": rng.normal(0, 1, 5000), "c": rng.choice(["a", "b"], 5000)})
    baseline = build_baseline(base, rng.random(5000), categorical=["c"])
    same = drift_report(baseline, pd.DataFrame({"x": rng.normal(0, 1, 5000), "c": rng.choice(["a", "b"], 5000)}),
                        rng.random(5000))
    assert same["status"] == "stable"
    shifted = drift_report(baseline, pd.DataFrame({"x": rng.normal(2, 1, 5000), "c": ["a"] * 5000}),
                           rng.random(5000) ** 4)
    assert shifted["status"] == "escalate"
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_drift_report_requires_minimum_sample():
    baseline = build_baseline(pd.DataFrame({"x": np.arange(500.0)}), np.random.rand(500), categorical=[])
    assert drift_report(baseline, pd.DataFrame({"x": [1.0, 2.0]}))["status"] == "insufficient_data"


def test_fairness_flags_significant_disparity_only():
    n = 4000
    y = np.zeros(n, dtype=int)
    group = np.array(["A"] * 2000 + ["B"] * 2000)
    flagged = np.zeros(n, dtype=bool)
    flagged[:40] = True            # A: 2% of legit flagged
    flagged[2000:2200] = True      # B: 10% of legit flagged
    report = fairness_report(y, flagged, {"segment": group})
    assert report["status"] == "review_required"
    assert [f["group"] for f in report["findings"]] == ["B"]

    balanced = np.zeros(n, dtype=bool)
    balanced[:100] = True
    balanced[2000:2100] = True
    assert fairness_report(y, balanced, {"segment": group})["status"] == "pass"


def test_reason_codes_rank_only_risk_raising_contributions():
    contributions = {"amount_vs_hist_avg": 2.0, "login_attempts": 0.5, "device_new_for_customer": 1.0,
                     "txn_hour": -3.0, "is_off_hours": 0.2, "channel": 0.1, "merchant_category": 0.05}
    reasons = principal_reasons(contributions, {})
    assert [r["code"] for r in reasons] == ["AMT_VS_HISTORY", "NEW_DEVICE", "LOGIN_ANOMALY", "CHANNEL_RISK"]
    assert all(r["contribution_log_odds"] > 0 for r in reasons)
    assert len(reasons) <= 4


# ── Registry governance ───────────────────────────────────────────────────────

@pytest.fixture
def temp_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("BTI_MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    registry.clear_cache()
    yield
    registry.clear_cache()


def _register(model_id, status, developer="dev.one"):
    registry.save_model(model_id, {"stub": True},
                        {"ownership": {"developer": developer}, "validation": {"status": status}})


def test_failed_model_cannot_become_champion(temp_registry):
    _register("m-failed", "failed")
    with pytest.raises(registry.RegistryError, match="validation"):
        registry.assign_role("m-failed", "champion", "risk.officer", "Attempted promotion of failed model")


def test_four_eyes_principle(temp_registry):
    _register("m-ok", "passed", developer="dev.one")
    with pytest.raises(registry.RegistryError, match="Four-eyes"):
        registry.assign_role("m-ok", "champion", "DEV.ONE", "Self-approval should be blocked")
    event = registry.assign_role("m-ok", "champion", "risk.officer", "Independent validation completed")
    assert event["previous"] is None
    assert registry.model_for_role("champion") == "m-ok"
    assert registry.read_index()["history"][-1]["approver"] == "risk.officer"


def test_registry_entries_are_immutable(temp_registry):
    _register("m-1", "passed")
    with pytest.raises(registry.RegistryError, match="immutable"):
        _register("m-1", "passed")


def test_promotion_requires_rationale(temp_registry):
    _register("m-2", "passed")
    with pytest.raises(registry.RegistryError, match="rationale"):
        registry.assign_role("m-2", "challenger", "risk.officer", "ok")


# ── Scorer (uses the registered model in models/registry) ─────────────────────

REGISTERED = registry.read_index().get("challenger") or registry.read_index().get("champion")


@pytest.mark.skipif(not REGISTERED, reason="No v3 model registered — run python -m bti.modeling.train")
def test_live_score_matches_batch_score():
    from bti.modeling.scorer import scorer
    df = _txns([
        {"transaction_id": f"t{i}", "customer_id": "C1", "transaction_date": f"2024-03-{i + 1:02d}",
         "transaction_time": "10:00:00", "transaction_amount": 100 + i} for i in range(5)
    ] + [{"transaction_id": "t9", "customer_id": "C1", "transaction_date": "2024-03-09", "transaction_time": "03:00:00",
          "transaction_amount": 5000, "device_id": "NEW", "login_attempts": 6}])
    batch = scorer.score_frame(df)["fraud_probability"].iloc[-1]
    live = scorer.score(df.iloc[-1].to_dict(), history=df.iloc[:-1])
    assert live.fraud_probability == pytest.approx(batch, abs=1e-6)
    assert live.reason_codes and live.reason_codes[0]["code"] in {"AMT_VS_HISTORY", "NEW_DEVICE", "LOGIN_ANOMALY",
                                                                   "AMT_HIGH", "NEW_IP"}
    assert 0 <= live.score <= 1000
