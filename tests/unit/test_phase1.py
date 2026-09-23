"""Phase 1: algorithm back-ends, extended features, rolling-fold tuning, challenger tournament."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap

from bti.modeling import algorithms, registry
from bti.modeling.features import (
    ALL_FEATURES, FEATURE_SETS, assert_feature_lineage, build_features, feature_specs, to_model_matrix,
)
from bti.modeling.tuning import rolling_folds

CLEAN = Path("data/processed/banking_transactions_clean.csv")


def _xy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.random(n), "b": rng.random(n), "c": rng.random(n)})
    X.loc[rng.random(n) < 0.1, "b"] = np.nan
    y = ((X["a"] > 0.75) | (X["b"].fillna(0) > 0.9) | (rng.random(n) < 0.02)).astype(int).to_numpy()
    return X, y


@pytest.mark.parametrize("algorithm", algorithms.ALGORITHMS)
def test_backends_are_monotone_explainable_and_trimmed(algorithm):
    X, y = _xy()
    model = algorithms.fit(algorithm, algorithms.DEFAULT_PARAMS[algorithm], [1, 0, 0],
                           X.iloc[:3000], y[:3000], X.iloc[3000:], y[3000:])
    grid = pd.DataFrame({"a": np.linspace(0, 1, 60), "b": 0.5, "c": 0.5})
    assert np.all(np.diff(model.predict_proba(grid)[:, 1]) >= -1e-9)          # monotone in "a"
    assert algorithms.shap_additivity_error(model, shap.TreeExplainer(model), X.iloc[:300]) < 1e-3
    assert algorithms.iterations_used(model) > 0
    assert np.isfinite(model.predict_proba(X.iloc[:20])[:, 1]).all()          # missing values handled


def test_extended_features_have_lineage_and_a_set():
    assert_feature_lineage(ALL_FEATURES)
    assert len(FEATURE_SETS["extended"]) > len(FEATURE_SETS["core"])
    assert {f.name for f in feature_specs("core")} == set(FEATURE_SETS["core"])
    with pytest.raises(ValueError):
        feature_specs("everything")


def _rows(spec):
    base = dict(currency="USD", merchant_name="M", device_id="D1", ip_location="1",
                historical_average_transaction_amount=100, account_balance_before=1000, failed_attempt_count=0,
                login_attempts=1, debit_credit_flag="Debit", channel="ATM", authorization_method="PIN",
                merchant_category="X", transaction_type="Purchase", customer_id="C1")
    return pd.DataFrame([{**base, "transaction_id": f"t{i}", "transaction_date": d, "transaction_time": t,
                          "transaction_amount": a} for i, (d, t, a) in enumerate(spec)])


def test_extended_features_are_point_in_time():
    df = _rows([("2024-01-01", "10:00:00", 100), ("2024-01-02", "11:00:00", 120), ("2024-01-03", "10:30:00", 950),
                ("2024-01-04", "10:15:00", 9500), ("2024-01-05", "03:00:00", 4700)])
    f = build_features(df)
    assert np.isnan(f["amount_zscore_customer"].iloc[1])                     # needs two prior amounts
    assert f["amount_zscore_customer"].iloc[2] == 50                         # clipped
    assert f["just_below_threshold"].tolist() == [0, 0, 1, 1, 1]
    assert f["cust_near_threshold_7d"].tolist() == [0, 0, 0, 1, 2]
    assert np.isnan(f["hour_deviation"].iloc[2]) and 7 < f["hour_deviation"].iloc[4] < 7.5
    later = pd.concat([df, _rows([("2024-01-06", "09:00:00", 99999)])], ignore_index=True)
    pd.testing.assert_frame_equal(f, build_features(later).iloc[:5])        # future rows change nothing


def test_model_matrix_uses_requested_feature_set():
    f = build_features(_rows([("2024-01-01", "10:00:00", 100)]))
    enc = {c: {"prior": 0.05, "rates": {}} for c in ("channel", "authorization_method", "merchant_category",
                                                     "transaction_type")}
    assert list(to_model_matrix(f, enc, FEATURE_SETS["extended"]).columns) == FEATURE_SETS["extended"]
    assert list(to_model_matrix(f, enc).columns) == FEATURE_SETS["core"]


def test_rolling_folds_validate_on_later_data_only():
    idx = np.arange(1000)
    for fit, val in rolling_folds(idx):
        assert fit.max() < val.min() and len(np.intersect1d(fit, val)) == 0 and len(val) > 0


@pytest.fixture
def small_data(tmp_path, monkeypatch):
    if not CLEAN.exists():
        pytest.skip("processed data not available")
    monkeypatch.setenv("BTI_MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    registry.clear_cache()
    path = tmp_path / "sample.csv"
    pd.read_csv(CLEAN, low_memory=False).sample(15000, random_state=7).to_csv(path, index=False)
    yield path
    registry.clear_cache()


def test_tournament_selects_on_calibration_window_and_respects_incumbent(small_data):
    from bti.modeling.tournament import run_tournament
    first = run_tournament("dev.one", ["hgb"], ["core"], data_path=small_data)
    assert first["decision"]["outcome"] in ("new_challenger", "no_eligible_candidate")
    if first["decision"]["outcome"] == "new_challenger":
        assert registry.model_for_role("challenger") == first["decision"]["challenger"]
    incumbent = registry.model_for_role("challenger")
    second = run_tournament("dev.one", ["lightgbm"], ["core"], data_path=small_data, min_gain=1.0)
    assert second["decision"]["outcome"] in ("incumbent_retained", "no_eligible_candidate")
    assert registry.model_for_role("challenger") == incumbent                 # impossible gain → no change
    assert Path(second["report_path"]).exists()
    assert len(registry.read_index()["models"]) == 2                          # every candidate is on record
    assert all(e["validation"] == "passed" for e in second["entrants"] if e["model_id"] == incumbent)


def test_lightgbm_model_scores_live_and_batch_identically(small_data):
    from bti.modeling.scorer import scorer
    from bti.modeling.train import train
    card = train(small_data, "dev.one", algorithm="lightgbm", feature_set="extended")
    df = pd.read_csv(small_data, low_memory=False).sort_values(["transaction_date", "transaction_time"]).head(400)
    batch = scorer.score_frame(df, role="challenger") if registry.model_for_role("challenger") == card["model_id"] \
        else None
    if batch is None:
        pytest.skip("candidate failed validation and did not take the challenger role")
    last = df.iloc[-1].to_dict()
    live = scorer.score(last, history=df.iloc[:-1], role="challenger")
    assert live.model_id == card["model_id"]
    assert live.fraud_probability == pytest.approx(float(batch["fraud_probability"].iloc[-1]), abs=1e-6)
    assert set(live.contributions) == set(FEATURE_SETS["extended"])


def test_rescore_replaces_legacy_scores_in_stored_history(tmp_path, monkeypatch):
    if not CLEAN.exists() or not registry.read_index().get("challenger"):
        pytest.skip("processed data or registered model not available")
    from datetime import datetime as dt
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from bti.database.models import AuditLog, Base, Transaction
    import bti.modeling.rescore as rescore

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(rescore, "SessionLocal", Session)
    sample = pd.read_csv(CLEAN, low_memory=False).head(300)
    path = tmp_path / "hist.csv"
    sample.to_csv(path, index=False)
    db = Session()
    db.add_all([Transaction(transaction_id=t, customer_id="C", account_id="A", transaction_date=dt(2024, 1, 1),
                            transaction_amount=1.0, final_risk_score=99.0, final_alert_tier="VERY HIGH",
                            rf_fraud_proba=0.9, lr_fraud_proba=0.9) for t in sample["transaction_id"][:50]])
    db.commit()
    db.close()

    summary = rescore.rescore_history(path, role="challenger")
    assert summary["scored"] == 300 and summary["updated_in_db"] == 50
    db = Session()
    rows = db.query(Transaction).all()
    assert all(r.rf_fraud_proba is None and r.lr_fraud_proba is None for r in rows)
    assert all(0 <= r.final_risk_score <= 100 for r in rows)
    assert {r.final_alert_tier for r in rows} <= {"LOW", "MEDIUM", "HIGH", "VERY HIGH", "CRITICAL"}
    assert db.query(AuditLog).filter(AuditLog.event_type == "HISTORY_RESCORED").count() == 1
    db.close()


def test_remediation_mode_excludes_incumbent_and_uses_non_inferiority(small_data):
    from bti.modeling.tournament import run_tournament
    run_tournament("dev.one", ["hgb"], ["core"], data_path=small_data)
    incumbent = registry.read_index()["models"][0]["model_id"]
    registry.assign_role(incumbent, "challenger", "dev.one", "Set incumbent for the remediation test")
    r = run_tournament("dev.one", ["lightgbm", "xgboost"], ["core"], data_path=small_data, min_gain=1.0,
                       remediation="Test finding")
    assert r["decision"]["outcome"] in ("remediation_replacement", "no_eligible_candidate")
    assert r["decision"].get("challenger") != incumbent or r["decision"]["outcome"] == "no_eligible_candidate"
    if r["decision"]["outcome"] == "remediation_replacement":
        assert registry.model_for_role("challenger") == r["decision"]["challenger"] != incumbent
        assert "Test finding" in registry.read_index()["history"][-1]["rationale"]
    assert r["remediation_finding"] == "Test finding"
