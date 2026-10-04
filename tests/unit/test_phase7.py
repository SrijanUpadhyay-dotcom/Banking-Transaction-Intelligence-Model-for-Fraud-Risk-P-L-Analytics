# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 7: feature version 3, online/batch feature parity, the Redis feature store and the scorer's fast path."""

import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bti.modeling.features import build_features, event_timestamps
from bti.streaming import online_features
from bti.streaming.feature_store import events_from_frame
from bti.streaming.parity import MemoryHistory, parity_report

CLEAN = Path("data/processed/banking_transactions_clean.csv")


def _frame(rows):
    base = {"currency": "USD", "account_balance_before": 1000.0, "historical_average_transaction_amount": 50.0,
            "debit_credit_flag": "Debit", "channel": "Mobile Banking", "authorization_method": "PIN",
            "merchant_category": "Retail", "transaction_type": "Purchase", "failed_attempt_count": 0,
            "login_attempts": 1, "device_id": "D1", "ip_location": "1.1.1.1", "merchant_name": "M1",
            "latitude": None, "longitude": None, "payee_id": None}
    return pd.DataFrame([{**base, **dict(zip(("transaction_id", "customer_id", "transaction_date", "transaction_time",
                                               "transaction_amount"), r))} for r in rows])


def _redis_url():
    import redis
    for url in filter(None, (os.environ.get("BTI_TEST_REDIS_URL"), "redis://localhost:6379/15",
                             "redis://localhost:6390/15")):
        try:
            if redis.Redis.from_url(url, socket_connect_timeout=0.2).ping():
                return url
        except Exception:
            continue
    return None


def test_v3_window_sums_are_exact_per_customer():
    rows = [(f"T{i}", "A" if i % 2 else "B", "2024-01-01", f"{10 + i // 6:02d}:{(i * 7) % 60:02d}:00", 0.1 + i * 1e5)
            for i in range(40)]
    df = _frame(rows)
    f = build_features(df, feature_version=3)
    ts = event_timestamps(df)
    for i in range(len(df)):
        prior = df[(df["customer_id"] == df.at[i, "customer_id"]) & (ts < ts[i]) & (ts >= ts[i] - pd.Timedelta("24h"))]
        assert f.at[i, "cust_amount_usd_24h"] == pytest.approx(prior["transaction_amount"].sum(), rel=1e-12, abs=1e-9)


def test_v3_degenerate_cases_are_unknown_and_v2_is_unchanged():
    # past hours 00:00 and 12:00 twice each: no usual hour; four identical past amounts: no spread
    rows = [("T0", "A", "2024-01-01", "00:00:00", 25.0), ("T1", "A", "2024-01-01", "12:00:00", 25.0),
            ("T2", "A", "2024-01-02", "00:00:00", 25.0), ("T3", "A", "2024-01-02", "12:00:00", 25.0),
            ("T4", "A", "2024-01-03", "06:00:00", 80.0)]
    v3, v2 = build_features(_frame(rows), feature_version=3), build_features(_frame(rows), feature_version=2)
    assert np.isnan(v3.at[4, "hour_deviation"]) and not np.isnan(v2.at[4, "hour_deviation"])
    assert np.isnan(v3.at[4, "amount_zscore_customer"])
    same = [c for c in v2.columns if c not in ("hour_deviation", "amount_zscore_customer")]
    pd.testing.assert_frame_equal(v3[same], v2[same])


def test_online_features_match_batch_on_a_small_frame():
    rows = [(f"T{i}", "AB"[i % 2], f"2024-01-{1 + i // 8:02d}", f"{(i * 5) % 24:02d}:{i % 60:02d}:00", 10 + i * 3.7)
            for i in range(48)]
    df = _frame(rows)
    df.loc[df.index % 5 == 0, "device_id"] = "D-SHARED"
    df.loc[df.index % 3 == 0, "payee_id"] = "P1"
    r = parity_report(df, sample=len(df), feature_version=3)
    assert r["exact"], r["mismatches"]
    assert set(parity_report(df, sample=len(df), feature_version=2)["mismatches"]) <= online_features.V2_INEXACT


def test_online_features_match_batch_on_real_transactions():
    if not CLEAN.exists():
        pytest.skip("processed data not available")
    df = pd.read_csv(CLEAN, low_memory=False).sample(20000, random_state=3)
    r = parity_report(df, sample=1500, feature_version=3)
    assert r["exact"] and r["passing"] == r["features"], r["mismatches"]
    v2 = parity_report(df, sample=1500, feature_version=2)
    assert set(v2["mismatches"]) <= online_features.V2_INEXACT          # only the known noisy v2 features


def test_feature_store_round_trip_through_redis():
    url = _redis_url()
    if url is None:
        pytest.skip("no Redis server reachable")
    from bti.streaming.feature_store import FeatureStore
    rows = [(f"T{i}", "AB"[i % 2], f"2024-01-{1 + i // 6:02d}", f"{(i * 5) % 24:02d}:{i % 60:02d}:00", 20 + i)
            for i in range(36)]
    df = _frame(rows)
    store = FeatureStore(url, prefix=f"bti:test:{uuid.uuid4().hex}:")
    try:
        store.write_events(events_from_frame(df))
        memory = parity_report(df, sample=len(df))
        live = parity_report(df, sample=len(df), store=store)
        assert live["exact"] and memory["exact"]
        txn = df.iloc[-1].to_dict()
        ts = int((event_timestamps(df).iloc[-1] - pd.Timestamp("1970-01-01")).total_seconds())
        customer, entity, security = store.read(txn, ts)
        assert all(e["ts"] < ts for e in customer) and security is None       # no security feed configured
        store.write_events(events_from_frame(df))                              # re-delivery is idempotent
        assert len(store.read(txn, ts)[0]) == len(customer)
        store.write_security_events([("A", ts - 3600, "sim_swap", "E1")])
        assert store.read(txn, ts)[2] is not None
    finally:
        store.flush()


def test_memory_history_excludes_the_transaction_itself():
    df = _frame([("T0", "A", "2024-01-01", "10:00:00", 5.0), ("T1", "A", "2024-01-01", "10:00:00", 6.0),
                 ("T2", "A", "2024-01-01", "10:00:01", 7.0)])
    h = MemoryHistory(events_from_frame(df))
    ts = int((event_timestamps(df).iloc[1] - pd.Timestamp("1970-01-01")).total_seconds())
    assert h.read(df.iloc[1].to_dict(), ts)[0] == []                       # same second: not yet visible
    assert len(h.read(df.iloc[2].to_dict(), ts + 1)[0]) == 2


def test_v2_models_need_a_certificate_for_the_fast_path(tmp_path, monkeypatch):
    from bti.streaming import parity
    monkeypatch.setattr(parity, "CERT_DIR", tmp_path)
    parity._certified.cache_clear()
    noisy = {"feature_version": 2, "feature_names": ["cust_amount_usd_7d", "txn_hour"]}
    assert parity.online_allowed("m3", {"feature_version": 3, "feature_names": ["cust_amount_usd_7d"]})
    assert parity.online_allowed("m2-clean", {"feature_version": 2, "feature_names": ["txn_hour"]})
    assert not parity.online_allowed("m2", noisy)
    (tmp_path / "m2.json").write_text('{"model_id": "m2", "certified": true}')
    parity._certified.cache_clear()
    assert parity.online_allowed("m2", noisy)
    (tmp_path / "m2.json").write_text('{"model_id": "m2", "certified": false}')
    parity._certified.cache_clear()
    assert not parity.online_allowed("m2", noisy)


class _FakeStore:
    """A feature store over memory, recording writes."""

    def __init__(self, df, fail=False):
        self.events = events_from_frame(df)
        self.fail = fail

    def read(self, txn, ts):
        from bti.streaming.feature_store import FeatureStoreUnavailable
        if self.fail:
            raise FeatureStoreUnavailable("down")
        return MemoryHistory(self.events).read(txn, ts)

    def write_events(self, events, trim=False):
        self.events.extend(events)


def test_scorer_uses_the_store_and_falls_back_when_it_is_down(monkeypatch):
    from bti.modeling import registry
    from bti.modeling.scorer import V3Scorer
    from bti.streaming import feature_store, parity
    model_id = registry.model_for_role("champion") or registry.model_for_role("challenger")
    if not model_id or not CLEAN.exists():
        pytest.skip("no registered model or processed data")
    art = registry.load_artifact(model_id)
    if not parity.online_allowed(model_id, art):
        pytest.skip("model not certified for the online path")
    df = pd.read_csv(CLEAN, low_memory=False)
    cust = df["customer_id"].value_counts().index[0]
    hist = df[df["customer_id"] == cust].sort_values(["transaction_date", "transaction_time"])
    txn = hist.iloc[-1].to_dict()
    history = df[pd.to_datetime(df["transaction_date"]) <= pd.to_datetime(txn["transaction_date"])]
    s = V3Scorer()
    batch = s.score(txn, history=history[history["transaction_id"] != txn["transaction_id"]], explain=False)
    store = _FakeStore(history[history["transaction_id"] != txn["transaction_id"]])
    monkeypatch.setattr(feature_store, "get_store", lambda: store)
    before = len(store.events)
    online = s.score(txn, db_session=object(), explain=False, record=True)
    assert online.feature_path == "online" and len(store.events) == before + 1
    assert online.fraud_probability == pytest.approx(batch.fraud_probability, abs=1e-9)
    store.fail = True
    monkeypatch.setattr("bti.modeling.scorer.fetch_history", lambda db, t, days: history[
        history["transaction_id"] != txn["transaction_id"]])
    fallback = s.score(txn, db_session=object(), explain=False)
    assert fallback.feature_path == "batch"
    assert fallback.fraud_probability == pytest.approx(batch.fraud_probability, abs=1e-9)


def test_numpy_model_row_and_booster_match_the_training_path():
    from bti.modeling import registry
    from bti.modeling.features import model_row, to_model_matrix
    from bti.modeling.scorer import predict_raw
    model_id = registry.model_for_role("champion") or registry.model_for_role("challenger")
    if not model_id or not CLEAN.exists():
        pytest.skip("no registered model or processed data")
    art = registry.load_artifact(model_id)
    names = art["feature_names"]
    df = pd.read_csv(CLEAN, low_memory=False).sample(3000, random_state=9)
    df.loc[df.index[:50], "channel"] = None                                # missing and unseen categories
    df.loc[df.index[50:100], "channel"] = "Carrier Pigeon"
    feats = build_features(df, feature_version=art.get("feature_version", 1))
    X = to_model_matrix(feats, art["encodings"], names)
    rows = np.vstack([model_row(r, art["encodings"], names) for r in feats.to_dict("records")])
    np.testing.assert_array_equal(rows, X.to_numpy())
    slow = art["estimator"].predict_proba(X)[:, 1]
    fast = np.array([predict_raw(art["estimator"], rows[[i]], names) for i in range(len(rows))])
    np.testing.assert_array_equal(fast, slow)


def test_async_explanations_fill_the_score_log_and_match_inline(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from bti.config import get_settings
    from bti.database.models import AuditLog, Base, ScoreLog
    from bti.modeling import registry
    from bti.operations import explanations
    from bti.operations.scoring_service import score_and_decide
    if not (registry.model_for_role("champion") or registry.model_for_role("challenger")):
        pytest.skip("no registered model")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    txn = {"transaction_id": "ASYNC-1", "customer_id": "C-ASYNC", "transaction_date": "2024-06-01",
           "transaction_time": "03:10:00", "transaction_amount": 4200.0, "currency": "USD", "channel": "Mobile Banking",
           "historical_average_transaction_amount": 80.0, "account_balance_before": 5000.0, "device_id": "D-NEW"}
    settings = get_settings()
    monkeypatch.setattr(settings, "feature_store_url", "")
    monkeypatch.setattr(settings, "explain_mode", "inline")
    inline = score_and_decide(dict(txn), None)
    monkeypatch.setattr(settings, "explain_mode", "async")
    sd = score_and_decide(dict(txn), db)
    assert sd.explanation == "pending" and sd.live.reason_codes == []
    assert sd.live.fraud_probability == inline.live.fraud_probability and sd.decision.action == inline.decision.action
    explanations.drain(30)
    got = explanations.get("ASYNC-1", db)
    assert got["status"] == "ready"
    assert [r["code"] for r in got["reason_codes"]] == [r["code"] for r in inline.live.reason_codes]
    db.expire_all()
    row = db.query(ScoreLog).filter_by(transaction_id="ASYNC-1", is_shadow=False).one()
    assert [r["code"] for r in row.reason_codes] == [r["code"] for r in inline.live.reason_codes]
    assert db.query(AuditLog).filter_by(event_type="EXPLANATION_ADDED", transaction_id="ASYNC-1").count() == 1
