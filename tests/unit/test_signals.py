"""
Feed-dependent signals: impossible travel, payee velocity, time since security event.

The synthetic dataset has no transaction location, payee key or security-event
log, so these tests use small hand-built feeds. They prove the plumbing
(point-in-time correctness, feed gating, live/batch parity), not predictive value.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bti.governance.reason_codes import REASON_CODES
from bti.modeling import registry
from bti.modeling.features import (
    FEATURE_SETS, FEEDS, NO_RECENT_EVENT_HOURS, SIGNAL_FEATURES, assert_feature_lineage, build_features,
    feed_coverage, feeds_required,
)

CLEAN = Path("data/processed/banking_transactions_clean.csv")
LONDON, NEW_YORK, READING = (51.5074, -0.1278), (40.7128, -74.0060), (51.4543, -0.9781)


def _rows(spec):
    base = dict(currency="GBP", merchant_name="M", device_id="D1", ip_location="1", channel="Mobile Banking",
                historical_average_transaction_amount=100, account_balance_before=1000, failed_attempt_count=0,
                login_attempts=1, debit_credit_flag="Debit", authorization_method="PIN", merchant_category="X",
                transaction_type="Transfer", transaction_amount=100)
    return pd.DataFrame([{**base, "transaction_id": f"t{i}", **row} for i, row in enumerate(spec)])


def test_signal_features_are_governed_and_explainable():
    assert_feature_lineage(SIGNAL_FEATURES)
    assert all(f.reason_code in REASON_CODES for f in SIGNAL_FEATURES)
    for name in ("core", "extended", "core-relative", "extended-relative"):
        assert not feeds_required(FEATURE_SETS[name])                      # never in a default set
    assert set(feeds_required(FEATURE_SETS["signals-relative"])) == set(FEEDS)


def test_impossible_travel_is_point_in_time():
    df = _rows([
        dict(customer_id="C1", transaction_date="2024-01-01", transaction_time="10:00:00",
             latitude=LONDON[0], longitude=LONDON[1]),
        dict(customer_id="C1", transaction_date="2024-01-01", transaction_time="11:00:00",
             latitude=NEW_YORK[0], longitude=NEW_YORK[1]),                  # 5,570 km in an hour
        dict(customer_id="C1", transaction_date="2024-01-01", transaction_time="12:00:00"),   # no location
        dict(customer_id="C1", transaction_date="2024-01-03", transaction_time="12:00:00",
             latitude=LONDON[0], longitude=LONDON[1]),                      # back two days later: plausible
        dict(customer_id="C2", transaction_date="2024-01-03", transaction_time="12:30:00",
             latitude=READING[0], longitude=READING[1]),                    # other customer: no prior
    ])
    f = build_features(df)
    assert np.isnan(f["geo_distance_prev_km"].iloc[0]) and np.isnan(f["impossible_travel"].iloc[0])
    assert f["geo_distance_prev_km"].iloc[1] == pytest.approx(5570, rel=0.01)
    assert f["impossible_travel"].iloc[1] == 1 and f["geo_speed_kmh"].iloc[1] > 5000
    assert np.isnan(f["impossible_travel"].iloc[2])
    assert f["impossible_travel"].iloc[3] == 0                             # measured from New York, 49h earlier
    assert np.isnan(f["geo_distance_prev_km"].iloc[4])
    later = pd.concat([df, _rows([dict(customer_id="C1", transaction_date="2024-01-04", transaction_time="09:00:00",
                                       latitude=-33.9, longitude=151.2)])], ignore_index=True)
    pd.testing.assert_frame_equal(f, build_features(later).iloc[:5])     # future rows change nothing


def test_payee_signals_detect_new_payees_and_mule_pattern():
    day = "2024-02-01"
    df = _rows([dict(customer_id="C1", transaction_date="2024-01-20", transaction_time="09:00:00", payee_id="P-OLD")]
               + [dict(customer_id="C1", transaction_date=day, transaction_time="09:00:00", payee_id="P-OLD")]
               + [dict(customer_id=f"V{i}", transaction_date="2024-01-30", transaction_time="10:00:00",
                       payee_id="P-MULE") for i in range(4)]
               + [dict(customer_id="C1", transaction_date=day, transaction_time=f"1{i}:00:00", payee_id=f"P-NEW{i}")
                  for i in range(3)]
               + [dict(customer_id="C1", transaction_date=day, transaction_time="15:00:00", payee_id="P-MULE"),
                  dict(customer_id="C1", transaction_date=day, transaction_time="16:00:00")])
    f = build_features(df)
    assert f["payee_new_for_customer"].iloc[1] == 0                       # paid before
    assert f["payee_new_for_customer"].iloc[6:9].tolist() == [1, 1, 1]
    assert f["cust_new_payees_24h"].iloc[6:9].tolist() == [0, 1, 2]
    assert f["payee_other_customer_txns_7d"].iloc[9] == 4                 # four other customers paid the mule
    assert np.isnan(f["payee_new_for_customer"].iloc[10])                 # no payee on this transaction
    assert np.isnan(f["payee_new_for_customer"].iloc[2])                  # customer with no history: unknown


def test_security_event_signals_only_see_earlier_events():
    df = _rows([dict(customer_id="C1", transaction_date="2024-03-10", transaction_time="12:00:00"),
                dict(customer_id="C2", transaction_date="2024-03-10", transaction_time="12:00:00"),
                dict(customer_id="C3", transaction_date="2024-03-10", transaction_time="12:00:00")])
    events = pd.DataFrame([
        {"customer_id": "C1", "event_time": "2024-03-10 10:00:00", "event_type": "sim_swap"},
        {"customer_id": "C1", "event_time": "2024-03-09 12:00:00", "event_type": "password_reset"},
        {"customer_id": "C1", "event_time": "2024-03-10 13:00:00", "event_type": "email_change"},   # after: ignored
        {"customer_id": "C2", "event_time": "2024-01-01 12:00:00", "event_type": "password_reset"},  # > 30 days
    ])
    f = build_features(df, security_events=events)
    assert f["hours_since_security_event"].tolist() == [2.0, NO_RECENT_EVENT_HOURS, NO_RECENT_EVENT_HOURS]
    assert f["hours_since_sim_swap"].tolist() == [2.0, NO_RECENT_EVENT_HOURS, NO_RECENT_EVENT_HOURS]
    assert f["security_events_7d"].tolist() == [2, 0, 0]
    without = build_features(df)
    assert without[["hours_since_security_event", "security_events_7d"]].isna().all().all()   # feed absent


def test_training_refuses_signal_features_without_the_feeds(tmp_path, monkeypatch):
    if not CLEAN.exists():
        pytest.skip("processed data not available")
    from bti.modeling.train import FeedUnavailableError, check_feeds, prepare
    path = tmp_path / "sample.csv"
    pd.read_csv(CLEAN, low_memory=False).sample(3000, random_state=1).to_csv(path, index=False)
    data = prepare(path)
    assert data.feeds["train"] == {"location": 0.0, "payee": 0.0, "security_events": 0.0}
    check_feeds(data, "extended-relative")
    with pytest.raises(FeedUnavailableError, match="location"):
        check_feeds(data, "signals-relative")


def _with_test_feeds(df: pd.DataFrame, seed: int = 5):
    """TEST FEED ONLY — random locations, payees and events; carries no fraud signal by design."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["latitude"] = rng.uniform(-40, 60, len(out))
    out["longitude"] = rng.uniform(-120, 140, len(out))
    out["payee_id"] = [f"P{int(i)}" for i in rng.integers(0, 400, len(out))]
    ts = pd.to_datetime(out["transaction_date"]) - pd.to_timedelta(rng.integers(1, 72, len(out)), unit="h")
    pick = rng.random(len(out)) < 0.2
    events = pd.DataFrame({"customer_id": out.loc[pick, "customer_id"].to_numpy(),
                           "event_time": ts[pick].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                           "event_type": rng.choice(["sim_swap", "password_reset", "device_enrolment"], pick.sum())})
    return out, events


def test_signal_model_trains_with_feeds_and_scores_live_like_batch(tmp_path, monkeypatch):
    if not CLEAN.exists():
        pytest.skip("processed data not available")
    monkeypatch.setenv("BTI_MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    registry.clear_cache()
    from bti.modeling.scorer import scorer
    from bti.modeling.train import MIN_FEED_COVERAGE, prepare, train_candidate

    df, events = _with_test_feeds(pd.read_csv(CLEAN, low_memory=False).sample(12000, random_state=3))
    path, events_path = tmp_path / "sample.csv", tmp_path / "events.csv"
    df.to_csv(path, index=False)
    events.to_csv(events_path, index=False)
    data = prepare(path, security_events_path=events_path)
    assert all(c >= MIN_FEED_COVERAGE for w in data.feeds.values() for c in w.values())
    card = train_candidate(data, "lightgbm", "signals-relative", developer="dev.test", auto_challenger=False)
    assert card["data"]["feed_coverage"]["train"]["security_events"] == 1.0   # every row: event age or "none"
    registry.assign_role(card["model_id"], "challenger", "dev.test", "Plumbing test for feed-dependent signals")
    assert registry.load_artifact(card["model_id"])["feeds"] == list(FEEDS)

    ordered = df.sort_values(["transaction_date", "transaction_time"]).tail(300)
    batch = scorer.score_frame(ordered, role="challenger", security_events=events)
    last = ordered.iloc[-1].to_dict()
    live = scorer.score(last, history=ordered.iloc[:-1], role="challenger", security_events=events)
    assert live.fraud_probability == pytest.approx(float(batch["fraud_probability"].iloc[-1]), abs=1e-6)
    assert set(live.contributions) == set(FEATURE_SETS["signals-relative"])
    registry.clear_cache()


def test_feed_coverage_reports_share_of_rows_with_signal():
    df = _rows([dict(customer_id="C1", transaction_date="2024-01-01", transaction_time="10:00:00",
                     latitude=1.0, longitude=1.0),
                dict(customer_id="C1", transaction_date="2024-01-02", transaction_time="10:00:00",
                     latitude=1.0, longitude=2.0)])
    cov = feed_coverage(build_features(df))
    assert cov["location"] == 0.5 and cov["payee"] == 0.0 and cov["security_events"] == 0.0
