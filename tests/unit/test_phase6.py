# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 6: synthetic rings, the point-in-time entity graph, learned graph features and the live snapshot."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from bti.graph.temporal import GRAPH_BASE_FEATURES, graph_features

CLEAN = Path("data/processed/banking_transactions_clean.csv")


def _txns(rows):
    return pd.DataFrame([dict(zip(("transaction_id", "customer_id", "transaction_date", "transaction_time", "device_id",
                                   "ip_location", "payee_id", "fraud_flag", "label_confirmed_at"), r)) for r in rows])


def test_graph_features_are_point_in_time_with_confirmation_delay():
    df = _txns([
        ("T1", "A", "2024-01-01", "10:00:00", "DEV-1", "1.1.1.1", None, 1, "2024-01-03 09:00:00"),
        ("T2", "B", "2024-01-01", "11:00:00", "DEV-1", "2.2.2.2", None, 0, None),   # same day: not visible
        ("T3", "B", "2024-01-02", "09:00:00", "DEV-9", "2.2.2.2", None, 0, None),   # linked, fraud not yet known
        ("T4", "B", "2024-01-04", "09:00:00", "DEV-9", "2.2.2.2", None, 0, None),   # A's fraud confirmed on the 3rd
        ("T5", "C", "2024-01-04", "09:30:00", "DEV-7", "3.3.3.3", None, 0, None),   # unrelated customer
    ])
    f = graph_features(df).set_index(df["transaction_id"])
    assert f.loc["T2", "graph_shared_entity_customers"] == 0               # A's use of DEV-1 was the same day
    assert f.loc["T3", "graph_component_customers"] == 2 and f.loc["T3", "graph_component_known_fraud"] == 0
    assert f.loc["T4", "graph_component_known_fraud"] == 1
    assert f.loc["T1", "graph_component_known_fraud"] == 0                  # own label is never visible
    assert f.loc["T5", "graph_component_customers"] == 1
    later = pd.concat([df, _txns([("T6", "C", "2024-02-01", "09:00:00", "DEV-1", "1.1.1.1", None, 1, None)])],
                      ignore_index=True)
    pd.testing.assert_frame_equal(graph_features(later).iloc[:5].reset_index(drop=True),
                                  f.reset_index(drop=True), check_names=False)    # future rows change nothing


def test_mule_payee_share_separates_a_mule_from_a_biller():
    rows = []
    for i in range(40):                                                      # a biller with a few frauds by volume
        rows.append((f"B{i}", f"C{i}", "2024-01-01", "10:00:00", f"D{i}", f"I{i}", "BILLER", int(i < 2),
                     "2024-01-02 00:00:00" if i < 2 else None))
    for i in range(4):                                                       # victims paying one mule payee
        rows.append((f"M{i}", f"V{i}", "2024-01-01", "11:00:00", f"DV{i}", f"IV{i}", "MULE", 1, "2024-01-02 00:00:00"))
    rows += [("X1", "Z1", "2024-01-05", "10:00:00", "DZ1", "IZ1", "BILLER", 0, None),
             ("X2", "Z2", "2024-01-05", "10:00:00", "DZ2", "IZ2", "MULE", 0, None)]
    f = graph_features(_txns(rows)).set_index(pd.Series([r[0] for r in rows]))
    assert f.loc["X1", "graph_payee_known_fraud_share"] == pytest.approx(2 / 40)
    assert f.loc["X2", "graph_payee_known_fraud_share"] == 1.0
    assert np.isnan(graph_features(_txns([("Q", "C", "2024-01-01", "10:00:00", "D", "I", None, 0, None)]))
                    ["graph_payee_senders"].iloc[0])


def test_synthetic_rings_are_labelled_and_not_given_away_by_payee_presence():
    if not CLEAN.exists():
        pytest.skip("processed data not available")
    from bti.graph.synthetic_rings import PAYMENT_TYPES, inject_rings
    base = pd.read_csv(CLEAN, low_memory=False).sample(8000, random_state=0)
    r = inject_rings(base, n_rings=4, seed=3)
    d, m = r["data"], r["manifest"]
    assert "SYNTHETIC" in m["notice"] and m["rings"] == 4 and m["ring_rows"] == int(d["synthetic_ring"].sum())
    payment = d["transaction_type"].isin(PAYMENT_TYPES)
    assert d.loc[payment & (d["synthetic_ring"] == 0), "payee_id"].notna().all()     # every payment has a payee
    ring = d[d["synthetic_ring"] == 1]
    assert set(ring.loc[ring["fraud_flag"] == 1, "fraud_type"]) == {"Authorised Push Payment", "Mule Account"}
    ts = pd.to_datetime(ring["transaction_date"] + " " + ring["transaction_time"])
    conf = pd.to_datetime(ring["label_confirmed_at"])
    assert (conf[ring["fraud_flag"] == 1] > ts[ring["fraud_flag"] == 1]).all()      # confirmed after the event


def test_spectral_similarity_is_zero_for_unlinked_customers():
    from scipy import sparse
    from bti.graph.learned import spectral_similarity
    customers = pd.Index(["A", "B", "C", "D", "E", "F"])
    # A-B-C share entity 0/1 (a ring, A known fraud); D, E, F each alone on their own entity
    rows, cols = [0, 1, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 3, 4]
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(6, 5))
    sim = spectral_similarity(customers, B, {"A"})
    assert sim[1] > 0.5 and sim[2] > 0 and (sim[3:] == 0).all()


@pytest.mark.skipif(not __import__("bti.graph.learned", fromlist=["x"]).torch_available(), reason="PyTorch not installed")
def test_learned_features_train_only_on_the_training_window():
    if not CLEAN.exists():
        pytest.skip("processed data not available")
    from bti.graph.learned import LEARNED_FEATURES, learned_features
    from bti.graph.synthetic_rings import inject_rings
    d = inject_rings(pd.read_csv(CLEAN, low_memory=False).sample(6000, random_state=1), n_rings=6, seed=5)["data"]
    ts = pd.to_datetime(d["transaction_date"])
    r = learned_features(d, ts.quantile(0.6), freq="28D")
    f = r["features"]
    assert list(f.columns) == LEARNED_FEATURES and r["sage_state"]["training_snapshots"] >= 4
    ring = d["synthetic_ring"].to_numpy() == 1
    assert np.nanmean(f.loc[ring, "graph_sage_score"]) > np.nanmean(f.loc[~ring, "graph_sage_score"])
    sim = f["graph_embed_fraud_similarity"].dropna()
    assert ((sim >= 0) & (sim <= 1)).all()          # weak on small samples: ring mules are confirmed after the fact


def test_nightly_snapshot_serves_the_same_features_live(tmp_path, monkeypatch):
    from bti.config import get_settings
    from bti.database.models import Base, FraudLabel, Transaction
    from bti.graph import snapshot
    monkeypatch.setattr(get_settings(), "graph_snapshot_path", str(tmp_path / "snap.joblib"))
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    for i, (cust, dev, day) in enumerate((("A", "DEV-S", 1), ("B", "DEV-S", 2), ("C", "DEV-S", 3), ("D", "DEV-X", 3))):
        db.add(Transaction(transaction_id=f"T{i}", customer_id=cust, account_id=cust, transaction_amount=10.0,
                           transaction_date=datetime(2024, 1, day), transaction_time="10:00:00", device_id=dev,
                           ip_location=f"IP{i}", fraud_flag=0))
    db.add(FraudLabel(transaction_id="T0", label=1, label_source="CHARGEBACK", event_at=datetime(2024, 1, 1),
                      created_at=datetime(2024, 1, 2)))
    db.commit()
    assert snapshot.live_features("m", {"customer_id": "A"}) is None                   # nothing built yet
    built = snapshot.build_snapshot(db)
    assert built["status"] == "built" and built["known_fraud_customers"] == 1
    f = snapshot.live_features("m", {"customer_id": "B", "device_id": "DEV-S", "ip_location": "IP9"})
    assert f["graph_component_customers"] == 3 and f["graph_component_known_fraud"] == 1
    assert f["graph_shared_entity_known_fraud"] == 1 and f["graph_sage_score"] is None
    assert snapshot.status()["status"] == "ok"
