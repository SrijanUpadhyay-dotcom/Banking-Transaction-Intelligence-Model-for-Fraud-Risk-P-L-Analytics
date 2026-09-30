# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Nightly graph snapshot for live scoring.

The job rebuilds the point-in-time entity graph from the transaction history in the database, with the same
code training uses (bti.graph.temporal).
- **Fraud confirmation times:** taken from the labels table (when each label was recorded). History rows
  with no recorded label use the assumed confirmation delay, as in training.
- **Learned scores:** for every scoring model that uses learned graph features (champion and challenger), the
  latest-snapshot spectral similarity and GraphSAGE score are computed with that model's own fitted weights.

At scoring time the scorer reads this snapshot. Graph features are therefore at most one day old, which is
the same lag training uses.

The snapshot is a single file (`graph.snapshot_path`). At a bank's scale the same state belongs in a keyed
store (Redis or the warehouse) behind the same `features()` call.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import AuditLog, FraudLabel, Transaction
from bti.graph.temporal import GRAPH_BASE_FEATURES, graph_features
from bti.logging_config import get_logger
from bti.modeling import registry

log = get_logger("graph.snapshot")
_lock = threading.Lock()
_cache: Dict = {"mtime": None, "snapshot": None}


def snapshot_path() -> Path:
    return Path(get_settings().graph_snapshot_path)


def _history(db: Session) -> pd.DataFrame:
    rows = db.query(Transaction.transaction_id, Transaction.customer_id, Transaction.device_id,
                    Transaction.ip_location, Transaction.payee_id, Transaction.transaction_date,
                    Transaction.transaction_time, Transaction.fraud_flag).all()
    df = pd.DataFrame(rows, columns=["transaction_id", "customer_id", "device_id", "ip_location", "payee_id",
                                     "transaction_date", "transaction_time", "fraud_flag"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.strftime("%Y-%m-%d")
    labels = pd.DataFrame(db.query(FraudLabel.transaction_id, FraudLabel.label, FraudLabel.created_at,
                                   FraudLabel.id).all(), columns=["transaction_id", "label", "created_at", "id"])
    if not labels.empty:
        latest = labels.sort_values(["created_at", "id"]).drop_duplicates("transaction_id", keep="last")
        latest = latest.set_index("transaction_id")
        hit = df["transaction_id"].isin(latest.index)
        df.loc[hit, "fraud_flag"] = df.loc[hit, "transaction_id"].map(latest["label"])
        df["label_confirmed_at"] = df["transaction_id"].map(latest["created_at"])
    return df


def build_snapshot(db: Session) -> Dict:
    s = get_settings()
    df = _history(db)
    if df.empty:
        return {"status": "no_history"}
    _, graph = graph_features(df, delay_days=s.graph_label_delay_days, return_graph=True)
    now = pd.Timestamp(datetime.utcnow())
    graph.confirm_until(now)
    learned = {}
    for role in ("champion", "challenger"):
        model_id = registry.model_for_role(role)
        if not model_id:
            continue
        g = registry.load_artifact(model_id).get("graph") or {}
        names = registry.load_artifact(model_id).get("feature_names", [])
        if {"graph_embed_fraud_similarity", "graph_sage_score"} & set(names):
            from bti.graph.learned import latest_scores
            learned[model_id] = latest_scores(df, g.get("sage_state"), s.graph_label_delay_days)
    snapshot = {"built_at": now.isoformat(), "rows": int(len(df)), "graph": graph, "learned": learned,
                "label_delay_days": s.graph_label_delay_days}
    path = snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(snapshot, tmp, compress=3)
    os.replace(tmp, path)
    summary = {"built_at": snapshot["built_at"], "rows": snapshot["rows"], "customers": len(graph.comp_customers),
               "known_fraud_customers": len(graph.known), "learned_models": list(learned)}
    db.add(AuditLog(ts=datetime.utcnow(), event_type="GRAPH_SNAPSHOT", payload=summary))
    db.commit()
    with _lock:
        _cache.update(mtime=None, snapshot=None)
    return {"status": "built", **summary}


def load_snapshot() -> Optional[Dict]:
    path = snapshot_path()
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    with _lock:
        if _cache["mtime"] != mtime:
            _cache.update(mtime=mtime, snapshot=joblib.load(path))
        return _cache["snapshot"]


def live_features(model_id: str, txn: Dict) -> Optional[Dict[str, float]]:
    """Graph features for one transaction from the latest snapshot (None when no snapshot exists)."""
    snap = load_snapshot()
    if snap is None:
        return None
    customer = str(txn.get("customer_id"))
    out = snap["graph"].features(customer, txn.get("device_id"), txn.get("ip_location"), txn.get("payee_id"))
    sim, sage = snap["learned"].get(model_id, {}).get(customer, (None, None))
    out["graph_embed_fraud_similarity"] = sim
    out["graph_sage_score"] = sage
    return out


def status() -> Dict:
    snap = load_snapshot()
    if snap is None:
        return {"status": "none", "path": str(snapshot_path())}
    return {"status": "ok", "built_at": snap["built_at"], "rows": snap["rows"],
            "known_fraud_customers": len(snap["graph"].known), "learned_models": list(snap["learned"]),
            "features": GRAPH_BASE_FEATURES}
