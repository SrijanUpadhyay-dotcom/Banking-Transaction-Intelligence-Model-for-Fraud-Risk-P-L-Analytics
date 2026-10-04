# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
BTI v3 real-time scorer.

For each transaction it loads point-in-time history (the customer's own
transactions plus any on the same device, IP or payee within the look-back, and
the customer's security events when the model uses them), runs the
same `build_features` used in training, applies the registered model and
calibrator, and explains the result with exact SHAP reason codes.

When the online feature store is configured (BTI_FEATURE_STORE_URL), live
scoring reads the history events from Redis and computes the same features in
pure Python (`bti.streaming.online_features`, held equal to the batch features
by the parity test). This is sub-millisecond instead of ~75 ms. The scorer falls
back to the database path when:
- the store is unreachable
- the model is feature version 1–2, uses a feature whose batch value carries
  rounding noise, and has no parity certificate (`bti.streaming.parity`)
- the model uses the security-event feed and the store does not carry it
Every live-scored transaction is written to the store after its features are
computed.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit

from bti.governance.reason_codes import principal_reasons
from bti.logging_config import get_logger
from bti.modeling import algorithms, registry
from bti.modeling.recalibration import apply_overlay, overlay_version
from bti.modeling.features import (
    ALL_FEATURE_NAMES, CATEGORICAL_FEATURES, FEATURE_NAMES, FEEDS, build_features, event_timestamps, model_row,
    to_model_matrix,
)

log = get_logger("modeling.scorer")

HISTORY_COLUMNS = [
    "transaction_id", "customer_id", "transaction_date", "transaction_time", "transaction_amount", "currency",
    "device_id", "ip_location", "merchant_name", "channel", "authorization_method", "merchant_category",
    "transaction_type", "debit_credit_flag", "historical_average_transaction_amount", "account_balance_before",
    "failed_attempt_count", "login_attempts", "latitude", "longitude", "payee_id",
]
MAX_HISTORY_ROWS = 5000
SECURITY_EVENT_COLUMNS = ["customer_id", "event_time", "event_type"]


@dataclass
class V3Score:
    transaction_id: str
    model_id: str
    model_role: str
    provisional: bool
    fraud_probability: float
    score: int                       # 0–1000
    risk_band: str
    reason_codes: List[dict]
    features: Dict[str, Optional[float]]
    history_rows_used: int
    latency_ms: float
    notes: List[str] = field(default_factory=list)
    contributions: Dict[str, float] = field(default_factory=dict)   # SHAP, log-odds, every feature
    base_probability: Optional[float] = None                         # calibrated probability at the SHAP baseline
    model_probability: Optional[float] = None                        # before any recalibration overlay
    calibration_overlay: Optional[int] = None                        # active overlay version, if any
    feature_path: str = "batch"                                      # "online" (feature store) or "batch"
    model_input: Optional[np.ndarray] = field(default=None, repr=False)   # for an asynchronous explanation


_EPOCH = datetime(1970, 1, 1)


def _epoch(txn: dict) -> Optional[int]:
    """Epoch seconds of the transaction, as `event_timestamps` reads them (fast path for ISO strings)."""
    d, t = txn.get("transaction_date"), txn.get("transaction_time")
    if isinstance(d, str) and len(d) >= 10:
        try:
            day = datetime.strptime(d[:10], "%Y-%m-%d") if len(d) == 10 or d[10] in " T" else None
            if day is not None:
                clock = "12:00:00" if t is None or (isinstance(t, float) and t != t) else str(t)[:8]
                h, m, sec = (int(x) for x in clock.split(":"))
                return int((day - _EPOCH).total_seconds()) + h * 3600 + m * 60 + sec
        except (ValueError, TypeError):
            pass
    ts = event_timestamps(pd.DataFrame([txn])).iloc[0]
    return None if pd.isna(ts) else int((ts - pd.Timestamp("1970-01-01")).total_seconds())


def _missing(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and v != v)


def predict_raw(estimator, X: np.ndarray, names: List[str]) -> float:
    """The uncalibrated fraud probability for one row; LightGBM goes straight to the booster."""
    booster = getattr(estimator, "booster_", None)
    if booster is not None and getattr(estimator, "n_classes_", 2) == 2:
        return float(booster.predict(X)[0])                  # identical to LGBMClassifier.predict_proba[:, 1]
    return float(estimator.predict_proba(pd.DataFrame(X, columns=names))[0, 1])


def risk_band(p: float) -> str:
    if p >= 0.90:
        return "CRITICAL"
    if p >= 0.50:
        return "HIGH"
    if p >= 0.10:
        return "MEDIUM"
    return "LOW"


def fetch_history(db_session, txn: dict, lookback_days: int, max_rows: int = MAX_HISTORY_ROWS) -> pd.DataFrame:
    """Customer, device and IP history from the transactions table within the look-back window."""
    from sqlalchemy import or_
    from bti.database.models import Transaction

    if db_session is None:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    ts = event_timestamps(pd.DataFrame([txn])).iloc[0]
    if pd.isna(ts):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    conds = [Transaction.customer_id == txn.get("customer_id")]
    if txn.get("device_id"):
        conds.append(Transaction.device_id == txn["device_id"])
    if txn.get("ip_location"):
        conds.append(Transaction.ip_location == txn["ip_location"])
    if txn.get("payee_id"):
        conds.append(Transaction.payee_id == txn["payee_id"])
    start = (ts - timedelta(days=lookback_days)).to_pydatetime()
    end = (ts + timedelta(days=1)).normalize().to_pydatetime()
    rows = (db_session.query(*[getattr(Transaction, c) for c in HISTORY_COLUMNS])
            .filter(or_(*conds), Transaction.transaction_date >= start, Transaction.transaction_date < end)
            .order_by(Transaction.transaction_date.desc())
            .limit(max_rows).all())
    hist = pd.DataFrame([tuple(r) for r in rows], columns=HISTORY_COLUMNS)
    if not hist.empty and txn.get("transaction_id") is not None:
        hist = hist[hist["transaction_id"].astype(str) != str(txn["transaction_id"])]
    return hist


def fetch_security_events(db_session, txn: dict, days: int = 31) -> Optional[pd.DataFrame]:
    """The customer's security events logged before the transaction (None without a database)."""
    from bti.database.models import SecurityEvent

    if db_session is None:
        return None
    ts = event_timestamps(pd.DataFrame([txn])).iloc[0]
    if pd.isna(ts):
        return None
    rows = (db_session.query(SecurityEvent.customer_id, SecurityEvent.event_time, SecurityEvent.event_type)
            .filter(SecurityEvent.customer_id == txn.get("customer_id"),
                    SecurityEvent.event_time >= (ts - timedelta(days=days)).to_pydatetime(),
                    SecurityEvent.event_time < ts.to_pydatetime())
            .all())
    return pd.DataFrame([tuple(r) for r in rows], columns=SECURITY_EVENT_COLUMNS)


class V3Scorer:
    """Scores with the champion; falls back to the challenger (marked provisional) until one is approved."""

    def __init__(self):
        self._explainers: Dict[str, object] = {}
        self._lock = threading.Lock()

    def resolve(self, role: str = "champion") -> tuple[str, str, bool]:
        model_id = registry.model_for_role(role)
        if model_id:
            return model_id, role, False
        if role == "champion":
            challenger = registry.model_for_role("challenger")
            if challenger:
                return challenger, "challenger", True
        raise registry.RegistryError("No model is registered for scoring. Run: python -m bti.modeling.train")

    def _explainer(self, model_id: str, estimator):
        with self._lock:
            if model_id not in self._explainers:
                import shap
                self._explainers[model_id] = shap.TreeExplainer(estimator)
            return self._explainers[model_id]

    def score(self, txn: dict, db_session=None, history: Optional[pd.DataFrame] = None,
              role: str = "champion", explain: bool = True,
              security_events: Optional[pd.DataFrame] = None, record: bool = False) -> V3Score:
        """Score one transaction. `record=True` writes it to the online feature store (live traffic only)."""
        t0 = time.perf_counter()
        model_id, used_role, provisional = self.resolve(role)
        art = registry.load_artifact(model_id)

        names = art.get("feature_names", FEATURE_NAMES)
        online = None
        if history is None and db_session is not None and security_events is None:
            online = self._online(model_id, art, names, txn)
        if online is not None:
            row, history_rows, event = online
        else:
            if history is None:
                history = fetch_history(db_session, txn, art["lookback_days"])
            if security_events is None and "security_events" in art.get("feeds", []):
                security_events = fetch_security_events(db_session, txn)
            frame = pd.concat([history, pd.DataFrame([txn])], ignore_index=True)
            row = build_features(frame, lookback_days=art["lookback_days"],
                                 feature_version=art.get("feature_version", 1),
                                 security_events=security_events).iloc[-1].to_dict()
            history_rows, event = int(len(history)), None
        if record:
            self._record(txn, row, event)
        graph_note = None
        if (art.get("graph") or {}).get("uses_graph"):
            from bti.graph.snapshot import live_features
            live = live_features(model_id, txn)
            if live is None:
                graph_note = "No graph snapshot yet: network features are unknown (run the nightly graph snapshot)."
            else:
                for col, value in live.items():
                    if col in row:
                        row[col] = np.nan if value is None else float(value)
        X = model_row(row, art["encodings"], names)

        raw = predict_raw(art["estimator"], X, names)
        p_model = float(art["calibrator"].predict(np.array([raw]))[0])
        p = float(apply_overlay(model_id, p_model))

        values = {c: (None if _missing(v) else (round(float(v), 4) if isinstance(v, (int, float, np.number))
                                                 and not isinstance(v, bool) else str(v)))
                  for c, v in row.items()}
        reasons: List[dict] = []
        contributions: Dict[str, float] = {}
        base_probability = None
        if explain:
            contributions, reasons, base_probability = self.explain(model_id, art, X, values)

        notes = []
        if provisional:
            notes.append("No champion approved yet — scored by the challenger model; treat as provisional.")
        if db_session is None and history is not None and history.empty and online is None:
            notes.append("No transaction history supplied — velocity and novelty features are uninformed.")
        if graph_note:
            notes.append(graph_note)
        never_missing = [c for c, share in art.get("train_missing_share", {}).items()
                         if share < 0.001 and c in row and _missing(row[c])]
        if never_missing:
            notes.append(f"Inputs not supplied and scored as unknown: {', '.join(never_missing)}. Send the source "
                         f"fields for a fully informed score.")
        for feed in art.get("feeds", []):
            if all(_missing(row.get(c)) for c in FEEDS[feed]["features"]):
                notes.append(f"The model uses the {feed.replace('_', ' ')} feed, but it gave no signal for this "
                             f"transaction; those features are treated as unknown.")

        return V3Score(
            transaction_id=str(txn.get("transaction_id", "UNKNOWN")),
            model_id=model_id,
            model_role=used_role,
            provisional=provisional,
            fraud_probability=round(p, 6),
            score=int(round(p * 1000)),
            risk_band=risk_band(p),
            reason_codes=reasons,
            features=values,
            history_rows_used=history_rows,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            notes=notes,
            contributions=contributions,
            base_probability=base_probability,
            model_probability=round(p_model, 6),
            calibration_overlay=overlay_version(model_id),
            feature_path="online" if online is not None else "batch",
            model_input=X,
        )

    def explain(self, model_id: str, art: dict, X: np.ndarray, values: Dict[str, object]):
        """(SHAP contributions, principal reasons, base probability) for one model input row."""
        names = art.get("feature_names", FEATURE_NAMES)
        explainer = self._explainer(model_id, art["estimator"])
        sv = algorithms.shap_matrix(explainer, pd.DataFrame(X, columns=names))
        contributions = {f: round(float(v), 6) for f, v in zip(names, sv[0])}
        reasons = principal_reasons(contributions, values)
        base_raw = float(expit(algorithms.shap_base(explainer)))
        base_probability = round(float(apply_overlay(
            model_id, float(art["calibrator"].predict(np.array([base_raw]))[0]))), 6)
        return contributions, reasons, base_probability

    # ── online feature store ──
    def _online(self, model_id: str, art: dict, names: List[str], txn: dict):
        """(features, history rows, store event) from the feature store, or None to use the database path."""
        from bti.streaming import online_features
        from bti.streaming.feature_store import FeatureStoreUnavailable, event_from_online, get_store
        try:
            store = get_store()
        except Exception as exc:                                   # redis client missing or misconfigured
            log.warning("Feature store unavailable", extra={"error": str(exc)})
            return None
        if store is None:
            return None
        from bti.streaming.parity import online_allowed
        version = art.get("feature_version", 1)
        if not online_allowed(model_id, art):          # v1–2 model using a noisy feature, not certified
            return None
        ts = _epoch(txn)
        if ts is None:
            return None
        try:
            customer, entity, security = store.read(txn, ts)
        except FeatureStoreUnavailable as exc:
            log.warning("Feature store read failed; using the database path", extra={"error": str(exc)})
            return None
        if security is None and "security_events" in art.get("feeds", []):
            return None
        f = online_features.compute(txn, ts, customer, entity, security, art["lookback_days"] * 86400,
                                    feature_version=version, categorical=CATEGORICAL_FEATURES)
        row = {c: f.get(c, np.nan) for c in ALL_FEATURE_NAMES}
        return row, len(customer), event_from_online(txn, ts, f)

    @staticmethod
    def _record(txn: dict, row: Dict[str, object], event: Optional[dict]) -> None:
        """Write a live transaction to the store; never fails the score."""
        try:
            from bti.streaming.feature_store import event_from_online, get_store
            store = get_store()
            if store is None:
                return
            if event is None:
                ts = _epoch(txn)
                if ts is None:
                    return
                r = row
                event = event_from_online(txn, ts, {"_usd": r["amount_usd"], "_hour": int(r["txn_hour"]),
                                                    "_near": bool(r["just_below_threshold"]),
                                                    "_share": r["amount_to_balance"],
                                                    "_new_payee_flag": r["payee_new_for_customer"]})
            store.write_events([event], trim=True)
        except Exception as exc:
            log.warning("Feature store write failed", extra={"transaction_id": txn.get("transaction_id"),
                                                             "error": str(exc)})

    def score_frame(self, df: pd.DataFrame, role: str = "champion",
                    security_events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Batch scoring where the frame itself is the history (backtests, file uploads)."""
        model_id, used_role, _ = self.resolve(role)
        art = registry.load_artifact(model_id)
        feats = build_features(df, lookback_days=art["lookback_days"], feature_version=art.get("feature_version", 1),
                               security_events=security_events)
        X = to_model_matrix(feats, art["encodings"], art.get("feature_names", FEATURE_NAMES))
        p = apply_overlay(model_id, art["calibrator"].predict(art["estimator"].predict_proba(X)[:, 1]))
        out = feats.copy()
        out["fraud_probability"] = p
        out["score"] = np.round(p * 1000).astype(int)
        out["model_id"] = model_id
        out["model_role"] = used_role
        return out


scorer = V3Scorer()
