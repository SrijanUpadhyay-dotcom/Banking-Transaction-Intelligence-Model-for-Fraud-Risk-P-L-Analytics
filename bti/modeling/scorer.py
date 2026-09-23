"""
BTI v3 real-time scorer.

For each transaction it loads point-in-time history (the customer's own
transactions plus any on the same device or IP within the look-back), runs the
same `build_features` used in training, applies the registered model and
calibrator, and explains the result with exact SHAP reason codes.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit

from bti.governance.reason_codes import principal_reasons
from bti.logging_config import get_logger
from bti.modeling import algorithms, registry
from bti.modeling.features import FEATURE_NAMES, build_features, event_timestamps, to_model_matrix

log = get_logger("modeling.scorer")

HISTORY_COLUMNS = [
    "transaction_id", "customer_id", "transaction_date", "transaction_time", "transaction_amount", "currency",
    "device_id", "ip_location", "merchant_name", "channel", "authorization_method", "merchant_category",
    "transaction_type", "debit_credit_flag", "historical_average_transaction_amount", "account_balance_before",
    "failed_attempt_count", "login_attempts",
]
MAX_HISTORY_ROWS = 5000


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
              role: str = "champion", explain: bool = True) -> V3Score:
        t0 = time.perf_counter()
        model_id, used_role, provisional = self.resolve(role)
        art = registry.load_artifact(model_id)

        if history is None:
            history = fetch_history(db_session, txn, art["lookback_days"])
        frame = pd.concat([history, pd.DataFrame([txn])], ignore_index=True)
        names = art.get("feature_names", FEATURE_NAMES)
        feats = build_features(frame, lookback_days=art["lookback_days"]).iloc[[-1]]
        X = to_model_matrix(feats, art["encodings"], names)

        raw = float(art["estimator"].predict_proba(X)[0, 1])
        p = float(art["calibrator"].predict(np.array([raw]))[0])

        values = {c: (None if pd.isna(v) else (round(float(v), 4) if isinstance(v, (int, float, np.number))
                                                else str(v)))
                  for c, v in feats.iloc[0].items()}
        reasons: List[dict] = []
        contributions: Dict[str, float] = {}
        base_probability = None
        if explain:
            explainer = self._explainer(model_id, art["estimator"])
            sv = algorithms.shap_matrix(explainer, X)
            contributions = {f: round(float(v), 6) for f, v in zip(names, sv[0])}
            reasons = principal_reasons(contributions, values)
            base_raw = float(expit(algorithms.shap_base(explainer)))
            base_probability = round(float(art["calibrator"].predict(np.array([base_raw]))[0]), 6)

        notes = []
        if provisional:
            notes.append("No champion approved yet — scored by the challenger model; treat as provisional.")
        if db_session is None and history is not None and history.empty:
            notes.append("No transaction history supplied — velocity and novelty features are uninformed.")

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
            history_rows_used=int(len(history)),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            notes=notes,
            contributions=contributions,
            base_probability=base_probability,
        )

    def score_frame(self, df: pd.DataFrame, role: str = "champion") -> pd.DataFrame:
        """Batch scoring where the frame itself is the history (backtests, file uploads)."""
        model_id, used_role, _ = self.resolve(role)
        art = registry.load_artifact(model_id)
        feats = build_features(df, lookback_days=art["lookback_days"])
        X = to_model_matrix(feats, art["encodings"], art.get("feature_names", FEATURE_NAMES))
        p = art["calibrator"].predict(art["estimator"].predict_proba(X)[:, 1])
        out = feats.copy()
        out["fraud_probability"] = p
        out["score"] = np.round(p * 1000).astype(int)
        out["model_id"] = model_id
        out["model_role"] = used_role
        return out


scorer = V3Scorer()
