"""
Real-time transaction fraud scoring engine.

Accepts a single transaction dict and returns a complete fraud risk assessment
in <100ms by:
  1. Applying all 19 rule-based checks (standalone + DB-context-aware)
  2. Running the ML ensemble (Isolation Forest + LR + RF)
  3. Computing the composite final_risk_score (0–100)
  4. Assigning the final_alert_tier

DB context (customer history) is queried when a session is provided, enabling
velocity checks that a standalone transaction cannot resolve.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from bti.config import get_settings
from bti.logging_config import get_logger
from bti.scoring.model_loader import ModelBundle

log = get_logger("scoring.realtime")
settings = get_settings()


# ── Rule weights (must match 03_fraud_rules_engine.py) ────────────────────────
RULE_WEIGHTS: Dict[str, int] = {
    "R01_high_value_vs_avg":       9,
    "R02_velocity_spike":          8,
    "R03_failed_auth":             7,
    "R04_multi_login":             6,
    "R05_off_hours":               5,
    "R06_geo_mismatch":            7,
    "R07_repeated_merchant":       6,
    "R08_abnormal_refund":         8,
    "R09_cb_heavy_merchant":       8,
    "R10_amount_outlier":          9,
    "R11_balance_inconsistency":   9,
    "R12_device_ip_mismatch":      8,
    "R13_segment_high_value":      7,
    "R14_channel_concentration":   6,
    "R15_duplicate":               9,
    "R16_rapid_sequential":        6,
    "R17_cross_border":            8,
    "R18_refund_ratio":            7,
    "R19_chargeback_ratio":        8,
}
TOTAL_WEIGHT = sum(RULE_WEIGHTS.values())

HIGH_RISK_CATS = {
    "Gaming & Gambling", "Crypto Exchanges",
    "Financial Services / Money Transfer", "Luxury Goods", "Online Marketplaces",
}
HIGH_CB_CATS = {
    "Gaming & Gambling", "Crypto Exchanges", "Travel & Airlines",
    "Luxury Goods", "Financial Services / Money Transfer",
}
RISKY_CHANNELS = {"USSD", "API/Open Banking", "Call Centre"}


@dataclass
class ScoreResult:
    """Complete fraud risk assessment for a single transaction."""
    transaction_id:      str
    fraud_rule_score:    float        # 0–100 weighted rule score
    rules_triggered:     int
    rules_fired:         List[str]    # names of rules that fired
    ml_iso_score:        float        # raw isolation forest anomaly score
    ml_lr_proba:         float        # logistic regression fraud probability
    ml_rf_proba:         float        # random forest fraud probability
    ml_anomaly_score:    float        # normalised 0–100
    final_risk_score:    float        # composite 0–100
    final_alert_tier:    str          # LOW / MEDIUM / HIGH / VERY HIGH / CRITICAL
    is_suspicious:       bool
    processing_time_ms:  float
    risk_score_input:    float        # pass-through of input risk_score field
    model_version:       str = "unknown"
    db_context_used:     bool = False  # whether customer history was queried


def _apply_standalone_rules(txn: dict) -> Dict[str, int]:
    """
    Rules that can be evaluated from a single transaction record alone.
    Returns {rule_name: 0|1}.
    """
    flags: Dict[str, int] = {}
    amount = float(txn.get("transaction_amount", 0))
    hist_avg = float(txn.get("historical_average_transaction_amount", 0) or 1)
    risk_score = float(txn.get("risk_score", 0))
    failed_auth = int(txn.get("failed_attempt_count", 0))
    login_att = int(txn.get("login_attempts", 1))
    hour_str = str(txn.get("transaction_time", "12:00:00"))[:2]
    hour = int(hour_str) if hour_str.isdigit() else 12
    ip = str(txn.get("ip_location", ""))
    bal_after = float(txn.get("account_balance_after", 0))
    dc_flag = str(txn.get("debit_credit_flag", "Debit"))
    merchant_cat = str(txn.get("merchant_category", ""))
    refund_flag = int(txn.get("refund_flag", 0))
    cb_flag = int(txn.get("chargeback_flag", 0))
    segment = str(txn.get("customer_segment", ""))
    channel = str(txn.get("channel", ""))
    device_id = str(txn.get("device_id", ""))

    # R01 — high value vs average
    ratio = amount / hist_avg if hist_avg else 0
    flags["R01_high_value_vs_avg"] = 1 if ratio >= settings.high_value_ratio else 0

    # R03 — failed auth
    flags["R03_failed_auth"] = 1 if failed_auth >= settings.failed_auth_threshold else 0

    # R04 — multiple login attempts
    flags["R04_multi_login"] = 1 if login_att >= settings.login_attempts_threshold else 0

    # R05 — off hours
    flags["R05_off_hours"] = 1 if hour < settings.off_hours_end else 0

    # R06 — geography mismatch proxy (non-RFC-1918 IP + high risk)
    known_private = ip.startswith(("192.", "10.", "172."))
    flags["R06_geo_mismatch"] = 1 if (not known_private and risk_score > 40) else 0

    # R08 — abnormal refund
    flags["R08_abnormal_refund"] = 1 if (
        refund_flag == 1 and (
            amount > hist_avg * 2 or merchant_cat in HIGH_RISK_CATS
        )
    ) else 0

    # R09 — chargeback in risky merchant category
    flags["R09_cb_heavy_merchant"] = 1 if (
        merchant_cat in HIGH_CB_CATS and cb_flag == 1
    ) else 0

    # R11 — balance goes negative on debit
    flags["R11_balance_inconsistency"] = 1 if (
        dc_flag == "Debit" and bal_after < 0
    ) else 0

    # R12 — device IP mismatch (proxy: high risk + BTI device prefix)
    flags["R12_device_ip_mismatch"] = 1 if (
        risk_score >= settings.high_risk_score and device_id.startswith("DEV-")
    ) else 0

    # R13 — high-risk segment + high value
    flags["R13_segment_high_value"] = 1 if (
        segment in {"Student", "NRI / Diaspora"} and amount > 5_000
    ) else 0

    # R14 — risky channel + high value
    flags["R14_channel_concentration"] = 1 if (
        channel in RISKY_CHANNELS and amount > hist_avg * 2
    ) else 0

    # R17 — cross-border suspicious
    flags["R17_cross_border"] = 1 if (
        merchant_cat in HIGH_CB_CATS and risk_score > 50 and amount > 1_000
    ) else 0

    return flags


def _apply_db_rules(txn: dict, db_session) -> Dict[str, int]:
    """
    Rules that need customer history from the database.
    Returns {rule_name: 0|1}. Falls back to 0 if DB unavailable.
    """
    from bti.database.models import Transaction
    from sqlalchemy import func, cast, Date

    flags: Dict[str, int] = {
        "R02_velocity_spike":   0,
        "R07_repeated_merchant": 0,
        "R10_amount_outlier":   0,
        "R15_duplicate":        0,
        "R16_rapid_sequential": 0,
        "R18_refund_ratio":     0,
        "R19_chargeback_ratio": 0,
    }

    if db_session is None:
        return flags

    customer_id = txn.get("customer_id")
    txn_date = txn.get("transaction_date", "")
    merchant = txn.get("merchant_name", "")
    amount = float(txn.get("transaction_amount", 0))
    segment = str(txn.get("customer_segment", ""))
    refund_flag = int(txn.get("refund_flag", 0))
    cb_flag = int(txn.get("chargeback_flag", 0))

    try:
        # R02 / R16: daily transaction count for this customer
        daily_count = (db_session.query(func.count(Transaction.transaction_id))
                       .filter(Transaction.customer_id == customer_id,
                               Transaction.month_year == str(txn_date)[:7])
                       .scalar() or 0)
        flags["R02_velocity_spike"] = 1 if daily_count > settings.velocity_spike_threshold else 0
        flags["R16_rapid_sequential"] = 1 if daily_count > 5 else 0

        # R07: repeated merchant same customer
        merchant_count = (db_session.query(func.count(Transaction.transaction_id))
                          .filter(Transaction.customer_id == customer_id,
                                  Transaction.merchant_name == merchant)
                          .scalar() or 0)
        flags["R07_repeated_merchant"] = 1 if merchant_count >= 5 else 0

        # R10: z-score within customer segment
        seg_stats = (db_session.query(
            func.avg(Transaction.transaction_amount),
            func.count(Transaction.transaction_id)
        ).filter(Transaction.customer_segment == segment).one())
        seg_mean = float(seg_stats[0] or amount)
        if seg_stats[1] and seg_stats[1] > 1:
            # approximate std from mean (rough but avoids a second query)
            z = abs(amount - seg_mean) / (seg_mean * 0.5 + 1e-9)
            flags["R10_amount_outlier"] = 1 if z > settings.amount_outlier_zscore else 0

        # R15: same customer+amount+merchant (duplicate proxy)
        dup_count = (db_session.query(func.count(Transaction.transaction_id))
                     .filter(Transaction.customer_id == customer_id,
                             Transaction.transaction_amount == amount,
                             Transaction.merchant_name == merchant)
                     .scalar() or 0)
        flags["R15_duplicate"] = 1 if dup_count >= 1 else 0

        # R18: merchant refund ratio
        m_total = (db_session.query(func.count(Transaction.transaction_id))
                   .filter(Transaction.merchant_name == merchant).scalar() or 0)
        m_refund = (db_session.query(func.sum(Transaction.refund_flag))
                    .filter(Transaction.merchant_name == merchant).scalar() or 0)
        flags["R18_refund_ratio"] = 1 if (m_total and m_refund / m_total > settings.refund_ratio_threshold) else 0

        # R19: customer chargeback ratio
        c_total = (db_session.query(func.count(Transaction.transaction_id))
                   .filter(Transaction.customer_id == customer_id).scalar() or 0)
        c_cb = (db_session.query(func.sum(Transaction.chargeback_flag))
                .filter(Transaction.customer_id == customer_id).scalar() or 0)
        flags["R19_chargeback_ratio"] = 1 if (c_total and c_cb / c_total > settings.chargeback_ratio_threshold) else 0

    except Exception as exc:
        log.warning(f"DB context query failed: {exc} — proceeding with standalone rules only")

    return flags


def _build_feature_vector(txn: dict, feature_cols: list):
    """
    Build the ML feature vector as a pandas DataFrame (preserving column names
    so sklearn scalers fitted on DataFrames don't emit feature-name warnings).
    Missing features are filled with 0.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    row = {}
    for col in feature_cols:
        if col.endswith("_enc"):
            base = col[:-4]
            val = txn.get(base, "unknown")
            try:
                row[col] = float(le.fit_transform([str(val)])[0])
            except Exception:
                row[col] = 0.0
        else:
            row[col] = float(txn.get(col, 0) or 0)

    return pd.DataFrame([row], columns=feature_cols)


def score_transaction(
    txn: dict,
    bundle: ModelBundle,
    db_session=None,
    base_risk_score: Optional[float] = None,
) -> ScoreResult:
    """
    Score a single transaction end-to-end.

    Args:
        txn:             Transaction dict (must include at least transaction_id,
                         transaction_amount, customer_id)
        bundle:          Loaded ModelBundle (from load_models())
        db_session:      Optional SQLAlchemy session for DB-context rules
        base_risk_score: Override the risk_score field (useful when the input
                         doesn't carry a pre-computed score)
    """
    t0 = time.perf_counter()

    if base_risk_score is not None:
        txn = {**txn, "risk_score": base_risk_score}

    # ── 1. Rule engine ─────────────────────────────────────────────────────────
    standalone_flags = _apply_standalone_rules(txn)
    db_flags = _apply_db_rules(txn, db_session)
    all_flags = {**standalone_flags, **db_flags}

    weighted_sum = sum(RULE_WEIGHTS[r] * v for r, v in all_flags.items() if r in RULE_WEIGHTS)
    fraud_rule_score = round(weighted_sum / TOTAL_WEIGHT * 100, 2)
    rules_fired = [r for r, v in all_flags.items() if v == 1]
    rules_triggered = len(rules_fired)

    # ── 2. ML ensemble ────────────────────────────────────────────────────────
    X = _build_feature_vector(txn, bundle.feature_cols)
    X = X.fillna(0).replace([float("inf"), float("-inf")], 0)

    # iso_scaler was fitted on a DataFrame (feature names present)
    X_iso = bundle.iso_scaler.transform(X)
    iso_score = float(-bundle.iso_forest.score_samples(X_iso)[0])

    # lr_scaler and rf_model were fitted on numpy arrays (no feature names)
    X_np = X.values
    X_lr = bundle.lr_scaler.transform(X_np)
    lr_proba = float(bundle.lr_model.predict_proba(X_lr)[0, 1])
    rf_proba = float(bundle.rf_model.predict_proba(X_np)[0, 1])

    # Normalise ISO score (approximate: clip to [0,1] using training range)
    iso_norm = min(max(iso_score * 2.5, 0), 1)
    ml_raw = iso_norm * 30 + lr_proba * 30 + rf_proba * 40
    ml_norm = round(min(ml_raw, 100), 2)

    # ── 3. Composite final score ───────────────────────────────────────────────
    risk_input = float(txn.get("risk_score", 50) or 50)
    final_score = round(
        risk_input * 0.35 +
        fraud_rule_score * 0.35 +
        ml_norm * 0.30,
        2
    )
    final_score = min(max(final_score, 0), 100)

    # ── 4. Alert tier ──────────────────────────────────────────────────────────
    if final_score >= 85:
        tier = "CRITICAL"
    elif final_score >= 70:
        tier = "VERY HIGH"
    elif final_score >= 50:
        tier = "HIGH"
    elif final_score >= 25:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    is_suspicious = rules_triggered >= 3 or fraud_rule_score > 40

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return ScoreResult(
        transaction_id=str(txn.get("transaction_id", "UNKNOWN")),
        fraud_rule_score=fraud_rule_score,
        rules_triggered=rules_triggered,
        rules_fired=rules_fired,
        ml_iso_score=round(iso_score, 6),
        ml_lr_proba=round(lr_proba, 4),
        ml_rf_proba=round(rf_proba, 4),
        ml_anomaly_score=ml_norm,
        final_risk_score=final_score,
        final_alert_tier=tier,
        is_suspicious=is_suspicious,
        processing_time_ms=elapsed_ms,
        risk_score_input=risk_input,
        model_version=bundle.trained_at,
        db_context_used=db_session is not None,
    )


class RealTimeScorer:
    """Stateful scorer — holds the loaded bundle in memory. Use as a singleton."""

    def __init__(self):
        self._bundle: Optional[ModelBundle] = None

    @property
    def bundle(self) -> ModelBundle:
        if self._bundle is None:
            from bti.scoring.model_loader import load_models
            self._bundle = load_models()
        return self._bundle

    def score(self, txn: dict, db_session=None) -> ScoreResult:
        return score_transaction(txn, self.bundle, db_session=db_session)

    def reload(self):
        from bti.scoring.model_loader import load_models, invalidate_cache
        invalidate_cache()
        self._bundle = load_models(force_reload=True)
        log.info("RealTimeScorer reloaded models from disk")
