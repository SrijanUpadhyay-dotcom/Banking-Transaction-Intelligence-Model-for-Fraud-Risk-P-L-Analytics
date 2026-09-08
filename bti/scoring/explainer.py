"""
SHAP-based Explainability Layer — Phase 1 of BTI v2 (Fraud Intelligence Layer)

For every scored transaction, produces:
  - Per-feature SHAP contributions (how much each feature pushed the score up/down)
  - Top-N drivers ranked by absolute impact
  - A plain-English narrative summary for fraud investigators

Uses TreeExplainer for the Random Forest (exact Shapley values, no approximation).
The explainer is built once and cached in-process.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from bti.logging_config import get_logger

log = get_logger("scoring.explainer")

_lock = threading.Lock()
_cached_explainer: Optional[Any] = None   # shap.TreeExplainer


# ── Human-readable feature descriptions ───────────────────────────────────────
# Maps model feature names → plain-English label for investigator reports.

_FEATURE_LABELS: Dict[str, str] = {
    # BTI synthetic model features
    "transaction_amount":                      "Transaction amount",
    "account_balance_before":                  "Account balance before transaction",
    "account_balance_after":                   "Account balance after transaction",
    "failed_attempt_count":                    "Failed authentication attempts",
    "login_attempts":                          "Login attempts before transaction",
    "risk_score":                              "Baseline risk score",
    "historical_average_transaction_amount":   "Customer's historical average amount",
    "monthly_customer_transaction_count":      "Monthly transaction count",
    "fee_income":                              "Fee income",
    "interchange_income":                      "Interchange income",
    "processing_cost":                         "Processing cost",
    "chargeback_loss":                         "Chargeback loss",
    "refund_loss":                             "Refund loss",
    "fraud_loss":                              "Fraud loss",
    "net_revenue":                             "Net revenue",
    "net_pnl_impact":                          "Net P&L impact",
    "amount_vs_hist_avg_ratio":                "Amount vs. customer historical average (ratio)",
    "fraud_rule_score":                        "Rule-based fraud score",
    "rules_triggered":                         "Number of fraud rules triggered",
    "is_off_hours":                            "Off-hours transaction (midnight–6am)",
    "reversal_flag":                           "Reversal flag",
    "refund_flag":                             "Refund flag",
    "chargeback_flag":                         "Chargeback flag",
    "channel_enc":                             "Transaction channel",
    "merchant_category_enc":                   "Merchant category",
    "customer_segment_enc":                    "Customer segment",
    "transaction_type_enc":                    "Transaction type",
    "authorization_method_enc":                "Authorization method",
    # IEEE-CIS retrained model features
    "amount":                                  "Transaction amount",
    "log_amount":                              "Transaction amount (log-scaled)",
    "amount_vs_card_median":                   "Amount vs. card's median spend",
    "amount_zscore":                           "Amount z-score (statistical outlier)",
    "hour_of_day":                             "Hour of day",
    "is_off_hours":                            "Off-hours transaction",
    "day_of_week":                             "Day of week",
    "productcd_fraud_rate":                    "Fraud rate for this product category",
    "is_card_not_present":                     "Card-not-present transaction",
    "is_credit":                               "Credit card (vs. debit)",
    "email_domain_known":                      "Recognised email domain",
    "is_anonymous_email":                      "Anonymous email address used",
    "C1": "Count of payment accounts linked to card",
    "C2": "Count of payment accounts (C2)",
    "C5": "Count of transactions on card (C5)",
    "C6": "Count of credit cards linked (C6)",
    "C13": "Count of declined transactions (C13)",
    "C14": "Count of merchant logins (C14)",
    "D1": "Days since previous transaction (D1)",
    "D2": "Days since previous transaction — card (D2)",
    "D3": "Days between this and prior transaction type (D3)",
}

for i in range(1, 15):
    _FEATURE_LABELS.setdefault(f"C{i}", f"Count feature C{i}")
for i in range(1, 16):
    _FEATURE_LABELS.setdefault(f"D{i}", f"Timedelta feature D{i} (days)")
for i in range(1, 10):
    _FEATURE_LABELS.setdefault(f"M{i}", f"Match flag M{i}")
for i in range(1, 21):
    _FEATURE_LABELS.setdefault(f"V{i}", f"Vesta engineered feature V{i}")


@dataclass
class FeatureContribution:
    feature:      str
    label:        str
    value:        float
    shap_value:   float
    direction:    str   # "increases_risk" | "decreases_risk"
    impact_pct:   float


@dataclass
class ExplanationResult:
    transaction_id:    str
    fraud_probability: float
    risk_tier:         str
    top_drivers:       List[FeatureContribution]
    narrative:         str
    base_probability:  float
    total_shap_sum:    float


def _get_explainer(rf_model: Any) -> Any:
    """Build and cache the SHAP TreeExplainer."""
    global _cached_explainer
    if _cached_explainer is not None:
        return _cached_explainer
    with _lock:
        if _cached_explainer is not None:
            return _cached_explainer
        import shap
        log.info("Building SHAP TreeExplainer (first call — cached after this)")
        _cached_explainer = shap.TreeExplainer(rf_model)
        return _cached_explainer


def explain(
    rf_model:       Any,
    feature_vector: np.ndarray,         # shape (1, n_features) or (n_features,)
    feature_cols:   List[str],
    transaction_id: str = "unknown",
    fraud_prob:     float = 0.0,
    risk_tier:      str = "UNKNOWN",
    top_n:          int = 8,
) -> ExplanationResult:
    """
    Compute SHAP values for one transaction and return a structured explanation.

    Parameters
    ----------
    rf_model       : fitted RandomForestClassifier
    feature_vector : 1-D or 2-D numpy array of feature values for one transaction
    feature_cols   : list of feature names matching the vector
    transaction_id : for labelling the result
    fraud_prob     : the RF's predicted fraud probability (0–1)
    risk_tier      : e.g. "HIGH", "CRITICAL"
    top_n          : how many top drivers to return
    """
    import shap

    fv = np.array(feature_vector).reshape(1, -1)

    explainer  = _get_explainer(rf_model)
    shap_vals  = explainer.shap_values(fv)

    # shap_values returns [class_0_vals, class_1_vals]; we want class 1 (fraud)
    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1]).reshape(-1)   # fraud class, flatten to 1-D
    else:
        sv = np.array(shap_vals).reshape(-1)

    base_prob = float(explainer.expected_value[1]) if isinstance(
        explainer.expected_value, (list, np.ndarray)
    ) else float(explainer.expected_value)

    # Build per-feature contributions
    contributions: List[FeatureContribution] = []
    for i, col in enumerate(feature_cols):
        sv_i = float(sv[i])
        contributions.append(FeatureContribution(
            feature    = col,
            label      = _FEATURE_LABELS.get(col, col),
            value      = float(fv[0, i]),
            shap_value = sv_i,
            direction  = "increases_risk" if sv_i > 0 else "decreases_risk",
            impact_pct = 0.0,   # filled below
        ))

    total_abs = sum(abs(c.shap_value) for c in contributions) + 1e-9
    for c in contributions:
        c.impact_pct = round(abs(c.shap_value) / total_abs * 100, 1)

    top_drivers = sorted(contributions, key=lambda c: abs(c.shap_value), reverse=True)[:top_n]

    narrative = _build_narrative(top_drivers, fraud_prob, risk_tier, base_prob)

    return ExplanationResult(
        transaction_id   = transaction_id,
        fraud_probability= fraud_prob,
        risk_tier        = risk_tier,
        top_drivers      = top_drivers,
        narrative        = narrative,
        base_probability = base_prob,
        total_shap_sum   = float(sv.sum()),
    )


def _build_narrative(
    drivers:   List[FeatureContribution],
    prob:      float,
    tier:      str,
    base_prob: float,
) -> str:
    """Generate a plain-English explanation for a fraud investigator."""

    risk_drivers   = [d for d in drivers if d.direction == "increases_risk"]
    safety_drivers = [d for d in drivers if d.direction == "decreases_risk"]

    lines = []
    lines.append(
        f"This transaction is assessed as {tier} risk "
        f"(fraud probability: {prob*100:.1f}%, "
        f"baseline: {base_prob*100:.1f}%)."
    )

    if risk_drivers:
        top = risk_drivers[0]
        lines.append(
            f"The strongest fraud signal is '{top.label}' "
            f"(value: {_fmt(top.value)}, contributing {top.impact_pct:.1f}% of total risk)."
        )
        if len(risk_drivers) > 1:
            others = ", ".join(
                f"'{d.label}' ({d.impact_pct:.1f}%)" for d in risk_drivers[1:4]
            )
            lines.append(f"Additional risk factors: {others}.")

    if safety_drivers:
        top_s = safety_drivers[0]
        lines.append(
            f"Mitigating factor: '{top_s.label}' "
            f"(reduces fraud likelihood by {top_s.impact_pct:.1f}%)."
        )

    lines.append("Recommended action: " + _recommend(prob))
    return " ".join(lines)


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _recommend(prob: float) -> str:
    if prob >= 0.85:
        return "Block transaction and escalate to senior fraud analyst immediately."
    if prob >= 0.63:
        return "Hold transaction and route to fraud analyst for manual review."
    if prob >= 0.30:
        return "Flag for enhanced monitoring; allow with step-up authentication."
    return "Allow — low fraud probability."


def invalidate_explainer_cache() -> None:
    global _cached_explainer
    with _lock:
        _cached_explainer = None
    log.info("SHAP explainer cache invalidated")
