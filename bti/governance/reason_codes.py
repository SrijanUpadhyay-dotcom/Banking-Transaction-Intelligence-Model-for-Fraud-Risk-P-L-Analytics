# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Reason codes derived from exact SHAP contributions.

Two audiences:
  * analyst text — specific, for investigators, QA and model validation
  * customer text — plain and deliberately non-specific, so a notice never
    teaches a fraudster which control fired and never hints at a suspicious-
    activity report (tipping-off offences exist in the UK, US, EU, India,
    Singapore, Hong Kong, UAE and Nigeria)

Up to four principal reasons are returned, the convention used for US adverse
action notices and a reasonable default for GDPR Art. 22 explanations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from bti.modeling.features import FEATURE_BY_NAME

MAX_REASONS = 4

REASON_CODES: Dict[str, Dict[str, str]] = {
    "AMT_HIGH": {
        "analyst": "Large transaction amount (USD equivalent)",
        "customer": "The size of this transaction",
    },
    "AMT_VS_HISTORY": {
        "analyst": "Amount far above this customer's historical average",
        "customer": "This transaction is different from your usual activity",
    },
    "AMT_VS_BALANCE": {
        "analyst": "Amount is a large share of the available balance",
        "customer": "The amount relative to your available balance",
    },
    "AUTH_FAILURES": {
        "analyst": "Failed authentication attempts in this session",
        "customer": "Unsuccessful verification attempts",
    },
    "LOGIN_ANOMALY": {
        "analyst": "Unusually many login attempts before the transaction",
        "customer": "Unusual sign-in activity",
    },
    "TIME_OF_DAY": {
        "analyst": "Transaction at an unusual time of day or week",
        "customer": "The timing of this transaction",
    },
    "VELOCITY": {
        "analyst": "Transaction frequency or spend velocity out of pattern",
        "customer": "Recent account activity",
    },
    "NEW_DEVICE": {
        "analyst": "Device not previously used by this customer",
        "customer": "The device used for this transaction",
    },
    "NEW_IP": {
        "analyst": "Network location not previously used by this customer",
        "customer": "The location or network used for this transaction",
    },
    "NEW_MERCHANT": {
        "analyst": "First payment to this merchant or payee",
        "customer": "This is a new payee or merchant for you",
    },
    "SHARED_DEVICE": {
        "analyst": "Device or IP recently used by other customers (possible mule / ring activity)",
        "customer": "The device or network used for this transaction",
    },
    "CHANNEL_RISK": {
        "analyst": "Channel or authentication method with elevated fraud rates",
        "customer": "How this transaction was initiated",
    },
    "MERCHANT_RISK": {
        "analyst": "Merchant category with elevated fraud rates",
        "customer": "The type of merchant",
    },
    "MERCHANT_VELOCITY": {
        "analyst": "Burst of activity at this merchant across customers (possible card testing or compromise)",
        "customer": "Recent activity at this merchant",
    },
    "DEVICE_VELOCITY": {
        "analyst": "High transaction rate on this device",
        "customer": "The device used for this transaction",
    },
    "STRUCTURING": {
        "analyst": "Amount just below a round threshold, repeated recently (possible limit testing)",
        "customer": "The pattern of recent transaction amounts",
    },
    "TXN_TYPE": {
        "analyst": "Transaction type with elevated fraud rates",
        "customer": "The type of transaction",
    },
    "GEO_VELOCITY": {
        "analyst": "Location implausibly far from the customer's previous transaction (impossible travel)",
        "customer": "The location of this transaction",
    },
    "NEW_PAYEE": {
        "analyst": "Payment to a payee new for this customer, or several new payees in a short time",
        "customer": "This is a new payee for you",
    },
    "PAYEE_VELOCITY": {
        "analyst": "Payee recently paid by many other customers (possible mule account)",
        "customer": "The account you are paying",
    },
    "NETWORK_LINK": {
        "analyst": "Linked through shared devices or IPs to other customers, some with confirmed fraud (possible ring)",
        "customer": "The device or network used for this transaction",
    },
    "MULE_PAYEE": {
        "analyst": "Payee has received payments later confirmed as fraud from several customers (possible mule account)",
        "customer": "The account you are paying",
    },
    "PAYEE_RISK": {
        "analyst": "Payee account recently opened, new to the bank, or held here (possible mule or scam account)",
        "customer": "The account you are paying",
    },
    "COP_MISMATCH": {
        "analyst": "Confirmation of Payee did not match, closely matched, or could not check the payee's name",
        "customer": "The name on the account you are paying",
    },
    "SCAM_PATTERN": {
        "analyst": "Repeated or escalating payments to the same new payee (investment / romance scam pattern)",
        "customer": "The pattern of your recent payments",
    },
    "MULE_FLOW": {
        "analyst": "Many inbound payments from new senders moved straight out (money-mule flow pattern)",
        "customer": "Recent activity on your account",
    },
    "SECURITY_EVENT": {
        "analyst": "Recent password reset, SIM swap, contact-detail change or new device enrolment",
        "customer": "Recent changes to your account security settings",
    },
}


def principal_reasons(contributions: Dict[str, float], feature_values: Dict[str, object],
                      max_reasons: int = MAX_REASONS, specs: Optional[Dict] = None) -> List[dict]:
    """
    Aggregate per-feature SHAP contributions (log-odds) into reason codes and
    return the codes that pushed risk up the most.
    """
    by_code: Dict[str, dict] = {}
    for feature, value in contributions.items():
        spec = (specs or FEATURE_BY_NAME).get(feature)
        if spec is None:
            continue
        entry = by_code.setdefault(spec.reason_code, {"contribution": 0.0, "features": []})
        entry["contribution"] += float(value)
        entry["features"].append({"feature": feature, "value": feature_values.get(feature),
                                  "contribution": round(float(value), 4)})
    risk_raising = {k: v for k, v in by_code.items() if v["contribution"] > 0}
    total = sum(v["contribution"] for v in risk_raising.values()) or 1.0
    ranked = sorted(risk_raising.items(), key=lambda kv: -kv[1]["contribution"])[:max_reasons]
    return [
        {
            "code": code,
            "rank": i,
            "analyst_text": REASON_CODES[code]["analyst"],
            "customer_text": REASON_CODES[code]["customer"],
            "contribution_log_odds": round(v["contribution"], 4),
            "share_of_risk": round(v["contribution"] / total, 4),
            "features": sorted(v["features"], key=lambda f: -f["contribution"]),
        }
        for i, (code, v) in enumerate(ranked, start=1)
    ]
