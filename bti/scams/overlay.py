# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Live APP-scam overlay for outbound payments.

For each outbound payment to a payee, alongside the transaction model:

1. The payee-risk features are built from the customer's and payee's history
   (`fetch_history` already loads rows for the same payee) plus the payment's
   own Confirmation-of-Payee result and payee account age.
2. The scam model (scam family: champion, else challenger) gives a calibrated
   probability, with SHAP reason codes.
3. The bank's reimbursement exposure and the customer's unreimbursed loss are
   computed (`bti.scams.reimbursement`). The cheapest intervention is chosen:
   none, a tailored scam warning, or hold and call.
4. If the payee is an on-us account under an open or confirmed mule alert, the
   overlay escalates to hold-and-call whatever the model says. Money must not
   reach a known mule.

**Modes** (`scams.mode`):
- `shadow` (default): the result is logged on the score log and returned, but
  the decision is unchanged. This is how a model earns its validation
  evidence.
- `active`: hold-and-call raises the decision to REVIEW, which opens a case
  carrying the scam reasons. A warning sets `scam.warning` for the channel to
  show.

A provisional (challenger-only) scam model can never act, whatever the mode.
The overlay never lowers a decision.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from bti.config import get_settings
from bti.logging_config import get_logger

log = get_logger("scams.overlay")


def _model():
    from bti.modeling import registry
    model_id = registry.model_for_role("champion", family="scam")
    provisional = model_id is None
    model_id = model_id or registry.model_for_role("challenger", family="scam")
    if not model_id:
        return None, None, True
    return model_id, registry.load_artifact(model_id, family="scam"), provisional


def is_payment(txn: Dict) -> bool:
    from bti.scams.features import PAYMENT_TYPES
    return bool(txn.get("payee_id")) and txn.get("transaction_type") in PAYMENT_TYPES and \
        str(txn.get("debit_credit_flag") or "Debit") == "Debit"


def payee_mule_flag(db, payee_customer_id: Optional[str]) -> Optional[Dict]:
    if db is None or not payee_customer_id:
        return None
    from bti.database.models import MuleAlert
    row = (db.query(MuleAlert).filter(MuleAlert.customer_id == str(payee_customer_id),
                                      MuleAlert.status.in_(("open", "confirmed_mule")))
           .order_by(MuleAlert.id.desc()).first())
    return {"alert_id": row.id, "status": row.status, "score": row.score} if row else None


def assess(txn: Dict, history: pd.DataFrame, db=None, explain: bool = True) -> Optional[Dict]:
    """Scam probability, exposure and intervention for one outbound payment; None if not a payment or no model."""
    if not is_payment(txn):
        return None
    model_id, art, provisional = _model()
    if art is None:
        return None
    from bti.scams.features import SCAM_FEATURE_BY_NAME, SCAM_FEATURE_NAMES, scam_features
    from bti.scams.reimbursement import _to_gbp, choose_intervention, exposure_gbp, in_scope
    row = {**txn, "debit_credit_flag": txn.get("debit_credit_flag") or "Debit"}
    frame = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    feats = scam_features(frame, graph=False).iloc[[-1]]
    graph_note = None
    graph_cols = [c for c in ("graph_payee_senders", "graph_payee_known_fraud_share") if c in art["feature_names"]]
    if graph_cols:                                   # same day-lagged graph state as training, from the snapshot
        from bti.graph.snapshot import live_features
        snap = live_features("scam", txn)
        if snap is None:
            graph_note = "No graph snapshot: payee graph features are unknown (run the nightly graph snapshot)."
        for col in graph_cols:
            if snap and snap.get(col) is not None:
                feats[col] = float(snap[col])
    X = feats[art["feature_names"]].astype(float)
    p = float(art["calibrator"].predict(art["estimator"].predict_proba(X)[:, 1])[0])
    last = frame.iloc[[-1]]
    exposure = float(exposure_gbp(last)[0])
    amount_gbp = float(np.nan_to_num(_to_gbp(last["transaction_amount"], last["currency"]))[0])
    r = get_settings().scam_reimbursement
    reimbursed_total = (max(0.0, min(amount_gbp, r["max_claim_gbp"])
                            - (0 if row.get("is_vulnerable") else r["excess_gbp"])) if in_scope(last)[0] else 0.0)
    customer_loss = max(0.0, amount_gbp - reimbursed_total)
    choice = choose_intervention(p, exposure, customer_loss)
    mule = payee_mule_flag(db, row.get("payee_customer_id"))
    action = choice["action"]
    if mule:
        action = "hold_and_call"
    reasons = []
    if explain:
        from bti.governance.reason_codes import principal_reasons
        contrib = art["estimator"].booster_.predict(X, pred_contrib=True)[0][:-1]
        values = {c: (None if pd.isna(v) else round(float(v), 4)) for c, v in X.iloc[0].items()}
        reasons = principal_reasons(dict(zip(art["feature_names"], contrib)), values, specs=SCAM_FEATURE_BY_NAME)
    mode = get_settings().scam_mode
    return {"model_id": model_id, "provisional": provisional, "mode": mode,
            "acting": mode == "active" and not provisional,
            "probability": round(p, 6), "exposure_gbp": round(exposure, 2),
            "customer_unreimbursed_loss_gbp": round(customer_loss, 2), "in_reimbursement_scope": bool(in_scope(last)[0]),
            "action": action, "expected_cost_gbp": choice["expected_cost_gbp"], "payee_mule_alert": mule,
            "warning": action == "warning", "reason_codes": reasons, "note": graph_note}


def apply(decision, scam: Optional[Dict]) -> None:
    """In active mode with an approved scam model, raise the decision for hold-and-call. Never lowers."""
    if not scam or not scam["acting"]:
        return
    if scam["action"] == "hold_and_call" and decision.action in ("APPROVE", "STEP_UP"):
        decision.action = "REVIEW"
        decision.guardrails_applied.append("APP-scam risk: payment held for a call with the customer")
    elif scam["action"] == "warning":
        decision.guardrails_applied.append("APP-scam risk: show the tailored scam warning before release")
