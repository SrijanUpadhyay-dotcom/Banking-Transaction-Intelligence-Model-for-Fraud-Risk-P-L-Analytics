# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
UK APP-scam reimbursement exposure, and the choice of scam intervention.

**The rules modelled.** These are the Payment Systems Regulator's mandatory
reimbursement requirements, in force since 7 October 2024. The bank's
compliance team must confirm the rules currently in force: the PSR's functions
are moving to the FCA.
- *Scope.* Consumers, micro-enterprises and charities, paying by Faster
  Payments or CHAPS between UK accounts. International wires are out of scope.
- *Limit.* Up to £85,000 per claim.
- *Split.* The sending and receiving payment providers share reimbursement
  50:50. When the payee account is at this bank, the bank is both, so it bears
  100%.
- *Excess.* The sending provider may apply an excess of up to £100, never to a
  vulnerable customer.
- *Not modelled.* The consumer-standard-of-caution exception (gross negligence)
  is decided case by case and does not apply to vulnerable customers. It is
  not netted off, so exposure is the conservative figure. Effective warnings
  matter to that assessment, which is for legal review.

**Exposure for one payment.**
`exposure = share × max(0, min(amount £, cap) − excess)`. Across a portfolio,
payments are grouped into claims (customer × payee within the claim window), so
the cap applies per claim, not per payment.

**Intervention.** For a payment with scam probability p, the cheapest of
three options:

| Option | Expected cost |
|---|---|
| none | p × (exposure + w × customer's unreimbursed loss) |
| warning | p × (…) × (1 − warning effectiveness) + warning friction × (1 − p) |
| hold and call | p × (…) × (1 − call effectiveness) + call cost + call friction × (1 − p) |

w is `customer_harm_weight`: the bank values its customer's own unreimbursed
loss as well as its own. Effectiveness figures are illustrative until measured.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from bti.config import get_settings
from bti.modeling.fx import RATES_TO_USD


def _to_gbp(amount, currency) -> np.ndarray:
    usd = pd.to_numeric(amount, errors="coerce") * pd.Series(currency).astype(str).str.upper().map(RATES_TO_USD).to_numpy()
    return np.asarray(usd, dtype=float) / RATES_TO_USD["GBP"]


def in_scope(df: pd.DataFrame) -> np.ndarray:
    r = get_settings().scam_reimbursement
    from bti.jurisdiction.policies import policy_for
    iso = df["country"].map(lambda c: getattr(policy_for(c), "iso2", None)).to_numpy() if "country" in df else None
    uk = (iso == r["jurisdiction"]) if iso is not None else np.zeros(len(df), bool)
    return uk & df["transaction_type"].isin(r["in_scope_types"]).to_numpy() & \
        (df["debit_credit_flag"].astype(str) == "Debit").to_numpy()


def exposure_gbp(df: pd.DataFrame) -> np.ndarray:
    """The bank's reimbursement if each payment were a scam (per payment; the cap is applied per payment)."""
    r = get_settings().scam_reimbursement
    gbp = np.nan_to_num(_to_gbp(df["transaction_amount"], df["currency"]))
    vulnerable = pd.to_numeric(df.get("is_vulnerable", 0), errors="coerce").fillna(0).astype(bool).to_numpy() \
        if "is_vulnerable" in df else np.zeros(len(df), bool)
    excess = np.where(vulnerable, 0.0, float(r["excess_gbp"]))
    reimbursable = np.maximum(0.0, np.minimum(gbp, r["max_claim_gbp"]) - excess)
    on_us = df["payee_customer_id"].notna().to_numpy() if "payee_customer_id" in df else np.zeros(len(df), bool)
    share = np.where(on_us, 1.0, float(r["sending_share"]))
    return np.where(in_scope(df), reimbursable * share, 0.0)


def claim_exposure(df: pd.DataFrame, probability: Optional[np.ndarray] = None) -> Dict:
    """Portfolio exposure with the cap applied per claim (customer × payee); expected if probabilities given."""
    r = get_settings().scam_reimbursement
    d = df.assign(_gbp=np.nan_to_num(_to_gbp(df["transaction_amount"], df["currency"])), _scope=in_scope(df),
                  _p=1.0 if probability is None else probability,
                  _onus=df["payee_customer_id"].notna() if "payee_customer_id" in df else False,
                  _vuln=pd.to_numeric(df.get("is_vulnerable", 0), errors="coerce").fillna(0).astype(bool)
                  if "is_vulnerable" in df else False)
    d = d[d["_scope"]]
    if d.empty:
        return {"claims": 0, "exposure_gbp": 0.0}
    g = d.groupby(["customer_id", "payee_id"], dropna=False).agg(gbp=("_gbp", "sum"), p=("_p", "max"),
                                                                 onus=("_onus", "max"), vuln=("_vuln", "max"))
    excess = np.where(g["vuln"], 0.0, float(r["excess_gbp"]))
    reimb = np.maximum(0.0, np.minimum(g["gbp"], r["max_claim_gbp"]) - excess)
    bank = reimb * np.where(g["onus"], 1.0, r["sending_share"])
    return {"claims": int(len(g)), "claims_at_cap": int((g["gbp"] >= r["max_claim_gbp"]).sum()),
            "reimbursable_gbp": round(float((reimb * g["p"]).sum()), 0),
            "bank_exposure_gbp": round(float((bank * g["p"]).sum()), 0),
            "on_us_claims": int(g["onus"].sum())}


def choose_intervention(p: float, exposure: float, customer_loss: float) -> Dict:
    s = get_settings()
    w = s.scam_customer_harm_weight
    harm = p * (exposure + w * customer_loss)
    options = {"none": harm}
    for name, c in s.scam_interventions.items():
        options[name] = harm * (1 - c["effectiveness"]) + c["cost_gbp"] + c["friction_gbp"] * (1 - p)
    action = min(options, key=options.get)
    return {"action": action, "expected_cost_gbp": {k: round(v, 2) for k, v in options.items()}}


def portfolio_report(data_path=None, window_start: str = None) -> Dict:
    """
    Out-of-time replay: the scam model's interventions on every outbound payment, against doing nothing.
    Reports the UK reimbursement exposure on confirmed scams (claim-capped), the expected exposure and customer
    loss averted, and the cost: calls, warnings and genuine customers interrupted.
    """
    import json
    from pathlib import Path
    from bti.modeling import registry
    from bti.scams.app_model import DATA, LABEL
    from bti.scams.features import SCAM_FEATURE_NAMES, payment_rows, scam_features
    s = get_settings()
    model_id = registry.model_for_role("champion", family="scam") or registry.model_for_role("challenger", family="scam")
    art = registry.load_artifact(model_id, family="scam")
    df = pd.read_csv(data_path or DATA, low_memory=False)
    rows = payment_rows(df)
    feats = scam_features(df)[rows]
    d = df[rows].reset_index(drop=True)
    p = art["calibrator"].predict(art["estimator"].predict_proba(feats[art["feature_names"]].astype(float))[:, 1])
    dates = pd.to_datetime(d["transaction_date"])
    start = pd.Timestamp(window_start) if window_start else dates.quantile(0.8)
    w = (dates >= start).to_numpy()
    d, p = d[w].reset_index(drop=True), p[w]
    y = (d["fraud_type"] == LABEL).to_numpy()
    exposure = exposure_gbp(d)
    gbp = np.nan_to_num(_to_gbp(d["transaction_amount"], d["currency"]))
    scope = in_scope(d)
    vuln = pd.to_numeric(d.get("is_vulnerable", 0), errors="coerce").fillna(0).astype(bool).to_numpy()
    reimbursed = np.where(scope, np.maximum(0, np.minimum(gbp, s.scam_reimbursement["max_claim_gbp"])
                                             - np.where(vuln, 0, s.scam_reimbursement["excess_gbp"])), 0)
    cust_loss = np.maximum(0, gbp - reimbursed)
    actions = np.array([choose_intervention(float(pi), float(e), float(c))["action"]
                        for pi, e, c in zip(p, exposure, cust_loss)])
    eff = np.select([actions == k for k in s.scam_interventions],
                    [v["effectiveness"] for v in s.scam_interventions.values()], 0.0)
    days = max(int(dates[w].dt.normalize().nunique()), 1)
    out = {
        "model_id": model_id, "window": [str(start.date()), str(dates.max().date())], "days": days,
        "payments": int(len(d)), "confirmed_scams": int(y.sum()),
        "uk_in_scope_scams": int((y & scope).sum()),
        "claims_exposure_if_no_intervention": claim_exposure(d[y]),
        "interventions": {a: int((actions == a).sum()) for a in ["none", *s.scam_interventions]},
        "calls_per_day": round(float((actions == "hold_and_call").sum() / days), 2),
        "genuine_payments_interrupted": {a: int(((actions == a) & ~y).sum()) for a in s.scam_interventions},
        "scams_intervened": {a: int(((actions == a) & y).sum()) for a in s.scam_interventions},
        "expected_bank_exposure_averted_gbp": round(float((exposure * eff)[y].sum()), 0),
        "expected_customer_loss_averted_gbp": round(float((cust_loss * eff)[y].sum()), 0),
        "intervention_cost_gbp": round(float(sum(
            ((actions == a) * (c["cost_gbp"] + np.where(y, 0, c["friction_gbp"]))).sum()
            for a, c in s.scam_interventions.items())), 0),
        "assumptions": {"interventions": s.scam_interventions, "reimbursement": s.scam_reimbursement,
                        "customer_harm_weight": s.scam_customer_harm_weight},
        "note": "Expected values: an intervention stops a scam with its configured effectiveness (illustrative until "
                "measured). Synthetic data; the consumer-caution exception is not netted off.",
    }
    path = Path("outputs/scams/reimbursement_exposure.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    import json
    print(json.dumps(portfolio_report(), indent=1, default=str))


if __name__ == "__main__":
    main()
