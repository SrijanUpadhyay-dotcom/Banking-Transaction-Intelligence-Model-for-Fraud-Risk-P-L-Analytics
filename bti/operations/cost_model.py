# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Cost model v2: decision economics from the bank's own data.

v1 prices every customer and every challenge alike. v2 makes three changes.

1. **Customer value.** The attrition cost of wrongly declining a customer uses
   that customer's annual net revenue from the P&L data (fees + interchange −
   processing cost, annualised). The value is bounded to [$75, $1,200] by
   default.
   - Unbounded value would make a Student ($1.8 a year here) about 700 times
     cheaper to decline than a Corporate client, which is differential
     treatment by wealth.
   - Even bounded, v2 leans that way. `backtest` measures the decision-level
     fairness effect before anyone switches it on.
2. **Measured step-up outcomes.** Friction and catch rate come from real
   challenge outcomes per method (SMS, push, 3-D Secure), once 200 challenges
   of that method have completed. Until then, stated priors are used.
   - Friction per transaction = message cost + abandonment rate × (lost margin
     + attrition on an abandoned payment).
3. **No segment pricing.** Segment is a protected proxy in the feature catalog.
   Pricing friction by segment would give customers different treatment for
   belonging to a segment, so v2 uses measured channel and method outcomes
   instead of segment. This is a deliberate departure from the original
   docket wording.

Capacity shadow prices (analyst capacity, challenge budget) stay as floors on
review cost and step-up friction, so v2 cannot push volumes past the budgets.
`decisioning.cost_model: v1 | v2` selects the model. The default is v1 until
the bank approves v2 on the backtest evidence.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import AuditLog, CustomerValue
from bti.logging_config import get_logger
from bti.modeling.fx import series_to_usd

log = get_logger("operations.cost_model")

_measured = {"at": 0.0, "rates": {}}
_lock = threading.Lock()


# ── Customer value ───────────────────────────────────────────────────────────

def customer_values_from_pnl(df: pd.DataFrame, as_of: Optional[pd.Timestamp] = None,
                             window_days: int = 365) -> pd.DataFrame:
    """Annualised net revenue per customer from transactions before `as_of` (point-in-time)."""
    ts = pd.to_datetime(df["transaction_date"], errors="coerce")
    as_of = as_of if as_of is not None else ts.max() + pd.Timedelta(days=1)
    m = (ts < as_of) & (ts >= as_of - pd.Timedelta(days=window_days))
    d = df.loc[m, ["customer_id", "net_revenue", "currency"]].copy()
    d["_ts"] = ts[m]
    d["net_usd"] = series_to_usd(pd.to_numeric(d["net_revenue"], errors="coerce").fillna(0), d["currency"])
    g = d.groupby("customer_id").agg(net=("net_usd", "sum"), first=("_ts", "min"), n=("net_usd", "size"))
    months = ((as_of - g["first"]).dt.days / 30.44).clip(lower=3, upper=12)
    return pd.DataFrame({"customer_id": g.index, "annual_value_usd": (g["net"] * 12 / months).round(2).to_numpy(),
                         "months_observed": months.round(2).to_numpy(), "transactions": g["n"].to_numpy()})


def refresh_customer_values(db: Session, source_path=None) -> Dict:
    from bti.modeling.train import default_data_path
    df = pd.read_csv(source_path or default_data_path(), low_memory=False,
                     usecols=["customer_id", "transaction_date", "net_revenue", "currency"])
    values = customer_values_from_pnl(df)
    now = datetime.utcnow()
    db.query(CustomerValue).delete()
    db.bulk_insert_mappings(CustomerValue, [{**r, "computed_at": now} for r in values.to_dict("records")])
    db.add(AuditLog(ts=now, event_type="CUSTOMER_VALUES_REFRESHED",
                    payload={"customers": int(len(values)), "median_usd": float(values["annual_value_usd"].median())}))
    db.commit()
    return {"customers": int(len(values)), "median_annual_value_usd": round(float(values["annual_value_usd"].median()), 2)}


def bounded_value(value: Optional[float]) -> float:
    lo, hi = get_settings().cost_v2_value_bounds
    default = 300.0
    if value is None or np.isnan(value):
        return default
    return float(min(max(value, lo), hi))


# ── Measured step-up outcomes ────────────────────────────────────────────────

def measured_stepup(db: Session, days: int = 90) -> Dict[str, Dict]:
    """Abandonment and catch rates per method when enough challenges completed; otherwise the priors."""
    from bti.operations.stepup import stepup_stats
    with _lock:
        if time.time() - _measured["at"] < 300 and _measured["rates"]:
            return _measured["rates"]
    s = get_settings()
    stats = stepup_stats(db, days)["methods"] if db is not None else {}
    rates = {}
    for method in ("sms_otp", "push", "3ds"):
        m = stats.get(method, {})
        enough = (m.get("completed") or 0) >= s.cost_v2_min_challenges
        rates[method] = {
            "abandonment_rate": m["abandonment_rate"] if enough and m.get("abandonment_rate") is not None
            else s.cost_v2_abandonment_prior,
            "catch_rate": m["catch_rate"] if enough and m.get("catch_rate") is not None else None,
            "source": "measured" if enough else "prior",
            "completed": m.get("completed", 0),
        }
    with _lock:
        _measured.update(at=time.time(), rates=rates)
    return rates


def clear_cache() -> None:
    with _lock:
        _measured.update(at=0.0, rates={})


# ── Per-transaction overrides ────────────────────────────────────────────────

def v2_overrides(amount_usd: float, customer_value: Optional[float], method: Optional[str],
                 rates: Dict[str, Dict], capacity: Optional[Dict] = None) -> Dict[str, float]:
    s = get_settings()
    value = bounded_value(customer_value)
    out: Dict[str, float] = {"customer_annual_value_usd": value}
    if method:
        r = rates.get(method, {"abandonment_rate": s.cost_v2_abandonment_prior, "catch_rate": None})
        lost = amount_usd * s.cost_v2_margin_rate + 0.02 * value          # margin + attrition on an abandoned payment
        out["step_up_friction_usd"] = round(s.cost_v2_message_cost_usd.get(method, 0.05)
                                            + r["abandonment_rate"] * lost, 4)
        if r.get("catch_rate") is not None:
            out["step_up_catch_rate"] = r["catch_rate"]
    if capacity:
        out["review_cost_usd"] = capacity.get("review_cost_usd", out.get("review_cost_usd"))
        if "step_up_friction_usd" in capacity:
            out["step_up_friction_usd"] = max(out.get("step_up_friction_usd", 0.0), capacity["step_up_friction_usd"])
    return {k: v for k, v in out.items() if v is not None}


def overrides_for(db: Optional[Session], txn: Dict, amount_usd: float, capacity: Optional[Dict]) -> Optional[Dict]:
    """The cost overrides live scoring should use under the configured cost model."""
    if get_settings().cost_model_version != "v2":
        return capacity
    from bti.operations.stepup import choose_method
    value = None
    if db is not None and txn.get("customer_id"):
        row = db.get(CustomerValue, str(txn["customer_id"]))
        value = row.annual_value_usd if row else None
    rates = measured_stepup(db) if db is not None else {}
    return v2_overrides(amount_usd, value, choose_method(txn.get("channel"), txn.get("transaction_type")), rates,
                        capacity)


# ── Backtest v1 against v2 ───────────────────────────────────────────────────

def backtest(model_id: Optional[str] = None) -> Dict:
    """
    Replay the out-of-time window under v1 and v2. Customer value is computed only from P&L before the window
    (point-in-time). Reports realised cost, action mix and the decision-level fairness of each version.
    """
    from bti.governance.fairness import decision_fairness, fairness_report
    from bti.jurisdiction.policies import policy_for
    from bti.modeling import registry
    from bti.modeling.reassess import model_probabilities
    from bti.modeling.train import prepare
    from bti.operations.capacity import capacity_overrides
    from bti.operations.decisioning import cost_model_for, decide
    from bti.operations.stepup import choose_method

    model_id = model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
    data = prepare(feature_version=registry.load_artifact(model_id).get("feature_version", 1))
    p = model_probabilities(model_id, data)
    te = data.te
    df = data.df.loc[te].reset_index(drop=True)
    y, prob = data.y[te], p[te]
    amt = data.features.loc[te, "amount_usd"].fillna(0).to_numpy(float)
    values = customer_values_from_pnl(data.df.loc[data.tr | data.ca].drop(columns=["_ts"]),
                                      as_of=data.cut_test).set_index("customer_id")["annual_value_usd"]
    capacity = capacity_overrides(model_id)
    provisional = registry.model_for_role("champion") != model_id
    rates = {m: {"abandonment_rate": get_settings().cost_v2_abandonment_prior, "catch_rate": None}
             for m in ("sms_otp", "push", "3ds")}
    results = {}
    for version in ("v1", "v2"):
        actions, costs = [], np.zeros(len(y))
        for i, row in df.iterrows():
            policy = policy_for(row["country"])
            ov = capacity if version == "v1" else v2_overrides(
                amt[i], values.get(row["customer_id"]), choose_method(row["channel"], row["transaction_type"]),
                rates, capacity)
            d = decide(float(prob[i]), float(amt[i]), policy, row["channel"], row["transaction_type"],
                       provisional_model=provisional, cost_overrides=ov)
            actions.append(d.action)
            cm = cost_model_for(policy, ov)
            loss = amt[i] * cm.loss_given_fraud
            a, fraud = d.action, y[i] == 1
            costs[i] = ((loss if fraud else 0) if a == "APPROVE" else
                        (loss * (1 - cm.step_up_catch_rate) if fraud else cm.step_up_friction_usd) if a == "STEP_UP" else
                        cm.review_cost_usd + (loss * (1 - cm.review_catch_rate) if fraud else cm.review_friction_usd)
                        if a == "REVIEW" else
                        (0 if fraud else cm.decline_friction_usd + cm.attrition_rate_on_false_decline
                         * cm.customer_annual_value_usd))
        a = np.array(actions)
        intervened = a != "APPROVE"
        groups = {c: df[c] for c in ("customer_segment", "customer_age_band", "country")}
        fair = fairness_report(y, intervened, groups)
        conditional = decision_fairness(y, intervened, groups, amt)
        by_segment = {seg: round(float(intervened[(df["customer_segment"] == seg).to_numpy() & (y == 0)].mean()), 5)
                      for seg in sorted(df["customer_segment"].unique())}
        results[version] = {
            "realised_cost_usd": round(float(costs.sum()), 2),
            "action_mix": {x: int((a == x).sum()) for x in ("APPROVE", "STEP_UP", "REVIEW", "DECLINE")},
            "fraud_intervened": int((intervened & (y == 1)).sum()),
            "genuine_intervened": int((intervened & (y == 0)).sum()),
            "genuine_intervention_rate_by_segment": by_segment,
            "fairness": {"status": fair["status"],
                         "findings": [{k: f[k] for k in ("attribute", "group", "fpr_ratio", "fpr_q_value")}
                                      for f in fair["findings"]]},
            "decision_fairness_amount_standardised": {
                "status": conditional["status"], "band_rates": conditional["band_rates"],
                "findings": [{k: f[k] for k in ("attribute", "group", "raw_ratio", "amount_standardised_ratio",
                                                "q_value")} for f in conditional["findings"]],
                "groups": {a["attribute"]: {g["group"]: [g["raw_ratio"], g["amount_standardised_ratio"]]
                                            for g in a["groups"]} for a in conditional["attributes"]}},
        }
    return {"model_id": model_id, "window": "out-of-time", "transactions": int(len(y)),
            "customer_value_source": "P&L before the window (point-in-time), bounded "
                                     f"{get_settings().cost_v2_value_bounds}",
            "step_up_rates": "priors (no measured challenges on synthetic data)",
            "v1": results["v1"], "v2": results["v2"],
            "cost_saving_v2_usd": round(results["v1"]["realised_cost_usd"] - results["v2"]["realised_cost_usd"], 2),
            "recommendation": _recommend(results)}


def _recommend(r: Dict) -> str:
    """Compare v2 with v1: v2 must not add amount-standardised disparities and must lower realised cost."""
    def keys(v):
        return {(f["attribute"], f["group"]) for f in r[v]["decision_fairness_amount_standardised"]["findings"]}
    added = keys("v2") - keys("v1")
    if added:
        return f"Keep v1: v2 adds decision disparities beyond what amounts explain: {sorted(added)}."
    if r["v2"]["realised_cost_usd"] >= r["v1"]["realised_cost_usd"]:
        return "Keep v1: v2 does not lower realised cost."
    return ("v2 lowers realised cost without adding decision disparities; it needs approval before switching "
            "(decisioning.cost_model). Existing disparities apply to both versions and are tracked separately.")
