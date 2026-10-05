# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Policy what-if simulator: the loss and friction impact of a policy change,
before it is made.

**Method.** A proposal is replayed against the current live policy on the same
labelled window: the model's out-of-time transactions, with outcomes known.
Both policies decide every transaction with the production decision function
(`_choose_actions`). Per-country loss-given-fraud is applied, and provisional
models downgrade declines to reviews.

**A proposal can change:**
- `cost_overrides`: any cost-model figure (loss-given-fraud, decline friction,
  step-up friction or catch rate, review cost, `min_decline_probability`, ...).
  It applies on top of the live capacity prices. An explicit loss-given-fraud
  replaces every jurisdiction's own figure.
- `max_review_rate` / `max_step_up_rate`: refit the capacity prices on the
  calibration window under the proposal. This is the same fit as
  `bti.operations.capacity`, but nothing is saved.
- `step_up_channels`: the channels that can challenge a customer, for example
  after rolling out in-app push.
- `model_id`: score the window with another registered model (challenger vs
  champion).
- `provisional`: whether declines are downgraded to reviews.

Decisions use the capacity shadow prices; realised cost is accounted at the
economic figures (default cost model plus any `cost_overrides`), because a
shadow price rations capacity and is not money spent.

**Reported, per policy and as differences:**
- fraud loss: prevented and residual
- genuine customers disturbed: declined, challenged, held for review
- realised total cost
- cases per day, and the analyst workload they imply (handling times from
  the staffing module)
- genuine-customer intervention rate by customer segment and age band, as
  ratios to the overall rate, so a change that shifts friction onto one group
  is visible before it ships

Differences carry 95% intervals from a paired bootstrap over days.

**Read-only.** Nothing live changes. Each run is saved under
outputs/planning/whatif/ with an id, as evidence for the change request. The
change itself goes through the governed paths: capacity refit by a named
fitter, cost-model approval, or a model promotion with four-eyes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bti.config import get_settings
from bti.operations import decisioning
from bti.operations.decisioning import ACTIONS, CostModel, _choose_actions, cost_model_for

OUT_DIR = Path("outputs/planning/whatif")
ALLOWED_KEYS = {"cost_overrides", "max_review_rate", "max_step_up_rate", "step_up_channels", "model_id",
                "provisional", "label"}


class WhatIfError(ValueError):
    pass


def _lgf(countries: pd.Series) -> np.ndarray:
    from bti.jurisdiction.policies import DEFAULT_LOSS_GIVEN_FRAUD, policy_for
    cache = {}
    out = np.empty(len(countries))
    for i, c in enumerate(countries):
        if c not in cache:
            pol = policy_for(c)
            cache[c] = pol.loss_given_fraud if pol else DEFAULT_LOSS_GIVEN_FRAUD
        out[i] = cache[c]
    return out


def _actions(w: pd.DataFrame, cm: CostModel, step_channels: set, provisional: bool) -> np.ndarray:
    can_step = np.array([(c in step_channels) or (t in decisioning.STEP_UP_TXN_TYPES)
                         for c, t in zip(w["channel"], w["transaction_type"])])
    actions = np.empty(len(w), dtype=object)
    lgf = w["_lgf"].to_numpy()
    for value in np.unique(lgf):                                 # per-jurisdiction loss-given-fraud
        m = lgf == value
        actions[m] = _choose_actions(w["p"].to_numpy()[m], w["amount_usd"].to_numpy()[m], can_step[m],
                                     replace(cm, loss_given_fraud=float(value)))
    if provisional:
        actions[actions == "DECLINE"] = "REVIEW"
    return actions.astype(str)


def _outcomes(w: pd.DataFrame, a: np.ndarray, cm: CostModel) -> pd.DataFrame:
    """Per-transaction realised components (same accounting as decisioning.backtest_policy)."""
    y = w["y"].to_numpy()
    fraud, genuine = y == 1, y == 0
    loss = w["amount_usd"].to_numpy() * w["_lgf"].to_numpy()
    catch = np.select([a == "DECLINE", a == "REVIEW", a == "STEP_UP"], [1.0, cm.review_catch_rate, cm.step_up_catch_rate], 0.0)
    friction = np.select([(a == "DECLINE") & genuine, (a == "REVIEW") & genuine, (a == "STEP_UP") & genuine],
                         [cm.decline_friction_usd + cm.attrition_rate_on_false_decline * cm.customer_annual_value_usd,
                          cm.review_friction_usd, cm.step_up_friction_usd], 0.0)
    return pd.DataFrame({
        "date": w["date"].to_numpy(),
        "prevented": np.where(fraud, loss * catch, 0.0),
        "residual_loss": np.where(fraud, loss * (1 - catch), 0.0),
        "friction_cost": friction,
        "review_cost": np.where(a == "REVIEW", cm.review_cost_usd, 0.0),
        "declined_genuine": ((a == "DECLINE") & genuine).astype(int),
        "challenged_genuine": ((a == "STEP_UP") & genuine).astype(int),
        "reviewed_genuine": ((a == "REVIEW") & genuine).astype(int),
        "disturbed_genuine": ((a != "APPROVE") & genuine).astype(int),
        "fraud_caught": (fraud & (a != "APPROVE")).astype(int),
        "cases": np.isin(a, ["REVIEW", "DECLINE"]).astype(int),
        "genuine": genuine.astype(int),
    })


def _summary(o: pd.DataFrame, a: np.ndarray, w: pd.DataFrame, days: int, aht_minutes: float) -> Dict:
    total_cost = o["residual_loss"].sum() + o["friction_cost"].sum() + o["review_cost"].sum()
    s = get_settings()
    cases_per_day = o["cases"].sum() / days
    fairness = {}
    for attr in ("customer_segment", "customer_age_band"):
        g = pd.DataFrame({"k": w[attr].astype("string").fillna("(unknown)").to_numpy(), "d": o["disturbed_genuine"],
                          "g": o["genuine"]})
        g = g[g["g"] == 1].groupby("k")["d"].agg(["mean", "size"])
        overall = o.loc[o["genuine"] == 1, "disturbed_genuine"].mean()
        fairness[attr] = {k: {"genuine_intervention_rate": round(float(r["mean"]), 4),
                              "ratio_to_overall": round(float(r["mean"] / overall), 2) if overall else None,
                              "genuine_customers": int(r["size"])} for k, r in g.iterrows()}
    return {
        "fraud_loss_prevented_usd": round(float(o["prevented"].sum()), 0),
        "residual_fraud_loss_usd": round(float(o["residual_loss"].sum()), 0),
        "frauds_intervened": int(o["fraud_caught"].sum()),
        "genuine_declined": int(o["declined_genuine"].sum()),
        "genuine_challenged": int(o["challenged_genuine"].sum()),
        "genuine_held_for_review": int(o["reviewed_genuine"].sum()),
        "genuine_disturbed": int(o["disturbed_genuine"].sum()),
        "realised_total_cost_usd": round(float(total_cost), 0),
        "cases_per_day": round(float(cases_per_day), 2),
        "analyst_workload_fte": round(float(cases_per_day * aht_minutes / 60 / s.planning_max_occupancy
                                            / (s.planning_paid_hours_per_day * (1 - s.planning_shrinkage))), 2),
        "action_mix": {x: round(float((a == x).mean()), 4) for x in ACTIONS},
        "fairness": fairness,
    }


def _bootstrap(base: pd.DataFrame, new: pd.DataFrame, n_boot: int, seed: int) -> Dict:
    cols = ["prevented", "residual_loss", "disturbed_genuine", "declined_genuine", "challenged_genuine", "cases"]
    b = base.groupby("date")[cols + ["friction_cost", "review_cost"]].sum()
    n = new.groupby("date")[cols + ["friction_cost", "review_cost"]].sum().reindex(b.index, fill_value=0)
    d = n - b
    d["total_cost"] = (n["residual_loss"] + n["friction_cost"] + n["review_cost"]) - \
        (b["residual_loss"] + b["friction_cost"] + b["review_cost"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    arr = d[cols + ["total_cost"]].to_numpy()
    sums = arr[idx].sum(axis=1)
    point = arr.sum(axis=0)
    return {c: {"change": round(float(point[i]), 1), "ci95": [round(float(np.percentile(sums[:, i], 2.5)), 1),
                                                               round(float(np.percentile(sums[:, i], 97.5)), 1)]}
            for i, c in enumerate(cols + ["total_cost"])}


def simulate(proposal: Dict, n_boot: int = 500, seed: int = 0, save: bool = True) -> Dict:
    unknown = set(proposal) - ALLOWED_KEYS
    if unknown:
        raise WhatIfError(f"unknown proposal keys: {sorted(unknown)}; allowed: {sorted(ALLOWED_KEYS)}")
    bad = set(proposal.get("cost_overrides", {})) - set(CostModel.__dataclass_fields__)
    if bad:
        raise WhatIfError(f"unknown cost-model fields: {sorted(bad)}")
    from bti.modeling import registry
    from bti.operations.capacity import capacity_overrides
    from bti.planning.replay import scored_window
    from bti.planning.staffing import case_rates, handling_minutes

    live_model, w = scored_window()
    w = w.assign(_lgf=_lgf(w["country"]))
    live_provisional = registry.model_for_role("champion") != live_model
    current_cm = cost_model_for(None, capacity_overrides(live_model))
    current = {"model_id": live_model, "cost_model": current_cm, "step_up_channels": set(decisioning.STEP_UP_CHANNELS),
               "provisional": live_provisional}

    p_model = proposal.get("model_id") or live_model
    pw = w
    if p_model != live_model:
        _, other = scored_window(p_model)
        if not other["transaction_id"].equals(w["transaction_id"]):
            raise WhatIfError("the proposed model was evaluated on a different window; compare on shared data")
        pw = w.assign(p=other["p"].to_numpy())
    cm = cost_model_for(None, {**(capacity_overrides(p_model) or {}), **proposal.get("cost_overrides", {})})
    if "loss_given_fraud" in proposal.get("cost_overrides", {}):        # an explicit figure replaces every jurisdiction's
        pw = pw.assign(_lgf=float(proposal["cost_overrides"]["loss_given_fraud"]))
    refit = None
    if proposal.get("max_review_rate") is not None or proposal.get("max_step_up_rate") is not None:
        _, cal = scored_window(p_model, window="calibration")
        base_cm = cost_model_for(None, proposal.get("cost_overrides", {}))
        from bti.operations.capacity import load_policy
        live_targets = ((load_policy(p_model) or load_policy(live_model) or {}).get("current", {}).get("targets")
                        or {"max_review_rate": get_settings().max_review_rate,
                            "max_step_up_rate": get_settings().max_step_up_rate})
        targets = {k: proposal[k] if proposal.get(k) is not None else live_targets.get(k)
                   for k in ("max_review_rate", "max_step_up_rate")}               # unspecified targets stay as live
        cm = decisioning.fit_capacity(cal["p"], cal["amount_usd"], cal["channel"], cal["transaction_type"], base_cm,
                                      targets["max_review_rate"], targets["max_step_up_rate"])
        refit = {"targets": targets, "review_cost_usd": cm.review_cost_usd,
                 "step_up_friction_usd": cm.step_up_friction_usd}
    step = set(proposal["step_up_channels"]) if "step_up_channels" in proposal else current["step_up_channels"]
    prov = bool(proposal.get("provisional", registry.model_for_role("champion") != p_model))

    days = max(int(w["date"].nunique()), 1)
    rates = case_rates(live_model)
    aht = handling_minutes()
    weights = rates["rates_by_queue"]
    aht_avg = sum(weights[q] * aht[q]["minutes"] for q in aht) / max(sum(weights.values()), 1e-9)

    a0 = _actions(w, current_cm, current["step_up_channels"], live_provisional)
    a1 = _actions(pw, cm, step, prov)
    # decisions use the capacity shadow prices; realised cost uses the economic figures (shadow prices ration
    # capacity, they are not money spent)
    econ0 = cost_model_for(None)
    econ1 = cost_model_for(None, proposal.get("cost_overrides", {}))
    o0, o1 = _outcomes(w, a0, econ0), _outcomes(pw, a1, econ1)
    s0, s1 = _summary(o0, a0, w, days, aht_avg), _summary(o1, a1, pw, days, aht_avg)
    max_ratio = lambda s: max(v["ratio_to_overall"] or 0 for attr in s["fairness"].values() for v in attr.values())
    report = {
        "id": hashlib.sha256(json.dumps(proposal, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "proposal": proposal, "status": "simulation only — nothing has been changed",
        "window": {"transactions": int(len(w)), "days": days, "from": str(w["date"].min().date()),
                   "to": str(w["date"].max().date()), "frauds": int(w["y"].sum())},
        "current_policy": {"model_id": live_model, "provisional": live_provisional,
                           "capacity_prices": capacity_overrides(live_model), **s0},
        "proposed_policy": {"model_id": p_model, "provisional": prov, "refitted_capacity_prices": refit,
                            "step_up_channels": sorted(step), **s1},
        "difference_with_95pct_ci": _bootstrap(o0, o1, n_boot, seed),
        "fairness_check": {"current_max_ratio": max_ratio(s0), "proposed_max_ratio": max_ratio(s1),
                           "flag": max_ratio(s1) > max(1.25, max_ratio(s0) + 0.1)},
        "how_to_apply": "Capacity targets: python -m bti.operations.capacity --fitted-by '<name>' ...; cost-model "
                        "changes: settings review and approval; model change: registry promotion with four-eyes.",
    }
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"{report['id']}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Policy what-if simulator (read-only)")
    parser.add_argument("proposal", help='JSON, e.g. \'{"max_review_rate": 0.03}\'')
    args = parser.parse_args()
    r = simulate(json.loads(args.proposal))
    keys = ("fraud_loss_prevented_usd", "residual_fraud_loss_usd", "genuine_declined", "genuine_challenged",
            "genuine_held_for_review", "realised_total_cost_usd", "cases_per_day", "analyst_workload_fte")
    print(f"what-if {r['id']} on {r['window']['transactions']} transactions ({r['window']['days']} days)")
    for k in keys:
        print(f"  {k:28s} {r['current_policy'][k]:>14,} -> {r['proposed_policy'][k]:>14,}")
    print(json.dumps(r["difference_with_95pct_ci"], indent=1))
    print("fairness:", r["fairness_check"])


if __name__ == "__main__":
    main()
