"""
Expected-cost decisioning.

Instead of one global threshold, each transaction gets the action with the
lowest expected cost given its calibrated fraud probability and amount:

  APPROVE  cost = p · loss
  STEP_UP  cost = (1-p) · challenge friction + p · loss · (1 - step-up catch rate)
  REVIEW   cost = analyst cost + (1-p) · hold friction + p · loss · (1 - review catch rate)
  DECLINE  cost = (1-p) · (decline friction + attrition · customer value)

where loss = USD amount × jurisdiction loss-given-fraud. A $40 purchase and a
$40,000 wire with the same probability therefore get different actions — which
is how fraud losses and customer friction are traded off in practice.

Guardrails:
  * never decline a customer who is more likely genuine than fraudulent
    (p below `min_decline_probability`); step-up or review instead
  * a provisional (not yet approved) model may never auto-decline — declines
    are downgraded to REVIEW
  * step-up is offered only on channels that can challenge the customer
  * in GDPR / UK GDPR jurisdictions a decline carries a human-review route
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Optional

import numpy as np

from bti.jurisdiction.policies import DEFAULT_LOSS_GIVEN_FRAUD, JurisdictionPolicy

ACTIONS = ("APPROVE", "STEP_UP", "REVIEW", "DECLINE")
STEP_UP_CHANNELS = {"Mobile Banking", "Internet Banking", "API/Open Banking"}
STEP_UP_TXN_TYPES = {"Card Not Present"}


@dataclass(frozen=True)
class CostModel:
    """Illustrative defaults — calibrate every figure with the bank's own loss and operations data."""
    loss_given_fraud: float = DEFAULT_LOSS_GIVEN_FRAUD
    decline_friction_usd: float = 15.0
    attrition_rate_on_false_decline: float = 0.02
    customer_annual_value_usd: float = 300.0
    step_up_friction_usd: float = 3.00     # abandonment, contact-centre calls and dissatisfaction, not just the OTP
    step_up_catch_rate: float = 0.60
    review_cost_usd: float = 8.0
    review_friction_usd: float = 4.0
    review_catch_rate: float = 0.90
    min_decline_probability: float = 0.50


@dataclass
class Decision:
    action: str
    expected_costs_usd: Dict[str, float]
    expected_saving_vs_approve_usd: float
    guardrails_applied: List[str]
    human_review_route: bool
    cost_model: Dict[str, float]


def cost_model_for(policy: Optional[JurisdictionPolicy], overrides: Optional[Dict] = None) -> CostModel:
    cm = CostModel()
    if policy is not None:
        cm = replace(cm, loss_given_fraud=policy.loss_given_fraud)
    if overrides:
        cm = replace(cm, **{k: float(v) for k, v in overrides.items() if hasattr(cm, k)})
    return cm


def expected_costs(p, amount_usd, cm: CostModel) -> Dict[str, np.ndarray]:
    p = np.asarray(p, dtype=float)
    loss = np.nan_to_num(np.asarray(amount_usd, dtype=float)) * cm.loss_given_fraud
    genuine = 1 - p
    return {
        "APPROVE": p * loss,
        "STEP_UP": genuine * cm.step_up_friction_usd + p * loss * (1 - cm.step_up_catch_rate),
        "REVIEW": cm.review_cost_usd + genuine * cm.review_friction_usd + p * loss * (1 - cm.review_catch_rate),
        "DECLINE": genuine * (cm.decline_friction_usd + cm.attrition_rate_on_false_decline * cm.customer_annual_value_usd),
    }


def step_up_available(channel: Optional[str], transaction_type: Optional[str]) -> bool:
    return (channel in STEP_UP_CHANNELS) or (transaction_type in STEP_UP_TXN_TYPES)


def decide(p: float, amount_usd: float, policy: Optional[JurisdictionPolicy] = None,
           channel: Optional[str] = None, transaction_type: Optional[str] = None,
           provisional_model: bool = False, cost_overrides: Optional[Dict] = None) -> Decision:
    cm = cost_model_for(policy, cost_overrides)
    costs = {a: float(v[0]) for a, v in expected_costs([p], [amount_usd], cm).items()}
    allowed = list(ACTIONS)
    guardrails: List[str] = []
    if not step_up_available(channel, transaction_type):
        allowed.remove("STEP_UP")
        guardrails.append("Step-up unavailable on this channel")
    if p < cm.min_decline_probability:
        allowed.remove("DECLINE")
        if costs["DECLINE"] < min(costs[a] for a in allowed):
            guardrails.append(f"Decline withheld: probability below {cm.min_decline_probability:.0%}")
    action = min(allowed, key=lambda a: costs[a])
    if action == "DECLINE" and provisional_model:
        action = "REVIEW"
        guardrails.append("Provisional model — automatic decline downgraded to analyst review")
    human_route = bool(action == "DECLINE" and policy is not None and policy.decline_requires_human_review_route)
    if human_route:
        guardrails.append("GDPR Art. 22 — decline notice must offer human review")
    return Decision(
        action=action,
        expected_costs_usd={a: round(c, 2) for a, c in costs.items()},
        expected_saving_vs_approve_usd=round(costs["APPROVE"] - costs[action], 2),
        guardrails_applied=guardrails,
        human_review_route=human_route,
        cost_model=asdict(cm),
    )


def _choose_actions(p: np.ndarray, amt: np.ndarray, can_step: np.ndarray, cm: CostModel) -> np.ndarray:
    costs = expected_costs(p, amt, cm)
    matrix = np.vstack([costs[a] for a in ACTIONS])
    matrix[ACTIONS.index("STEP_UP"), ~can_step] = np.inf
    matrix[ACTIONS.index("DECLINE"), p < cm.min_decline_probability] = np.inf
    return np.array(ACTIONS)[np.argmin(matrix, axis=0)]


_BUDGETED = {"REVIEW": "review_cost_usd", "STEP_UP": "step_up_friction_usd"}


def fit_action_budget(p, amount_usd, channels: Iterable, txn_types: Iterable, cm: CostModel,
                      action: str, max_rate: float) -> CostModel:
    """
    Raise an action's unit cost (a capacity shadow price) until the share of
    traffic routed to it fits `max_rate` — analyst capacity for REVIEW, the
    customer-challenge budget for STEP_UP. The fitted cost is then used live.
    """
    field = _BUDGETED[action]
    p = np.asarray(p, dtype=float)
    amt = np.nan_to_num(np.asarray(amount_usd, dtype=float))
    can_step = np.array([step_up_available(c, t) for c, t in zip(channels, txn_types)])

    def rate(cost: float) -> float:
        return float((_choose_actions(p, amt, can_step, replace(cm, **{field: cost})) == action).mean())

    start = getattr(cm, field)
    if rate(start) <= max_rate:
        return cm
    lo, hi = start, max(start * 2, 1.0)
    while rate(hi) > max_rate and hi < 1e9:
        hi *= 2
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rate(mid) > max_rate else (lo, mid)
    return replace(cm, **{field: round(hi, 2)})


def fit_capacity(p, amount_usd, channels, txn_types, cm: CostModel, max_review_rate: Optional[float] = None,
                 max_step_up_rate: Optional[float] = None) -> CostModel:
    channels, txn_types = list(channels), list(txn_types)
    for _ in range(3):   # the two budgets interact; a few alternating passes converge
        if max_step_up_rate is not None:
            cm = fit_action_budget(p, amount_usd, channels, txn_types, cm, "STEP_UP", max_step_up_rate)
        if max_review_rate is not None:
            cm = fit_action_budget(p, amount_usd, channels, txn_types, cm, "REVIEW", max_review_rate)
    return cm


def backtest_policy(y, p, amount_usd, channels: Iterable, txn_types: Iterable,
                    cm: CostModel = CostModel(), reference_threshold: Optional[float] = None) -> Dict:
    """
    Replay the expected-cost policy on labelled history and compare realised
    cost with (a) approving everything and (b) a single-threshold decline rule.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    amt = np.nan_to_num(np.asarray(amount_usd, dtype=float))
    can_step = np.array([step_up_available(c, t) for c, t in zip(channels, txn_types)])
    action = _choose_actions(p, amt, can_step, cm)

    def realised(actions: np.ndarray) -> Dict:
        loss = amt * cm.loss_given_fraud
        fraud, genuine = y == 1, y == 0
        cost = np.zeros(len(y))
        a = actions
        cost += np.where((a == "APPROVE") & fraud, loss, 0)
        cost += np.where((a == "STEP_UP") & fraud, loss * (1 - cm.step_up_catch_rate), 0)
        cost += np.where((a == "STEP_UP") & genuine, cm.step_up_friction_usd, 0)
        cost += np.where(a == "REVIEW", cm.review_cost_usd, 0)
        cost += np.where((a == "REVIEW") & fraud, loss * (1 - cm.review_catch_rate), 0)
        cost += np.where((a == "REVIEW") & genuine, cm.review_friction_usd, 0)
        cost += np.where((a == "DECLINE") & genuine,
                         cm.decline_friction_usd + cm.attrition_rate_on_false_decline * cm.customer_annual_value_usd, 0)
        prevented = np.where(fraud & (a == "DECLINE"), loss, 0) \
            + np.where(fraud & (a == "REVIEW"), loss * cm.review_catch_rate, 0) \
            + np.where(fraud & (a == "STEP_UP"), loss * cm.step_up_catch_rate, 0)
        return {
            "total_cost_usd": round(float(cost.sum()), 2),
            "fraud_loss_prevented_usd": round(float(prevented.sum()), 2),
            "genuine_customers_declined": int(((a == "DECLINE") & genuine).sum()),
            "false_decline_rate": round(float(((a == "DECLINE") & genuine).sum() / max(genuine.sum(), 1)), 5),
            "genuine_customers_challenged": int(((a == "STEP_UP") & genuine).sum()),
            "cases_for_review": int((a == "REVIEW").sum()),
            "action_mix": {x: int((a == x).sum()) for x in ACTIONS},
        }

    approve_all = realised(np.full(len(y), "APPROVE"))
    policy = realised(action)
    result = {
        "n": int(len(y)),
        "fraud_value_usd": round(float(amt[y == 1].sum() * cm.loss_given_fraud), 2),
        "expected_cost_policy": policy,
        "approve_all": approve_all,
        "saving_vs_approve_all_usd": round(approve_all["total_cost_usd"] - policy["total_cost_usd"], 2),
        "cost_model": asdict(cm),
    }
    if reference_threshold is not None:
        threshold = realised(np.where(p >= reference_threshold, "DECLINE", "APPROVE"))
        result["single_threshold_decline"] = {"threshold": reference_threshold, **threshold}
        result["saving_vs_single_threshold_usd"] = round(threshold["total_cost_usd"] - policy["total_cost_usd"], 2)
    return result
