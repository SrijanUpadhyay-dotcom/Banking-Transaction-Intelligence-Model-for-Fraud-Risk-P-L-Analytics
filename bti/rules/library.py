"""
Starter rule library, created as drafts for analysts to simulate, adjust and approve.

Two rules answer validation findings directly:
- *Velocity burst* covers card-testing and bot bursts, which the model does not
  react to: its velocity features carry no weight (Phase 3 finding).
- *First transaction draining the balance* covers the new-customer
  account-takeover gap left when amount_to_balance was removed for fairness.
"""

from __future__ import annotations

from typing import Dict, List

from sqlalchemy.orm import Session

from bti.database.models import RuleVersion
from bti.rules.lifecycle import create_rule

STARTER_RULES: List[Dict] = [
    {"name": "Velocity burst",
     "description": "Five or more transactions by the customer in the previous hour: card testing or a bot. The model's "
                    "velocity features carry no weight on current data, so this rule covers the gap.",
     "condition": {"field": "cust_txn_count_1h", "op": ">=", "value": 5}, "action": "REVIEW"},
    {"name": "Failed-authentication burst",
     "description": "Three or more failed credential attempts in the session.",
     "condition": {"field": "failed_attempt_count", "op": ">=", "value": 3}, "action": "STEP_UP"},
    {"name": "New device with login anomaly",
     "description": "A device the customer has never used, after four or more login attempts.",
     "condition": {"all": [{"field": "device_new_for_customer", "op": "==", "value": 1},
                           {"field": "login_attempts", "op": ">=", "value": 4}]}, "action": "REVIEW"},
    {"name": "Repeated just-below-threshold amounts",
     "description": "Two or more amounts just under a reporting or limit threshold in seven days (structuring or limit "
                    "probing).",
     "condition": {"field": "cust_near_threshold_7d", "op": ">=", "value": 2}, "action": "REVIEW"},
    {"name": "First transaction draining the balance",
     "description": "No transaction history and the amount is 80% or more of the balance: covers new-customer account "
                    "takeover after amount_to_balance was removed from the model for fairness.",
     "condition": {"all": [{"field": "secs_since_last_txn", "op": "is_null"},
                           {"field": "amount_to_balance", "op": ">=", "value": 0.8}]}, "action": "STEP_UP"},
]


def seed(db: Session, author: str) -> List[Dict]:
    existing = {r.name for r in db.query(RuleVersion).all()}
    return [create_rule(db, r["name"], r["description"], r["condition"], r["action"], author)
            for r in STARTER_RULES if r["name"] not in existing]
