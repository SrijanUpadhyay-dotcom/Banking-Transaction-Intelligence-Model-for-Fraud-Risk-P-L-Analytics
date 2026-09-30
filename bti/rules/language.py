"""
The rule condition language.

A condition is a JSON tree, never code:

    {"all": [{"field": "failed_attempt_count", "op": ">=", "value": 3},
             {"not": {"field": "channel", "op": "in", "value": ["Branch"]}}]}

- **Combinators:** `all`, `any`, `not`.
- **Leaves:** `{"field", "op", "value"}`.
- **Operators:** `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `is_null`, `not_null`.
- **Missing values:** a missing value makes every comparison false except
  `is_null`.

**Which fields a rule may use.** The same lineage rule the model obeys: only
fields known before authorisation.

- the model's features
- the model's fraud probability
- the currency
- identifiers for block and watch lists (merchant, payee, device, IP)

**Which fields are refused.** Protected attributes and their proxies (segment,
age band, country, city), post-event fields and label-derived fields are
refused with the reason.

One vectorised evaluator serves both history simulation and live scoring, so a
rule cannot behave differently in the two.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bti.modeling.features import ALL_FEATURE_NAMES, CATEGORICAL_FEATURES, FEATURE_BY_NAME, SOURCE_FIELDS, Availability

OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "not_in", "is_null", "not_null")
NUMERIC_OPS = (">", ">=", "<", "<=")
MAX_DEPTH, MAX_LEAVES, MAX_LIST = 6, 50, 10_000
LIST_FIELDS = ("merchant_name", "payee_id", "device_id", "ip_location")
EXTRA_FIELDS = {
    "fraud_probability": "Calibrated fraud probability from the scoring model",
    "currency": "Transaction currency",
    **{f: f"{f.replace('_', ' ').capitalize()} (identifier: for block and watch lists)" for f in LIST_FIELDS},
}


class RuleError(ValueError):
    pass


def allowed_fields() -> Dict[str, str]:
    out = {name: FEATURE_BY_NAME[name].description for name in ALL_FEATURE_NAMES}
    out.update(EXTRA_FIELDS)
    return out


def _refusal(field: str) -> str:
    src = SOURCE_FIELDS.get(field)
    if src is not None and src.availability is Availability.PROTECTED:
        return f"'{field}' is a protected attribute or proxy ({src.note}); rules may not use it"
    if src is not None and src.availability in (Availability.POST_EVENT, Availability.LABEL_DERIVED):
        return f"'{field}' is {src.availability.value} — not known when the decision is made"
    return f"'{field}' is not a governed rule field; see GET /api/v1/rules/fields"


def validate(condition: Dict) -> List[str]:
    """Return the fields a valid condition uses; raise RuleError explaining the first problem."""
    fields: List[str] = []
    allowed = allowed_fields()

    def walk(node, depth):
        if depth > MAX_DEPTH:
            raise RuleError(f"Conditions may nest at most {MAX_DEPTH} deep")
        if not isinstance(node, dict) or not node:
            raise RuleError("Each condition node must be a non-empty object")
        if "all" in node or "any" in node:
            key = "all" if "all" in node else "any"
            if set(node) != {key} or not isinstance(node[key], list) or not node[key]:
                raise RuleError(f"'{key}' must be the only key and hold a non-empty list")
            for child in node[key]:
                walk(child, depth + 1)
            return
        if "not" in node:
            if set(node) != {"not"}:
                raise RuleError("'not' must be the only key")
            walk(node["not"], depth + 1)
            return
        if set(node) - {"field", "op", "value"} or "field" not in node or "op" not in node:
            raise RuleError(f"A leaf needs 'field', 'op' and (except null checks) 'value': {node}")
        field, op = node["field"], node["op"]
        if field not in allowed:
            raise RuleError(_refusal(field))
        if op not in OPS:
            raise RuleError(f"Unknown operator {op!r}; use one of {OPS}")
        value = node.get("value")
        if op in ("is_null", "not_null"):
            if "value" in node:
                raise RuleError(f"'{op}' takes no value")
        elif op in ("in", "not_in"):
            if not isinstance(value, list) or not value or len(value) > MAX_LIST:
                raise RuleError(f"'{op}' needs a list of 1–{MAX_LIST} values")
        elif op in NUMERIC_OPS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuleError(f"'{op}' on {field} needs a number")
            if field in CATEGORICAL_FEATURES or field in LIST_FIELDS or field == "currency":
                raise RuleError(f"'{op}' is not meaningful on the text field {field}")
        elif value is None:
            raise RuleError(f"'{op}' needs a value")
        fields.append(field)

    walk(condition, 0)
    if len(fields) > MAX_LEAVES:
        raise RuleError(f"At most {MAX_LEAVES} conditions per rule")
    return sorted(set(fields))


def evaluate(condition: Dict, frame: pd.DataFrame) -> np.ndarray:
    """Boolean hit per row of `frame` (columns are field names)."""
    def leaf(node) -> np.ndarray:
        field, op, value = node["field"], node["op"], node.get("value")
        col = frame[field] if field in frame.columns else pd.Series([np.nan] * len(frame), index=frame.index)
        null = col.isna().to_numpy()
        if op == "is_null":
            return null
        if op == "not_null":
            return ~null
        if op in ("in", "not_in"):
            hit = col.astype("string").isin([str(v) for v in value]).fillna(False).to_numpy(bool)
            return (hit if op == "in" else ~hit) & ~null
        if op in NUMERIC_OPS:
            num = pd.to_numeric(col, errors="coerce").to_numpy(float)
            with np.errstate(invalid="ignore"):
                out = {">": num > value, ">=": num >= value, "<": num < value, "<=": num <= value}[op]
            return out & ~np.isnan(num)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            num = pd.to_numeric(col, errors="coerce").to_numpy(float)
            eq = num == float(value)
        else:
            eq = col.astype("string").fillna("").to_numpy() == str(value)
        return (eq if op == "==" else ~eq) & ~null

    def walk(node) -> np.ndarray:
        if "all" in node:
            return np.logical_and.reduce([walk(c) for c in node["all"]])
        if "any" in node:
            return np.logical_or.reduce([walk(c) for c in node["any"]])
        if "not" in node:
            return ~walk(node["not"])
        return leaf(node)

    return np.asarray(walk(condition), dtype=bool).reshape(len(frame))


def describe(condition: Dict) -> str:
    """Human-readable form for documentation and the audit trail."""
    if "all" in condition:
        return "(" + " AND ".join(describe(c) for c in condition["all"]) + ")"
    if "any" in condition:
        return "(" + " OR ".join(describe(c) for c in condition["any"]) + ")"
    if "not" in condition:
        return "NOT " + describe(condition["not"])
    value = condition.get("value")
    if isinstance(value, list) and len(value) > 5:
        value = f"[{len(value)} values]"
    return f"{condition['field']} {condition['op']}" + ("" if condition["op"] in ("is_null", "not_null") else f" {value}")
