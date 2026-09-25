"""
Capacity-constrained decision policy for live scoring.

Expected-cost decisioning prices each action per transaction. Left alone, it
sends every transaction whose expected loss beats an $8 review to an analyst:
about a fifth of all traffic on the synthetic data, far beyond any fraud team.

This module fits capacity "shadow prices" per model: the review cost and
step-up friction at which the policy stays within the bank's analyst capacity
and customer-challenge budget. The fit is on the model's calibration window,
checked on the out-of-time window, and stored per model with its history.
Live and shadow decisions then use their own model's prices.

Usage:
  python -m bti.operations.capacity --fitted-by "Srijan Upadhyay" --max-review-rate 0.02 --max-step-up-rate 0.05
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from bti.config import get_settings
from bti.modeling import registry
from bti.operations.decisioning import (
    ACTIONS, CostModel, _choose_actions, fit_capacity, step_up_available,
)

_lock = threading.Lock()
_cache: Dict[str, Optional[Dict]] = {}
FITTED_FIELDS = ("review_cost_usd", "step_up_friction_usd")


def policy_dir() -> Path:
    return registry.registry_dir() / "policies"


def _path(model_id: str) -> Path:
    return policy_dir() / f"{model_id}.json"


def load_policy(model_id: str) -> Optional[Dict]:
    with _lock:
        if model_id not in _cache:
            p = _path(model_id)
            _cache[model_id] = json.loads(p.read_text()) if p.exists() else None
        return _cache[model_id]


def capacity_overrides(model_id: str) -> Optional[Dict[str, float]]:
    policy = load_policy(model_id)
    return dict(policy["current"]["overrides"]) if policy else None


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _mix(p, amt, channels, types, cm: CostModel) -> Dict[str, float]:
    can_step = np.array([step_up_available(c, t) for c, t in zip(channels, types)])
    a = _choose_actions(np.asarray(p, float), np.nan_to_num(np.asarray(amt, float)), can_step, cm)
    return {x: round(float((a == x).mean()), 5) for x in ACTIONS}


def fit_live_policy(fitted_by: str, model_id: Optional[str] = None, max_review_rate: Optional[float] = None,
                    max_step_up_rate: Optional[float] = None, data_path=None) -> Dict:
    from bti.modeling.reassess import model_probabilities
    from bti.modeling.train import prepare

    if not fitted_by or not fitted_by.strip():
        raise ValueError("fitted_by is required")
    s = get_settings()
    max_review_rate = max_review_rate if max_review_rate is not None else s.max_review_rate
    max_step_up_rate = max_step_up_rate if max_step_up_rate is not None else s.max_step_up_rate
    model_id = model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
    if not model_id:
        raise registry.RegistryError("No model to fit a policy for")
    version = registry.load_artifact(model_id).get("feature_version", 1)
    data = prepare(data_path, feature_version=version)
    p = model_probabilities(model_id, data)
    amt = data.features["amount_usd"].to_numpy(float)
    ch, tt = data.df["channel"].to_numpy(), data.df["transaction_type"].to_numpy()
    ca, te = data.ca, data.te

    base = CostModel()
    fitted = fit_capacity(p[ca], amt[ca], ch[ca], tt[ca], base, max_review_rate, max_step_up_rate)
    overrides = {f: getattr(fitted, f) for f in FITTED_FIELDS}
    entry = {
        "fitted_at": datetime.now(timezone.utc).isoformat(), "fitted_by": fitted_by.strip(),
        "targets": {"max_review_rate": max_review_rate, "max_step_up_rate": max_step_up_rate},
        "overrides": overrides,
        "base_cost_model": asdict(base),
        "fitted_on": "calibration window", "data_sha256": data.sha256,
        "action_mix": {
            "unconstrained_calibration": _mix(p[ca], amt[ca], ch[ca], tt[ca], base),
            "calibration": _mix(p[ca], amt[ca], ch[ca], tt[ca], fitted),
            "out_of_time": _mix(p[te], amt[te], ch[te], tt[te], fitted),
        },
        "note": "Fitted with the default loss-given-fraud; jurisdictions with a different loss-given-fraud land "
                "slightly above or below the targets. Provisional models downgrade declines to reviews, which "
                "adds to review volume.",
    }
    policy_dir().mkdir(parents=True, exist_ok=True)
    existing = load_policy(model_id) or {"model_id": model_id, "history": []}
    existing["history"] = existing.get("history", []) + ([existing["current"]] if "current" in existing else [])
    existing["current"] = entry
    tmp = _path(model_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2, default=str))
    tmp.replace(_path(model_id))
    with _lock:
        _cache.pop(model_id, None)
    return {"model_id": model_id, **entry}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the live capacity policy for the scoring model")
    parser.add_argument("--fitted-by", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-review-rate", type=float, default=None)
    parser.add_argument("--max-step-up-rate", type=float, default=None)
    args = parser.parse_args()
    r = fit_live_policy(args.fitted_by, args.model, args.max_review_rate, args.max_step_up_rate)
    print(f"{r['model_id']}: review cost ${r['overrides']['review_cost_usd']}, "
          f"step-up friction ${r['overrides']['step_up_friction_usd']}")
    for window, mix in r["action_mix"].items():
        print(f"  {window:26s} " + "  ".join(f"{a} {v:.2%}" for a, v in mix.items()))


if __name__ == "__main__":
    main()
