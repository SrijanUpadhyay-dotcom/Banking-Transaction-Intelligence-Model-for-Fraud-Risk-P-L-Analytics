"""
Bounded recalibration on matured labels, a minor model change.

Fraud rates move: seasonality, new attack waves, and changes in the bank's
book. The model's ranking can stay good while its probabilities drift, and the
expected-cost decisions depend on the probabilities. So a small overlay is
refitted on top of the model's own calibrated probability:

    p' = sigmoid(α · logit(p) + β)

It is monotone (α > 0), so the ranking, reason codes and SHAP explanations are
unchanged. A candidate overlay is accepted only if all of these hold:

- the window holds at least `min_fraud` matured frauds and `min_genuine`
  genuine transactions
- fitted on the earlier 70% of the window, it lowers ECE on the later 30%,
  compared with both the current overlay and no overlay
- α lies in [0.67, 1.5] and |β| ≤ 1.0 (anything bigger is a model problem
  for retraining, not recalibration)
- it changes no more than 5% of decisions at the decline line (p = 0.5) and
  at the model's reference threshold

Accepted overlays are refitted on the full window, versioned per model, applied
by the scorer to live and batch scores, logged to the audit chain as
MODEL_RECALIBRATED, and noted in the registry as a minor change. Rejections are
logged with their reasons.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sqlalchemy.orm import Session

from bti.database.models import AuditLog, ScoreLog
from bti.modeling import registry
from bti.modeling.metrics import expected_calibration_error

BOUNDS = {"alpha": (0.67, 1.5), "beta_abs": 1.0, "max_decision_change": 0.05, "min_fraud": 100,
          "min_genuine": 1000, "holdout_share": 0.3}
_CLIP = 1e-6
_lock = threading.Lock()
_cache: Dict[str, Optional[Dict]] = {}


def _dir() -> Path:
    return registry.registry_dir() / "calibrations"


def load_overlay(model_id: str) -> Optional[Dict]:
    with _lock:
        if model_id not in _cache:
            path = _dir() / f"{model_id}.json"
            _cache[model_id] = json.loads(path.read_text()) if path.exists() else None
        return _cache[model_id]


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _transform(p, alpha: float, beta: float) -> np.ndarray:
    return expit(alpha * logit(np.clip(np.asarray(p, dtype=float), _CLIP, 1 - _CLIP)) + beta)


def apply_overlay(model_id: str, p):
    """Model probability → recalibrated probability (identity when no overlay is active)."""
    overlay = load_overlay(model_id)
    if not overlay or not overlay.get("current"):
        return p
    c = overlay["current"]
    out = _transform(p, c["alpha"], c["beta"])
    return float(out) if np.ndim(p) == 0 else out


def overlay_version(model_id: str) -> Optional[int]:
    overlay = load_overlay(model_id)
    return overlay["current"]["version"] if overlay and overlay.get("current") else None


def fit_overlay(y: np.ndarray, p: np.ndarray):
    z = logit(np.clip(p, _CLIP, 1 - _CLIP)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y.astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _matured(db: Session, model_id: str, start: datetime, end: datetime, maturity_days: int, now: datetime):
    from bti.operations.feedback import labelled_scores
    frame = labelled_scores(db, start, end, maturity_days, shadow=None, now=now)
    frame = frame[(frame["model_id"] == model_id) & frame["label"].notna()]
    if frame.empty:
        return frame
    base = dict(db.query(ScoreLog.transaction_id, ScoreLog.model_probability).filter(
        ScoreLog.model_id == model_id, ScoreLog.scored_at >= start, ScoreLog.scored_at < end).all())
    frame = frame.sort_values("scored_at", kind="stable").drop_duplicates("transaction_id", keep="last").copy()
    frame["model_probability"] = [base.get(t) if base.get(t) is not None else p
                                  for t, p in zip(frame["transaction_id"], frame["fraud_probability"])]
    return frame


def recalibrate(db: Session, model_id: Optional[str] = None, window_days: int = 90, maturity_days: int = 90,
                now: Optional[datetime] = None, apply: bool = True, actor: str = "bti.modeling.recalibration") -> Dict:
    now = now or datetime.utcnow()
    model_id = model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
    end = now - timedelta(days=maturity_days)
    start = end - timedelta(days=window_days)
    frame = _matured(db, model_id, start, end, maturity_days, now)
    result: Dict = {"model_id": model_id, "window": {"from": start.isoformat(), "to": end.isoformat()},
                    "bounds": BOUNDS, "applied": False}
    n_fraud = int((frame["label"] == 1).sum()) if not frame.empty else 0
    n_genuine = int((frame["label"] == 0).sum()) if not frame.empty else 0
    result["labels"] = {"fraud": n_fraud, "genuine": n_genuine}
    if n_fraud < BOUNDS["min_fraud"] or n_genuine < BOUNDS["min_genuine"]:
        result["status"] = "insufficient_labels"
        result["reasons"] = [f"{n_fraud} matured frauds and {n_genuine} genuine; need {BOUNDS['min_fraud']} and "
                             f"{BOUNDS['min_genuine']}"]
        return _log(db, result, now)

    y = frame["label"].astype(int).to_numpy()
    p = frame["model_probability"].astype(float).to_numpy()
    cut = int(len(y) * (1 - BOUNDS["holdout_share"]))
    alpha, beta = fit_overlay(y[:cut], p[:cut])
    current = (load_overlay(model_id) or {}).get("current")
    hold_y, hold_p = y[cut:], p[cut:]
    candidate = _transform(hold_p, alpha, beta)
    active = _transform(hold_p, current["alpha"], current["beta"]) if current else hold_p
    ece = {"no_overlay": round(expected_calibration_error(hold_y, hold_p), 5),
           "current_overlay": round(expected_calibration_error(hold_y, active), 5),
           "candidate": round(expected_calibration_error(hold_y, candidate), 5)}
    ref = registry.load_artifact(model_id).get("reference_threshold")
    lines = [0.5] + ([float(ref)] if ref is not None else [])
    changed = float(np.mean(np.any([(candidate >= t) != (active >= t) for t in lines], axis=0)))
    reasons = []
    if not BOUNDS["alpha"][0] <= alpha <= BOUNDS["alpha"][1]:
        reasons.append(f"α {alpha:.3f} outside {BOUNDS['alpha']}: a change this large needs retraining")
    if abs(beta) > BOUNDS["beta_abs"]:
        reasons.append(f"|β| {abs(beta):.3f} above {BOUNDS['beta_abs']}: a change this large needs retraining")
    if not ece["candidate"] < min(ece["no_overlay"], ece["current_overlay"]):
        reasons.append(f"holdout ECE {ece['candidate']} does not beat the current {ece['current_overlay']} and "
                       f"no-overlay {ece['no_overlay']}")
    if changed > BOUNDS["max_decision_change"]:
        reasons.append(f"{changed:.1%} of decisions would change (limit {BOUNDS['max_decision_change']:.0%})")
    result.update({"holdout_ece": ece, "candidate": {"alpha": round(alpha, 5), "beta": round(beta, 5)},
                   "decision_change_share": round(changed, 5), "reasons": reasons})
    if reasons:
        result["status"] = "rejected"
        return _log(db, result, now)

    alpha_full, beta_full = fit_overlay(y, p)
    if not (BOUNDS["alpha"][0] <= alpha_full <= BOUNDS["alpha"][1] and abs(beta_full) <= BOUNDS["beta_abs"]):
        result["status"] = "rejected"
        result["reasons"] = ["full-window refit fell outside the bounds"]
        return _log(db, result, now)
    result["status"] = "accepted"
    if apply:
        store = load_overlay(model_id) or {"model_id": model_id, "history": []}
        version = (current["version"] + 1) if current else 1
        entry = {"version": version, "alpha": round(alpha_full, 6), "beta": round(beta_full, 6),
                 "fitted_at": datetime.now(timezone.utc).isoformat(), "fitted_by": actor,
                 "window": result["window"], "labels": result["labels"], "holdout_ece": ece}
        store["history"] = store.get("history", []) + ([current] if current else [])
        store["current"] = entry
        _dir().mkdir(parents=True, exist_ok=True)
        tmp = _dir() / f"{model_id}.json.tmp"
        tmp.write_text(json.dumps(store, indent=2))
        tmp.replace(_dir() / f"{model_id}.json")
        with _lock:
            _cache.pop(model_id, None)
        registry.add_note(model_id, actor, "Minor change: recalibration",
                          f"Overlay v{version} (α {entry['alpha']}, β {entry['beta']}) fitted on "
                          f"{n_fraud + n_genuine:,} matured transactions; holdout ECE {ece['current_overlay']} → "
                          f"{ece['candidate']}; {changed:.1%} of decisions change. Ranking unchanged.", entry)
        result["applied"], result["version"] = True, version
    return _log(db, result, now)


def rollback(db: Session, model_id: str, actor: str, reason: str) -> Dict:
    """Return to the previous overlay (or none)."""
    store = load_overlay(model_id)
    if not store or not store.get("current"):
        raise ValueError(f"{model_id} has no active overlay")
    previous = store["history"][-1] if store.get("history") else None
    store["history"] = store["history"][:-1] if previous else []
    store.setdefault("rolled_back", []).append({**store["current"], "rolled_back_at": datetime.utcnow().isoformat(),
                                                "rolled_back_by": actor, "reason": reason})
    store["current"] = previous
    (_dir() / f"{model_id}.json").write_text(json.dumps(store, indent=2))
    with _lock:
        _cache.pop(model_id, None)
    registry.add_note(model_id, actor, "Recalibration rolled back", reason)
    db.add(AuditLog(ts=datetime.utcnow(), event_type="MODEL_RECALIBRATION_ROLLED_BACK",
                    payload={"model_id": model_id, "actor": actor, "reason": reason,
                             "now_active": previous["version"] if previous else None}))
    db.commit()
    return {"model_id": model_id, "active_version": previous["version"] if previous else None}


def _log(db: Session, result: Dict, now: datetime) -> Dict:
    event = "MODEL_RECALIBRATED" if result.get("applied") else "RECALIBRATION_CHECK"
    db.add(AuditLog(ts=now, event_type=event, payload=json.loads(json.dumps(result, default=str))))
    db.commit()
    return result
