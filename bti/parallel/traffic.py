"""
Randomised traffic split between the incumbent and BTI, with fallback.

**Assignment.**
- A salted hash of the customer ID (or the transaction ID) puts each unit in the
  BTI arm with probability `bti_share`.
- Assignment by customer is the default. A customer then always gets the same
  system, so their experience is consistent and one arm's decisions cannot
  contaminate the other's.
- The assignment is deterministic, so anyone can recompute it from the salt.

**Governance.**
- An experiment is proposed by one person and started by a different approver
  (four-eyes).
- The BTI share is capped (`parallel_run.max_bti_share`, 10% by default).
- An experiment can only start when an approved champion model exists: live
  customer decisions are never handed to a provisional model.
- Only one experiment runs at a time. Stopping one is immediate.

**Routing (`route`).**
- Every call is scored by BTI under a time budget and logged with both systems'
  answers.
- BTI arm: BTI's decision is the effective one.
- Control arm, or no running experiment: the incumbent's decision is the
  effective one.
- If BTI errors or exceeds the budget, the incumbent's decision is used and the
  fallback is recorded. If the caller did not send the incumbent's decision, the
  response says to apply it.

**Analysis (`experiment_report`).** Arms are compared on matured outcomes:

- fraud loss in basis points of value
- fraud detection
- false declines and customer friction

Confidence intervals come from a bootstrap that resamples customers, because the
customer is the unit of randomisation. A sample-ratio-mismatch check guards
against a broken split.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chisquare, norm
from sqlalchemy.orm import Session, sessionmaker

from bti.config import get_settings
from bti.database.models import AuditLog, RoutedDecision, TrafficExperiment
from bti.logging_config import get_logger
from bti.modeling import registry
from bti.modeling.fx import to_usd
from bti.operations.feedback import DEFAULT_MATURITY_DAYS, latest_labels
from bti.parallel.incumbent import normalise_decision

log = get_logger("parallel.traffic")

UNITS = ("customer", "transaction")
INTERVENTIONS = ("STEP_UP", "REVIEW", "DECLINE")
_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="bti-route")


class ExperimentError(ValueError):
    pass


# ── Assignment ────────────────────────────────────────────────────────────────

def bucket(salt: str, key: str) -> float:
    """Uniform [0, 1) from a salted SHA-256 of the unit key."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def assign_arm(experiment: TrafficExperiment, customer_id: Optional[str], transaction_id: str) -> str:
    key = customer_id if experiment.unit == "customer" and customer_id else transaction_id
    return "bti" if bucket(experiment.salt, str(key)) < experiment.bti_share else "control"


# ── Experiment lifecycle ─────────────────────────────────────────────────────

def _audit(db: Session, event: str, payload: Dict) -> None:
    db.add(AuditLog(ts=datetime.utcnow(), event_type=event, payload=payload))


def _as_dict(e: TrafficExperiment) -> Dict:
    return {c.name: (getattr(e, c.name).isoformat() if isinstance(getattr(e, c.name), datetime)
                     else getattr(e, c.name)) for c in TrafficExperiment.__table__.columns}


def propose(db: Session, name: str, bti_share: float, proposed_by: str, rationale: str,
            unit: str = "customer") -> Dict:
    cap = get_settings().parallel_max_bti_share
    if not 0 < bti_share <= cap:
        raise ExperimentError(f"bti_share must be above 0 and at most {cap:.0%} (parallel_run.max_bti_share)")
    if unit not in UNITS:
        raise ExperimentError(f"unit must be one of {UNITS}")
    if not proposed_by or not proposed_by.strip():
        raise ExperimentError("proposed_by is required")
    if not rationale or len(rationale.strip()) < 10:
        raise ExperimentError("A rationale of at least 10 characters is required")
    if db.query(TrafficExperiment).filter(TrafficExperiment.name == name).first():
        raise ExperimentError(f"An experiment named {name!r} already exists")
    e = TrafficExperiment(name=name, bti_share=bti_share, unit=unit, salt=secrets.token_hex(8), status="proposed",
                          proposed_by=proposed_by.strip(), rationale=rationale.strip())
    db.add(e)
    db.flush()
    _audit(db, "TRAFFIC_EXPERIMENT_PROPOSED", _as_dict(e))
    db.commit()
    return _as_dict(e)


def approve(db: Session, experiment_id: int, approver: str) -> Dict:
    e = db.get(TrafficExperiment, experiment_id)
    if e is None:
        raise ExperimentError(f"No experiment {experiment_id}")
    if e.status != "proposed":
        raise ExperimentError(f"Experiment {experiment_id} is {e.status}; only a proposed experiment can start")
    if not approver or not approver.strip():
        raise ExperimentError("An approver is required")
    if approver.strip().lower() == e.proposed_by.strip().lower():
        raise ExperimentError("Four-eyes principle: the approver must differ from the proposer")
    champion = registry.model_for_role("champion")
    if not champion:
        raise ExperimentError("No approved champion model: live customer decisions cannot be routed to a "
                              "provisional model. Approve a champion first.")
    if db.query(TrafficExperiment).filter(TrafficExperiment.status == "running").first():
        raise ExperimentError("Another experiment is running; stop it first")
    e.status, e.approved_by, e.started_at, e.model_id = "running", approver.strip(), datetime.utcnow(), champion
    _audit(db, "TRAFFIC_EXPERIMENT_STARTED", _as_dict(e))
    db.commit()
    return _as_dict(e)


def stop(db: Session, experiment_id: int, stopped_by: str, reason: str) -> Dict:
    e = db.get(TrafficExperiment, experiment_id)
    if e is None:
        raise ExperimentError(f"No experiment {experiment_id}")
    if e.status not in ("proposed", "running"):
        raise ExperimentError(f"Experiment {experiment_id} is already {e.status}")
    if not stopped_by or not reason:
        raise ExperimentError("stopped_by and reason are required")
    e.status, e.stopped_at, e.stopped_by, e.stop_reason = "stopped", datetime.utcnow(), stopped_by, reason
    _audit(db, "TRAFFIC_EXPERIMENT_STOPPED", _as_dict(e))
    db.commit()
    return _as_dict(e)


def running_experiment(db: Session) -> Optional[TrafficExperiment]:
    return db.query(TrafficExperiment).filter(TrafficExperiment.status == "running").first()


def list_experiments(db: Session) -> List[Dict]:
    return [_as_dict(e) for e in db.query(TrafficExperiment).order_by(TrafficExperiment.id.desc()).all()]


# ── Routing with fallback ────────────────────────────────────────────────────

@dataclass
class Routed:
    transaction_id: str
    arm: str
    experiment_id: Optional[int]
    effective_decision: Optional[str]
    decided_by: str
    bti_decision: Optional[str]
    bti_probability: Optional[float]
    bti_model_id: Optional[str]
    incumbent_decision: Optional[str]
    fallback_reason: Optional[str]
    bti_latency_ms: Optional[float]
    instruction: str


def _score_in_own_session(bind, txn: dict) -> Dict:
    from bti.operations.scoring_service import score_and_decide
    session = sessionmaker(bind=bind)()
    try:
        sd = score_and_decide(txn, session, explain=False)
        return {"decision": sd.decision.action, "probability": sd.live.fraud_probability,
                "model_id": sd.live.model_id, "amount_usd": sd.amount_usd}
    finally:
        session.close()


def route(db: Session, txn: dict, incumbent_decision: Optional[str] = None,
          timeout_ms: Optional[float] = None) -> Routed:
    timeout_ms = timeout_ms or get_settings().parallel_bti_timeout_ms
    tid = str(txn.get("transaction_id"))
    inc = normalise_decision(incumbent_decision) if incumbent_decision else None
    if incumbent_decision and inc is None:
        raise ExperimentError(f"Unknown incumbent decision code {incumbent_decision!r}")
    exp = running_experiment(db)
    arm = assign_arm(exp, txn.get("customer_id"), tid) if exp else "none"

    t0 = time.perf_counter()
    bti, fallback = None, None
    future = _pool.submit(_score_in_own_session, db.get_bind(), txn)
    try:
        bti = future.result(timeout=timeout_ms / 1000)
    except FutureTimeout:
        fallback = "timeout"
    except Exception as exc:                                   # scoring error, no model, bad input
        fallback = "error"
        log.warning("BTI scoring failed in the router", extra={"transaction_id": tid, "error": str(exc)})
    latency = round((time.perf_counter() - t0) * 1000, 2)

    if arm == "bti" and bti is not None:
        effective, decided_by, instruction = bti["decision"], "BTI", "apply effective_decision"
    else:
        effective, decided_by = inc, "INCUMBENT" if inc else "NONE"
        instruction = "apply effective_decision" if inc else "apply the incumbent's decision"
        if arm != "bti":
            fallback = None if bti is not None else fallback   # control arm: BTI failure is not a fallback

    amount_usd = bti["amount_usd"] if bti else _amount_usd(txn)
    db.add(RoutedDecision(
        experiment_id=exp.id if exp else None, transaction_id=tid, customer_id=txn.get("customer_id"), arm=arm,
        bti_model_id=bti["model_id"] if bti else None, bti_probability=bti["probability"] if bti else None,
        bti_decision=bti["decision"] if bti else None, bti_latency_ms=latency if bti else None,
        incumbent_decision=inc, effective_decision=effective, decided_by=decided_by,
        fallback_reason=fallback if arm == "bti" else None, amount_usd=amount_usd, routed_at=datetime.utcnow()))
    db.commit()
    return Routed(tid, arm, exp.id if exp else None, effective, decided_by, bti["decision"] if bti else None,
                  bti["probability"] if bti else None, bti["model_id"] if bti else None, inc,
                  fallback if arm == "bti" else None, latency, instruction)


def _amount_usd(txn: dict) -> Optional[float]:
    try:
        return to_usd(float(txn.get("transaction_amount") or 0), (txn.get("currency") or "USD").upper())
    except Exception:
        return None


# ── Arm comparison ───────────────────────────────────────────────────────────

def _arm_metrics(g: pd.DataFrame) -> Dict:
    y, value = g["label"].astype(int).to_numpy(), g["amount_usd"].fillna(0).to_numpy(float)
    iv = g["effective_decision"].isin(INTERVENTIONS).to_numpy()
    dec = (g["effective_decision"] == "DECLINE").to_numpy()
    fraud, genuine = y == 1, y == 0
    return {
        "transactions": int(len(g)), "customers": int(g["customer_id"].nunique()),
        "fraud": int(fraud.sum()),
        "fraud_loss_bps": round(float(value[fraud & ~iv].sum() / max(value.sum(), 1e-9) * 10_000), 3),
        "tdr": round(float(iv[fraud].mean()), 4) if fraud.any() else None,
        "false_decline_rate": round(float(dec[genuine].mean()), 5) if genuine.any() else None,
        "genuine_intervention_rate": round(float(iv[genuine].mean()), 5) if genuine.any() else None,
    }


def _customer_totals(g: pd.DataFrame) -> pd.DataFrame:
    y, value = g["label"].astype(int), g["amount_usd"].fillna(0).astype(float)
    iv = g["effective_decision"].isin(INTERVENTIONS)
    dec = g["effective_decision"] == "DECLINE"
    t = pd.DataFrame({"customer_id": g["customer_id"], "value": value,
                      "missed_value": value.where((y == 1) & ~iv, 0.0),
                      "fraud": (y == 1).astype(int), "caught": ((y == 1) & iv).astype(int),
                      "genuine": (y == 0).astype(int), "declined": ((y == 0) & dec).astype(int),
                      "disturbed": ((y == 0) & iv).astype(int)})
    return t.groupby("customer_id").sum()


def _ratios(t: np.ndarray) -> np.ndarray:
    """Columns of `t` are the sums of _customer_totals (value, missed_value, fraud, caught, genuine, declined, disturbed)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([t[..., 1] / t[..., 0] * 10_000, t[..., 3] / t[..., 2], t[..., 5] / t[..., 4],
                         t[..., 6] / t[..., 4]], axis=-1)


def _cluster_bootstrap(frame: pd.DataFrame, reps: int, seed: int) -> Dict:
    """Customers are resampled within each arm (the unit of randomisation); ratios are recomputed from totals."""
    rng = np.random.default_rng(seed)
    names = ("fraud_loss_bps", "tdr", "false_decline_rate", "genuine_intervention_rate")
    totals = {arm: _customer_totals(frame[frame["arm"] == arm]).to_numpy(float) for arm in ("bti", "control")}
    draws = {arm: np.stack([t[rng.integers(0, len(t), len(t))].sum(axis=0) for _ in range(reps)])
             for arm, t in totals.items()}
    diff = _ratios(draws["bti"]) - _ratios(draws["control"])
    out = {}
    for j, name in enumerate(names):
        d = diff[:, j][np.isfinite(diff[:, j])]
        out[name] = [round(float(np.percentile(d, 2.5)), 5), round(float(np.percentile(d, 97.5)), 5)] if len(d) else None
    return out


def minimum_detectable_difference(p: float, n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> Optional[float]:
    """Smallest absolute difference in a proportion detectable with these arm sizes."""
    if not n1 or not n2 or not 0 < p < 1:
        return None
    return round(float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))), 5)


def experiment_report(db: Session, experiment_id: int, maturity_days: int = DEFAULT_MATURITY_DAYS,
                      now: Optional[datetime] = None, bootstrap_reps: int = 300) -> Dict:
    e = db.get(TrafficExperiment, experiment_id)
    if e is None:
        raise ExperimentError(f"No experiment {experiment_id}")
    rows = db.query(RoutedDecision).filter(RoutedDecision.experiment_id == experiment_id).all()
    frame = pd.DataFrame([{c.name: getattr(r, c.name) for c in RoutedDecision.__table__.columns} for r in rows])
    out: Dict = {"experiment": _as_dict(e), "routed": int(len(frame))}
    if frame.empty:
        out["status"] = "no_traffic"
        return out
    frame = frame.sort_values("routed_at", kind="stable").drop_duplicates("transaction_id", keep="last")
    frame["customer_id"] = frame["customer_id"].fillna(frame["transaction_id"])

    units = frame["customer_id"] if e.unit == "customer" else frame["transaction_id"]
    per_arm = frame.assign(unit=units).groupby("arm")["unit"].nunique()
    observed = [int(per_arm.get("bti", 0)), int(per_arm.get("control", 0))]
    expected = [sum(observed) * e.bti_share, sum(observed) * (1 - e.bti_share)]
    srm_p = float(chisquare(observed, expected).pvalue) if sum(observed) and min(expected) > 0 else None
    bti_rows = frame["arm"] == "bti"
    out["split"] = {"units_per_arm": {"bti": observed[0], "control": observed[1]}, "configured_bti_share": e.bti_share,
                    "srm_p_value": round(srm_p, 6) if srm_p is not None else None,
                    "sample_ratio_mismatch": bool(srm_p is not None and srm_p < 0.001)}
    out["fallback"] = {"bti_arm_transactions": int(bti_rows.sum()),
                       "fallbacks": frame.loc[bti_rows, "fallback_reason"].value_counts().to_dict(),
                       "fallback_rate": round(float(frame.loc[bti_rows, "fallback_reason"].notna().mean()), 5)
                       if bti_rows.any() else None}

    labels = latest_labels(db, frame["transaction_id"].tolist())
    frame = frame.merge(labels[["transaction_id", "label"]] if not labels.empty
                        else pd.DataFrame(columns=["transaction_id", "label"]), on="transaction_id", how="left")
    cutoff = (now or datetime.utcnow()) - pd.Timedelta(days=maturity_days)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame.loc[frame["label"].isna() & (pd.to_datetime(frame["routed_at"]) < cutoff), "label"] = 0.0
    mature = frame[frame["label"].notna()]
    out["labels"] = {"mature": int(len(mature)), "immature": int(len(frame) - len(mature)),
                     "fraud": int((mature["label"] == 1).sum())}
    arms = {arm: mature[mature["arm"] == arm] for arm in ("bti", "control")}
    if any(a.empty or (a["label"] == 1).sum() < 10 for a in arms.values()):
        out["status"] = "insufficient_labels"
        out["detail"] = "Each arm needs at least 10 matured frauds before outcomes are compared."
        return out
    metrics = {arm: _arm_metrics(g) for arm, g in arms.items()}
    ci = _cluster_bootstrap(mature[mature["arm"].isin(["bti", "control"])], bootstrap_reps, seed=11)
    out["arms"] = metrics
    out["bti_minus_control_ci95"] = ci
    p_fraud = float(mature["label"].mean())
    out["minimum_detectable_tdr_difference"] = minimum_detectable_difference(
        0.5, metrics["bti"]["fraud"], metrics["control"]["fraud"])
    out["base_fraud_rate"] = round(p_fraud, 5)
    loss_ci = ci.get("fraud_loss_bps")
    fd_ci = ci.get("false_decline_rate")
    if out["split"]["sample_ratio_mismatch"]:
        verdict = "invalid_split"
    elif loss_ci and loss_ci[1] < 0 and not (fd_ci and fd_ci[0] > 0):
        verdict = "bti_lower_loss"
    elif loss_ci and loss_ci[0] > 0:
        verdict = "bti_higher_loss"
    else:
        verdict = "no_significant_difference"
    out["status"], out["verdict"] = "ok", verdict
    return out
