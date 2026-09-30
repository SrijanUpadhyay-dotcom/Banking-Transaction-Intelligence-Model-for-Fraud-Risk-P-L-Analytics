"""
Outcomes analysis on matured labels (SR 11-7 ongoing monitoring; PRA SS1/23
Principle 5).

Each quarter, the live scores whose outcomes have matured are compared with
what the model promised at development. Only transactions past the label
maturity window count, and unreported ones by then are treated as genuine.

**Discrimination.** ROC-AUC, PR-AUC and KS against the development
out-of-time figures.

**Calibration.**
- A decile table of predicted against observed fraud rates.
- ECE.
- Calibration-in-the-large: mean predicted rate against observed.

**Fairness.** The same pooled, corrected false-positive-rate test as
development, plus a test of the *decisions*: genuine-customer intervention
rates standardised for payment amount, so a group is flagged only when it is
intervened on more than its amounts explain.
- The two halves of the quarter are the two windows, and a finding must
  recur in both.
- It runs on the monitoring attributes recorded at scoring (segment, age band,
  country). These are never model inputs.
- Groups on the model's development watchlist are re-tested by name.

**Findings.** Degradation beyond tolerance raises a finding in the tracker
(source "monitoring"), unless an open finding with the same title exists. The
result goes to the hash-chained audit log, and issues alert.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from bti.database.models import AuditLog, ScoreLog, ValidationFinding
from bti.governance.fairness import fairness_assessment
from bti.logging_config import get_logger
from bti.modeling import registry
from bti.modeling.metrics import classification_metrics, threshold_for_alert_rate
from bti.operations.feedback import DEFAULT_MATURITY_DAYS, labelled_scores

log = get_logger("governance.outcomes")

EVENT_TYPE = "OUTCOMES_ANALYSIS"
MIN_FRAUD = 30
TOLERANCE = {"pr_auc_drop": 0.05, "roc_auc_drop": 0.03, "max_ece": 0.02, "calibration_in_the_large": 0.20}
ATTRIBUTES = ("customer_segment", "customer_age_band", "country")


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> List[Dict]:
    order = np.argsort(p, kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, bins), 1):
        if len(idx) == 0:
            continue
        rows.append({"decile": i, "n": int(len(idx)), "mean_predicted": round(float(p[idx].mean()), 5),
                     "observed_rate": round(float(y[idx].mean()), 5)})
    return rows


def _frame(db: Session, start, end, maturity_days, now) -> pd.DataFrame:
    scores = labelled_scores(db, start, end, maturity_days, shadow=False, now=now)
    if scores.empty:
        return scores
    scores = scores.sort_values("scored_at", kind="stable").drop_duplicates("transaction_id", keep="last")
    q = db.query(ScoreLog.transaction_id, ScoreLog.monitoring_attributes, ScoreLog.scored_at).filter(
        ScoreLog.is_shadow.is_(False))
    if start:
        q = q.filter(ScoreLog.scored_at >= start)
    if end:
        q = q.filter(ScoreLog.scored_at < end)
    attrs = pd.DataFrame(q.all(), columns=["transaction_id", "monitoring_attributes", "scored_at"])
    attrs = attrs.sort_values("scored_at", kind="stable").drop_duplicates("transaction_id", keep="last")
    expanded = pd.DataFrame([a or {} for a in attrs["monitoring_attributes"]], index=attrs.index)
    for c in ATTRIBUTES:
        attrs[c] = expanded[c] if c in expanded else None
    return scores.merge(attrs[["transaction_id", *ATTRIBUTES]], on="transaction_id", how="left")


def _analyse_model(model_id: str, g: pd.DataFrame) -> Dict:
    known = g["label"].notna().to_numpy()
    y = g.loc[known, "label"].astype(int).to_numpy()
    p = g.loc[known, "fraud_probability"].astype(float).to_numpy()
    out: Dict = {"model_id": model_id, "scored": int(len(g)), "matured": int(known.sum()), "fraud": int(y.sum())}
    if y.sum() < MIN_FRAUD or (y == 0).sum() < MIN_FRAUD:
        out["status"] = "insufficient_labels"
        out["detail"] = f"{int(y.sum())} matured frauds; at least {MIN_FRAUD} are needed."
        return out
    live = classification_metrics(y, p)
    card = registry.load_card(model_id)
    dev = card["performance"]["metrics"]["out_of_time"]
    citl = (p.mean() - y.mean()) / y.mean() if y.mean() else None
    flags = []
    if dev.get("pr_auc") and dev["pr_auc"] - live["pr_auc"] > TOLERANCE["pr_auc_drop"]:
        flags.append(("PR-AUC degraded on matured outcomes", "high", "performance",
                      f"Live PR-AUC {live['pr_auc']} vs development {dev['pr_auc']}"))
    if dev.get("roc_auc") and dev["roc_auc"] - live["roc_auc"] > TOLERANCE["roc_auc_drop"]:
        flags.append(("ROC-AUC degraded on matured outcomes", "medium", "performance",
                      f"Live ROC-AUC {live['roc_auc']} vs development {dev['roc_auc']}"))
    if live["ece"] > TOLERANCE["max_ece"] or (citl is not None and abs(citl) > TOLERANCE["calibration_in_the_large"]):
        flags.append(("Calibration drift on matured outcomes", "medium", "monitoring",
                      f"ECE {live['ece']} (tolerance {TOLERANCE['max_ece']}); mean predicted {p.mean():.4%} vs "
                      f"observed {y.mean():.4%} ({citl:+.0%}). Recalibrate on matured labels."))

    # Fairness: two halves of the period as the two windows
    fair = None
    gk = g.loc[known].reset_index(drop=True)
    groups = {c: gk[c] for c in ATTRIBUTES if c in gk and gk[c].notna().mean() > 0.5}
    if groups:
        order = pd.to_datetime(gk["scored_at"])
        mid = order.sort_values().iloc[len(order) // 2]
        windows = {"first_half": (order < mid).to_numpy(), "second_half": (order >= mid).to_numpy()}
        art_threshold = registry.load_artifact(model_id).get("reference_threshold")
        thresholds = {"stress_10pct_budget": threshold_for_alert_rate(p, 0.10)}
        if art_threshold is not None:
            thresholds["reference"] = float(art_threshold)
        fair = fairness_assessment(y, p, thresholds, windows, groups)
        for f in fair["findings"]:
            flags.append((f"Live fairness finding: {f['attribute']} = {f['group']}", "high", "fairness",
                          f"Pooled false-positive-rate ratio {f['pooled_fpr_ratio']} (q {f['pooled_q_value']}) at "
                          f"{f['operating_point']}, above the norm in both halves of the period."))
        dev_watch = {(w["attribute"], w["group"]) for w in card.get("fairness", {}).get("watchlist", [])}
        retest = []
        for attr, group in sorted(dev_watch):
            ratios = [g_["fpr_ratio"] for run in fair["operating_points"].values()
                      for a in run["pooled"]["attributes"] if a["attribute"] == attr
                      for g_ in a["groups"] if g_["group"] == group and g_["fpr_ratio"] is not None]
            retest.append({"attribute": attr, "group": group, "worst_pooled_ratio_live": max(ratios) if ratios else None})
        fair = {k: v for k, v in fair.items() if k != "operating_points"} | {"watchlist_retest": retest}
        from bti.governance.fairness import decision_fairness
        intervened = (gk["decision"].astype(str) != "APPROVE").to_numpy()
        decisions = decision_fairness(y, intervened, groups, gk["amount_usd"].astype(float).fillna(0).to_numpy())
        fair["decisions_amount_standardised"] = {k: decisions[k] for k in ("status", "findings", "band_rates")}
        for f in decisions["findings"]:
            flags.append((f"Live decision disparity beyond amounts: {f['attribute']} = {f['group']}", "medium",
                          "fairness", f"Genuine customers intervened {f['amount_standardised_ratio']}x what their "
                                      f"payment amounts explain (raw ratio {f['raw_ratio']}, q {f['q_value']})."))
    out.update({
        "status": "ok",
        "live": live, "development_out_of_time": {k: dev.get(k) for k in ("roc_auc", "pr_auc", "ks", "ece")},
        "calibration": {"ece": live["ece"], "mean_predicted": round(float(p.mean()), 5),
                        "observed_rate": round(float(y.mean()), 5),
                        "calibration_in_the_large": round(float(citl), 4) if citl is not None else None,
                        "table": calibration_table(y, p)},
        "fairness": fair if fair is not None else {"status": "not_run",
                                                  "detail": "Monitoring attributes were not supplied with scores"},
        "flags": [{"title": t, "severity": s, "category": c, "detail": d} for t, s, c, d in flags],
    })
    return out


def outcomes_analysis(db: Session, start: Optional[datetime] = None, end: Optional[datetime] = None,
                      maturity_days: int = DEFAULT_MATURITY_DAYS, now: Optional[datetime] = None) -> Dict:
    frame = _frame(db, start, end, maturity_days, now)
    result = {"window": {"from": start.isoformat() if start else None, "to": end.isoformat() if end else None},
              "maturity_days": maturity_days, "tolerances": TOLERANCE}
    if frame.empty:
        return {**result, "status": "no_scores", "models": []}
    result["models"] = [_analyse_model(m, g) for m, g in frame.groupby("model_id")]
    result["status"] = "ok" if any(m["status"] == "ok" for m in result["models"]) else "insufficient_labels"
    return result


def run_quarterly(db: Session, now: Optional[datetime] = None, maturity_days: int = DEFAULT_MATURITY_DAYS,
                  notify: bool = True, raised_by: str = "bti.governance.outcomes") -> Dict:
    """The quarter whose outcomes have just matured: [now − maturity − 91 days, now − maturity)."""
    now = now or datetime.utcnow()
    end = now - timedelta(days=maturity_days)
    result = outcomes_analysis(db, end - timedelta(days=91), end, maturity_days, now)
    raised = []
    for m in result.get("models", []):
        for flag in m.get("flags", []):
            title = f"{flag['title']} ({end - timedelta(days=91):%Y-%m-%d} to {end:%Y-%m-%d})"
            open_same = db.query(ValidationFinding).filter(
                ValidationFinding.model_id == m["model_id"], ValidationFinding.title == title,
                ValidationFinding.status.in_(("open", "remediating"))).first()
            if open_same is None:
                from bti.governance.validation import raise_finding
                owner = registry.load_card(m["model_id"]).get("ownership", {}).get("developer") or "model owner"
                raised.append(raise_finding(db, m["model_id"], title, flag["severity"], flag["category"],
                                            "monitoring", flag["detail"], raised_by, owner=owner)["id"])
    result["findings_raised"] = raised
    result["alert"] = None
    if notify and raised:
        from bti.alerts import AlertDispatcher
        result["alert"] = AlertDispatcher().dispatch_event(
            "OUTCOMES_ANALYSIS", "WARNING", {"findings_raised": raised,
                                             "flags": [f for m in result["models"] for f in m.get("flags", [])]})
    import json
    db.add(AuditLog(ts=now, event_type=EVENT_TYPE, payload=json.loads(json.dumps(result, default=str))))
    db.commit()
    log.info("Outcomes analysis complete", extra={"status": result["status"], "findings_raised": len(raised)})
    return result


def outcomes_history(db: Session, limit: int = 8) -> List[Dict]:
    rows = (db.query(AuditLog.payload).filter(AuditLog.event_type == EVENT_TYPE)
            .order_by(AuditLog.ts.desc(), AuditLog.id.desc()).limit(limit).all())
    return [r[0] for r in rows]
