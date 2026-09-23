"""
Live fraud-operations KPIs, champion/challenger comparison and live drift —
all computed from the score log and confirmed labels.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy.orm import Session

from bti.database.models import ScoreLog
from bti.governance.monitoring import drift_report
from bti.modeling import registry
from bti.operations.feedback import DEFAULT_MATURITY_DAYS, labelled_scores

INTERVENTIONS = ("STEP_UP", "REVIEW", "DECLINE")


def _auc(y: np.ndarray, p: np.ndarray) -> Dict:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4)}


def operational_kpis(frame: pd.DataFrame) -> Dict:
    if frame.empty:
        return {"volume": 0, "detail": "No scored transactions in the window."}
    amt = pd.to_numeric(frame["amount_usd"], errors="coerce").fillna(0).to_numpy()
    decision = frame["decision"].astype(str).to_numpy()
    intervened = np.isin(decision, INTERVENTIONS)
    out = {
        "volume": int(len(frame)),
        "value_usd": round(float(amt.sum()), 2),
        "action_rates": {a: round(float((decision == a).mean()), 5) for a in ("APPROVE",) + INTERVENTIONS},
        "intervention_rate": round(float(intervened.mean()), 5),
    }
    known = frame["label"].notna().to_numpy()
    out["labelled"] = int(known.sum())
    if not known.any():
        out["detail"] = "No transactions are past the label maturity window yet; outcome KPIs pending."
        return out
    y = frame.loc[known, "label"].astype(int).to_numpy()
    a, d, iv = amt[known], decision[known], intervened[known]
    fraud, genuine = y == 1, y == 0
    fraud_value = a[fraud].sum()
    out.update({
        "fraud_count": int(fraud.sum()),
        "tdr": round(float(iv[fraud].mean()), 4) if fraud.any() else None,
        "vdr": round(float(a[fraud & iv].sum() / fraud_value), 4) if fraud_value else None,
        "hit_rate": round(float(fraud[iv].mean()), 4) if iv.any() else None,
        "false_positive_ratio": round(float((genuine & iv).sum() / max((fraud & iv).sum(), 1)), 2),
        "false_decline_rate": round(float(((d == "DECLINE") & genuine).sum() / max(genuine.sum(), 1)), 5),
        "fraud_loss_bps": round(float(a[fraud & ~iv].sum() / max(a.sum(), 1e-9) * 10_000), 3),
        **_auc(y, frame.loc[known, "fraud_probability"].to_numpy(float)),
    })
    return out


def kpi_report(db: Session, start: Optional[datetime] = None, end: Optional[datetime] = None,
               maturity_days: int = DEFAULT_MATURITY_DAYS) -> Dict:
    frame = labelled_scores(db, start, end, maturity_days, shadow=False)
    report = {"window": {"from": start, "to": end}, "maturity_days": maturity_days,
              "overall": operational_kpis(frame)}
    if not frame.empty:
        report["by_jurisdiction"] = {j: operational_kpis(g) for j, g in frame.groupby(frame["jurisdiction"].fillna("??"))}
    return report


def champion_challenger(db: Session, start: Optional[datetime] = None, end: Optional[datetime] = None,
                        maturity_days: int = DEFAULT_MATURITY_DAYS) -> Dict:
    frame = labelled_scores(db, start, end, maturity_days, shadow=None)
    if frame.empty:
        return {"status": "no_data"}
    live = frame[~frame["is_shadow"].astype(bool)]
    shadow = frame[frame["is_shadow"].astype(bool)]
    if shadow.empty:
        return {"status": "no_shadow_scores",
                "detail": "No challenger is scoring in shadow mode for this window."}
    paired = live.merge(shadow, on="transaction_id", suffixes=("_live", "_shadow"))
    if paired.empty:
        return {"status": "no_overlap"}
    out = {
        "status": "ok",
        "paired_transactions": int(len(paired)),
        "live_model": paired["model_id_live"].iloc[-1],
        "shadow_model": paired["model_id_shadow"].iloc[-1],
        "decision_agreement": round(float((paired["decision_live"] == paired["decision_shadow"]).mean()), 4),
        "probability_correlation": round(float(np.corrcoef(paired["fraud_probability_live"],
                                                           paired["fraud_probability_shadow"])[0, 1]), 4)
        if len(paired) > 2 else None,
        "mean_abs_probability_gap": round(float((paired["fraud_probability_live"]
                                                 - paired["fraud_probability_shadow"]).abs().mean()), 5),
    }
    known = paired["label_live"].notna()
    if known.any():
        y = paired.loc[known, "label_live"].astype(int).to_numpy()
        live_perf = _auc(y, paired.loc[known, "fraud_probability_live"].to_numpy(float))
        shadow_perf = _auc(y, paired.loc[known, "fraud_probability_shadow"].to_numpy(float))
        out.update({"labelled_pairs": int(known.sum()), "live": live_perf, "shadow": shadow_perf})
        if live_perf["pr_auc"] is not None:
            delta = shadow_perf["pr_auc"] - live_perf["pr_auc"]
            out["pr_auc_delta"] = round(delta, 4)
            out["recommendation"] = ("Challenger outperforms on matured labels — prepare promotion evidence"
                                     if delta > 0.01 else "Keep current champion")
    return out


def live_drift(db: Session, start: Optional[datetime] = None, end: Optional[datetime] = None,
               model_id: Optional[str] = None, shadow: bool = False, limit: int = 50_000) -> Dict:
    model_id = model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
    if not model_id:
        return {"status": "no_model"}
    card = registry.load_card(model_id)
    q = db.query(ScoreLog.features, ScoreLog.fraud_probability).filter(ScoreLog.model_id == model_id,
                                                                      ScoreLog.is_shadow == shadow)
    if start:
        q = q.filter(ScoreLog.scored_at >= start)
    if end:
        q = q.filter(ScoreLog.scored_at < end)
    rows = q.order_by(ScoreLog.scored_at.desc()).limit(limit).all()
    feats = pd.DataFrame([r[0] or {} for r in rows])
    scores = np.array([r[1] for r in rows], dtype=float)
    return {"model_id": model_id, **drift_report(card["monitoring"]["baseline"], feats, scores)}


def scoring_latency(db: Session, hours: int, sla_ms: float) -> Dict:
    """Model scoring latency percentiles by model over the last `hours`, from the score log."""
    since = datetime.utcnow() - timedelta(hours=hours)
    rows = (db.query(ScoreLog.model_id, ScoreLog.is_shadow, ScoreLog.latency_ms)
            .filter(ScoreLog.scored_at >= since).all())
    frame = pd.DataFrame(rows, columns=["model_id", "is_shadow", "latency_ms"])
    out = {"window_hours": hours, "sla_ms": sla_ms, "models": []}
    for (model_id, shadow), g in frame.groupby(["model_id", "is_shadow"]):
        lat = g["latency_ms"].astype(float).to_numpy()
        out["models"].append({
            "model_id": model_id, "shadow": bool(shadow), "scored": int(len(lat)),
            "p50_ms": round(float(np.percentile(lat, 50)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "p99_ms": round(float(np.percentile(lat, 99)), 1),
            "share_over_sla": round(float((lat > sla_ms).mean()), 4),
        })
    return out
