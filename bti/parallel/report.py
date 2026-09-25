"""
Parallel-run report: BTI against the incumbent on the same transactions.

The headline comparison is at *equal intervention rate*. BTI's riskiest
transactions are taken in the same number as the incumbent intervened on, and
the two sets are compared on fraud caught (count and value) and on genuine
customers disturbed. That removes the easy win of simply intervening more.

- **Paired test.** McNemar's test on the frauds one system caught and the other
  missed.
- **Confidence intervals.** Bootstrap intervals on the detection-rate
  differences.
- **Which labels count.** Only labels past the maturity window count, so a
  young window never looks clean.

Weekly, three views are produced:

- the last 7 days (operations, agreement, latency, feed coverage)
- the cohort that matured this week (outcomes)
- everything to date (outcomes, cumulative)

A regression alerts through the monitoring channels.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import AuditLog, ScoreLog
from bti.logging_config import get_logger
from bti.operations.feedback import DEFAULT_MATURITY_DAYS, labelled_scores
from bti.parallel.incumbent import ACTIONS, latest_incumbent

log = get_logger("parallel.report")

EVENT_TYPE = "PARALLEL_RUN_REPORT"
INTERVENTIONS = ("STEP_UP", "REVIEW", "DECLINE")
MIN_LABELLED_FRAUD = 30
ALPHA = 0.05


def _rate(num, den) -> Optional[float]:
    return round(float(num) / float(den), 5) if den else None


def _percentiles(values) -> Optional[Dict]:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not len(v):
        return None
    return {"n": int(len(v)), **{f"p{q}_ms": round(float(np.percentile(v, q)), 1) for q in (50, 95, 99)}}


def _system_outcomes(y, value, intervened, declined) -> Dict:
    fraud, genuine = y == 1, y == 0
    fraud_value = value[fraud].sum()
    return {
        "interventions": int(intervened.sum()),
        "intervention_rate": _rate(intervened.sum(), len(y)),
        "fraud_caught": int((fraud & intervened).sum()),
        "tdr": _rate((fraud & intervened).sum(), fraud.sum()),
        "vdr": round(float(value[fraud & intervened].sum() / fraud_value), 4) if fraud_value else None,
        "fraud_value_caught_usd": round(float(value[fraud & intervened].sum()), 2),
        "fraud_value_missed_usd": round(float(value[fraud & ~intervened].sum()), 2),
        "genuine_intervened": int((genuine & intervened).sum()),
        "genuine_intervention_rate": _rate((genuine & intervened).sum(), genuine.sum()),
        "false_decline_rate": _rate((genuine & declined).sum(), genuine.sum()),
        "hit_rate": _rate((fraud & intervened).sum(), intervened.sum()),
        "false_positive_ratio": round(float((genuine & intervened).sum() / max((fraud & intervened).sum(), 1)), 2),
    }


def _top_k(prob: np.ndarray, k: int) -> np.ndarray:
    chosen = np.zeros(len(prob), dtype=bool)
    if k > 0:
        chosen[np.argsort(-prob, kind="stable")[:k]] = True
    return chosen


def _bootstrap(y, value, prob, inc_iv, reps: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    tdr_d, vdr_d = [], []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        yy, vv, pp, ii = y[i], value[i], prob[i], inc_iv[i]
        fraud = yy == 1
        if not fraud.any():
            continue
        bti = _top_k(pp, int(ii.sum()))
        tdr_d.append((fraud & bti).sum() / fraud.sum() - (fraud & ii).sum() / fraud.sum())
        fv = vv[fraud].sum()
        if fv:
            vdr_d.append((vv[fraud & bti].sum() - vv[fraud & ii].sum()) / fv)

    def ci(d):
        return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)] if d else None
    return {"reps": reps, "tdr_difference_ci95": ci(tdr_d), "vdr_difference_ci95": ci(vdr_d)}


def compare_systems(frame: pd.DataFrame, incumbent_name: str = "SAS", bootstrap_reps: int = 500,
                    seed: int = 7) -> Dict:
    """
    `frame`: one row per transaction with bti_probability, bti_decision, incumbent_decision, amount_usd and label
    (1 fraud / 0 genuine / NaN not yet mature); optional bti_latency_ms, incumbent_latency_ms.
    """
    paired = frame[frame["bti_decision"].notna() & frame["incumbent_decision"].notna()].copy()
    out: Dict = {"incumbent": incumbent_name, "paired_transactions": int(len(paired))}
    if paired.empty:
        out["status"] = "no_paired_transactions"
        return out

    bd, idc = paired["bti_decision"].astype(str), paired["incumbent_decision"].astype(str)
    matrix = pd.crosstab(bd, idc).reindex(index=list(ACTIONS), columns=list(ACTIONS), fill_value=0)
    bti_iv, inc_iv = bd.isin(INTERVENTIONS).to_numpy(), idc.isin(INTERVENTIONS).to_numpy()
    out["operational"] = {
        "decision_agreement": round(float((bd == idc).mean()), 4),
        "intervention_agreement": round(float((bti_iv == inc_iv).mean()), 4),
        "agreement_matrix": {"rows": "BTI", "columns": incumbent_name,
                             "counts": {b: {i: int(matrix.loc[b, i]) for i in ACTIONS} for b in ACTIONS}},
        "bti_action_rates": {a: round(float((bd == a).mean()), 5) for a in ACTIONS},
        "incumbent_action_rates": {a: round(float((idc == a).mean()), 5) for a in ACTIONS},
    }
    out["latency"] = {"bti": _percentiles(paired.get("bti_latency_ms", [])),
                      "incumbent": _percentiles(paired.get("incumbent_latency_ms", []))}

    known = paired["label"].notna().to_numpy()
    y = paired.loc[known, "label"].astype(int).to_numpy()
    out["labels"] = {"mature": int(known.sum()), "immature": int((~known).sum()),
                     "fraud": int((y == 1).sum()), "genuine": int((y == 0).sum())}
    if (y == 1).sum() < MIN_LABELLED_FRAUD:
        out["status"] = "insufficient_labels"
        out["detail"] = (f"{int((y == 1).sum())} matured frauds among paired transactions; at least "
                         f"{MIN_LABELLED_FRAUD} are needed before outcomes are compared.")
        return out

    k = paired.loc[known]
    value = pd.to_numeric(k["amount_usd"], errors="coerce").fillna(0).to_numpy(float)
    prob = k["bti_probability"].to_numpy(float)
    b_iv, i_iv = bti_iv[known], inc_iv[known]
    b_dec = (k["bti_decision"] == "DECLINE").to_numpy()
    i_dec = (k["incumbent_decision"] == "DECLINE").to_numpy()
    equal = _top_k(prob, int(i_iv.sum()))
    fraud, genuine = y == 1, y == 0

    both, bti_only = int((fraud & equal & i_iv).sum()), int((fraud & equal & ~i_iv).sum())
    inc_only, neither = int((fraud & ~equal & i_iv).sum()), int((fraud & ~equal & ~i_iv).sum())
    discordant = bti_only + inc_only
    p_mcnemar = float(binomtest(min(bti_only, inc_only), discordant, 0.5).pvalue) if discordant else 1.0

    inc_out = _system_outcomes(y, value, i_iv, i_dec)
    equal_out = _system_outcomes(y, value, equal, np.zeros_like(equal))
    equal_out.pop("false_decline_rate")
    boot = _bootstrap(y, value, prob, i_iv, bootstrap_reps, seed)
    tdr_diff = (equal_out["tdr"] or 0) - (inc_out["tdr"] or 0)
    vdr_diff = (equal_out["vdr"] or 0) - (inc_out["vdr"] or 0)

    ci = boot["tdr_difference_ci95"]
    if p_mcnemar < ALPHA and bti_only > inc_only and ci and ci[0] > 0:
        verdict = "bti_detects_more"
    elif p_mcnemar < ALPHA and inc_only > bti_only and ci and ci[1] < 0:
        verdict = "incumbent_detects_more"
    else:
        verdict = "no_significant_difference"

    out.update({
        "status": "ok",
        "verdict": verdict,
        "at_equal_intervention_rate": {
            "definition": f"BTI's {int(i_iv.sum())} highest-probability transactions vs the {int(i_iv.sum())} "
                          f"{incumbent_name} intervened on (matured, paired)",
            "incumbent": {k_: inc_out[k_] for k_ in ("interventions", "tdr", "vdr", "fraud_value_caught_usd",
                                                     "genuine_intervened", "hit_rate")},
            "bti": {k_: equal_out[k_] for k_ in ("interventions", "tdr", "vdr", "fraud_value_caught_usd",
                                                 "genuine_intervened", "hit_rate")},
            "tdr_difference": round(tdr_diff, 4),
            "vdr_difference": round(vdr_diff, 4),
            "incremental_fraud_value_usd": round(equal_out["fraud_value_caught_usd"]
                                                 - inc_out["fraud_value_caught_usd"], 2),
            **boot,
        },
        "fraud_overlap": {
            "caught_by_both": both, "bti_only": bti_only, f"{incumbent_name.lower()}_only": inc_only,
            "missed_by_both": neither,
            "bti_only_value_usd": round(float(value[fraud & equal & ~i_iv].sum()), 2),
            f"{incumbent_name.lower()}_only_value_usd": round(float(value[fraud & ~equal & i_iv].sum()), 2),
            "mcnemar_p_value": round(p_mcnemar, 5),
        },
        "customer_friction": {
            "genuine_disturbed_by_bti_only": int((genuine & equal & ~i_iv).sum()),
            f"genuine_disturbed_by_{incumbent_name.lower()}_only": int((genuine & ~equal & i_iv).sum()),
        },
        "at_actual_decisions": {"bti": _system_outcomes(y, value, b_iv, b_dec), "incumbent": inc_out},
    })
    return out


# ── Database side ────────────────────────────────────────────────────────────

def build_frame(db: Session, start: Optional[datetime] = None, end: Optional[datetime] = None,
                maturity_days: int = DEFAULT_MATURITY_DAYS, now: Optional[datetime] = None) -> pd.DataFrame:
    """Latest live BTI decision per transaction in the window, joined to the incumbent's and to outcomes."""
    bti = labelled_scores(db, start, end, maturity_days, shadow=False, now=now)
    if bti.empty:
        return pd.DataFrame(columns=["transaction_id", "bti_probability", "bti_decision", "incumbent_decision",
                                     "amount_usd", "label"])
    bti = bti.sort_values("scored_at", kind="stable").drop_duplicates("transaction_id", keep="last")
    lat = db.query(ScoreLog.transaction_id, ScoreLog.latency_ms).filter(ScoreLog.is_shadow.is_(False))
    if start:
        lat = lat.filter(ScoreLog.scored_at >= start)
    if end:
        lat = lat.filter(ScoreLog.scored_at < end)
    latency = pd.DataFrame(lat.all(), columns=["transaction_id", "bti_latency_ms"]).drop_duplicates(
        "transaction_id", keep="last")
    slack = timedelta(days=2)
    inc = latest_incumbent(db, start=start - slack if start else None, end=end + slack if end else None)
    inc = inc.rename(columns={"decision": "incumbent_decision", "score": "incumbent_score",
                              "latency_ms": "incumbent_latency_ms"})
    frame = (bti.rename(columns={"fraud_probability": "bti_probability", "decision": "bti_decision"})
                .merge(latency, on="transaction_id", how="left"))
    if inc.empty:
        frame["incumbent_decision"] = None
        return frame
    return frame.merge(inc[["transaction_id", "incumbent_decision", "incumbent_score", "incumbent_latency_ms",
                            "system"]], on="transaction_id", how="left")


def parallel_run_report(db: Session, start: Optional[datetime] = None, end: Optional[datetime] = None,
                        maturity_days: int = DEFAULT_MATURITY_DAYS, now: Optional[datetime] = None,
                        bootstrap_reps: int = 500) -> Dict:
    frame = build_frame(db, start, end, maturity_days, now)
    system = get_settings().incumbent_system
    report = compare_systems(frame, system, bootstrap_reps=bootstrap_reps)
    n_bti = int(frame["bti_decision"].notna().sum()) if not frame.empty else 0
    report["window"] = {"from": start.isoformat() if start else None, "to": end.isoformat() if end else None}
    report["maturity_days"] = maturity_days
    report["coverage"] = {"bti_decisions": n_bti, "with_incumbent_decision": report["paired_transactions"],
                          "pairing_rate": _rate(report["paired_transactions"], n_bti)}
    return report


def run_weekly_report(db: Session, now: Optional[datetime] = None, maturity_days: int = DEFAULT_MATURITY_DAYS,
                      notify: bool = True) -> Dict:
    now = now or datetime.utcnow()
    week = timedelta(days=7)
    matured_end = now - timedelta(days=maturity_days)
    views = {
        "last_7_days": parallel_run_report(db, now - week, now, maturity_days, now),
        "matured_cohort": parallel_run_report(db, matured_end - week, matured_end, maturity_days, now),
        "to_date": parallel_run_report(db, None, now, maturity_days, now),
    }
    issues: List[Dict] = []
    cumulative = views["to_date"]
    if cumulative.get("verdict") == "incumbent_detects_more":
        issues.append({"severity": "CRITICAL", "issue": "Incumbent detects significantly more fraud at equal "
                                                         "intervention rate on matured labels"})
    pairing = views["last_7_days"]["coverage"]["pairing_rate"]
    if pairing is not None and pairing < 0.95:
        issues.append({"severity": "WARNING", "issue": f"Only {pairing:.1%} of BTI decisions in the last 7 days have "
                                                       f"an incumbent decision; check the feed"})
    lat = (views["last_7_days"].get("latency") or {}).get("bti")
    sla = get_settings().scoring_latency_sla_ms
    if lat and lat["p99_ms"] > sla:
        issues.append({"severity": "WARNING", "issue": f"BTI p99 latency {lat['p99_ms']} ms exceeds the {sla} ms SLA"})
    result = {"generated_at": now.isoformat(), "issues": issues, "views": views, "alert": None}
    if notify and issues:
        from bti.alerts import AlertDispatcher
        worst = "CRITICAL" if any(i["severity"] == "CRITICAL" for i in issues) else "WARNING"
        result["alert"] = AlertDispatcher().dispatch_event("PARALLEL_RUN", worst, {"issues": issues})
    db.add(AuditLog(ts=now, event_type=EVENT_TYPE, payload=_jsonable(result)))
    db.commit()
    log.info("Parallel-run report generated", extra={"issues": len(issues)})
    return result


def report_history(db: Session, limit: int = 12) -> List[Dict]:
    rows = (db.query(AuditLog.payload).filter(AuditLog.event_type == EVENT_TYPE)
            .order_by(AuditLog.ts.desc(), AuditLog.id.desc()).limit(limit).all())
    return [r[0] for r in rows]


def _jsonable(obj):
    import json
    return json.loads(json.dumps(obj, default=str))
