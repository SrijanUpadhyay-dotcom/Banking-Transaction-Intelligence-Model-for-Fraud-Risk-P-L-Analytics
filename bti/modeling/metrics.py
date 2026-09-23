"""
Validation metrics for fraud models — statistical and operational.

Statistical: ROC-AUC, PR-AUC, KS, Gini, Brier, expected calibration error.
Operational (what fraud operations report to the business):
  alert rate, precision (hit rate), transaction detection rate (TDR),
  value detection rate (VDR — share of fraud *dollars* intercepted),
  account detection rate (ADR), and false-positive ratio (legit alerts per
  fraud alert, the industry "N:1" figure).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    n_pos, n_neg = max(ys.sum(), 1), max(len(ys) - ys.sum(), 1)
    return float(np.max(np.abs(np.cumsum(ys) / n_pos - np.cumsum(1 - ys) / n_neg)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-frequency bins — stable under heavy class imbalance."""
    order = np.argsort(p, kind="stable")
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            ece += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(ece)


def classification_metrics(y, p) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "n_fraud": int(y.sum()), "base_rate": float(y.mean()) if len(y) else 0.0}
    auc = roc_auc_score(y, p)
    return {
        "n": int(len(y)),
        "n_fraud": int(y.sum()),
        "base_rate": round(float(y.mean()), 5),
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y, p), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "log_loss": round(float(log_loss(y, p)), 5),
        "ece": round(expected_calibration_error(y, p), 5),
    }


def operating_point(y, p, threshold: float, amounts=None, accounts=None) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    flag = p >= threshold
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    out = {
        "threshold": round(float(threshold), 6),
        "alerts": int(flag.sum()),
        "alert_rate": round(float(flag.mean()), 5) if len(p) else 0.0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "tdr": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "false_positive_ratio": round(fp / tp, 2) if tp else None,
        "legit_customers_impacted_per_10k": round(float(fp / max((y == 0).sum(), 1) * 10_000), 1),
    }
    if amounts is not None:
        a = np.nan_to_num(np.asarray(amounts, dtype=float))
        fraud_value = a[y == 1].sum()
        out["vdr"] = round(float(a[flag & (y == 1)].sum() / fraud_value), 4) if fraud_value else 0.0
        out["fraud_value_intercepted_usd"] = round(float(a[flag & (y == 1)].sum()), 2)
        out["fraud_value_missed_usd"] = round(float(a[~flag & (y == 1)].sum()), 2)
    if accounts is not None:
        acc = pd.Series(np.asarray(accounts))
        fraud_accounts = acc[y == 1].unique()
        caught_accounts = acc[flag & (y == 1)].unique()
        out["adr"] = round(len(caught_accounts) / len(fraud_accounts), 4) if len(fraud_accounts) else 0.0
    return out


def threshold_for_alert_rate(p, alert_rate: float) -> float:
    p = np.asarray(p, dtype=float)
    if alert_rate <= 0:
        return float(np.inf)
    return float(np.quantile(p, 1 - alert_rate, method="higher"))


def alert_budget_table(y, p, budgets=(0.005, 0.01, 0.02, 0.05, 0.10), amounts=None, accounts=None) -> List[dict]:
    """Performance at fixed investigation capacities — how fraud ops actually set thresholds."""
    return [
        {"alert_budget": b, **operating_point(y, p, threshold_for_alert_rate(p, b), amounts, accounts)}
        for b in budgets
    ]


def lift_table(y, p, amounts=None, n_bins: int = 10) -> List[dict]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    a = np.nan_to_num(np.asarray(amounts, dtype=float)) if amounts is not None else None
    order = np.argsort(-p, kind="stable")
    n, n_fraud = len(y), max(int(y.sum()), 1)
    fraud_value = a[y == 1].sum() if a is not None else 0.0
    rows, cum_fraud, cum_value, cum_n = [], 0, 0.0, 0
    for i, idx in enumerate(np.array_split(order, n_bins), start=1):
        nf = int(y[idx].sum())
        cum_fraud += nf
        cum_n += len(idx)
        captured = cum_fraud / n_fraud
        pop = cum_n / n
        row = {
            "decile": i,
            "score_min": round(float(p[idx].min()), 5),
            "score_max": round(float(p[idx].max()), 5),
            "n": int(len(idx)),
            "n_fraud": nf,
            "fraud_rate": round(nf / len(idx), 4) if len(idx) else 0.0,
            "cum_fraud_captured": round(captured, 4),
            "cum_lift": round(captured / pop, 3) if pop else 0.0,
            "decile_lift": round((nf / len(idx)) / (n_fraud / n), 3) if len(idx) else 0.0,
        }
        if a is not None:
            cum_value += a[idx][y[idx] == 1].sum()
            row["cum_value_captured"] = round(float(cum_value / fraud_value), 4) if fraud_value else 0.0
        rows.append(row)
    return rows


def segment_performance(y, p, segments, threshold: float, min_n: int = 200) -> List[dict]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    seg = pd.Series(np.asarray(segments)).astype("string").fillna("(missing)")
    rows = []
    for name in sorted(seg.unique()):
        m = (seg == name).to_numpy()
        if m.sum() < min_n:
            continue
        ys, ps = y[m], p[m]
        flag = ps >= threshold
        legit, fraud = ys == 0, ys == 1
        rows.append({
            "segment": str(name),
            "n": int(m.sum()),
            "fraud_rate": round(float(ys.mean()), 4),
            "roc_auc": round(float(roc_auc_score(ys, ps)), 4) if 0 < ys.sum() < len(ys) else None,
            "alert_rate": round(float(flag.mean()), 4),
            "tdr": round(float(flag[fraud].mean()), 4) if fraud.any() else None,
            "fpr": round(float(flag[legit].mean()), 5) if legit.any() else None,
        })
    return rows
