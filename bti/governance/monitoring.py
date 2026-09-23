"""
Population stability monitoring (PSI for the score, CSI for each feature).

Thresholds follow common bank practice:
  PSI < 0.10  stable
  0.10–0.25   moderate shift — investigate
  > 0.25      significant shift — escalate; recalibrate or retrain
The baseline is captured at training time and stored in the model card, so
live traffic is always compared with the population the model was validated on.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25
_EPS = 1e-4
_MISSING = "__missing__"
_OTHER = "__other__"


def psi_status(value: float) -> str:
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "investigate"
    return "escalate"


def psi(expected: List[float], actual: List[float]) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_edges(x: pd.Series, n_bins: int) -> List[float]:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return []
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return [float(e) for e in edges]


def _numeric_distribution(x: pd.Series, edges: List[float]) -> List[float]:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(float)
    missing = np.isnan(vals)
    bins = np.searchsorted(np.asarray(edges), vals[~missing], side="right")
    counts = np.bincount(bins, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, missing.sum())
    return (counts / max(len(vals), 1)).tolist()


def _categorical_distribution(x: pd.Series, categories: List[str]) -> List[float]:
    s = x.astype("string").fillna(_MISSING)
    s = s.where(s.isin(categories + [_MISSING]), _OTHER)
    counts = s.value_counts()
    keys = categories + [_OTHER, _MISSING]
    return [float(counts.get(k, 0)) / max(len(s), 1) for k in keys]


def build_baseline(features: pd.DataFrame, scores: np.ndarray, categorical: List[str],
                   n_bins: int = 10) -> Dict:
    baseline = {"n": int(len(features)), "features": {}, "score": {}}
    for col in features.columns:
        if col in categorical:
            cats = sorted(features[col].dropna().astype(str).unique().tolist())
            baseline["features"][col] = {"type": "categorical", "categories": cats,
                                         "dist": _categorical_distribution(features[col], cats)}
        else:
            edges = _numeric_edges(features[col], n_bins)
            baseline["features"][col] = {"type": "numeric", "edges": edges,
                                         "dist": _numeric_distribution(features[col], edges)}
    s = pd.Series(np.asarray(scores, dtype=float))
    edges = _numeric_edges(s, n_bins)
    baseline["score"] = {"edges": edges, "dist": _numeric_distribution(s, edges)}
    return baseline


def drift_report(baseline: Dict, features: pd.DataFrame, scores: Optional[np.ndarray] = None,
                 min_n: int = 100) -> Dict:
    n = int(len(features))
    report: Dict = {"n": n, "baseline_n": baseline.get("n"), "features": []}
    if n < min_n:
        report["status"] = "insufficient_data"
        report["detail"] = f"Need at least {min_n} scored transactions for a stable PSI; got {n}."
        return report

    worst = "stable"
    rank = {"stable": 0, "investigate": 1, "escalate": 2}
    for col, spec in baseline["features"].items():
        if col not in features.columns:
            continue
        if spec["type"] == "categorical":
            actual = _categorical_distribution(features[col], spec["categories"])
        else:
            actual = _numeric_distribution(features[col], spec["edges"])
        value = psi(spec["dist"], actual)
        status = psi_status(value)
        worst = max(worst, status, key=rank.get)
        report["features"].append({"feature": col, "csi": round(value, 4), "status": status})
    report["features"].sort(key=lambda r: -r["csi"])

    if scores is not None and baseline.get("score", {}).get("edges") is not None:
        value = psi(baseline["score"]["dist"],
                    _numeric_distribution(pd.Series(np.asarray(scores, dtype=float)), baseline["score"]["edges"]))
        report["score_psi"] = round(value, 4)
        report["score_status"] = psi_status(value)
        worst = max(worst, report["score_status"], key=rank.get)

    report["status"] = worst
    report["thresholds"] = {"stable_below": PSI_STABLE, "escalate_above": PSI_SIGNIFICANT}
    return report
