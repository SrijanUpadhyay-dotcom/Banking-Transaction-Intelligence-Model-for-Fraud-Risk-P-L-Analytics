"""
Fairness testing for fraud decisions across protected / proxy attributes.

In fraud screening the customer harm is a *legitimate* customer being held or
declined, so the primary test is false-positive-rate parity: the share of
legitimate customers in each group who are flagged, relative to the overall
rate. A group is flagged for review only when the disparity is both material
(ratio above `ratio_threshold`) and statistically significant after
Benjamini–Hochberg correction across every group tested in the report. Without
the correction, testing ~26 groups at alpha 0.05 produces false findings as a
matter of course. `fairness_assessment` is the gate-grade test: it pools
independent out-of-sample windows and requires the disparity to recur in each.

Also reported: detection-rate parity (is fraud protection equally effective
for every group) and the adverse impact ratio on approval rates.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_RATIO_THRESHOLD = 1.25
DEFAULT_ALPHA = 0.05
DEFAULT_MIN_N = 200


def _two_proportion_p(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 1.0
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def fairness_report(
    y, flagged, groups: Dict[str, pd.Series],
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    alpha: float = DEFAULT_ALPHA,
    min_n: int = DEFAULT_MIN_N,
) -> Dict:
    y = np.asarray(y, dtype=int)
    flag = np.asarray(flagged, dtype=bool)
    legit, fraud = y == 0, y == 1
    overall_fpr = float(flag[legit].mean()) if legit.any() else 0.0
    overall_tdr = float(flag[fraud].mean()) if fraud.any() else 0.0

    attributes, candidates = [], []
    for attr, values in groups.items():
        seg = pd.Series(np.asarray(values)).astype("string").fillna("(missing)")
        rows = []
        approval = {}
        for name in sorted(seg.unique()):
            m = (seg == name).to_numpy()
            if m.sum() < min_n:
                continue
            gl, gf = m & legit, m & fraud
            fp_g, n_l = int(flag[gl].sum()), int(gl.sum())
            fp_rest, n_rest = int(flag[legit & ~m].sum()), int((legit & ~m).sum())
            fpr = fp_g / n_l if n_l else 0.0
            ratio = fpr / overall_fpr if overall_fpr else None
            p_value = _two_proportion_p(fp_g, n_l, fp_rest, n_rest)
            approval[str(name)] = 1 - float(flag[m].mean())
            row = {
                "group": str(name),
                "n": int(m.sum()),
                "legit_flagged": fp_g,
                "fpr": round(fpr, 5),
                "fpr_ratio": round(ratio, 3) if ratio is not None else None,
                "fpr_p_value": round(p_value, 4),
                "tdr": round(float(flag[gf].mean()), 4) if gf.any() else None,
            }
            row["_material"] = bool(ratio is not None and ratio > ratio_threshold and fp_g >= 5)
            candidates.append((attr, row))
            rows.append(row)
        best = max(approval.values()) if approval else 0
        for row in rows:
            row["approval_air"] = round(approval[row["group"]] / best, 4) if best else None
        attributes.append({"attribute": attr, "groups": rows})

    q_values = benjamini_hochberg([row["fpr_p_value"] for _, row in candidates])
    findings = []
    for (attr, row), q in zip(candidates, q_values):
        row["fpr_q_value"] = round(q, 4)
        row["disparity_flag"] = bool(row.pop("_material") and q < alpha)
        if row["disparity_flag"]:
            findings.append({"attribute": attr, **row})

    return {
        "method": "False-positive-rate parity, two-proportion z-test with Benjamini–Hochberg correction across "
                  "all groups; approval adverse impact ratio",
        "tests": len(candidates),
        "overall_fpr": round(overall_fpr, 5),
        "overall_tdr": round(overall_tdr, 4),
        "ratio_threshold": ratio_threshold,
        "alpha": alpha,
        "min_group_n": min_n,
        "attributes": attributes,
        "findings": findings,
        "status": "review_required" if findings else "pass",
    }


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """False-discovery-rate adjusted q-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0, 1)
    return q.tolist()


def fairness_assessment(
    y, p, thresholds: Dict[str, float], windows: Dict[str, np.ndarray], groups: Dict[str, pd.Series],
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD, alpha: float = DEFAULT_ALPHA, min_n: int = DEFAULT_MIN_N,
) -> Dict:
    """
    Gate-grade fairness assessment over two or more out-of-sample windows.

    For each operating point the windows are pooled for power and tested with
    Benjamini–Hochberg correction across groups. A group is a *finding* (fails
    the gate) only if the pooled disparity is material and significant after
    correction AND its false-positive rate is above the overall rate in every
    window separately. Real disparities have a mechanism and recur; a sampling
    artefact is confined to one window and dilutes when pooled.

    A group that is material and nominally significant (uncorrected p < alpha)
    in any single window but is not a finding goes on the *watchlist*: reported,
    not gating, and re-tested by production monitoring.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    masks = {w: np.asarray(m, dtype=bool) for w, m in windows.items()}
    pooled = np.logical_or.reduce(list(masks.values()))
    columns = {a: pd.Series(np.asarray(v)) for a, v in groups.items()}

    def report(mask, cut):
        return fairness_report(y[mask], p[mask] >= cut, {a: v[mask] for a, v in columns.items()},
                               ratio_threshold=ratio_threshold, alpha=alpha, min_n=min_n)

    def index(r):
        return {(a["attribute"], g["group"]): g for a in r["attributes"] for g in a["groups"]}

    runs, findings, watchlist = {}, [], []
    for name, cut in thresholds.items():
        run = {"threshold": round(float(cut), 6), "pooled": report(pooled, cut)}
        run.update({w: report(m, cut) for w, m in masks.items()})
        runs[name] = run
        per_window = {w: index(run[w]) for w in masks}
        confirmed = set()
        for f in run["pooled"]["findings"]:
            key = (f["attribute"], f["group"])
            window_ratios = {w: per_window[w].get(key, {}).get("fpr_ratio") for w in masks}
            base = {"operating_point": name, "attribute": f["attribute"], "group": f["group"],
                    "pooled_fpr_ratio": f["fpr_ratio"], "pooled_q_value": f["fpr_q_value"],
                    "legit_flagged": f["legit_flagged"], "window_ratios": window_ratios}
            if all(r is not None and r > 1 for r in window_ratios.values()):
                findings.append(base)
                confirmed.add(key)
            else:
                missing = [w for w, r in window_ratios.items() if r is None]
                watchlist.append({**base, "reason": (f"too few customers to test in {', '.join(missing)}" if missing
                                                     else "not above the overall rate in every window")})
                confirmed.add(key)
        for w in masks:
            for key, g in per_window[w].items():
                if key in confirmed or not (g["fpr_ratio"] and g["fpr_ratio"] > ratio_threshold
                                            and g["legit_flagged"] >= 5 and g["fpr_p_value"] < alpha):
                    continue
                pooled_row = index(run["pooled"]).get(key, {})
                watchlist.append({
                    "operating_point": name, "attribute": key[0], "group": key[1], "window": w,
                    "fpr_ratio": g["fpr_ratio"], "fpr_p_value": g["fpr_p_value"],
                    "pooled_fpr_ratio": pooled_row.get("fpr_ratio"), "pooled_q_value": pooled_row.get("fpr_q_value"),
                    "window_ratios": {v: per_window[v].get(key, {}).get("fpr_ratio") for v in masks},
                    "reason": "material in one window only; not significant once windows are pooled and corrected",
                })
                confirmed.add(key)

    return {
        "status": "review_required" if findings else "pass",
        "method": (f"False-positive-rate parity at each operating point on the pooled out-of-sample windows "
                   f"({', '.join(masks)}); two-proportion z-test with Benjamini–Hochberg correction across groups; "
                   f"a finding needs pooled FPR ratio > {ratio_threshold}, q < {alpha}, and an FPR above the overall "
                   f"rate in every window. Single-window signals are kept on a non-gating watchlist."),
        "windows": list(masks),
        "findings": findings,
        "watchlist": watchlist,
        "operating_points": runs,
    }


DEFAULT_AMOUNT_BANDS = (0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0)


def decision_fairness(y, intervened, groups: Dict[str, pd.Series], amount, bands=DEFAULT_AMOUNT_BANDS,
                      ratio_threshold: float = DEFAULT_RATIO_THRESHOLD, alpha: float = DEFAULT_ALPHA,
                      min_n: int = DEFAULT_MIN_N) -> Dict:
    """
    Fairness of the *decisions*, not the scores: how often genuine customers in each group are intervened on
    (step-up, review or decline).

    Expected-cost decisions weigh probability by amount, so groups that make large payments are intervened on
    more. That is a legitimate risk factor, but it can hide or mimic group effects. So each group gets two ratios:
    - raw: the group's genuine intervention rate against the overall rate
    - amount-standardised: observed interventions against those expected if the group had each amount band's
      overall rate (indirect standardisation over genuine-transaction amount bands). A ratio above 1 means the
      group is intervened on more than its amounts explain.

    A finding needs a standardised ratio above `ratio_threshold`, at least 5 interventions, and a
    Benjamini–Hochberg-corrected Poisson test below alpha.
    """
    from scipy.stats import poisson

    y = np.asarray(y, dtype=int)
    iv = np.asarray(intervened, dtype=bool)
    amt = np.nan_to_num(np.asarray(amount, dtype=float))
    genuine = y == 0
    edges = np.unique(np.quantile(amt[genuine], bands)) if genuine.any() else np.array([0.0, 1.0])
    band = np.clip(np.searchsorted(edges, amt, side="right") - 1, 0, max(len(edges) - 2, 0))
    band_rate = {b: float(iv[genuine & (band == b)].mean()) for b in np.unique(band[genuine])}
    expected_each = np.array([band_rate.get(b, 0.0) for b in band])
    overall = float(iv[genuine].mean()) if genuine.any() else 0.0

    candidates, attributes = [], []
    for attr, values in groups.items():
        seg = pd.Series(np.asarray(values)).astype("string").fillna("(missing)").to_numpy()
        rows = []
        for name in sorted(set(seg)):
            m = (seg == name) & genuine
            if m.sum() < min_n:
                continue
            observed, expected = int(iv[m].sum()), float(expected_each[m].sum())
            sir = observed / expected if expected > 0 else None
            if expected > 0:
                p_lo, p_hi = poisson.cdf(observed, expected), poisson.sf(observed - 1, expected)
                p_value = min(1.0, 2 * min(p_lo, p_hi))
            else:
                p_value = 1.0
            row = {"group": str(name), "genuine": int(m.sum()), "intervened": observed,
                   "intervention_rate": round(float(iv[m].mean()), 5),
                   "raw_ratio": round(float(iv[m].mean()) / overall, 3) if overall else None,
                   "expected_from_amounts": round(expected, 1),
                   "amount_standardised_ratio": round(sir, 3) if sir is not None else None,
                   "p_value": round(p_value, 5)}
            row["_material"] = bool(sir is not None and sir > ratio_threshold and observed >= 5)
            candidates.append((attr, row))
            rows.append(row)
        attributes.append({"attribute": attr, "groups": rows})
    q = benjamini_hochberg([r["p_value"] for _, r in candidates])
    findings = []
    for (attr, row), qv in zip(candidates, q):
        row["q_value"] = round(qv, 5)
        if row.pop("_material") and qv < alpha:
            findings.append({"attribute": attr, **row})
    return {"method": "Genuine-customer intervention rate, raw and indirectly standardised over amount bands "
                      f"(quantiles {list(bands)}); Poisson test with Benjamini–Hochberg correction",
            "overall_genuine_intervention_rate": round(overall, 5),
            "band_rates": {f"{edges[b]:.0f}–{edges[min(b + 1, len(edges) - 1)]:.0f} USD": round(r, 4)
                           for b, r in sorted(band_rate.items())},
            "attributes": attributes, "findings": findings,
            "status": "review_required" if findings else "pass"}
