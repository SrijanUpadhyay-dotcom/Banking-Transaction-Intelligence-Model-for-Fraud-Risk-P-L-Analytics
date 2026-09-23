"""
Fairness testing for fraud decisions across protected / proxy attributes.

In fraud screening the customer harm is a *legitimate* customer being held or
declined, so the primary test is false-positive-rate parity: the share of
legitimate customers in each group who are flagged, relative to the overall
rate. A group is flagged for review only when the disparity is both material
(ratio above `ratio_threshold`) and statistically significant (two-proportion
z-test), so small-sample noise does not generate findings.

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

    attributes, findings = [], []
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
            row["disparity_flag"] = bool(ratio is not None and ratio > ratio_threshold
                                         and p_value < alpha and fp_g >= 5)
            if row["disparity_flag"]:
                findings.append({"attribute": attr, **row})
            rows.append(row)
        best = max(approval.values()) if approval else 0
        for row in rows:
            row["approval_air"] = round(approval[row["group"]] / best, 4) if best else None
        attributes.append({"attribute": attr, "groups": rows})

    return {
        "method": "False-positive-rate parity with two-proportion z-test; approval adverse impact ratio",
        "overall_fpr": round(overall_fpr, 5),
        "overall_tdr": round(overall_tdr, 4),
        "ratio_threshold": ratio_threshold,
        "alpha": alpha,
        "min_group_n": min_n,
        "attributes": attributes,
        "findings": findings,
        "status": "review_required" if findings else "pass",
    }
