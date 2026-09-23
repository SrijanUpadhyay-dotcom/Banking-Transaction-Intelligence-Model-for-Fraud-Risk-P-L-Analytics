"""
Hyperparameter search on rolling out-of-time folds inside the training window.

Fold k fits on the oldest `start_k` share of training rows and validates on the
next `width` share, so every validation slice is later than the data it was
fitted on. Category encodings are re-learned per fold, early stopping uses the
most recent slice of each fold's fit rows, and the calibration and final
out-of-time windows are never touched — they stay clean for selection and
reporting.
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from bti.logging_config import get_logger
from bti.modeling import algorithms
from bti.modeling.features import (
    CATEGORICAL_FEATURES, FEATURE_SETS, fit_category_encodings, out_of_fold_category_encoding, to_model_matrix,
)

log = get_logger("modeling.tuning")

FOLD_STARTS = (0.55, 0.70, 0.85)
FOLD_WIDTH = 0.15
EARLY_STOP_SHARE = 0.15


def rolling_folds(train_idx: np.ndarray, starts=FOLD_STARTS, width=FOLD_WIDTH) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(train_idx)
    return [(train_idx[: int(n * s)], train_idx[int(n * s): int(n * min(s + width, 1.0))]) for s in starts]


def _fold_score(data, algorithm: str, names: List[str], params: Dict, monotone: List[int],
                fit_idx: np.ndarray, val_idx: np.ndarray) -> Dict:
    features, y = data.features, data.y
    enc = fit_category_encodings(features.iloc[fit_idx], y[fit_idx])
    X_fit = to_model_matrix(features.iloc[fit_idx], enc, names)
    X_fit[CATEGORICAL_FEATURES] = out_of_fold_category_encoding(features.iloc[fit_idx], y[fit_idx]).to_numpy()
    X_val = to_model_matrix(features.iloc[val_idx], enc, names)
    cut = int(len(fit_idx) * (1 - EARLY_STOP_SHARE))
    model = algorithms.fit(algorithm, params, monotone, X_fit.iloc[:cut], y[fit_idx][:cut],
                           X_fit.iloc[cut:], y[fit_idx][cut:])
    p = model.predict_proba(X_val)[:, 1]
    yv = y[val_idx]
    return {"pr_auc": round(float(average_precision_score(yv, p)), 5),
            "roc_auc": round(float(roc_auc_score(yv, p)), 5),
            "iterations": algorithms.iterations_used(model)}


def tune(data, algorithm: str, feature_set: str, monotone_increasing: List[str],
         grid: Optional[Dict[str, List]] = None) -> Dict:
    grid = grid or algorithms.TUNING_GRID[algorithm]
    names = FEATURE_SETS[feature_set]
    monotone = [1 if n in monotone_increasing else 0 for n in names]
    folds = rolling_folds(np.flatnonzero(data.tr))
    keys = sorted(grid)
    results = []
    t0 = time.time()
    for values in itertools.product(*(grid[k] for k in keys)):
        params = {**algorithms.DEFAULT_PARAMS[algorithm], **dict(zip(keys, values))}
        scores = [_fold_score(data, algorithm, names, params, monotone, f, v) for f, v in folds]
        results.append({"params": dict(zip(keys, values)),
                        "mean_pr_auc": round(float(np.mean([s["pr_auc"] for s in scores])), 5),
                        "std_pr_auc": round(float(np.std([s["pr_auc"] for s in scores])), 5),
                        "folds": scores})
    results.sort(key=lambda r: -r["mean_pr_auc"])
    best = results[0]
    log.info("Tuning complete", extra={"algorithm": algorithm, "feature_set": feature_set,
                                        "best": best["params"], "seconds": round(time.time() - t0, 1)})
    return {
        "method": f"Grid search, {len(folds)} rolling out-of-time folds inside the training window "
                  f"(fit on oldest {', '.join(f'{s:.0%}' for s in FOLD_STARTS)}; validate on the next "
                  f"{FOLD_WIDTH:.0%}); selection by mean PR-AUC",
        "grid": grid,
        "best_params": {**algorithms.DEFAULT_PARAMS[algorithm], **best["params"]},
        "best_mean_pr_auc": best["mean_pr_auc"],
        "default_mean_pr_auc": next((r["mean_pr_auc"] for r in results
                                     if all(algorithms.DEFAULT_PARAMS[algorithm].get(k) == v
                                            for k, v in r["params"].items())), None),
        "results": results,
        "seconds": round(time.time() - t0, 1),
    }
