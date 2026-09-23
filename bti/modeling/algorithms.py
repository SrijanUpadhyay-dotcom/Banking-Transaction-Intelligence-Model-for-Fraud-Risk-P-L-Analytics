"""
Gradient-boosting back-ends behind one interface: histogram GBM (scikit-learn),
LightGBM and XGBoost.

Every algorithm gets the same treatment so candidates are comparable:
monotonic constraints, early stopping on the most recent slice of the training
window (never a random split), deterministic seeds, and SHAP helpers whose
additivity is checked against the model's own raw margin.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

ALGORITHMS = ("hgb", "lightgbm", "xgboost")
SHORT = {"hgb": "hgb", "lightgbm": "lgbm", "xgboost": "xgb"}
DESCRIPTION = {
    "hgb": "scikit-learn HistGradientBoostingClassifier (histogram gradient boosting)",
    "lightgbm": "LightGBM LGBMClassifier (leaf-wise histogram gradient boosting)",
    "xgboost": "XGBoost XGBClassifier (hist tree method)",
}
EARLY_STOPPING_ROUNDS = 50
N_JOBS = 4

DEFAULT_PARAMS: Dict[str, Dict] = {
    "hgb": dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0),
    "lightgbm": dict(learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, reg_lambda=1.0),
    "xgboost": dict(learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0),
}

TUNING_GRID: Dict[str, Dict[str, List]] = {
    "hgb": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 50]},
    "lightgbm": {"learning_rate": [0.03, 0.06], "num_leaves": [15, 31, 63], "min_child_samples": [20, 50]},
    "xgboost": {"learning_rate": [0.03, 0.06], "max_depth": [4, 6, 8], "min_child_weight": [1, 5]},
}


def _check(algorithm: str) -> None:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")


def fit(algorithm: str, params: Dict, monotone: List[int], X_fit: pd.DataFrame, y_fit: np.ndarray,
        X_val: pd.DataFrame, y_val: np.ndarray):
    """
    Early-stop on (X_val, y_val), then — for LightGBM and XGBoost — refit with
    exactly the best number of trees, so the stored model contains only the
    trees it predicts with and SHAP explains the same function that scores.
    `monotone` aligns with X_fit.columns.
    """
    _check(algorithm)
    if algorithm == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            **params, max_iter=2000, early_stopping=True, n_iter_no_change=EARLY_STOPPING_ROUNDS,
            validation_fraction=None, categorical_features=None, random_state=42,
            monotonic_cst=dict(zip(X_fit.columns, monotone)))
        return model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
    if algorithm == "lightgbm":
        import lightgbm as lgb

        def make(n):
            return lgb.LGBMClassifier(**params, n_estimators=n, monotone_constraints=list(monotone),
                                      random_state=42, deterministic=True, force_row_wise=True,
                                      n_jobs=N_JOBS, verbose=-1)
        probe = make(2000).fit(X_fit, y_fit, eval_X=(X_val,), eval_y=(y_val,), eval_metric="binary_logloss",
                               callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        return make(max(int(probe.best_iteration_), 1)).fit(X_fit, y_fit)
    import xgboost as xgb

    def make(n, early):
        return xgb.XGBClassifier(**params, n_estimators=n, monotone_constraints=tuple(monotone),
                                 tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=N_JOBS,
                                 early_stopping_rounds=EARLY_STOPPING_ROUNDS if early else None)
    probe = make(2000, True).fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return make(int(probe.best_iteration) + 1, False).fit(X_fit, y_fit)


def iterations_used(model) -> int:
    if hasattr(model, "n_iter_") and model.__class__.__module__.startswith("sklearn"):
        return int(model.n_iter_)
    return int(getattr(model, "n_estimators", -1))


def raw_margin(model, X: pd.DataFrame) -> np.ndarray:
    """Log-odds before the sigmoid — what SHAP values must add up to."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    if model.__class__.__module__.startswith("lightgbm"):
        return np.asarray(model.predict(X, raw_score=True), dtype=float)
    return np.asarray(model.predict(X, output_margin=True), dtype=float)


def shap_matrix(explainer, X: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    return values[..., -1] if values.ndim == 3 else values


def shap_base(explainer) -> float:
    return float(np.ravel(explainer.expected_value)[-1])


def shap_additivity_error(model, explainer, X: pd.DataFrame) -> float:
    return float(np.abs(shap_matrix(explainer, X).sum(1) + shap_base(explainer) - raw_margin(model, X)).max())
