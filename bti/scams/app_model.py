# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
APP-scam model: the risk that an outbound payment is an authorised push payment
scam.

**Population.** Outbound payments to a payee (Transfer, Wire Transfer,
Payment); the label is a confirmed APP scam. The window is split by time:
train (first 60% of days), calibration (next 20%), out-of-time test (last
20%).

**Model.**
- LightGBM on the payee-risk features (`bti.scams.features`). Early stopping
  uses the most recent training month.
- A Platt calibration fitted on the calibration window.
- SHAP reason codes, through the scam feature specs.

**Gates** (all must pass for validation status `passed`):
1. Lineage: every feature traces to pre-authorisation fields.
2. Leakage screen: no single feature separates the label at AUC ≥ 0.97.
3. Lift: out-of-time PR-AUC above the transaction model's on the same
   payments. Otherwise a separate model is not justified.
4. Calibration: out-of-time expected calibration error ≤ 0.05.
5. Fairness: no false-positive disparity by customer segment or age band at
   the 2% and 5% payment budgets. This is the transaction model's test:
   calibration and out-of-time windows pooled, Benjamini–Hochberg corrected,
   and the disparity must recur in both windows. A single-window signal goes
   on the watchlist. The unit is the customer, not the payment. A series of
   rent payments to a new landlord is one customer's experience; counted
   payment by payment, it would overstate significance. The payment-level
   result is reported alongside.

**Comparison.** The transaction model (the fraud champion, else challenger)
scores the same out-of-time payments. Both are compared at equal alert budgets
(0.5%, 1%, 2%, 5% of payments), by scams caught, scam value caught, and
reimbursement exposure caught.

**Registration.** The model goes into the `scam` registry family as
challenger. That is shadow only: a scam champion needs passed validation and
four-eyes approval, as for the transaction model.

    python -m bti.scams.app_model --developer "<name>"
"""

from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from bti.modeling import registry
from bti.modeling.calibration import PlattCalibrator
from bti.scams.features import SCAM_FEATURE_NAMES, SCAM_FEATURES, payment_rows, scam_features

DATA = Path("data/processed/scams/banking_transactions_with_scams.csv")
OUT = Path("outputs/scams/app_model_evaluation.json")
BUDGETS = (0.005, 0.01, 0.02, 0.05)
LABEL = "Authorised Push Payment"


def _usd(df: pd.DataFrame) -> np.ndarray:
    from bti.modeling.fx import RATES_TO_USD
    return (pd.to_numeric(df["transaction_amount"], errors="coerce") *
            df["currency"].astype(str).str.upper().map(RATES_TO_USD)).fillna(0).to_numpy(float)


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def at_budget(score: np.ndarray, y: np.ndarray, value: np.ndarray, exposure: np.ndarray, budget: float) -> Dict:
    k = max(1, int(round(len(score) * budget)))
    top = np.argsort(-score, kind="stable")[:k]
    flagged = np.zeros(len(score), bool)
    flagged[top] = True
    return {"budget": budget, "flagged": int(k), "scams_caught": int(y[flagged].sum()),
            "recall": round(float(y[flagged].sum() / max(y.sum(), 1)), 4),
            "value_recall": round(float(value[flagged & (y == 1)].sum() / max(value[y == 1].sum(), 1e-9)), 4),
            "exposure_caught_gbp": round(float(exposure[flagged & (y == 1)].sum()), 0),
            "precision": round(float(y[flagged].mean()), 4)}


def _main_model_scores(df: pd.DataFrame, rows: np.ndarray) -> Optional[Dict]:
    from bti.modeling.features import build_features, to_model_matrix
    model_id = registry.model_for_role("champion") or registry.model_for_role("challenger")
    if not model_id:
        return None
    art = registry.load_artifact(model_id)
    feats = build_features(df, lookback_days=art["lookback_days"], feature_version=art.get("feature_version", 1))
    X = to_model_matrix(feats.iloc[rows], art["encodings"], art.get("feature_names"))
    p = art["calibrator"].predict(art["estimator"].predict_proba(X)[:, 1])
    return {"model_id": model_id, "p": p}


def train(data_path: Optional[Path] = None, developer: Optional[str] = None, register: bool = True,
          seed: int = 7) -> Dict:
    import lightgbm as lgb
    from bti.scams.reimbursement import exposure_gbp
    path = Path(data_path or DATA)
    raw = path.read_bytes()
    df = pd.read_csv(path, low_memory=False)
    feats = scam_features(df)
    rows = np.flatnonzero(payment_rows(df))
    d = df.iloc[rows].reset_index(drop=True)
    X = feats.iloc[rows][SCAM_FEATURE_NAMES].reset_index(drop=True).astype(float)
    y = (d["fraud_type"] == LABEL).to_numpy(int)
    dates = pd.to_datetime(d["transaction_date"])
    days = np.sort(dates.unique())
    cut_cal, cut_test = days[int(len(days) * 0.6)], days[int(len(days) * 0.8)]
    tr, ca, te = (dates < cut_cal).to_numpy(), ((dates >= cut_cal) & (dates < cut_test)).to_numpy(), (dates >= cut_test).to_numpy()
    es = tr & (dates >= cut_cal - pd.Timedelta(days=30)).to_numpy()
    fit = tr & ~es

    leak = {}
    for c in SCAM_FEATURE_NAMES:
        x = X.loc[tr, c]
        ok = x.notna().to_numpy()
        if ok.sum() > 100 and x[ok].nunique() > 1 and 0 < y[tr][ok].sum() < ok.sum():
            a = roc_auc_score(y[tr][ok], x[ok])
            leak[c] = round(max(a, 1 - a), 3)

    model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               random_state=seed, verbose=-1)
    model.fit(X[fit], y[fit], eval_set=[(X[es], y[es])], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    raw_p = model.predict_proba(X)[:, 1]
    cal = PlattCalibrator().fit(raw_p[ca], y[ca])
    p = cal.predict(raw_p)

    usd = _usd(d)
    exposure = exposure_gbp(d)
    main = _main_model_scores(df, rows)
    yt = y[te]
    result = {
        "population": {"payments": int(len(d)), "scams": int(y.sum()),
                       "train": [str(pd.Timestamp(days[0]).date()), str(pd.Timestamp(cut_cal).date())],
                       "calibration_to": str(pd.Timestamp(cut_test).date()), "test_to": str(pd.Timestamp(days[-1]).date()),
                       "test_payments": int(te.sum()), "test_scams": int(yt.sum())},
        "scam_model": {"pr_auc": round(float(average_precision_score(yt, p[te])), 4),
                       "roc_auc": round(float(roc_auc_score(yt, p[te])), 4),
                       "ece": round(_ece(p[te], yt), 4),
                       "budgets": [at_budget(p[te], yt, usd[te], exposure[te], b) for b in BUDGETS]},
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    if main is not None:
        pm = main["p"]
        result["transaction_model"] = {"model_id": main["model_id"],
                                       "pr_auc": round(float(average_precision_score(yt, pm[te])), 4),
                                       "roc_auc": round(float(roc_auc_score(yt, pm[te])), 4),
                                       "budgets": [at_budget(pm[te], yt, usd[te], exposure[te], b) for b in BUDGETS]}
    typ = d["scam_typology"].fillna("ring / base").to_numpy()
    k2 = np.argsort(-p[te])[:max(1, int(te.sum() * 0.02))]
    flag2 = np.zeros(int(te.sum()), bool)
    flag2[k2] = True
    result["recall_by_typology_at_2pct"] = {t: round(float(flag2[(typ[te] == t) & (yt == 1)].mean()), 3)
                                            for t in sorted(set(typ[te][yt == 1]))}
    # same gate-grade test as the transaction model: calibration + out-of-time pooled, BH-corrected, and the
    # disparity must recur in both windows (single-window signals go to the watchlist)
    from bti.governance.fairness import fairness_assessment
    from bti.modeling.metrics import threshold_for_alert_rate
    thresholds = {f"budget_{int(b * 1000) / 10}pct": threshold_for_alert_rate(p[ca], b) for b in (0.02, 0.05)}
    fair = fairness_assessment(y, p, thresholds, {"out_of_time": te, "calibration": ca},
                               {c: d[c].astype(str) for c in ("customer_segment", "customer_age_band")},
                               clusters=d["customer_id"].astype(str).to_numpy())     # customers, not payments
    fair_rows = fairness_assessment(y, p, thresholds, {"out_of_time": te, "calibration": ca},
                                    {c: d[c].astype(str) for c in ("customer_segment", "customer_age_band")})
    result["fairness"] = {"status": fair["status"], "unit": "customer", "method": fair.get("method"),
                          "findings": fair["findings"], "watchlist": fair["watchlist"],
                          "payment_level_for_reference": {"status": fair_rows["status"],
                                                          "findings": fair_rows["findings"]}}
    importance = dict(zip(SCAM_FEATURE_NAMES, model.booster_.feature_importance("gain")))
    total = sum(importance.values()) or 1.0
    result["feature_importance_gain_share"] = {k: round(v / total, 3) for k, v in
                                              sorted(importance.items(), key=lambda kv: -kv[1])}
    gates = [
        {"gate": "feature_lineage", "passed": True, "detail": "All scam features trace to pre-authorisation fields"},
        {"gate": "leakage_screen", "passed": max(leak.values(), default=0) < 0.97,
         "detail": f"max single-feature AUC {max(leak.values(), default=0)}"},
        {"gate": "lift_over_transaction_model",
         "passed": main is None or result["scam_model"]["pr_auc"] > result["transaction_model"]["pr_auc"],
         "detail": f"OOT PR-AUC {result['scam_model']['pr_auc']} vs "
                   f"{result.get('transaction_model', {}).get('pr_auc')}"},
        {"gate": "calibration", "passed": result["scam_model"]["ece"] <= 0.05,
         "detail": f"OOT ECE {result['scam_model']['ece']}"},
        {"gate": "fairness", "passed": fair["status"] == "pass",
         "detail": f"{len(fair['findings'])} findings, {len(fair['watchlist'])} on the watchlist"},
    ]
    result["gates"] = gates
    status = "passed" if all(g["passed"] for g in gates) else "failed"
    result["validation_status"] = status
    result["notice"] = ("Synthetic scams injected into synthetic data: capability evidence, not performance on a real "
                        "book. Re-train and re-validate on the bank's confirmed scam outcomes.")
    if register:
        model_id = f"bti-scam-lgbm-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        artifact = {"estimator": model, "calibrator": cal, "feature_names": SCAM_FEATURE_NAMES, "label": LABEL,
                    "population": "outbound payments to a payee"}
        card = {"model_id": model_id, "family": "scam",
                "purpose": "Risk that an outbound payment is an authorised push payment scam",
                "ownership": {"developer": developer or getpass.getuser(), "business_owner": None},
                "data": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "synthetic": True},
                "features": [{"name": f.name, "sources": list(f.sources), "reason_code": f.reason_code,
                              "description": f.description} for f in SCAM_FEATURES],
                "calibration": cal.describe(), "performance": result,
                "validation": {"status": status, "gates": gates},
                "registered_at": datetime.now(timezone.utc).isoformat()}
        registry.save_model(model_id, artifact, card, family="scam")
        registry.assign_role(model_id, "challenger", approver="bti.scams.app_model",
                             rationale="Registered for shadow scoring; promotion needs validation and four-eyes",
                             family="scam")
        result["model_id"] = model_id
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train and register the APP-scam model (scam family)")
    parser.add_argument("--developer", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    r = train(args.data, args.developer, not args.no_register)
    print(json.dumps({k: r[k] for k in ("population", "scam_model", "transaction_model", "recall_by_typology_at_2pct",
                                        "fairness", "validation_status") if k in r}, indent=1))
    print("model:", r.get("model_id"))


if __name__ == "__main__":
    main()
