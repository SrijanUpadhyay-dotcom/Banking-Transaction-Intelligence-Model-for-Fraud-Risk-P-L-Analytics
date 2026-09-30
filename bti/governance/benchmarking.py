# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Benchmarking, sensitivity analysis and stress testing (SR 11-7 "outcomes and
benchmarking"; PRA SS1/23 Principle 4).

Every figure below is computed on the out-of-time window, which is never used
for training or selection.

- **Benchmarks.** The model is compared with three alternatives:
  - a logistic regression on the same features, the conventional
    interpretable alternative
  - the other candidates registered on the same data, i.e. the alternatives
    considered
  - a simple pre-authorisation rules engine
- **Uncertainty.** Bootstrap 95% intervals on ROC-AUC and PR-AUC, plus
  month-by-month stability.
- **Sensitivity.** Each numeric input is moved ±1 training standard deviation
  on a sample of transactions. The report gives the mean change in probability
  and the share of decisions that flip at the 2% and 10% alert-budget
  thresholds.
- **Monotonicity.** Every constrained feature is swept across its range. The
  probability must never fall as risk rises.
- **Stress scenarios.** Transactions are perturbed at the source-field level,
  features rebuilt, and each scenario compared with the baseline at fixed
  thresholds:
  - an inflation / FX shock
  - a spending spike
  - a velocity surge
  - missing profile data
  - a surge of new-to-bank customers
  - a channel shift
  - a doubled fraud prior

Results are stored per model under `models/registry/validation/<model_id>/` and
the latest is rendered into the documentation pack.

Usage:
  python -m bti.governance.benchmarking [--model <model_id>]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bti.modeling import registry
from bti.modeling.features import CATEGORICAL_FEATURES, build_features, out_of_fold_category_encoding, to_model_matrix
from bti.modeling.metrics import classification_metrics, threshold_for_alert_rate

STRESS_FLAGS = {"alert_rate_relative_change": 0.5, "tdr_drop": 0.05, "roc_auc_drop": 0.02}


def _validation_dir(model_id: str) -> Path:
    return registry.registry_dir() / "validation" / model_id


def _tdr(y, flagged) -> Optional[float]:
    return round(float(flagged[y == 1].mean()), 4) if (y == 1).any() else None


def _point(y, p, cut) -> Dict:
    flag = p >= cut
    return {"alert_rate": round(float(flag.mean()), 5), "tdr": _tdr(y, flag),
            "genuine_flag_rate": round(float(flag[y == 0].mean()), 5) if (y == 0).any() else None,
            "precision": round(float(y[flag].mean()), 4) if flag.any() else None}


def _bootstrap_auc(y, p, reps=500, seed=0) -> Dict:
    rng = np.random.default_rng(seed)
    roc, pr = [], []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        roc.append(roc_auc_score(y[i], p[i]))
        pr.append(average_precision_score(y[i], p[i]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"reps": reps, "roc_auc_ci95": q(roc), "pr_auc_ci95": q(pr)}


def _predict(art, X: pd.DataFrame) -> np.ndarray:
    return art["calibrator"].predict(art["estimator"].predict_proba(X)[:, 1])


def benchmark_report(model_id: Optional[str] = None, data_path=None, sample: int = 3000, seed: int = 0) -> Dict:
    from bti.modeling.train import MONOTONE_INCREASING, prepare
    from bti.parallel.simulate import rules_stand_in

    model_id = model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
    art = registry.load_artifact(model_id)
    card = registry.load_card(model_id)
    names = art.get("feature_names")
    data = prepare(data_path, feature_version=art.get("feature_version", 1))
    if card.get("data", {}).get("sha256") not in (None, data.sha256):
        raise ValueError(f"{model_id} was trained on different data; benchmark it on its own training data")
    y, tr, ca, te = data.y, data.tr, data.ca, data.te
    X = to_model_matrix(data.features, art["encodings"], names)
    p = _predict(art, X)
    cut2, cut10 = threshold_for_alert_rate(p[ca], 0.02), threshold_for_alert_rate(p[ca], 0.10)
    rng = np.random.default_rng(seed)
    report: Dict = {"model_id": model_id, "generated_at": datetime.now(timezone.utc).isoformat(),
                    "window": "out-of-time (last 25% by time)", "data_sha256": data.sha256}

    # ── Benchmarks ───────────────────────────────────────────────────────────
    model_m = classification_metrics(y[te], p[te])
    X_lr = X.copy()
    X_lr.loc[tr, [c for c in CATEGORICAL_FEATURES if c in names]] = out_of_fold_category_encoding(
        data.features[tr], y[tr])[[c for c in CATEGORICAL_FEATURES if c in names]].to_numpy()
    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                       LogisticRegression(max_iter=2000, C=1.0))
    lr.fit(X_lr[tr], y[tr])
    lr_p = lr.predict_proba(X_lr)[:, 1]
    rules = rules_stand_in(data.features.loc[te]).incumbent_score.to_numpy(float)
    alternatives = []
    for m in registry.read_index()["models"]:
        c = registry.load_card(m["model_id"])
        if m["model_id"] != model_id and c.get("data", {}).get("sha256") == data.sha256:
            alternatives.append({"model_id": m["model_id"], "validation": c["validation"]["status"],
                                 "feature_set": c["features"].get("feature_set"),
                                 "oot_roc_auc": c["performance"]["metrics"]["out_of_time"].get("roc_auc"),
                                 "oot_pr_auc": c["performance"]["metrics"]["out_of_time"].get("pr_auc")})
    alternatives.sort(key=lambda a: -(a["oot_pr_auc"] or 0))
    better = [a for a in alternatives if a["validation"] == "passed" and (a["oot_pr_auc"] or 0) > model_m["pr_auc"]]
    report["benchmarks"] = {
        "model": {"roc_auc": model_m["roc_auc"], "pr_auc": model_m["pr_auc"], "ece": model_m["ece"],
                  "at_2pct_budget": _point(y[te], p[te], cut2), "at_10pct_budget": _point(y[te], p[te], cut10),
                  **_bootstrap_auc(y[te], p[te], seed=seed)},
        "logistic_regression": {"roc_auc": round(float(roc_auc_score(y[te], lr_p[te])), 4),
                                "pr_auc": round(float(average_precision_score(y[te], lr_p[te])), 4),
                                "at_2pct_budget": _point(y[te], lr_p[te], threshold_for_alert_rate(lr_p[ca], 0.02)),
                                "specification": "median imputation with missing indicators, standardised, L2 C=1"},
        "rules_stand_in": {"roc_auc": round(float(roc_auc_score(y[te], rules)), 4),
                           "pr_auc": round(float(average_precision_score(y[te], rules)), 4),
                           "specification": "bti.parallel.simulate.rules_stand_in (pre-authorisation rules)"},
        "alternatives_considered": alternatives[:12],
        "passed_alternatives_with_higher_oot_pr_auc": [a["model_id"] for a in better],
        "note": "Alternatives were selected on the calibration window, so a higher out-of-time figure among them "
                "is expected by chance; differences inside the bootstrap interval are noise.",
    }

    # ── Temporal stability ───────────────────────────────────────────────────
    months = data.df.loc[te, "_ts"].dt.to_period("M").astype(str).to_numpy()
    stability = []
    for mth in sorted(set(months)):
        m = months == mth
        yy, pp = y[te][m], p[te][m]
        met = classification_metrics(yy, pp)
        stability.append({"month": mth, "n": int(m.sum()), "fraud": int(yy.sum()), "roc_auc": met.get("roc_auc"),
                          "pr_auc": met.get("pr_auc"), "ece": met.get("ece"),
                          "alert_rate_2pct_threshold": round(float((pp >= cut2).mean()), 5)})
    report["temporal_stability"] = stability

    # ── Sensitivity ──────────────────────────────────────────────────────────
    idx = rng.choice(np.flatnonzero(te), min(sample, int(te.sum())), replace=False)
    Xs, base = X.iloc[idx], p[idx]
    numeric = [c for c in names if c not in CATEGORICAL_FEATURES]
    sd = X.loc[tr, numeric].std()
    sens = []
    for f in numeric:
        row = {"feature": f}
        for sign, label in ((1, "plus_1sd"), (-1, "minus_1sd")):
            Xp = Xs.copy()
            Xp[f] = Xp[f] + sign * sd[f]
            pp = _predict(art, Xp)
            row[label] = {"mean_abs_change": round(float(np.abs(pp - base).mean()), 5),
                          "flips_at_2pct_threshold": round(float(((pp >= cut2) != (base >= cut2)).mean()), 5),
                          "flips_at_10pct_threshold": round(float(((pp >= cut10) != (base >= cut10)).mean()), 5)}
        row["max_flip_rate"] = max(row[k]["flips_at_10pct_threshold"] for k in ("plus_1sd", "minus_1sd"))
        sens.append(row)
    sens.sort(key=lambda r: -max(r["plus_1sd"]["mean_abs_change"], r["minus_1sd"]["mean_abs_change"]))
    report["sensitivity"] = {"sample": int(len(idx)), "perturbation": "±1 training-window standard deviation",
                             "features": sens}

    # ── Monotonicity ─────────────────────────────────────────────────────────
    mono = []
    grid_rows = Xs.iloc[:300]
    for f in [c for c in numeric if c in MONOTONE_INCREASING]:
        values = np.unique(np.nanpercentile(X.loc[tr, f].to_numpy(float), np.linspace(1, 99, 25)))
        preds = []
        for v in values:
            Xg = grid_rows.copy()
            Xg[f] = v
            preds.append(_predict(art, Xg))
        preds = np.vstack(preds)                                       # grid × rows
        worst = float(np.max(np.maximum(0, preds[:-1] - preds[1:]))) if len(values) > 1 else 0.0
        mono.append({"feature": f, "grid_points": int(len(values)), "max_decrease": round(worst, 8),
                     "monotone": worst <= 1e-9})
    report["monotonicity"] = {"all_monotone": all(m["monotone"] for m in mono), "features": mono}

    # ── Stress scenarios ─────────────────────────────────────────────────────
    report["stress"] = _stress(data, art, names, cut2, cut10, rng)
    report["summary"] = _summary(report)
    report["proposed_findings"] = proposed_findings(report)
    return report


def _rescore(data, art, names, df: pd.DataFrame) -> np.ndarray:
    feats = build_features(df.drop(columns=["_ts"]), lookback_days=data.lookback_days,
                           feature_version=art.get("feature_version", 1), security_events=data.security_events)
    return _predict(art, to_model_matrix(feats, art["encodings"], names))


def _stress(data, art, names, cut2, cut10, rng) -> List[Dict]:
    y, te = data.y, data.te
    base_df = data.df
    p0 = _rescore(data, art, names, base_df)
    te_idx = np.flatnonzero(te)

    def outcome(name, description, p, label_preserving=True, weights=None):
        yy, pp = y[te], p[te]
        row = {"scenario": name, "description": description,
               "at_2pct_threshold": _point(yy, pp, cut2), "at_10pct_threshold": _point(yy, pp, cut10),
               "mean_probability": round(float(pp.mean()), 5)}
        if label_preserving and len(np.unique(yy)) > 1:
            row["roc_auc"] = round(float(roc_auc_score(yy, pp)), 4)
        if weights is not None:
            row["observed_fraud_rate"] = round(float(np.average(yy, weights=weights)), 5)
            row["mean_probability_weighted"] = round(float(np.average(pp, weights=weights)), 5)
        return row

    base = outcome("baseline", "Out-of-time window as observed", p0)
    rows = [base]

    def scenario(name, description, mutate):
        df = base_df.copy()
        mutate(df)
        rows.append(outcome(name, description, _rescore(data, art, names, df)))

    def inflation(df):
        for c in ("transaction_amount", "account_balance_before", "historical_average_transaction_amount"):
            df.loc[te, c] = pd.to_numeric(df.loc[te, c], errors="coerce") * 1.5
    scenario("inflation_fx_shock", "Amounts, balances and profile averages all +50% (price level / FX move)",
             inflation)

    def spike(df):
        df.loc[te, "transaction_amount"] = pd.to_numeric(df.loc[te, "transaction_amount"], errors="coerce") * 2
    scenario("spending_spike", "Every transaction amount doubled, profiles unchanged", spike)

    def missing(df):
        pick = te_idx[rng.random(len(te_idx)) < 0.3]
        df.loc[pick, ["historical_average_transaction_amount", "account_balance_before", "login_attempts"]] = np.nan
    scenario("missing_profile_data", "30% of transactions arrive without profile average, balance and login count",
             missing)

    def new_to_bank(df):
        customers = df.loc[te, "customer_id"].unique()
        chosen = set(rng.choice(customers, int(len(customers) * 0.3), replace=False))
        m = te & df["customer_id"].isin(chosen).to_numpy()
        df.loc[m, "customer_id"] = "NEW-" + df.loc[m, "customer_id"].astype(str)
    scenario("new_to_bank_surge", "30% of out-of-time customers arrive with no history", new_to_bank)

    def channels(df):
        m = te & df["channel"].isin(["Branch", "ATM"]).to_numpy() & (rng.random(len(df)) < 0.3)
        df.loc[m, "channel"] = "Mobile Banking"
    scenario("channel_shift", "30% of branch and ATM transactions move to mobile banking", channels)

    X = to_model_matrix(data.features, art["encodings"], names).copy()
    for c, add in (("cust_txn_count_1h", 3), ("cust_txn_count_24h", 5), ("cust_txn_count_7d", 8)):
        if c in X.columns:
            X.loc[te, c] = X.loc[te, c] + add
    rows.append(outcome("velocity_surge", "Every customer shows +3 transactions in the prior hour, +5 in 24h, +8 in 7d "
                        "(bot or account-testing wave)", _predict(art, X)))

    w = np.where(y[te] == 1, 2.0, 1.0)
    rows.append(outcome("fraud_prior_doubled", "Fraud twice as common, same patterns (reweighted): tests whether "
                        "calibrated probabilities track a prior shift", p0, weights=w))

    b2 = base["at_2pct_threshold"]
    for r in rows[1:]:
        a = r["at_2pct_threshold"]
        flags = []
        if b2["alert_rate"] and abs(a["alert_rate"] - b2["alert_rate"]) / b2["alert_rate"] > STRESS_FLAGS["alert_rate_relative_change"]:
            flags.append(f"alert rate {b2['alert_rate']:.2%} → {a['alert_rate']:.2%}")
        if a["tdr"] is not None and b2["tdr"] is not None and b2["tdr"] - a["tdr"] > STRESS_FLAGS["tdr_drop"]:
            flags.append(f"detection {b2['tdr']:.1%} → {a['tdr']:.1%}")
        if "roc_auc" in r and base.get("roc_auc") and base["roc_auc"] - r["roc_auc"] > STRESS_FLAGS["roc_auc_drop"]:
            flags.append(f"ROC-AUC {base['roc_auc']} → {r['roc_auc']}")
        r["flags"] = flags
    return rows


def _summary(r: Dict) -> Dict:
    b = r["benchmarks"]
    stressed = [s["scenario"] for s in r["stress"] if s.get("flags")]
    most_sensitive = r["sensitivity"]["features"][0]["feature"] if r["sensitivity"]["features"] else None
    return {
        "beats_logistic_regression_pr_auc": b["model"]["pr_auc"] > b["logistic_regression"]["pr_auc"],
        "pr_auc_gain_vs_logistic": round(b["model"]["pr_auc"] - b["logistic_regression"]["pr_auc"], 4),
        "pr_auc_gain_vs_rules": round(b["model"]["pr_auc"] - b["rules_stand_in"]["pr_auc"], 4),
        "monotonicity_holds": r["monotonicity"]["all_monotone"],
        "most_sensitive_feature": most_sensitive,
        "stress_scenarios_flagged": stressed,
    }


VELOCITY_FEATURES = ("cust_txn_count_1h", "cust_txn_count_24h", "cust_txn_count_7d", "cust_amount_usd_24h",
                     "merchant_txn_count_1h", "device_txn_count_24h")


def proposed_findings(r: Dict) -> List[Dict]:
    """Findings the report supports, in the tracker's format (title, severity, category, description)."""
    out = []
    b = r["benchmarks"]
    ci = b["model"]["pr_auc_ci95"]
    gain = b["model"]["pr_auc"] - b["logistic_regression"]["pr_auc"]
    if gain < (ci[1] - ci[0]) / 2:
        out.append(("Gain over the logistic-regression benchmark is within noise", "low", "methodology",
                    f"Out-of-time PR-AUC {b['model']['pr_auc']} vs {b['logistic_regression']['pr_auc']} for a "
                    f"logistic regression on the same features (model 95% CI {ci}). The added complexity is not yet "
                    f"justified by accuracy; re-benchmark on bank data."))
    sens = r["sensitivity"]["features"]
    if sens and sens[0]["max_flip_rate"] > 0.5:
        out.append((f"Decisions concentrated on one input: {sens[0]['feature']}", "medium", "data",
                    f"A one-standard-deviation rise in {sens[0]['feature']} flips {sens[0]['max_flip_rate']:.0%} of "
                    f"decisions at the 10% alert-budget threshold. Its source field must be defined and maintained "
                    f"point-in-time exactly as in training; a change in how the bank computes it moves decisions."))
    velocity = [f for f in sens if f["feature"] in VELOCITY_FEATURES]
    if velocity and all(max(f["plus_1sd"]["mean_abs_change"], f["minus_1sd"]["mean_abs_change"]) < 0.001
                        for f in velocity):
        out.append(("Velocity features carry no weight", "medium", "methodology",
                    "Transaction-velocity features change the probability by less than 0.001 at ±1 SD, and the "
                    "velocity-surge stress scenario changes no decision. The model would not react to a card-testing "
                    "or bot burst. Synthetic customers average ~6 transactions, so velocity was uninformative in "
                    "training; re-test on bank data and keep velocity rules in the incumbent until then."))
    for row in r["stress"]:
        if row.get("flags"):
            out.append((f"Stress scenario degrades performance: {row['scenario']}", "medium", "robustness",
                        f"{row['description']}: {'; '.join(row['flags'])} at the 2% alert-budget threshold."))
    prior = next((row for row in r["stress"] if row["scenario"] == "fraud_prior_doubled"), None)
    if prior and prior.get("observed_fraud_rate"):
        gap = abs(prior["mean_probability_weighted"] - prior["observed_fraud_rate"]) / prior["observed_fraud_rate"]
        if gap > 0.05:
            out.append(("Probabilities do not follow a shift in the fraud rate", "low", "monitoring",
                        f"With fraud twice as common, mean probability is {prior['mean_probability_weighted']:.2%} "
                        f"against an observed {prior['observed_fraud_rate']:.2%}. Quarterly outcomes analysis must "
                        f"trigger recalibration when calibration drifts."))
    return [{"title": t, "severity": sev, "category": cat, "description": d} for t, sev, cat, d in out]


def run_and_store(model_id: Optional[str] = None) -> Dict:
    report = benchmark_report(model_id)
    folder = _validation_dir(report["model_id"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"benchmark-{datetime.now(timezone.utc):%Y%m%d%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    report["path"] = str(path)
    return report


def latest_report(model_id: str) -> Optional[Dict]:
    files = sorted(_validation_dir(model_id).glob("benchmark-*.json"))
    return json.loads(files[-1].read_text()) if files else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark, sensitivity and stress-test a registered model")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    r = run_and_store(args.model)
    b, s = r["benchmarks"], r["summary"]
    print(f"{r['model_id']}  (out-of-time)")
    print(f"  model               ROC {b['model']['roc_auc']}  PR {b['model']['pr_auc']}  "
          f"PR 95% CI {b['model']['pr_auc_ci95']}")
    print(f"  logistic regression ROC {b['logistic_regression']['roc_auc']}  PR {b['logistic_regression']['pr_auc']}")
    print(f"  rules stand-in      ROC {b['rules_stand_in']['roc_auc']}  PR {b['rules_stand_in']['pr_auc']}")
    print(f"  monotonicity holds: {s['monotonicity_holds']}; most sensitive: {s['most_sensitive_feature']}")
    for row in r["stress"]:
        a = row["at_2pct_threshold"]
        print(f"  {row['scenario']:22s} alert {a['alert_rate']:.2%}  TDR {a['tdr']}  "
              f"ROC {row.get('roc_auc', '—')}  {'; '.join(row.get('flags', [])) or ''}")
    print(f"Stored: {r['path']}")


if __name__ == "__main__":
    main()
