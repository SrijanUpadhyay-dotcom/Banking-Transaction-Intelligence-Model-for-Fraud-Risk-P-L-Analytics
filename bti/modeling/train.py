"""
Train, validate and register a BTI v3 fraud model.

Design choices a model validator will look for:
  * Only pre-authorisation features (lineage enforced in bti.modeling.features)
  * Out-of-time validation: train on the oldest 60% of the timeline, calibrate
    on the next 15%, test on the most recent 25% — never a random split
  * Platt calibration so the output is a usable probability (strictly
    monotone, so ranking — and therefore AUC — is untouched)
  * Categorical fields enter as out-of-fold fraud rates so SHAP reason codes
    are exact (verified by an additivity check)
  * Monotonic constraints so risk cannot fall as failed logins / velocity rise
  * Any of three algorithms (histogram GBM, LightGBM, XGBoost) and two feature
    sets (core, extended), all through identical gates; early stopping on the
    most recent 15% of the training window
  * Operating threshold chosen on the calibration window, then applied to the
    test window (no peeking at test data)
  * Automated validation gates; the result is registered, never auto-promoted
    to champion (four-eyes approval happens through the registry)

Usage:
  python -m bti.modeling.train --developer "jane.doe" [--algorithm lightgbm] [--feature-set extended]
  (compare several candidates with python -m bti.modeling.tournament)
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import sklearn

from bti.config import get_settings
from bti.governance.fairness import fairness_assessment
from bti.governance.monitoring import build_baseline
from bti.logging_config import get_logger
from bti.modeling import algorithms, fx, registry
from bti.modeling.calibration import PlattCalibrator
from bti.modeling.features import (
    CATEGORICAL_FEATURES, DEFAULT_LOOKBACK_DAYS, FEATURE_SET_SUFFIX, FEATURE_SETS, FEATURE_VERSION, PROTECTED_ATTRIBUTES, SOURCE_FIELDS,
    Availability, assert_feature_lineage, build_features, event_timestamps, feature_specs, feed_coverage,
    feeds_required,
    fit_category_encodings, leakage_audit, out_of_fold_category_encoding, to_model_matrix,
)
from bti.modeling.metrics import (
    alert_budget_table, classification_metrics, lift_table, operating_point, segment_performance,
    threshold_for_alert_rate,
)

log = get_logger("modeling.train")

LABEL = "fraud_flag"
REFERENCE_ALERT_BUDGET = 0.02
EARLY_STOP_SHARE = 0.15
MONOTONE_INCREASING = [
    "failed_attempt_count", "login_attempts", "amount_vs_hist_avg", "device_new_for_customer",
    "ip_new_for_customer", "cust_txn_count_1h", "cust_txn_count_24h", "cust_txn_count_7d",
    "amount_usd", "log_amount_usd", "amount_to_balance",
    "device_txn_count_24h", "amount_zscore_customer", "cust_near_threshold_7d", "hour_deviation",
    "balance_share_vs_own",
]
REMEDIATION_LOG = [
    {
        "finding": "Fairness gate failed at the 10% alert-budget stress point: legitimate Student-segment "
                   "transactions flagged 1.34x the overall rate (p=0.006).",
        "root_cause": "SHAP gap analysis: amount_usd and amount_to_balance raised risk for small absolute "
                      "amounts — the model learned that the low-amount region holds fraud, making absolute "
                      "amount a proxy for the Student segment.",
        "remediation": "Monotone-increasing constraints on amount_usd, log_amount_usd and amount_to_balance "
                       "(a larger amount can never lower risk).",
        "retest": "Student ratio at 10% budget fell to 1.12x; fairness gate passes; OOT ROC-AUC 0.9566 -> 0.9581.",
        "superseded_model": "bti-v3-hgb-20260923160131",
        "reassessment": "Re-tested under the pooled, corrected method (2026-09-24): the Student disparity was "
                        "confined to the out-of-time window (1.34x there, 0.97x in the calibration window; pooled "
                        "1.20x, q=0.44), so it would now sit on the watchlist rather than fail the gate. The "
                        "monotone constraints are kept on their own merits: a larger amount should never lower risk.",
    },
]
GATES = {"min_oot_roc_auc": 0.75, "max_oot_ece": 0.02, "max_feature_single_auc": 0.97}
SEGMENT_COLUMNS = ["channel", "country", "customer_segment", "customer_age_band"]

REGULATORY_MAPPING = [
    {"framework": "US Federal Reserve SR 11-7 / OCC 2011-12", "scope": "Model risk management"},
    {"framework": "UK PRA SS1/23", "scope": "Model risk management principles for banks"},
    {"framework": "EU AI Act", "scope": "Annex III 5(b) exempts AI used to detect financial fraud from the "
                                        "high-risk credit-scoring category; GDPR Art. 22 still applies to "
                                        "solely automated decisions"},
    {"framework": "MAS FEAT Principles (Singapore)", "scope": "Fairness, ethics, accountability, transparency"},
    {"framework": "HKMA high-level principles on AI", "scope": "Governance, explainability, data quality"},
    {"framework": "CBUAE Model Management Standards", "scope": "Model lifecycle governance (UAE)"},
]

LIMITATIONS = [
    "Trained on synthetic data; performance must be re-established on the bank's own labelled history "
    "before production use.",
    "FX amounts use indicative reference rates unless overridden with the bank's official rate feed.",
    "The synthetic customer base has ~6 transactions per customer, so velocity features are sparse; "
    "their contribution will be larger on real transaction streams.",
    "Labels are taken at face value; production labels need a maturity window (chargebacks arrive 30-120 "
    "days after the transaction) before performance is measured.",
    "historical_average_transaction_amount is a profile field supplied by the source system and must be "
    "maintained point-in-time to avoid look-ahead.",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).resolve().parents[2]).decode().strip()
    except Exception:
        return None


class FeedUnavailableError(RuntimeError):
    """A feature set needs a data feed (location, payee, security events) the training data does not carry."""


MIN_FEED_COVERAGE = 0.01


def default_security_events_path() -> Optional[Path]:
    path = Path(get_settings().processed_data_dir) / "security_events.csv"
    return path if path.exists() else None


def default_data_path() -> Path:
    settings = get_settings()
    clean = Path(settings.processed_data_dir) / "banking_transactions_clean.csv"
    return clean if clean.exists() else Path(settings.raw_data_path)


@dataclass
class TrainingData:
    """Everything candidates share: data, features, labels and the out-of-time split."""
    path: Path
    sha256: str
    df: pd.DataFrame
    features: pd.DataFrame
    y: np.ndarray
    tr: np.ndarray
    ca: np.ndarray
    te: np.ndarray
    es_fit: np.ndarray       # training rows used to fit
    es_val: np.ndarray       # most recent training rows, used only for early stopping
    cut_calib: pd.Timestamp
    cut_test: pd.Timestamp
    source_audit: list
    lookback_days: int
    feature_version: int = FEATURE_VERSION
    security_events: Optional[pd.DataFrame] = None
    security_events_sha256: Optional[str] = None
    feeds: Optional[Dict[str, Dict[str, float]]] = None


def prepare(data_path: Optional[Path] = None, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
            feature_version: int = FEATURE_VERSION, security_events_path: Optional[Path] = None) -> TrainingData:
    assert_feature_lineage()
    data_path = Path(data_path or default_data_path())
    events_path = security_events_path or (default_security_events_path() if data_path == default_data_path() else None)
    events = pd.read_csv(events_path) if events_path else None
    log.info("Loading training data", extra={"path": str(data_path)})
    df = pd.read_csv(data_path, low_memory=False)
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce").fillna(0).astype(int)
    df["_ts"] = event_timestamps(df)
    df = df.dropna(subset=["_ts"]).sort_values("_ts", kind="stable").reset_index(drop=True)
    cut_calib, cut_test = df["_ts"].quantile(0.60), df["_ts"].quantile(0.75)
    tr = (df["_ts"] < cut_calib).to_numpy()
    tr_idx = np.flatnonzero(tr)
    es_start = tr_idx[int(len(tr_idx) * (1 - EARLY_STOP_SHARE))]
    es_val = tr & (np.arange(len(df)) >= es_start)
    features = build_features(df, lookback_days=lookback_days, feature_version=feature_version,
                              security_events=events)
    ca = ((df["_ts"] >= cut_calib) & (df["_ts"] < cut_test)).to_numpy()
    te = (df["_ts"] >= cut_test).to_numpy()
    return TrainingData(
        path=data_path, sha256=_sha256(data_path), df=df, features=features,
        y=df[LABEL].to_numpy(), tr=tr, ca=ca, te=te, es_fit=tr & ~es_val, es_val=es_val,
        cut_calib=cut_calib, cut_test=cut_test,
        source_audit=leakage_audit(df.drop(columns=["_ts"]), LABEL), lookback_days=lookback_days,
        feature_version=feature_version, security_events=events,
        security_events_sha256=_sha256(Path(events_path)) if events_path else None,
        feeds={w: feed_coverage(features[m]) for w, m in (("train", tr), ("calibration", ca), ("out_of_time", te))},
    )


def assess_fairness(data: TrainingData, p: np.ndarray) -> Dict:
    """Fairness on the out-of-sample windows at the reference threshold and the 5% / 10% stress budgets."""
    attrs = [c for c in PROTECTED_ATTRIBUTES if c in data.df.columns and c not in ("geography", "city")]
    thresholds = {"reference": threshold_for_alert_rate(p[data.ca], REFERENCE_ALERT_BUDGET),
                  "stress_5pct_budget": threshold_for_alert_rate(p[data.ca], 0.05),
                  "stress_10pct_budget": threshold_for_alert_rate(p[data.ca], 0.10)}
    return fairness_assessment(data.y, p, thresholds, {"out_of_time": data.te, "calibration": data.ca},
                               {c: data.df[c] for c in attrs})


def check_feeds(data: TrainingData, feature_set: str) -> None:
    """Refuse a feature set whose data feeds are absent or too thin in any window of the split."""
    for feed in feeds_required(FEATURE_SETS[feature_set]):
        thin = {w: c[feed] for w, c in (data.feeds or {}).items() if c[feed] < MIN_FEED_COVERAGE}
        if thin or not data.feeds:
            raise FeedUnavailableError(f"Feature set '{feature_set}' needs the {feed} feed, which covers "
                                       f"{thin} of rows (minimum {MIN_FEED_COVERAGE:.0%} in every window)")


def _model_id(algorithm: str, feature_set: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base = f"bti-v3-{algorithms.SHORT[algorithm]}{FEATURE_SET_SUFFIX[feature_set]}-{stamp}"
    model_id, n = base, 1
    while (registry.registry_dir() / model_id).exists():
        n += 1
        model_id = f"{base}-{n}"
    return model_id


def train_candidate(data: TrainingData, algorithm: str = "hgb", feature_set: str = "core",
                    params: Optional[Dict] = None, developer: Optional[str] = None, register: bool = True,
                    tuning: Optional[Dict] = None, auto_challenger: bool = True) -> Dict:
    names = FEATURE_SETS[feature_set]
    specs = feature_specs(feature_set)
    check_feeds(data, feature_set)
    params = dict(params or algorithms.DEFAULT_PARAMS[algorithm])
    df, features, y, tr, ca, te = data.df, data.features, data.y, data.tr, data.ca, data.te

    encodings = fit_category_encodings(features[tr], y[tr])
    X = to_model_matrix(features, encodings, names)
    X.loc[tr, CATEGORICAL_FEATURES] = out_of_fold_category_encoding(features[tr], y[tr]).to_numpy()

    engineered_audit = leakage_audit(pd.concat([X.loc[tr], df.loc[tr, [LABEL]]], axis=1), LABEL,
                                     threshold=GATES["max_feature_single_auc"])

    monotone = [1 if n in MONOTONE_INCREASING else 0 for n in names]
    estimator = algorithms.fit(algorithm, params, monotone, X[data.es_fit], y[data.es_fit],
                               X[data.es_val], y[data.es_val])
    raw = estimator.predict_proba(X)[:, 1]
    calibrator = PlattCalibrator().fit(raw[ca], y[ca])
    p = calibrator.predict(raw)

    amount_usd = features["amount_usd"].to_numpy()
    accounts = df.get("account_id", df["customer_id"]).to_numpy()
    threshold = threshold_for_alert_rate(p[ca], REFERENCE_ALERT_BUDGET)
    metrics = {
        "train": classification_metrics(y[tr], p[tr]),
        "calibration": classification_metrics(y[ca], p[ca]),
        "out_of_time": classification_metrics(y[te], p[te]),
        "out_of_time_uncalibrated": classification_metrics(y[te], raw[te]),
    }
    oot = {
        "reference_operating_point": {
            "chosen_on": "calibration window",
            "alert_budget": REFERENCE_ALERT_BUDGET,
            **operating_point(y[te], p[te], threshold, amount_usd[te], accounts[te]),
        },
        "alert_budgets": alert_budget_table(y[te], p[te], amounts=amount_usd[te], accounts=accounts[te]),
        "lift_table": lift_table(y[te], p[te], amounts=amount_usd[te]),
        "segments": {c: segment_performance(y[te], p[te], df.loc[te, c], threshold)
                     for c in SEGMENT_COLUMNS if c in df.columns},
    }
    fairness = assess_fairness(data, p)

    legacy = {}
    legacy_scores = _legacy_scores(df.loc[te])
    if legacy_scores is not None:
        legacy = {"note": "Legacy composite weights label-derived risk_score at 35%; figures are not valid "
                          "evidence of performance and are shown only to document the leakage.",
                  **classification_metrics(y[te], legacy_scores / 100.0)}

    importance = _global_importance(estimator, X[te])
    if not importance:
        raise RuntimeError("SHAP explanations failed the additivity check — the model cannot be registered")

    leaking_features = [r["column"] for r in engineered_audit if r["suspected_leak"]]
    gates = [
        {"gate": "feature_lineage", "passed": True, "detail": "All model features trace to pre-authorisation fields"},
        {"gate": "no_single_feature_leak", "passed": not leaking_features,
         "detail": f"Features above {GATES['max_feature_single_auc']} single-feature AUC: {leaking_features or 'none'}"},
        {"gate": "oot_discrimination", "passed": metrics["out_of_time"].get("roc_auc", 0) >= GATES["min_oot_roc_auc"],
         "detail": f"OOT ROC-AUC {metrics['out_of_time'].get('roc_auc')} vs minimum {GATES['min_oot_roc_auc']}"},
        {"gate": "oot_calibration", "passed": metrics["out_of_time"].get("ece", 1) <= GATES["max_oot_ece"],
         "detail": f"OOT ECE {metrics['out_of_time'].get('ece')} vs maximum {GATES['max_oot_ece']}"},
        {"gate": "fairness", "passed": fairness["status"] == "pass",
         "detail": f"{len(fairness['findings'])} false-positive-rate disparities significant on the pooled "
                   f"out-of-sample windows and present in each ({len(fairness['watchlist'])} on the "
                   f"non-gating watchlist)"},
    ]
    status = "passed" if all(g["passed"] for g in gates) else "failed"

    created = datetime.now(timezone.utc)
    model_id = _model_id(algorithm, feature_set)
    card = {
        "model_id": model_id,
        "model_family": "BTI v3 transaction fraud model",
        "created_at": created.isoformat(),
        "ownership": {"developer": developer or getpass.getuser(), "business_owner": None,
                      "validator": None},
        "purpose": {
            "intended_use": "Real-time probability that a payment or account transaction is fraudulent, "
                            "used to approve, step-up, review or decline before funds move.",
            "out_of_scope": ["Credit underwriting or credit-limit decisions",
                             "AML transaction monitoring / SAR decisions (separate typologies and regulation)",
                             "Sole basis for account closure without human review"],
            "decision_type": "Decision support with automated action under documented policy thresholds",
        },
        "regulatory_mapping": REGULATORY_MAPPING,
        "data": {
            "path": str(data.path),
            "sha256": data.sha256,
            "security_events_sha256": data.security_events_sha256,
            "feed_coverage": data.feeds,
            "rows": int(len(df)),
            "fraud_rate": round(float(y.mean()), 5),
            "window": [str(df["_ts"].min()), str(df["_ts"].max())],
            "split": {"method": "out_of_time",
                      "train": {"to": str(data.cut_calib), "rows": int(tr.sum()), "fraud": int(y[tr].sum()),
                                "early_stopping_rows": int(data.es_val.sum())},
                      "calibration": {"from": str(data.cut_calib), "to": str(data.cut_test),
                                      "rows": int(ca.sum()), "fraud": int(y[ca].sum())},
                      "out_of_time_test": {"from": str(data.cut_test), "rows": int(te.sum()),
                                           "fraud": int(y[te].sum())}},
            "fx": {"reporting_currency": "USD", "rates_as_of": fx.RATES_AS_OF, "rates": fx.RATES_TO_USD},
        },
        "features": {
            "lookback_days": data.lookback_days,
            "feature_set": feature_set,
            "feature_version": data.feature_version,
            "model_features": [{**asdict(f), "kind": f.kind.value} for f in specs],
            "monotone_increasing": [n for n in MONOTONE_INCREASING if n in names],
            "excluded_source_fields": [
                {"field": f.name, "classification": f.availability.value, "reason": f.note}
                for f in SOURCE_FIELDS.values()
                if f.availability not in (Availability.PRE_AUTH, Availability.IDENTIFIER)
            ],
            "source_leakage_audit": [r for r in data.source_audit if r["single_feature_auc"] >= 0.6],
            "engineered_leakage_audit": engineered_audit,
            "global_importance": importance,
            "category_encodings": encodings,
        },
        "methodology": {
            "algorithm": algorithms.DESCRIPTION[algorithm],
            "algorithm_key": algorithm,
            "calibration": {**calibrator.describe(), "fitted_on": "calibration window"},
            "categorical_encoding": "Smoothed fraud rate per category (m=50), out-of-fold on the training "
                                    "window; applied as a fitted parameter at inference",
            "hyperparameters": params,
            "early_stopping": f"{algorithms.EARLY_STOPPING_ROUNDS} rounds on the most recent "
                              f"{EARLY_STOP_SHARE:.0%} of the training window",
            "iterations_used": algorithms.iterations_used(estimator),
            "tuning": tuning,
            "threshold_policy": f"Reference threshold = {REFERENCE_ALERT_BUDGET:.0%} alert budget on the "
                                "calibration window",
            "reference_threshold": threshold,
        },
        "performance": {"metrics": metrics, **oot, "legacy_benchmark": legacy},
        "fairness": fairness,
        "monitoring": {
            "baseline_window": "calibration",
            "baseline": build_baseline(features.loc[ca, names], p[ca], CATEGORICAL_FEATURES),
            "plan": {
                "population_stability": "Score PSI and per-feature CSI weekly; investigate >= 0.10, "
                                        "escalate >= 0.25",
                "performance": "ROC-AUC, PR-AUC, VDR on labels older than the 90-day maturity window, monthly",
                "calibration": "Recalibrate when OOT ECE exceeds 0.02 on matured labels",
                "fairness": "Re-run false-positive-rate parity quarterly",
                "revalidation": "Full independent revalidation annually or on material change",
            },
        },
        "validation": {"status": status, "gates": gates, "thresholds": GATES,
                       "remediation_log": REMEDIATION_LOG},
        "limitations": LIMITATIONS,
        "provenance": {"git_sha": _git_sha(), "python": platform.python_version(),
                       "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                       **_library_versions(algorithm)},
    }
    artifact = {"estimator": estimator, "calibrator": calibrator, "encodings": encodings,
                "feature_names": names, "feeds": feeds_required(names), "lookback_days": data.lookback_days,
                "algorithm": algorithm,
                "feature_version": data.feature_version,
                "reference_threshold": threshold, "model_id": model_id}

    if register:
        registry.save_model(model_id, artifact, card)
        if auto_challenger and (registry.model_for_role("challenger") is None or status == "passed"):
            registry.assign_role(model_id, "challenger", approver="bti.modeling.train",
                                 rationale="Newly trained model enters shadow (challenger) mode automatically")
        log.info("Model registered", extra={"model_id": model_id, "validation": status,
                                             "oot_auc": metrics["out_of_time"].get("roc_auc")})
    return card


def train(data_path: Optional[Path] = None, developer: Optional[str] = None,
          lookback_days: int = DEFAULT_LOOKBACK_DAYS, register: bool = True, algorithm: str = "hgb",
          feature_set: str = "core", params: Optional[Dict] = None) -> Dict:
    return train_candidate(prepare(data_path, lookback_days), algorithm, feature_set, params, developer, register)


def _library_versions(algorithm: str) -> Dict[str, str]:
    if algorithm == "lightgbm":
        import lightgbm
        return {"lightgbm": lightgbm.__version__}
    if algorithm == "xgboost":
        import xgboost
        return {"xgboost": xgboost.__version__}
    return {}


def _legacy_scores(test_rows: pd.DataFrame) -> Optional[np.ndarray]:
    if "final_risk_score" in test_rows.columns:
        return pd.to_numeric(test_rows["final_risk_score"], errors="coerce").fillna(0).to_numpy() / 100.0
    scored = Path(get_settings().processed_data_dir) / "banking_transactions_ml_scored.csv"
    if not scored.exists() or "transaction_id" not in test_rows.columns:
        return None
    legacy = pd.read_csv(scored, usecols=["transaction_id", "final_risk_score"]).set_index("transaction_id")
    values = test_rows["transaction_id"].map(legacy["final_risk_score"])
    return pd.to_numeric(values, errors="coerce").fillna(0).to_numpy() / 100.0


def _global_importance(estimator, X: pd.DataFrame, sample: int = 2000) -> list:
    try:
        import shap
        Xs = X.sample(min(sample, len(X)), random_state=0)
        explainer = shap.TreeExplainer(estimator)
        values = algorithms.shap_matrix(explainer, Xs)
        additivity = algorithms.shap_additivity_error(estimator, explainer, Xs)
        if additivity > 1e-3:
            raise RuntimeError(f"SHAP additivity error {additivity:.4g} — explanations would be unreliable")
        mean_abs = np.abs(values).mean(axis=0)
        total = mean_abs.sum() or 1.0
        return sorted([{"feature": f, "mean_abs_shap": round(float(v), 5), "share": round(float(v / total), 4)}
                       for f, v in zip(X.columns, mean_abs)], key=lambda r: -r["mean_abs_shap"])
    except Exception as exc:
        log.warning(f"Global SHAP importance unavailable: {exc}")
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register a BTI v3 fraud model")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--developer", default=None)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--algorithm", choices=algorithms.ALGORITHMS, default="hgb")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="core")
    args = parser.parse_args()
    card = train(args.data, args.developer, args.lookback_days, algorithm=args.algorithm,
                 feature_set=args.feature_set)
    m = card["performance"]["metrics"]["out_of_time"]
    op = card["performance"]["reference_operating_point"]
    print(f"Registered {card['model_id']}  validation={card['validation']['status']}")
    print(f"OOT  ROC-AUC {m.get('roc_auc')}  PR-AUC {m.get('pr_auc')}  KS {m.get('ks')}  ECE {m.get('ece')}")
    print(f"At {op['alert_budget']:.0%} alert budget: precision {op['precision']}  TDR {op['tdr']}  "
          f"VDR {op.get('vdr')}  FP ratio {op['false_positive_ratio']}:1")
    for g in card["validation"]["gates"]:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['gate']}: {g['detail']}")


if __name__ == "__main__":
    main()
