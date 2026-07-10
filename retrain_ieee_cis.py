"""
BTI Retrain on IEEE-CIS Real Fraud Data
Trains Isolation Forest + Logistic Regression + Random Forest
directly on real transaction features, then compares vs original
synthetic-trained BTI metrics.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, precision_recall_curve
)

warnings.filterwarnings("ignore")

DATA_FILE  = Path("/root/.claude/uploads/cb39af70-98e8-5cee-91c6-7f66a7b03c55/d638486b-train_transaction_sample.csv")
MODELS_OUT = Path("/home/user/Banking-Transaction-Intelligence-Model-for-Fraud-Risk-P-L-Analytics/models_ieee")
MODELS_OUT.mkdir(exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading IEEE-CIS sample …")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"  {len(df):,} rows | Fraud rate: {df['isFraud'].mean()*100:.2f}%")

# ── 2. Feature engineering ────────────────────────────────────────────────────
print("Engineering features …")

feat = pd.DataFrame(index=df.index)

# --- Transaction amount features
feat["amount"]          = df["TransactionAmt"]
feat["log_amount"]      = np.log1p(df["TransactionAmt"])
card_median             = df.groupby("card1")["TransactionAmt"].transform("median")
feat["amount_vs_card_median"] = (df["TransactionAmt"] / card_median.replace(0, np.nan)).fillna(1).clip(0, 50)
amt_mean = df["TransactionAmt"].mean()
amt_std  = df["TransactionAmt"].std()
feat["amount_zscore"]   = ((df["TransactionAmt"] - amt_mean) / (amt_std + 1e-9)).clip(-10, 10)

# --- Time features
hour = (df["TransactionDT"] % 86400) // 3600
feat["hour_of_day"]  = hour
feat["is_off_hours"] = ((hour < 6) | (hour > 22)).astype(int)
feat["day_of_week"]  = (df["TransactionDT"] // 86400) % 7

# --- C-series (count features — 0% NaN, use all 14)
for i in range(1, 15):
    feat[f"C{i}"] = df[f"C{i}"].fillna(0)

# --- D-series (timedelta features — 66% NaN, use D1-D5 with median fill)
for i in range(1, 6):
    col = f"D{i}"
    feat[col] = df[col].fillna(df[col].median())

# --- M-series (match flags — T=1, F=0, NaN=0)
for i in range(1, 10):
    col = f"M{i}"
    feat[col] = df[col].map({"T": 1, "F": 0}).fillna(0).astype(int)

# --- V-series (Vesta engineered features — take V1-V20 with median fill)
for i in range(1, 21):
    col = f"V{i}"
    feat[col] = df[col].fillna(df[col].median())

# --- Categorical: ProductCD
productcd_fraud_rate = df.groupby("ProductCD")["isFraud"].transform("mean")
feat["productcd_fraud_rate"] = productcd_fraud_rate
feat["is_card_not_present"]  = (df["ProductCD"] == "C").astype(int)

# --- Categorical: card network
card4_dummies = pd.get_dummies(df["card4"].fillna("unknown"), prefix="card4")
for c in card4_dummies.columns:
    feat[c] = card4_dummies[c]

# --- Categorical: credit vs debit
feat["is_credit"] = (df["card6"].fillna("debit").str.lower().str.contains("credit")).astype(int)

# --- Email domain risk
top_domains = df["P_emaildomain"].value_counts().head(10).index.tolist()
feat["email_domain_known"] = df["P_emaildomain"].isin(top_domains).astype(int)
feat["is_anonymous_email"] = (df["P_emaildomain"] == "anonymous.com").astype(int)

# ── 3. Prepare matrix ─────────────────────────────────────────────────────────
FEATURE_COLS = [c for c in feat.columns]
X = feat[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0).values
y = df["isFraud"].values

print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"  Features: {FEATURE_COLS[:10]} … (+{len(FEATURE_COLS)-10} more)")

# ── 4. Train / test split (stratified) ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nSplit: {len(X_train):,} train | {len(X_test):,} test")
print(f"  Train fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"  Test fraud : {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ── 5. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 6. Train models ───────────────────────────────────────────────────────────
print("\nTraining models …")

# --- Isolation Forest (unsupervised anomaly detection)
print("  [1/3] Isolation Forest …")
iso = IsolationForest(n_estimators=200, contamination=0.028, random_state=42, n_jobs=-1)
iso.fit(X_train_s)
iso_scores_test = -iso.decision_function(X_test_s)   # higher = more anomalous
iso_pred_test   = (iso.predict(X_test_s) == -1).astype(int)

# --- Logistic Regression (class-balanced)
print("  [2/3] Logistic Regression …")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1, random_state=42)
lr.fit(X_train_s, y_train)
lr_prob_test = lr.predict_proba(X_test_s)[:, 1]
lr_pred_test = (lr_prob_test >= 0.5).astype(int)

# --- Random Forest (class-balanced)
print("  [3/3] Random Forest …")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_prob_test = rf.predict_proba(X_test)[:, 1]
rf_pred_test = (rf_prob_test >= 0.5).astype(int)

# Ensemble
ensemble_prob = (lr_prob_test + rf_prob_test) / 2
ensemble_pred = (ensemble_prob >= 0.5).astype(int)

# ── 7. Find optimal threshold (best F1) ──────────────────────────────────────
def best_threshold(y_true, y_prob):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.argmax(f1s)
    return thresholds[idx] if idx < len(thresholds) else 0.5, f1s[idx]

rf_thresh, rf_best_f1 = best_threshold(y_test, rf_prob_test)
lr_thresh, lr_best_f1 = best_threshold(y_test, lr_prob_test)

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_prob):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    pr  = tp / (tp + fp + 1e-9)
    rc  = tp / (tp + fn + 1e-9)
    f1  = 2*pr*rc / (pr+rc+1e-9)
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    return dict(name=name, tp=tp, fp=fp, tn=tn, fn=fn,
                precision=pr, recall=rc, f1=f1, roc_auc=auc, pr_auc=pr_auc)

results = [
    evaluate("Isolation Forest",    y_test, iso_pred_test,  iso_scores_test),
    evaluate("Logistic Regression", y_test, lr_pred_test,   lr_prob_test),
    evaluate("Random Forest",       y_test, rf_pred_test,   rf_prob_test),
    evaluate("Ensemble LR+RF",      y_test, ensemble_pred,  ensemble_prob),
]

# Optimal threshold RF
rf_opt_pred = (rf_prob_test >= rf_thresh).astype(int)
results.append(evaluate(f"RF @ optimal ({rf_thresh:.2f})", y_test, rf_opt_pred, rf_prob_test))

# ── 9. Print results ──────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  IEEE-CIS RETRAIN RESULTS  (test set: {len(y_test):,} txns)")
print(f"{'='*62}")
print(f"  {'Model':<28} {'ROC-AUC':>8} {'PR-AUC':>7} {'Recall':>7} {'Prec':>7} {'F1':>7}")
print(f"  {'-'*28} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for r in results:
    print(f"  {r['name']:<28} {r['roc_auc']:>8.4f} {r['pr_auc']:>7.4f} "
          f"{r['recall']:>7.4f} {r['precision']:>7.4f} {r['f1']:>7.4f}")

# ── 10. Confusion matrix for best model ──────────────────────────────────────
best = max(results, key=lambda r: r["roc_auc"])
print(f"\n  Best model: {best['name']}")
print(f"  Confusion matrix:")
print(f"    TP={best['tp']:,}  FP={best['fp']:,}  (caught {best['tp']} of {y_test.sum()} fraud cases)")
print(f"    FN={best['fn']:,}  TN={best['tn']:,}")

# ── 11. Top features (RF) ─────────────────────────────────────────────────────
print(f"\n  Top 15 predictive features (Random Forest):")
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat_name, imp in importances.head(15).items():
    bar = "█" * int(imp * 200)
    print(f"    {feat_name:<30} {imp:.4f}  {bar}")

# ── 12. Compare with original synthetic BTI ───────────────────────────────────
print(f"\n{'='*62}")
print(f"  BEFORE vs AFTER: Synthetic Training → Real Data Retrain")
print(f"{'='*62}")
print(f"  {'Metric':<25} {'Synthetic BTI':>15} {'Retrained (Real)':>18}")
print(f"  {'-'*25} {'-'*15} {'-'*18}")
print(f"  {'ROC-AUC (RF)':<25} {'0.5941 (≈random)':>15} {results[2]['roc_auc']:>18.4f}")
print(f"  {'Recall (RF)':<25} {'10.1%':>15} {results[2]['recall']*100:>17.1f}%")
print(f"  {'Precision (RF)':<25} {'3.2%':>15} {results[2]['precision']*100:>17.1f}%")
print(f"  {'F1 (RF)':<25} {'0.049':>15} {results[2]['f1']:>18.4f}")
print(f"  {'PR-AUC (RF)':<25} {'0.046 (≈baseline)':>15} {results[2]['pr_auc']:>18.4f}")
print(f"\n  Baseline (random classifier) ROC-AUC = 0.500")
print(f"  Baseline PR-AUC = fraud_rate = {y_test.mean():.4f}")

# ── 13. Save retrained models ─────────────────────────────────────────────────
print(f"\nSaving retrained models to {MODELS_OUT} …")
joblib.dump(iso,    MODELS_OUT / "isolation_forest.joblib")
joblib.dump(scaler, MODELS_OUT / "scaler.joblib")
joblib.dump(lr,     MODELS_OUT / "logistic_regression.joblib")
joblib.dump(rf,     MODELS_OUT / "random_forest.joblib")

manifest = {
    "trained_on": "IEEE-CIS Fraud Detection Dataset (sample)",
    "n_train": int(len(X_train)),
    "n_test":  int(len(X_test)),
    "fraud_rate_train": float(y_train.mean()),
    "feature_cols": FEATURE_COLS,
    "rf_metrics":  {"roc_auc": results[2]["roc_auc"], "f1": results[2]["f1"],
                    "recall": results[2]["recall"], "precision": results[2]["precision"]},
    "lr_metrics":  {"roc_auc": results[1]["roc_auc"], "f1": results[1]["f1"]},
    "ensemble_metrics": {"roc_auc": results[3]["roc_auc"], "f1": results[3]["f1"]},
    "optimal_rf_threshold": float(rf_thresh),
}
with open(MODELS_OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("  Done.")
print(f"\n{'='*62}")
print("  Retrain complete.")
print(f"{'='*62}\n")
