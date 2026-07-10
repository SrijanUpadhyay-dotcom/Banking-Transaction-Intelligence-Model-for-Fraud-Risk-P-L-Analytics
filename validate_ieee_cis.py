"""
IEEE-CIS Fraud Detection Dataset Validation
Maps IEEE-CIS columns to BTI model features, scores 20k transactions,
and evaluates against ground-truth isFraud labels.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
MODELS_DIR  = PROJECT_DIR / "models"
DATA_FILE   = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path("/root/.claude/uploads/cb39af70-98e8-5cee-91c6-7f66a7b03c55/d638486b-train_transaction_sample.csv")
)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading BTI models …")
iso_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")
iso_scaler = joblib.load(MODELS_DIR / "iso_scaler.joblib")
lr_model   = joblib.load(MODELS_DIR / "logistic_regression.joblib")
lr_scaler  = joblib.load(MODELS_DIR / "lr_scaler.joblib")
rf_model   = joblib.load(MODELS_DIR / "random_forest.joblib")
le_dict    = joblib.load(MODELS_DIR / "label_encoders.joblib")

with open(MODELS_DIR / "manifest.json") as f:
    manifest = json.load(f)
FEATURE_COLS = manifest["feature_cols"]

# Show label encoder classes
print("\nLabel encoder categories:")
for col, enc in le_dict.items():
    print(f"  {col}: {list(enc.classes_)}")

# ── Load IEEE-CIS sample ──────────────────────────────────────────────────────
print(f"\nLoading IEEE-CIS sample from {DATA_FILE} …")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"  Shape: {df.shape}")
print(f"  Fraud rate: {df['isFraud'].mean()*100:.2f}% ({df['isFraud'].sum()} fraud / {len(df)} total)")

ground_truth = df["isFraud"].values

# ── Feature mapping ───────────────────────────────────────────────────────────
print("\nBuilding BTI feature matrix from IEEE-CIS columns …")

feat = pd.DataFrame(index=df.index)

# ── Direct / derived numerics ─────────────────────────────────────────────────

feat["transaction_amount"] = df["TransactionAmt"]

# account_balance_before / after — not in IEEE-CIS; use median proxy
median_amt = df["TransactionAmt"].median()
feat["account_balance_before"] = median_amt * 10
feat["account_balance_after"]  = feat["account_balance_before"] - feat["transaction_amount"]

# failed_attempt_count — C5 as proxy; clip to 0-10
feat["failed_attempt_count"] = df["C5"].fillna(0).clip(0, 10).astype(int)

# login_attempts — C1 as proxy
feat["login_attempts"] = df["C1"].fillna(1).clip(1, 20).astype(int)

# risk_score — proxy from C6 (count of credit cards seen) * 10
feat["risk_score"] = (df["C6"].fillna(0) * 10).clip(0, 80)

# historical_average_transaction_amount — per-card median
feat["historical_average_transaction_amount"] = (
    df.groupby("card1")["TransactionAmt"].transform("median").fillna(median_amt)
)

# monthly_customer_transaction_count — per-card count in sample
feat["monthly_customer_transaction_count"] = (
    df.groupby("card1")["TransactionID"].transform("count").fillna(1)
)

# ── P&L columns (derived from amount) ────────────────────────────────────────
is_credit = df["card6"].fillna("debit").str.lower().str.contains("credit")
ic_rate   = np.where(is_credit, 0.0175, 0.005)

feat["fee_income"]         = feat["transaction_amount"] * 0.005
feat["interchange_income"] = feat["transaction_amount"] * ic_rate
feat["processing_cost"]    = feat["transaction_amount"] * 0.003 + 0.30
feat["chargeback_loss"]    = 0.0
feat["refund_loss"]        = 0.0
feat["fraud_loss"]         = 0.0
feat["net_revenue"]        = feat["fee_income"] + feat["interchange_income"] - feat["processing_cost"]
feat["net_pnl_impact"]     = feat["net_revenue"]

# ── Ratio ─────────────────────────────────────────────────────────────────────
hist_avg = feat["historical_average_transaction_amount"].replace(0, np.nan)
feat["amount_vs_hist_avg_ratio"] = (feat["transaction_amount"] / hist_avg).fillna(1.0)

# ── Time features ─────────────────────────────────────────────────────────────
hour_of_day = (df["TransactionDT"] % 86400) // 3600
feat["is_off_hours"] = ((hour_of_day < 6) | (hour_of_day > 22)).astype(int)

# ── Flags (not available in IEEE-CIS — default 0) ────────────────────────────
feat["reversal_flag"]   = 0
feat["refund_flag"]     = 0
feat["chargeback_flag"] = 0

# ── Fraud rule score (derived from available IEEE-CIS signals) ────────────────
r01 = (feat["amount_vs_hist_avg_ratio"] >= 5).astype(int)           # high value vs avg
r03 = (feat["failed_attempt_count"] >= 3).astype(int)               # failed auth
r04 = (feat["login_attempts"] >= 4).astype(int)                     # multiple logins
r05 = feat["is_off_hours"]                                           # off hours
r06 = (df["addr2"].fillna(87) != 87).astype(int)                    # non-domestic addr
z_scores = (feat["transaction_amount"] - feat["transaction_amount"].mean()) / (feat["transaction_amount"].std() + 1e-9)
r10 = (z_scores.abs() > 3.5).astype(int)                            # amount outlier
r12 = (feat["risk_score"] >= 60).astype(int)                        # high risk score
r16 = (feat["monthly_customer_transaction_count"] > 15).astype(int) # velocity

raw_score    = r01*9 + r03*8 + r04*7 + r05*8 + r06*7 + r10*7 + r12*6 + r16*7
total_weight = 9 + 8 + 7 + 8 + 7 + 7 + 6 + 7   # 59
feat["fraud_rule_score"] = (raw_score / total_weight * 100).clip(0, 100)
feat["rules_triggered"]  = r01 + r03 + r04 + r05 + r06 + r10 + r12 + r16

# ── Categorical encoding ──────────────────────────────────────────────────────
def safe_encode(series, enc):
    classes = set(enc.classes_)
    return series.map(lambda x: enc.transform([x])[0] if x in classes else len(enc.classes_))

# ProductCD → merchant_category
productcd_map = {
    "W": "Other",
    "H": "Travel & Airlines",
    "C": "Online Marketplaces",
    "S": "Retail / General Merchandise",
    "R": "Financial Services / Money Transfer",
}
enc_merch     = le_dict["merchant_category"]
merchant_cats = df["ProductCD"].map(productcd_map).fillna("Other")
feat["merchant_category_enc"] = safe_encode(merchant_cats, enc_merch)

# channel_enc: default "Mobile Banking"
enc_channel = le_dict["channel"]
default_ch  = "Mobile Banking" if "Mobile Banking" in enc_channel.classes_ else enc_channel.classes_[0]
feat["channel_enc"] = enc_channel.transform([default_ch])[0]

# customer_segment_enc: default "Retail"
enc_seg     = le_dict["customer_segment"]
default_seg = "Retail" if "Retail" in enc_seg.classes_ else enc_seg.classes_[0]
feat["customer_segment_enc"] = enc_seg.transform([default_seg])[0]

# transaction_type_enc: card6 (credit/debit) → Purchase or Card Not Present
enc_txtype = le_dict["transaction_type"]
tx_types   = df["card6"].fillna("debit").map(
    lambda x: "Card Not Present" if "credit" in str(x).lower() else "Purchase"
)
feat["transaction_type_enc"] = safe_encode(tx_types, enc_txtype)

# authorization_method_enc: default "PIN"
enc_auth     = le_dict["authorization_method"]
default_auth = "PIN" if "PIN" in enc_auth.classes_ else enc_auth.classes_[0]
feat["authorization_method_enc"] = enc_auth.transform([default_auth])[0]

# ── Assemble feature matrix ───────────────────────────────────────────────────
X = feat[FEATURE_COLS].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Any NaN remaining: {X.isnull().any().any()}")

# ── Score with all 3 models ───────────────────────────────────────────────────
print("\nScoring with BTI models …")

X_iso        = iso_scaler.transform(X)
iso_scores   = iso_forest.decision_function(X_iso)
iso_pred     = (iso_forest.predict(X_iso) == -1).astype(int)

X_lr    = lr_scaler.transform(X)
lr_prob = lr_model.predict_proba(X_lr)[:, 1]
lr_pred = (lr_prob >= 0.5).astype(int)

rf_prob = rf_model.predict_proba(X)[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)

ensemble_pred = ((iso_pred + lr_pred + rf_pred) >= 2).astype(int)
iso_norm      = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
ensemble_prob = (lr_prob + rf_prob + iso_norm) / 3

# ── Evaluate ──────────────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score, average_precision_score

y = ground_truth

def evaluate(name, y_true, y_pred, y_prob=None):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    spec      = tn / (tn + fp + 1e-9)
    auc       = roc_auc_score(y_true, y_prob)   if y_prob is not None else None
    aucpr     = average_precision_score(y_true, y_prob) if y_prob is not None else None

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Confusion Matrix:  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Specificity: {spec:.4f}")
    if auc  is not None: print(f"  ROC-AUC   : {auc:.4f}")
    if aucpr is not None: print(f"  PR-AUC    : {aucpr:.4f}")
    return dict(model=name, precision=precision, recall=recall, f1=f1,
                roc_auc=auc, pr_auc=aucpr, tp=tp, fp=fp, tn=tn, fn=fn)

print(f"\n{'='*55}")
print(f"  VALIDATION RESULTS vs IEEE-CIS Ground Truth")
print(f"  Dataset: {len(df):,} transactions | Fraud rate: {y.mean()*100:.2f}%")
print(f"{'='*55}")

results = [
    evaluate("Isolation Forest",    y, iso_pred,      iso_norm),
    evaluate("Logistic Regression", y, lr_pred,       lr_prob),
    evaluate("Random Forest",       y, rf_pred,       rf_prob),
    evaluate("Ensemble (2-of-3)",   y, ensemble_pred, ensemble_prob),
]

print(f"\n{'='*55}")
print(f"  SUMMARY TABLE")
print(f"{'='*55}")
print(f"  {'Model':<25} {'Precision':>9} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print(f"  {'-'*25} {'-'*9} {'-'*8} {'-'*8} {'-'*9}")
for r in results:
    auc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] is not None else "  N/A  "
    print(f"  {r['model']:<25} {r['precision']:>9.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {auc_str:>9}")

print(f"\n{'='*55}")
print(f"  RF FRAUD PROBABILITY DISTRIBUTION")
print(f"{'='*55}")
rf_fraud = rf_prob[y == 1]
rf_legit = rf_prob[y == 0]
print(f"  Known fraud (n={len(rf_fraud):,}):  mean={rf_fraud.mean():.4f}  median={np.median(rf_fraud):.4f}  p75={np.percentile(rf_fraud,75):.4f}")
print(f"  Legit txns  (n={len(rf_legit):,}): mean={rf_legit.mean():.4f}  median={np.median(rf_legit):.4f}  p75={np.percentile(rf_legit,75):.4f}")

print(f"\n  RF Threshold sensitivity:")
print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged':>8}")
for thresh in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    p  = (rf_prob >= thresh).astype(int)
    tp = ((p == 1) & (y == 1)).sum()
    fp = ((p == 1) & (y == 0)).sum()
    fn = ((p == 0) & (y == 1)).sum()
    pr = tp / (tp + fp + 1e-9)
    rc = tp / (tp + fn + 1e-9)
    f1 = 2*pr*rc/(pr+rc+1e-9)
    print(f"  {thresh:>10.1f} {pr:>10.4f} {rc:>8.4f} {f1:>8.4f} {int(p.sum()):>8,}")

print(f"\n{'='*55}")
print("  Validation complete.")
print(f"{'='*55}\n")
