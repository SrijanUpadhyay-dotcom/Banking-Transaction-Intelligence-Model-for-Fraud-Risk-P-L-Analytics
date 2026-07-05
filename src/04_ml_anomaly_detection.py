"""
Banking Transaction Intelligence Model
Part 4: Machine Learning Anomaly Detection Layer
Models: Isolation Forest (primary), Logistic Regression, Random Forest, Z-Score baseline.
Outputs: ml_anomaly_score, ml_fraud_prediction, model performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve, f1_score)


os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING FOR ML
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "transaction_amount",
    "account_balance_before",
    "account_balance_after",
    "failed_attempt_count",
    "login_attempts",
    "risk_score",
    "historical_average_transaction_amount",
    "monthly_customer_transaction_count",
    "fee_income",
    "interchange_income",
    "processing_cost",
    "chargeback_loss",
    "refund_loss",
    "fraud_loss",
    "net_revenue",
    "net_pnl_impact",
    "amount_vs_hist_avg_ratio",
    "fraud_rule_score",
    "rules_triggered",
    "is_off_hours",
    "reversal_flag",
    "refund_flag",
    "chargeback_flag",
]

CATEGORICAL_FEATURES = [
    "channel",
    "merchant_category",
    "customer_segment",
    "transaction_type",
    "authorization_method",
]


def prepare_features(df):
    df = df.copy()

    # Encode categorical variables — save each encoder so inference can reuse it
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            enc = LabelEncoder()
            df[col + "_enc"] = enc.fit_transform(df[col].astype(str))
            label_encoders[col] = enc

    encoded_cats = [c + "_enc" for c in CATEGORICAL_FEATURES if c in df.columns]
    feature_cols = NUMERIC_FEATURES + encoded_cats

    # Drop missing
    feature_df = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    return feature_df, feature_cols, label_encoders


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: ISOLATION FOREST (unsupervised anomaly detection)
# ─────────────────────────────────────────────────────────────────────────────

def run_isolation_forest(df, X):
    print("\n── Model 1: Isolation Forest (Unsupervised) ─────────────────")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.06,   # ~6% expected anomaly rate
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_scaled)

    # -1 = anomaly, +1 = normal
    iso_preds  = iso.predict(X_scaled)
    iso_scores = iso.score_samples(X_scaled)  # lower = more anomalous

    df["iso_forest_flag"]  = (iso_preds == -1).astype(int)
    df["iso_forest_score"] = (-iso_scores).round(6)  # invert: higher = more anomalous

    n_flagged = df["iso_forest_flag"].sum()
    print(f"  Transactions flagged as anomalous : {n_flagged:,} ({n_flagged/len(df)*100:.1f}%)")

    # Compare with known fraud labels
    if "fraud_flag" in df.columns:
        overlap = ((df["iso_forest_flag"] == 1) & (df["fraud_flag"] == 1)).sum()
        print(f"  Overlap with known fraud labels  : {overlap:,}")

    return df, iso, scaler


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Z-SCORE BASELINE (per-segment)
# ─────────────────────────────────────────────────────────────────────────────

def run_zscore_detection(df):
    print("\n── Model 2: Z-Score Anomaly Detection ───────────────────────")

    df["z_score_amount"] = df.groupby("customer_segment")["transaction_amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    ).round(4)

    df["z_score_flag"] = (df["z_score_amount"].abs() > 3.0).astype(int)
    n = df["z_score_flag"].sum()
    print(f"  Z-score anomalies (|z| > 3.0)    : {n:,}  ({n/len(df)*100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: LOGISTIC REGRESSION (supervised, uses fraud_flag label)
# ─────────────────────────────────────────────────────────────────────────────

def run_logistic_regression(df, X, feature_cols):
    print("\n── Model 3: Logistic Regression (Supervised) ────────────────")

    y = df["fraud_flag"].values
    X_vals = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_vals, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)

    y_pred  = lr.predict(X_test_s)
    y_proba = lr.predict_proba(X_test_s)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Apply to full dataset
    X_full_s = scaler.transform(X_vals)
    df["lr_fraud_proba"] = lr.predict_proba(X_full_s)[:, 1].round(4)
    df["lr_fraud_flag"]  = (df["lr_fraud_proba"] >= 0.40).astype(int)

    return df, lr, scaler, {"roc_auc": roc, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: RANDOM FOREST (supervised)
# ─────────────────────────────────────────────────────────────────────────────

def run_random_forest(df, X, feature_cols):
    print("\n── Model 4: Random Forest Classifier (Supervised) ──────────")

    y = df["fraud_flag"].values
    X_vals = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_vals, y, test_size=0.25, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Feature importance plot
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_features = importances.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_features.sort_values().plot(kind="barh", ax=ax, color="#1f6b3a")
    ax.set_title("Random Forest – Top 15 Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/charts/rf_feature_importance.png", dpi=150)
    plt.close()
    print("  Chart saved: outputs/charts/rf_feature_importance.png")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Legitimate", "Fraud"]); ax2.set_yticklabels(["Legitimate", "Fraud"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax2.set_title("Random Forest – Confusion Matrix", fontweight="bold")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    fig2.tight_layout()
    fig2.savefig("outputs/charts/rf_confusion_matrix.png", dpi=150)
    plt.close()
    print("  Chart saved: outputs/charts/rf_confusion_matrix.png")

    # Full dataset prediction
    df["rf_fraud_proba"] = rf.predict_proba(X_vals)[:, 1].round(4)
    df["rf_fraud_flag"]  = (df["rf_fraud_proba"] >= 0.40).astype(int)

    return df, rf, {"roc_auc": roc, "f1": f1, "importances": importances}


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE ML SCORE
# ─────────────────────────────────────────────────────────────────────────────

def build_composite_ml_score(df):
    print("\n── Composite ML Anomaly Score ───────────────────────────────")

    # Weighted ensemble
    df["ml_anomaly_score"] = (
        df["iso_forest_score"] * 30 +
        df["lr_fraud_proba"]   * 30 +
        df["rf_fraud_proba"]   * 40
    ).round(4)

    # Normalise 0–100
    mn, mx = df["ml_anomaly_score"].min(), df["ml_anomaly_score"].max()
    df["ml_anomaly_score_norm"] = ((df["ml_anomaly_score"] - mn) / (mx - mn) * 100).round(2)

    # Final ML prediction: any of the 3 models flags it
    df["ml_fraud_prediction"] = (
        (df["iso_forest_flag"] == 1) |
        (df["lr_fraud_flag"]   == 1) |
        (df["rf_fraud_flag"]   == 1)
    ).astype(int)

    n = df["ml_fraud_prediction"].sum()
    print(f"  ML-flagged transactions : {n:,}  ({n/len(df)*100:.1f}%)")

    # Combined final risk score
    df["final_risk_score"] = (
        df["risk_score"] * 0.35 +
        df["fraud_rule_score"] * 0.35 +
        df["ml_anomaly_score_norm"] * 0.30
    ).round(2).clip(0, 100)

    # Final tier
    df["final_alert_tier"] = pd.cut(
        df["final_risk_score"],
        bins=[-1, 25, 50, 70, 85, 101],
        labels=["LOW", "MEDIUM", "HIGH", "VERY HIGH", "CRITICAL"]
    )

    print("\nFinal alert tier distribution:")
    print(df["final_alert_tier"].value_counts().to_string())
    return df


def create_eda_charts(df):
    print("\n── EDA Charts ───────────────────────────────────────────────")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Banking Transaction Intelligence – EDA Overview", fontsize=14, fontweight="bold")

    # 1. Monthly transaction volume
    monthly = df.groupby("month_year")["transaction_id"].count()
    if not monthly.empty:
        monthly.plot(ax=axes[0, 0], kind="bar", color="#1f6b3a", rot=45)
        axes[0, 0].set_title("Monthly Transaction Volume")
        axes[0, 0].set_ylabel("Count"); axes[0, 0].set_xlabel("")

    # 2. Transaction amount distribution
    df["transaction_amount"].clip(upper=df["transaction_amount"].quantile(0.99)).hist(
        bins=50, ax=axes[0, 1], color="#0a4a8a", edgecolor="white"
    )
    axes[0, 1].set_title("Transaction Amount Distribution (99th pct)")
    axes[0, 1].set_xlabel("Amount"); axes[0, 1].set_ylabel("Frequency")

    # 3. Fraud by channel
    fraud_channel = df[df["fraud_flag"] == 1]["channel"].value_counts()
    fraud_channel.plot(ax=axes[0, 2], kind="barh", color="#8b0000", edgecolor="white")
    axes[0, 2].set_title("Fraud by Channel"); axes[0, 2].set_xlabel("Fraud Count")

    # 4. Alert tier pie
    tier_counts = df["final_alert_tier"].value_counts()
    colors = ["#2ca02c", "#ffdd44", "#ff7f0e", "#d62728", "#9467bd"]
    if not tier_counts.empty:
        tier_counts.plot(ax=axes[1, 0], kind="pie", colors=colors,
                         autopct="%1.1f%%", startangle=90)
        axes[1, 0].set_title("Final Alert Tier Distribution"); axes[1, 0].set_ylabel("")

    # 5. Net P&L by segment
    pnl_seg = df.groupby("customer_segment")["net_pnl_impact"].sum()
    pnl_seg.sort_values().plot(ax=axes[1, 1], kind="barh",
                                color=["#d62728" if v < 0 else "#1f6b3a" for v in pnl_seg.sort_values()])
    axes[1, 1].set_title("Net P&L Impact by Customer Segment")
    axes[1, 1].set_xlabel("Net P&L ($)")

    # 6. Risk score distribution
    df["final_risk_score"].hist(bins=30, ax=axes[1, 2], color="#6a0dad", edgecolor="white")
    axes[1, 2].set_title("Final Risk Score Distribution")
    axes[1, 2].set_xlabel("Risk Score (0–100)"); axes[1, 2].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig("outputs/charts/eda_overview.png", dpi=150)
    plt.close()
    print("  Chart saved: outputs/charts/eda_overview.png")


def save_models(iso, iso_scaler, lr, lr_scaler, rf, feature_cols, label_encoders, metrics):
    """Persist trained models and metadata to disk for real-time serving."""
    import joblib, json
    from datetime import datetime
    os.makedirs("models", exist_ok=True)

    joblib.dump(iso,            "models/isolation_forest.joblib")
    joblib.dump(iso_scaler,     "models/iso_scaler.joblib")
    joblib.dump(lr,             "models/logistic_regression.joblib")
    joblib.dump(lr_scaler,      "models/lr_scaler.joblib")
    joblib.dump(rf,             "models/random_forest.joblib")
    joblib.dump(label_encoders, "models/label_encoders.joblib")

    manifest = {
        "trained_at":   datetime.utcnow().isoformat() + "Z",
        "feature_cols": list(feature_cols),
        "lr_metrics":   metrics.get("lr", {}),
        "rf_metrics":   metrics.get("rf", {}),
        "models": {
            "isolation_forest":    "models/isolation_forest.joblib",
            "iso_scaler":          "models/iso_scaler.joblib",
            "logistic_regression": "models/logistic_regression.joblib",
            "lr_scaler":           "models/lr_scaler.joblib",
            "random_forest":       "models/random_forest.joblib",
            "label_encoders":      "models/label_encoders.joblib",
        },
    }
    with open("models/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n── Model Artifacts Saved ─────────────────────────────────")
    for name, path in manifest["models"].items():
        size_kb = os.path.getsize(path) // 1024
        print(f"  {name:<25} → {path} ({size_kb}KB)")
    print(f"  manifest                  → models/manifest.json")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/banking_transactions_flagged.csv",
                     parse_dates=["transaction_date"])

    X, feature_cols, label_encoders = prepare_features(df)

    df, iso, iso_scaler     = run_isolation_forest(df, X)
    df                      = run_zscore_detection(df)
    df, lr, lr_scaler, lr_m = run_logistic_regression(df, X, feature_cols)
    df, rf, rf_m            = run_random_forest(df, X, feature_cols)
    df                      = build_composite_ml_score(df)
    create_eda_charts(df)

    save_models(iso, iso_scaler, lr, lr_scaler, rf, feature_cols, label_encoders,
                metrics={"lr": lr_m, "rf": {k: v for k, v in rf_m.items()
                                             if k != "importances"}})

    df.to_csv("data/processed/banking_transactions_ml_scored.csv", index=False)
    print("\nML anomaly detection complete. File saved.")
