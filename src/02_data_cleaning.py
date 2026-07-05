"""
Banking Transaction Intelligence Model
Part 2: Data Cleaning & Validation Layer
Implements banking-grade data quality checks, standardisation, and enrichment.
"""

import pandas as pd
import numpy as np
import os

def load_raw_data(path="data/raw/banking_transactions_raw.csv"):
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    print(f"Loaded raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def run_quality_report(df):
    print("\n── Data Quality Report ──────────────────────────────────────")
    total = len(df)
    report = []
    for col in df.columns:
        nulls   = df[col].isna().sum()
        nullpct = nulls / total * 100
        dtype   = str(df[col].dtype)
        n_unique = df[col].nunique()
        report.append({"column": col, "dtype": dtype, "nulls": nulls,
                        "null_pct": round(nullpct, 2), "unique_values": n_unique})
    qdf = pd.DataFrame(report)
    issues = qdf[qdf["null_pct"] > 0]
    print(f"  Total columns with nulls: {len(issues)}")
    print(qdf[["column", "dtype", "nulls", "null_pct"]].to_string(index=False))
    return qdf


def clean_and_validate(df):
    print("\n── Cleaning & Validation ────────────────────────────────────")

    # 1. Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_id"])
    print(f"  Duplicate transaction_ids removed : {before - len(df)}")

    # 2. Enforce correct dtypes
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").clip(0, 100)

    # 3. Clamp negative amounts (shouldn't exist for purchases; keep for credits/reversals)
    neg_mask = (df["transaction_amount"] < 0) & (df["debit_credit_flag"] == "Debit")
    df.loc[neg_mask, "transaction_amount"] = df.loc[neg_mask, "transaction_amount"].abs()
    print(f"  Negative debit amounts fixed     : {neg_mask.sum()}")

    # 4. Fill known nulls
    df["fraud_type"] = df["fraud_type"].fillna("None")
    df["failed_attempt_count"] = df["failed_attempt_count"].fillna(0).astype(int)
    df["login_attempts"] = df["login_attempts"].fillna(1).astype(int)

    # 5. Balance consistency check – flag rows where balance movement is wrong
    df["balance_delta"] = (df["account_balance_after"] - df["account_balance_before"]).round(2)
    df["balance_check"] = np.where(
        df["debit_credit_flag"] == "Debit",
        np.abs(df["balance_delta"] + df["transaction_amount"]) < 1,   # should decrease
        np.abs(df["balance_delta"] - df["transaction_amount"]) < 1    # should increase
    )
    balance_errors = (~df["balance_check"]).sum()
    print(f"  Balance inconsistency flags       : {balance_errors}")

    # 6. Date range validation
    date_errors = df[
        (df["transaction_date"] < pd.Timestamp("2023-01-01")) |
        (df["transaction_date"] > pd.Timestamp("2024-12-31"))
    ]
    print(f"  Out-of-range dates                : {len(date_errors)}")

    # 7. Risk score imputation (nulls → median)
    median_risk = df["risk_score"].median()
    df["risk_score"] = df["risk_score"].fillna(median_risk).astype(int)

    print(f"\n  ✓ Clean dataset shape: {df.shape}")
    return df


def enrich_data(df):
    print("\n── Feature Enrichment ───────────────────────────────────────")

    # Temporal features
    df["transaction_year"]    = df["transaction_date"].dt.year
    df["transaction_month"]   = df["transaction_date"].dt.month
    df["transaction_quarter"] = df["transaction_date"].dt.quarter
    df["transaction_week"]    = df["transaction_date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]         = df["transaction_date"].dt.day_name()
    df["is_weekend"]          = df["transaction_date"].dt.dayofweek >= 5

    # Hour extraction from time string
    df["transaction_hour"] = df["transaction_time"].str[:2].astype(int, errors="ignore")
    df["is_off_hours"] = df["transaction_hour"].apply(
        lambda h: 1 if (h < 6 or h > 22) else 0
    )

    # Transaction size bands
    df["amount_band"] = pd.cut(
        df["transaction_amount"],
        bins=[0, 100, 500, 2_000, 10_000, 50_000, float("inf")],
        labels=["Micro (<100)", "Small (100–500)", "Medium (500–2K)",
                "Large (2K–10K)", "High Value (10K–50K)", "Very High (50K+)"]
    )

    # Anomaly ratio: how much the transaction deviates from customer historical average
    df["amount_vs_hist_avg_ratio"] = (
        df["transaction_amount"] / df["historical_average_transaction_amount"].replace(0, np.nan)
    ).round(4)

    # Composite risk flag: combines multiple signals
    df["is_high_risk"] = (
        (df["risk_score"] >= 60) |
        (df["fraud_flag"] == 1) |
        (df["chargeback_flag"] == 1) |
        (df["amount_vs_hist_avg_ratio"] > 5)
    ).astype(int)

    # Revenue leakage flag
    df["is_revenue_leakage"] = (
        (df["net_pnl_impact"] < 0) &
        (df["fraud_flag"] == 0)
    ).astype(int)

    # Month-Year label for aggregations
    df["month_year"] = df["transaction_date"].dt.to_period("M").astype(str)

    new_cols = ['transaction_year','transaction_month','transaction_quarter',
                'day_of_week','is_weekend','is_off_hours','amount_band',
                'amount_vs_hist_avg_ratio','is_high_risk','is_revenue_leakage','month_year']
    print(f"  New columns added: {new_cols}")
    print(f"  ✓ Enriched dataset shape: {df.shape}")
    return df


def save_clean_data(df, path="data/processed/banking_transactions_clean.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  ✓ Clean data saved to {path}")


if __name__ == "__main__":
    df_raw   = load_raw_data()
    run_quality_report(df_raw)
    df_clean = clean_and_validate(df_raw)
    df_clean = enrich_data(df_clean)
    save_clean_data(df_clean)
    print("\nData cleaning complete.")
