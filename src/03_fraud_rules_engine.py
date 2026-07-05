"""
Banking Transaction Intelligence Model
Part 3: Rule-Based Fraud & Anomaly Detection Engine
Implements 19 banking fraud detection rules. Each rule is tagged, scored, and
appended to the transaction record. Final composite fraud_rule_score is output.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# RULE DEFINITIONS
# Each rule: name, weight (0–10), logic function
# ─────────────────────────────────────────────────────────────────────────────

def rule_high_value_vs_avg(df):
    """R01: Transaction ≥ 5× customer historical average."""
    ratio = df["transaction_amount"] / df["historical_average_transaction_amount"].replace(0, np.nan)
    return (ratio >= 5).astype(int), ratio.round(4)

def rule_velocity_spike(df):
    """R02: Customer daily transaction count > 10."""
    daily_counts = df.groupby(["customer_id", "transaction_date"])["transaction_id"].transform("count")
    return (daily_counts > 10).astype(int), daily_counts

def rule_failed_auth_attempts(df):
    """R03: ≥ 3 failed authentication attempts on a single transaction."""
    return (df["failed_attempt_count"] >= 3).astype(int), df["failed_attempt_count"]

def rule_multiple_login_attempts(df):
    """R04: ≥ 4 login attempts before transaction."""
    return (df["login_attempts"] >= 4).astype(int), df["login_attempts"]

def rule_off_hours_transaction(df):
    """R05: Transaction between 00:00–05:59 local time."""
    hour = df["transaction_time"].str[:2].astype(int, errors="ignore").fillna(12)
    return (hour < 6).astype(int), hour

def rule_geography_mismatch(df):
    """R06: Device IP country differs from account registered country (proxy: off-hours + high risk)."""
    suspicious_ip_pattern = df["ip_location"].str.startswith(("192.", "10.", "172.")).fillna(False)
    return (~suspicious_ip_pattern & (df["risk_score"] > 40)).astype(int), df["risk_score"]

def rule_repeated_merchant(df):
    """R07: Same customer transacts with the same merchant ≥ 5× on the same day."""
    daily_merch = df.groupby(["customer_id", "transaction_date", "merchant_name"])["transaction_id"].transform("count")
    return (daily_merch >= 5).astype(int), daily_merch

def rule_abnormal_refund(df):
    """R08: Refund amount > original average + refund in flagged category."""
    high_risk_cats = ["Gaming & Gambling", "Crypto Exchanges", "Financial Services / Money Transfer",
                      "Luxury Goods", "Online Marketplaces"]
    cond = (df["refund_flag"] == 1) & (
        (df["transaction_amount"] > df["historical_average_transaction_amount"] * 2) |
        (df["merchant_category"].isin(high_risk_cats))
    )
    return cond.astype(int), df["refund_flag"]

def rule_chargeback_heavy_merchant(df):
    """R09: Merchant category with historically high chargeback rates."""
    high_cb_cats = ["Gaming & Gambling", "Crypto Exchanges", "Travel & Airlines",
                    "Luxury Goods", "Financial Services / Money Transfer"]
    return (df["merchant_category"].isin(high_cb_cats) & (df["chargeback_flag"] == 1)).astype(int), df["chargeback_flag"]

def rule_amount_outlier_zscore(df):
    """R10: Z-score of transaction amount > 3.5 within customer segment."""
    df["_z"] = df.groupby("customer_segment")["transaction_amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    return (df["_z"].abs() > 3.5).astype(int), df["_z"].round(3)

def rule_balance_inconsistency(df):
    """R11: Account balance after debit transaction is negative."""
    return (
        (df["debit_credit_flag"] == "Debit") &
        (df["account_balance_after"] < 0)
    ).astype(int), df["account_balance_after"]

def rule_device_ip_mismatch(df):
    """R12: Transaction device_id differs from customer's typical device (proxy via unknown prefix)."""
    # In the synthetic dataset, fraud transactions got random device IDs
    known_prefix = "DEV-"
    return (
        (df["risk_score"] >= 60) &
        df["device_id"].str.startswith(known_prefix, na=False)
    ).astype(int), df["device_id"]

def rule_high_risk_segment_high_value(df):
    """R13: Student or NRI/Diaspora segment with transaction > 5,000."""
    return (
        df["customer_segment"].isin(["Student", "NRI / Diaspora"]) &
        (df["transaction_amount"] > 5_000)
    ).astype(int), df["transaction_amount"]

def rule_channel_fraud_concentration(df):
    """R14: Flagged transaction channels — USSD, API, Call Centre — with high amount."""
    risky_channels = ["USSD", "API/Open Banking", "Call Centre"]
    return (
        df["channel"].isin(risky_channels) &
        (df["transaction_amount"] > df["historical_average_transaction_amount"] * 2)
    ).astype(int), df["channel"]

def rule_duplicate_transaction(df):
    """R15: Same customer, same amount, same merchant within 10 minutes (same date proxy)."""
    df["_dup_key"] = (df["customer_id"].astype(str) + "_" +
                      df["transaction_amount"].astype(str) + "_" +
                      df["merchant_name"] + "_" +
                      df["transaction_date"].astype(str))
    dup_counts = df.groupby("_dup_key")["transaction_id"].transform("count")
    return (dup_counts > 1).astype(int), dup_counts

def rule_rapid_sequential_transactions(df):
    """R16: Customer makes > 5 transactions on a single day (velocity rule)."""
    daily_count = df.groupby(["customer_id", "transaction_date"])["transaction_id"].transform("count")
    return (daily_count > 5).astype(int), daily_count

def rule_cross_border_suspicious(df):
    """R17: Cross-border transaction + high risk score + unusual merchant category."""
    cross_border_cats = ["Crypto Exchanges", "Financial Services / Money Transfer",
                         "Gaming & Gambling", "Luxury Goods"]
    return (
        df["merchant_category"].isin(cross_border_cats) &
        (df["risk_score"] > 50) &
        (df["transaction_amount"] > 1_000)
    ).astype(int), df["risk_score"]

def rule_high_refund_to_sale(df):
    """R18: Merchant has refund_flag ratio > 30% of all its transactions."""
    merch_total  = df.groupby("merchant_name")["transaction_id"].transform("count")
    merch_refund = df.groupby("merchant_name")["refund_flag"].transform("sum")
    ratio = (merch_refund / merch_total.replace(0, np.nan)).fillna(0)
    return (ratio > 0.30).astype(int), ratio.round(4)

def rule_high_chargeback_ratio(df):
    """R19: Customer chargeback_flag ratio > 10% of all transactions."""
    cust_total = df.groupby("customer_id")["transaction_id"].transform("count")
    cust_cb    = df.groupby("customer_id")["chargeback_flag"].transform("sum")
    ratio = (cust_cb / cust_total.replace(0, np.nan)).fillna(0)
    return (ratio > 0.10).astype(int), ratio.round(4)


# Rule registry: (function, weight, short_name)
RULES = [
    (rule_high_value_vs_avg,            9, "R01_high_value_vs_avg"),
    (rule_velocity_spike,               8, "R02_velocity_spike"),
    (rule_failed_auth_attempts,         7, "R03_failed_auth"),
    (rule_multiple_login_attempts,      6, "R04_multi_login"),
    (rule_off_hours_transaction,        5, "R05_off_hours"),
    (rule_geography_mismatch,           7, "R06_geo_mismatch"),
    (rule_repeated_merchant,            6, "R07_repeated_merchant"),
    (rule_abnormal_refund,              8, "R08_abnormal_refund"),
    (rule_chargeback_heavy_merchant,    8, "R09_cb_heavy_merchant"),
    (rule_amount_outlier_zscore,        9, "R10_amount_outlier"),
    (rule_balance_inconsistency,        9, "R11_balance_inconsistency"),
    (rule_device_ip_mismatch,           8, "R12_device_ip_mismatch"),
    (rule_high_risk_segment_high_value, 7, "R13_segment_high_value"),
    (rule_channel_fraud_concentration,  6, "R14_channel_concentration"),
    (rule_duplicate_transaction,        9, "R15_duplicate"),
    (rule_rapid_sequential_transactions,6, "R16_rapid_sequential"),
    (rule_cross_border_suspicious,      8, "R17_cross_border"),
    (rule_high_refund_to_sale,          7, "R18_refund_ratio"),
    (rule_high_chargeback_ratio,        8, "R19_chargeback_ratio"),
]


def apply_fraud_rules(df):
    print("Applying 19 fraud detection rules...")
    df = df.copy()
    rule_flags = []

    for rule_fn, weight, name in RULES:
        flag, _ = rule_fn(df)
        df[name] = flag
        rule_flags.append((name, weight))

    # Composite weighted fraud rule score (0–100)
    total_weight = sum(w for _, w in rule_flags)
    df["fraud_rule_score"] = 0.0
    for name, weight in rule_flags:
        df["fraud_rule_score"] += df[name] * weight
    df["fraud_rule_score"] = (df["fraud_rule_score"] / total_weight * 100).round(2)

    # Rules triggered count
    flag_cols = [n for n, _ in rule_flags]
    df["rules_triggered"] = df[flag_cols].sum(axis=1).astype(int)

    # Suspicious flag: ≥ 3 rules triggered or fraud_rule_score > 40
    df["is_suspicious"] = (
        (df["rules_triggered"] >= 3) | (df["fraud_rule_score"] > 40)
    ).astype(int)

    # Alert tier
    df["alert_tier"] = pd.cut(
        df["fraud_rule_score"],
        bins=[-1, 20, 40, 60, 80, 101],
        labels=["Green", "Yellow", "Orange", "Red", "Critical"]
    )

    # Clean up temp columns
    for col in ["_z", "_dup_key"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    suspicious = df["is_suspicious"].sum()
    print(f"  Suspicious transactions flagged : {suspicious:,}  ({suspicious/len(df)*100:.1f}%)")
    print(f"  Alert tier breakdown:")
    print(df["alert_tier"].value_counts().to_string())
    return df


def generate_rules_summary(df):
    """Return a summary DataFrame showing how many transactions each rule flagged."""
    rule_names = [n for _, _, n in RULES]
    summary = []
    for name in rule_names:
        if name in df.columns:
            n_flagged = df[name].sum()
            summary.append({
                "rule": name,
                "transactions_flagged": int(n_flagged),
                "pct_of_total": round(n_flagged / len(df) * 100, 2),
            })
    return pd.DataFrame(summary).sort_values("transactions_flagged", ascending=False)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/banking_transactions_clean.csv",
                     parse_dates=["transaction_date"])
    df = apply_fraud_rules(df)
    df.to_csv("data/processed/banking_transactions_flagged.csv", index=False)

    summary = generate_rules_summary(df)
    print("\nRule Summary:")
    print(summary.to_string(index=False))
    print("\nFraud rules engine complete.")
