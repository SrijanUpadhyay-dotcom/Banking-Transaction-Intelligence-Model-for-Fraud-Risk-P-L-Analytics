"""
Banking Transaction Intelligence Model
Part 5: FP&A and P&L Analytics Layer
Calculates all P&L KPIs, variance analysis, profitability by dimension, and
revenue leakage. Outputs summary DataFrames and charts for the Excel workbook.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/pnl",    exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# P&L FORMULA LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

def calc_pnl_kpis(df):
    """Compute top-level P&L KPIs for the full dataset."""
    gross_tv       = df["transaction_amount"].sum()
    fee_income     = df["fee_income"].sum()
    int_income     = df["interchange_income"].sum()
    processing_cost= df["processing_cost"].sum()
    refund_loss    = df["refund_loss"].sum()
    chargeback_loss= df["chargeback_loss"].sum()
    fraud_loss     = df["fraud_loss"].sum()
    net_revenue    = df["net_revenue"].sum()
    net_pnl        = df["net_pnl_impact"].sum()

    total_tx       = len(df)
    fraud_tx       = df["fraud_flag"].sum()
    suspicious_tx  = df.get("is_suspicious", pd.Series([0]*len(df))).sum()

    # Formula results
    fraud_loss_rate       = fraud_loss / gross_tv * 100 if gross_tv else 0
    chargeback_ratio      = chargeback_loss / gross_tv * 100 if gross_tv else 0
    revenue_leakage_pct   = abs(min(0, net_pnl)) / fee_income * 100 if fee_income else 0
    cost_to_income_ratio  = processing_cost / (fee_income + int_income) * 100 if (fee_income + int_income) else 0
    risk_adj_revenue      = net_revenue - fraud_loss - chargeback_loss
    suspicious_tx_ratio   = suspicious_tx / total_tx * 100 if total_tx else 0
    refund_loss_ratio     = refund_loss / gross_tv * 100 if gross_tv else 0
    reversal_ratio        = df["reversal_flag"].sum() / total_tx * 100 if total_tx else 0
    exception_rate        = df.get("is_high_risk", pd.Series([0]*len(df))).sum() / total_tx * 100 if total_tx else 0

    return {
        "Gross Transaction Value ($)":  round(gross_tv, 2),
        "Fee Income ($)":               round(fee_income, 2),
        "Interchange Income ($)":       round(int_income, 2),
        "Processing Cost ($)":          round(processing_cost, 2),
        "Refund Loss ($)":              round(refund_loss, 2),
        "Chargeback Loss ($)":          round(chargeback_loss, 2),
        "Fraud Loss ($)":               round(fraud_loss, 2),
        "Net Revenue ($)":              round(net_revenue, 2),
        "Net P&L Impact ($)":           round(net_pnl, 2),
        "Risk-Adjusted Revenue ($)":    round(risk_adj_revenue, 2),
        "Total Transactions":           int(total_tx),
        "Fraud Transactions":           int(fraud_tx),
        "Suspicious Transactions":      int(suspicious_tx),
        "Fraud Loss Rate (%)":          round(fraud_loss_rate, 4),
        "Chargeback Ratio (%)":         round(chargeback_ratio, 4),
        "Revenue Leakage % of Fee":     round(revenue_leakage_pct, 4),
        "Cost-to-Income Ratio (%)":     round(cost_to_income_ratio, 2),
        "Suspicious Tx Ratio (%)":      round(suspicious_tx_ratio, 2),
        "Refund Loss Ratio (%)":        round(refund_loss_ratio, 4),
        "Reversal Ratio (%)":           round(reversal_ratio, 4),
        "Exception Rate (%)":           round(exception_rate, 2),
    }


def monthly_variance(df):
    """Month-on-month P&L variance analysis."""
    df["month_year"] = pd.to_datetime(df["transaction_date"]).dt.to_period("M").astype(str)
    grp = df.groupby("month_year").agg(
        total_transactions=("transaction_id", "count"),
        gross_tv=("transaction_amount", "sum"),
        fee_income=("fee_income", "sum"),
        interchange_income=("interchange_income", "sum"),
        processing_cost=("processing_cost", "sum"),
        fraud_loss=("fraud_loss", "sum"),
        chargeback_loss=("chargeback_loss", "sum"),
        refund_loss=("refund_loss", "sum"),
        net_revenue=("net_revenue", "sum"),
        net_pnl=("net_pnl_impact", "sum"),
        fraud_count=("fraud_flag", "sum"),
    ).reset_index()

    grp = grp.sort_values("month_year")
    grp["net_pnl_mom_variance"] = grp["net_pnl"].diff()
    grp["net_pnl_mom_pct"]      = grp["net_pnl"].pct_change().mul(100).round(2)
    grp["fraud_loss_mom_pct"]   = grp["fraud_loss"].pct_change().mul(100).round(2)
    grp["revenue_mom_pct"]      = grp["net_revenue"].pct_change().mul(100).round(2)
    return grp


def channel_profitability(df):
    return df.groupby("channel").agg(
        transactions=("transaction_id", "count"),
        gross_tv=("transaction_amount", "sum"),
        fee_income=("fee_income", "sum"),
        interchange_income=("interchange_income", "sum"),
        processing_cost=("processing_cost", "sum"),
        fraud_loss=("fraud_loss", "sum"),
        chargeback_loss=("chargeback_loss", "sum"),
        net_revenue=("net_revenue", "sum"),
        net_pnl=("net_pnl_impact", "sum"),
        fraud_count=("fraud_flag", "sum"),
    ).assign(
        fraud_rate=lambda x: (x["fraud_count"] / x["transactions"] * 100).round(2),
        pnl_per_tx=lambda x: (x["net_pnl"] / x["transactions"]).round(4),
    ).reset_index().sort_values("net_pnl", ascending=False)


def customer_segment_profitability(df):
    return df.groupby("customer_segment").agg(
        transactions=("transaction_id", "count"),
        gross_tv=("transaction_amount", "sum"),
        fee_income=("fee_income", "sum"),
        fraud_loss=("fraud_loss", "sum"),
        chargeback_loss=("chargeback_loss", "sum"),
        net_revenue=("net_revenue", "sum"),
        net_pnl=("net_pnl_impact", "sum"),
        fraud_count=("fraud_flag", "sum"),
        avg_risk_score=("risk_score", "mean"),
    ).assign(
        fraud_rate=lambda x: (x["fraud_count"] / x["transactions"] * 100).round(2),
        avg_pnl_per_tx=lambda x: (x["net_pnl"] / x["transactions"]).round(4),
    ).reset_index().sort_values("net_pnl", ascending=False)


def merchant_risk_analysis(df):
    merch = df.groupby(["merchant_name", "merchant_category"]).agg(
        transactions=("transaction_id", "count"),
        gross_tv=("transaction_amount", "sum"),
        fee_income=("fee_income", "sum"),
        chargeback_loss=("chargeback_loss", "sum"),
        refund_loss=("refund_loss", "sum"),
        fraud_loss=("fraud_loss", "sum"),
        net_pnl=("net_pnl_impact", "sum"),
        fraud_count=("fraud_flag", "sum"),
        chargeback_count=("chargeback_flag", "sum"),
        avg_risk=("risk_score", "mean"),
    ).reset_index()

    merch["fraud_rate"]      = (merch["fraud_count"] / merch["transactions"] * 100).round(2)
    merch["chargeback_rate"] = (merch["chargeback_count"] / merch["transactions"] * 100).round(2)
    merch["refund_rate"]     = (merch["refund_loss"] / merch["gross_tv"].replace(0, np.nan) * 100).round(4)
    _risk_labels = ["Low", "Moderate", "Medium", "High", "Critical"]
    _risk_bins = pd.qcut(merch["avg_risk"], q=5, labels=False, duplicates="drop")
    _n_bins = int(_risk_bins.max() + 1) if not _risk_bins.isna().all() else 1
    merch["merchant_risk_rank"] = _risk_bins.map(
        lambda x: _risk_labels[int(x)] if not pd.isna(x) else "Low"
    )
    return merch.sort_values("fraud_loss", ascending=False)


def exception_queue(df):
    """Generate the investigation exception queue — top suspicious transactions."""
    cols = [
        "transaction_id", "customer_id", "transaction_date", "transaction_amount",
        "transaction_type", "channel", "merchant_category", "merchant_name",
        "customer_segment", "fraud_flag", "fraud_type", "risk_score",
        "fraud_rule_score", "ml_anomaly_score_norm", "final_risk_score",
        "final_alert_tier", "rules_triggered", "alert_tier",
        "chargeback_flag", "refund_flag", "reversal_flag",
        "net_pnl_impact", "fraud_loss", "chargeback_loss",
    ]
    avail = [c for c in cols if c in df.columns]
    exc = df[df.get("is_suspicious", df["fraud_flag"]) == 1][avail]
    exc = exc.sort_values("final_risk_score" if "final_risk_score" in exc.columns else "risk_score",
                          ascending=False)
    return exc.head(500)


# ─────────────────────────────────────────────────────────────────────────────
# P&L CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def create_pnl_charts(df):
    print("Creating P&L charts...")

    mv = monthly_variance(df)
    ch = channel_profitability(df)
    sg = customer_segment_profitability(df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Banking Transaction Intelligence – P&L Analytics", fontsize=14, fontweight="bold")

    # 1. Monthly Net P&L
    ax = axes[0, 0]
    colors = ["#1f6b3a" if v >= 0 else "#8b0000" for v in mv["net_pnl"]]
    ax.bar(mv["month_year"], mv["net_pnl"], color=colors, edgecolor="white")
    ax.set_title("Monthly Net P&L Impact ($)")
    ax.set_xlabel("Month"); ax.set_ylabel("Net P&L ($)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # 2. Monthly Fraud Loss vs Net Revenue
    ax = axes[0, 1]
    ax.plot(mv["month_year"], mv["net_revenue"], marker="o", color="#0a4a8a",
            label="Net Revenue", linewidth=2)
    ax.plot(mv["month_year"], mv["fraud_loss"], marker="s", color="#d62728",
            label="Fraud Loss", linewidth=2, linestyle="--")
    ax.set_title("Net Revenue vs Fraud Loss (Monthly)")
    ax.legend(); ax.tick_params(axis="x", rotation=45); ax.grid(alpha=0.3)

    # 3. Channel Net P&L
    ax = axes[0, 2]
    ch_sorted = ch.sort_values("net_pnl")
    bar_colors = ["#1f6b3a" if v >= 0 else "#8b0000" for v in ch_sorted["net_pnl"]]
    ax.barh(ch_sorted["channel"], ch_sorted["net_pnl"], color=bar_colors)
    ax.set_title("Net P&L by Channel"); ax.set_xlabel("Net P&L ($)")
    ax.grid(axis="x", alpha=0.3)

    # 4. Segment profitability
    ax = axes[1, 0]
    seg_colors = ["#1f6b3a" if v >= 0 else "#8b0000" for v in sg["net_pnl"]]
    ax.bar(sg["customer_segment"], sg["net_pnl"], color=seg_colors, edgecolor="white")
    ax.set_title("Net P&L by Customer Segment"); ax.set_ylabel("Net P&L ($)")
    ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)

    # 5. P&L waterfall-style components
    ax = axes[1, 1]
    kpis = calc_pnl_kpis(df)
    components = ["Fee Income ($)", "Interchange Income ($)", "Processing Cost ($)",
                  "Refund Loss ($)", "Chargeback Loss ($)", "Fraud Loss ($)"]
    vals = [kpis[c] for c in components]
    bar_c2 = ["#2ca02c" if c in ["Fee Income ($)", "Interchange Income ($)"] else "#d62728"
               for c in components]
    ax.bar([c.replace(" ($)", "").replace(" &", "\n&") for c in components], vals, color=bar_c2)
    ax.set_title("P&L Component Breakdown"); ax.set_ylabel("Amount ($)")
    ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)

    # 6. MoM Net P&L % Change
    ax = axes[1, 2]
    mom_data = mv.dropna(subset=["net_pnl_mom_pct"])
    bar_cols = ["#1f6b3a" if v >= 0 else "#8b0000" for v in mom_data["net_pnl_mom_pct"]]
    ax.bar(mom_data["month_year"], mom_data["net_pnl_mom_pct"], color=bar_cols, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Month-on-Month Net P&L Variance (%)")
    ax.set_xlabel("Month"); ax.set_ylabel("MoM Change (%)")
    ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("outputs/charts/pnl_dashboard.png", dpi=150)
    plt.close()
    print("  Chart saved: outputs/charts/pnl_dashboard.png")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/banking_transactions_ml_scored.csv",
                     parse_dates=["transaction_date"])

    kpis = calc_pnl_kpis(df)
    print("\n── Top-Level P&L KPIs ───────────────────────────────────────")
    for k, v in kpis.items():
        print(f"  {k:<40} {v:>20,.4f}" if isinstance(v, float) else f"  {k:<40} {v:>20,}")

    mv  = monthly_variance(df)
    ch  = channel_profitability(df)
    sg  = customer_segment_profitability(df)
    mr  = merchant_risk_analysis(df)
    exc = exception_queue(df)

    mv.to_csv("outputs/pnl/monthly_variance.csv",  index=False)
    ch.to_csv("outputs/pnl/channel_profitability.csv", index=False)
    sg.to_csv("outputs/pnl/segment_profitability.csv", index=False)
    mr.to_csv("outputs/pnl/merchant_risk.csv",     index=False)
    exc.to_csv("outputs/pnl/exception_queue.csv",  index=False)

    create_pnl_charts(df)
    print("\nP&L analytics complete.")
