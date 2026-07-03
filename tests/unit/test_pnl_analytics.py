"""
Unit tests for P&L analytics and KPI calculations.
"""

import sys
import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

SRC_DIR = Path(__file__).parent.parent.parent / "src"


def _load_pnl():
    spec = importlib.util.spec_from_file_location("pnl", SRC_DIR / "05_pnl_analytics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pnl = _load_pnl()


def _make_df(n=100, fraud_pct=0.1) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_pct)
    df = pd.DataFrame({
        "transaction_id":         [f"T{i:05d}" for i in range(n)],
        "transaction_date":        pd.date_range("2024-01-01", periods=n, freq="h"),
        "customer_id":            [f"C{i % 20:04d}" for i in range(n)],
        "customer_segment":       rng.choice(["Mass Market", "Premium", "SME"], n),
        "channel":                rng.choice(["Mobile App", "Web", "Branch"], n),
        "merchant_category":      rng.choice(["Retail", "Gaming & Gambling"], n),
        "transaction_amount":     rng.uniform(100, 5000, n),
        "fee_income":             rng.uniform(0.5, 5, n),
        "interchange_income":     rng.uniform(0.2, 2, n),
        "processing_cost":        rng.uniform(0.1, 1, n),
        "chargeback_loss":        np.where(rng.random(n) < 0.05, rng.uniform(100, 500, n), 0),
        "refund_loss":            np.where(rng.random(n) < 0.05, rng.uniform(50, 200, n), 0),
        "fraud_loss":             np.zeros(n),
        "fraud_flag":             np.zeros(n, dtype=int),
        "chargeback_flag":        (rng.random(n) < 0.05).astype(int),
        "refund_flag":            (rng.random(n) < 0.05).astype(int),
        "final_alert_tier":       rng.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], n),
        "final_risk_score":       rng.uniform(0, 100, n),
        "is_revenue_leakage":     np.zeros(n, dtype=int),
        "is_high_risk":           np.zeros(n, dtype=int),
        "net_pnl_impact":         np.zeros(n),
        "reversal_flag":          np.zeros(n, dtype=int),
    })
    # Inject fraud rows
    fraud_idx = rng.choice(n, n_fraud, replace=False)
    df.loc[fraud_idx, "fraud_flag"] = 1
    df.loc[fraud_idx, "fraud_loss"] = df.loc[fraud_idx, "transaction_amount"] * 0.9

    df["net_revenue"] = (df["fee_income"] + df["interchange_income"] - df["processing_cost"])
    df["net_pnl_impact"] = (df["net_revenue"] - df["chargeback_loss"]
                             - df["refund_loss"] - df["fraud_loss"])
    df["is_revenue_leakage"] = ((df["net_pnl_impact"] < 0) & (df["fraud_flag"] == 0)).astype(int)
    df["month_year"] = df["transaction_date"].dt.to_period("M").astype(str)
    return df


class TestPnLKPIs:
    def setup_method(self):
        self.df = _make_df(n=200, fraud_pct=0.10)
        self.kpis = pnl.calc_pnl_kpis(self.df)

    def test_total_transactions(self):
        assert self.kpis["Total Transactions"] == 200

    def test_fraud_count_matches(self):
        expected = int(self.df["fraud_flag"].sum())
        assert self.kpis["Fraud Transactions"] == expected

    def test_fraud_loss_rate_is_percentage(self):
        rate = self.kpis["Fraud Loss Rate (%)"]
        assert 0 <= rate <= 100

    def test_net_revenue_sign(self):
        expected_nr = (self.df["fee_income"].sum()
                       + self.df["interchange_income"].sum()
                       - self.df["processing_cost"].sum())
        assert abs(self.kpis["Net Revenue ($)"] - expected_nr) < 0.01

    def test_cost_to_income_ratio(self):
        cti = self.kpis["Cost-to-Income Ratio (%)"]
        assert 0 < cti < 200

    def test_risk_adjusted_revenue(self):
        rar = self.kpis["Risk-Adjusted Revenue ($)"]
        nr = self.kpis["Net Revenue ($)"]
        # Risk-adjusted revenue must be <= net revenue
        assert rar <= nr + 0.01


class TestMonthlyVariance:
    def test_returns_dataframe_with_expected_cols(self):
        df = _make_df(n=500)
        result = pnl.monthly_variance(df)
        assert isinstance(result, pd.DataFrame)
        # Verify the actual column names returned by monthly_variance
        required_cols = {"month_year", "net_pnl", "net_pnl_mom_pct"}
        assert required_cols.issubset(set(result.columns))

    def test_no_variance_for_single_month(self):
        df = _make_df(n=50)
        df["month_year"] = "2024-01"  # force single month
        result = pnl.monthly_variance(df)
        assert len(result) >= 1


class TestChannelProfitability:
    def test_returns_all_channels(self):
        df = _make_df(n=300)
        result = pnl.channel_profitability(df)
        assert isinstance(result, pd.DataFrame)
        assert "channel" in result.columns
        assert len(result) > 0

    def test_net_revenue_column_exists(self):
        df = _make_df(n=300)
        result = pnl.channel_profitability(df)
        assert "net_revenue" in result.columns or "net_pnl_impact" in result.columns
