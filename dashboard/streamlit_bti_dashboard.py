"""
BTI Terminal — Banking Transaction Intelligence
Streamlit Interactive Dashboard
Run: streamlit run dashboard/streamlit_bti_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTI Terminal — Banking Transaction Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark terminal theme via custom CSS
st.markdown("""
<style>
  /* Dark background */
  .stApp { background-color: #0a0e1a; color: #e8ecf0; }
  .block-container { padding-top: 1rem; }

  /* Header */
  h1, h2, h3 { color: #f4a261 !important; font-family: "Courier New", monospace; }
  h4, h5, h6 { color: #52b788 !important; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #141d35;
    border: 1px solid #1e2d50;
    border-top: 2px solid #f4a261;
    padding: 12px;
  }
  div[data-testid="metric-container"] label { color: #8892a4 !important; font-size: 10px !important; letter-spacing: 2px; text-transform: uppercase; }
  div[data-testid="stMetricValue"] > div { color: #f4a261 !important; font-family: "Courier New", monospace; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0f1629; border-right: 1px solid #1e2d50; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label,
  section[data-testid="stSidebar"] .stSlider label { color: #8892a4 !important; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }

  /* Tables */
  .stDataFrame { border: 1px solid #1e2d50; }
  div[data-testid="stDataFrame"] { border: 1px solid #2a4080; }

  /* Buttons */
  .stButton > button { background: #1a2442; color: #f4a261; border: 1px solid #2a4080; font-family: "Courier New", monospace; letter-spacing: 1px; }
  .stButton > button:hover { background: #f4a261; color: #0a0e1a; }

  /* Tabs */
  button[data-baseweb="tab"] { color: #8892a4; font-family: "Courier New", monospace; font-size: 11px; letter-spacing: 1px; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #f4a261 !important; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Alert box */
  .bti-alert {
    background: rgba(230,57,70,0.12);
    border: 1px solid #7b1d1d;
    border-left: 3px solid #e63946;
    padding: 8px 14px;
    font-family: "Courier New", monospace;
    font-size: 11px;
    color: #f4a261;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    paths = [
        "data/processed/banking_transactions_ml_scored.csv",
        "data/processed/banking_transactions_flagged.csv",
        "data/processed/banking_transactions_clean.csv",
        "data/raw/banking_transactions_raw.csv",
    ]
    # Try to find the project root
    for root in [".", "..", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
        for path in paths:
            full = os.path.join(root, path)
            if os.path.exists(full):
                df = pd.read_csv(full, parse_dates=["transaction_date"])
                return df

    # Generate sample data if nothing found
    st.warning("⚠ Dataset not found. Generating sample data. Run src/01_data_generation.py for full dataset.")
    return generate_sample_data()


def generate_sample_data(n=5000):
    """Generate a small sample dataset for demo purposes."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-12-31", periods=n)
    channels = ["Mobile Banking","Internet Banking","ATM","Branch","POS Terminal","USSD","API/Open Banking"]
    segments = ["Retail","Premium","Private Banking","SME","Corporate","Student"]
    merchants = ["Crypto Exchanges","Gaming & Gambling","Grocery","Retail","Travel & Airlines","Utilities","Luxury Goods"]

    df = pd.DataFrame({
        "transaction_id": [f"TXN-{str(i).zfill(8)}" for i in range(1,n+1)],
        "customer_id":    [f"CUST-{np.random.randint(1,1001):06d}" for _ in range(n)],
        "transaction_date": dates,
        "transaction_amount": np.abs(np.random.exponential(800, n)).clip(1, 100000),
        "transaction_type": np.random.choice(["Purchase","Transfer","Withdrawal","Payment","Refund"], n),
        "channel":         np.random.choice(channels, n),
        "merchant_category": np.random.choice(merchants, n),
        "customer_segment":  np.random.choice(segments, n),
        "geography":  np.random.choice(["United States","United Kingdom","India","Singapore","UAE"], n),
        "fraud_flag":  np.random.choice([0,1], n, p=[0.95,0.05]),
        "chargeback_flag": np.random.choice([0,1], n, p=[0.99,0.01]),
        "refund_flag": np.random.choice([0,1], n, p=[0.96,0.04]),
        "reversal_flag": np.random.choice([0,1], n, p=[0.97,0.03]),
        "risk_score":  np.random.randint(5,100,n),
        "fee_income":  np.abs(np.random.normal(8,5,n)).clip(0.1,50),
        "interchange_income": np.abs(np.random.normal(15,10,n)).clip(0,100),
        "processing_cost": np.abs(np.random.normal(3,2,n)).clip(0.1,20),
        "fraud_loss": 0.0,
        "chargeback_loss": 0.0,
        "refund_loss": 0.0,
        "net_revenue": 0.0,
        "net_pnl_impact": 0.0,
        "currency": "USD",
        "transaction_status": np.random.choice(["Completed","Failed","Reversed"], n, p=[0.85,0.10,0.05]),
        "debit_credit_flag": np.random.choice(["Debit","Credit"], n, p=[0.85,0.15]),
        "fraud_rule_score": np.random.uniform(0,100,n),
        "final_risk_score": np.random.uniform(0,100,n),
    })

    # Calculate P&L
    df["fraud_loss"]     = df["transaction_amount"] * df["fraud_flag"] * np.random.uniform(0.8,1.0,n)
    df["chargeback_loss"]= df["transaction_amount"] * df["chargeback_flag"] * 1.05
    df["refund_loss"]    = df["transaction_amount"] * df["refund_flag"] * 0.15
    df["net_revenue"]    = df["fee_income"] + df["interchange_income"] - df["processing_cost"]
    df["net_pnl_impact"] = df["net_revenue"] - df["fraud_loss"] - df["chargeback_loss"] - df["refund_loss"]

    df["month_year"] = df["transaction_date"].dt.to_period("M").astype(str)
    df["fraud_rule_score"] = df["fraud_rule_score"].round(2)
    df["final_risk_score"] = df["final_risk_score"].round(2)
    df["is_suspicious"] = ((df["risk_score"] >= 60) | (df["fraud_flag"] == 1)).astype(int)
    df["final_alert_tier"] = pd.cut(df["final_risk_score"], bins=[-1,25,50,70,85,101],
                                     labels=["LOW","MEDIUM","HIGH","VERY HIGH","CRITICAL"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        plot_bgcolor  = "#0f1629",
        paper_bgcolor = "#0a0e1a",
        font          = dict(color="#e8ecf0", family="Courier New"),
        title_font    = dict(color="#f4a261"),
        xaxis=dict(gridcolor="#1e2d50", color="#8892a4"),
        yaxis=dict(gridcolor="#1e2d50", color="#8892a4"),
        legend=dict(bgcolor="#141d35", bordercolor="#2a4080"),
    )
)
COLORS = {
    "green":  "#52b788",
    "amber":  "#f4a261",
    "red":    "#e63946",
    "blue":   "#4895ef",
    "purple": "#b07fef",
    "gold":   "#e9c46a",
}


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(90deg,#050a16,#0f1629,#050a16);
            border-bottom:2px solid #2a4080;padding:12px 20px;
            display:flex;align-items:center;justify-content:space-between;
            margin-bottom:12px">
  <div>
    <span style="font-family:'Courier New',monospace;font-size:28px;
                 font-weight:bold;color:#f4a261;letter-spacing:4px;
                 text-shadow:0 0 20px rgba(244,162,97,0.4)">BTI TERMINAL</span>
    <span style="font-family:'Courier New',monospace;font-size:10px;
                 color:#8892a4;letter-spacing:3px;margin-left:16px">
      BANKING TRANSACTION INTELLIGENCE
    </span>
  </div>
  <div style="font-family:'Courier New',monospace;font-size:9px;color:#8892a4;
              letter-spacing:2px">
    FRAUD | FP&A | P&L | RISK | TX MONITORING
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────

df_full = load_data()

with st.sidebar:
    st.markdown("### ⚙ FILTERS")

    # Date range
    min_date = df_full["transaction_date"].min().date()
    max_date = df_full["transaction_date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Segment filter
    segments_available = sorted(df_full["customer_segment"].dropna().unique().tolist())
    selected_segments = st.multiselect("Customer Segment", segments_available, default=segments_available)

    # Channel filter
    channels_available = sorted(df_full["channel"].dropna().unique().tolist())
    selected_channels = st.multiselect("Channel", channels_available, default=channels_available)

    # Risk score filter
    risk_min, risk_max = st.slider("Risk Score Range", 0, 100, (0, 100))

    # Fraud only toggle
    show_fraud_only = st.checkbox("Show Fraud Transactions Only", value=False)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:9px;color:#8892a4;font-family:'Courier New',monospace;line-height:1.8">
    DATASET: {len(df_full):,} transactions<br>
    PERIOD: {min_date} → {max_date}<br>
    GENERATED: {date.today()}<br>
    STATUS: <span style="color:#52b788">● ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)


# Apply filters
if len(date_range) == 2:
    df = df_full[
        (df_full["transaction_date"] >= pd.Timestamp(date_range[0])) &
        (df_full["transaction_date"] <= pd.Timestamp(date_range[1]))
    ].copy()
else:
    df = df_full.copy()

if selected_segments:
    df = df[df["customer_segment"].isin(selected_segments)]
if selected_channels:
    df = df[df["channel"].isin(selected_channels)]

df = df[(df["risk_score"] >= risk_min) & (df["risk_score"] <= risk_max)]
if show_fraud_only:
    df = df[df["fraud_flag"] == 1]


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE KPIs
# ─────────────────────────────────────────────────────────────────────────────

gtv            = df["transaction_amount"].sum()
net_rev        = df["net_revenue"].sum()
net_pnl        = df["net_pnl_impact"].sum()
fraud_loss     = df["fraud_loss"].sum()
cb_loss        = df["chargeback_loss"].sum()
refund_loss    = df["refund_loss"].sum()
total_tx       = len(df)
fraud_tx       = int(df["fraud_flag"].sum())
fraud_rate     = fraud_tx / total_tx * 100 if total_tx else 0
suspicious     = int(df.get("is_suspicious", df["fraud_flag"]).sum()) if "is_suspicious" in df.columns else fraud_tx
risk_adj_rev   = net_rev - fraud_loss - cb_loss
fee_income_sum = df["fee_income"].sum()
proc_cost_sum  = df["processing_cost"].sum()
int_income_sum = df["interchange_income"].sum()
cti            = proc_cost_sum / (fee_income_sum + int_income_sum) * 100 if (fee_income_sum + int_income_sum) else 0


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📊 F1 OVERVIEW",
    "🚨 F2 FRAUD MONITOR",
    "💰 F3 P&L ANALYTICS",
    "👤 F4 CUSTOMER RISK",
    "🏪 F5 MERCHANT RISK",
    "📡 F6 CHANNEL P&L",
    "⚠️ F7 EXCEPTION QUEUE",
    "📝 F8 ANALYST MEMO",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="bti-alert">⚠ ALERT: Fraud exposure above threshold · Chargeback spike in Gaming/Crypto · 500 transactions in exception queue</div>', unsafe_allow_html=True)
    st.markdown("")

    # KPI Row 1
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("GROSS TX VALUE",       f"${gtv/1e6:,.1f}M",       f"{len(df):,} transactions")
    c2.metric("NET REVENUE",          f"${net_rev/1e6:,.2f}M",   f"Fee + Interchange")
    c3.metric("NET P&L IMPACT",       f"${net_pnl/1e6:,.2f}M",   f"After all losses")
    c4.metric("RISK-ADJ REVENUE",     f"${risk_adj_rev/1e6:,.2f}M", "After Fraud + CB")

    # KPI Row 2
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("FRAUD LOSS",           f"${fraud_loss/1e6:,.2f}M", f"{fraud_rate:.1f}% fraud rate", delta_color="inverse")
    c6.metric("CHARGEBACK LOSS",      f"${cb_loss/1e6:,.2f}M",   f"{fraud_tx:,} fraud txns", delta_color="inverse")
    c7.metric("SUSPICIOUS TXNS",      f"{suspicious:,}",          f"{suspicious/total_tx*100:.1f}% of portfolio")
    c8.metric("COST-TO-INCOME",       f"{cti:.1f}%",              "Processing / Revenue")

    st.markdown("---")

    # Monthly trend + Channel breakdown
    col1, col2 = st.columns(2)

    with col1:
        if "month_year" in df.columns:
            monthly = df.groupby("month_year").agg(
                net_pnl=("net_pnl_impact","sum"),
                net_rev=("net_revenue","sum"),
                fraud_loss=("fraud_loss","sum"),
            ).reset_index().sort_values("month_year")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly["month_year"], y=monthly["net_pnl"],
                name="Net P&L", marker_color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in monthly["net_pnl"]],
            ))
            fig.add_trace(go.Scatter(
                x=monthly["month_year"], y=monthly["net_rev"],
                name="Net Revenue", line=dict(color=COLORS["blue"], width=2),
                mode="lines+markers",
            ))
            fig.update_layout(
                PLOTLY_TEMPLATE["layout"],
                title="Monthly Net P&L & Revenue",
                xaxis_tickangle=45, height=320,
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fraud_by_cat = df[df["fraud_flag"]==1].groupby("merchant_category")["fraud_flag"].count().sort_values(ascending=True)
        fig2 = px.bar(
            fraud_by_cat.reset_index(),
            x="fraud_flag", y="merchant_category",
            orientation="h",
            color_discrete_sequence=[COLORS["red"]],
            labels={"fraud_flag":"Fraud Count","merchant_category":""},
            title="Fraud Count by Merchant Category",
        )
        fig2.update_layout(PLOTLY_TEMPLATE["layout"], height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Alert tier distribution
    col3, col4 = st.columns(2)
    with col3:
        if "final_alert_tier" in df.columns:
            tier_counts = df["final_alert_tier"].value_counts().reset_index()
            tier_counts.columns = ["tier", "count"]
            tier_color_map = {
                "CRITICAL": COLORS["red"], "VERY HIGH": "#c45e20",
                "HIGH": COLORS["amber"],   "MEDIUM": COLORS["blue"],
                "LOW": COLORS["green"],
            }
            fig3 = px.pie(
                tier_counts, values="count", names="tier",
                color="tier", color_discrete_map=tier_color_map,
                title="Transaction Risk Tier Distribution",
                hole=0.4,
            )
            fig3.update_layout(PLOTLY_TEMPLATE["layout"], height=300)
            fig3.update_traces(textinfo="percent+label")
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        seg_pnl = df.groupby("customer_segment")["net_pnl_impact"].sum().sort_values()
        fig4 = px.bar(
            seg_pnl.reset_index(),
            x="net_pnl_impact", y="customer_segment",
            orientation="h",
            color="net_pnl_impact",
            color_continuous_scale=[[0, COLORS["red"]], [0.5, COLORS["amber"]], [1, COLORS["green"]]],
            labels={"net_pnl_impact": "Net P&L ($)", "customer_segment": ""},
            title="Net P&L by Customer Segment",
        )
        fig4.update_layout(PLOTLY_TEMPLATE["layout"], height=300, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: FRAUD MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    fc1,fc2,fc3,fc4 = st.columns(4)
    fc1.metric("FRAUD TRANSACTIONS",  f"{fraud_tx:,}",        f"{fraud_rate:.2f}% rate")
    fc2.metric("FRAUD LOSS ($)",      f"${fraud_loss/1e6:,.2f}M", "Confirmed fraud")
    fc3.metric("CHARGEBACK LOSS",     f"${cb_loss/1e6:,.2f}M",  f"{int(df['chargeback_flag'].sum()):,} CBs")
    fc4.metric("FRAUD LOSS RATE",     f"{fraud_loss/gtv*100:.3f}%", "of Gross TX Value")

    col1, col2 = st.columns(2)

    with col1:
        fraud_channel = df[df["fraud_flag"]==1].groupby("channel")["transaction_amount"].agg(["count","sum"]).reset_index()
        fraud_channel.columns = ["channel","fraud_count","fraud_value"]
        fig = px.bar(fraud_channel.sort_values("fraud_count"), x="fraud_count", y="channel",
                     orientation="h", title="Fraud Transactions by Channel",
                     color_discrete_sequence=[COLORS["red"]])
        fig.update_layout(PLOTLY_TEMPLATE["layout"], height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "fraud_type" in df.columns:
            ft = df[df["fraud_flag"]==1]["fraud_type"].value_counts().head(10)
            fig2 = px.bar(ft.reset_index(), x="count", y="fraud_type",
                          orientation="h", title="Top 10 Fraud Typologies",
                          color_discrete_sequence=[COLORS["amber"]])
            fig2.update_layout(PLOTLY_TEMPLATE["layout"], height=320)
            st.plotly_chart(fig2, use_container_width=True)

    # Risk score distribution
    fig3 = px.histogram(df, x="risk_score", nbins=50, title="Risk Score Distribution",
                        color_discrete_sequence=[COLORS["blue"]])
    fig3.update_layout(PLOTLY_TEMPLATE["layout"], height=250)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: P&L ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    pc1,pc2,pc3,pc4,pc5 = st.columns(5)
    pc1.metric("Gross TX Value",     f"${gtv/1e9:.3f}B")
    pc2.metric("Fee Income",         f"${fee_income_sum/1e6:.2f}M")
    pc3.metric("Interchange",        f"${int_income_sum/1e6:.2f}M")
    pc4.metric("Refund Loss",        f"${refund_loss/1e6:.2f}M")
    pc5.metric("Net Revenue",        f"${net_rev/1e6:.2f}M")

    col1, col2 = st.columns(2)

    with col1:
        if "month_year" in df.columns:
            mv = df.groupby("month_year").agg(
                net_pnl=("net_pnl_impact","sum"),
                fraud_loss=("fraud_loss","sum"),
                cb_loss=("chargeback_loss","sum"),
                refund_loss=("refund_loss","sum"),
            ).reset_index().sort_values("month_year")

            fig = go.Figure()
            for col_name, color, name in [
                ("fraud_loss", COLORS["red"], "Fraud Loss"),
                ("cb_loss", COLORS["amber"], "Chargeback Loss"),
                ("refund_loss", COLORS["purple"], "Refund Loss"),
            ]:
                fig.add_trace(go.Bar(name=name, x=mv["month_year"], y=mv[col_name],
                                     marker_color=color))
            fig.update_layout(PLOTLY_TEMPLATE["layout"], barmode="stack",
                               title="Monthly Loss Breakdown (Stacked)", height=320, xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        ch = df.groupby("channel").agg(
            net_pnl=("net_pnl_impact","sum"),
            gross_tv=("transaction_amount","sum"),
            fraud_loss=("fraud_loss","sum"),
        ).reset_index()

        fig2 = px.scatter(ch, x="gross_tv", y="net_pnl", size="fraud_loss",
                          color="net_pnl", text="channel",
                          title="Channel: Gross TV vs Net P&L (size = Fraud Loss)",
                          color_continuous_scale=[[0,COLORS["red"]],[0.5,COLORS["amber"]],[1,COLORS["green"]]])
        fig2.update_traces(textposition="top center")
        fig2.update_layout(PLOTLY_TEMPLATE["layout"], height=320, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # P&L formula reference
    st.markdown("##### FP&A Formula Reference")
    formulas = {
        "Net Revenue":          "= Fee Income + Interchange Income − Processing Cost",
        "Net P&L Impact":       "= Net Revenue − CB Loss − Refund Loss − Fraud Loss",
        "Fraud Loss Rate (%)":  "= (Fraud Loss / Gross TX Value) × 100",
        "Risk-Adj Revenue":     "= Net Revenue − Fraud Loss − Chargeback Loss",
        "Revenue Leakage %":    "= |MIN(0, Net P&L)| / Fee Income × 100",
        "Cost-to-Income Ratio": "= Processing Cost / (Fee + Interchange) × 100",
        "MoM Variance %":       "= (Current − Prior P&L) / |Prior P&L| × 100",
        "Exception Rate %":     "= High-Risk Txns / Total Txns × 100",
    }
    fdf = pd.DataFrame(list(formulas.items()), columns=["KPI", "Formula"])
    st.dataframe(fdf, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CUSTOMER RISK
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    cust = df.groupby(["customer_id", "customer_segment", "geography"]).agg(
        total_tx=("transaction_id","count") if "transaction_id" in df.columns else ("transaction_amount","count"),
        total_spend=("transaction_amount","sum"),
        fraud_count=("fraud_flag","sum"),
        fraud_loss=("fraud_loss","sum"),
        cb_count=("chargeback_flag","sum"),
        net_pnl=("net_pnl_impact","sum"),
        avg_risk=("risk_score","mean"),
    ).reset_index()

    cust["fraud_rate"] = (cust["fraud_count"] / cust["total_tx"] * 100).round(2)
    cust["risk_tier"]  = pd.cut(cust["avg_risk"], bins=[-1,25,50,70,85,101],
                                 labels=["LOW","MEDIUM","HIGH","VERY HIGH","CRITICAL"])
    cust = cust.sort_values("avg_risk", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        seg_counts = df.groupby("customer_segment").agg(
            txns=("transaction_amount","count"),
            fraud_count=("fraud_flag","sum"),
        ).reset_index()
        seg_counts["fraud_rate"] = seg_counts["fraud_count"] / seg_counts["txns"] * 100

        fig = px.bar(seg_counts, x="customer_segment", y="fraud_rate",
                     color="fraud_rate",
                     color_continuous_scale=[[0,COLORS["green"]],[0.5,COLORS["amber"]],[1,COLORS["red"]]],
                     title="Fraud Rate by Customer Segment (%)")
        fig.update_layout(PLOTLY_TEMPLATE["layout"], height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        geo_fraud = df[df["fraud_flag"]==1].groupby("geography")["fraud_flag"].count().reset_index()
        fig2 = px.bar(geo_fraud.sort_values("fraud_flag"), x="fraud_flag", y="geography",
                      orientation="h", title="Fraud Transactions by Geography",
                      color_discrete_sequence=[COLORS["purple"]])
        fig2.update_layout(PLOTLY_TEMPLATE["layout"], height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Top 50 High-Risk Customers")
    display_cols = ["customer_id","customer_segment","geography","total_tx","fraud_count",
                    "fraud_rate","fraud_loss","net_pnl","avg_risk","risk_tier"]
    disp = [c for c in display_cols if c in cust.columns]
    st.dataframe(
        cust[disp].head(50).style
            .background_gradient(subset=["avg_risk"] if "avg_risk" in disp else [], cmap="RdYlGn_r")
            .format({"fraud_loss":"${:,.0f}","net_pnl":"${:,.0f}","avg_risk":"{:.1f}","fraud_rate":"{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: MERCHANT RISK
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    merch = df.groupby(["merchant_category"]).agg(
        txns=("transaction_amount","count"),
        gross_tv=("transaction_amount","sum"),
        fraud_count=("fraud_flag","sum"),
        fraud_loss=("fraud_loss","sum"),
        cb_count=("chargeback_flag","sum"),
        cb_loss=("chargeback_loss","sum"),
        refund_loss=("refund_loss","sum"),
        net_pnl=("net_pnl_impact","sum"),
        avg_risk=("risk_score","mean"),
    ).reset_index()

    merch["fraud_rate"] = (merch["fraud_count"] / merch["txns"] * 100).round(2)
    merch["cb_rate"]    = (merch["cb_count"] / merch["txns"] * 100).round(3)
    merch = merch.sort_values("fraud_loss", ascending=False)

    fig = px.scatter(
        merch, x="fraud_rate", y="cb_rate",
        size="fraud_loss", color="avg_risk",
        text="merchant_category",
        title="Merchant Risk Matrix: Fraud Rate vs Chargeback Rate",
        color_continuous_scale=[[0,COLORS["green"]],[0.5,COLORS["amber"]],[1,COLORS["red"]]],
        labels={"fraud_rate":"Fraud Rate (%)","cb_rate":"Chargeback Rate (%)"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(PLOTLY_TEMPLATE["layout"], height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        merch.style
            .background_gradient(subset=["fraud_loss"], cmap="RdYlGn_r")
            .format({"fraud_loss":"${:,.0f}","cb_loss":"${:,.0f}","net_pnl":"${:,.0f}",
                     "fraud_rate":"{:.1f}%","cb_rate":"{:.2f}%","avg_risk":"{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: CHANNEL P&L
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    ch = df.groupby("channel").agg(
        txns=("transaction_amount","count"),
        gross_tv=("transaction_amount","sum"),
        fee_income=("fee_income","sum"),
        interchange=("interchange_income","sum"),
        processing_cost=("processing_cost","sum"),
        fraud_loss=("fraud_loss","sum"),
        cb_loss=("chargeback_loss","sum"),
        net_rev=("net_revenue","sum"),
        net_pnl=("net_pnl_impact","sum"),
        fraud_count=("fraud_flag","sum"),
        avg_risk=("risk_score","mean"),
    ).reset_index()
    ch["fraud_rate"] = (ch["fraud_count"]/ch["txns"]*100).round(2)
    ch["pnl_per_tx"] = (ch["net_pnl"]/ch["txns"]).round(4)
    ch["cti"]        = (ch["processing_cost"]/(ch["fee_income"]+ch["interchange"])*100).round(2)
    ch = ch.sort_values("net_pnl", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ch, x="channel", y="net_pnl",
                     color="net_pnl",
                     color_continuous_scale=[[0,COLORS["red"]],[0.5,COLORS["amber"]],[1,COLORS["green"]]],
                     title="Net P&L by Channel")
        fig.update_layout(PLOTLY_TEMPLATE["layout"], height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(ch, x="channel", y="fraud_rate",
                      color="fraud_rate",
                      color_continuous_scale=[[0,COLORS["green"]],[0.5,COLORS["amber"]],[1,COLORS["red"]]],
                      title="Fraud Rate by Channel (%)")
        fig2.update_layout(PLOTLY_TEMPLATE["layout"], height=320, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        ch.style.format({
            "gross_tv":"${:,.0f}","fee_income":"${:,.0f}","net_pnl":"${:,.0f}",
            "fraud_rate":"{:.1f}%","cti":"{:.1f}%","pnl_per_tx":"${:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: EXCEPTION QUEUE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("##### Investigation Exception Queue — High-Risk Transactions")

    exc_cols = ["transaction_id","customer_id","transaction_date","transaction_amount",
                "transaction_type","channel","merchant_category","customer_segment",
                "fraud_flag","fraud_type","risk_score","chargeback_flag","net_pnl_impact","fraud_loss"]
    avail = [c for c in exc_cols if c in df.columns]
    exc = df[avail].copy()

    # Sort by risk
    sort_col = "final_risk_score" if "final_risk_score" in df.columns else "risk_score"
    exc = df[avail + ([sort_col] if sort_col not in avail and sort_col in df.columns else [])].copy()
    exc = exc.sort_values(sort_col if sort_col in exc.columns else "risk_score", ascending=False)

    # Filter to suspicious
    if "is_suspicious" in df.columns:
        exc_filter = df["is_suspicious"] == 1
    else:
        exc_filter = (df["fraud_flag"] == 1) | (df["risk_score"] >= 60)

    exc_display = exc[exc_filter.values[:len(exc)]].head(200)

    ec1,ec2,ec3 = st.columns(3)
    ec1.metric("Queue Size", f"{len(exc_display):,}", "Pending review")
    ec2.metric("Fraud Confirmed", f"{int(exc_display['fraud_flag'].sum() if 'fraud_flag' in exc_display.columns else 0):,}")
    ec3.metric("P&L at Risk", f"${abs(exc_display['net_pnl_impact'].sum() if 'net_pnl_impact' in exc_display.columns else 0)/1e6:.2f}M")

    st.dataframe(
        exc_display.style
            .background_gradient(subset=["risk_score"] if "risk_score" in exc_display.columns else [], cmap="RdYlGn_r")
            .format({"transaction_amount":"${:,.2f}","net_pnl_impact":"${:,.2f}",
                     "fraud_loss":"${:,.2f}","risk_score":"{:.0f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8: ANALYST MEMO
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown(f"""
    <div style="background:#0f1629;border:1px solid #2a4080;padding:24px;
                font-family:'Courier New',monospace;line-height:2">
    <div style="color:#f4a261;font-size:16px;font-weight:bold;letter-spacing:3px;margin-bottom:16px">
    ANALYST INVESTIGATION MEMO — CONFIDENTIAL
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:11px;margin-bottom:16px">
      <div><span style="color:#8892a4">TO: </span>Senior VP, Fraud Risk & Financial Intelligence</div>
      <div><span style="color:#8892a4">FROM: </span>Transaction Monitoring & Analytics Team</div>
      <div><span style="color:#8892a4">DATE: </span>{date.today()}</div>
      <div><span style="color:#e63946">CLASSIFICATION: RESTRICTED</span></div>
    </div>
    <div style="border-top:1px solid #1e2d50;padding-top:16px">
    <div style="color:#f4a261;font-weight:bold;letter-spacing:2px;margin-bottom:8px;font-size:13px">
    RE: BANKING TRANSACTION INTELLIGENCE — FRAUD, RISK & P&L ANALYTICS REVIEW
    </div>
    <div style="color:#f4a261;font-size:10px;letter-spacing:2px;margin-top:12px">1. EXECUTIVE SUMMARY</div>
    <div style="font-size:11px;margin-top:4px">
    A comprehensive analytics review of {total_tx:,} banking transactions across digital, card, ATM,
    branch, and payment channels identified statistically significant fraud clusters, revenue leakage
    patterns, and P&L variances. Approximately {fraud_rate:.1f}% of transactions were confirmed
    fraudulent, with gross fraud loss of ${fraud_loss/1e6:.2f}M.
    </div>
    <div style="color:#f4a261;font-size:10px;letter-spacing:2px;margin-top:12px">2. KEY FINDINGS</div>
    <div style="font-size:11px;margin-top:4px">
    ▸ Fraud Loss: ${fraud_loss/1e6:.2f}M ({fraud_loss/gtv*100:.3f}% of Gross TX Value)<br>
    ▸ Chargeback Loss: ${cb_loss/1e6:.2f}M | Refund Loss: ${refund_loss/1e6:.2f}M<br>
    ▸ Net P&L Impact: ${net_pnl/1e6:.2f}M | Risk-Adjusted Revenue: ${risk_adj_rev/1e6:.2f}M<br>
    ▸ Suspicious Transactions: {suspicious:,} ({suspicious/total_tx*100:.1f}% of portfolio)<br>
    ▸ Cost-to-Income Ratio: {cti:.1f}%
    </div>
    <div style="color:#f4a261;font-size:10px;letter-spacing:2px;margin-top:12px">3. RECOMMENDED ACTIONS</div>
    <div style="font-size:11px;margin-top:4px">
    1. Implement velocity controls on USSD and API channels — limit 3 tx/15 min above $500.<br>
    2. Deploy real-time device fingerprinting for all Card Not Present transactions above $500.<br>
    3. Escalate exception queue to Level 2 investigation — P&L exposure exceeds materiality threshold.<br>
    4. Place Crypto Exchange and Gaming merchants on enhanced monitoring.<br>
    5. Commission full balance reconciliation for inconsistency-flagged transactions.
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
