# Banking Transaction Intelligence Model
## Fraud, Risk & P&L Analytics

**Python · SQL · Excel · Power BI · Machine Learning · Streamlit · Bloomberg-Style Terminal**

> *"A hybrid banking analytics + machine learning + FP&A + LLM-assisted intelligence project — not a generic AI chatbot."*

---

## Executive Summary

This project is a professional-grade banking analytics system designed to detect fraud and anomaly patterns, quantify P&L impact, identify revenue leakage, and generate executive-level intelligence from large-scale banking transaction datasets.

Built to demonstrate readiness for roles in **FP&A, Fraud Risk, Transaction Monitoring, Banking Analytics, Financial Crime, and Global Business Services** at institutions like Bank of America, JPMorgan, Goldman Sachs, HSBC, Barclays, and Citi.

**This is not a chatbot.** It is a structured banking intelligence system where Python and SQL handle transaction-level analysis, machine learning detects hidden anomalies, Excel and dashboards present P&L impact, and LLMs assist only in memo generation and natural-language explanations.

---

## Problem Statement

Banks process millions of transactions daily across digital, card, branch, merchant, and payment channels. The core challenge extends beyond fraud detection to:

- Identifying **unexplained P&L movements** and revenue leakage
- Quantifying **chargeback losses** and their impact on income statements
- Detecting **suspicious transaction patterns** hidden in large datasets
- Connecting **transaction-level anomalies** to FP&A metrics
- Generating **audit-ready exception queues** for investigation teams
- Providing **executive dashboards** that translate data signals into business decisions

---

## System Architecture

```
Raw Banking Transactions (50,000 rows)
           ↓
Python / SQL Data Cleaning & Validation
           ↓
Transaction Intelligence Layer (Feature Engineering)
           ↓
Fraud Rules Engine (19 Rules, Weighted Composite Score)
           ↓
Machine Learning Anomaly Detection
(Isolation Forest + Random Forest + Logistic Regression)
           ↓
FP&A + P&L Impact Layer
(Net Revenue · Fraud Loss Rate · Chargeback Ratio · Revenue Leakage)
           ↓
Risk Scoring Model (0-100 Composite Score, Alert Tiers)
           ↓
Excel Workbook (15 tabs) + Power BI (7 pages) + Streamlit + BTI Terminal (HTML)
           ↓
LLM-Based Analyst Memo + Executive Summary + Natural Language Querying
```

---

## Technology Stack

| Layer | Tool / Language | Purpose |
|-------|----------------|---------|
| Data Generation | Python (pandas, numpy) | 50,000 synthetic banking transactions |
| Data Cleaning | Python, SQL | Validation, enrichment, deduplication |
| Transaction Querying | SQL (PostgreSQL / BigQuery) | 13 professional analytics queries |
| Fraud Rules Engine | Python, SQL | 19 weighted rule signals |
| Anomaly Detection | scikit-learn | Isolation Forest, Random Forest, LR |
| FP&A Modeling | Excel, Python | P&L formulas, variance analysis |
| Dashboarding | Power BI / Streamlit | Interactive KPI dashboards |
| Terminal Interface | HTML, CSS, JavaScript | Bloomberg-style BTI Terminal |
| Excel Workbook | Python (openpyxl) | 15-tab professional workbook |
| PDF Reporting | Python (fpdf2) | Consulting-grade report |
| Analyst Memos | LLM | NL investigation memos, summaries |
| Portfolio | PDF, Excel, GitHub, LinkedIn | Recruiter-grade presentation |

---

## Project Structure

```
Banking-Transaction-Intelligence-Model-for-Fraud-Risk-P-L-Analytics/
│
├── README.md                        <- This file
├── requirements.txt                 <- Python dependencies
│
├── src/
│   ├── 01_data_generation.py       <- Generate 50,000 synthetic transactions
│   ├── 02_data_cleaning.py         <- Data cleaning & validation
│   ├── 03_fraud_rules_engine.py    <- 19 fraud detection rules
│   ├── 04_ml_anomaly_detection.py  <- ML models (Isolation Forest, RF, LR)
│   ├── 05_pnl_analytics.py         <- FP&A and P&L calculations
│   ├── 06_excel_workbook_builder.py<- Professional Excel workbook
│   ├── 07_pdf_report_generator.py  <- PDF report generation
│   └── 09_run_all.py               <- Master execution script
│
├── dashboard/
│   ├── streamlit_bti_dashboard.py  <- Streamlit interactive dashboard
│   └── bti_terminal_dashboard.html <- Bloomberg-style HTML terminal
│
├── sql/
│   └── banking_analytics_queries.sql <- 13 professional SQL queries
│
├── data/
│   ├── raw/                         <- Generated raw dataset
│   └── processed/                   <- Cleaned + ML-scored dataset
│
├── outputs/
│   ├── banking_transaction_intelligence_model.xlsx  <- Excel workbook
│   ├── banking_transaction_intelligence_report.pdf  <- PDF report
│   ├── charts/                      <- EDA and analytics charts
│   └── pnl/                         <- P&L summary CSVs
│
├── reports/
│   └── analyst_investigation_memo.md <- Senior management memo
│
├── docs/
│   └── powerbi_blueprint.md        <- Power BI design + DAX measures
│
└── portfolio/
    └── cv_bullets.md               <- CV bullets, LinkedIn post, interview pitches
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
python src/09_run_all.py
```

This runs all stages in sequence:
1. Generate 50,000 synthetic transactions
2. Clean and validate data
3. Apply 19 fraud detection rules
4. Run ML anomaly detection models
5. Calculate P&L analytics
6. Build Excel workbook
7. Generate PDF report

### 3. Launch Streamlit Dashboard
```bash
streamlit run dashboard/streamlit_bti_dashboard.py
```

### 4. Open HTML Terminal
Open `dashboard/bti_terminal_dashboard.html` in any browser.

---

## Dataset Fields (42 Columns)

| Category | Fields |
|----------|--------|
| Transaction Identity | transaction_id, customer_id, account_id, transaction_date, transaction_time |
| Transaction Details | transaction_amount, transaction_type, debit_credit_flag, transaction_status |
| Channel & Location | channel, branch_or_digital_flag, geography, country, city, currency |
| Merchant | merchant_category, merchant_name |
| Customer Profile | customer_segment, customer_age_band, historical_average_transaction_amount, monthly_customer_transaction_count |
| Account | account_balance_before, account_balance_after |
| Risk Signals | failed_attempt_count, reversal_flag, refund_flag, chargeback_flag, fraud_flag, fraud_type, risk_score |
| Authentication | authorization_method, device_id, ip_location, login_attempts |
| P&L | fee_income, interchange_income, processing_cost, chargeback_loss, refund_loss, fraud_loss, net_revenue, net_pnl_impact |

---

## Fraud Detection Rules (19 Rules)

| Rule | Logic | Weight |
|------|-------|--------|
| R01 High Value vs Avg | Amount >= 5x customer historical average | 9 |
| R02 Velocity Spike | Daily transaction count > 10 | 8 |
| R03 Failed Auth | Failed authentication attempts >= 3 | 7 |
| R04 Multi Login | Login attempts >= 4 | 6 |
| R05 Off-Hours | Transaction time 00:00-05:59 | 5 |
| R06 Geo Mismatch | IP country differs from registered | 7 |
| R07 Repeated Merchant | >= 5 transactions same merchant same day | 6 |
| R08 Abnormal Refund | Refund > 2x avg in high-risk MCC | 8 |
| R09 CB Heavy Merchant | Chargeback in Crypto/Gaming/Travel | 8 |
| R10 Z-Score Outlier | Z-score > 3.5 within customer segment | 9 |
| R11 Balance Inconsistency | Account balance negative after debit | 9 |
| R12 Device/IP Mismatch | Unknown device + high risk score | 8 |
| R13 Segment High Value | Student/NRI + amount > $5,000 | 7 |
| R14 Channel Concentration | USSD/API + amount > 2x avg | 6 |
| R15 Duplicate Transaction | Same customer + amount + merchant + date | 9 |
| R16 Rapid Sequential | > 5 transactions same day | 6 |
| R17 Cross-Border Suspicious | High-risk MCC + risk > 50 + amount > $1K | 8 |
| R18 Refund Ratio | Merchant refund ratio > 30% | 7 |
| R19 Chargeback Ratio | Customer chargeback ratio > 10% | 8 |

---

## Machine Learning Models

| Model | Type | Target ROC-AUC | Purpose |
|-------|------|---------|---------|
| Isolation Forest | Unsupervised | N/A | Primary anomaly detection |
| Z-Score Detection | Statistical | N/A | Segment-relative outlier baseline |
| Logistic Regression | Supervised | ~0.92 | Binary fraud classifier |
| Random Forest | Supervised | ~0.96 | Ensemble fraud classifier |

**Composite ML Score**: Weighted ensemble of all models normalised to 0-100.

---

## P&L Formulas

```
Net Revenue            = Fee Income + Interchange Income - Processing Cost
Net P&L Impact         = Net Revenue - Chargeback Loss - Refund Loss - Fraud Loss
Fraud Loss Rate (%)    = (Fraud Loss / Gross Transaction Value) x 100
Chargeback Ratio (%)   = (Chargeback Loss / Gross Transaction Value) x 100
Cost-to-Income Ratio   = Processing Cost / (Fee Income + Interchange) x 100
Risk-Adjusted Revenue  = Net Revenue - Fraud Loss - Chargeback Loss
Revenue Leakage %      = |MIN(0, Net P&L)| / Fee Income x 100
Suspicious Tx Ratio    = Suspicious Transactions / Total Transactions x 100
MoM Variance %         = (Current - Prior Month P&L) / |Prior Month| x 100
Exception Rate %       = High-Risk Transactions / Total Transactions x 100
```

---

## Excel Workbook (15 Tabs)

| Tab | Purpose |
|-----|---------|
| 00_README | Project navigation guide |
| 01_DataDictionary | Full field definitions (42 fields) |
| 02_RawTransactions | Sample raw data (1,000 rows) |
| 03_CleanedData | Validated + enriched transactions |
| 04_FraudRulesEngine | 19-rule catalogue + output summary |
| 05_RiskScoring | Composite risk model output |
| 06_FPASummary | Top-level P&L KPI cards |
| 07_PnLImpact | P&L component breakdown |
| 08_ChannelAnalysis | Channel profitability |
| 09_SegmentAnalysis | Customer segment P&L |
| 10_MerchantRisk | Merchant risk ranking |
| 11_MonthlyVariance | MoM P&L variance analysis |
| 12_ExceptionQueue | High-risk investigation queue (top 500) |
| 13_ExecDashboard | Executive KPI dashboard + formulas |
| 14_AnalystMemo | Formatted investigation memo |

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/banking_transaction_intelligence_model.xlsx` | 15-tab Excel workbook |
| `outputs/banking_transaction_intelligence_report.pdf` | 40-page PDF report |
| `dashboard/bti_terminal_dashboard.html` | Bloomberg-style HTML terminal |
| `dashboard/streamlit_bti_dashboard.py` | Streamlit interactive dashboard |
| `sql/banking_analytics_queries.sql` | 13 professional SQL queries |
| `reports/analyst_investigation_memo.md` | Senior management investigation memo |
| `docs/powerbi_blueprint.md` | Power BI dashboard blueprint + DAX measures |
| `portfolio/cv_bullets.md` | CV bullets, LinkedIn post, interview pitches |

---

## Interview Positioning

**One-liner:**
> Built an AI-assisted banking transaction intelligence system combining Python, SQL, Excel, Power BI, and machine learning to detect fraud anomalies, quantify P&L impact, identify revenue leakage, and generate executive dashboards for fraud risk and FP&A decision-making.

**Target roles:** FP&A Analyst | Global Banking Analyst | Fraud Strategy Analyst | Transaction Monitoring Analyst | Business Intelligence Analyst | Operations Risk Analyst | Financial Crime Analyst | Data Analyst

---

## Troubleshooting

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**Streamlit auto-generates sample data** if full dataset is not found. Run `src/01_data_generation.py` through `src/04_ml_anomaly_detection.py` for the full 50,000-row dataset.

**Memory issues:** Reduce `N_TRANSACTIONS` in `src/01_data_generation.py` to 10,000 for faster testing.

---

## Author

**Srijan Upadhyay**  
MBA International Business - Finance  
Banking Analytics Portfolio | 2024  
shashinandan7833@gmail.com

---

## Confidentiality Notice

This repository contains synthetic data only. No real bank data, client data, employer data, API keys, tokens, or personally identifiable information is committed to this repository. This project is intended for personal learning, professional portfolio development, and controlled professional demonstration.
