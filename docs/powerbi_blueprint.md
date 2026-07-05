# Power BI Dashboard Blueprint
## Banking Transaction Intelligence Model — 7-Page Dashboard

---

## PAGE 1: EXECUTIVE OVERVIEW

### KPIs (Card Visuals)
| KPI | DAX Measure |
|-----|-------------|
| Gross Transaction Value | `[Total Gross TV]` |
| Net Revenue | `[Total Net Revenue]` |
| Net P&L Impact | `[Total Net PnL]` |
| Fraud Loss | `[Total Fraud Loss]` |
| Chargeback Loss | `[Total Chargeback Loss]` |
| Fraud Loss Rate | `[Fraud Loss Rate %]` |
| Suspicious Tx Ratio | `[Suspicious Tx Ratio %]` |
| Risk-Adjusted Revenue | `[Risk Adjusted Revenue]` |

### Charts
- **Line chart**: Monthly Net Revenue vs Fraud Loss (time axis = Month-Year)
- **Bar chart**: Net P&L by Customer Segment
- **Donut chart**: Transaction count by Channel
- **KPI card**: MoM Net P&L Variance % with conditional icon (▲/▼)

### Slicers
- Date (Month-Year range)
- Customer Segment (multi-select)
- Channel (multi-select)
- Geography (multi-select)
- Alert Tier (Critical / High / Medium / Low)

### Business Interpretation
Provides the C-suite with a single-screen view of overall banking performance, connecting revenue generation to fraud and chargeback losses. Designed for weekly management reporting.

---

## PAGE 2: FRAUD MONITORING

### KPIs
- Fraud Transaction Count
- Fraud Loss ($)
- Fraud Loss Rate (%)
- Chargeback Loss
- ML-Flagged Transactions
- Rules-Triggered Count

### Charts
- **Stacked bar**: Fraud by Channel + Fraud Type (colour-coded by typology)
- **Treemap**: Fraud Loss by Merchant Category
- **Line chart**: Monthly Fraud Count trend with MoM delta
- **Table**: Top 20 Fraud Transactions (sortable, with Risk Score column)

### Slicers
- Fraud Type (multi-select)
- Risk Score (≥ slider)
- Channel

---

## PAGE 3: P&L IMPACT ANALYSIS

### KPIs
- Gross Transaction Value
- Fee Income
- Interchange Income
- Processing Cost
- Refund Loss
- Net P&L Impact

### Charts
- **Waterfall chart**: P&L component decomposition (Fee + Interchange − Processing − Fraud − CB − Refund)
- **Clustered bar**: Monthly P&L variance (current vs prior period)
- **Area chart**: Cumulative Net Revenue vs Cumulative Losses
- **Matrix**: P&L by Segment × Channel (heat-shaded)

### DAX Measures
```dax
-- Net Revenue
Net Revenue = [Fee Income] + [Interchange Income] - [Processing Cost]

-- P&L Impact
Net PnL Impact = [Net Revenue] - [Chargeback Loss] - [Refund Loss] - [Fraud Loss]

-- MoM Variance %
MoM PnL Variance % =
  VAR cur = [Total Net PnL]
  VAR prv = CALCULATE([Total Net PnL],
              DATEADD('Date'[Date], -1, MONTH))
  RETURN DIVIDE(cur - prv, ABS(prv), 0) * 100
```

---

## PAGE 4: CUSTOMER RISK INTELLIGENCE

### KPIs
- High-Risk Customer Count (score ≥ 80)
- Average Customer Risk Score
- Top Customer Fraud Loss

### Charts
- **Scatter plot**: Customer Total Spend vs Fraud Loss (sized by Risk Score, coloured by Segment)
- **Bar chart**: Fraud Rate by Customer Segment
- **Table**: Customer Risk Leaderboard (Top 50 by composite risk score)
- **Map visual**: Customer geography with fraud concentration overlay

### Slicers
- Customer Segment
- Risk Tier
- Geography

---

## PAGE 5: MERCHANT & CHANNEL ANALYTICS

### Charts
- **Bar chart**: Chargeback Rate by Merchant Category
- **Matrix**: Channel × Merchant Category P&L heat map
- **Bar chart**: Channel Net P&L (green = profitable, red = loss)
- **Gauge**: USSD/API fraud concentration vs threshold

### Table
Merchant risk table: Name | Category | Fraud Rate | CB Rate | Fraud Loss | Risk Tier

---

## PAGE 6: EXCEPTION QUEUE

### Table
Top 500 high-risk transactions with columns:
TX ID | Customer | Date | Amount | Channel | Merchant Cat | Fraud Type | Risk Score | Alert Tier | P&L Impact

### Filters
- Alert Tier = Critical / High
- Investigation Status (New / Under Review / Closed)
- Date Range

### Business Use
Connects directly to Level 2 investigation workflow. Integrates with AML/KYC case management via API.

---

## PAGE 7: AI INVESTIGATION SUMMARY

### Visual Types
- **Text box**: Auto-generated analyst memo summary (LLM-generated text imported as measure or visual)
- **KPI cards**: Summary statistics for the period
- **Bar chart**: Top anomaly patterns detected by ML model
- **Table**: Model performance metrics (Precision, Recall, F1, ROC-AUC)

---

## COMPLETE DAX MEASURE LIBRARY

```dax
-- ─────────────────────────────────────────────────────────────────────
-- CORE TRANSACTION MEASURES
-- ─────────────────────────────────────────────────────────────────────

Total Transactions = COUNTROWS(banking_transactions)

Fraud Transactions =
  CALCULATE(COUNTROWS(banking_transactions),
            banking_transactions[fraud_flag] = 1)

Suspicious Transactions =
  CALCULATE(COUNTROWS(banking_transactions),
            banking_transactions[is_suspicious] = 1)

High Risk Customer Count =
  CALCULATE(DISTINCTCOUNT(banking_transactions[customer_id]),
            banking_transactions[risk_score] >= 80)

-- ─────────────────────────────────────────────────────────────────────
-- FINANCIAL MEASURES
-- ─────────────────────────────────────────────────────────────────────

Total Gross TV = SUM(banking_transactions[transaction_amount])

Fee Income = SUM(banking_transactions[fee_income])

Interchange Income = SUM(banking_transactions[interchange_income])

Processing Cost = SUM(banking_transactions[processing_cost])

Fraud Loss = SUM(banking_transactions[fraud_loss])

Chargeback Loss = SUM(banking_transactions[chargeback_loss])

Refund Loss = SUM(banking_transactions[refund_loss])

Net Revenue =
  [Fee Income] + [Interchange Income] - [Processing Cost]

Net PnL Impact =
  [Net Revenue] - [Chargeback Loss] - [Refund Loss] - [Fraud Loss]

Risk Adjusted Revenue =
  [Net Revenue] - [Fraud Loss] - [Chargeback Loss]

-- ─────────────────────────────────────────────────────────────────────
-- RATE / RATIO MEASURES
-- ─────────────────────────────────────────────────────────────────────

Fraud Loss Rate % =
  DIVIDE([Fraud Loss], [Total Gross TV], 0) * 100

Chargeback Ratio % =
  DIVIDE([Chargeback Loss], [Total Gross TV], 0) * 100

Suspicious Tx Ratio % =
  DIVIDE([Suspicious Transactions], [Total Transactions], 0) * 100

Revenue Leakage % =
  DIVIDE(
    ABS(MIN(0, [Net PnL Impact])),
    [Fee Income], 0
  ) * 100

Cost to Income Ratio % =
  DIVIDE(
    [Processing Cost],
    [Fee Income] + [Interchange Income], 0
  ) * 100

Refund Loss Ratio % =
  DIVIDE([Refund Loss], [Total Gross TV], 0) * 100

-- ─────────────────────────────────────────────────────────────────────
-- TIME INTELLIGENCE MEASURES
-- ─────────────────────────────────────────────────────────────────────

Prior Month PnL =
  CALCULATE([Net PnL Impact],
            DATEADD('Date'[Date], -1, MONTH))

MoM PnL Variance =
  [Net PnL Impact] - [Prior Month PnL]

MoM PnL Variance % =
  DIVIDE([MoM PnL Variance], ABS([Prior Month PnL]), 0) * 100

YTD Net Revenue =
  CALCULATE([Net Revenue], DATESYTD('Date'[Date]))

YoY Revenue Growth % =
  DIVIDE(
    [Net Revenue] - CALCULATE([Net Revenue], SAMEPERIODLASTYEAR('Date'[Date])),
    ABS(CALCULATE([Net Revenue], SAMEPERIODLASTYEAR('Date'[Date]))),
    0
  ) * 100

-- ─────────────────────────────────────────────────────────────────────
-- CONDITIONAL FORMATTING MEASURES
-- ─────────────────────────────────────────────────────────────────────

PnL Color =
  IF([Net PnL Impact] >= 0, "#52b788", "#e63946")

Fraud Rate Category =
  SWITCH(TRUE(),
    [Fraud Loss Rate %] >= 2.0, "CRITICAL",
    [Fraud Loss Rate %] >= 1.5, "HIGH",
    [Fraud Loss Rate %] >= 1.0, "MEDIUM",
    "LOW"
  )
```

---

## DEPLOYMENT NOTES

- **Data source**: CSV export from `data/processed/banking_transactions_ml_scored.csv`
- **Scheduled refresh**: Daily (if connected to database) or manual for CSV
- **Row-level security**: Apply by `customer_segment` or `geography` for regional access
- **Published to**: Power BI Service workspace — share with Finance / Risk stakeholders
- **Mobile layout**: Configure for executive mobile view (KPI cards only)
