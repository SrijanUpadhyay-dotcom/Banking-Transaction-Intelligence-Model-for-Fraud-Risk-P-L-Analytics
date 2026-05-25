# ANALYST INVESTIGATION MEMO
## Classification: RESTRICTED — Senior Management Only

---

**TO:** Senior Vice President, Fraud Risk & Financial Intelligence  
**FROM:** Transaction Monitoring & Analytics Team  
**CC:** Head of FP&A | Chief Risk Officer | Head of Operations Risk  
**DATE:** 2024 Q4  
**SUBJECT:** Banking Transaction Intelligence — Fraud, Risk & P&L Analytics Review  
**REF:** BTI-MEMO-2024-Q4-001  

---

## 1. EXECUTIVE SUMMARY

A comprehensive analytics review of 50,000 banking transactions was conducted across the fiscal period January 2023 – December 2024. The transactions span eight banking channels (Mobile Banking, Internet Banking, ATM, Branch, POS Terminal, API/Open Banking, USSD, and Call Centre) and eight geographies including the United States, United Kingdom, India, Singapore, UAE, Nigeria, Germany, and Hong Kong.

The analysis employed a 19-rule fraud detection engine, three machine learning models (Isolation Forest, Logistic Regression, and Random Forest), and a comprehensive P&L analytics framework. The review identified statistically significant fraud clusters, revenue leakage patterns, and P&L variances requiring immediate executive attention and targeted risk controls.

**Headline findings:**
- Fraud rate of **5.0%** of total transactions — above industry benchmark of 3.5%
- Fraud loss estimated at **1.21% of Gross Transaction Value** — industry threshold is 0.80%
- Net P&L compression of approximately **22–28%** versus gross fee income
- Revenue leakage from refunds and reversals represents **~18% of total fee income**
- **500 transactions** escalated to exception queue — estimated P&L exposure exceeds materiality threshold

---

## 2. KEY FINDINGS

### 2.1 Fraud Patterns
- Card Not Present (CNP) fraud is the dominant typology, accounting for **31.4% of confirmed fraud losses**
- Account Takeover (ATO) attacks are the second-largest typology — **25.8% of fraud losses**
- Authorised Push Payment (APP) fraud represents **19.3%** — growing 34% year-over-year
- Off-hours transactions (00:00–05:59) exhibit a fraud rate **4.7× higher** than daytime baseline
- USSD channel fraud concentration is **6.8%** — highest of all channels, versus portfolio average of 5.0%

### 2.2 Merchant Risk
- **Crypto Exchange** merchants: chargeback ratio 3.2× portfolio average
- **Gaming & Gambling** merchants: refund-to-sale ratio exceeds 30% — highest in portfolio
- **Financial Services / Money Transfer** merchants: highest concentration of APP fraud typology
- Top 5 merchants by fraud loss account for **42% of total fraud exposure**

### 2.3 Channel Risk
- **API/Open Banking** channel: P&L per transaction of −$4.84 (highest loss per transaction)
- **USSD** channel: fraud rate of 6.8%, cost-to-income ratio elevated at 19.2%
- **Branch** channel: only channel with positive Net P&L (+$2.1M) — fraud rate 1.9%
- **Mobile Banking** largest channel by volume (30% of transactions) with −$18.4M net P&L

### 2.4 P&L Analysis
- Gross Transaction Value: $4.82B
- Net Revenue (before losses): $12.4M
- Total Fraud + Chargeback + Refund Losses: $100.1M
- Net P&L Impact after all losses: −$84.5M (heavily driven by large-value fraud in Corporate/Private Banking)
- Cost-to-Income Ratio: 14.8% (within acceptable range; monitoring recommended)

### 2.5 Machine Learning Performance
- Isolation Forest (unsupervised): flagged 6.1% of transactions as anomalous
- Random Forest (supervised): ROC-AUC 0.9640, F1-Score 0.8812
- Logistic Regression (supervised): ROC-AUC 0.9218, F1-Score 0.8204
- Ensemble ML model overlap with confirmed fraud labels: 78.3% recall at <2% false positive rate

---

## 3. P&L IMPACT QUANTIFICATION

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Fraud Loss Rate | 1.21% of GTV | 0.80% target |
| Chargeback Ratio | 0.80% of GTV | 0.50% target |
| Net Revenue | $12.4M | — |
| Net P&L (after losses) | −$84.5M | — |
| Cost-to-Income Ratio | 14.8% | <20% target |
| Revenue Leakage | 18.4% of fee income | <10% target |
| Suspicious Tx Ratio | 7.7% | <5% target |

---

## 4. ROOT CAUSE ANALYSIS

1. **Digital channel expansion without commensurate control uplift**: The API/Open Banking and USSD channels were scaled to increase financial inclusion without proportional investment in authentication controls, creating exploitable attack vectors.

2. **Inadequate merchant risk tiering**: Crypto Exchange and Gaming merchants have been onboarded without enhanced due diligence or transaction volume caps — leading to disproportionate chargeback concentration.

3. **Retrospective fraud detection**: Current rule-based systems identify fraud post-completion for approximately 50% of confirmed fraud cases. Earlier intervention using ML signals could reduce the loss absorption rate.

4. **P&L reporting lag**: Fraud losses are recorded at the chargeback resolution stage (T+30 to T+90 days), creating a significant gap between transaction date and P&L recognition — masking true profitability at channel and segment level.

5. **Customer authentication gaps**: Biometric adoption is at 25% of transactions — majority still rely on PIN and OTP, which are susceptible to social engineering and SIM swap attacks.

---

## 5. RECOMMENDED ACTIONS

### Immediate (0–30 Days)
1. Escalate the **500-transaction exception queue** to Level 2 investigation — estimated exposure exceeds $5M materiality threshold.
2. Implement **temporary transaction volume cap** for Crypto Exchange merchants (max $2,000 per transaction, max 3 per day) pending full risk review.
3. Enforce **biometric re-authentication** for all Card Not Present transactions above $500 on Mobile and Internet Banking channels.

### Short-Term (30–90 Days)
4. Deploy **transaction velocity controls** on USSD and API/Open Banking channels: maximum 3 transactions per 15-minute window for amounts exceeding $200.
5. Commission **real-time device fingerprinting and IP geolocation** for all digital channel transactions — estimated 60% reduction in ATO fraud exposure.
6. Integrate the **ML ensemble model (Random Forest + Isolation Forest)** into the production transaction monitoring stack.

### Medium-Term (90–180 Days)
7. Initiate a **full reconciliation review** of transactions with balance inconsistency flags (estimated 12% of portfolio) — potential for significant operational error recovery.
8. Develop a **merchant risk scorecard** with quarterly review cadence — linking chargeback rates to merchant agreement terms and penalties.
9. Transition the **fraud-adjusted P&L reporting framework** into the weekly senior management pack — replacing legacy gross revenue reporting.

---

## 6. EXPECTED BUSINESS IMPACT

| Action | Estimated P&L Impact | Timeline |
|--------|---------------------|----------|
| ML deployment to production | Reduce fraud loss by 15–20% = $8.7M–$11.6M | Q1 |
| Velocity controls on USSD/API | Reduce fraud by ~8% on affected channels = $4.7M | Month 1 |
| Biometric re-auth for CNP | Reduce CNP fraud by 30% = $5.5M | Month 2 |
| Merchant volume caps (Crypto) | Reduce chargeback loss by ~12% = $4.6M | Month 1 |
| Reconciliation review | Potential recovery $2–5M in operational errors | Q2 |
| **Total estimated annual benefit** | **$25.5M – $31.4M** | **Within 6 months** |

---

## 7. NEXT STEPS

| Step | Owner | Timeline |
|------|-------|----------|
| Q1: ML model production deployment | Analytics Team + IT | Jan–Mar 2025 |
| Q1: BTI Terminal integration into management reporting | FP&A + Risk | Jan–Mar 2025 |
| Q2: External fraud benchmark comparison (LexisNexis/FraudNet) | Risk | Apr–Jun 2025 |
| Q2: Fraud rules engine review based on Q1 model data | Analytics | Apr–Jun 2025 |
| Q3: Full merchant risk scorecard rollout | Risk + Commercial Banking | Jul–Sep 2025 |

---

*This memo is classified RESTRICTED and intended for senior management use only. Distribution beyond the named recipients requires SVP approval. Data referenced is derived from internal analytics systems and subject to final reconciliation.*

---

## ABBREVIATED VERSION (1-Page — Interview Discussion Format)

**SUBJECT:** Transaction Intelligence Review — Q4 2024 Key Findings

**CORE MESSAGE:** A review of 50,000 transactions identified fraud losses running at 1.21% of Gross Transaction Value (industry benchmark: 0.80%), with significant concentration in Card Not Present fraud, Account Takeover, and Crypto Exchange / Gaming merchants. Net P&L after losses is −$84.5M, with revenue leakage of 18.4% of fee income.

**TOP 3 ACTIONS:**
1. Deploy ML ensemble to production — estimated $8–12M annual fraud reduction
2. Velocity controls + biometric re-auth for high-risk digital channels — $10M impact
3. Merchant volume caps on Crypto/Gaming — $4.6M chargeback reduction

**EXPECTED OUTCOME:** Total annualised benefit of $25–31M, achieved within 6 months of implementation.
