"""
Banking Transaction Intelligence Model
Part 1: Synthetic Dataset Generation (50,000 transactions)
Generates a realistic banking transaction dataset with fraud, risk, and P&L fields.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

N_TRANSACTIONS = 50_000
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 12, 31)

CHANNELS = ["Mobile Banking", "Internet Banking", "ATM", "Branch", "POS Terminal",
            "USSD", "API/Open Banking", "Call Centre"]
CHANNEL_WEIGHTS = [0.30, 0.22, 0.18, 0.10, 0.10, 0.04, 0.04, 0.02]

MERCHANT_CATEGORIES = [
    "Retail / General Merchandise", "Grocery & Supermarkets", "Fuel & Petroleum",
    "Restaurants & Food Delivery", "Travel & Airlines", "Hotels & Accommodation",
    "Healthcare & Pharmacy", "Electronics & Technology", "Utilities & Telecom",
    "Financial Services / Money Transfer", "Online Marketplaces", "Gaming & Gambling",
    "Luxury Goods", "Education", "Entertainment & Streaming", "Auto & Transportation",
    "Insurance", "Government & Taxes", "Crypto Exchanges", "Real Estate"
]
_mcw_raw = [0.15, 0.12, 0.08, 0.10, 0.06, 0.04, 0.06, 0.07,
            0.05, 0.07, 0.06, 0.02, 0.02, 0.02, 0.02, 0.03,
            0.01, 0.01, 0.01, 0.02]
MERCHANT_CATEGORY_WEIGHTS = [w / sum(_mcw_raw) for w in _mcw_raw]

TRANSACTION_TYPES = ["Purchase", "Transfer", "Withdrawal", "Deposit", "Payment",
                     "Refund", "Reversal", "Fee Charge", "Interest Credit",
                     "Chargeback", "Wire Transfer", "Card Not Present"]
TRANSACTION_TYPE_WEIGHTS = [0.28, 0.18, 0.12, 0.10, 0.10, 0.05, 0.03, 0.04,
                             0.02, 0.02, 0.04, 0.02]

CUSTOMER_SEGMENTS = ["Retail", "Premium", "Private Banking", "SME", "Corporate",
                     "Student", "NRI / Diaspora"]
CUSTOMER_SEGMENT_WEIGHTS = [0.45, 0.20, 0.05, 0.12, 0.08, 0.07, 0.03]

AGE_BANDS = ["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
AGE_BAND_WEIGHTS = [0.12, 0.28, 0.25, 0.18, 0.10, 0.07]

GEOGRAPHIES = {
    "United States": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                      "Miami", "San Francisco", "Boston", "Atlanta"],
    "United Kingdom": ["London", "Manchester", "Birmingham", "Leeds", "Glasgow"],
    "India":          ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"],
    "Singapore":      ["Singapore City"],
    "UAE":            ["Dubai", "Abu Dhabi"],
    "Nigeria":        ["Lagos", "Abuja"],
    "Germany":        ["Frankfurt", "Berlin", "Munich"],
    "Hong Kong":      ["Hong Kong"],
}
GEO_WEIGHTS = [0.28, 0.18, 0.20, 0.08, 0.06, 0.05, 0.09, 0.06]

CURRENCIES = {"United States": "USD", "United Kingdom": "GBP", "India": "INR",
              "Singapore": "SGD", "UAE": "AED", "Nigeria": "NGN",
              "Germany": "EUR", "Hong Kong": "HKD"}

AUTH_METHODS = ["PIN", "Biometric", "OTP", "Password", "Token", "Contactless", "Signature"]
AUTH_METHOD_WEIGHTS = [0.30, 0.25, 0.20, 0.12, 0.07, 0.04, 0.02]

FRAUD_TYPES = ["Card Not Present Fraud", "Account Takeover", "Identity Theft",
               "Friendly Fraud", "First Party Fraud", "Mule Account",
               "Synthetic Identity", "Phishing", "SIM Swap", "Authorised Push Payment"]

TRANSACTION_STATUSES = ["Completed", "Failed", "Pending", "Reversed", "Declined",
                        "Under Review", "Fraudulent"]
STATUS_WEIGHTS = [0.78, 0.07, 0.03, 0.04, 0.04, 0.02, 0.02]

MERCHANTS_BY_CATEGORY = {
    "Retail / General Merchandise": ["Walmart", "Target", "Amazon", "Costco", "Best Buy"],
    "Grocery & Supermarkets":       ["Whole Foods", "Tesco", "Kroger", "Sainsbury's", "ALDI"],
    "Fuel & Petroleum":             ["Shell", "BP", "ExxonMobil", "Chevron", "TotalEnergies"],
    "Restaurants & Food Delivery":  ["McDonald's", "Uber Eats", "DoorDash", "Starbucks", "Domino's"],
    "Travel & Airlines":            ["United Airlines", "British Airways", "Emirates", "Delta", "Lufthansa"],
    "Hotels & Accommodation":       ["Marriott", "Hilton", "Hyatt", "IHG", "Airbnb"],
    "Healthcare & Pharmacy":        ["CVS Health", "Walgreens", "Boots", "Apollo Pharmacy", "Rite Aid"],
    "Electronics & Technology":     ["Apple Store", "Samsung", "Best Buy", "Currys", "Flipkart"],
    "Utilities & Telecom":          ["AT&T", "Verizon", "Vodafone", "Jio", "Comcast"],
    "Financial Services / Money Transfer": ["PayPal", "Western Union", "Wise", "Remitly", "MoneyGram"],
    "Online Marketplaces":          ["eBay", "Etsy", "Alibaba", "Flipkart", "Lazada"],
    "Gaming & Gambling":            ["Steam", "Bet365", "DraftKings", "FanDuel", "Unibet"],
    "Luxury Goods":                 ["Louis Vuitton", "Gucci", "Rolex", "Tiffany & Co", "Hermès"],
    "Education":                    ["Coursera", "Udemy", "Tuition Direct", "Chegg", "Khan Academy"],
    "Entertainment & Streaming":    ["Netflix", "Spotify", "Disney+", "Apple TV+", "YouTube Premium"],
    "Auto & Transportation":        ["Uber", "Lyft", "AutoZone", "Hertz", "National Car"],
    "Insurance":                    ["Allstate", "Progressive", "GEICO", "AXA", "Aviva"],
    "Government & Taxes":           ["IRS Payment", "HMRC", "eBay GST", "Council Tax", "DMV"],
    "Crypto Exchanges":             ["Binance", "Coinbase", "Kraken", "Bybit", "OKX"],
    "Real Estate":                  ["Zillow Rental", "Rightmove Deposit", "Century 21", "RE/MAX", "JLL"],
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def random_date_between(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def generate_device_id():
    return f"DEV-{random.randint(100000, 999999)}"

def generate_ip():
    return f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER & ACCOUNT MASTER
# ─────────────────────────────────────────────────────────────────────────────

N_CUSTOMERS = 8_000
customer_ids  = [f"CUST-{str(i).zfill(6)}" for i in range(1, N_CUSTOMERS + 1)]
account_ids   = [f"ACC-{str(i).zfill(6)}"  for i in range(1, N_CUSTOMERS + 1)]

customer_segments = np.random.choice(CUSTOMER_SEGMENTS, size=N_CUSTOMERS, p=CUSTOMER_SEGMENT_WEIGHTS)
age_bands         = np.random.choice(AGE_BANDS, size=N_CUSTOMERS, p=AGE_BAND_WEIGHTS)

geo_list   = list(GEOGRAPHIES.keys())
cust_geos  = np.random.choice(geo_list, size=N_CUSTOMERS, p=GEO_WEIGHTS)

# Historical average transaction amounts per segment
segment_avg = {
    "Retail":          500,
    "Premium":         2_500,
    "Private Banking": 15_000,
    "SME":             8_000,
    "Corporate":       50_000,
    "Student":         150,
    "NRI / Diaspora":  3_000,
}
hist_avg = np.array([
    np.random.normal(segment_avg[seg], segment_avg[seg] * 0.3)
    for seg in customer_segments
]).clip(50, 200_000)

monthly_tx_count = np.random.randint(1, 80, size=N_CUSTOMERS)

# Account balances
account_balances = {
    acc: np.random.uniform(500, 150_000)
    for acc in account_ids
}

# Device and IP per customer (mostly consistent, but some anomalies added later)
customer_devices = {cid: generate_device_id() for cid in customer_ids}
customer_ips     = {cid: generate_ip()        for cid in customer_ids}

customer_master = pd.DataFrame({
    "customer_id":    customer_ids,
    "account_id":     account_ids,
    "customer_segment": customer_segments,
    "customer_age_band": age_bands,
    "geography":      cust_geos,
    "historical_avg_tx_amount": hist_avg,
    "monthly_tx_count": monthly_tx_count,
})
customer_master.set_index("customer_id", inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

print("Generating 50,000 synthetic banking transactions...")

records = []

for i in range(N_TRANSACTIONS):
    tx_id   = f"TXN-{str(i+1).zfill(8)}"
    cust_id = np.random.choice(customer_ids)
    cm      = customer_master.loc[cust_id]
    acc_id  = cm["account_id"]
    segment = cm["customer_segment"]
    age_band= cm["customer_age_band"]
    geo     = cm["geography"]
    hist_avg_amt = cm["historical_avg_tx_amount"]
    mo_count     = cm["monthly_tx_count"]

    # Transaction date & time
    tx_datetime = random_date_between(START_DATE, END_DATE)
    tx_date     = tx_datetime.date()
    tx_time     = tx_datetime.strftime("%H:%M:%S")

    # Transaction type and debit/credit
    tx_type = np.random.choice(TRANSACTION_TYPES, p=TRANSACTION_TYPE_WEIGHTS)
    dc_flag = "Credit" if tx_type in ["Deposit", "Refund", "Interest Credit"] else "Debit"

    # Channel
    channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
    branch_flag = "Branch" if channel == "Branch" else "Digital"

    # Merchant
    merch_cat = np.random.choice(MERCHANT_CATEGORIES, p=MERCHANT_CATEGORY_WEIGHTS)
    merch_name = random.choice(MERCHANTS_BY_CATEGORY[merch_cat])

    # Geography details
    country  = geo
    city     = random.choice(GEOGRAPHIES[geo])
    currency = CURRENCIES[geo]

    # Amount: generally close to historical average, sometimes abnormal
    anomaly_factor = 1.0
    is_fraud = False
    fraud_type = None

    amount_base = max(1.0, np.random.exponential(hist_avg_amt * 0.7))

    # ── FRAUD INJECTION (~5%) ─────────────────────────────────────────────
    roll = random.random()
    if roll < 0.05:
        is_fraud = True
        fraud_type = random.choice(FRAUD_TYPES)
        # Fraud transactions are often large outliers
        anomaly_factor = np.random.uniform(3.0, 15.0)
        amount_base *= anomaly_factor

    amount = round(amount_base, 2)
    amount = max(1.0, amount)

    # Account balance
    bal_before = account_balances.get(acc_id, 5000.0)
    bal_after  = bal_before - amount if dc_flag == "Debit" else bal_before + amount
    account_balances[acc_id] = max(0.0, bal_after)

    # Transaction status
    status = np.random.choice(TRANSACTION_STATUSES, p=STATUS_WEIGHTS)
    if is_fraud:
        status = np.random.choice(["Completed", "Under Review", "Fraudulent"],
                                   p=[0.50, 0.30, 0.20])

    # Auth method
    auth_method = np.random.choice(AUTH_METHODS, p=AUTH_METHOD_WEIGHTS)

    # Device & IP – sometimes mismatch for fraud/anomaly
    if is_fraud and random.random() < 0.6:
        device_id  = generate_device_id()   # unknown device
        ip_location = generate_ip()          # unknown IP
        login_attempts = random.randint(3, 10)
    else:
        device_id  = customer_devices[cust_id]
        ip_location = customer_ips[cust_id]
        login_attempts = random.choices([1, 2, 3, 4, 5], weights=[70, 15, 8, 4, 3])[0]

    failed_attempts = random.choices([0, 1, 2, 3, 4], weights=[75, 14, 6, 3, 2])[0]

    # Reversal, refund, chargeback flags
    reversal_flag   = 1 if tx_type == "Reversal" else (1 if random.random() < 0.03 else 0)
    refund_flag     = 1 if tx_type == "Refund"   else (1 if random.random() < 0.04 else 0)
    chargeback_flag = 1 if (is_fraud and random.random() < 0.35) else (1 if random.random() < 0.01 else 0)

    # Risk score (0–100)
    risk_score = 5
    if is_fraud:               risk_score += 60
    if amount > hist_avg_amt * 3: risk_score += 15
    if failed_attempts > 2:    risk_score += 10
    if login_attempts > 3:     risk_score += 8
    if chargeback_flag:        risk_score += 12
    if reversal_flag:          risk_score += 5
    if merch_cat in ["Gaming & Gambling", "Crypto Exchanges", "Financial Services / Money Transfer"]:
        risk_score += 8
    risk_score = min(100, risk_score + random.randint(-3, 3))

    # ── P&L COMPONENTS ────────────────────────────────────────────────────
    fee_rate        = {"Purchase": 0.0025, "Transfer": 0.005, "Withdrawal": 0.002,
                       "Wire Transfer": 0.008, "Payment": 0.002, "Card Not Present": 0.003}
    fee_income      = round(amount * fee_rate.get(tx_type, 0.001), 2)

    interchange_rate = 0.0175 if merch_cat not in ["Government & Taxes", "Utilities & Telecom"] else 0.005
    interchange_income = round(amount * interchange_rate, 2) if dc_flag == "Debit" and tx_type == "Purchase" else 0.0

    processing_cost  = round(amount * 0.003 + random.uniform(0.10, 0.50), 2)
    chargeback_loss  = round(amount * 1.05, 2) if chargeback_flag else 0.0
    refund_loss      = round(amount * 0.15, 2) if refund_flag else 0.0
    fraud_loss       = round(amount, 2) if (is_fraud and status == "Completed") else 0.0

    net_revenue      = round(fee_income + interchange_income - processing_cost, 2)
    net_pnl_impact   = round(net_revenue - chargeback_loss - refund_loss - fraud_loss, 2)

    fraud_flag = 1 if is_fraud else 0

    records.append({
        "transaction_id":          tx_id,
        "customer_id":             cust_id,
        "account_id":              acc_id,
        "transaction_date":        tx_date,
        "transaction_time":        tx_time,
        "transaction_amount":      amount,
        "transaction_type":        tx_type,
        "debit_credit_flag":       dc_flag,
        "channel":                 channel,
        "branch_or_digital_flag":  branch_flag,
        "merchant_category":       merch_cat,
        "merchant_name":           merch_name,
        "customer_segment":        segment,
        "customer_age_band":       age_band,
        "geography":               geo,
        "country":                 country,
        "city":                    city,
        "currency":                currency,
        "account_balance_before":  round(bal_before, 2),
        "account_balance_after":   round(max(0.0, bal_after), 2),
        "transaction_status":      status,
        "failed_attempt_count":    failed_attempts,
        "reversal_flag":           reversal_flag,
        "refund_flag":             refund_flag,
        "chargeback_flag":         chargeback_flag,
        "fraud_flag":              fraud_flag,
        "fraud_type":              fraud_type if is_fraud else None,
        "risk_score":              risk_score,
        "authorization_method":    auth_method,
        "device_id":               device_id,
        "ip_location":             ip_location,
        "login_attempts":          login_attempts,
        "historical_average_transaction_amount": round(hist_avg_amt, 2),
        "monthly_customer_transaction_count":    int(mo_count),
        "fee_income":              fee_income,
        "interchange_income":      interchange_income,
        "processing_cost":         processing_cost,
        "chargeback_loss":         chargeback_loss,
        "refund_loss":             refund_loss,
        "fraud_loss":              fraud_loss,
        "net_revenue":             net_revenue,
        "net_pnl_impact":          net_pnl_impact,
    })

    if (i + 1) % 10_000 == 0:
        print(f"  Generated {i+1:,} transactions...")

df = pd.DataFrame(records)

# Ensure output dirs exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

output_path = "data/raw/banking_transactions_raw.csv"
df.to_csv(output_path, index=False)

print(f"\n✓ Dataset saved to {output_path}")
print(f"  Shape       : {df.shape}")
print(f"  Fraud txns  : {df['fraud_flag'].sum():,}  ({df['fraud_flag'].mean()*100:.1f}%)")
print(f"  Total amount: ${df['transaction_amount'].sum():,.0f}")
print(f"  Net revenue : ${df['net_revenue'].sum():,.0f}")
print(f"  Net P&L     : ${df['net_pnl_impact'].sum():,.0f}")
print(f"  Fraud loss  : ${df['fraud_loss'].sum():,.0f}")
print(f"  Chargeback  : ${df['chargeback_loss'].sum():,.0f}")
