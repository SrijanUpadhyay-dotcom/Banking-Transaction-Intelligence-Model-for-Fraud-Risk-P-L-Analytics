# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Payee-risk features for APP-scam detection, point-in-time.

An authorised push payment is made by the genuine customer: their own device,
their own credentials, often a convincing story. Account-takeover signals see
nothing unusual. The evidence is in the payee and in how the payment relates to
the customer's history:

| Feature | What it captures |
|---|---|
| `payee_new_for_customer` | first payment to this payee |
| `payee_account_age_days` | days since the payee's account opened (unknown without the feed) |
| `cop_no_match`, `cop_close_match`, `cop_unavailable` | Confirmation-of-Payee result |
| `payee_new_to_bank`, `payee_first_seen_days` | how long anyone at the bank has paid this payee |
| `payee_distinct_senders_30d` | other customers paying it recently: mule fan-in |
| `payee_payments_24h` | burst of payments to the payee |
| `payee_on_us` | the payee account is held at the bank |
| `cust_payments_to_payee_30d` | repeat payments: investment and romance scams escalate |
| `cust_new_payees_7d` | several new payees in a week |
| `amount_vs_cust_max_payment` | amount ÷ the customer's largest earlier payment |
| `amount_vs_hist_avg`, `amount_to_balance`, `txn_hour`, `is_off_hours` | the core features |
| `graph_payee_senders`, `graph_payee_known_fraud_share` | Phase 6: fraud confirmed before the day |

Every count uses only events strictly before the payment's second. Every
feature declares its source fields and passes the same lineage check as the
transaction model's features.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
import pandas as pd

from bti.modeling.features import FeatureSpec, Kind, assert_feature_lineage, event_timestamps

_N = Kind.NUMERIC
_W = ("transaction_date", "transaction_time")
_P = ("customer_id", "payee_id") + _W
PAYMENT_TYPES = ("Transfer", "Wire Transfer", "Payment")

SCAM_FEATURES: List[FeatureSpec] = [
    FeatureSpec("payee_new_for_customer", _N, _P, "NEW_PAYEE", "First payment from this customer to this payee"),
    FeatureSpec("payee_account_age_days", _N, ("payee_id", "payee_account_opened_date") + _W, "PAYEE_RISK",
                "Days since the payee's account was opened (unknown without the payee-intelligence feed)"),
    FeatureSpec("cop_no_match", _N, ("cop_result",), "COP_MISMATCH", "Confirmation of Payee: name does not match"),
    FeatureSpec("cop_close_match", _N, ("cop_result",), "COP_MISMATCH", "Confirmation of Payee: close match"),
    FeatureSpec("cop_unavailable", _N, ("cop_result",), "COP_MISMATCH", "Confirmation of Payee: check not possible"),
    FeatureSpec("payee_new_to_bank", _N, _P, "PAYEE_RISK", "No customer of the bank has paid this payee before"),
    FeatureSpec("payee_first_seen_days", _N, _P, "PAYEE_RISK", "Days since anyone at the bank first paid this payee"),
    FeatureSpec("payee_distinct_senders_30d", _N, _P, "PAYEE_VELOCITY",
                "Other customers who paid this payee in the last 30 days (mule fan-in)"),
    FeatureSpec("payee_payments_24h", _N, _P, "PAYEE_VELOCITY", "Payments to this payee by anyone in the last 24 hours"),
    FeatureSpec("payee_on_us", _N, ("payee_customer_id",), "PAYEE_RISK", "Payee account is held at this bank"),
    FeatureSpec("cust_payments_to_payee_30d", _N, _P, "SCAM_PATTERN",
                "This customer's earlier payments to this payee in 30 days (escalating scams)"),
    FeatureSpec("cust_new_payees_7d", _N, _P, "NEW_PAYEE", "New payees this customer paid in the last 7 days"),
    FeatureSpec("amount_vs_cust_max_payment", _N, ("customer_id", "transaction_amount", "currency") + _W,
                "AMT_VS_HISTORY", "Amount ÷ the customer's largest earlier outbound payment (365 days)"),
    FeatureSpec("amount_vs_hist_avg", _N, ("transaction_amount", "historical_average_transaction_amount"),
                "AMT_VS_HISTORY", "Amount ÷ customer's historical average"),
    FeatureSpec("amount_to_balance", _N, ("transaction_amount", "account_balance_before"), "AMT_VS_BALANCE",
                "Amount ÷ available balance"),
    FeatureSpec("txn_hour", _N, ("transaction_time",), "TIME_OF_DAY", "Hour of day"),
    FeatureSpec("is_off_hours", _N, ("transaction_time",), "TIME_OF_DAY", "Between 22:00 and 06:00"),
    FeatureSpec("graph_payee_senders", _N, ("payee_id", "customer_id", "confirmed_fraud_labels") + _W, "MULE_PAYEE",
                "Other customers who have paid this payee (graph, day lag)"),
    FeatureSpec("graph_payee_known_fraud_share", _N, ("payee_id", "customer_id", "confirmed_fraud_labels") + _W,
                "MULE_PAYEE", "Confirmed frauds paid to this payee per sender (fraud confirmed before the day)"),
]
SCAM_FEATURE_NAMES = [f.name for f in SCAM_FEATURES]
SCAM_FEATURE_BY_NAME = {f.name: f for f in SCAM_FEATURES}
assert_feature_lineage(SCAM_FEATURES)


def payment_rows(df: pd.DataFrame) -> np.ndarray:
    """Outbound payments to a payee: the population the scam model scores."""
    return (df["transaction_type"].isin(PAYMENT_TYPES) & (df["debit_credit_flag"].astype(str) == "Debit")
            & df["payee_id"].notna()).to_numpy()


def scam_features(df: pd.DataFrame, graph: bool = True) -> pd.DataFrame:
    """Payee-risk features for every payment row of `df` (other rows get NaN), using only earlier events."""
    from bti.modeling.fx import RATES_TO_USD
    ts = event_timestamps(df)
    secs = ((ts - pd.Timestamp("1970-01-01")).dt.total_seconds()).to_numpy()
    usd = pd.to_numeric(df["transaction_amount"], errors="coerce").to_numpy(float) * \
        df["currency"].astype(str).str.upper().map(RATES_TO_USD).to_numpy(float)
    is_pay = payment_rows(df)
    out = pd.DataFrame(np.nan, index=df.index, columns=SCAM_FEATURE_NAMES)
    order = np.argsort(secs, kind="stable")
    cust = df["customer_id"].astype(str).to_numpy()
    payee = df["payee_id"].astype(str).to_numpy()
    day = 86400

    pair_times: Dict = defaultdict(deque)          # (customer, payee) -> payment times within 365d
    payee_events: Dict = defaultdict(deque)        # payee -> (t, customer) within 30d
    payee_first: Dict = {}
    cust_payments: Dict = defaultdict(deque)       # customer -> (t, usd) outbound payments within 365d
    cust_new_payee_times: Dict = defaultdict(deque)
    pending: List = []                             # same-second events wait so they never see each other
    cols = {c: np.full(len(df), np.nan) for c in SCAM_FEATURE_NAMES}
    last_t = None

    def flush():
        for i in pending:
            c, p, t = cust[i], payee[i], secs[i]
            if not pair_times[(c, p)]:
                cust_new_payee_times[c].append(t)
            pair_times[(c, p)].append(t)
            payee_events[p].append((t, c))
            payee_first.setdefault(p, t)
            cust_payments[c].append((t, usd[i]))
        pending.clear()

    for i in order:
        if not is_pay[i]:
            continue
        t = secs[i]
        if last_t is not None and t != last_t:
            flush()
        last_t = t
        c, p = cust[i], payee[i]
        pt = pair_times[(c, p)]
        while pt and pt[0] < t - 365 * day:
            pt.popleft()
        pe = payee_events[p]
        while pe and pe[0][0] < t - 30 * day:
            pe.popleft()
        cp = cust_payments[c]
        while cp and cp[0][0] < t - 365 * day:
            cp.popleft()
        nt = cust_new_payee_times[c]
        while nt and nt[0] < t - 7 * day:
            nt.popleft()
        cols["payee_new_for_customer"][i] = float(len(pt) == 0)
        cols["cust_payments_to_payee_30d"][i] = float(sum(1 for x in pt if x >= t - 30 * day))
        cols["payee_distinct_senders_30d"][i] = float(len({cc for _, cc in pe if cc != c}))
        cols["payee_payments_24h"][i] = float(sum(1 for x, _ in pe if x >= t - day))
        first = payee_first.get(p)
        cols["payee_new_to_bank"][i] = float(first is None)
        cols["payee_first_seen_days"][i] = (t - first) / day if first is not None else np.nan
        cols["cust_new_payees_7d"][i] = float(len(nt))
        prior_max = max((a for _, a in cp if not np.isnan(a)), default=np.nan)
        cols["amount_vs_cust_max_payment"][i] = usd[i] / prior_max if prior_max and prior_max > 0 else np.nan
        pending.append(i)

    for k, v in cols.items():
        out[k] = v
    pay_idx = df.index[is_pay]
    opened = pd.to_datetime(df.get("payee_account_opened_date"), errors="coerce") if "payee_account_opened_date" in df \
        else pd.Series(pd.NaT, index=df.index)
    out.loc[pay_idx, "payee_account_age_days"] = ((ts.dt.normalize() - opened).dt.days.clip(lower=0))[pay_idx]
    cop = df["cop_result"].astype("string") if "cop_result" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    for name, value in (("cop_no_match", "no_match"), ("cop_close_match", "close_match"),
                        ("cop_unavailable", "unavailable")):
        out.loc[pay_idx, name] = np.where(cop[pay_idx].isna(), np.nan, (cop[pay_idx] == value).astype(float))
    onus = df["payee_customer_id"].notna() if "payee_customer_id" in df else pd.Series(False, index=df.index)
    out.loc[pay_idx, "payee_on_us"] = onus[pay_idx].astype(float)
    amount = pd.to_numeric(df["transaction_amount"], errors="coerce")
    hist = pd.to_numeric(df.get("historical_average_transaction_amount"), errors="coerce")
    bal = pd.to_numeric(df.get("account_balance_before"), errors="coerce")
    out.loc[pay_idx, "amount_vs_hist_avg"] = (amount / hist.where(hist > 0))[pay_idx]
    out.loc[pay_idx, "amount_to_balance"] = (amount / bal.clip(lower=1.0))[pay_idx]
    hour = ts.dt.hour
    out.loc[pay_idx, "txn_hour"] = hour[pay_idx].astype(float)
    out.loc[pay_idx, "is_off_hours"] = ((hour < 6) | (hour >= 22))[pay_idx].astype(float)
    if graph:
        from bti.graph.temporal import graph_features
        g = graph_features(df)
        out.loc[pay_idx, "graph_payee_senders"] = g.loc[pay_idx, "graph_payee_senders"].to_numpy()
        out.loc[pay_idx, "graph_payee_known_fraud_share"] = g.loc[pay_idx, "graph_payee_known_fraud_share"].to_numpy()
    return out
