# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Synthetic mule-ring generator: a capability test harness, not training data.

The synthetic dataset has almost no network structure. Only 50 devices are
shared, each by two customers, and no IP or payee is shared. Graph methods
therefore have nothing to find in it. This module injects rings shaped like
real mule networks into a *copy* of the data, so graph features can be tested
for whether they detect ring structure at all:

- **Mule accounts.** 3–8 new customers per ring, sharing 1–2 devices and 1–2
  IPs. They warm up with a few ordinary transactions before the ring goes
  active.
- **Victim payments.** Existing customers pay a mule's account, using their own
  usual device, IP, channel and typical amounts. Labelled Authorised Push
  Payment fraud.
- **Cash-out.** Mules move the money on to exit accounts from the shared
  devices. Labelled Mule Account fraud.
- **Confirmation delays.** Victims report within 1–10 days. Mule cash-outs are
  confirmed when the ring is uncovered, 20–60 days later. Existing frauds are
  confirmed 15–90 days after the event. Graph features may only use a
  confirmation made before the day of the transaction.

Payees are given to *every* payment-type transaction, including shared biller
payees used by thousands of customers. Without that, the mere presence of a
payee would give the rings away, and graph features would need to cope with
giant benign components.

Results on this data show *capability* (can the method see a ring shaped like
this?), not lift on a real book.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PAYMENT_TYPES = ("Transfer", "Wire Transfer", "Payment")
NOTICE = ("SYNTHETIC rings injected into a copy of synthetic data for capability testing of graph methods; "
          "not evidence of performance on a real book.")


def _ts(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["transaction_date"].astype(str) + " " + df["transaction_time"].astype(str).str[:8],
                          errors="coerce")


def assign_payees(df: pd.DataFrame, rng: np.random.Generator, n_billers: int = 30,
                  biller_share: float = 0.2) -> pd.DataFrame:
    """Give every payment-type row a payee: each customer's regular payees, plus popular billers."""
    out = df.copy()
    pay = out["transaction_type"].isin(PAYMENT_TYPES).to_numpy()
    customers = out.loc[pay, "customer_id"].astype(str).unique()
    regular = {c: [f"PAYEE-{rng.integers(0, 10**7):07d}" for _ in range(rng.integers(1, 5))] for c in customers}
    billers = [f"BILLER-{i:03d}" for i in range(n_billers)]
    payees = []
    for c, use_biller in zip(out.loc[pay, "customer_id"].astype(str), rng.random(pay.sum()) < biller_share):
        payees.append(billers[rng.integers(0, n_billers)] if use_biller else regular[c][rng.integers(0, len(regular[c]))])
    out["payee_id"] = None
    out.loc[pay, "payee_id"] = payees
    return out


def confirmation_times(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    ts = _ts(df)
    delay = pd.to_timedelta(rng.integers(15, 91, len(df)), unit="D")
    out = (ts + delay).where(pd.to_numeric(df["fraud_flag"], errors="coerce").fillna(0).astype(int) == 1)
    return out


def inject_rings(df: pd.DataFrame, n_rings: int = 15, seed: int = 11) -> Dict:
    rng = np.random.default_rng(seed)
    base = assign_payees(df, rng)
    base["label_confirmed_at"] = confirmation_times(base, rng)
    ts = _ts(base)
    start, end = ts.min(), ts.max()
    by_customer = {c: g for c, g in base.groupby("customer_id")}
    customers = list(by_customer)
    rows = []
    n_txn = 0

    def row_from(template: pd.Series, **fields) -> Dict:
        r = template.to_dict()
        r.update(fields)
        return r

    for ring in range(n_rings):
        mules = [f"CUST-M{ring:03d}{m}" for m in range(rng.integers(3, 9))]
        devices = [f"DEV-RING-{ring:03d}-{d}" for d in range(rng.integers(1, 3))]
        ips = [f"10.{ring}.{d}.{rng.integers(1, 250)}" for d in range(rng.integers(1, 3))]
        active = start + timedelta(days=int(rng.integers(30, max((end - start).days - 70, 31))))
        window = int(rng.integers(14, 61))
        uncovered = active + timedelta(days=window + int(rng.integers(20, 61)))
        country_row = base.sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
        for mule in mules:
            seg_row = base.sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
            profile = {"customer_id": mule, "account_id": mule.replace("CUST", "ACC"),
                       "customer_segment": seg_row["customer_segment"],
                       "customer_age_band": seg_row["customer_age_band"],
                       "country": country_row["country"], "geography": country_row["geography"],
                       "city": country_row["city"], "currency": country_row["currency"]}
            for w in range(rng.integers(1, 4)):                          # account warming, genuine
                t = active - timedelta(days=int(rng.integers(3, 30)), minutes=int(rng.integers(0, 1440)))
                tpl = base[base["transaction_type"] == "Purchase"].sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
                n_txn += 1
                rows.append(row_from(tpl, **profile, transaction_id=f"TXN-R{n_txn:07d}",
                                     transaction_date=t.strftime("%Y-%m-%d"), transaction_time=t.strftime("%H:%M:%S"),
                                     device_id=devices[rng.integers(0, len(devices))],
                                     ip_location=ips[rng.integers(0, len(ips))], payee_id=None,
                                     fraud_flag=0, fraud_type="None", label_confirmed_at=pd.NaT))
        inbound_total = {m: 0.0 for m in mules}
        for v in range(rng.integers(5, 26)):                             # victim payments into mule accounts
            victim = customers[rng.integers(0, len(customers))]
            hist = by_customer[victim]
            tpl = hist.sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
            t = active + timedelta(days=int(rng.integers(0, window)), minutes=int(rng.integers(0, 1440)))
            amount = round(float(hist["transaction_amount"].median()) * float(rng.uniform(1, 3)), 2)
            mule = mules[rng.integers(0, len(mules))]
            inbound_total[mule] += amount
            n_txn += 1
            rows.append(row_from(tpl, transaction_id=f"TXN-R{n_txn:07d}", transaction_type="Transfer",
                                 debit_credit_flag="Debit", transaction_amount=amount,
                                 transaction_date=t.strftime("%Y-%m-%d"), transaction_time=t.strftime("%H:%M:%S"),
                                 payee_id=f"PAYEE-M-{mule}", fraud_flag=1, fraud_type="Authorised Push Payment",
                                 label_confirmed_at=t + timedelta(days=int(rng.integers(1, 11)))))
        for mule in mules:                                               # cash-out through the shared devices
            total = inbound_total[mule] or float(rng.uniform(500, 3000))
            k = int(rng.integers(2, 7))
            tpl = base[base["transaction_type"] == "Transfer"].sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
            for part in rng.dirichlet(np.ones(k)) * total:
                t = active + timedelta(days=int(rng.integers(1, window + 1)), minutes=int(rng.integers(0, 1440)))
                n_txn += 1
                rows.append(row_from(tpl, customer_id=mule, account_id=mule.replace("CUST", "ACC"),
                                     country=country_row["country"], geography=country_row["geography"],
                                     city=country_row["city"], currency=country_row["currency"],
                                     transaction_id=f"TXN-R{n_txn:07d}", transaction_type="Transfer",
                                     debit_credit_flag="Debit", transaction_amount=round(float(part), 2),
                                     transaction_date=t.strftime("%Y-%m-%d"), transaction_time=t.strftime("%H:%M:%S"),
                                     device_id=devices[rng.integers(0, len(devices))],
                                     ip_location=ips[rng.integers(0, len(ips))], payee_id=f"PAYEE-X-{ring:03d}",
                                     account_balance_before=round(total * float(rng.uniform(1.0, 1.5)), 2),
                                     historical_average_transaction_amount=round(total / k, 2),
                                     fraud_flag=1, fraud_type="Mule Account", label_confirmed_at=uncovered))
    rings = pd.DataFrame(rows)
    combined = pd.concat([base, rings], ignore_index=True)
    combined["synthetic_ring"] = combined["transaction_id"].astype(str).str.startswith("TXN-R").astype(int)
    manifest = {"notice": NOTICE, "seed": seed, "rings": n_rings, "base_rows": int(len(base)),
                "ring_rows": int(len(rings)), "ring_fraud_rows": int(rings["fraud_flag"].sum()),
                "ring_fraud_share_of_all_fraud": round(float(rings["fraud_flag"].sum()
                                                              / combined["fraud_flag"].astype(int).sum()), 4),
                "payment_rows_with_payee": int(combined["payee_id"].notna().sum())}
    return {"data": combined, "manifest": manifest}


def write(df: pd.DataFrame, out_dir: Path, n_rings: int = 15, seed: int = 11) -> Dict:
    result = inject_rings(df, n_rings, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "banking_transactions_with_synthetic_rings.csv"
    result["data"].to_csv(path, index=False)
    (out_dir / "synthetic_rings_manifest.json").write_text(json.dumps(result["manifest"], indent=2))
    return {"path": str(path), **result["manifest"]}
