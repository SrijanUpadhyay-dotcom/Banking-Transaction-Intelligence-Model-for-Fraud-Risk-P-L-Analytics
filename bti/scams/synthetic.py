# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Synthetic APP-scam and mule-account generator: a capability test harness, not
training data for a real book.

It builds on the Phase 6 ring data (`bti.graph.synthetic_rings`) and adds what
scam and mule models need, which the base data does not carry.

**Payee attributes** on every payment, from feeds a bank would have:
- *Payee account opening date.* Known for on-us payees; for about 60% of
  external payees through data sharing; otherwise unknown.
- *Confirmation-of-Payee result* (`match` / `close_match` / `no_match` /
  `unavailable`), fixed per customer–payee pair.
  - Genuine payees mostly match.
  - Scam payees often match too, because scammers open accounts in matching
    names or coach victims past the warning. They are only somewhat more often
    `no_match` or `close_match`.

**APP-scam episodes.** Genuine customers pay new scam payees, using their own
device, IP and channel, so authentication signals stay normal:

| Typology | Pattern |
|---|---|
| Purchase | one payment near the usual amount |
| Impersonation / safe account | 1–4 large payments in a day, to a days-old account, often on-us |
| Investment | 2–8 escalating payments over weeks |
| Romance | 3–12 payments over months |

Each episode is confirmed when the victim reports it, later for investment and
romance scams.

**Benign lookalikes,** so no single signal gives the answer away:
- large first payments to new payees, some with young accounts or name
  near-misses (a house deposit, a private car sale)
- recurring payments to the same payee (rent, savings, family support), 8%
  to young accounts, so repeat payments are not themselves suspicious
- busy small-business, landlord and student accounts that receive from many
  senders but keep the money
- benign on-us payees: 15% of customers' regular payees bank here too, so
  "the payee is our customer" is not itself suspicious
- households (500 groups of 2–4 customers) sharing a device and their home
  broadband IP for part of their activity, while half the ring mules use their own phones, so a shared
  device alone does not identify a mule
- benign pass-through accounts (150): salary moved on to the customer's own
  savings elsewhere, or withdrawn as cash, within a day
- new customers: 12% of accounts opened shortly before their first activity

**Mule accounts.**
- *Inbound credits.* Every scam or ring payment into an on-us mule also
  appears as a credit on the mule's account, with the sender as
  `counterparty_id`.
- *Standalone mules.* Young accounts that receive from many external senders
  and pass 70–95% straight out (transfers, ATM, crypto) within two days.
- *Labels.* Mule labels go in a separate table: activation date, and the date
  the account is uncovered.

**Fairness.** Mules and victims are drawn at random across segments and ages.
Real mule herders target young people, but a generator that copied that would
teach a model age by proxy.

`is_vulnerable` (about 8% of customers) is a protected attribute. It is used
only for reimbursement exposure, never as a model input.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from bti.graph.synthetic_rings import PAYMENT_TYPES, _ts, inject_rings

NOTICE = ("SYNTHETIC scams and mule accounts injected into a copy of synthetic data for capability testing; "
          "not evidence of performance on a real book.")
COP_GENUINE = (("match", 0.92), ("close_match", 0.04), ("no_match", 0.02), ("unavailable", 0.02))
COP_SCAM = (("match", 0.55), ("close_match", 0.15), ("no_match", 0.20), ("unavailable", 0.10))
TYPOLOGIES = (("purchase", 0.35), ("impersonation", 0.20), ("investment", 0.25), ("romance", 0.20))


def _pick(rng, table):
    names, probs = zip(*table)
    return names[rng.choice(len(names), p=probs)]


def inject_scams(df: pd.DataFrame, n_rings: int = 15, n_episodes: int = 260, n_benign_large: int = 600,
                 n_standalone_mules: int = 40, n_busy_benign: int = 80, n_recurring_benign: int = 900,
                 n_households: int = 500, n_passthrough_benign: int = 150, new_customer_share: float = 0.12,
                 seed: int = 21) -> Dict:
    rng = np.random.default_rng(seed)
    rings = inject_rings(df, n_rings, seed)
    data = rings["data"]
    data["synthetic_scam"] = 0
    data["scam_typology"] = None
    data["counterparty_id"] = None
    data["payee_customer_id"] = None
    ts = _ts(data)
    start, end = ts.min(), ts.max()
    customers = data.loc[~data["customer_id"].astype(str).str.startswith("CUST-M"), "customer_id"].astype(str).unique()
    by_customer = {c: g for c, g in data.groupby("customer_id")}
    pay_mask = data["transaction_type"].isin(PAYMENT_TYPES) & (data["debit_credit_flag"] == "Debit")
    rows: List[Dict] = []
    n = [0]

    def new_id():
        n[0] += 1
        return f"TXN-S{n[0]:07d}"

    def row(template: pd.Series, t: pd.Timestamp, **fields) -> Dict:
        r = template.to_dict()
        r.update(transaction_id=new_id(), transaction_date=t.strftime("%Y-%m-%d"),
                 transaction_time=t.strftime("%H:%M:%S"), synthetic_scam=1, **fields)
        r.pop("synthetic_scam_benign", None)
        return r

    # ── on-us mules: ring mules plus standalone ones ──
    ring_mules = sorted({c for c in data["customer_id"].astype(str) if c.startswith("CUST-M")})
    mule_info: Dict[str, Dict] = {}
    for m in ring_mules:
        g = data[data["customer_id"] == m]
        mt = _ts(g)
        mule_info[m] = {"kind": "ring", "activated_at": mt[g["fraud_flag"] == 1].min(),
                        "uncovered_at": pd.to_datetime(g["label_confirmed_at"]).max(), "template": g.iloc[0]}
    for k in range(n_standalone_mules):
        cid = f"CUST-SM{k:04d}"
        active = start + timedelta(days=int(rng.integers(60, (end - start).days - 60)))
        tpl = data.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
        mule_info[cid] = {"kind": "standalone", "activated_at": active,
                          "uncovered_at": active + timedelta(days=int(rng.integers(20, 75))), "template": tpl,
                          "window": int(rng.integers(7, 42))}
    onus_mules = list(mule_info)

    # ── APP-scam episodes ──
    for e in range(n_episodes):
        victim = customers[rng.integers(0, len(customers))]
        hist = by_customer[victim]
        tpl = hist.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
        median = float(pd.to_numeric(hist["transaction_amount"], errors="coerce").median() or 100.0)
        balance = float(pd.to_numeric(hist["account_balance_before"], errors="coerce").max() or median * 10)
        typ = _pick(rng, TYPOLOGIES)
        t0 = start + timedelta(days=int(rng.integers(30, (end - start).days - 30)), minutes=int(rng.integers(0, 1440)))
        on_us = typ == "impersonation" and rng.random() < 0.4 or rng.random() < 0.1
        mule = onus_mules[rng.integers(0, len(onus_mules))] if on_us else None
        payee = f"PAYEE-M-{mule}" if mule else f"PAYEE-SC-{e:05d}"
        if typ == "purchase":
            times, amounts = [t0], [median * rng.uniform(0.5, 2.0)]
        elif typ == "impersonation":
            k = int(rng.integers(1, 5))
            times = [t0 + timedelta(minutes=int(m)) for m in np.sort(rng.integers(0, 600, k))]
            amounts = list(np.minimum(median * rng.uniform(3, 10, k), max(balance, median) * rng.uniform(0.3, 0.9, k)))
        elif typ == "investment":
            k = int(rng.integers(2, 9))
            times = [t0 + timedelta(days=int(d)) for d in np.sort(rng.integers(0, 70, k))]
            amounts = list(median * np.linspace(1, rng.uniform(2, 5), k) * rng.uniform(0.8, 1.2, k))
        else:
            k = int(rng.integers(3, 13))
            times = [t0 + timedelta(days=int(d)) for d in np.sort(rng.integers(0, 180, k))]
            amounts = list(median * rng.uniform(0.5, 2.5, k))
        last = max(times)
        delay = rng.integers(1, 30) if typ in ("purchase", "impersonation") else rng.integers(14, 90)
        confirmed = last + timedelta(days=int(delay))
        for t, amt in zip(times, amounts):
            if t > end:
                continue
            rows.append(row(tpl, t, transaction_type="Transfer", debit_credit_flag="Debit",
                            channel=tpl["channel"] if tpl["channel"] in ("Mobile Banking", "Internet Banking")
                            else "Mobile Banking", transaction_amount=round(float(amt), 2), payee_id=payee,
                            payee_customer_id=mule, fraud_flag=1, fraud_type="Authorised Push Payment",
                            scam_typology=typ, label_confirmed_at=confirmed))
            if mule:
                mtpl = mule_info[mule]["template"]
                rows.append(row(mtpl, t + timedelta(seconds=int(rng.integers(5, 120))), customer_id=mule,
                                account_id=mule.replace("CUST", "ACC"), transaction_type="Transfer",
                                debit_credit_flag="Credit", transaction_amount=round(float(amt), 2),
                                counterparty_id=f"CP-{victim}", payee_id=None, fraud_flag=0, fraud_type="None",
                                label_confirmed_at=pd.NaT))

    # ring victim payments also land on the ring mule's account as credits
    ring_pay = data[data["payee_id"].astype(str).str.startswith("PAYEE-M-")]
    data.loc[ring_pay.index, "payee_customer_id"] = ring_pay["payee_id"].str.replace("PAYEE-M-", "", regex=False)
    for _, r in ring_pay.iterrows():
        mule = r["payee_id"].replace("PAYEE-M-", "")
        t = _ts(pd.DataFrame([r])).iloc[0] + timedelta(seconds=int(rng.integers(5, 120)))
        rows.append(row(mule_info[mule]["template"], t, customer_id=mule, account_id=mule.replace("CUST", "ACC"),
                        transaction_type="Transfer", debit_credit_flag="Credit",
                        transaction_amount=r["transaction_amount"], counterparty_id=f"CP-{r['customer_id']}",
                        payee_id=None, fraud_flag=0, fraud_type="None", label_confirmed_at=pd.NaT))

    # ── standalone mules: external inflows, rapid pass-through ──
    for cid, info in mule_info.items():
        if info["kind"] != "standalone":
            continue
        tpl, active, window = info["template"], info["activated_at"], info["window"]
        profile = dict(customer_id=cid, account_id=cid.replace("CUST", "ACC"), payee_id=None, label_confirmed_at=pd.NaT,
                       device_id=f"DEV-SM-{cid}", ip_location=f"172.{rng.integers(16, 31)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}")
        for w in range(int(rng.integers(1, 4))):                        # warming
            t = active - timedelta(days=int(rng.integers(2, 25)), minutes=int(rng.integers(0, 1440)))
            rows.append(row(tpl, t, **profile, transaction_type="Purchase", debit_credit_flag="Debit",
                            transaction_amount=round(float(rng.uniform(5, 60)), 2), fraud_flag=0, fraud_type="None"))
        slow = rng.random() < 0.3                                       # "low and slow": few senders, slower exit
        senders = [f"CP-EXT-{rng.integers(0, 10**7):07d}" for _ in range(int(rng.integers(2, 5)))] if slow else None
        for i in range(int(rng.integers(4, 10) if slow else rng.integers(5, 31))):
            t = active + timedelta(days=int(rng.integers(0, window)), minutes=int(rng.integers(0, 1440)))
            amt = round(float(rng.lognormal(6.3, 0.8)), 2)
            rows.append(row(tpl, t, **{**profile, "payee_id": None}, transaction_type="Transfer",
                            debit_credit_flag="Credit", transaction_amount=amt,
                            counterparty_id=senders[rng.integers(0, len(senders))] if slow
                            else f"CP-EXT-{rng.integers(0, 10**7):07d}", fraud_flag=0, fraud_type="None"))
            for part in rng.dirichlet(np.ones(int(rng.integers(1, 3)))) * amt * rng.uniform(0.7, 0.95):
                out_t = t + timedelta(hours=float(rng.uniform(24, 120) if slow else rng.uniform(0.2, 48)))
                kind = rng.choice(["transfer", "atm", "crypto"], p=[0.6, 0.25, 0.15])
                fields = {"transaction_type": "Transfer", "payee_id": f"PAYEE-X-SM{rng.integers(0, 50):03d}"} \
                    if kind == "transfer" else \
                    {"transaction_type": "Withdrawal", "channel": "ATM", "payee_id": None} if kind == "atm" else \
                    {"transaction_type": "Purchase", "merchant_category": "Crypto Exchanges", "payee_id": None}
                rows.append(row(tpl, out_t, **{**profile, **fields, "label_confirmed_at": info["uncovered_at"]},
                                debit_credit_flag="Debit", transaction_amount=round(float(part), 2), fraud_flag=1,
                                fraud_type="Mule Account"))

    # ── benign lookalikes ──
    for b in range(n_benign_large):                                     # big first payments to new payees
        c = customers[rng.integers(0, len(customers))]
        hist = by_customer[c]
        tpl = hist.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
        median = float(pd.to_numeric(hist["transaction_amount"], errors="coerce").median() or 100.0)
        t = start + timedelta(days=int(rng.integers(30, (end - start).days)), minutes=int(rng.integers(0, 1440)))
        rows.append(row(tpl, t, transaction_type="Transfer", debit_credit_flag="Debit",
                        channel="Mobile Banking" if rng.random() < 0.6 else "Internet Banking",
                        transaction_amount=round(median * float(rng.uniform(2, 15)), 2),
                        payee_id=f"PAYEE-BN-{b:05d}", fraud_flag=0, fraud_type="None", label_confirmed_at=pd.NaT))
    for b in range(n_recurring_benign):                                 # rent, savings, family support
        c = customers[rng.integers(0, len(customers))]
        hist = by_customer[c]
        tpl = hist.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
        median = float(pd.to_numeric(hist["transaction_amount"], errors="coerce").median() or 100.0)
        every = int(rng.choice([7, 14, 30]))
        first = start + timedelta(days=int(rng.integers(0, (end - start).days - 60)), minutes=int(rng.integers(0, 1440)))
        amount = median * float(rng.uniform(0.5, 4))
        young = rng.random() < 0.08                                     # e.g. a new landlord's account
        payee = f"PAYEE-RB{'Y' if young else ''}-{b:05d}"
        for k in range(int(rng.integers(3, 20))):
            t = first + timedelta(days=k * every + int(rng.integers(-1, 2)))
            if t > end:
                break
            rows.append(row(tpl, t, transaction_type="Transfer", debit_credit_flag="Debit",
                            transaction_amount=round(amount * float(rng.uniform(0.9, 1.1)), 2), payee_id=payee,
                            fraud_flag=0, fraud_type="None", label_confirmed_at=pd.NaT))
    for c in rng.choice(customers, n_passthrough_benign, replace=False):   # salary straight on to savings / cash
        tpl = by_customer[c].iloc[0]
        employer = f"CP-EMP-{rng.integers(0, 10**6):06d}"
        own = [f"PAYEE-OWN-{c}-{k}" for k in range(int(rng.integers(1, 3)))]
        cash_user = rng.random() < 0.3
        first = start + timedelta(days=int(rng.integers(0, 120)))
        for k in range(int(rng.integers(6, 24))):
            t = first + timedelta(days=30 * k + int(rng.integers(-2, 3)), hours=float(rng.uniform(0, 12)))
            if t > end:
                break
            amt = float(rng.lognormal(7.3, 0.4))
            rows.append(row(tpl, t, transaction_type="Deposit", debit_credit_flag="Credit",
                            transaction_amount=round(amt, 2), counterparty_id=employer, payee_id=None, fraud_flag=0,
                            fraud_type="None", label_confirmed_at=pd.NaT))
            out_t = t + timedelta(hours=float(rng.uniform(1, 30)))
            fields = {"transaction_type": "Withdrawal", "channel": "ATM", "payee_id": None} if cash_user else \
                {"transaction_type": "Transfer", "payee_id": own[rng.integers(0, len(own))]}
            rows.append(row(tpl, out_t, **fields, debit_credit_flag="Debit",
                            transaction_amount=round(amt * float(rng.uniform(0.8, 1.0)), 2), fraud_flag=0,
                            fraud_type="None", label_confirmed_at=pd.NaT))
    busy = rng.choice(customers, n_busy_benign, replace=False)
    for c in busy:                                                      # many senders, money stays
        tpl = by_customer[c].iloc[0]
        senders = [f"CP-EXT-B{rng.integers(0, 10**7):07d}" for _ in range(int(rng.integers(5, 40)))]
        for _ in range(int(rng.integers(10, 60))):
            t = start + timedelta(days=int(rng.integers(0, (end - start).days)), minutes=int(rng.integers(0, 1440)))
            rows.append(row(tpl, t, transaction_type="Transfer", debit_credit_flag="Credit",
                            transaction_amount=round(float(rng.lognormal(5.5, 0.7)), 2), payee_id=None,
                            counterparty_id=senders[rng.integers(0, len(senders))], fraud_flag=0, fraud_type="None",
                            label_confirmed_at=pd.NaT))

    # benign on-us payees: 15% of regular payees are other customers of the bank (family, landlords, friends)
    regular = data["payee_id"].astype(str).str.fullmatch(r"PAYEE-\d{7}")
    pool = pd.Series(data.loc[regular, "payee_id"].unique())
    onus_payees = pool[rng.random(len(pool)) < 0.15]
    owner = {pid: customers[rng.integers(0, len(customers))] for pid in onus_payees}
    hit = data["payee_id"].isin(owner)
    data.loc[hit, "payee_customer_id"] = data.loc[hit, "payee_id"].map(owner)
    for _, r in data[hit].iterrows():
        recipient = owner[r["payee_id"]]
        t = _ts(pd.DataFrame([r])).iloc[0] + timedelta(seconds=int(rng.integers(5, 120)))
        rows.append(row(by_customer[recipient].iloc[0], t, customer_id=recipient, transaction_type="Transfer",
                        debit_credit_flag="Credit", transaction_amount=r["transaction_amount"],
                        counterparty_id=f"CP-{r['customer_id']}", payee_id=None, fraud_flag=0, fraud_type="None",
                        label_confirmed_at=pd.NaT))

    combined = pd.concat([data, pd.DataFrame(rows)], ignore_index=True)
    combined = _devices(combined, mule_info, customers, n_households, rng)
    combined = _payee_and_account_attributes(combined, mule_info, rng, new_customer_share)
    mules = pd.DataFrame([{"customer_id": c, "kind": i["kind"], "activated_at": i["activated_at"],
                           "uncovered_at": i["uncovered_at"]} for c, i in mule_info.items()])
    scam = combined[(combined["fraud_type"] == "Authorised Push Payment")]
    manifest = {"notice": NOTICE, "seed": seed, "rows": int(len(combined)), "injected_rows": int(len(rows)),
                "app_scam_payments": int(len(scam)),
                "app_scam_by_typology": scam["scam_typology"].fillna("ring").value_counts().to_dict(),
                "mule_accounts": {"ring": int((mules["kind"] == "ring").sum()),
                                  "standalone": int((mules["kind"] == "standalone").sum())},
                "benign_large_new_payee_payments": n_benign_large, "busy_benign_accounts": n_busy_benign,
                "recurring_benign_series": n_recurring_benign, "households_sharing_devices": n_households,
                "benign_pass_through_accounts": n_passthrough_benign, "new_customer_share": new_customer_share}
    return {"data": combined, "mules": mules, "manifest": manifest}


def _devices(df: pd.DataFrame, mule_info: Dict, customers, n_households: int, rng) -> pd.DataFrame:
    """
    Benign device sharing (households of 2–4 customers sharing a tablet or laptop for part of their activity), and
    recruited ring mules who use their own phones: half the ring mules move to a personal device. Without this,
    a shared device alone would identify a mule.
    """
    out = df.copy()
    for h in range(n_households):
        members = rng.choice(customers, int(rng.integers(2, 5)), replace=False)
        rows = out.index[out["customer_id"].isin(members)]
        use = rows[rng.random(len(rows)) < 0.5]
        out.loc[use, "device_id"] = f"DEV-HH-{h:04d}"
        home = rows[rng.random(len(rows)) < 0.6]                        # home broadband
        out.loc[home, "ip_location"] = f"192.168.{h // 250}.{h % 250 + 1}"
    for m, info in mule_info.items():
        if info["kind"] == "ring" and rng.random() < 0.5:
            out.loc[out["customer_id"] == m, "device_id"] = f"DEV-OWN-{m}"
    return out


def _payee_and_account_attributes(df: pd.DataFrame, mule_info: Dict, rng: np.random.Generator,
                                  new_customer_share: float = 0.12) -> pd.DataFrame:
    out = df.copy()
    ts = _ts(out)
    # customer account opening dates (CIF)
    first = ts.groupby(out["customer_id"]).min()
    opened = {c: f - timedelta(days=int(rng.integers(1, 60) if rng.random() < new_customer_share
                                         else rng.integers(30, 3650))) for c, f in first.items()}
    for c, info in mule_info.items():
        if c in first.index:
            young = info["kind"] == "standalone" or rng.random() < 0.75
            opened[c] = first[c] - timedelta(days=int(rng.integers(1, 45) if young else rng.integers(180, 2000)))
    out["account_opened_date"] = out["customer_id"].map(opened).dt.strftime("%Y-%m-%d")
    vulnerable = {c: bool(rng.random() < 0.08) for c in first.index}
    out["is_vulnerable"] = out["customer_id"].map(vulnerable).astype(int)
    # base credits get regular counterparties (employer, landlord tenant, own transfers)
    credit = (out["debit_credit_flag"] == "Credit") & out["counterparty_id"].isna()
    out.loc[credit, "counterparty_id"] = [f"CP-{c}-{rng.integers(0, 3)}" for c in out.loc[credit, "customer_id"]]
    # payee attributes: account opening date and Confirmation-of-Payee, per payee / customer-payee pair
    pay = out["payee_id"].notna() & (out["debit_credit_flag"] == "Debit")
    payees = out.loc[pay, "payee_id"].astype(str)
    first_use = ts[pay].groupby(payees).min()
    onus_owner = out.loc[pay & out["payee_customer_id"].notna(), ["payee_id", "payee_customer_id"]] \
        .drop_duplicates("payee_id").set_index("payee_id")["payee_customer_id"].astype(str).to_dict()
    payee_opened, coverage = {}, {}
    for p, f in first_use.items():
        if p in onus_owner and not p.startswith("PAYEE-M-") and onus_owner[p] in opened:
            payee_opened[p] = pd.Timestamp(min(opened[onus_owner[p]], f))
        elif p.startswith("BILLER-"):
            payee_opened[p] = f - timedelta(days=int(rng.integers(1800, 7300)))
        elif p.startswith("PAYEE-M-"):
            payee_opened[p] = pd.Timestamp(opened.get(p.replace("PAYEE-M-", ""), f - timedelta(days=30)))
        elif p.startswith("PAYEE-SC-"):
            payee_opened[p] = f - timedelta(days=int(rng.integers(3, 120) if rng.random() < 0.75 else rng.integers(200, 3000)))
        elif p.startswith("PAYEE-RBY-"):
            payee_opened[p] = f - timedelta(days=int(rng.integers(5, 90)))
        elif p.startswith("PAYEE-RB-"):
            payee_opened[p] = f - timedelta(days=int(rng.integers(200, 5000)))
        elif p.startswith("PAYEE-BN-"):
            payee_opened[p] = f - timedelta(days=int(rng.integers(5, 120) if rng.random() < 0.3 else rng.integers(200, 5000)))
        else:
            payee_opened[p] = f - timedelta(days=int(rng.integers(0, 90) if rng.random() < 0.10 else rng.integers(365, 5500)))
        on_us = p in onus_owner                                  # own records for every on-us payee
        coverage[p] = on_us or rng.random() < 0.6
    out["payee_account_opened_date"] = None
    known = pay & payees.reindex(out.index).map(coverage).fillna(False).astype(bool)
    out.loc[known, "payee_account_opened_date"] = payees[known].map(payee_opened).dt.strftime("%Y-%m-%d")
    pairs = out.loc[pay, ["customer_id", "payee_id"]].astype(str).drop_duplicates()
    scam_payee = lambda p: p.startswith(("PAYEE-M-", "PAYEE-SC-"))
    cop = {}
    for c, p in pairs.itertuples(index=False):
        table = COP_SCAM if scam_payee(p) else COP_GENUINE
        if p.startswith("PAYEE-BN-") and rng.random() < 0.15:
            cop[(c, p)] = "close_match"
        else:
            cop[(c, p)] = _pick(rng, table)
    out["cop_result"] = None
    keys = list(zip(out.loc[pay, "customer_id"].astype(str), out.loc[pay, "payee_id"].astype(str)))
    out.loc[pay, "cop_result"] = [cop[k] for k in keys]
    onus = out["payee_id"].astype(str).str.startswith("PAYEE-M-") & out["payee_customer_id"].isna()
    out.loc[onus, "payee_customer_id"] = out.loc[onus, "payee_id"].str.replace("PAYEE-M-", "", regex=False)
    return out


def write(df: pd.DataFrame, out_dir: Path = Path("data/processed/scams"), seed: int = 21) -> Dict:
    r = inject_scams(df, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    r["data"].to_csv(out_dir / "banking_transactions_with_scams.csv", index=False)
    r["mules"].to_csv(out_dir / "mule_labels.csv", index=False)
    (out_dir / "synthetic_scams_manifest.json").write_text(json.dumps(r["manifest"], indent=2, default=str))
    return r["manifest"]


def main() -> None:
    from bti.modeling.train import default_data_path
    manifest = write(pd.read_csv(default_data_path(), low_memory=False))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
