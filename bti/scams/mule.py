# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Mule-account detection: is this customer's account being used to receive and
move on the proceeds of fraud?

**Unit.** One snapshot per account per week. Every account active in the
previous 30 days is included. Features use only events before the snapshot
date.

**Inbound flows:**
- credits in 7 and 30 days
- distinct senders in 30 days, and the share of them never seen before
- inbound value

**Pass-through:**
- outbound ÷ inbound value
- share of inbound money sent out again within two days (`fast_out_share`)
- share of outflow that is cash or crypto
- new payees paid

**Account:** age in days, from the customer master.

**Reported inbound fraud:** payments into this account that the sender's
bank (here, the bank itself, on-us) had confirmed as fraud before the
snapshot. In the UK, sending banks notify the receiving bank of APP-scam
claims, and that report is often the first hard signal.

**Network (Phase 6):** customers sharing this account's devices or IPs, and
confirmed fraud in its network component, with day-lagged confirmations.

**Label.** A snapshot is positive from a mule's activation date until it is
uncovered. Snapshots before activation are left out (ambiguous), and so are
snapshots after the account is closed.

**Evaluation** (out-of-time, by snapshot date). Each weekly scan raises the
top `mule_alert_budget` share of active accounts. The report gives:
- mules detected, days from activation to the first alert, and lead time
  before the account was uncovered
- false alerts per scan
- a simple rule as the baseline: 5+ senders and 70% pass-through
- customer-level fairness, by the same test as the other models

**Live.** `run_scan(db)` scores today's snapshot from the transactions table.
Accounts in the budget are raised as `mule_alerts`, with an audit event and a
webhook. Analysts confirm or clear them through the API. Payments to an on-us
payee under an open or confirmed alert are held by the scam overlay.

    python -m bti.scams.mule --developer "<name>"
"""

from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from bti.modeling import registry
from bti.modeling.calibration import PlattCalibrator
from bti.modeling.features import FeatureSpec, Kind, assert_feature_lineage, event_timestamps
from bti.modeling.fx import RATES_TO_USD

DATA = Path("data/processed/scams/banking_transactions_with_scams.csv")
LABELS = Path("data/processed/scams/mule_labels.csv")
OUT = Path("outputs/scams/mule_model_evaluation.json")
_N = Kind.NUMERIC
_W = ("transaction_date", "transaction_time")
_IN = ("customer_id", "counterparty_id", "debit_credit_flag", "transaction_amount", "currency") + _W
_OUT = ("customer_id", "debit_credit_flag", "transaction_type", "transaction_amount", "currency", "merchant_category",
        "channel", "payee_id") + _W
_G = ("customer_id", "device_id", "ip_location", "confirmed_fraud_labels") + _W

MULE_FEATURES: List[FeatureSpec] = [
    FeatureSpec("account_age_days", _N, ("account_opened_date",) + _W, "PAYEE_RISK", "Days since the account opened"),
    FeatureSpec("in_count_7d", _N, _IN, "MULE_FLOW", "Inbound credits in 7 days"),
    FeatureSpec("in_count_30d", _N, _IN, "MULE_FLOW", "Inbound credits in 30 days"),
    FeatureSpec("in_value_30d_usd", _N, _IN, "MULE_FLOW", "Inbound value in 30 days (USD)"),
    FeatureSpec("in_distinct_senders_30d", _N, _IN, "MULE_FLOW", "Distinct senders in 30 days"),
    FeatureSpec("in_new_sender_share", _N, _IN, "MULE_FLOW", "Share of 30-day senders never seen before"),
    FeatureSpec("out_value_30d_usd", _N, _OUT, "MULE_FLOW", "Outbound value in 30 days (USD)"),
    FeatureSpec("pass_through_ratio", _N, _IN + _OUT, "MULE_FLOW", "Outbound ÷ inbound value over 30 days"),
    FeatureSpec("fast_out_share", _N, _IN + _OUT, "MULE_FLOW", "Share of inbound value sent on within two days"),
    FeatureSpec("cashout_share", _N, _OUT, "MULE_FLOW", "Share of outflow taken as cash (ATM) or crypto"),
    FeatureSpec("out_new_payees_30d", _N, _OUT, "MULE_FLOW", "New payees paid in 30 days"),
    FeatureSpec("inbound_reported_fraud_30d", _N, ("payee_customer_id", "confirmed_fraud_labels") + _W, "MULE_PAYEE",
                "Payments into this account confirmed as fraud before the snapshot (sending-bank reports)"),
    FeatureSpec("graph_shared_entity_customers", _N, _G, "NETWORK_LINK",
                "Most other customers seen on this account's devices or IPs"),
    FeatureSpec("graph_component_known_fraud", _N, _G, "NETWORK_LINK", "Confirmed fraud in this account's network"),
]
MULE_FEATURE_NAMES = [f.name for f in MULE_FEATURES]
MULE_FEATURE_BY_NAME = {f.name: f for f in MULE_FEATURES}
assert_feature_lineage(MULE_FEATURES)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"customer_id": df["customer_id"].astype(str), "ts": event_timestamps(df)})
    d["day"] = d["ts"].dt.normalize()
    d["usd"] = (pd.to_numeric(df["transaction_amount"], errors="coerce") *
                df["currency"].astype(str).str.upper().map(RATES_TO_USD)).fillna(0).to_numpy()
    d["credit"] = (df["debit_credit_flag"].astype(str) == "Credit").to_numpy()
    d["counterparty"] = df["counterparty_id"].astype("string") if "counterparty_id" in df else pd.NA
    d["payee"] = df["payee_id"].astype("string") if "payee_id" in df else pd.NA
    d["cashout"] = ((df["transaction_type"] == "Withdrawal") | (df.get("channel") == "ATM")
                    | (df.get("merchant_category") == "Crypto Exchanges")).to_numpy()
    d["opened"] = pd.to_datetime(df.get("account_opened_date"), errors="coerce") if "account_opened_date" in df else pd.NaT
    d["payee_customer"] = df["payee_customer_id"].astype("string") if "payee_customer_id" in df else pd.NA
    d["fraud"] = pd.to_numeric(df.get("fraud_flag", 0), errors="coerce").fillna(0).astype(int).to_numpy()
    d["confirmed_at"] = pd.to_datetime(df.get("label_confirmed_at"), errors="coerce") if "label_confirmed_at" in df \
        else pd.NaT
    return d


def snapshots(df: pd.DataFrame, dates: Optional[List[pd.Timestamp]] = None, freq: str = "7D",
              graph: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per (account, snapshot date) for accounts active in the 30 days before the date."""
    d = _prep(df)
    if graph is None:
        from bti.graph.temporal import graph_features
        graph = graph_features(df)
    d["g_shared"] = graph["graph_shared_entity_customers"].to_numpy()
    d["g_component"] = graph["graph_component_known_fraud"].to_numpy()
    if dates is None:
        dates = list(pd.date_range(d["day"].min() + pd.Timedelta(days=35), d["day"].max(), freq=freq))
    opened = d.groupby("customer_id")["opened"].min()
    first_sender = d[d["credit"]].groupby(["customer_id", "counterparty"])["ts"].min()
    first_payee = d[~d["credit"] & d["payee"].notna()].groupby(["customer_id", "payee"])["ts"].min()
    rows = []
    for D in dates:
        w = d[(d["ts"] < D) & (d["ts"] >= D - pd.Timedelta(days=30))]
        if w.empty:
            continue
        cust = w["customer_id"].unique()
        inn, out = w[w["credit"]], w[~w["credit"]]
        f = pd.DataFrame(index=pd.Index(cust, name="customer_id"))
        f["in_count_7d"] = inn[inn["ts"] >= D - pd.Timedelta(days=7)].groupby("customer_id").size()
        f["in_count_30d"] = inn.groupby("customer_id").size()
        f["in_value_30d_usd"] = inn.groupby("customer_id")["usd"].sum()
        f["in_distinct_senders_30d"] = inn.groupby("customer_id")["counterparty"].nunique()
        new = inn.assign(first=[first_sender.get((c, s)) for c, s in zip(inn["customer_id"], inn["counterparty"])])
        new["is_new"] = new["first"] >= D - pd.Timedelta(days=30)
        f["in_new_sender_share"] = new.drop_duplicates(["customer_id", "counterparty"]).groupby("customer_id")["is_new"].mean()
        f["out_value_30d_usd"] = out.groupby("customer_id")["usd"].sum()
        f = f.fillna({"in_count_7d": 0, "in_count_30d": 0, "in_value_30d_usd": 0, "in_distinct_senders_30d": 0,
                      "out_value_30d_usd": 0})
        f["pass_through_ratio"] = (f["out_value_30d_usd"] / f["in_value_30d_usd"]).where(f["in_value_30d_usd"] > 0)
        # fast-out: inbound value per day matched with outbound on the same or the next two days
        din = inn.groupby(["customer_id", "day"])["usd"].sum().unstack(fill_value=0)
        dout = out.groupby(["customer_id", "day"])["usd"].sum().unstack(fill_value=0)
        days = pd.date_range(D - pd.Timedelta(days=30), D, freq="D")
        din = din.reindex(columns=days, fill_value=0)
        dout = dout.reindex(index=din.index, columns=days, fill_value=0).to_numpy()
        # match each day's inflow with outflow on the same or the next two days; each outflow is used only once
        inflow, remaining = din.to_numpy().astype(float), dout.astype(float).copy()
        matched = np.zeros(len(inflow))
        for j in range(inflow.shape[1]):
            need = inflow[:, j].copy()
            for k in range(j, min(j + 3, inflow.shape[1])):
                take = np.minimum(need, remaining[:, k])
                remaining[:, k] -= take
                need -= take
                matched += take
        total = inflow.sum(axis=1)
        f["fast_out_share"] = pd.Series(np.where(total > 0, matched / np.maximum(total, 1e-9), np.nan), index=din.index)
        cash = out[out["cashout"]].groupby("customer_id")["usd"].sum()
        f["cashout_share"] = (cash.reindex(f.index).fillna(0) / f["out_value_30d_usd"]).where(f["out_value_30d_usd"] > 0)
        po = out[out["payee"].notna()]
        newp = [first_payee.get((c, p)) for c, p in zip(po["customer_id"], po["payee"])]
        f["out_new_payees_30d"] = po.assign(first=newp)[lambda x: x["first"] >= D - pd.Timedelta(days=30)] \
            .groupby("customer_id")["payee"].nunique()
        f["out_new_payees_30d"] = f["out_new_payees_30d"].fillna(0)
        reported = d[(d["payee_customer"].notna()) & (d["fraud"] == 1) & (d["confirmed_at"] < D)
                     & (d["ts"] >= D - pd.Timedelta(days=30)) & (d["ts"] < D)]
        f["inbound_reported_fraud_30d"] = reported.groupby("payee_customer").size().reindex(f.index).fillna(0)
        f["graph_shared_entity_customers"] = w.groupby("customer_id")["g_shared"].max()
        f["graph_component_known_fraud"] = w.sort_values("ts").groupby("customer_id")["g_component"].last()
        f["account_age_days"] = (D - opened.reindex(f.index)).dt.days
        f["snapshot_date"] = D
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)[["customer_id", "snapshot_date"] + MULE_FEATURE_NAMES]


def label(snaps: pd.DataFrame, mules: pd.DataFrame) -> pd.Series:
    """1 while a mule is active, 0 for other accounts, NaN before activation or after the account is uncovered."""
    m = mules.set_index("customer_id")
    act = pd.to_datetime(snaps["customer_id"].map(m["activated_at"]))
    unc = pd.to_datetime(snaps["customer_id"].map(m["uncovered_at"]))
    is_mule = snaps["customer_id"].isin(m.index)
    y = pd.Series(0.0, index=snaps.index)
    y[is_mule] = np.nan
    active = is_mule & (snaps["snapshot_date"] >= act) & (snaps["snapshot_date"] <= unc)
    y[active] = 1.0
    return y


def _alerts(snaps: pd.DataFrame, score: np.ndarray, budget: float) -> np.ndarray:
    flag = np.zeros(len(snaps), bool)
    for _, idx in snaps.groupby("snapshot_date").indices.items():
        k = max(1, int(round(len(idx) * budget)))
        flag[idx[np.argsort(-score[idx])[:k]]] = True
    return flag


def _detection(snaps: pd.DataFrame, flag: np.ndarray, mules: pd.DataFrame, start: pd.Timestamp) -> Dict:
    m = mules.set_index("customer_id")
    tested = [c for c in m.index if pd.Timestamp(m.loc[c, "uncovered_at"]) >= start
              and pd.Timestamp(m.loc[c, "activated_at"]) <= snaps["snapshot_date"].max()]
    first = snaps[flag].groupby("customer_id")["snapshot_date"].min()
    detected, after_activation, lead = 0, [], []
    for c in tested:
        f = first.get(c)
        a, u = pd.Timestamp(m.loc[c, "activated_at"]), pd.Timestamp(m.loc[c, "uncovered_at"])
        if f is not None and a <= f <= u:
            detected += 1
            after_activation.append((f - a).days)
            lead.append((u - f).days)
    kinds = {}
    for kind in sorted(set(m["kind"])):
        ks = [c for c in tested if m.loc[c, "kind"] == kind]
        kd = sum(1 for c in ks if first.get(c) is not None
                 and pd.Timestamp(m.loc[c, "activated_at"]) <= first[c] <= pd.Timestamp(m.loc[c, "uncovered_at"]))
        kinds[kind] = {"mules": len(ks), "detected": kd}
    return {"mules_in_window": len(tested), "detected": detected, "by_kind": kinds,
            "detection_rate": round(detected / max(len(tested), 1), 3),
            "median_days_after_activation": float(np.median(after_activation)) if after_activation else None,
            "median_lead_days_before_uncovered": float(np.median(lead)) if lead else None}


def train(data_path: Optional[Path] = None, labels_path: Optional[Path] = None, developer: Optional[str] = None,
          register: bool = True, seed: int = 7) -> Dict:
    import lightgbm as lgb
    from bti.config import get_settings
    from bti.governance.fairness import fairness_assessment
    from bti.modeling.metrics import threshold_for_alert_rate
    path, lpath = Path(data_path or DATA), Path(labels_path or LABELS)
    df = pd.read_csv(path, low_memory=False)
    mules = pd.read_csv(lpath, parse_dates=["activated_at", "uncovered_at"])
    snaps = snapshots(df)
    y_all = label(snaps, mules)
    keep = y_all.notna().to_numpy()
    snaps, y = snaps[keep].reset_index(drop=True), y_all[keep].astype(int).to_numpy()
    X = snaps[MULE_FEATURE_NAMES].astype(float)
    dates = np.sort(snaps["snapshot_date"].unique())
    cut_cal, cut_test = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.8)]
    sd = snaps["snapshot_date"]
    tr, ca, te = (sd < cut_cal).to_numpy(), ((sd >= cut_cal) & (sd < cut_test)).to_numpy(), (sd >= cut_test).to_numpy()
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    model.fit(X[tr], y[tr])
    raw = model.predict_proba(X)[:, 1]
    cal = PlattCalibrator().fit(raw[ca], y[ca])
    p = cal.predict(raw)
    budget = get_settings().mule_alert_budget
    flag = _alerts(snaps, p, budget)
    rule = ((X["in_distinct_senders_30d"] >= 5) & (X["pass_through_ratio"] >= 0.7)).to_numpy()
    groups = df.drop_duplicates("customer_id").set_index(df.drop_duplicates("customer_id")["customer_id"].astype(str))
    attrs = {c: snaps["customer_id"].map(groups[c]).astype(str) for c in ("customer_segment", "customer_age_band")}
    fair = fairness_assessment(y, p, {f"budget_{budget:.1%}": threshold_for_alert_rate(p[ca], budget)},
                               {"out_of_time": te, "calibration": ca}, attrs, clusters=snaps["customer_id"].to_numpy())
    start = pd.Timestamp(cut_test)
    result = {
        "snapshots": {"total": int(len(snaps)), "positive": int(y.sum()), "test": int(te.sum()),
                      "test_positive": int(y[te].sum()), "accounts": int(snaps["customer_id"].nunique()),
                      "test_from": str(start.date())},
        "model": {"pr_auc": round(float(average_precision_score(y[te], p[te])), 4),
                  "roc_auc": round(float(roc_auc_score(y[te], p[te])), 4),
                  "alert_budget": budget,
                  "alerts_per_scan": round(float(flag[te].sum() / max(len(np.unique(sd[te])), 1)), 1),
                  "alert_precision": round(float(y[te][flag[te]].mean()), 3) if flag[te].any() else None,
                  "detection": _detection(snaps[te], flag[te], mules, start)},
        "rule_baseline": {"rule": "5+ distinct senders and pass-through >= 70% in 30 days",
                          "alerts_per_scan": round(float(rule[te].sum() / max(len(np.unique(sd[te])), 1)), 1),
                          "alert_precision": round(float(y[te][rule[te]].mean()), 3) if rule[te].any() else None,
                          "detection": _detection(snaps[te], rule[te], mules, start)},
        "fairness": {"status": fair["status"], "unit": "customer", "findings": fair["findings"],
                     "watchlist": fair["watchlist"]},
    }
    # ablation: flow and account features only (no network signals), to show what each kind of evidence carries
    flow = [c for c in MULE_FEATURE_NAMES if not c.startswith("graph_")]
    ab = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    ab.fit(X.loc[tr, flow], y[tr])
    pa = ab.predict_proba(X[flow])[:, 1]
    fa = _alerts(snaps, pa, budget)
    result["ablation_without_network_features"] = {
        "pr_auc": round(float(average_precision_score(y[te], pa[te])), 4),
        "detection": _detection(snaps[te], fa[te], mules, start)}
    imp = dict(zip(MULE_FEATURE_NAMES, model.booster_.feature_importance("gain")))
    tot = sum(imp.values()) or 1.0
    result["feature_importance_gain_share"] = {k: round(v / tot, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    beats_rule = (result["model"]["detection"]["detection_rate"] >= result["rule_baseline"]["detection"]["detection_rate"])
    gates = [
        {"gate": "feature_lineage", "passed": True, "detail": "All mule features trace to pre-authorisation fields"},
        {"gate": "beats_rule_baseline", "passed": bool(beats_rule),
         "detail": f"detection {result['model']['detection']['detection_rate']} vs rule "
                   f"{result['rule_baseline']['detection']['detection_rate']}"},
        {"gate": "fairness", "passed": fair["status"] == "pass",
         "detail": f"{len(fair['findings'])} findings, {len(fair['watchlist'])} on the watchlist"},
    ]
    result["gates"] = gates
    result["validation_status"] = "passed" if all(g["passed"] for g in gates) else "failed"
    result["notice"] = ("Synthetic mules in synthetic data: capability evidence, not performance on a real book.")
    if register:
        model_id = f"bti-mule-lgbm-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        artifact = {"estimator": model, "calibrator": cal, "feature_names": MULE_FEATURE_NAMES,
                    "unit": "account weekly snapshot"}
        card = {"model_id": model_id, "family": "mule", "purpose": "Risk that an account is operating as a money mule",
                "ownership": {"developer": developer or getpass.getuser(), "business_owner": None},
                "data": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "synthetic": True},
                "features": [{"name": f.name, "sources": list(f.sources), "description": f.description}
                             for f in MULE_FEATURES],
                "calibration": cal.describe(), "performance": result,
                "validation": {"status": result["validation_status"], "gates": gates},
                "registered_at": datetime.now(timezone.utc).isoformat()}
        registry.save_model(model_id, artifact, card, family="mule")
        registry.assign_role(model_id, "challenger", approver="bti.scams.mule",
                             rationale="Registered for shadow mule scans; promotion needs validation and four-eyes",
                             family="mule")
        result["model_id"] = model_id
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    return result


# ── live scan ────────────────────────────────────────────────────────────────
def run_scan(db, as_of: Optional[pd.Timestamp] = None, notify: bool = True) -> Dict:
    """Score today's account snapshot from the transactions table; raise the top of the budget as mule alerts."""
    from bti.config import get_settings
    from bti.database.models import AuditLog, MuleAlert, Transaction
    model_id = registry.model_for_role("champion", family="mule") or registry.model_for_role("challenger", family="mule")
    if not model_id:
        return {"status": "no mule model registered"}
    art = registry.load_artifact(model_id, family="mule")
    from sqlalchemy import func
    cols = [c.name for c in Transaction.__table__.columns]
    latest = db.query(func.max(Transaction.transaction_date)).scalar()
    if latest is None:
        return {"status": "no transactions"}
    as_of = pd.Timestamp(as_of or pd.Timestamp(latest) + pd.Timedelta(days=1)).normalize()
    # 120 days of history: the 30-day window plus enough to know which senders and payees are new
    query = db.query(*[getattr(Transaction, c) for c in cols]).filter(
        Transaction.transaction_date >= (as_of - pd.Timedelta(days=120)).to_pydatetime(),
        Transaction.transaction_date < as_of.to_pydatetime())
    df = pd.DataFrame([tuple(r) for r in query.all()], columns=cols)
    if df.empty:
        return {"status": "no recent transactions", "as_of": str(as_of.date())}
    snaps = snapshots(df, dates=[as_of])
    if snaps.empty:
        return {"status": "no active accounts", "as_of": str(as_of.date())}
    X = snaps[art["feature_names"]].astype(float)
    p = art["calibrator"].predict(art["estimator"].predict_proba(X)[:, 1])
    k = max(1, int(round(len(snaps) * get_settings().mule_alert_budget)))
    top = np.argsort(-p)[:k]
    contrib = art["estimator"].booster_.predict(X.iloc[top], pred_contrib=True)[:, :-1]
    new = []
    for rank, (i, c) in enumerate(zip(top, contrib), start=1):
        cid = str(snaps.loc[i, "customer_id"])
        if db.query(MuleAlert).filter(MuleAlert.customer_id == cid, MuleAlert.status.in_(("open", "confirmed_mule"))).first():
            continue
        reasons = [{"feature": f, "value": None if pd.isna(X.iloc[i][f]) else round(float(X.iloc[i][f]), 3),
                    "contribution": round(float(v), 3), "description": MULE_FEATURE_BY_NAME[f].description}
                   for f, v in sorted(zip(art["feature_names"], c), key=lambda kv: -kv[1])[:4] if v > 0]
        row = MuleAlert(customer_id=cid, snapshot_date=str(as_of.date()), model_id=model_id, score=round(float(p[i]), 6),
                        rank=rank, reasons=reasons, status="open", created_at=datetime.utcnow())
        db.add(row)
        db.add(AuditLog(ts=datetime.utcnow(), event_type="MULE_ALERT",
                        payload={"customer_id": cid, "score": round(float(p[i]), 6), "rank": rank, "model_id": model_id}))
        new.append({"customer_id": cid, "score": round(float(p[i]), 4), "rank": rank})
    db.commit()
    if new and notify:
        try:
            from bti.alerts import AlertDispatcher
            AlertDispatcher()._post_webhook({"source": "BTI-MuleScan", "alerts": new})
        except Exception:
            pass
    return {"as_of": str(as_of.date()), "model_id": model_id, "accounts_scored": int(len(snaps)),
            "budget": k, "new_alerts": new}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train and register the mule-account model (mule family)")
    parser.add_argument("--developer", default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    r = train(developer=args.developer, register=not args.no_register)
    print(json.dumps({k: r[k] for k in ("snapshots", "model", "rule_baseline", "fairness", "validation_status")},
                     indent=1, default=str))
    print("importance:", r["feature_importance_gain_share"])
    print("model:", r.get("model_id"))


if __name__ == "__main__":
    main()
