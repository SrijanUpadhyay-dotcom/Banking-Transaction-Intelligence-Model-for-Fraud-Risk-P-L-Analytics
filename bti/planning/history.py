# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Transaction history for planning, from the processed extract or the database.

One row per transaction:
- `date`
- slice keys: `country`, `channel`, `merchant_category`, `merchant_name`,
  `transaction_type`, `customer_segment`
- `fraud` (0/1) and `fraud_type`
- `amount_usd` and `loss_usd`: the realised fraud loss after recovery,
  converted to USD
- `confirmed_at`: when the fraud was confirmed, where known

`confirmed_at` drives label maturity. Recent fraud is under-reported until
disputes and investigations close, so forecasts and early warning must use
the population as it was known on a date, not as it is known today
(`known_as_of`). The completion factors in `bti.planning.forecast` then
scale the known counts up to the expected ultimate counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bti.modeling.fx import RATES_TO_USD

SLICES = ("country", "channel", "merchant_category", "merchant_name", "transaction_type", "customer_segment")


def _usd(values: pd.Series, currency: pd.Series) -> pd.Series:
    rate = currency.astype(str).str.upper().map(RATES_TO_USD)
    return pd.to_numeric(values, errors="coerce").fillna(0.0) * rate


def from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Planning history from a transactions frame in the extract's schema."""
    out = pd.DataFrame({"transaction_id": df["transaction_id"].astype(str),
                        "date": pd.to_datetime(df["transaction_date"], errors="coerce").dt.normalize()})
    for col in SLICES + ("fraud_type",):
        out[col] = df[col].astype("string") if col in df.columns else pd.NA
    out["fraud"] = pd.to_numeric(df.get("fraud_flag", 0), errors="coerce").fillna(0).astype(int)
    currency = df["currency"] if "currency" in df.columns else pd.Series("USD", index=df.index)
    out["amount_usd"] = _usd(df["transaction_amount"], currency)
    loss = df["fraud_loss"] if "fraud_loss" in df.columns else df["transaction_amount"]
    out["loss_usd"] = np.where(out["fraud"] == 1, _usd(loss, currency), 0.0)
    out["confirmed_at"] = (pd.to_datetime(df["label_confirmed_at"], errors="coerce")
                           if "label_confirmed_at" in df.columns else pd.NaT)
    out["hour"] = pd.to_numeric(df.get("transaction_time", pd.Series("12", index=df.index))
                                .astype(str).str.slice(0, 2), errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)


def load(path: Optional[str] = None) -> pd.DataFrame:
    from bti.modeling.train import default_data_path
    return from_frame(pd.read_csv(path or default_data_path(), low_memory=False))


def from_database(db) -> pd.DataFrame:
    """Planning history from the transactions table, with confirmations from the labels table where present."""
    from bti.database.models import FraudLabel, Transaction
    bind = db.get_bind()
    cols = ["transaction_id", "transaction_date", "transaction_time", "transaction_amount", "currency", "fraud_flag",
            "fraud_loss", "fraud_type", *[c for c in SLICES if c in Transaction.__table__.columns]]
    df = pd.read_sql_table(Transaction.__tablename__, bind, columns=cols)
    labels = pd.read_sql_table(FraudLabel.__tablename__, bind,
                               columns=["transaction_id", "label", "created_at", "fraud_type", "loss_amount",
                                        "recovered_amount", "id"])
    if not labels.empty:
        latest = labels.sort_values("id").drop_duplicates("transaction_id", keep="last").set_index("transaction_id")
        ids = df["transaction_id"].astype(str)
        hit = ids.isin(latest.index)
        df.loc[hit, "fraud_flag"] = ids[hit].map(latest["label"]).to_numpy()
        net = (latest["loss_amount"].fillna(0) - latest["recovered_amount"].fillna(0)).clip(lower=0)
        has_loss = ids.map(latest["loss_amount"]).notna() & hit
        df.loc[has_loss, "fraud_loss"] = ids[has_loss].map(net).to_numpy()
        df["label_confirmed_at"] = ids.map(latest["created_at"].where(latest["label"] == 1))
    return from_frame(df)


def known_as_of(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    History as it was known at `as_of`: transactions up to that date, with fraud counted only if confirmed by
    then. Rows without a confirmation time are treated as known immediately.
    """
    h = history[history["date"] <= as_of].copy()
    unknown_yet = (h["fraud"] == 1) & h["confirmed_at"].notna() & (h["confirmed_at"] > as_of)
    h.loc[unknown_yet, ["fraud"]] = 0
    h.loc[unknown_yet, ["loss_usd"]] = 0.0
    return h
