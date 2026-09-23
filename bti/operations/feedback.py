"""
Confirmed-outcome feedback loop.

Fraud labels arrive late: chargebacks 30–120 days after the transaction,
investigator outcomes days to weeks later. Labels are stored append-only; the
latest label per transaction wins. For performance measurement a scored
transaction with no fraud report after the maturity window is treated as
genuine — the standard convention — and younger unlabelled transactions are
excluded so recent weeks do not look artificially clean.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from bti.database.models import FraudLabel, ScoreLog

LABEL_SOURCES: Dict[str, int] = {
    "CHARGEBACK": 1,
    "CUSTOMER_REPORTED_FRAUD": 1,
    "INVESTIGATOR_CONFIRMED": 1,
    "LAW_ENFORCEMENT": 1,
    "INVESTIGATOR_CLEARED": 0,
    "CUSTOMER_CONFIRMED_GENUINE": 0,
}
DEFAULT_MATURITY_DAYS = 90


class LabelError(ValueError):
    pass


def record_labels(db: Session, labels: Iterable[dict]) -> Dict:
    rows: List[FraudLabel] = []
    for i, item in enumerate(labels):
        source = str(item.get("label_source", "")).upper()
        if source not in LABEL_SOURCES:
            raise LabelError(f"Row {i}: label_source must be one of {sorted(LABEL_SOURCES)}")
        label = int(item.get("label", LABEL_SOURCES[source]))
        if label != LABEL_SOURCES[source]:
            raise LabelError(f"Row {i}: label {label} contradicts source {source}")
        if not item.get("transaction_id"):
            raise LabelError(f"Row {i}: transaction_id is required")
        event_at = item.get("event_at") or datetime.utcnow()
        if isinstance(event_at, str):
            event_at = pd.Timestamp(event_at).to_pydatetime()
        rows.append(FraudLabel(
            transaction_id=str(item["transaction_id"]), label=label, label_source=source,
            fraud_type=item.get("fraud_type"), event_at=event_at, loss_amount=item.get("loss_amount"),
            recovered_amount=item.get("recovered_amount"), currency=item.get("currency"),
            reported_by=item.get("reported_by"), notes=item.get("notes"),
        ))
    db.add_all(rows)
    db.commit()
    return {"recorded": len(rows), "fraud": sum(r.label for r in rows),
            "genuine": sum(1 - r.label for r in rows)}


def latest_labels(db: Session, transaction_ids: Optional[List[str]] = None) -> pd.DataFrame:
    q = db.query(FraudLabel.id, FraudLabel.transaction_id, FraudLabel.label, FraudLabel.label_source,
                 FraudLabel.event_at)
    if transaction_ids is not None:
        q = q.filter(FraudLabel.transaction_id.in_(transaction_ids))
    df = pd.DataFrame(q.all(), columns=["id", "transaction_id", "label", "label_source", "event_at"])
    if df.empty:
        return df
    return (df.sort_values(["event_at", "id"]).groupby("transaction_id", as_index=False).last()
              .drop(columns="id"))


def labelled_scores(db: Session, start: Optional[datetime] = None, end: Optional[datetime] = None,
                    maturity_days: int = DEFAULT_MATURITY_DAYS, shadow: Optional[bool] = False,
                    now: Optional[datetime] = None) -> pd.DataFrame:
    """Scores joined with outcomes. label is NaN where the outcome is not yet knowable."""
    cols = [ScoreLog.transaction_id, ScoreLog.customer_id, ScoreLog.model_id, ScoreLog.model_role,
            ScoreLog.is_shadow, ScoreLog.fraud_probability, ScoreLog.decision, ScoreLog.amount_usd,
            ScoreLog.jurisdiction, ScoreLog.scored_at]
    q = db.query(*cols)
    if start:
        q = q.filter(ScoreLog.scored_at >= start)
    if end:
        q = q.filter(ScoreLog.scored_at < end)
    if shadow is not None:
        q = q.filter(ScoreLog.is_shadow == shadow)
    scores = pd.DataFrame(q.all(), columns=[c.key for c in cols])
    if scores.empty:
        return scores.assign(label=pd.Series(dtype=float), label_source=pd.Series(dtype=str))
    labels = latest_labels(db, scores["transaction_id"].unique().tolist())
    out = scores.merge(labels[["transaction_id", "label", "label_source"]] if not labels.empty
                       else pd.DataFrame(columns=["transaction_id", "label", "label_source"]),
                       on="transaction_id", how="left")
    cutoff = (now or datetime.utcnow()) - timedelta(days=maturity_days)
    mature_unlabelled = out["label"].isna() & (pd.to_datetime(out["scored_at"]) < cutoff)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out.loc[mature_unlabelled, "label"] = 0.0
    out.loc[mature_unlabelled, "label_source"] = "MATURED_NO_REPORT"
    return out


def label_status(db: Session, maturity_days: int = DEFAULT_MATURITY_DAYS) -> Dict:
    by_source = dict(db.query(FraudLabel.label_source, func.count(FraudLabel.id))
                     .group_by(FraudLabel.label_source).all())
    scored = db.query(func.count(func.distinct(ScoreLog.transaction_id))).scalar() or 0
    cutoff = datetime.utcnow() - timedelta(days=maturity_days)
    mature = (db.query(func.count(func.distinct(ScoreLog.transaction_id)))
              .filter(ScoreLog.scored_at < cutoff).scalar() or 0)
    labelled = db.query(func.count(func.distinct(FraudLabel.transaction_id))).scalar() or 0
    return {
        "labels_by_source": by_source,
        "transactions_scored": int(scored),
        "transactions_with_explicit_label": int(labelled),
        "transactions_past_maturity": int(mature),
        "maturity_days": maturity_days,
        "measurable_share": round(mature / scored, 4) if scored else 0.0,
    }
