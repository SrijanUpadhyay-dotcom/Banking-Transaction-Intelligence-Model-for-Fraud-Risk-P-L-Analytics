"""
Rescore stored transaction history with the scoring v3 model.

The transactions table was seeded with the legacy composite score, which read
label-derived inputs. The analyst exception queue, the transaction risk filters
and the customer-risk analytics all read `final_risk_score` /
`final_alert_tier` from that table, so they are rewritten here with the v3
calibrated probability (× 100) and tier — the same meaning those fields have on
/score since Phase 0. The legacy ML columns (`lr_fraud_proba`,
`rf_fraud_proba`) are cleared. The original legacy values remain in
data/processed/banking_transactions_ml_scored.csv for audit.

Usage:
  python -m bti.modeling.rescore
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import text

from bti.database.connection import SessionLocal
from bti.database.models import AuditLog
from bti.logging_config import get_logger
from bti.modeling.scorer import scorer
from bti.scoring.v3_adapter import tier_for

log = get_logger("modeling.rescore")


def rescore_history(csv_path: Optional[Path] = None, role: str = "champion", batch: int = 5000) -> Dict:
    from bti.modeling.train import default_data_path
    csv_path = Path(csv_path or default_data_path())
    df = pd.read_csv(csv_path, low_memory=False)
    scored = scorer.score_frame(df, role=role)
    model_id = scored["model_id"].iloc[0]
    probability = scored["fraud_probability"].to_numpy()
    rows = [{"tid": str(t), "score": round(float(p) * 100, 2), "tier": tier_for(float(p))}
            for t, p in zip(df["transaction_id"], probability)]

    db = SessionLocal()
    try:
        updated = 0
        for i in range(0, len(rows), batch):
            result = db.execute(text(
                "UPDATE transactions SET final_risk_score = :score, final_alert_tier = :tier, "
                "lr_fraud_proba = NULL, rf_fraud_proba = NULL WHERE transaction_id = :tid"), rows[i:i + batch])
            updated += result.rowcount or 0
        tiers = pd.Series([r["tier"] for r in rows]).value_counts().to_dict()
        summary = {"model_id": model_id, "model_role": scored["model_role"].iloc[0], "source": str(csv_path),
                   "scored": len(rows), "updated_in_db": int(updated), "tiers": tiers}
        db.add(AuditLog(ts=datetime.utcnow(), event_type="HISTORY_RESCORED", payload=summary))
        db.commit()
    finally:
        db.close()
    log.info("History rescored", extra=summary)
    return summary


def main() -> None:
    s = rescore_history()
    print(f"Rescored {s['scored']:,} transactions with {s['model_id']} ({s['model_role']}); "
          f"{s['updated_in_db']:,} rows updated. Tiers: {s['tiers']}")


if __name__ == "__main__":
    main()
