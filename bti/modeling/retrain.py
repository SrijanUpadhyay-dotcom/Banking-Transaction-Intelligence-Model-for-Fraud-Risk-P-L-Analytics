"""
Continuous retraining: scheduled or triggered, never self-promoting.

A daily check decides whether to retrain. It may, at most once every
`min_interval_days` (7), when any of these holds:
- **schedule:** `interval_days` (30) have passed since the last retraining.
- **drift:** a drift check escalated (PSI ≥ 0.25) since then.
- **performance:** outcomes analysis raised a performance finding since then.
- **new labels:** at least `min_new_labels` (500) confirmed labels have
  arrived since then.

**The training extract.** The bank's latest transaction extract, with the
feedback loop applied:
- Confirmed labels from case dispositions, chargebacks and investigations
  override the extract's fraud flag, and the latest label wins.
- Transactions younger than the label-maturity window are dropped unless they
  carry an explicit label. Otherwise frauds not yet reported would be learned
  as genuine.

**What happens to the result.** The tournament trains candidates on that
extract and judges the incumbent on the same split.
- A winner takes the challenger role only, which runs in shadow beside the
  champion.
- Promotion to champion stays behind the human, four-eyes, independently
  validated gate.
- Per-event online updates are deliberately not done: each would be a model
  change a validator must review.

Usage:
  python -m bti.modeling.retrain --check       # show triggers
  python -m bti.modeling.retrain --force       # retrain now
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import AuditLog, FraudLabel, ValidationFinding
from bti.logging_config import get_logger
from bti.modeling import registry

log = get_logger("modeling.retrain")

EVENT_TYPE = "RETRAIN_RUN"


def _last_run(db: Session) -> Optional[datetime]:
    row = (db.query(AuditLog.ts).filter(AuditLog.event_type == EVENT_TYPE)
           .order_by(AuditLog.ts.desc(), AuditLog.id.desc()).first())
    return row[0] if row else None


def retraining_triggers(db: Session, now: Optional[datetime] = None) -> Dict:
    s = get_settings()
    now = now or datetime.utcnow()
    last = _last_run(db)
    since = last or datetime(1970, 1, 1)
    drift = [r.payload for r in db.query(AuditLog).filter(AuditLog.event_type == "DRIFT_CHECK",
                                                          AuditLog.ts > since).all()
             if (r.payload or {}).get("status") == "escalate"]
    performance = (db.query(ValidationFinding).filter(ValidationFinding.category == "performance",
                                                      ValidationFinding.source == "monitoring",
                                                      ValidationFinding.raised_at > since).count())
    new_labels = db.query(FraudLabel).filter(FraudLabel.created_at > since).count()
    reasons = []
    if last is None or now - last >= timedelta(days=s.retrain_interval_days):
        reasons.append("schedule" if last else "no previous retraining")
    if drift:
        reasons.append(f"drift escalated ({len(drift)} check(s))")
    if performance:
        reasons.append(f"{performance} performance finding(s) from outcomes analysis")
    if new_labels >= s.retrain_min_new_labels:
        reasons.append(f"{new_labels} new confirmed labels")
    too_soon = last is not None and now - last < timedelta(days=s.retrain_min_interval_days)
    return {"checked_at": now.isoformat(), "last_retrain": last.isoformat() if last else None,
            "reasons": reasons, "new_labels_since_last": new_labels, "too_soon": too_soon,
            "should_run": bool(reasons) and not too_soon,
            "policy": {"interval_days": s.retrain_interval_days, "min_interval_days": s.retrain_min_interval_days,
                       "min_new_labels": s.retrain_min_new_labels}}


def build_training_extract(db: Session, source_path: Optional[Path] = None, maturity_days: Optional[int] = None,
                           out_dir: Optional[Path] = None) -> Dict:
    from bti.modeling.features import event_timestamps
    from bti.modeling.train import LABEL, default_data_path
    from bti.operations.feedback import latest_labels

    s = get_settings()
    source_path = Path(source_path or default_data_path())
    maturity_days = maturity_days if maturity_days is not None else s.retrain_label_maturity_days
    df = pd.read_csv(source_path, low_memory=False)
    labels = latest_labels(db, df["transaction_id"].astype(str).tolist())
    overrides, flipped = 0, 0
    labelled_ids = set()
    if not labels.empty:
        lab = labels.set_index("transaction_id")["label"].astype(int)
        ids = df["transaction_id"].astype(str)
        hit = ids.isin(lab.index)
        new = ids[hit].map(lab).to_numpy()
        old = pd.to_numeric(df.loc[hit, LABEL], errors="coerce").fillna(0).astype(int).to_numpy()
        overrides, flipped = int(hit.sum()), int((new != old).sum())
        df.loc[hit, LABEL] = new
        labelled_ids = set(ids[hit])
    ts = event_timestamps(df)
    cutoff = ts.max() - pd.Timedelta(days=maturity_days)
    immature = (ts > cutoff) & ~df["transaction_id"].astype(str).isin(labelled_ids)
    extract = df[~immature]
    out_dir = Path(out_dir or Path(s.processed_data_dir) / "retraining")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"extract-{datetime.utcnow():%Y%m%d%H%M%S}.csv"
    extract.to_csv(path, index=False)
    return {"path": str(path), "source": str(source_path), "rows": int(len(extract)),
            "dropped_immature": int(immature.sum()), "maturity_cutoff": str(cutoff),
            "labels_applied": overrides, "labels_changed_flag": flipped,
            "fraud_rate": round(float(pd.to_numeric(extract[LABEL], errors="coerce").fillna(0).mean()), 5)}


def run_retraining(db: Session, force: bool = False, notify: bool = True, source_path: Optional[Path] = None,
                   now: Optional[datetime] = None, algorithms=None, feature_sets=None,
                   extract_dir: Optional[Path] = None) -> Dict:
    from bti.modeling.tournament import run_tournament

    s = get_settings()
    now = now or datetime.utcnow()
    triggers = retraining_triggers(db, now)
    if not force and not triggers["should_run"]:
        return {"status": "not_triggered", "triggers": triggers}
    incumbent = registry.model_for_role("challenger") or registry.model_for_role("champion")
    incumbent_set = registry.load_card(incumbent)["features"].get("feature_set") if incumbent else None
    feature_sets = feature_sets or sorted({*(s.pipeline_feature_sets or []), *([incumbent_set] if incumbent_set else [])})
    extract = build_training_extract(db, source_path, out_dir=extract_dir)
    developer = s.model_developer or "bti.modeling.retrain"
    report = run_tournament(developer, algorithms or s.pipeline_algorithms, feature_sets, tune=False,
                            data_path=extract["path"])
    result = {"status": "retrained", "run_at": now.isoformat(), "forced": force, "triggers": triggers,
              "extract": extract, "feature_sets": feature_sets, "decision": report["decision"],
              "tournament_report": report["report_path"],
              "note": "A new challenger runs in shadow only; champion promotion stays behind validation "
                      "sign-off and four-eyes approval."}
    db.add(AuditLog(ts=now, event_type=EVENT_TYPE, payload=json.loads(json.dumps(result, default=str))))
    db.commit()
    if notify:
        from bti.alerts import AlertDispatcher
        result["alert"] = AlertDispatcher().dispatch_event(
            "MODEL_RETRAINED", "INFO", {"decision": report["decision"], "reasons": triggers["reasons"]})
    log.info("Retraining complete", extra={"outcome": report["decision"]["outcome"]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous retraining (winners become shadow challengers only)")
    parser.add_argument("--check", action="store_true", help="Show triggers without retraining")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    from bti.database.connection import SessionLocal
    from bti.database.init_db import create_tables
    create_tables()
    db = SessionLocal()
    try:
        out = retraining_triggers(db) if args.check else run_retraining(db, force=args.force)
    finally:
        db.close()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
