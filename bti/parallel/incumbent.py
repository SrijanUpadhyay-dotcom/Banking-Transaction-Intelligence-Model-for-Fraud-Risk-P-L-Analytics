"""
Incumbent (e.g. SAS) score and decision ingestion.

Every platform names its actions differently, so decisions are normalised to
BTI's four actions: APPROVE / STEP_UP / REVIEW / DECLINE. The default map covers
common vendor vocabularies; a bank adds its own codes under
`parallel_run.decision_map` in settings.yaml. Unknown codes are rejected row by
row, never guessed. Records are append-only; the latest per transaction wins.

Usage (batch file from the incumbent's decision log):
  python -m bti.parallel.incumbent --file sas_decisions.csv --system SAS \\
      --column-map '{"TXN_ID": "transaction_id", "SAS_SCORE": "score", "ACTION_CD": "decision"}'
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import IncumbentDecision, ScoreLog

ACTIONS = ("APPROVE", "STEP_UP", "REVIEW", "DECLINE")
DEFAULT_DECISION_MAP: Dict[str, str] = {
    **{c: "APPROVE" for c in ("APPROVE", "APPROVED", "ACCEPT", "ALLOW", "PASS", "AUTHORISE", "AUTHORIZE", "OK")},
    **{c: "STEP_UP" for c in ("STEP_UP", "STEPUP", "CHALLENGE", "AUTHENTICATE", "OTP", "3DS", "SCA")},
    **{c: "REVIEW" for c in ("REVIEW", "REFER", "REFERRAL", "QUEUE", "HOLD", "ALERT", "CASE", "PEND")},
    **{c: "DECLINE" for c in ("DECLINE", "DECLINED", "DENY", "REJECT", "BLOCK", "STOP")},
}
FIELDS = ("transaction_id", "customer_id", "score", "decision", "executed_decision", "decided_at", "latency_ms",
          "amount", "currency", "rule_ids")
MAX_REJECTS_REPORTED = 50


def decision_map() -> Dict[str, str]:
    custom = {str(k).strip().upper(): str(v).strip().upper() for k, v in get_settings().incumbent_decision_map.items()}
    bad = {k: v for k, v in custom.items() if v not in ACTIONS}
    if bad:
        raise ValueError(f"parallel_run.decision_map targets must be one of {ACTIONS}: {bad}")
    return {**DEFAULT_DECISION_MAP, **custom}


def normalise_decision(code, mapping: Optional[Dict[str, str]] = None) -> Optional[str]:
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    return (mapping or decision_map()).get(str(code).strip().upper())


def _timestamp(value, default: datetime) -> datetime:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return default
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def _number(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(v) else v


def ingest(db: Session, rows: Iterable[dict], system: Optional[str] = None,
           batch_id: Optional[str] = None) -> Dict:
    """Validate and store incumbent decisions. Valid rows are kept; each rejected row is reported with a reason."""
    system = (system or get_settings().incumbent_system).strip()
    batch_id = batch_id or uuid.uuid4().hex[:12]
    mapping, now = decision_map(), datetime.utcnow()
    records: List[IncumbentDecision] = []
    rejected: List[Dict] = []
    seen: Dict[str, int] = {}
    for i, row in enumerate(rows):
        tid = row.get("transaction_id")
        if tid is None or str(tid).strip() == "" or (isinstance(tid, float) and pd.isna(tid)):
            rejected.append({"row": i, "reason": "missing transaction_id"})
            continue
        decision = normalise_decision(row.get("decision"), mapping)
        if decision is None:
            rejected.append({"row": i, "transaction_id": str(tid),
                             "reason": f"unknown decision code {row.get('decision')!r}; add it to "
                                       f"parallel_run.decision_map"})
            continue
        executed = row.get("executed_decision")
        executed_norm = normalise_decision(executed, mapping) if executed not in (None, "") else None
        if executed not in (None, "") and not (isinstance(executed, float) and pd.isna(executed)) \
                and executed_norm is None:
            rejected.append({"row": i, "transaction_id": str(tid), "reason": f"unknown executed_decision {executed!r}"})
            continue
        try:
            decided_at = _timestamp(row.get("decided_at"), now)
        except (ValueError, TypeError):
            rejected.append({"row": i, "transaction_id": str(tid), "reason": f"bad decided_at {row.get('decided_at')!r}"})
            continue
        rules = row.get("rule_ids")
        if isinstance(rules, str):
            rules = [r.strip() for r in rules.replace(";", ",").split(",") if r.strip()]
        seen[str(tid)] = seen.get(str(tid), 0) + 1
        records.append(IncumbentDecision(
            transaction_id=str(tid).strip(), system=system, customer_id=(str(row["customer_id"])
                                                                       if row.get("customer_id") else None),
            score=_number(row.get("score")), decision=decision, raw_decision=str(row.get("decision")),
            executed_decision=executed_norm, decided_at=decided_at, latency_ms=_number(row.get("latency_ms")),
            amount=_number(row.get("amount")), currency=(str(row["currency"]).upper() if row.get("currency") else None),
            rule_ids=rules if isinstance(rules, list) else None, batch_id=batch_id, received_at=now,
        ))
    db.add_all(records)
    db.commit()
    return {
        "system": system, "batch_id": batch_id, "accepted": len(records), "rejected": len(rejected),
        "rejections": rejected[:MAX_REJECTS_REPORTED],
        "repeated_in_batch": sum(1 for c in seen.values() if c > 1),
        "decisions": pd.Series([r.decision for r in records], dtype=str).value_counts().to_dict(),
    }


def read_file(path, column_map: Optional[Dict[str, str]] = None) -> List[dict]:
    """Read a CSV / Parquet export and rename its columns to BTI's field names."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path, dtype=str)
    return frame_to_rows(df, column_map)


def frame_to_rows(df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None) -> List[dict]:
    df = df.rename(columns=column_map or {})
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "transaction_id" not in df.columns or "decision" not in df.columns:
        raise ValueError(f"The file needs transaction_id and decision columns (after column_map); found "
                         f"{list(df.columns)}")
    keep = [c for c in FIELDS if c in df.columns]
    return df[keep].astype(object).where(df[keep].notna(), None).to_dict("records")


CHUNK = 5000


def latest_incumbent(db: Session, transaction_ids: Optional[List[str]] = None,
                     start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
    if transaction_ids is not None and len(transaction_ids) > CHUNK:
        parts = [latest_incumbent(db, transaction_ids[i:i + CHUNK], start, end)
                 for i in range(0, len(transaction_ids), CHUNK)]
        return pd.concat(parts, ignore_index=True)
    cols = [IncumbentDecision.id, IncumbentDecision.transaction_id, IncumbentDecision.system,
            IncumbentDecision.score, IncumbentDecision.decision, IncumbentDecision.executed_decision,
            IncumbentDecision.decided_at, IncumbentDecision.latency_ms, IncumbentDecision.received_at]
    q = db.query(*cols)
    if transaction_ids is not None:
        q = q.filter(IncumbentDecision.transaction_id.in_(transaction_ids))
    if start:
        q = q.filter(IncumbentDecision.decided_at >= start)
    if end:
        q = q.filter(IncumbentDecision.decided_at < end)
    df = pd.DataFrame(q.all(), columns=[c.key for c in cols])
    if df.empty:
        return df.drop(columns="id")
    return (df.sort_values(["received_at", "id"], kind="stable")
              .drop_duplicates("transaction_id", keep="last").drop(columns="id").reset_index(drop=True))


def ingestion_status(db: Session, start: Optional[datetime] = None, end: Optional[datetime] = None) -> Dict:
    """How well the two decision streams line up: the parallel run is only as good as the join."""
    inc = latest_incumbent(db, start=start, end=end)
    q = db.query(func.distinct(ScoreLog.transaction_id)).filter(ScoreLog.is_shadow.is_(False))
    if start:
        q = q.filter(ScoreLog.scored_at >= start)
    if end:
        q = q.filter(ScoreLog.scored_at < end)
    bti_ids = {r[0] for r in q.all()}
    inc_ids = set(inc["transaction_id"]) if not inc.empty else set()
    both = bti_ids & inc_ids
    return {
        "window": {"from": start, "to": end},
        "incumbent_transactions": len(inc_ids),
        "bti_transactions": len(bti_ids),
        "matched": len(both),
        "bti_coverage_of_incumbent": round(len(both) / len(inc_ids), 4) if inc_ids else None,
        "incumbent_coverage_of_bti": round(len(both) / len(bti_ids), 4) if bti_ids else None,
        "incumbent_decisions": inc["decision"].value_counts().to_dict() if not inc.empty else {},
        "systems": sorted(inc["system"].unique().tolist()) if not inc.empty else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an incumbent decision export into BTI")
    parser.add_argument("--file", required=True)
    parser.add_argument("--system", default=None)
    parser.add_argument("--column-map", default="{}", help="JSON object: source column → BTI field")
    args = parser.parse_args()
    from bti.database.connection import SessionLocal
    from bti.database.init_db import create_tables
    create_tables()
    db = SessionLocal()
    try:
        result = ingest(db, read_file(args.file, json.loads(args.column_map)), system=args.system)
    finally:
        db.close()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
