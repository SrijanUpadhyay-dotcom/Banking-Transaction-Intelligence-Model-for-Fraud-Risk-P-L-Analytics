# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Tamper-evident audit storage.

The audit log is protected in three layers:

1. **Hash chain.** Every AuditLog row gets a sequence number and a SHA-256 over
   its content and the previous row's hash, assigned on insert under a lock on
   the chain head. Changing, removing or reordering any row breaks every hash
   after it, and `verify_chain` finds the first break.
2. **Append-only in the database.** Triggers reject UPDATE and DELETE on sealed
   rows, on SQLite and PostgreSQL. Rows written before the chain existed can be
   sealed once, and are then immutable too.
3. **Sealed daily archive.** `archive_day` writes each day's rows to a
   read-only JSONL file plus a manifest holding the file's SHA-256 and the
   chain position. `verify_archive` re-checks both against the database.

Retention: segments older than `audit.retention_days` (7 years by default) are
*reported* as eligible for disposal. Nothing is deleted by code; disposal is a
controlled procedure, subject to Compliance's legal-hold check.

Production: point `audit.archive_dir` at write-once storage, such as S3 Object
Lock in compliance mode, Azure immutable blob storage, or a WORM appliance, so
that even an administrator cannot alter a sealed segment. A local directory
with read-only files is tamper-evident, not tamper-proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import event, select, text, update
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import AuditChainHead, AuditLog

GENESIS = "0" * 64
FIELDS = ("seq", "ts", "event_type", "transaction_id", "analyst_id", "pipeline_run_id", "payload", "source_ip",
          "user_agent", "prev_hash")


def _canonical(record: Dict) -> bytes:
    body = {k: record.get(k) for k in FIELDS}
    ts = body["ts"]
    body["ts"] = ts.isoformat() if isinstance(ts, datetime) else ts
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False).encode()


def row_hash(record: Dict) -> str:
    return hashlib.sha256(_canonical(record)).hexdigest()


def _record(row: AuditLog) -> Dict:
    return {k: getattr(row, k) for k in FIELDS}


@event.listens_for(AuditLog, "before_insert")
def _chain_on_insert(mapper, connection, target: AuditLog) -> None:
    if target.ts is None:
        target.ts = datetime.utcnow()
    head_table = AuditChainHead.__table__
    # Take the write lock first (UPDATE), then read: concurrent writers serialise on the head row.
    locked = connection.execute(update(head_table).where(head_table.c.id == 1)
                                .values(seq=head_table.c.seq + 1)).rowcount
    if not locked:
        connection.execute(head_table.insert().values(id=1, seq=1, head_hash=GENESIS, updated_at=datetime.utcnow()))
    seq, prev = connection.execute(select(head_table.c.seq, head_table.c.head_hash)
                                   .where(head_table.c.id == 1)).one()
    target.seq, target.prev_hash = seq, prev
    target.row_hash = row_hash(_record(target))
    connection.execute(update(head_table).where(head_table.c.id == 1)
                       .values(head_hash=target.row_hash, updated_at=datetime.utcnow()))


# ── Database guards ──────────────────────────────────────────────────────────

_SQLITE_GUARDS = [
    """CREATE TRIGGER IF NOT EXISTS audit_logs_no_update BEFORE UPDATE ON audit_logs
       WHEN OLD.row_hash IS NOT NULL
       BEGIN SELECT RAISE(ABORT, 'audit_logs is append-only: sealed rows cannot be changed'); END""",
    """CREATE TRIGGER IF NOT EXISTS audit_logs_no_delete BEFORE DELETE ON audit_logs
       BEGIN SELECT RAISE(ABORT, 'audit_logs is append-only: rows cannot be deleted'); END""",
]
_POSTGRES_GUARDS = [
    """CREATE OR REPLACE FUNCTION bti_audit_append_only() RETURNS trigger AS $$
       BEGIN
         IF TG_OP = 'DELETE' OR OLD.row_hash IS NOT NULL THEN
           RAISE EXCEPTION 'audit_logs is append-only';
         END IF;
         RETURN NEW;
       END; $$ LANGUAGE plpgsql""",
    "DROP TRIGGER IF EXISTS audit_logs_append_only ON audit_logs",
    """CREATE TRIGGER audit_logs_append_only BEFORE UPDATE OR DELETE ON audit_logs
       FOR EACH ROW EXECUTE FUNCTION bti_audit_append_only()""",
]


def install_guards(bind) -> str:
    dialect = bind.dialect.name
    statements = {"sqlite": _SQLITE_GUARDS, "postgresql": _POSTGRES_GUARDS}.get(dialect)
    if statements is None:
        return f"no guards for dialect {dialect}"
    with bind.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))
    return f"append-only guards installed ({dialect})"


def seal_legacy(db: Session) -> int:
    """Chain rows written before the chain existed, in id order. Each can be sealed once."""
    rows = db.query(AuditLog).filter(AuditLog.row_hash.is_(None)).order_by(AuditLog.id).all()
    if not rows:
        return 0
    head = db.get(AuditChainHead, 1)
    seq, prev = (head.seq, head.head_hash) if head else (0, GENESIS)
    for row in rows:
        seq += 1
        row.seq, row.prev_hash = seq, prev
        row.row_hash = prev = row_hash(_record(row))
    if head is None:
        db.add(AuditChainHead(id=1, seq=seq, head_hash=prev, updated_at=datetime.utcnow()))
    else:
        head.seq, head.head_hash, head.updated_at = seq, prev, datetime.utcnow()
    db.commit()
    return len(rows)


# ── Verification ─────────────────────────────────────────────────────────────

def verify_chain(db: Session, batch: int = 5000) -> Dict:
    head = db.get(AuditChainHead, 1)
    unsealed = db.query(AuditLog).filter(AuditLog.row_hash.is_(None)).count()
    prev, expected_seq, checked = GENESIS, 1, 0
    last_seq = 0
    while True:
        rows = (db.query(AuditLog).filter(AuditLog.seq.isnot(None), AuditLog.seq > last_seq)
                .order_by(AuditLog.seq).limit(batch).all())
        if not rows:
            break
        for row in rows:
            problem = None
            if row.seq != expected_seq:
                problem = f"sequence gap: expected {expected_seq}, found {row.seq} (a row was removed)"
            elif row.prev_hash != prev:
                problem = "previous-hash link broken (a row was removed or reordered)"
            elif row.row_hash != row_hash(_record(row)):
                problem = "content does not match its hash (the row was altered)"
            if problem:
                return {"status": "broken", "rows_checked": checked,
                        "first_break": {"seq": row.seq, "id": row.id, "event_type": row.event_type,
                                        "ts": row.ts.isoformat() if row.ts else None, "problem": problem},
                        "unsealed_rows": unsealed}
            prev, expected_seq, checked, last_seq = row.row_hash, expected_seq + 1, checked + 1, row.seq
    head_ok = head is None and checked == 0 or (head is not None and head.seq == checked and head.head_hash == prev)
    return {"status": "intact" if head_ok else "broken", "rows_checked": checked,
            "head": {"seq": head.seq, "hash": head.head_hash} if head else None,
            "first_break": None if head_ok else {"problem": "chain head does not match the last row "
                                                            "(rows were removed from the end)"},
            "unsealed_rows": unsealed}


# ── Sealed daily archive and retention ───────────────────────────────────────

def archive_dir() -> Path:
    return Path(get_settings().audit_archive_dir)


def _segment_paths(day: date):
    folder = archive_dir() / f"{day:%Y}" / f"{day:%m}"
    return folder / f"audit-{day:%Y-%m-%d}.jsonl", folder / f"audit-{day:%Y-%m-%d}.manifest.json"


def archive_day(db: Session, day: Optional[date] = None) -> Dict:
    """Write one UTC day of sealed audit rows to a read-only segment with a manifest. Never overwrites."""
    day = day or (datetime.utcnow() - timedelta(days=1)).date()
    start = datetime(day.year, day.month, day.day)
    rows = (db.query(AuditLog).filter(AuditLog.ts >= start, AuditLog.ts < start + timedelta(days=1),
                                      AuditLog.row_hash.isnot(None)).order_by(AuditLog.seq).all())
    data_path, manifest_path = _segment_paths(day)
    if manifest_path.exists():
        return {"day": day.isoformat(), "status": "already_archived", **json.loads(manifest_path.read_text())}
    if not rows:
        return {"day": day.isoformat(), "status": "no_rows"}
    data_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({**{k: v for k, v in _record(r).items()}, "ts": r.ts.isoformat(), "row_hash": r.row_hash},
                        sort_keys=True, default=str, ensure_ascii=False) for r in rows]
    content = ("\n".join(lines) + "\n").encode()
    data_path.write_bytes(content)
    manifest = {"day": day.isoformat(), "rows": len(rows), "first_seq": rows[0].seq, "last_seq": rows[-1].seq,
                "last_row_hash": rows[-1].row_hash, "file": data_path.name,
                "file_sha256": hashlib.sha256(content).hexdigest(),
                "sealed_at": datetime.utcnow().isoformat(),
                "retain_until": (day + timedelta(days=get_settings().audit_retention_days)).isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for path in (data_path, manifest_path):
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return {"status": "archived", **manifest}


def verify_archive(db: Session, day: date) -> Dict:
    data_path, manifest_path = _segment_paths(day)
    if not manifest_path.exists():
        return {"day": day.isoformat(), "status": "not_archived"}
    manifest = json.loads(manifest_path.read_text())
    content = data_path.read_bytes() if data_path.exists() else b""
    problems = []
    if hashlib.sha256(content).hexdigest() != manifest["file_sha256"]:
        problems.append("segment file does not match its manifest hash")
    lines = [json.loads(line) for line in content.decode().splitlines() if line]
    if len(lines) != manifest["rows"]:
        problems.append(f"segment holds {len(lines)} rows, manifest says {manifest['rows']}")
    by_seq = {r.seq: r.row_hash for r in db.query(AuditLog.seq, AuditLog.row_hash).filter(
        AuditLog.seq >= manifest["first_seq"], AuditLog.seq <= manifest["last_seq"]).all()}
    mismatched = [line["seq"] for line in lines if by_seq.get(line["seq"]) != line["row_hash"]]
    if mismatched:
        problems.append(f"{len(mismatched)} rows differ from the database (first seq {mismatched[0]})")
    return {"day": day.isoformat(), "status": "intact" if not problems else "broken", "problems": problems,
            "manifest": manifest}


def retention_report(today: Optional[date] = None) -> Dict:
    today = today or datetime.utcnow().date()
    segments: List[Dict] = []
    for manifest_path in sorted(archive_dir().glob("*/*/*.manifest.json")):
        m = json.loads(manifest_path.read_text())
        segments.append({"day": m["day"], "rows": m["rows"], "retain_until": m["retain_until"],
                         "eligible_for_disposal": date.fromisoformat(m["retain_until"]) < today})
    return {"retention_days": get_settings().audit_retention_days, "segments": len(segments),
            "eligible_for_disposal": [s["day"] for s in segments if s["eligible_for_disposal"]],
            "policy": "Eligible segments are disposed of only through the controlled procedure, after "
                      "Compliance confirms no legal hold applies. No code path deletes audit records.",
            "oldest": segments[0]["day"] if segments else None, "newest": segments[-1]["day"] if segments else None}
