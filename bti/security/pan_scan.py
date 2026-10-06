# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Card-number discovery: proof that no primary account number is stored at rest.

**Why it matters.** PCI DSS v4.0 requirement 12.5.2 asks the entity to confirm
its scope at least once a year: where account data actually lives. BTI's
design keeps PANs out entirely:
- the ISO 8583 adapter tokenises them on arrival (`bti.streaming.tokenize`)
- the logs are redacted (`bti.security.hardening`)

This scan checks that the design holds in practice. It looks through:
- every text column of every database table
- the generated outputs and reports
- the application logs
- the model registry cards

**Match rule.** 13–19 digits, optionally separated by spaces or hyphens,
that pass the Luhn check and start with an issued-card prefix: Visa 4,
Mastercard 51–55 / 2221–2720, Amex 34/37, Discover/RuPay 60/65, UnionPay 62,
JCB 35, Diners 30/36/38. The 999999 test range used by the simulator is
excluded, and so are digit runs inside decimal numbers
(model cards and P&L outputs are full of them) or joined to letters (hex hashes in the audit chain). Matches are reported masked (first 6, last 4) with their location,
never in full.

    python -m bti.security.pan_scan
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

CANDIDATE = re.compile(r"(?<![\w.,])(?:\d[ -]?){12,18}\d(?!\w|[.,]\d)")    # not part of a decimal number
ISSUED = re.compile(r"^(4|5[1-5]|2(2[2-9]|[3-6]\d|7[01]|720)|3[47]|60|65|62|35|30|36|38)")
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".log", ".md", ".txt", ".html"}


def luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


def find_pans(text: str) -> List[str]:
    """Masked PAN-like values found in `text`."""
    out = []
    for m in CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and ISSUED.match(digits) and not digits.startswith("999999") and luhn(digits):
            out.append(f"{digits[:6]}{'*' * (len(digits) - 10)}{digits[-4:]}")
    return out


def scan_files(roots: Iterable[Path], max_bytes: int = 50 * 1024 * 1024) -> List[Dict]:
    hits = []
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for f in files:
            if f.suffix.lower() not in TEXT_SUFFIXES or f.stat().st_size > max_bytes:
                continue
            found = find_pans(f.read_text(errors="ignore"))
            if found:
                hits.append({"location": str(f), "matches": len(found), "examples": found[:3]})
    return hits


def scan_database(bind, limit_rows: int = 200000) -> List[Dict]:
    from sqlalchemy import inspect, text
    hits = []
    insp = inspect(bind)
    with bind.connect() as conn:
        for table in insp.get_table_names():
            cols = [c["name"] for c in insp.get_columns(table)
                    if any(t in str(c["type"]).upper() for t in ("CHAR", "TEXT", "JSON", "CLOB"))]
            if not cols:
                continue
            quote = bind.dialect.identifier_preparer.quote
            # identifiers come from the database's own schema and are quoted; the limit is an int
            q = text(f'SELECT {", ".join(quote(c) for c in cols)} FROM {quote(table)} '  # nosec B608
                     f'LIMIT {int(limit_rows)}')
            count, examples = 0, []
            for row in conn.execute(q):
                for v in row:
                    if v is None:
                        continue
                    found = find_pans(v if isinstance(v, str) else json.dumps(v, default=str))
                    count += len(found)
                    examples.extend(found[: 3 - len(examples)] if len(examples) < 3 else [])
            if count:
                hits.append({"location": f"db:{table}", "matches": count, "examples": examples})
    return hits


def run(bind=None, roots: Optional[List[Path]] = None) -> Dict:
    if bind is None:
        from bti.database.connection import engine as bind
    roots = roots or [Path("outputs"), Path("logs"), Path("models")]
    db_hits = scan_database(bind)
    file_hits = scan_files(roots)
    return {"database_locations_with_pans": db_hits, "file_locations_with_pans": file_hits,
            "clean": not db_hits and not file_hits,
            "scope": {"database": str(bind.url).split("@")[-1], "paths": [str(r) for r in roots]}}


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
