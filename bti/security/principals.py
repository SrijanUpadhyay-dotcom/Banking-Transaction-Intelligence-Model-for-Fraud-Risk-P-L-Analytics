# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Principals: who is calling the API.

Every person and every system that calls BTI is a *principal*, with:
- its own key, stored only as a SHA-256 hash
- one or more roles (`bti.security.access`)
- an active flag; deactivating revokes access at once

Keys are 32 random bytes, shown once at creation. They are never stored in
clear or logged.

The registry lives in `config/principals.json` (`BTI_PRINCIPALS_FILE`),
outside version control. In production it is generated from the bank's
identity provider (joiners, movers, leavers). Access reviews read it
through `python -m bti.security.principals list`.

**The `system` role** marks a service authenticating on behalf of humans
who were authenticated upstream, for example an API gateway in front of SSO.
Such a principal may name the acting person in a request. Every other
principal must *be* the person it names: approver, author, reviewer,
validator.

**Legacy single key.** The old shared `BTI_API_KEY` maps to a
`legacy-service` principal with `admin` and `system` roles, only when
`BTI_ALLOW_LEGACY_API_KEY=true`. It is off by default, and the readiness
probe and the security self-assessment both flag it. A shared admin key
cannot support four-eyes.

    python -m bti.security.principals add --id jane.doe --name "Jane Doe" --roles analyst
    python -m bti.security.principals list
    python -m bti.security.principals deactivate --id jane.doe
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bti.config import get_settings

ROLES = ("scoring", "analyst", "operations", "model_risk", "auditor", "admin", "system")
LEGACY_ID = "legacy-service"
DEV_ID = "dev-local"
_lock = threading.Lock()
_cache: Dict[str, object] = {"mtime": None, "principals": {}}


@dataclass
class Principal:
    id: str
    name: str
    roles: List[str]
    key_sha256: str = ""
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def has(self, *roles: str) -> bool:
        return bool(set(self.roles) & set(roles)) or "admin" in self.roles


def principals_file() -> Path:
    return Path(os.environ.get("BTI_PRINCIPALS_FILE") or get_settings().principals_file)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load() -> Dict[str, Principal]:
    path = principals_file()
    mtime = path.stat().st_mtime if path.exists() else None
    with _lock:
        if _cache["mtime"] != mtime or _cache.get("path") != str(path):
            items = json.loads(path.read_text()) if path.exists() else []
            _cache.update(mtime=mtime, path=str(path),
                          principals={p["id"]: Principal(**p) for p in items})
        return dict(_cache["principals"])


def _save(principals: Dict[str, Principal]) -> None:
    path = principals_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(p) for p in principals.values()], indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    with _lock:
        _cache["mtime"] = None


def add(principal_id: str, name: str, roles: List[str]) -> Tuple[Principal, str]:
    unknown = set(roles) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles {sorted(unknown)}; allowed {ROLES}")
    if not principal_id or principal_id in (LEGACY_ID, DEV_ID):
        raise ValueError("invalid principal id")
    principals = load()
    if principal_id in principals:
        raise ValueError(f"principal {principal_id} exists; deactivate it and create a new id to rotate")
    key = secrets.token_urlsafe(32)
    p = Principal(principal_id, name, sorted(set(roles)), _hash(key))
    principals[principal_id] = p
    _save(principals)
    return p, key


def deactivate(principal_id: str) -> Principal:
    principals = load()
    if principal_id not in principals:
        raise ValueError(f"no principal {principal_id}")
    principals[principal_id].active = False
    _save(principals)
    return principals[principal_id]


def authenticate(key: Optional[str]) -> Optional[Principal]:
    """The active principal holding `key`, or None. Constant-time comparison against every stored hash."""
    s = get_settings()
    if not key:
        return None
    digest = _hash(key)
    found = None
    for p in load().values():
        if hmac.compare_digest(digest, p.key_sha256) and p.active:
            found = p
    if found is None and s.allow_legacy_api_key and s.api_key and hmac.compare_digest(key, s.api_key):
        found = Principal(LEGACY_ID, "Legacy shared key", ["admin", "system"], active=True)
    return found


def dev_principal() -> Principal:
    return Principal(DEV_ID, "Local development (no authentication)", ["admin", "system"])


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Manage BTI API principals")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--id", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--roles", required=True, help=f"comma-separated: {', '.join(ROLES)}")
    d = sub.add_parser("deactivate")
    d.add_argument("--id", required=True)
    sub.add_parser("list")
    args = parser.parse_args()
    if args.cmd == "add":
        p, key = add(args.id, args.name, [r.strip() for r in args.roles.split(",") if r.strip()])
        print(f"Created {p.id} with roles {p.roles}.\nKey (shown once, store it in a secret manager):\n{key}")
    elif args.cmd == "deactivate":
        print(f"Deactivated {deactivate(args.id).id}")
    else:
        for p in load().values():
            print(f"{p.id:24s} {'active' if p.active else 'INACTIVE':8s} {','.join(p.roles):40s} {p.name}")


if __name__ == "__main__":
    main()
