# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Certification evidence pack: one command collects the system-generated evidence
an auditor asks for, and makes the collection tamper-evident.

    python -m bti.security.evidence            # -> outputs/evidence/<UTC timestamp>/

**Contents:**

| File | What it evidences |
|---|---|
| `access_matrix.json` | Every API route and the roles allowed (SOC 2 CC6.1/CC6.3; ISO 27001 A.5.15, A.8.3) |
| `principals_review.json` | Who has access, with roles, active state and creation date, for periodic access reviews. No key material (CC6.2; A.5.18) |
| `security_assessment.json` | Automated self-assessment: access probes, injection probes, SAST, dependency audit, secret and card-data scans (CC7.1; A.8.8, A.8.28, A.8.29) |
| `sbom.cyclonedx.json` | Software bill of materials for the deployable dependency set, with known vulnerabilities (A.8.8; PCI DSS 6.3.2) |
| `audit_chain.json` | Verification of the hash-chained audit log and archive retention (CC7.2, CC4.1; A.8.15; PCI DSS 10.3) |
| `change_history.json` | Model promotions and post-registration notes (all families), rule versions with author and approver, capacity-policy history (CC8.1; A.8.32) |
| `validation.json` | Model inventory, validation findings, independent sign-offs (SR 11-7; CC3.2) |
| `config_snapshot.json` | Security-relevant settings, with secrets removed (CC8.1; A.8.9) |
| `residency.json`, `readiness.json` | Data residency and operational readiness (C1.1, A1.2; A.5.34) |
| `manifest.json` | SHA-256 of every file above, the git commit, and the generation time |

**Tamper evidence.** The manifest's own SHA-256 is written to the hash-chained
audit log as an `EVIDENCE_PACK` event. A pack altered after generation no
longer matches its anchored hash.

**Scope.** These are system-generated artefacts only. A SOC 2 or ISO 27001
audit also needs organisational evidence that no code produces: policies,
risk assessment, HR screening and training, vendor management, incident
records, management review. See docs/CERTIFICATION_READINESS.md.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

SECRET_WORDS = ("key", "secret", "password", "token", "smtp_user")
ROOT = Path("outputs/evidence")


def _write(folder: Path, name: str, data) -> Path:
    path = folder / name
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def _config_snapshot() -> Dict:
    from bti.config import get_settings
    s = get_settings().model_dump()
    clean = {}
    for k, v in s.items():
        if any(w in k.lower() for w in SECRET_WORDS) and k not in ("allow_legacy_api_key",):
            clean[k] = "<redacted: set>" if v else "<not set>"
        elif "url" in k.lower() and isinstance(v, str) and "@" in v:
            clean[k] = v.split("@")[-1]
        else:
            clean[k] = v
    return clean


def _access_matrix() -> Dict:
    import re
    from api.main import app
    from bti.security import access
    routes = [(m.upper(), p) for p, ops in app.openapi()["paths"].items() for m in ops]
    rows = access.matrix([(m, re.sub(r"\{[^}]+\}", "X1", p)) for m, p in routes])
    for row, (_, template) in zip(rows, routes):
        row["path"] = template
    return {"roles": {"scoring": "authorisation host / stream consumer", "analyst": "fraud analysts",
                      "operations": "fraud operations and platform", "model_risk": "independent validation",
                      "auditor": "read-only audit", "admin": "everything", "system": "upstream SSO gateway"},
            "identity_bound_fields": list(access.BINDING_FIELDS), "routes": rows}


def _principals() -> Dict:
    from bti.security.principals import load, principals_file
    return {"source": str(principals_file()),
            "principals": [{"id": p.id, "name": p.name, "roles": p.roles, "active": p.active,
                            "created_at": p.created_at} for p in load().values()]}


def _change_history(db) -> Dict:
    from bti.database.models import RuleVersion
    from bti.modeling import registry
    families = {}
    for fam in registry.FAMILIES:
        idx = registry.read_index(fam)
        families[fam] = {"roles": {"champion": idx.get("champion"), "challenger": idx.get("challenger")},
                         "models": idx.get("models", []), "role_history": idx.get("history", []),
                         "notes": idx.get("notes", [])}
    rules = [{c: getattr(r, c) for c in ("rule_id", "version", "action", "author", "created_at", "status",
                                         "approved_by", "approved_at", "retired_by", "retired_at")}
             for r in db.query(RuleVersion).order_by(RuleVersion.id).all()] if db is not None else []
    policies = {}
    pdir = registry.registry_dir() / "policies"
    if pdir.exists():
        for f in pdir.glob("*.json"):
            d = json.loads(f.read_text())
            policies[d.get("model_id", f.stem)] = [{k: e.get(k) for k in ("fitted_at", "fitted_by", "targets")}
                                                   for e in d.get("history", []) + [d.get("current", {})]]
    return {"model_registry": families, "rule_versions": rules, "capacity_policies": policies}


def _validation(db) -> Dict:
    from bti.governance import validation
    out = {}
    for name, fn in (("findings", lambda: validation.list_findings(db)),
                     ("signoffs", lambda: validation.list_signoffs(db)),
                     ("inventory", lambda: validation.sync_inventory(db))):
        try:
            out[name] = fn()
        except Exception as exc:
            out[name] = {"error": str(exc)}
    return out


def _sbom(folder: Path) -> Optional[Path]:
    path = folder / "sbom.cyclonedx.json"
    r = subprocess.run([sys.executable, "-m", "pip_audit", "-r", "requirements.txt", "-f", "cyclonedx-json",
                        "--progress-spinner", "off", "-o", str(path)], capture_output=True, text=True, timeout=1200)
    return path if path.exists() else None


def build(db=None, run_assessment: bool = True, sbom: bool = True) -> Dict:
    from bti.governance.audit_chain import retention_report, verify_chain
    from bti.operations.readiness import readiness
    from bti.operations.residency import check
    own = db is None
    if own:
        from bti.database.connection import SessionLocal
        db = SessionLocal()
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = ROOT / stamp
        folder.mkdir(parents=True, exist_ok=True)
        files = [
            _write(folder, "access_matrix.json", _access_matrix()),
            _write(folder, "principals_review.json", _principals()),
            _write(folder, "audit_chain.json", {"verification": verify_chain(db), "retention": retention_report()}),
            _write(folder, "change_history.json", _change_history(db)),
            _write(folder, "validation.json", _validation(db)),
            _write(folder, "config_snapshot.json", _config_snapshot()),
            _write(folder, "residency.json", check()),
            _write(folder, "readiness.json", readiness()),
        ]
        if run_assessment:
            from bti.security.assessment import run
            files.append(_write(folder, "security_assessment.json", run()))
        sbom_path = _sbom(folder) if sbom else None
        if sbom_path:
            files.append(sbom_path)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        manifest = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": commit,
                    "working_tree_clean": not dirty,
                    "files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
        mpath = _write(folder, "manifest.json", manifest)
        digest = hashlib.sha256(mpath.read_bytes()).hexdigest()
        from bti.database.models import AuditLog
        db.add(AuditLog(ts=datetime.utcnow(), event_type="EVIDENCE_PACK",
                        payload={"folder": str(folder), "manifest_sha256": digest, "files": len(files)}))
        db.commit()
        return {"folder": str(folder), "files": sorted(manifest["files"]), "manifest_sha256": digest,
                "git_commit": commit}
    finally:
        if own:
            db.close()


def verify(folder: str, db=None) -> Dict:
    """Recompute every hash in a pack and check the manifest hash against the anchored audit event."""
    from bti.database.models import AuditLog
    f = Path(folder)
    manifest = json.loads((f / "manifest.json").read_text())
    changed = [n for n, h in manifest["files"].items() if hashlib.sha256((f / n).read_bytes()).hexdigest() != h]
    digest = hashlib.sha256((f / "manifest.json").read_bytes()).hexdigest()
    own = db is None
    if own:
        from bti.database.connection import SessionLocal
        db = SessionLocal()
    try:
        anchored = any(e.payload and e.payload.get("manifest_sha256") == digest
                       for e in db.query(AuditLog).filter(AuditLog.event_type == "EVIDENCE_PACK").all())
    finally:
        if own:
            db.close()
    return {"folder": folder, "files_changed": changed, "manifest_anchored_in_audit_log": anchored,
            "intact": not changed and anchored}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build or verify a certification evidence pack")
    parser.add_argument("command", nargs="?", choices=["build", "verify"], default="build")
    parser.add_argument("--folder", default=None)
    parser.add_argument("--skip-assessment", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.folder), indent=2))
    else:
        print(json.dumps(build(run_assessment=not args.skip_assessment), indent=2))


if __name__ == "__main__":
    main()
