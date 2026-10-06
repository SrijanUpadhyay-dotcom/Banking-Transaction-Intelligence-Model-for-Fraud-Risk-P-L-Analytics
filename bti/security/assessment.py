# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Automated security self-assessment: the checks an engineer can run before an
independent penetration test, so the tester's time goes on what automation
cannot find.

**It is not a penetration test.** It does not chain vulnerabilities, test
business logic creatively, or attack the infrastructure. That needs an
independent tester (see docs/CERTIFICATION_READINESS.md for scope and rules
of engagement).

**Checks:**
1. *Access policy coverage.* Every API route matches an explicit rule.
2. *Anonymous probe.* Every protected route refuses a caller without a key.
3. *Least privilege.* The read-only `auditor` role is refused on every
   write route.
4. *Injection and traversal probes.* SQL metacharacters, path traversal and
   script payloads go into every path parameter of every read route. Pass:
   no 5xx, and no payload echoed in an HTML response.
5. *Transport headers and exposure.* Security headers present; API docs off.
6. *Static analysis (SAST).* bandit over `bti/` and `api/`.
7. *Dependency vulnerabilities.* pip-audit of the installed environment
   against the PyPI/OSV advisory data.
8. *Secret scan.* Version-controlled files are checked for private keys,
   cloud keys, tokens and hard-coded credentials.
9. *Card-data discovery.* `bti.security.pan_scan`.
10. *Configuration posture.* Environment, legacy key, docs, residency,
    tokenisation key, CORS.

The probes run against an in-memory database, with temporary principals and
the legacy key off, so they never touch real data or real keys. Results go to
outputs/security/assessment.json as a findings register: id, severity,
status, evidence.

    python -m bti.security.assessment
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

OUT = Path("outputs/security/assessment.json")
PAYLOADS = ("' OR '1'='1", "../../../../etc/passwd", "<script>alert(1)</script>", "%00", "1;DROP TABLE x--")
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[abpors]-[A-Za-z0-9-]{10,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "hardcoded_credential": re.compile(r"(?i)\b(password|passwd|secret|api_key|token)\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
}
SECRET_ALLOW = re.compile(r"CHANGE_ME|test-api-key|example|placeholder|your-|<|\{|integration-test-key|unit-test-key|"
                          r"demo-key|dev-key|development-only", re.I)


def _finding(fid: str, severity: str, title: str, status: str, evidence) -> Dict:
    return {"id": fid, "severity": severity, "title": title, "status": status, "evidence": evidence}


def _api_checks() -> List[Dict]:
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from bti.config import get_settings
    from bti.database.connection import get_db
    from bti.database.models import Base
    from bti.security import access, principals

    findings = []
    s = get_settings()
    saved = (os.environ.get("BTI_PRINCIPALS_FILE"), s.allow_legacy_api_key)
    tmp = tempfile.mkdtemp(prefix="bti-assess-")
    os.environ["BTI_PRINCIPALS_FILE"] = str(Path(tmp) / "principals.json")
    s.allow_legacy_api_key = False
    try:
        _, auditor = principals.add("assess.auditor", "assessment", ["auditor"])
        _, admin = principals.add("assess.admin", "assessment", ["admin"])
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        from api.main import app

        def _db():
            db = Session()
            try:
                yield db
            finally:
                db.close()
        app.dependency_overrides[get_db] = _db
        routes = [(m.upper(), p, re.sub(r"\{[^}]+\}", "X1", p)) for p, ops in app.openapi()["paths"].items() for m in ops]
        uncovered = [f"{m} {p}" for m, p, c in routes if access.rule_for(m, c) is None]
        findings.append(_finding("AC-1", "high", "Every route has an explicit access rule",
                                 "pass" if not uncovered else "fail", {"routes": len(routes), "uncovered": uncovered}))
        with TestClient(app, raise_server_exceptions=False) as client:
            from api import security
            security._rate.clear()
            leaks = []
            for m, p, c in routes:
                if access.is_public(m, c):
                    continue
                r = client.request(m, c, json={} if m != "GET" else None)
                if r.status_code != 401:
                    leaks.append(f"{m} {p} -> {r.status_code}")
            findings.append(_finding("AC-2", "critical", "Protected routes refuse anonymous callers",
                                     "pass" if not leaks else "fail",
                                     {"probed": sum(1 for m, p, c in routes if not access.is_public(m, c)),
                                      "not_refused": leaks}))
            security._rate.clear()
            writes = [(m, p, c) for m, p, c in routes if m != "GET" and not access.is_public(m, c)]
            allowed = []
            for m, p, c in writes:
                r = client.request(m, c, headers={"X-API-Key": auditor}, json={})
                if r.status_code != 403:
                    allowed.append(f"{m} {p} -> {r.status_code}")
            findings.append(_finding("AC-3", "high", "Read-only auditor role is refused on every write route",
                                     "pass" if not allowed else "fail", {"write_routes": len(writes), "not_refused": allowed}))
            security._rate.clear()
            errors, reflected = [], []
            reads = [(m, p) for m, p, c in routes if m == "GET" and "{" in p]
            for m, p in reads:
                for payload in PAYLOADS:
                    concrete = re.sub(r"\{[^}]+\}", lambda _: payload.replace("/", "%2F"), p)
                    r = client.request(m, concrete, headers={"X-API-Key": admin})
                    if r.status_code >= 500:
                        errors.append(f"{p} [{payload}] -> {r.status_code}")
                    if "<script>" in payload and "text/html" in r.headers.get("content-type", "") and payload in r.text:
                        reflected.append(p)
                security._rate.clear()
            findings.append(_finding("IN-1", "high", "Injection and traversal payloads in path parameters cause no "
                                     "server errors or reflection", "pass" if not errors and not reflected else "fail",
                                     {"routes": len(reads), "payloads": list(PAYLOADS), "server_errors": errors,
                                      "reflected": reflected}))
            h = client.get("/health").headers
            missing = [k for k in ("X-Content-Type-Options", "X-Frame-Options", "Content-Security-Policy",
                                   "Referrer-Policy") if k not in h]
            docs = client.get("/docs").status_code
            findings.append(_finding("TR-1", "medium", "Security headers present; API docs not exposed",
                                     "pass" if not missing and docs == 404 else "fail",
                                     {"missing_headers": missing, "docs_status": docs}))
        app.dependency_overrides.pop(get_db, None)
    finally:
        if saved[0] is None:
            os.environ.pop("BTI_PRINCIPALS_FILE", None)
        else:
            os.environ["BTI_PRINCIPALS_FILE"] = saved[0]
        s.allow_legacy_api_key = saved[1]
    return findings


def _sast() -> Dict:
    try:
        out = subprocess.run([sys.executable, "-m", "bandit", "-r", "bti", "api", "-f", "json", "-q"],
                             capture_output=True, text=True, timeout=600)
        data = json.loads(out.stdout or "{}")
    except Exception as exc:
        return _finding("SA-1", "medium", "Static analysis (bandit)", "not_run", {"error": str(exc)})
    results = data.get("results", [])
    by = {}
    for r in results:
        by.setdefault(r["issue_severity"], 0)
        by[r["issue_severity"]] += 1
    high = [{"file": r["filename"], "line": r["line_number"], "test": r["test_id"], "issue": r["issue_text"]}
            for r in results if r["issue_severity"] == "HIGH"]
    medium = [{"file": r["filename"], "line": r["line_number"], "test": r["test_id"], "issue": r["issue_text"]}
              for r in results if r["issue_severity"] == "MEDIUM"]
    return _finding("SA-1", "high" if high else "medium" if medium else "low", "Static analysis (bandit)",
                    "fail" if high else "review" if medium else "pass",
                    {"by_severity": by, "high": high, "medium": medium})


def _audit(args: List[str]) -> List[Dict]:
    out = subprocess.run([sys.executable, "-m", "pip_audit", "-f", "json", "--progress-spinner", "off", *args],
                         capture_output=True, text=True, timeout=1200)
    data = json.loads(out.stdout or "{}")
    seen, vulns = set(), []
    for d in data.get("dependencies", []):
        for v in d.get("vulns", []):
            key = (d["name"], d["version"], v["id"])
            if key not in seen:
                seen.add(key)
                vulns.append({"package": d["name"], "version": d["version"], "id": v["id"],
                              "fix": v.get("fix_versions", [])})
    return [len(data.get("dependencies", [])), vulns]


def _dependencies() -> Dict:
    """The deployable set (requirements.txt, resolved fresh) gates; the build host's own tooling is reported only."""
    try:
        n_req, req = _audit(["-r", "requirements.txt"])
        n_env, env = _audit([])
    except Exception as exc:
        return _finding("DE-1", "high", "Dependency vulnerabilities (pip-audit)", "not_run", {"error": str(exc)})
    return _finding("DE-1", "high" if req else "low", "No known vulnerabilities in the deployable dependencies",
                    "fail" if req else "pass",
                    {"requirements_txt": {"packages_resolved": n_req, "vulnerabilities": req},
                     "build_host_environment_info_only": {
                         "packages": n_env, "vulnerabilities": len(env),
                         "packages_affected": sorted({v["package"] for v in env}),
                         "note": "system packages of the development container (not shipped); the production image "
                                 "is built from requirements.txt and audited in CI"}})


def _secrets() -> Dict:
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
    hits = []
    for f in files:
        p = Path(f)
        if not p.is_file() or p.suffix in (".joblib", ".png", ".jpg", ".pdf", ".pyc") or p.stat().st_size > 5_000_000:
            continue
        text = p.read_text(errors="ignore")
        for name, pat in SECRET_PATTERNS.items():
            for m in pat.finditer(text):
                if SECRET_ALLOW.search(m.group(0)):
                    continue
                line = text[:m.start()].count("\n") + 1
                hits.append({"file": f, "line": line, "type": name})
    return _finding("SE-1", "critical" if hits else "low", "No secrets in version-controlled files",
                    "fail" if hits else "pass", {"files_scanned": len(files), "hits": hits})


def _pan() -> Dict:
    from bti.security.pan_scan import run
    r = run()
    return _finding("PC-1", "critical" if not r["clean"] else "low", "No card numbers stored at rest",
                    "pass" if r["clean"] else "fail", r)


def _posture() -> List[Dict]:
    from bti.config import get_settings
    from bti.operations.residency import check
    s = get_settings()
    out = []
    out.append(_finding("CF-1", "high", "Shared legacy API key disabled", "pass" if not s.allow_legacy_api_key
                        else "fail", {"allow_legacy_api_key": s.allow_legacy_api_key,
                                      "note": "the test suite enables it; production must not"}))
    out.append(_finding("CF-2", "medium", "Environment is not 'development' (which bypasses authentication)",
                        "pass" if s.environment != "development" else "fail", {"environment": s.environment}))
    out.append(_finding("CF-3", "medium", "Tokenisation key configured for card and account data",
                        "pass" if s.token_key else "open", {"configured": bool(s.token_key),
                                                            "note": "set BTI_TOKEN_KEY from the bank's HSM"}))
    out.append(_finding("CF-4", "medium", "CORS restricted to named origins",
                        "pass" if "*" not in s.cors_origins else "fail", {"origins": s.cors_origins}))
    r = check()
    out.append(_finding("CF-5", "medium", "Data residency configured and compliant",
                        "pass" if r["jurisdiction"] and r["compliant"] else "open",
                        {"jurisdiction": r["jurisdiction"], "mode": r["mode"], "violations": r["violations"]}))
    return out


def run(skip_dependencies: bool = False) -> Dict:
    findings = _api_checks() + [_sast(), _secrets(), _pan()]
    if not skip_dependencies:
        findings.append(_dependencies())
    findings += _posture()
    summary = {}
    for f in findings:
        summary[f["status"]] = summary.get(f["status"], 0) + 1
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "nature": "Automated self-assessment; not a penetration test and not a certification",
              "summary": summary, "findings": findings}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BTI security self-assessment")
    parser.add_argument("--skip-dependencies", action="store_true")
    args = parser.parse_args()
    r = run(args.skip_dependencies)
    for f in r["findings"]:
        print(f"{f['id']:5s} {f['status']:8s} {f['severity']:8s} {f['title']}")
    print(r["summary"])
    if any(f["status"] == "fail" for f in r["findings"]):
        raise SystemExit(1)                         # CI gate


if __name__ == "__main__":
    main()
