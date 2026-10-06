# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Access policy: which roles may call which endpoints.

**One table, default deny.** `POLICY` is an ordered list of
(methods, path pattern, roles); the first match wins. An endpoint that no rule
matches is admin-only. A test checks that every route the API exposes matches
an explicit rule, so a new endpoint cannot ship without an access decision.
The table is also the access-control matrix in the certification evidence
pack.

**Roles** (`admin` may do everything):

| Role | Who | May |
|---|---|---|
| `scoring` | The authorisation host, stream consumer, SAS enrichment | Score transactions, issue and verify step-ups, ingest security events |
| `analyst` | Fraud analysts | Customer data, cases, alerts, early-warning and mule-alert reviews, the copilot |
| `operations` | Fraud operations and platform engineers | Run jobs; fit capacity; ingest labels and incumbent decisions; author rules; recalibrate; reload; plan |
| `model_risk` | Independent validation (second line) | Findings, sign-offs, champion promotion, approval of experiments and rules |
| `auditor` | Internal audit, external auditors | Read everything; change nothing |

Aggregate reports, which carry no customer-level data, are readable by every
human role.

**Public endpoints:**
- `/health`
- `/readyz`, which shows nothing sensitive unless the caller is authenticated
- the workbench page (it calls the API with the analyst's own key)
- the step-up callback, authenticated by its HMAC signature instead of a key

**Identity binding.** In a write request, the fields that say *who is acting*
must equal the authenticated principal's id:
- `approver`, `author`, `actor`, `analyst`, `reviewed_by`, `validator`,
  `fitted_by`, `signed_by`, `checker`

Four-eyes rests on these checks, which compare the approver with the
developer or maker. Only a `system` principal (a gateway that authenticated
the person upstream) may name someone else. Fields that name *another* person
are not bound: `assigned_to`, `owner`. `checked_by` is accepted only if it names a
checker who already confirmed with their own credentials (`POST /cases/{id}/check`).
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

ALL = ("GET", "POST", "PUT", "PATCH", "DELETE")
R, W = ("GET",), ("POST", "PUT", "PATCH", "DELETE")
HUMANS = ("analyst", "operations", "model_risk", "auditor")
PUBLIC = "public"
BINDING_FIELDS = ("approver", "author", "actor", "analyst", "reviewed_by", "validator", "fitted_by", "signed_by",
                  "checker")

_P = r"/api/v1"
POLICY: List[Tuple[Sequence[str], str, Sequence[str]]] = [
    # public
    (R, r"^/(health|readyz|workbench)$", (PUBLIC,)),
    (("POST",), _P + r"/stepup/challenges/[^/]+/callback$", (PUBLIC,)),            # HMAC-signed by the gateway
    # scoring services
    (("POST",), _P + r"/(v3/score(/batch)?|score/?|score/explain|score/upload|sas/enrich(/batch|/upload)?"
                     r"|scams/assess|operations/decide|parallel/decide|v3/security-events)$", ("scoring", "analyst")),
    (("POST",), _P + r"/stepup/challenges(/[^/]+/verify)?$", ("scoring",)),
    (R, _P + r"/(v3/(model|feeds)|score/(model-info|upload/template)|sas/(health|schema))$",
     ("scoring",) + HUMANS),
    # customer-level data: analysts and auditors
    (R, _P + r"/(transactions(/.*)?|analytics/risk/.*|cases(/.*)?|alerts/?|v3/explanations/[^/]+"
             r"|stepup/challenges/[^/]+|planning/early-warning|scams/mule-alerts)$", ("analyst", "auditor")),
    # analyst actions
    (W, _P + r"/(alerts/refresh|alerts/[^/]+|cases|cases/next|cases/[^/]+/(assign|disposition|pending-customer|check)"
             r"|copilot/(ask|summarize/ring)|graph/analyze(/upload)?|planning/early-warning/[^/]+/review"
             r"|scams/mule-alerts/[^/]+/review)$", ("analyst",)),
    # model risk (second line)
    (W, _P + r"/(governance/(findings(/[^/]+)?|signoffs|models/[^/]+/promote)"
             r"|parallel/experiments/[^/]+/approve|rules/versions/[^/]+/approve)$", ("model_risk",)),
    # operations
    (W, _P + r"/(cases/sla-check|governance/(audit/archive|drift/run|outcomes/run|check|models/[^/]+/benchmark)"
             r"|operations/(cost-model/backtest|cost-model/customer-values/refresh|graph/snapshot|labels"
             r"|policy/backtest|policy/capacity/fit|recalibration/(run|rollback)|retraining/run)"
             r"|parallel/(experiments|experiments/[^/]+/stop|incumbent/(decisions|upload)|reconcile/run|report/run)"
             r"|pipeline/run(/sync)?|planning/(whatif|early-warning/run)|rules|rules/[^/]+/versions"
             r"|rules/versions/[^/]+/(simulate|retire)|scams/mule-alerts/run|score/reload-models|stepup/expire"
             r"|analytics/model/compare|governance/psd2/tra-eligibility)$", ("operations",)),
    # aggregate reports: every human role
    (R, _P + r"/(analytics/(pnl/.*|model/lift)|governance/.*|operations/.*|parallel/.*|pipeline/status"
             r"|planning/(forecast/.*|whatif/[^/]+)|rules(/.*)?|scams/(models|reports/.*)|stepup/stats"
             r"|copilot/health|graph/health)$", HUMANS),
]
_COMPILED = [(set(m), re.compile(p), roles) for m, p, roles in POLICY]


def rule_for(method: str, path: str) -> Optional[Tuple[str, Sequence[str]]]:
    """(pattern, roles) of the first matching rule, or None (default deny: admin only)."""
    for methods, pattern, roles in _COMPILED:
        if method.upper() in methods and pattern.match(path):
            return pattern.pattern, roles
    return None


def allowed_roles(method: str, path: str) -> Sequence[str]:
    r = rule_for(method, path)
    return r[1] if r else ("admin",)


def is_public(method: str, path: str) -> bool:
    return PUBLIC in allowed_roles(method, path)


def matrix(routes: Iterable[Tuple[str, str]]) -> List[dict]:
    """The access-control matrix for a list of (method, example path): evidence for access reviews."""
    out = []
    for method, path in routes:
        r = rule_for(method, path)
        out.append({"method": method, "path": path, "roles": list(r[1]) if r else ["admin"],
                    "rule": r[0] if r else "default deny (admin only)"})
    return out


def identity_mismatches(body, principal_id: str) -> List[str]:
    """Binding fields in a JSON body that name someone other than the authenticated principal."""
    if not isinstance(body, dict):
        return []
    return [f for f in BINDING_FIELDS
            if f in body and body[f] is not None and str(body[f]).strip().lower() != principal_id.lower()]
