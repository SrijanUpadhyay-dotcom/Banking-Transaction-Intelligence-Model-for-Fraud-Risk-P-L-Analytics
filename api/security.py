# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
API authentication and authorisation (Phase 10).

`authorize` runs on every API route as a global dependency:
1. Resolve the principal from `X-API-Key` (`bti.security.principals`). In the
   development environment only, a request without a key acts as a local
   admin. That is never available in production.
2. Check the route's roles in the access policy (`bti.security.access`;
   default deny). Public routes pass without a key.
3. For writes, bind the identity fields in the JSON body (approver, author,
   reviewer, validator…) to the authenticated principal. A `system`
   principal (an upstream SSO gateway) is exempt.
4. Apply a per-principal rate limit.

Results: 401 with no or an invalid key, 403 with the wrong role or a
mismatched identity, 429 over the rate limit. The principal is available to
handlers as `request.state.principal`.

`require_api_key` remains on the routes that declared it, as a second
explicit check that a principal is present.
"""

import json
import threading
import time
from collections import defaultdict
from typing import Optional

from fastapi import Header, HTTPException, Request

from bti.config import get_settings
from bti.logging_config import get_logger
from bti.security import access, principals

log = get_logger("api.security")
_rate_lock = threading.Lock()
_rate: dict = defaultdict(lambda: [0, 0])          # principal -> [window minute, count]


def _rate_limit(key: str, role_bucket: str) -> None:
    limits = get_settings().rate_limit_per_minute
    limit = int(limits.get(role_bucket, limits.get("default", 600)))
    minute = int(time.time() // 60)
    with _rate_lock:
        window = _rate[key]
        if window[0] != minute:
            window[0], window[1] = minute, 0
        window[1] += 1
        over = window[1] > limit
    if over:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; retry next minute",
                            headers={"Retry-After": "60"})


async def authorize(request: Request, x_api_key: Optional[str] = Header(default=None)) -> None:
    settings = get_settings()
    method, path = request.method.upper(), request.url.path
    principal = principals.authenticate(x_api_key)
    if principal is None and x_api_key is None and settings.environment == "development":
        principal = principals.dev_principal()
    if access.is_public(method, path):
        request.state.principal = principal
        _rate_limit(principal.id if principal else f"ip:{request.client.host if request.client else '?'}",
                    "default" if principal else "anonymous")
        return
    if principal is None:
        log.warning("Unauthenticated request refused", extra={"method": method, "path": path})
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
    roles = access.allowed_roles(method, path)
    if not principal.has(*roles):
        log.warning("Forbidden", extra={"principal": principal.id, "method": method, "path": path,
                                        "required": list(roles)})
        raise HTTPException(status_code=403, detail=f"Requires one of the roles: {', '.join(roles)}")
    if method in access.W and "system" not in principal.roles and \
            "application/json" in request.headers.get("content-type", ""):
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        bad = access.identity_mismatches(body, principal.id)
        if bad:
            log.warning("Identity mismatch", extra={"principal": principal.id, "fields": bad, "path": path})
            raise HTTPException(status_code=403, detail=f"{', '.join(bad)} must be the authenticated principal "
                                                        f"({principal.id})")
    request.state.principal = principal
    _rate_limit(principal.id, "scoring" if "scoring" in principal.roles else "default")


def require_api_key(request: Request, x_api_key: Optional[str] = Header(default=None)) -> None:
    """Explicit guard kept on state-changing routes: a principal must have been authenticated."""
    if getattr(request.state, "principal", None) is None:
        principal = principals.authenticate(x_api_key)
        if principal is None and not (x_api_key is None and get_settings().environment == "development"):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
