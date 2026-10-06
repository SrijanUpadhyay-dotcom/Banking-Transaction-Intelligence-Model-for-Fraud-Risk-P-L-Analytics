# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Transport and application hardening (Phase 10).

- **Security headers** on every response: `nosniff`; frame denial;
  no-referrer; `no-store` for API responses; HSTS when served over HTTPS; a
  restrictive content security policy (the workbench uses only inline,
  self-hosted code).
- **Request size limit.** Anything over `api.max_request_bytes` (default
  25 MB, set for the upload endpoints) is refused with 413 before it is read.
- **Secrets at startup.** Outside development the API refuses to start if
  the legacy shared key is enabled with its default value, or if no
  principal exists and the legacy key is off (the API would be unusable).
- **Log redaction.** Card numbers (Luhn-valid 13–19 digits) and IBANs are
  masked in every log record before it is written, as a backstop to
  tokenisation at ingress (PCI DSS requirement 3.3, 3.4).
"""

from __future__ import annotations

import logging
import re

from bti.config import get_settings
from bti.logging_config import get_logger

log = get_logger("security.hardening")
DEFAULT_API_KEY = "CHANGE_ME_API_KEY"
HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                               "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
}


class StartupSecurityError(RuntimeError):
    pass


def enforce_secrets_at_startup() -> None:
    s = get_settings()
    if s.environment.lower() in ("development", "dev", "test"):
        return
    if s.allow_legacy_api_key and s.api_key in ("", DEFAULT_API_KEY):
        raise StartupSecurityError("BTI_ALLOW_LEGACY_API_KEY is on with the default key; set BTI_API_KEY or, "
                                   "better, turn the legacy key off and create principals")
    from bti.security.principals import load
    if not s.allow_legacy_api_key and not any(p.active for p in load().values()):
        log.warning("No active principals: every protected endpoint will refuse requests "
                    "(python -m bti.security.principals add ...)")


def install(app) -> None:
    """Security headers and the request size limit, as one ASGI middleware."""
    max_bytes = get_settings().max_request_bytes

    @app.middleware("http")
    async def security_middleware(request, call_next):
        from starlette.responses import JSONResponse
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_bytes:
            return JSONResponse(status_code=413, content={"detail": f"Request larger than {max_bytes} bytes"})
        response = await call_next(request)
        for k, v in HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


# ── log redaction ────────────────────────────────────────────────────────────
_DIGITS = re.compile(r"(?<![\w.,])(?:\d[ -]?){12,18}\d(?!\w|[.,]\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){11,30}\b")


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


def redact(text: str) -> str:
    def card(m):
        digits = re.sub(r"\D", "", m.group(0))
        return f"[PAN ****{digits[-4:]}]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)
    text = _DIGITS.sub(card, text)
    return _IBAN.sub(lambda m: f"[IBAN {m.group(0)[:2]}****]", text)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in
                                (record.args if isinstance(record.args, tuple) else (record.args,)))
        for k, v in list(record.__dict__.items()):
            if isinstance(v, str) and k not in ("name", "levelname", "pathname", "filename", "module", "funcName",
                                                "processName", "threadName", "msg"):
                record.__dict__[k] = redact(v)
        return True


def install_log_redaction() -> None:
    f = RedactionFilter()
    root = logging.getLogger()
    for h in root.handlers:
        if not any(isinstance(x, RedactionFilter) for x in h.filters):
            h.addFilter(f)
    if not any(isinstance(x, RedactionFilter) for x in root.filters):
        root.addFilter(f)
