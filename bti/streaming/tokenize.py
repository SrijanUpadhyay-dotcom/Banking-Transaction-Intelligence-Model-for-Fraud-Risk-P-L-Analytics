# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Keyed tokenisation of card numbers and account identifiers.

Card numbers (PANs) and IBANs arrive in payment messages but must never be
stored in BTI. PCI DSS requirement 3 forbids keeping a PAN unprotected, and
account numbers are personal data under GDPR and India's DPDP Act. Each
identifier is replaced on arrival by an HMAC-SHA256 token under a secret key:
- the token is stable, so a card or account keeps its history
- without the key, it cannot be reversed or recomputed from a guessed number.
  A plain hash of a 16-digit PAN can be brute-forced in minutes.

The key comes from BTI_TOKEN_KEY, set in the bank's secret store or HSM.
Without it, tokenisation refuses to run, except in a development environment,
which uses a fixed key and logs a warning.

Only the last four digits of a PAN may be kept, for display.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re

from bti.config import get_settings
from bti.logging_config import get_logger

log = get_logger("streaming.tokenize")
_DEV_KEY = b"bti-development-only-token-key"
_warned = False


class TokenisationError(RuntimeError):
    pass


def _key() -> bytes:
    global _warned
    settings = get_settings()
    if settings.token_key:
        return settings.token_key.encode()
    if settings.environment.lower() in ("development", "dev", "test"):
        if not _warned:
            log.warning("BTI_TOKEN_KEY not set: using the development tokenisation key (never in production)")
            _warned = True
        return _DEV_KEY
    raise TokenisationError("BTI_TOKEN_KEY is not set. Card numbers and account identifiers cannot be accepted "
                            "without a tokenisation key; set it from the bank's secret store.")


def token(kind: str, value: str) -> str:
    """Stable keyed token, e.g. token('PAN', '4111...') -> 'PAN-5W3Q...' (26 base32 characters)."""
    normalised = re.sub(r"[\s-]", "", str(value)).upper()
    if not normalised:
        raise TokenisationError(f"empty {kind}")
    digest = hmac.new(_key(), f"{kind}:{normalised}".encode(), hashlib.sha256).digest()
    return f"{kind}-{base64.b32encode(digest[:16]).decode().rstrip('=')}"


def luhn_ok(pan: str) -> bool:
    digits = [int(c) for c in pan if c.isdigit()]
    if len(digits) != len(pan) or not 12 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


def last4(value: str) -> str:
    digits = re.sub(r"\D", "", str(value))
    return digits[-4:] if len(digits) >= 4 else ""
