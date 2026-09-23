"""Shared API-key guard for state-changing endpoints (model promotion, label ingestion)."""

import hmac
from typing import Optional

from fastapi import Header, HTTPException

from bti.config import get_settings


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    settings = get_settings()
    if settings.environment != "development":
        if not hmac.compare_digest(x_api_key or "", settings.api_key or ""):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
