# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Test-wide settings: fix the API key before any module reads (and caches) the settings."""

import os

os.environ.setdefault("BTI_API_KEY", "test-api-key")
# Existing integration tests authenticate with the legacy shared key; the Phase 10 tests switch it off and use
# principals. Production keeps it off (bti.security.principals).
os.environ.setdefault("BTI_ALLOW_LEGACY_API_KEY", "true")
os.environ.setdefault("BTI_PRINCIPALS_FILE", os.path.join(os.path.dirname(__file__), ".principals.test.json"))

from bti.config import get_settings  # noqa: E402

get_settings.cache_clear()
