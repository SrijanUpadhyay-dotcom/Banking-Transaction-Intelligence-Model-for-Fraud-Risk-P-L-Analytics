"""Test-wide settings: fix the API key before any module reads (and caches) the settings."""

import os

os.environ.setdefault("BTI_API_KEY", "test-api-key")

from bti.config import get_settings  # noqa: E402

get_settings.cache_clear()
