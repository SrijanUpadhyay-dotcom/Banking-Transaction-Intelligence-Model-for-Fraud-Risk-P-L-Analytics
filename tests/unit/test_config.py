"""Unit tests for configuration management."""

import os
import pytest
from pathlib import Path


class TestSettings:
    def test_settings_load_without_error(self):
        from bti.config import get_settings
        s = get_settings()
        assert s is not None

    def test_default_database_url_is_sqlite(self):
        from bti.config import get_settings
        s = get_settings()
        # Default should be SQLite unless overridden
        if "BTI_DATABASE_URL" not in os.environ:
            assert "sqlite" in s.database_url

    def test_version_is_set(self):
        from bti.config import get_settings
        s = get_settings()
        assert s.app_version == "2.0.0"

    def test_fraud_thresholds_are_positive(self):
        from bti.config import get_settings
        s = get_settings()
        assert s.velocity_spike_threshold > 0
        assert s.failed_auth_threshold > 0
        assert s.high_value_ratio > 1.0
        assert 0 < s.fraud_prob_threshold < 1.0

    def test_paths_are_absolute_strings(self):
        from bti.config import get_settings
        s = get_settings()
        # These should be absolute paths
        assert Path(s.outputs_dir).is_absolute() or s.outputs_dir
        assert Path(s.models_dir).is_absolute() or s.models_dir
