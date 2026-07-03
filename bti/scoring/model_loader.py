"""
Model loader with in-process caching.
Models are loaded from disk once and held in memory for the lifetime of the process.
Thread-safe via a module-level lock.
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from bti.config import get_settings
from bti.logging_config import get_logger

log = get_logger("scoring.model_loader")
_lock = threading.Lock()
_cached_bundle: Optional["ModelBundle"] = None


@dataclass
class ModelBundle:
    """All ML artifacts needed for real-time scoring, loaded into memory once."""
    iso_forest:     Any
    iso_scaler:     Any
    lr_model:       Any
    lr_scaler:      Any
    rf_model:       Any
    feature_cols:   List[str]
    trained_at:     str
    label_encoders: dict = field(default_factory=dict)
    lr_roc_auc:     float = 0.0
    rf_roc_auc:     float = 0.0
    rf_f1:          float = 0.0
    manifest_path:  str = ""


def load_models(models_dir: Optional[str] = None, force_reload: bool = False) -> ModelBundle:
    """
    Load models from disk. Returns the cached bundle on subsequent calls
    unless force_reload=True.

    Raises FileNotFoundError if models haven't been trained yet.
    """
    global _cached_bundle
    if _cached_bundle is not None and not force_reload:
        return _cached_bundle

    with _lock:
        if _cached_bundle is not None and not force_reload:
            return _cached_bundle

        import joblib
        settings = get_settings()
        mdir = Path(models_dir or settings.models_dir)
        manifest_path = mdir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Model manifest not found at {manifest_path}. "
                "Run the pipeline first: python main.py pipeline --force"
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        log.info("Loading ML models from disk", extra={"models_dir": str(mdir)})

        le_path = mdir / "label_encoders.joblib"
        label_encoders = joblib.load(le_path) if le_path.exists() else {}

        bundle = ModelBundle(
            iso_forest=joblib.load(mdir / "isolation_forest.joblib"),
            iso_scaler=joblib.load(mdir / "iso_scaler.joblib"),
            lr_model=joblib.load(mdir / "logistic_regression.joblib"),
            lr_scaler=joblib.load(mdir / "lr_scaler.joblib"),
            rf_model=joblib.load(mdir / "random_forest.joblib"),
            feature_cols=manifest["feature_cols"],
            trained_at=manifest.get("trained_at", "unknown"),
            label_encoders=label_encoders,
            lr_roc_auc=manifest.get("lr_metrics", {}).get("roc_auc", 0.0),
            rf_roc_auc=manifest.get("rf_metrics", {}).get("roc_auc", 0.0),
            rf_f1=manifest.get("rf_metrics", {}).get("f1", 0.0),
            manifest_path=str(manifest_path),
        )
        _cached_bundle = bundle

        log.info(
            "Models loaded",
            extra={
                "feature_cols": len(bundle.feature_cols),
                "trained_at": bundle.trained_at,
                "lr_roc_auc": bundle.lr_roc_auc,
                "rf_roc_auc": bundle.rf_roc_auc,
            }
        )
        return bundle


def models_available(models_dir: Optional[str] = None) -> bool:
    settings = get_settings()
    mdir = Path(models_dir or settings.models_dir)
    return (mdir / "manifest.json").exists()


def invalidate_cache() -> None:
    """Call after retraining to force the next request to reload from disk."""
    global _cached_bundle
    with _lock:
        _cached_bundle = None
    log.info("Model cache invalidated — next request will reload from disk")
