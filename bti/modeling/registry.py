"""
File-based model registry with champion / challenger roles.

Layout:
  models/registry/index.json            roles, model list, promotion history
  models/registry/<model_id>/model.joblib
  models/registry/<model_id>/card.json   metadata, validation, monitoring baseline

Promotion is gated: the model's automated validation must have passed and the
approver must differ from the developer (four-eyes principle).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from bti.config import get_settings

ROLES = ("champion", "challenger")
_lock = threading.Lock()
_cache: Dict[str, Any] = {}


class RegistryError(RuntimeError):
    pass


def registry_dir() -> Path:
    override = os.environ.get("BTI_MODEL_REGISTRY_DIR")
    return Path(override) if override else Path(get_settings().models_dir) / "registry"


def _index_path() -> Path:
    return registry_dir() / "index.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def read_index() -> Dict:
    path = _index_path()
    if not path.exists():
        return {"champion": None, "challenger": None, "models": [], "history": []}
    return json.loads(path.read_text())


def save_model(model_id: str, artifact: Dict, card: Dict) -> Path:
    folder = registry_dir() / model_id
    if folder.exists():
        raise RegistryError(f"Model {model_id} already registered; registry entries are immutable")
    folder.mkdir(parents=True)
    joblib.dump(artifact, folder / "model.joblib", compress=3)
    _atomic_write_json(folder / "card.json", card)
    with _lock:
        index = read_index()
        index["models"].append({
            "model_id": model_id,
            "registered_at": _now(),
            "developer": card.get("ownership", {}).get("developer"),
            "validation_status": card.get("validation", {}).get("status"),
        })
        _atomic_write_json(_index_path(), index)
    return folder


def load_card(model_id: str) -> Dict:
    path = registry_dir() / model_id / "card.json"
    if not path.exists():
        raise RegistryError(f"Unknown model {model_id}")
    return json.loads(path.read_text())


def load_artifact(model_id: str) -> Dict:
    with _lock:
        if model_id not in _cache:
            path = registry_dir() / model_id / "model.joblib"
            if not path.exists():
                raise RegistryError(f"Unknown model {model_id}")
            _cache[model_id] = joblib.load(path)
        return _cache[model_id]


def model_for_role(role: str) -> Optional[str]:
    if role not in ROLES:
        raise RegistryError(f"Role must be one of {ROLES}")
    return read_index().get(role)


def assign_role(model_id: str, role: str, approver: str, rationale: str) -> Dict:
    """Assign a model to a role. Champion promotion requires passed validation and four-eyes."""
    if role not in ROLES:
        raise RegistryError(f"Role must be one of {ROLES}")
    if not approver or not approver.strip():
        raise RegistryError("An approver is required")
    if not rationale or len(rationale.strip()) < 10:
        raise RegistryError("A rationale of at least 10 characters is required for the audit trail")
    card = load_card(model_id)
    developer = card.get("ownership", {}).get("developer")
    status = card.get("validation", {}).get("status")
    if role == "champion":
        if status != "passed":
            raise RegistryError(f"Model {model_id} validation status is '{status}'; only 'passed' models "
                                "can become champion")
        if developer and approver.strip().lower() == str(developer).strip().lower():
            raise RegistryError("Four-eyes principle: the approver must differ from the model developer")

    with _lock:
        index = read_index()
        if model_id not in {m["model_id"] for m in index["models"]}:
            raise RegistryError(f"Model {model_id} is not in the registry index")
        previous = index.get(role)
        index[role] = model_id
        if role == "champion" and index.get("challenger") == model_id:
            index["challenger"] = None
        event = {"at": _now(), "role": role, "model_id": model_id, "previous": previous,
                 "approver": approver.strip(), "rationale": rationale.strip()}
        index["history"].append(event)
        _atomic_write_json(_index_path(), index)
    return event


def clear_cache() -> None:
    with _lock:
        _cache.clear()
