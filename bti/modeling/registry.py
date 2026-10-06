# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
File-based model registry with champion / challenger roles.

Layout:
  models/registry/index.json            roles, model list, promotion history, post-registration notes
  models/registry/<model_id>/model.joblib
  models/registry/<model_id>/card.json   metadata, validation, monitoring baseline

Promotion is gated: the model's automated validation must have passed and the
approver must differ from the developer (four-eyes principle).

**Families.** Each model family has its own registry, index and roles, with
the same governance (immutable entries, validation gate, four-eyes, notes):
- `fraud`: the transaction model, in models/registry
- `scam`: APP scams, Phase 9, in models/registry_scam
- `mule`: mule accounts, Phase 9, in models/registry_mule

A function called without `family` acts on `fraud`, so existing code is
unchanged. Keeping families apart means jobs that iterate the fraud registry,
such as tournaments, fairness reassessment and the validation inventory,
never meet a scam or mule artifact they cannot score.

Registered cards are never edited. When a model is re-assessed after
registration (a corrected test, a new finding, a retraction), the result is
appended to `notes` in the index, so the record shows both what was believed at
registration and what is known now.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from bti.config import get_settings

ROLES = ("champion", "challenger")
FAMILIES = ("fraud", "scam", "mule")
_lock = threading.Lock()
_cache: Dict[str, Any] = {}


class RegistryError(RuntimeError):
    pass


def registry_dir(family: str = "fraud") -> Path:
    if family not in FAMILIES:
        raise RegistryError(f"family must be one of {FAMILIES}")
    override = os.environ.get("BTI_MODEL_REGISTRY_DIR")
    base = Path(override) if override else Path(get_settings().models_dir) / "registry"
    return base if family == "fraud" else base.parent / f"{base.name}_{family}"


def _index_path(family: str = "fraud") -> Path:
    return registry_dir(family) / "index.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def read_index(family: str = "fraud") -> Dict:
    path = _index_path(family)
    if not path.exists():
        return {"champion": None, "challenger": None, "models": [], "history": []}
    return json.loads(path.read_text())


def save_model(model_id: str, artifact: Dict, card: Dict, family: str = "fraud") -> Path:
    folder = registry_dir(family) / model_id
    if folder.exists():
        raise RegistryError(f"Model {model_id} already registered; registry entries are immutable")
    folder.mkdir(parents=True)
    joblib.dump(artifact, folder / "model.joblib", compress=3)
    _atomic_write_json(folder / "card.json", card)
    with _lock:
        index = read_index(family)
        index["models"].append({
            "model_id": model_id,
            "registered_at": _now(),
            "developer": card.get("ownership", {}).get("developer"),
            "validation_status": card.get("validation", {}).get("status"),
        })
        _atomic_write_json(_index_path(family), index)
    return folder


def load_card(model_id: str, family: str = "fraud") -> Dict:
    path = registry_dir(family) / model_id / "card.json"
    if not path.exists():
        raise RegistryError(f"Unknown model {model_id}")
    return json.loads(path.read_text())


def load_artifact(model_id: str, family: str = "fraud") -> Dict:
    key = model_id if family == "fraud" else f"{family}:{model_id}"
    with _lock:
        if key not in _cache:
            path = registry_dir(family) / model_id / "model.joblib"
            if not path.exists():
                raise RegistryError(f"Unknown model {model_id}")
            _cache[key] = joblib.load(path)
        return _cache[key]


def model_for_role(role: str, family: str = "fraud") -> Optional[str]:
    if role not in ROLES:
        raise RegistryError(f"Role must be one of {ROLES}")
    return read_index(family).get(role)


def check_four_eyes(model_id: str, approver: str, family: str = "fraud") -> None:
    developer = load_card(model_id, family).get("ownership", {}).get("developer")
    if developer and approver and approver.strip().lower() == str(developer).strip().lower():
        raise RegistryError("Four-eyes principle: the approver must differ from the model developer")


def assign_role(model_id: str, role: str, approver: str, rationale: str, family: str = "fraud") -> Dict:
    """Assign a model to a role. Champion promotion requires passed validation and four-eyes."""
    if role not in ROLES:
        raise RegistryError(f"Role must be one of {ROLES}")
    if not approver or not approver.strip():
        raise RegistryError("An approver is required")
    if not rationale or len(rationale.strip()) < 10:
        raise RegistryError("A rationale of at least 10 characters is required for the audit trail")
    card = load_card(model_id, family)
    developer = card.get("ownership", {}).get("developer")
    status = card.get("validation", {}).get("status")
    if role == "champion":
        if status != "passed":
            raise RegistryError(f"Model {model_id} validation status is '{status}'; only 'passed' models "
                                "can become champion")
        if developer and approver.strip().lower() == str(developer).strip().lower():
            raise RegistryError("Four-eyes principle: the approver must differ from the model developer")

    with _lock:
        index = read_index(family)
        if model_id not in {m["model_id"] for m in index["models"]}:
            raise RegistryError(f"Model {model_id} is not in the registry index")
        previous = index.get(role)
        index[role] = model_id
        if role == "champion" and index.get("challenger") == model_id:
            index["challenger"] = None
        event = {"at": _now(), "role": role, "model_id": model_id, "previous": previous,
                 "approver": approver.strip(), "rationale": rationale.strip()}
        index["history"].append(event)
        _atomic_write_json(_index_path(family), index)
    return event


def add_note(model_id: str, author: str, subject: str, detail: str, data: Optional[Dict] = None,
             family: str = "fraud") -> Dict:
    """Append a post-registration note to a model's record. Notes are never edited or removed."""
    if not author or not author.strip():
        raise RegistryError("A note needs an author")
    if not subject or not detail or len(detail.strip()) < 10:
        raise RegistryError("A note needs a subject and a detail of at least 10 characters")
    with _lock:
        index = read_index(family)
        if model_id not in {m["model_id"] for m in index["models"]}:
            raise RegistryError(f"Model {model_id} is not in the registry index")
        note = {"at": _now(), "model_id": model_id, "author": author.strip(), "subject": subject.strip(),
                "detail": detail.strip(), "data": data or {}}
        index.setdefault("notes", []).append(note)
        _atomic_write_json(_index_path(family), index)
    return note


def notes_for(model_id: str, family: str = "fraud") -> List[Dict]:
    return [n for n in read_index(family).get("notes", []) if n["model_id"] == model_id]


def clear_cache() -> None:
    with _lock:
        _cache.clear()
