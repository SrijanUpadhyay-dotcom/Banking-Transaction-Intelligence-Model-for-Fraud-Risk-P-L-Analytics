"""
Structured logging with JSON output for production, human-readable for dev.
Every fraud decision gets an immutable audit trail entry.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines for log aggregation pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields passed via extra={}
        for key, val in record.__dict__.items():
            if key not in ("msg", "args", "levelname", "name", "module", "funcName",
                           "lineno", "exc_info", "exc_text", "stack_info",
                           "created", "msecs", "relativeCreated", "thread",
                           "threadName", "processName", "process", "taskName",
                           "pathname", "filename", "levelno", "message"):
                log_obj[key] = val
        return json.dumps(log_obj)


class AuditLogger:
    """
    Append-only JSONL audit log for all fraud scoring decisions.
    Required for regulatory compliance (AML/BSA/FCA).
    """

    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event_type: str, transaction_id: str, details: dict,
               analyst_id: Optional[str] = None) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "transaction_id": transaction_id,
            "analyst_id": analyst_id,
            **details,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> logging.Logger:
    """Call once at application startup."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

    root.addHandler(handler)

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "bti.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "apscheduler.executors.default"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("bti")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"bti.{name}")
