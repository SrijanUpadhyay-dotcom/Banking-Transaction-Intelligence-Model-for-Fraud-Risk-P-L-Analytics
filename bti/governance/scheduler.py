"""
In-process monitoring scheduler (weekly drift check by default).

With several API workers, enable it on exactly one of them
(BTI_MONITORING_SCHEDULER_ENABLED=false elsewhere), or run the job from an
external scheduler via POST /api/v1/governance/drift/run.
"""

from __future__ import annotations

from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from bti.config import get_settings
from bti.logging_config import get_logger

log = get_logger("governance.scheduler")

JOB_ID = "weekly_drift_check"
_scheduler: Optional[BackgroundScheduler] = None


def _drift_job() -> None:
    from bti.database.connection import SessionLocal
    from bti.governance.drift_job import run_drift_check
    db = SessionLocal()
    try:
        run_drift_check(db)
    except Exception:
        log.exception("Scheduled drift check failed")
    finally:
        db.close()


def start() -> Optional[BackgroundScheduler]:
    global _scheduler
    settings = get_settings()
    if not settings.monitoring_scheduler_enabled or _scheduler is not None:
        return _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC", daemon=True)
    _scheduler.add_job(_drift_job, CronTrigger.from_crontab(settings.drift_check_cron, timezone="UTC"),
                       id=JOB_ID, max_instances=1, coalesce=True, misfire_grace_time=3600)
    _scheduler.start()
    log.info("Monitoring scheduler started", extra={"cron": settings.drift_check_cron})
    return _scheduler


def stop() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def status() -> dict:
    settings = get_settings()
    job = _scheduler.get_job(JOB_ID) if _scheduler else None
    return {
        "enabled": settings.monitoring_scheduler_enabled,
        "running": _scheduler is not None,
        "cron_utc": settings.drift_check_cron,
        "window_days": settings.drift_window_days,
        "next_run": job.next_run_time.isoformat() if job and job.next_run_time else None,
    }
