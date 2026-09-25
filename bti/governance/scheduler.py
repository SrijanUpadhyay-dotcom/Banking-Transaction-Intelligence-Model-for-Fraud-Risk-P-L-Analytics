"""
In-process monitoring scheduler: weekly drift check, weekly parallel-run report
against the incumbent, and daily reconciliation of the two decision logs.

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
PARALLEL_JOB_ID = "weekly_parallel_run_report"
RECONCILE_JOB_ID = "daily_reconciliation"
_scheduler: Optional[BackgroundScheduler] = None


def _run(name: str, fn) -> None:
    from bti.database.connection import SessionLocal
    db = SessionLocal()
    try:
        fn(db)
    except Exception:
        log.exception(f"Scheduled {name} failed")
    finally:
        db.close()


def _drift_job() -> None:
    from bti.governance.drift_job import run_drift_check
    _run("drift check", run_drift_check)


def _parallel_job() -> None:
    from bti.parallel.report import run_weekly_report
    _run("parallel-run report", run_weekly_report)


def _reconcile_job() -> None:
    from bti.parallel.reconcile import reconcile
    _run("reconciliation", reconcile)


def start() -> Optional[BackgroundScheduler]:
    global _scheduler
    settings = get_settings()
    if not settings.monitoring_scheduler_enabled or _scheduler is not None:
        return _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC", daemon=True)
    for fn, cron, job_id in ((_drift_job, settings.drift_check_cron, JOB_ID),
                             (_parallel_job, settings.parallel_report_cron, PARALLEL_JOB_ID),
                             (_reconcile_job, settings.reconciliation_cron, RECONCILE_JOB_ID)):
        _scheduler.add_job(fn, CronTrigger.from_crontab(cron, timezone="UTC"), id=job_id, max_instances=1,
                           coalesce=True, misfire_grace_time=3600)
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
    def next_run(job_id):
        job = _scheduler.get_job(job_id) if _scheduler else None
        return job.next_run_time.isoformat() if job and job.next_run_time else None
    return {
        "enabled": settings.monitoring_scheduler_enabled,
        "running": _scheduler is not None,
        "cron_utc": settings.drift_check_cron,
        "window_days": settings.drift_window_days,
        "next_run": next_run(JOB_ID),
        "jobs": {
            JOB_ID: {"cron_utc": settings.drift_check_cron, "next_run": next_run(JOB_ID)},
            PARALLEL_JOB_ID: {"cron_utc": settings.parallel_report_cron, "next_run": next_run(PARALLEL_JOB_ID)},
            RECONCILE_JOB_ID: {"cron_utc": settings.reconciliation_cron, "next_run": next_run(RECONCILE_JOB_ID)},
        },
    }
