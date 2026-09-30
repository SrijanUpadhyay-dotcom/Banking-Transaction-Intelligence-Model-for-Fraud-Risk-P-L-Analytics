# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
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
ARCHIVE_JOB_ID = "daily_audit_archive"
OUTCOMES_JOB_ID = "quarterly_outcomes_analysis"
GOVERNANCE_JOB_ID = "weekly_governance_check"
CASE_SLA_JOB_ID = "case_sla_check"
RETRAIN_JOB_ID = "daily_retraining_check"
RECALIBRATION_JOB_ID = "daily_recalibration"
STEPUP_JOB_ID = "stepup_expiry"
VALUES_JOB_ID = "daily_customer_values"
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


def _archive_job() -> None:
    from bti.governance.audit_chain import archive_day
    _run("audit archive", archive_day)


def _outcomes_job() -> None:
    from bti.governance.outcomes import run_quarterly
    _run("outcomes analysis", run_quarterly)


def _governance_job() -> None:
    from bti.governance.validation import run_governance_check
    _run("governance check", run_governance_check)


def _case_sla_job() -> None:
    from bti.operations.cases import check_sla
    _run("case SLA check", check_sla)


def _retrain_job() -> None:
    from bti.modeling.retrain import run_retraining
    _run("retraining check", run_retraining)


def _stepup_job() -> None:
    from bti.operations.stepup import expire
    _run("step-up expiry", expire)


def _values_job() -> None:
    from bti.operations.cost_model import refresh_customer_values
    _run("customer value refresh", refresh_customer_values)


def _recalibration_job() -> None:
    from bti.modeling import registry
    from bti.modeling.recalibration import recalibrate

    def both(db):
        for role in ("champion", "challenger"):
            model_id = registry.model_for_role(role)
            if model_id:
                recalibrate(db, model_id)
    _run("recalibration", both)


def start() -> Optional[BackgroundScheduler]:
    global _scheduler
    settings = get_settings()
    if not settings.monitoring_scheduler_enabled or _scheduler is not None:
        return _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC", daemon=True)
    for fn, cron, job_id in ((_drift_job, settings.drift_check_cron, JOB_ID),
                             (_parallel_job, settings.parallel_report_cron, PARALLEL_JOB_ID),
                             (_reconcile_job, settings.reconciliation_cron, RECONCILE_JOB_ID),
                             (_archive_job, settings.audit_archive_cron, ARCHIVE_JOB_ID),
                             (_outcomes_job, settings.outcomes_cron, OUTCOMES_JOB_ID),
                             (_governance_job, settings.governance_check_cron, GOVERNANCE_JOB_ID),
                             (_case_sla_job, settings.case_sla_check_cron, CASE_SLA_JOB_ID),
                             (_retrain_job, settings.retrain_check_cron, RETRAIN_JOB_ID),
                             (_recalibration_job, settings.recalibration_cron, RECALIBRATION_JOB_ID),
                             (_stepup_job, settings.stepup_expiry_cron, STEPUP_JOB_ID),
                             (_values_job, settings.customer_value_cron, VALUES_JOB_ID)):
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
            ARCHIVE_JOB_ID: {"cron_utc": settings.audit_archive_cron, "next_run": next_run(ARCHIVE_JOB_ID)},
            OUTCOMES_JOB_ID: {"cron_utc": settings.outcomes_cron, "next_run": next_run(OUTCOMES_JOB_ID)},
            GOVERNANCE_JOB_ID: {"cron_utc": settings.governance_check_cron, "next_run": next_run(GOVERNANCE_JOB_ID)},
            CASE_SLA_JOB_ID: {"cron_utc": settings.case_sla_check_cron, "next_run": next_run(CASE_SLA_JOB_ID)},
            RETRAIN_JOB_ID: {"cron_utc": settings.retrain_check_cron, "next_run": next_run(RETRAIN_JOB_ID)},
            RECALIBRATION_JOB_ID: {"cron_utc": settings.recalibration_cron, "next_run": next_run(RECALIBRATION_JOB_ID)},
            STEPUP_JOB_ID: {"cron_utc": settings.stepup_expiry_cron, "next_run": next_run(STEPUP_JOB_ID)},
            VALUES_JOB_ID: {"cron_utc": settings.customer_value_cron, "next_run": next_run(VALUES_JOB_ID)},
        },
    }
