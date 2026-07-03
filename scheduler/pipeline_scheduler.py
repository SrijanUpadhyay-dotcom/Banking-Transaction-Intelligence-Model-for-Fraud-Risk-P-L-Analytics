"""
APScheduler-based pipeline orchestration.
Runs the full BTI pipeline on a configurable cron schedule.
Also schedules ML model retraining on a weekly cadence.

Start standalone:
    python scheduler/pipeline_scheduler.py

Or import and call start_scheduler() from main.py lifespan if co-locating.
"""

import signal
import sys
import time
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

# Add project root to path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from bti.config import get_settings
from bti.logging_config import setup_logging, get_logger

settings = get_settings()
log = setup_logging(log_level=settings.log_level)
log = get_logger("scheduler")


def run_pipeline_job():
    """Daily ETL + fraud scoring job."""
    log.info("Scheduled pipeline job started")
    from bti.pipeline import run_full_pipeline
    result = run_full_pipeline(force=False)
    if result["status"] == "SUCCESS":
        log.info("Pipeline job completed", extra=result)
    else:
        log.error("Pipeline job failed", extra=result)


def refresh_alert_queue_job():
    """Refresh the exception queue after each pipeline run."""
    log.info("Refreshing exception queue")
    try:
        from bti.database import SessionLocal, Transaction, FraudAlert
        from datetime import datetime
        db = SessionLocal()
        try:
            existing_ids = {row[0] for row in db.query(FraudAlert.transaction_id)
                            .filter(FraudAlert.status == "OPEN").all()}
            new_txns = (db.query(Transaction)
                        .filter(Transaction.final_alert_tier.in_(["CRITICAL", "VERY HIGH"]))
                        .filter(~Transaction.transaction_id.in_(existing_ids))
                        .all())
            for txn in new_txns:
                db.add(FraudAlert(
                    transaction_id=txn.transaction_id,
                    alert_tier=txn.final_alert_tier,
                    final_risk_score=txn.final_risk_score or 0,
                    fraud_rule_score=txn.fraud_rule_score,
                    ml_anomaly_score_norm=txn.ml_anomaly_score_norm,
                    rules_triggered=txn.rules_triggered,
                    status="OPEN",
                ))
            db.commit()
            log.info(f"Alert queue refreshed — {len(new_txns)} new alerts created")
        finally:
            db.close()
    except Exception:
        log.exception("Alert queue refresh failed")


def materialise_pnl_job():
    """Materialise monthly P&L summary table — accelerates API responses."""
    log.info("Materialising P&L summary")
    try:
        from bti.database import SessionLocal, Transaction, PnLSummary
        from sqlalchemy import func, text
        db = SessionLocal()
        try:
            result = db.execute(text("""
                SELECT
                    month_year                         AS period_month,
                    COUNT(*)                            AS total_transactions,
                    ROUND(SUM(transaction_amount), 2)   AS total_volume,
                    ROUND(SUM(fee_income + interchange_income), 2)    AS total_fee_income,
                    ROUND(SUM(interchange_income), 2)   AS total_interchange,
                    ROUND(SUM(processing_cost), 2)      AS total_processing_cost,
                    ROUND(SUM(chargeback_loss), 2)      AS total_chargeback_loss,
                    ROUND(SUM(refund_loss), 2)          AS total_refund_loss,
                    ROUND(SUM(fraud_loss), 2)           AS total_fraud_loss,
                    ROUND(SUM(net_revenue), 2)          AS net_revenue,
                    ROUND(SUM(net_pnl_impact), 2)       AS net_pnl_impact,
                    CAST(SUM(fraud_flag) AS INT)        AS fraud_count,
                    ROUND(SUM(fraud_loss)/NULLIF(SUM(transaction_amount),0)*100, 4) AS fraud_loss_rate,
                    ROUND(SUM(chargeback_flag)*1.0/COUNT(*)*100, 4)  AS chargeback_ratio,
                    ROUND(SUM(processing_cost)/NULLIF(SUM(fee_income+interchange_income),0)*100,4) AS cost_to_income,
                    ROUND(SUM(net_revenue)-SUM(fraud_loss)-SUM(chargeback_loss),2) AS risk_adj_revenue
                FROM transactions
                WHERE month_year IS NOT NULL
                GROUP BY month_year
            """))
            rows = [dict(r._mapping) for r in result]
            if not rows:
                return

            db.query(PnLSummary).delete()
            for row in rows:
                db.add(PnLSummary(**row))
            db.commit()
            log.info(f"P&L summary materialised — {len(rows)} months")
        finally:
            db.close()
    except Exception:
        log.exception("P&L materialisation failed")


def job_listener(event):
    if event.exception:
        log.error(f"Job {event.job_id} failed: {event.exception}")
    else:
        log.info(f"Job {event.job_id} completed successfully")


def start_scheduler():
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_listener(job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    # Daily pipeline at 02:00 UTC
    scheduler.add_job(
        run_pipeline_job,
        trigger=CronTrigger.from_crontab("0 2 * * *"),
        id="daily_pipeline",
        name="Daily ETL + Fraud Scoring",
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )

    # Alert queue refresh at 02:45 UTC (after pipeline completes)
    scheduler.add_job(
        refresh_alert_queue_job,
        trigger=CronTrigger.from_crontab("45 2 * * *"),
        id="alert_refresh",
        name="Exception Queue Refresh",
        max_instances=1,
    )

    # P&L materialisation at 03:00 UTC
    scheduler.add_job(
        materialise_pnl_job,
        trigger=CronTrigger.from_crontab("0 3 * * *"),
        id="pnl_materialise",
        name="Monthly P&L Materialisation",
        max_instances=1,
    )

    def _shutdown(signum, _frame):
        log.info("Scheduler received shutdown signal — stopping")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("Scheduler started — jobs registered",
             extra={"jobs": [j.id for j in scheduler.get_jobs()]})
    scheduler.start()


if __name__ == "__main__":
    start_scheduler()
