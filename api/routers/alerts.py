"""
/api/v1/alerts — exception queue management for fraud analysts.
"""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from bti.database import get_db, FraudAlert, Transaction
from bti.logging_config import get_logger, AuditLogger
from bti.config import get_settings
from api.schemas import AlertResponse, AlertUpdateRequest

router = APIRouter(prefix="/alerts", tags=["Fraud Alerts"])
log = get_logger("api.alerts")
settings = get_settings()


def _get_audit() -> AuditLogger:
    return AuditLogger(settings.audit_log_path)


@router.get("/", response_model=list[AlertResponse])
def list_alerts(
    status: Optional[str] = Query(default=None, description="OPEN|REVIEWING|CLOSED|ESCALATED"),
    tier: Optional[str] = Query(default=None, description="CRITICAL|VERY HIGH|HIGH"),
    assigned_to: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    db: Session = Depends(get_db),
):
    query = db.query(FraudAlert)
    filters = []
    if status:
        filters.append(FraudAlert.status == status)
    if tier:
        filters.append(FraudAlert.alert_tier == tier)
    if assigned_to:
        filters.append(FraudAlert.assigned_to == assigned_to)
    if filters:
        query = query.filter(and_(*filters))
    return query.order_by(desc(FraudAlert.final_risk_score)).limit(limit).all()


@router.patch("/{alert_id}", response_model=AlertResponse)
def update_alert(
    alert_id: int,
    body: AlertUpdateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    alert = db.query(FraudAlert).filter(FraudAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    if body.status:
        alert.status = body.status
    if body.assigned_to:
        alert.assigned_to = body.assigned_to
    if body.resolution:
        alert.resolution = body.resolution
    if body.notes:
        alert.notes = body.notes
    if body.status in ("CLOSED", "ESCALATED"):
        alert.resolved_at = datetime.utcnow()

    db.commit()
    db.refresh(alert)

    background_tasks.add_task(
        _get_audit().record,
        "ALERT_UPDATED", alert.transaction_id,
        {"alert_id": alert_id, "new_status": body.status, "resolution": body.resolution}
    )
    log.info("Alert updated", extra={"alert_id": alert_id, "status": body.status})
    return alert


@router.post("/refresh", summary="Re-populate exception queue from scored transactions")
def refresh_exception_queue(db: Session = Depends(get_db)):
    """
    Scans the transactions table and creates FraudAlert records for any
    CRITICAL/HIGH-risk transactions that don't already have an open alert.
    Run after each pipeline execution.
    """
    critical_tiers = ("CRITICAL", "VERY HIGH")
    # Exclude transactions that already have any alert (open OR previously resolved)
    # so that closing/escalating an alert does not re-open it on the next refresh
    existing_ids = {row[0] for row in db.query(FraudAlert.transaction_id).all()}

    new_alerts = (db.query(Transaction)
                  .filter(Transaction.final_alert_tier.in_(critical_tiers))
                  .filter(~Transaction.transaction_id.in_(existing_ids))
                  .all())

    created = 0
    for txn in new_alerts:
        alert = FraudAlert(
            transaction_id=txn.transaction_id,
            alert_tier=txn.final_alert_tier,
            final_risk_score=txn.final_risk_score or 0,
            fraud_rule_score=txn.fraud_rule_score,
            ml_anomaly_score_norm=txn.ml_anomaly_score_norm,
            rules_triggered=txn.rules_triggered,
            status="OPEN",
        )
        db.add(alert)
        created += 1

    db.commit()
    log.info("Exception queue refreshed", extra={"new_alerts": created})
    return {"created": created, "message": f"Created {created} new alerts"}
