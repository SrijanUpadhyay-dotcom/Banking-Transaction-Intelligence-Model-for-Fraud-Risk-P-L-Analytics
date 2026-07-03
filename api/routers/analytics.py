"""
/api/v1/analytics — P&L KPIs, monthly trends, customer risk profiles.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text

from bti.database import get_db, Transaction, PnLSummary
from api.schemas import PnLKPIs, PnLPeriod, CustomerRiskProfile

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/pnl/kpis", response_model=PnLKPIs)
def get_pnl_kpis(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    customer_segment: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Aggregate KPIs across the full dataset or a filtered window."""
    query = db.query(Transaction)
    if date_from:
        query = query.filter(Transaction.transaction_date >= date_from)
    if date_to:
        query = query.filter(Transaction.transaction_date <= date_to)
    if customer_segment:
        query = query.filter(Transaction.customer_segment == customer_segment)

    total = query.count()
    if total == 0:
        raise HTTPException(status_code=404, detail="No transactions match the filter")

    agg = query.with_entities(
        func.sum(Transaction.transaction_amount).label("total_volume"),
        func.sum(Transaction.net_revenue).label("net_revenue"),
        func.sum(Transaction.net_pnl_impact).label("net_pnl"),
        func.sum(Transaction.fraud_flag).label("fraud_count"),
        func.sum(Transaction.fraud_loss).label("fraud_loss"),
        func.sum(Transaction.chargeback_flag).label("chargeback_count"),
        func.sum(Transaction.chargeback_loss).label("chargeback_loss"),
        func.sum(Transaction.processing_cost).label("processing_cost"),
        func.sum(Transaction.fee_income + Transaction.interchange_income).label("gross_income"),
        func.sum(Transaction.is_revenue_leakage).label("leakage_count"),
    ).one()

    fraud_loss_rate = (agg.fraud_loss / agg.total_volume * 100) if agg.total_volume else 0
    cb_ratio = (agg.chargeback_count / total * 100) if total else 0
    cti = (agg.processing_cost / agg.gross_income * 100) if agg.gross_income else 0
    risk_adj_rev = agg.net_revenue - (agg.fraud_loss + agg.chargeback_loss)

    return PnLKPIs(
        total_transactions=total,
        total_volume=round(agg.total_volume or 0, 2),
        net_revenue=round(agg.net_revenue or 0, 2),
        net_pnl_impact=round(agg.net_pnl or 0, 2),
        fraud_count=int(agg.fraud_count or 0),
        fraud_loss_rate=round(fraud_loss_rate, 4),
        chargeback_ratio=round(cb_ratio, 4),
        cost_to_income=round(cti, 4),
        risk_adj_revenue=round(risk_adj_rev or 0, 2),
        revenue_leakage_count=int(agg.leakage_count or 0),
    )


@router.get("/pnl/monthly", response_model=List[PnLPeriod])
def get_monthly_pnl(
    limit: int = Query(default=24, le=60),
    db: Session = Depends(get_db),
):
    """Returns monthly P&L trends, newest first."""
    rows = (db.query(PnLSummary)
            .order_by(desc(PnLSummary.period_month))
            .limit(limit).all())
    if not rows:
        # Fall back to live aggregation if materialised table is empty
        rows = _live_monthly_agg(db, limit)
    return rows


def _live_monthly_agg(db: Session, limit: int) -> list:
    """Aggregate directly from transactions when pnl_summary is not populated."""
    result = db.execute(text("""
        SELECT
            month_year                         AS period_month,
            COUNT(*)                            AS total_transactions,
            ROUND(SUM(transaction_amount), 2)   AS total_volume,
            ROUND(SUM(net_revenue), 2)          AS net_revenue,
            ROUND(SUM(net_pnl_impact), 2)       AS net_pnl_impact,
            CAST(SUM(fraud_flag) AS INT)        AS fraud_count,
            ROUND(SUM(fraud_loss) / NULLIF(SUM(transaction_amount), 0) * 100, 4) AS fraud_loss_rate,
            ROUND(SUM(chargeback_flag) * 1.0 / COUNT(*) * 100, 4)  AS chargeback_ratio,
            ROUND(SUM(processing_cost) / NULLIF(SUM(fee_income + interchange_income), 0) * 100, 4) AS cost_to_income
        FROM transactions
        WHERE month_year IS NOT NULL
        GROUP BY month_year
        ORDER BY month_year DESC
        LIMIT :lim
    """), {"lim": limit})
    return [PnLPeriod(**dict(row._mapping)) for row in result]


@router.get("/risk/customer/{customer_id}", response_model=CustomerRiskProfile)
def get_customer_risk(customer_id: str, db: Session = Depends(get_db)):
    rows = db.query(Transaction).filter(Transaction.customer_id == customer_id).all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    scores = [r.final_risk_score for r in rows if r.final_risk_score is not None]
    return CustomerRiskProfile(
        customer_id=customer_id,
        avg_final_risk_score=round(sum(scores) / len(scores), 2) if scores else 0,
        max_final_risk_score=round(max(scores), 2) if scores else 0,
        total_transactions=len(rows),
        fraud_count=sum(r.fraud_flag or 0 for r in rows),
        suspicious_count=sum(r.is_suspicious or 0 for r in rows),
        total_volume=round(sum(r.transaction_amount or 0 for r in rows), 2),
        avg_amount=round(sum(r.transaction_amount or 0 for r in rows) / len(rows), 2),
        primary_segment=max(set(r.customer_segment for r in rows if r.customer_segment),
                            key=lambda s: sum(1 for r in rows if r.customer_segment == s),
                            default=None),
        primary_channel=max(set(r.channel for r in rows if r.channel),
                            key=lambda c: sum(1 for r in rows if r.channel == c),
                            default=None),
        chargeback_count=sum(r.chargeback_flag or 0 for r in rows),
    )


@router.get("/risk/top-customers", response_model=List[CustomerRiskProfile])
def get_top_risk_customers(limit: int = Query(default=20, le=100), db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT
            customer_id,
            ROUND(AVG(final_risk_score), 2)        AS avg_final_risk_score,
            ROUND(MAX(final_risk_score), 2)        AS max_final_risk_score,
            COUNT(*)                               AS total_transactions,
            SUM(fraud_flag)                        AS fraud_count,
            SUM(is_suspicious)                     AS suspicious_count,
            ROUND(SUM(transaction_amount), 2)      AS total_volume,
            ROUND(AVG(transaction_amount), 2)      AS avg_amount,
            MAX(customer_segment)                  AS primary_segment,
            MAX(channel)                           AS primary_channel,
            SUM(chargeback_flag)                   AS chargeback_count
        FROM transactions
        GROUP BY customer_id
        ORDER BY avg_final_risk_score DESC
        LIMIT :lim
    """), {"lim": limit})
    return [CustomerRiskProfile(**dict(row._mapping)) for row in result]
