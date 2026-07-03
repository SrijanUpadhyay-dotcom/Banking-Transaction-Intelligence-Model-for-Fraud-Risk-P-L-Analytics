"""
/api/v1/transactions endpoints — paginated listing, filtering, detail lookup.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_

from bti.database import get_db, Transaction
from api.schemas import TransactionScored, TransactionPage

router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.get("/", response_model=TransactionPage)
def list_transactions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, le=500),
    channel: Optional[str] = None,
    customer_segment: Optional[str] = None,
    fraud_only: bool = False,
    suspicious_only: bool = False,
    min_risk_score: Optional[float] = None,
    max_risk_score: Optional[float] = None,
    alert_tier: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Transaction)

    filters = []
    if channel:
        filters.append(Transaction.channel == channel)
    if customer_segment:
        filters.append(Transaction.customer_segment == customer_segment)
    if fraud_only:
        filters.append(Transaction.fraud_flag == 1)
    if suspicious_only:
        filters.append(Transaction.is_suspicious == 1)
    if min_risk_score is not None:
        filters.append(Transaction.final_risk_score >= min_risk_score)
    if max_risk_score is not None:
        filters.append(Transaction.final_risk_score <= max_risk_score)
    if alert_tier:
        filters.append(Transaction.final_alert_tier == alert_tier)
    if date_from:
        filters.append(Transaction.transaction_date >= date_from)
    if date_to:
        filters.append(Transaction.transaction_date <= date_to)

    if filters:
        query = query.filter(and_(*filters))

    total = query.count()
    offset = (page - 1) * page_size
    items = (query.order_by(desc(Transaction.final_risk_score))
             .offset(offset).limit(page_size).all())

    return TransactionPage(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{transaction_id}", response_model=TransactionScored)
def get_transaction(transaction_id: str, db: Session = Depends(get_db)):
    txn = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
    return txn


@router.get("/customer/{customer_id}", response_model=TransactionPage)
def get_customer_transactions(
    customer_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, le=200),
    db: Session = Depends(get_db),
):
    query = db.query(Transaction).filter(Transaction.customer_id == customer_id)
    total = query.count()
    offset = (page - 1) * page_size
    items = query.order_by(desc(Transaction.transaction_date)).offset(offset).limit(page_size).all()
    return TransactionPage(items=items, total=total, page=page,
                           page_size=page_size, pages=(total + page_size - 1) // page_size)
