"""
Database initialisation — creates all tables and optionally seeds from CSV.
Run once on first deployment or after schema migrations.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from bti.database.connection import engine, SessionLocal
from bti.database.models import Base, Transaction, PnLSummary
from bti.logging_config import get_logger

log = get_logger("db.init")


def create_tables() -> None:
    log.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    added = add_missing_columns()
    log.info("Tables created successfully", extra={"columns_added": added})


def add_missing_columns(bind=None) -> list:
    """
    Additive migration: create_all does not alter existing tables, so nullable
    columns added to the ORM since the database was created are added here.
    Never drops or changes a column.
    """
    from sqlalchemy import inspect, text
    bind = bind or engine
    inspector = inspect(bind)
    added = []
    with bind.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing or not column.nullable or column.primary_key:
                    continue
                ddl = column.type.compile(dialect=bind.dialect)
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl}'))
                added.append(f"{table.name}.{column.name}")
    return added


def seed_from_csv(csv_path: str, batch_size: int = 5000) -> int:
    """
    Load the processed ML-scored CSV into the transactions table.
    Skips rows whose transaction_id already exists (idempotent).
    Returns count of rows inserted.
    """
    path = Path(csv_path)
    if not path.exists():
        log.warning("Seed CSV not found — skipping", extra={"path": str(path)})
        return 0

    log.info("Seeding transactions from CSV", extra={"path": str(path)})
    df = pd.read_csv(path, low_memory=False)

    # Normalise column names to match ORM model attributes
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename ambiguous column
    if "branch_or_digital_flag" in df.columns:
        df.rename(columns={"branch_or_digital_flag": "branch_or_digital"}, inplace=True)

    # Parse dates
    for col in ("transaction_date", "ingested_at", "updated_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Keep only columns that exist in the ORM
    orm_cols = {c.key for c in Transaction.__table__.columns}
    df = df[[c for c in df.columns if c in orm_cols]].copy()

    db: Session = SessionLocal()
    try:
        existing_ids = {row[0] for row in db.query(Transaction.transaction_id).all()}
        new_rows = df[~df["transaction_id"].isin(existing_ids)]
        log.info(f"Rows to insert: {len(new_rows):,} (skipping {len(existing_ids):,} existing)")

        inserted = 0
        for start in range(0, len(new_rows), batch_size):
            batch = new_rows.iloc[start:start + batch_size]
            records = batch.where(pd.notna(batch), None).to_dict(orient="records")
            db.bulk_insert_mappings(Transaction, records)
            db.commit()
            inserted += len(records)
            log.info(f"Inserted batch {start // batch_size + 1} — {inserted:,} rows so far")

        log.info("Seeding complete", extra={"rows_inserted": inserted})
        return inserted
    except Exception:
        db.rollback()
        log.exception("Seeding failed — rolled back")
        raise
    finally:
        db.close()


def init_database(seed_csv: str | None = None) -> None:
    create_tables()
    if seed_csv:
        seed_from_csv(seed_csv)


if __name__ == "__main__":
    from bti.config import get_settings
    settings = get_settings()
    seed_path = os.path.join(settings.processed_data_dir, "banking_transactions_ml_scored.csv")
    init_database(seed_csv=seed_path if Path(seed_path).exists() else None)
