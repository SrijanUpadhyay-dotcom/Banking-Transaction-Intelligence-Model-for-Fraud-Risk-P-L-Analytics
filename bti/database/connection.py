"""
Database connection pool. Supports SQLite (default) and PostgreSQL.
Switch via BTI_DATABASE_URL environment variable — no code changes needed.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from bti.config import get_settings

settings = get_settings()

_is_sqlite = settings.database_url.startswith("sqlite")

# SQLite requires check_same_thread=False; pool_size/max_overflow only valid for non-SQLite
connect_args = {"check_same_thread": False} if _is_sqlite else {}
engine_kwargs = {"echo": settings.db_echo, "connect_args": connect_args}
if not _is_sqlite:
    engine_kwargs["pool_size"] = settings.db_pool_size
    engine_kwargs["max_overflow"] = 10

engine = create_engine(settings.database_url, **engine_kwargs)

# Enable WAL mode for SQLite — allows concurrent reads with writes
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session, closes it after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
