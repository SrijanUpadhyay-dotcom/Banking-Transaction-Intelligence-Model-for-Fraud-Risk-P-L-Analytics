"""
Banking Transaction Intelligence — FastAPI Application Entry Point

Endpoints:
  GET  /health                     — liveness probe
  GET  /api/v1/transactions/       — paginated transaction listing
  GET  /api/v1/transactions/{id}   — single transaction
  GET  /api/v1/transactions/customer/{id}
  GET  /api/v1/alerts/             — exception queue
  PATCH /api/v1/alerts/{id}        — update alert status
  POST /api/v1/alerts/refresh      — populate exception queue
  GET  /api/v1/analytics/pnl/kpis
  GET  /api/v1/analytics/pnl/monthly
  GET  /api/v1/analytics/risk/customer/{id}
  GET  /api/v1/analytics/risk/top-customers
  POST /api/v1/pipeline/run        — async pipeline trigger (requires X-API-Key)
  POST /api/v1/pipeline/run/sync   — sync pipeline trigger (dev)
  GET  /api/v1/pipeline/status
  POST /api/v1/score/              — real-time single-transaction scoring (<100ms)
  GET  /api/v1/score/model-info    — current model version and metrics
  POST /api/v1/v3/score            — leak-free v3 score + expected-cost decision + reason codes
  GET  /api/v1/governance/models   — model inventory, cards, documentation, promotion
  GET  /api/v1/operations/kpis     — fraud-ops KPIs, labels feedback, champion/challenger
  POST /api/v1/score/reload-models — reload models after retraining

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bti.config import get_settings
from bti.logging_config import setup_logging, get_logger
from bti.database.init_db import create_tables
from bti.governance import scheduler as monitoring_scheduler
from api import metrics
from api.routers import (
    transactions, alerts, analytics, pipeline, scoring, graph, copilot, sas, v3, governance, operations,
    parallel,
)

settings = get_settings()
log = setup_logging(log_level=settings.log_level, log_format="json")

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("BTI API starting up", extra={"version": settings.app_version,
                                            "env": settings.environment})
    create_tables()
    monitoring_scheduler.start()
    yield
    monitoring_scheduler.stop()
    log.info("BTI API shutting down")


app = FastAPI(
    title="Banking Transaction Intelligence API",
    description=(
        "Production REST API for real-time fraud detection, risk scoring, "
        "and P&L analytics. Powers the BTI dashboard and analyst exception queue."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception:
        metrics.record(request.url.path, 500, round((time.time() - t0) * 1000, 1))
        raise
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    metrics.record(request.url.path, response.status_code, elapsed_ms)
    log.info("HTTP request", extra={
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "ms": elapsed_ms,
    })
    return response


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    from bti.database.connection import engine
    try:
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "unreachable"

    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.environment,
        "database": db_status,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


# ── Routers ────────────────────────────────────────────────────────────────────

API_PREFIX = "/api/v1"
app.include_router(transactions.router, prefix=API_PREFIX)
app.include_router(alerts.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)
app.include_router(pipeline.router, prefix=API_PREFIX)
app.include_router(scoring.router, prefix=API_PREFIX)
app.include_router(graph.router,   prefix=API_PREFIX)
app.include_router(copilot.router, prefix=API_PREFIX)
app.include_router(sas.router,    prefix=API_PREFIX)
app.include_router(v3.router,     prefix=API_PREFIX)
app.include_router(governance.router, prefix=API_PREFIX)
app.include_router(operations.router, prefix=API_PREFIX)
app.include_router(parallel.router, prefix=API_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=(settings.environment == "development"),
        log_level=settings.log_level.lower(),
    )
