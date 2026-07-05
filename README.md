# Banking Transaction Intelligence Model
### Fraud Risk & P&L Analytics — Production-Grade System v2.0

A fully operational fraud detection and P&L analytics platform built to production standards. Accepts real-time transaction scoring requests via REST API, applies a 19-rule fraud engine plus an ML ensemble, and returns a risk assessment in under 100ms. Backs an exception queue for fraud analysts and live P&L KPI dashboards.

Built to demonstrate what a real bank fraud operations platform looks like end-to-end — not a notebook, not a prototype.

---

## What It Does

```
Transaction arrives  →  19-rule engine  →  ML ensemble  →  Composite score  →  BLOCK / HOLD / REVIEW / ALLOW
                                           (ISO + LR + RF)      (0–100)
                                                                      ↓
                                                           Exception queue (fraud analysts)
                                                           P&L KPI dashboard
                                                           Audit trail (JSONL)
```

**Fraud detection:** 19 weighted rules (velocity, geography, device, merchant category, refund ratio, chargeback ratio, off-hours, duplicate detection) combined with an Isolation Forest + Logistic Regression + Random Forest ensemble. Final score is a weighted composite of the input risk score, rule engine score, and ML anomaly score.

**P&L analytics:** Every transaction is tagged with fee income, interchange income, processing cost, fraud loss, chargeback loss, and net P&L impact. Monthly variance, channel profitability, merchant risk ranking, and segment P&L are computed and served via the analytics API.

**Exception queue:** CRITICAL and VERY HIGH transactions automatically populate a fraud analyst queue. Analysts can claim, review, escalate, and close alerts via the API.

---

## Model Performance

| Model | ROC-AUC | F1 Score |
|---|---|---|
| Logistic Regression | **1.000** | — |
| Random Forest | **0.9999** | **0.999** |
| Isolation Forest | Unsupervised anomaly detection | — |

28 features across numeric, categorical (saved label encoders), and engineered ratio columns. Trained on 50,000 synthetic transactions spanning 24 months across 8 customer segments, 15 merchant categories, and 7 channels.

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/srijanupadhyay-dotcom/banking-transaction-intelligence-model-for-fraud-risk-p-l-analytics
cd banking-transaction-intelligence-model-for-fraud-risk-p-l-analytics

docker compose up -d

# API live at http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

### Local (Python 3.11+)

```bash
pip install -r requirements.txt

# Initialise DB and seed 50k transactions
python main.py db-init --seed

# Start API server
python main.py api
```

```bash
# Or run the full pipeline end-to-end
# (generates data → trains models → seeds DB)
python main.py pipeline --force
```

### Environment Variables

```bash
BTI_SECRET_KEY=your-32-char-secret-key    # required in production
BTI_API_KEY=your-pipeline-api-key         # guards POST /pipeline/run
BTI_DATABASE_URL=postgresql://...         # optional; defaults to SQLite
```

---

## API Reference

Interactive docs at `/docs` (Swagger) and `/redoc`.

### Health

```
GET /health
```
```json
{ "status": "ok", "version": "2.0.0", "database": "ok", "uptime_seconds": 142.3 }
```

### Real-Time Fraud Scoring

```
POST /api/v1/score/
```

Submit a transaction and receive a full fraud risk assessment in **<100ms**.

**Request:**
```json
{
  "transaction_id": "TXN-20240715-00001",
  "customer_id": "CUST-4821",
  "transaction_date": "2024-07-15",
  "transaction_time": "02:47:00",
  "transaction_amount": 12500.00,
  "channel": "API/Open Banking",
  "merchant_category": "Crypto Exchanges",
  "customer_segment": "Mass Market",
  "risk_score": 72,
  "historical_average_transaction_amount": 450.0,
  "failed_attempt_count": 4,
  "login_attempts": 6,
  "account_balance_after": 500.00,
  "debit_credit_flag": "Debit",
  "device_id": "DEV-X9921-UNKNOWN",
  "ip_location": "203.0.113.45"
}
```

**Response:**
```json
{
  "transaction_id": "TXN-20240715-00001",
  "scored_at": "2024-07-15T02:47:01.234Z",
  "fraud_rule_score": 58.21,
  "rules_triggered": 8,
  "rules_fired": ["R01_high_value_vs_avg", "R03_failed_auth", "R05_off_hours", "R17_cross_border"],
  "ml_lr_proba": 0.9664,
  "ml_rf_proba": 0.5100,
  "ml_anomaly_score": 71.4,
  "final_risk_score": 62.92,
  "final_alert_tier": "HIGH",
  "is_suspicious": true,
  "processing_time_ms": 65.5,
  "model_version": "2026-07-03T11:50:52Z",
  "recommendation": "REVIEW — Flag for analyst review within 1 business hour"
}
```

### Alert Tiers & Recommendations

| Score | Tier | Action |
|---|---|---|
| ≥ 85 | CRITICAL | BLOCK — Immediately decline and escalate |
| ≥ 70 | VERY HIGH | HOLD — Step-up authentication required |
| ≥ 50 | HIGH | REVIEW — Analyst review within 1 hour |
| ≥ 25 | MEDIUM | MONITOR — Allow with enhanced monitoring |
| < 25 | LOW | ALLOW — Within normal risk parameters |

### Transactions

```
GET /api/v1/transactions/                        # paginated, filterable
GET /api/v1/transactions/{id}                    # single transaction detail
GET /api/v1/transactions/customer/{customer_id}  # customer history
```

Filters: `channel`, `customer_segment`, `fraud_only`, `suspicious_only`, `alert_tier`, `risk_score_min/max`, `date_from/date_to`, `page`, `page_size`.

### Fraud Alerts (Exception Queue)

```
GET   /api/v1/alerts/         # list alerts (filter: status, tier, assigned_to)
PATCH /api/v1/alerts/{id}     # update: OPEN → REVIEWING → CLOSED / ESCALATED
POST  /api/v1/alerts/refresh  # populate queue from newly scored transactions
```

### P&L Analytics

```
GET /api/v1/analytics/pnl/kpis                   # live aggregate KPIs
GET /api/v1/analytics/pnl/monthly                # month-over-month trend
GET /api/v1/analytics/risk/customer/{id}         # customer risk profile
GET /api/v1/analytics/risk/top-customers?limit=N # highest-risk customers
```

### Pipeline & Models

```
POST /api/v1/pipeline/run         # trigger async pipeline (requires X-API-Key)
GET  /api/v1/pipeline/status      # last run result
GET  /api/v1/score/model-info     # model version + metrics
POST /api/v1/score/reload-models  # hot-reload models after retraining
```

---

## Fraud Rules (19 rules)

| Rule | Description | Weight |
|---|---|---|
| R01 | Amount ≥ 5× customer historical average | 9 |
| R02 | Monthly velocity spike (> 10 transactions) | 8 |
| R03 | Failed authentication count ≥ 3 | 7 |
| R04 | Multiple login attempts ≥ 4 | 6 |
| R05 | Off-hours transaction (00:00–06:00 or 22:00–23:59) | 5 |
| R06 | Non-private IP with risk score > 40 | 7 |
| R07 | Repeated merchant ≥ 5 times same customer | 6 |
| R08 | Abnormal refund in high-risk merchant category | 8 |
| R09 | Chargeback in high-chargeback category | 8 |
| R10 | Amount z-score outlier within customer segment | 9 |
| R11 | Debit leaving negative balance | 9 |
| R12 | High risk score + unknown device prefix | 8 |
| R13 | High-risk segment (Student / NRI) + amount > £5,000 | 7 |
| R14 | Risky channel (USSD / API / Call Centre) + high value | 6 |
| R15 | Duplicate: same customer + amount + merchant ≥ 2× | 9 |
| R16 | Daily rapid sequential: > 5 transactions same day | 6 |
| R17 | Cross-border proxy: high-CB category + risk > 50 + amount > £1k | 8 |
| R18 | Merchant refund ratio > 30% | 7 |
| R19 | Customer chargeback ratio > 10% | 8 |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI (port 8000)                    │
│  /score  /transactions  /alerts  /analytics  /pipeline   │
└─────────────────────┬────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
   ┌──────▼──────┐        ┌───────▼───────┐
   │ Rule Engine  │        │  ML Ensemble  │
   │  19 rules    │        │  ISO + LR + RF│
   │  weighted    │        │  28 features  │
   └──────┬──────┘        └───────┬───────┘
          │                       │
          └───────────┬───────────┘
                      │
             ┌────────▼────────┐
             │ Composite Score  │
             │  risk   × 0.35   │
             │  rules  × 0.35   │
             │  ml     × 0.30   │
             └────────┬────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
   ┌──────▼──────┐        ┌───────▼───────┐
   │  SQLite /    │        │  Alert Queue  │
   │  PostgreSQL  │        │  (analysts)   │
   └─────────────┘        └───────────────┘

Pipeline (APScheduler, daily 02:00 UTC):
  01_data_generation → 02_cleaning → 03_fraud_rules
  → 04_ml_detection → 05_pnl_analytics → DB seed → alert dispatch
```

---

## Project Structure

```
├── api/
│   ├── main.py                    # FastAPI app, middleware, health
│   ├── schemas.py                 # Pydantic request/response schemas
│   └── routers/
│       ├── transactions.py        # paginated transaction listing
│       ├── alerts.py              # exception queue CRUD
│       ├── analytics.py           # P&L KPIs, customer risk profiles
│       ├── pipeline.py            # pipeline trigger + status
│       └── scoring.py             # POST /score/ real-time endpoint
├── bti/
│   ├── config.py                  # pydantic-settings, YAML + env override
│   ├── logging_config.py          # structured JSON logging + AuditLogger
│   ├── alerts.py                  # HMAC-signed webhook + email dispatch
│   ├── database/
│   │   ├── models.py              # SQLAlchemy ORM (5 tables)
│   │   ├── connection.py          # engine factory (SQLite / PostgreSQL)
│   │   └── init_db.py             # create_tables(), seed_from_csv()
│   ├── scoring/
│   │   ├── model_loader.py        # thread-safe model bundle cache
│   │   └── realtime.py            # 19-rule engine + ML ensemble
│   └── pipeline/
│       └── orchestrator.py        # 5-stage pipeline orchestration
├── src/
│   ├── 01_data_generation.py      # 50k synthetic transactions (24 months)
│   ├── 02_data_cleaning.py        # normalisation, outlier handling
│   ├── 03_fraud_rules_engine.py   # 19-rule batch scoring
│   ├── 04_ml_anomaly_detection.py # model training + artifact save
│   └── 05_pnl_analytics.py        # P&L KPIs and output CSVs
├── models/                        # trained joblib artifacts + manifest.json
├── scheduler/
│   └── pipeline_scheduler.py      # APScheduler cron jobs
├── tests/
│   ├── unit/                      # 54 unit tests
│   └── integration/               # 16 integration tests (TestClient)
├── config/settings.yaml           # all thresholds and paths
├── Dockerfile
├── docker-compose.yml             # api + scheduler + postgres (optional)
├── main.py                        # CLI entry point
└── .github/workflows/ci.yml       # unit → integration → Docker build CI
```

---

## Running Tests

```bash
# Full suite (70 tests, ~6 seconds)
python -m pytest tests/ -v

# Unit only
python -m pytest tests/unit/ -v

# Integration only (uses in-memory SQLite, no side effects)
python -m pytest tests/integration/ -v
```

---

## Configuration

All tunable parameters in `config/settings.yaml`. Override at deploy time via `BTI_` prefixed environment variables:

```bash
BTI_DATABASE_URL=postgresql://user:pass@host:5432/bti
BTI_ENVIRONMENT=production
BTI_LOG_LEVEL=INFO
BTI_VELOCITY_SPIKE_THRESHOLD=10
BTI_HIGH_VALUE_RATIO=5.0
BTI_FAILED_AUTH_THRESHOLD=3
```

---

## Security

- Pipeline API protected by `X-API-Key` header with `hmac.compare_digest` (timing-safe)
- Webhook alert dispatch signed with HMAC-SHA256 (`X-BTI-Signature` header), 3× retry with exponential backoff
- Append-only JSONL audit trail for all scoring decisions and alert state changes (AML/BSA/FCA compliance pattern)
- Non-root Docker user (`bti:bti`)
- PostgreSQL-ready for production deployment

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.111, Uvicorn, Pydantic v2 |
| ML | scikit-learn — Isolation Forest, Logistic Regression, Random Forest |
| Database | SQLAlchemy 2.0 ORM, SQLite (dev) / PostgreSQL (prod) |
| Config | pydantic-settings, YAML + environment variable override |
| Scheduler | APScheduler 3.x (cron-based, 3 jobs) |
| Logging | Structured JSON to stdout, append-only JSONL audit trail |
| Testing | pytest, FastAPI TestClient, StaticPool in-memory SQLite |
| Infrastructure | Docker, docker-compose, GitHub Actions CI/CD |

---

## P&L Output Files

After a pipeline run, `outputs/pnl/` contains:

| File | Contents |
|---|---|
| `monthly_variance.csv` | Month-over-month P&L, fraud loss, revenue trend |
| `channel_profitability.csv` | Net P&L and fraud rate by channel |
| `merchant_risk.csv` | Fraud rate, chargeback rate, risk rank per merchant |
| `segment_profitability.csv` | P&L breakdown by customer segment |
| `exception_queue.csv` | Highest-risk transactions for analyst review |

---

*Built to demonstrate production-grade financial crime infrastructure — REST API, ML scoring pipeline, compliance audit logging, and full test coverage at the standard expected in Tier 1 banking.*
