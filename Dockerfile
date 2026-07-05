FROM python:3.11-slim AS base

# Security: run as non-root
RUN addgroup --system bti && adduser --system --ingroup bti bti

WORKDIR /app

# Install system deps first (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=bti:bti . .

# Create runtime directories
RUN mkdir -p data/raw data/processed outputs/charts outputs/pnl models logs \
    && chown -R bti:bti data outputs models logs

USER bti

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
