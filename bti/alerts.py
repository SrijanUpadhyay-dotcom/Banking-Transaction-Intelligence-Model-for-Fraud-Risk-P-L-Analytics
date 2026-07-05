"""
Alert notification system — sends email and/or webhook on CRITICAL/HIGH alerts.
Designed to integrate with PagerDuty, Slack, or any webhook endpoint.
"""

import hashlib
import hmac
import json
import logging
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

try:
    import httpx
    _httpx_available = True
except ImportError:
    _httpx_available = False

from bti.config import get_settings
from bti.logging_config import get_logger

log = get_logger("alerts")


def _sign_payload(payload_bytes: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


class AlertDispatcher:
    """
    Dispatches fraud alerts via email and/or webhook.
    Instantiate once and call dispatch() per alert batch.
    """

    def __init__(self):
        self.settings = get_settings()

    def dispatch(self, alerts: list[dict]) -> None:
        if not self.settings.alerts_enabled or not alerts:
            return

        critical = [a for a in alerts if a.get("final_alert_tier") == "CRITICAL"]
        high = [a for a in alerts if a.get("final_alert_tier") == "VERY HIGH"]

        if not critical and not high:
            return

        log.warning(
            "Dispatching fraud alerts",
            extra={"critical": len(critical), "high": len(high)}
        )

        if self.settings.webhook_url:
            self._send_webhook(critical + high)

        if self.settings.smtp_user and self.settings.smtp_host:
            self._send_email(critical, high)

    def _send_webhook(self, alerts: list[dict], retries: int = 3) -> None:
        if not _httpx_available:
            log.warning("httpx not installed — webhook skipped")
            return

        payload = {
            "source": "BTI-FraudEngine",
            "alert_count": len(alerts),
            "alerts": [
                {
                    "transaction_id": a.get("transaction_id"),
                    "tier": a.get("final_alert_tier"),
                    "risk_score": a.get("final_risk_score"),
                    "customer_id": a.get("customer_id"),
                    "amount": a.get("transaction_amount"),
                    "channel": a.get("channel"),
                }
                for a in alerts[:50]  # cap payload size
            ],
        }
        payload_bytes = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.settings.webhook_secret:
            headers["X-BTI-Signature"] = _sign_payload(payload_bytes, self.settings.webhook_secret)

        for attempt in range(retries):
            try:
                with httpx.Client(timeout=10) as client:
                    r = client.post(self.settings.webhook_url, content=payload_bytes, headers=headers)
                    r.raise_for_status()
                log.info("Webhook sent", extra={"status": r.status_code, "alerts": len(alerts)})
                return
            except Exception as exc:
                wait = 2 ** attempt
                log.warning(f"Webhook attempt {attempt + 1} failed: {exc}. Retrying in {wait}s")
                if attempt < retries - 1:
                    time.sleep(wait)
        log.error("All webhook attempts failed", extra={"url": self.settings.webhook_url})

    def _send_email(self, critical: list[dict], high: list[dict]) -> None:
        body_lines = [
            "Banking Transaction Intelligence — Fraud Alert Summary",
            "=" * 60,
            f"CRITICAL alerts: {len(critical)}",
            f"VERY HIGH alerts: {len(high)}",
            "",
        ]
        for a in (critical + high)[:20]:
            body_lines.append(
                f"  [{a.get('final_alert_tier')}] TXN {a.get('transaction_id')} | "
                f"Score {a.get('final_risk_score', 0):.1f} | "
                f"${a.get('transaction_amount', 0):,.2f} | "
                f"{a.get('channel', '')} | {a.get('customer_id', '')}"
            )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[BTI ALERT] {len(critical)} CRITICAL + {len(high)} HIGH fraud alerts"
        msg["From"] = self.settings.smtp_user
        msg["To"] = "fraud-team@bank.internal"
        msg.attach(MIMEText("\n".join(body_lines), "plain"))

        try:
            with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port) as srv:
                srv.starttls()
                srv.login(self.settings.smtp_user, self.settings.smtp_password)
                srv.send_message(msg)
            log.info("Alert email sent", extra={"critical": len(critical), "high": len(high)})
        except Exception:
            log.exception("Failed to send alert email")
