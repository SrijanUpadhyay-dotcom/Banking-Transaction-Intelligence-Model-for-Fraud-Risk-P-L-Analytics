# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Data-residency guard.

An in-country deployment (for example India, `residency.jurisdiction: IN`)
must not send customer or payment data to any endpoint outside the
jurisdiction. The guard lists every outbound endpoint the configuration makes
BTI talk to:
- database
- feature store (Redis)
- Kafka brokers
- SMTP relay and alert webhook
- step-up provider webhook
- the investigation copilot's LLM endpoint, when an API key is configured

It checks each host against `residency.in_country_hosts`. That list holds
hostnames, `*.suffix` patterns and CIDR ranges which the bank attests are
inside the jurisdiction.

The guard cannot know where a host physically is. It enforces the bank's own
attested inventory, which makes any endpoint outside that inventory a visible
decision rather than a silent one. Two modes:
- `warn` reports violations on /readyz and in the logs
- `enforce` stops the API and the stream consumer from starting, and refuses
  copilot calls to an endpoint outside the list

Local files (SQLite, model registry, graph snapshot) live on the host's own
disk, so they are in-country if and only if the deployment is. The bank
attests that for the hosting region itself.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import os
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from bti.config import get_settings
from bti.logging_config import get_logger

log = get_logger("operations.residency")


class ResidencyViolation(RuntimeError):
    pass


def _host(url: str) -> Optional[str]:
    if not url:
        return None
    if "://" not in url:
        url = "//" + url
    parts = urlsplit(url)
    if parts.scheme.startswith("sqlite"):
        return "local-disk"
    return parts.hostname


def endpoints() -> List[Dict]:
    s = get_settings()
    out = [{"name": "database", "host": _host(s.database_url), "carries": "scores, cases, audit log, transactions"}]
    if s.feature_store_url:
        out.append({"name": "feature_store", "host": _host(s.feature_store_url), "carries": "transaction history"})
    for i, server in enumerate(s.streaming_bootstrap_servers.split(",")):
        if server.strip():
            out.append({"name": f"kafka_broker_{i + 1}", "host": _host(server.strip()),
                        "carries": "transactions and decisions"})
    if s.smtp_host:
        out.append({"name": "smtp", "host": s.smtp_host, "carries": "alert e-mails (case details)"})
    if s.webhook_url:
        out.append({"name": "alert_webhook", "host": _host(s.webhook_url), "carries": "alerts (case details)"})
    if s.stepup_webhook_url:
        out.append({"name": "stepup_webhook", "host": _host(s.stepup_webhook_url), "carries": "customer contact"})
    if os.environ.get("ANTHROPIC_API_KEY"):
        out.append({"name": "copilot_llm", "host": _host(os.environ.get("ANTHROPIC_BASE_URL",
                                                                        "https://api.anthropic.com")),
                    "carries": "case and transaction details sent to the LLM"})
    return out


def in_country(host: Optional[str], allowed: List[str]) -> bool:
    if host is None:
        return True
    if host == "local-disk":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    for rule in allowed:
        rule = str(rule).strip().lower()
        if ip is not None and "/" in rule:
            try:
                if ip in ipaddress.ip_network(rule, strict=False):
                    return True
            except ValueError:
                continue
        elif host.lower() == rule or fnmatch.fnmatch(host.lower(), rule):
            return True
    return False


def check() -> Dict:
    s = get_settings()
    jurisdiction = (s.residency_jurisdiction or "").upper() or None
    rows = []
    for e in endpoints():
        rows.append({**e, "in_country": in_country(e["host"], s.residency_in_country_hosts)})
    violations = [r for r in rows if not r["in_country"]] if jurisdiction else []
    return {"jurisdiction": jurisdiction, "mode": s.residency_mode, "enforced": bool(jurisdiction),
            "endpoints": rows, "violations": violations, "compliant": not violations,
            "note": None if jurisdiction else "No residency jurisdiction configured; the guard is off."}


def enforce_at_startup(component: str) -> Dict:
    """Log the residency verdict; raise in enforce mode when an endpoint is outside the jurisdiction."""
    result = check()
    if result["violations"]:
        names = ", ".join(f"{v['name']} ({v['host']})" for v in result["violations"])
        msg = f"Data-residency ({result['jurisdiction']}): endpoints outside the attested in-country list: {names}"
        if result["mode"] == "enforce":
            log.error(msg, extra={"component": component})
            raise ResidencyViolation(msg)
        log.warning(msg, extra={"component": component})
    return result


def allow_endpoint(name: str) -> None:
    """Raise before sending data to endpoint `name` if residency is enforced and it is outside."""
    result = check()
    if result["mode"] == "enforce" and any(v["name"] == name for v in result["violations"]):
        raise ResidencyViolation(f"{name} is outside the {result['jurisdiction']} in-country list; "
                                 f"data-residency mode is enforce")
