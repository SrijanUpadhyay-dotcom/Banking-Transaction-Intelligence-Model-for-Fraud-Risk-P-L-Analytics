# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Scheduled planning jobs: weekly loss and staffing forecasts from the database history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict

from bti.logging_config import get_logger

log = get_logger("planning.jobs")
MIN_DAYS = 120


def weekly_forecasts(db) -> Dict:
    """Loss forecast (with backtest) and staffing forecast from the transactions table; written to outputs."""
    from bti.config import get_settings
    from bti.planning import forecast as loss, staffing
    from bti.planning.history import from_database
    history = from_database(db)
    span = (history["date"].max() - history["date"].min()).days if not history.empty else 0
    if span < MIN_DAYS:
        log.warning("Planning forecast skipped: too little history", extra={"days": span})
        return {"status": "skipped", "reason": f"{span} days of history; at least {MIN_DAYS} needed"}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": "database",
              "forecast": loss.forecast(history, fit_days=min(365, span))}
    if span >= 365 + 120:
        report["backtest"] = loss.backtest(history)
    loss.OUT.parent.mkdir(parents=True, exist_ok=True)
    loss.OUT.write_text(json.dumps(report, indent=2, default=str))
    staff = staffing.forecast(history, staffing.case_rates(), staffing.handling_minutes(db),
                              fit_days=min(365, span), analysts_fte=get_settings().planning_analysts_fte)
    staffing.OUT.write_text(json.dumps(staff, indent=2, default=str))
    return {"status": "ok", "as_of": report["forecast"]["as_of"],
            "loss_90d_p50": report["forecast"]["total"]["90d"]["loss_usd"]["p50"],
            "cases_per_day_p50": staff["plans"]["30d"]["cases_per_day"]["p50"]}
