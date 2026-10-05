# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Alert-volume and staffing forecast, and the review-capacity setting it implies.

**Volume.** Daily transaction volume is forecast with the same over-dispersed
Poisson model as fraud counts (`bti.planning.forecast.CountModel`: day-of-week
plus a shrunk trend), as Monte Carlo paths.

**Cases.** Every REVIEW and every DECLINE opens a case: declines need a
disposition too, because no chargeback will ever label them.
- *Rates.* The rate per queue comes from replaying the live policy, meaning the
  model and its fitted capacity prices, on the model's out-of-time window.
  Provisional models turn declines into reviews.
- *Queues.* Cases are routed to queues by the same rules as live.
- *Paths.* Volume paths are thinned binomially into case paths.

**Handling time.** Median minutes from assignment to close, per queue, from
the case table once it holds 30 closed cases. Until then, the configured
defaults are used (`planning.aht_minutes`; illustrative, replace with the
bank's).

**Staffing.**
- Cases arrive with the transactions' hourly profile.
- Analysts are pooled across queues. For each hour, Erlang C gives the fewest
  analysts who pick up the target share of cases (`service_level_target`)
  within the strictest active SLA, without exceeding `max_occupancy`.
- The raw workload (cases × handling time ÷ occupancy) is reported alongside.
- Analyst-hours are converted to rostered FTE with shrinkage (leave, training,
  meetings) and paid hours.
- The plan is given at the median and at the 90th-percentile day ("plan to
  P90").
- A 24-hour cover floor (one analyst on duty around the clock) is reported
  separately, because for a small book it dominates. `--scale` projects the
  volume to a bank's size.

**Review capacity.** Given the analysts actually rostered (`--analysts`), the
sustainable cases per day become a recommended `max_review_rate`. This is only
a proposal. Applying it means refitting the live capacity prices under a named
fitter (`python -m bti.operations.capacity --fitted-by … --max-review-rate …`),
which keeps the change on the governed path.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from bti.config import get_settings
from bti.planning.forecast import HORIZONS, CountModel

OUT = Path("outputs/planning/staffing_forecast.json")


# ── Erlang C ─────────────────────────────────────────────────────────────────
def erlang_c(agents: int, traffic: float) -> float:
    """Probability an arriving case waits (M/M/N), via the stable Erlang B recursion."""
    if agents <= traffic:
        return 1.0
    b = 1.0
    for n in range(1, agents + 1):
        b = traffic * b / (n + traffic * b)
    return agents * b / (agents - traffic * (1 - b))


def service_level(agents: int, arrivals_per_hour: float, aht_minutes: float, sla_minutes: float) -> float:
    traffic = arrivals_per_hour * aht_minutes / 60.0
    if traffic == 0:
        return 1.0
    if agents <= traffic:
        return 0.0
    return 1.0 - erlang_c(agents, traffic) * math.exp(-(agents - traffic) * sla_minutes / aht_minutes)


def agents_needed(arrivals_per_hour: float, aht_minutes: float, sla_minutes: float, target: float = 0.9,
                  max_occupancy: float = 0.85) -> int:
    if arrivals_per_hour <= 0:
        return 0
    traffic = arrivals_per_hour * aht_minutes / 60.0
    n = max(1, math.ceil(traffic / max_occupancy))
    while service_level(n, arrivals_per_hour, aht_minutes, sla_minutes) < target:
        n += 1
    return n


# ── inputs ───────────────────────────────────────────────────────────────────
def case_rates(model_id: Optional[str] = None) -> Dict:
    """Cases per transaction, by queue, from replaying the live policy on the model's out-of-time window."""
    from bti.modeling import registry
    from bti.operations.capacity import capacity_overrides
    from bti.operations.cases import choose_queue
    from bti.operations.decisioning import _choose_actions, cost_model_for, step_up_available
    from bti.planning.replay import scored_window
    model_id, w = scored_window(model_id)
    cm = cost_model_for(None, capacity_overrides(model_id))
    can_step = np.array([step_up_available(c, t) for c, t in zip(w["channel"], w["transaction_type"])])
    actions = _choose_actions(w["p"].to_numpy(), w["amount_usd"].to_numpy(), can_step, cm)
    provisional = registry.model_for_role("champion") != model_id
    cases = np.isin(actions, ["REVIEW", "DECLINE"])
    queues = pd.Series([choose_queue(p, a) for p, a in zip(w["p"], w["amount_usd"])])[cases]
    rates = (queues.value_counts() / len(w)).to_dict()
    return {"model_id": model_id, "provisional": provisional, "transactions": int(len(w)),
            "case_rate": round(float(cases.mean()), 5),
            "rates_by_queue": {q: round(float(rates.get(q, 0.0)), 5) for q in get_settings().case_queues},
            "action_mix": {a: round(float((actions == a).mean()), 5) for a in ("APPROVE", "STEP_UP", "REVIEW", "DECLINE")},
            "capacity_policy": capacity_overrides(model_id)}


def handling_minutes(db=None, min_closed: int = 30) -> Dict[str, Dict]:
    defaults = get_settings().planning_aht_minutes
    out = {q: {"minutes": float(defaults.get(q, 20)), "source": "default"} for q in get_settings().case_queues}
    if db is None:
        return out
    from bti.database.models import FraudCase
    rows = db.query(FraudCase.queue, FraudCase.assigned_at, FraudCase.closed_at).filter(
        FraudCase.closed_at.isnot(None), FraudCase.assigned_at.isnot(None)).all()
    frame = pd.DataFrame(rows, columns=["queue", "assigned_at", "closed_at"])
    for q in out:
        g = frame[frame["queue"] == q]
        if len(g) >= min_closed:
            minutes = (pd.to_datetime(g["closed_at"]) - pd.to_datetime(g["assigned_at"])).dt.total_seconds() / 60
            out[q] = {"minutes": round(float(np.median(minutes.clip(lower=1))), 1), "source": f"{len(g)} closed cases"}
    return out


def hourly_profile(history: pd.DataFrame) -> np.ndarray:
    hours = history["hour"].dropna().astype(int).clip(0, 23)
    counts = np.bincount(hours, minlength=24).astype(float) + 1.0       # smoothing: no empty hours
    return counts / counts.sum()


# ── forecast ─────────────────────────────────────────────────────────────────
def forecast(history: pd.DataFrame, rates: Dict, aht: Dict[str, Dict], as_of: Optional[pd.Timestamp] = None,
             horizons: Sequence[int] = HORIZONS, fit_days: int = 365, n_paths: int = 2000, seed: int = 0,
             analysts_fte: Optional[float] = None, volume_scale: float = 1.0) -> Dict:
    s = get_settings()
    as_of = pd.Timestamp(as_of or history["date"].max()).normalize()
    start = as_of - pd.Timedelta(days=fit_days - 1)
    days = pd.date_range(start, as_of, freq="D")
    volume = history[(history["date"] >= start) & (history["date"] <= as_of)].groupby("date").size() \
        .reindex(days, fill_value=0)
    model = CountModel(dow=True, trend=True).fit(days, volume.to_numpy(float))
    future = pd.date_range(as_of + pd.Timedelta(days=1), periods=max(horizons), freq="D")
    rng = np.random.default_rng(seed)
    vol_paths = model.simulate(future, n_paths, rng)
    if volume_scale != 1.0:                                     # project the book to another size (what-if)
        vol_paths = np.round(vol_paths * volume_scale).astype(np.int64)
    profile = hourly_profile(history[history["date"] >= start])
    queues = s.case_queues
    case_paths = {q: rng.binomial(vol_paths, rates["rates_by_queue"].get(q, 0.0)) for q in queues}
    all_cases = sum(case_paths.values())

    def staffing_for(daily_cases: Dict[str, float]) -> Dict:
        """Workload FTE (continuous) and pooled Erlang C FTE (analysts work any queue; strictest SLA governs)."""
        productive = s.planning_paid_hours_per_day * (1 - s.planning_shrinkage)
        total = sum(daily_cases.values())
        workload_minutes = sum(daily_cases[q] * aht[q]["minutes"] for q in queues)
        weighted_aht = workload_minutes / total if total else 0.0
        strictest = min(r["sla_minutes"] for q, r in queues.items() if daily_cases[q] > 0) if total else 0
        per_hour = total * profile
        agents = [agents_needed(a, weighted_aht, strictest, s.planning_service_level_target,
                                s.planning_max_occupancy) for a in per_hour]
        return {"cases_per_day": round(total, 2),
                "by_queue": {q: round(daily_cases[q], 2) for q in queues},
                "workload_hours_per_day": round(workload_minutes / 60 / s.planning_max_occupancy, 2),
                "workload_fte": round(workload_minutes / 60 / s.planning_max_occupancy / productive, 2),
                "erlang_analyst_hours_per_day": int(sum(agents)), "peak_hour_analysts": int(max(agents)),
                "erlang_fte": round(sum(agents) / productive, 2)}

    plans = {}
    for h in horizons:
        window = slice(0, h)
        daily_mean = {q: float(case_paths[q][:, window].mean()) for q in queues}
        daily_p90 = {q: float(np.percentile(case_paths[q][:, window], 90)) for q in queues}
        plans[f"{h}d"] = {
            "transactions_per_day": {"p50": round(float(np.median(vol_paths[:, window])), 1),
                                     "p90": round(float(np.percentile(vol_paths[:, window], 90)), 1)},
            "cases_per_day": {"p50": round(float(np.median(all_cases[:, window])), 1),
                              "p90": round(float(np.percentile(all_cases[:, window], 90)), 1)},
            "cases_total": {"p50": int(np.median(all_cases[:, window].sum(axis=1))),
                            "p90": int(np.percentile(all_cases[:, window].sum(axis=1), 90))},
            "staffing_at_mean_day": staffing_for(daily_mean),
            "staffing_at_p90_day": staffing_for(daily_p90),
        }
    productive = s.planning_paid_hours_per_day * (1 - s.planning_shrinkage)
    floor_fte = round(24 / productive, 2)
    out = {"as_of": as_of.date().isoformat(), "volume_scale": volume_scale, "volume_model": model.describe(),
           "case_rates": rates,
           "handling_minutes": aht,
           "assumptions": {"service_level_target": s.planning_service_level_target,
                           "max_occupancy": s.planning_max_occupancy, "shrinkage": s.planning_shrinkage,
                           "paid_hours_per_day": s.planning_paid_hours_per_day,
                           "sla_minutes": {q: r["sla_minutes"] for q, r in queues.items()}},
           "plans": plans,
           "around_the_clock_floor_fte": floor_fte,
           "note": "workload_fte is the case work itself; erlang_fte also keeps the strictest queue's SLA every hour "
                   "(pooled analysts). For a small book the Erlang figure is set by around-the-clock cover; for a "
                   "large one it converges to the workload plus a queueing margin."}
    if analysts_fte is not None:
        out["review_capacity"] = recommend_review_rate(analysts_fte, plans[f"{horizons[0]}d"], rates, aht)
    return out


def recommend_review_rate(analysts_fte: float, plan: Dict, rates: Dict, aht: Dict[str, Dict]) -> Dict:
    """Sustainable cases per day for the rostered analysts, as a proposed max_review_rate (not applied)."""
    s = get_settings()
    productive_minutes = analysts_fte * s.planning_paid_hours_per_day * (1 - s.planning_shrinkage) * 60 \
        * s.planning_max_occupancy
    total_rate = sum(rates["rates_by_queue"].values()) or 1e-9
    weighted_aht = sum(rates["rates_by_queue"][q] * aht[q]["minutes"] for q in aht) / total_rate
    sustainable = productive_minutes / weighted_aht
    volume_p90 = plan["transactions_per_day"]["p90"]
    declines = rates["action_mix"]["DECLINE"]        # declines open cases too (reviews, if the model is provisional)
    proposed = max(0.0, min(sustainable / max(volume_p90, 1e-9) - declines, 0.25))
    warning = None
    if sustainable / max(volume_p90, 1e-9) <= declines:
        warning = (f"Declines alone ({declines:.2%} of transactions, {declines * volume_p90:,.0f}/day at P90) open more "
                   f"cases than {analysts_fte:g} FTE can work ({sustainable:,.0f}/day): no room for reviews. Add "
                   f"analysts, shorten decline handling, or reduce declines through the cost model.")
    return {"analysts_fte": analysts_fte, "weighted_handling_minutes": round(weighted_aht, 1), "warning": warning,
            "sustainable_cases_per_day": round(sustainable, 1), "planning_volume_p90_per_day": volume_p90,
            "current_max_review_rate": s.max_review_rate, "proposed_max_review_rate": round(proposed, 4),
            "status": "proposal — not applied",
            "apply_with": f"python -m bti.operations.capacity --fitted-by '<name>' --max-review-rate {proposed:.4f}"}


def run(path: Optional[str] = None, analysts_fte: Optional[float] = None, db=None, volume_scale: float = 1.0) -> Dict:
    from bti.planning.history import load
    history = load(path)
    report = forecast(history, case_rates(), handling_minutes(db), analysts_fte=analysts_fte,
                      volume_scale=volume_scale)
    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Alert-volume and staffing forecast")
    parser.add_argument("--data", default=None)
    parser.add_argument("--analysts", type=float, default=None, help="rostered analyst FTE, for a review-rate proposal")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply forecast volume (project to a bank's size)")
    args = parser.parse_args()
    r = run(args.data, args.analysts if args.analysts is not None else get_settings().planning_analysts_fte,
            volume_scale=args.scale)
    print(f"case rate {r['case_rates']['case_rate']:.2%} of transactions; AHT "
          + ", ".join(f"{q} {v['minutes']:.0f} min ({v['source']})" for q, v in r["handling_minutes"].items()))
    for h, p in r["plans"].items():
        print(f"  {h}: {p['transactions_per_day']['p50']:.0f} txns/day, cases/day p50 {p['cases_per_day']['p50']} "
              f"p90 {p['cases_per_day']['p90']}; workload FTE {p['staffing_at_mean_day']['workload_fte']} "
              f"(P90 day {p['staffing_at_p90_day']['workload_fte']}); Erlang FTE {p['staffing_at_mean_day']['erlang_fte']} "
              f"(P90 day {p['staffing_at_p90_day']['erlang_fte']})")
    print(f"  24-hour cover floor: {r['around_the_clock_floor_fte']} FTE")
    if "review_capacity" in r:
        print(json.dumps(r["review_capacity"], indent=1))


if __name__ == "__main__":
    main()
