# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Attack early warning: change-point detection by typology, merchant and corridor.

**Families** (segments watched):
- `typology`: fraud type
- `merchant_category`
- `merchant`: merchant name; card testing and merchant compromise show here
  first
- `corridor`: account country × channel. With the ISO 20022 feed this
  becomes debtor country → creditor country.

**Signals** per segment per day:
- `fraud`: confirmed fraud, as known on the day. It is exact but late,
  because fraud is confirmed days to weeks afterwards.
- `alerts`: transactions the model intervened on (any action but APPROVE).
  These exist the same day, so an attack shows here before a single dispute
  is filed. Typology has no alert signal, because a fraud's type is only
  known once it is confirmed.

**Test.** A rate-based Poisson CUSUM per segment and signal:
- *Expected count:* the segment's volume that day × its own baseline rate.
  The baseline covers the 90 days before a 7-day guard band, so an attack
  under way does not raise its own baseline. It is shrunk towards the
  portfolio rate for thin segments. A rise in volume alone therefore does not
  alarm; a rise in the rate does.
- *Statistic:* `S = max(0, S + x·ln ρ − E·(ρ − 1))`, tuned to detect a
  doubling (ρ = 2).

**False alarms.** Watching hundreds of segments guarantees false alarms
unless the budget is set per family. Each family gets
`false_alarms_per_family_per_month` (default 1), shared across its signals.
Each segment's threshold h comes from the in-control average run length that
implies (segments × signals × 30 days per false alarm), using a simulated (expected count × threshold) → ARL table.
So h is right for a segment's own volume. A thin merchant and a busy corridor
get different thresholds.

**Evaluation** (`evaluate`) injects four attacks into the history and reports
the detection delay for each, plus the false-alarm rate on the clean data:
- card testing at one merchant
- a typology surge
- a corridor spike
- a merchant-category compromise

**Live.** A daily job (`run_daily`) records new alarms in `early_warnings`,
writes an audit event and posts them to the alert webhook. Alarms are
acknowledged or closed through the API.

    python -m bti.planning.early_warning evaluate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from bti.planning.history import known_as_of

OUT = Path("outputs/planning/early_warning_evaluation.json")
CALIBRATION = Path("outputs/planning/early_warning_calibration.json")
MULTIPLIERS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
FAMILIES = {"typology": ("fraud_type",), "merchant_category": ("merchant_category",), "merchant": ("merchant_name",),
            "corridor": ("country", "channel")}
SIGNALS = {"typology": ("fraud",), "merchant_category": ("fraud", "alerts"), "merchant": ("fraud", "alerts"),
           "corridor": ("fraud", "alerts")}
RHO = 2.0


# ── thresholds ───────────────────────────────────────────────────────────────
_E_GRID = np.geomspace(0.002, 60.0, 20)
_H_GRID = np.linspace(0.4, 24.0, 60)


@lru_cache(maxsize=4)
def _arl_table(rho: float = RHO, chains: int = 400, steps: int = 3000, seed: int = 7) -> np.ndarray:
    """In-control average run length (days) for each (expected daily count, threshold), by simulation."""
    rng = np.random.default_rng(seed)
    a = np.log(rho)
    table = np.empty((len(_E_GRID), len(_H_GRID)))
    for i, mean in enumerate(_E_GRID):
        s = np.zeros((chains, len(_H_GRID)))
        alarms = np.zeros(len(_H_GRID))
        b = mean * (rho - 1)
        for _ in range(steps):
            inc = rng.poisson(mean, chains) * a - b
            s = np.maximum(0.0, s + inc[:, None])
            hit = s >= _H_GRID
            alarms += hit.sum(axis=0)
            s[hit] = 0.0
        arl = (chains * steps) / np.maximum(alarms, 0.5)
        table[i] = np.maximum.accumulate(arl)                    # monotone in h (removes simulation noise)
    return table


def threshold(expected_daily: np.ndarray, target_arl: float, rho: float = RHO) -> np.ndarray:
    """CUSUM threshold giving roughly `target_arl` days between false alarms at this expected daily count."""
    table = _arl_table(rho)
    h_by_e = np.array([np.interp(np.log(target_arl), np.log(row), _H_GRID) for row in table])
    e = np.clip(np.asarray(expected_daily, float), _E_GRID[0], _E_GRID[-1])
    return np.interp(np.log(e), np.log(_E_GRID), h_by_e)


# ── daily matrices ───────────────────────────────────────────────────────────
def _segment_key(frame: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    key = frame[list(cols)].astype("string").fillna("(unknown)")
    return key.iloc[:, 0] if len(cols) == 1 else key.agg(" × ".join, axis=1)


def _matrix(frame: pd.DataFrame, key: pd.Series, value: Optional[str], days: pd.DatetimeIndex) -> pd.DataFrame:
    g = frame.assign(_k=key.to_numpy())
    if value is None:
        m = g.groupby(["date", "_k"]).size()
    else:
        m = g.groupby(["date", "_k"])[value].sum()
    return m.unstack(fill_value=0).reindex(days, fill_value=0)


def monitor(history: pd.DataFrame, alerts: Optional[pd.Series] = None, as_of: Optional[pd.Timestamp] = None,
            families: Optional[Sequence[str]] = None, rho: float = RHO, false_alarms_per_family_per_month: float = 1.0,
            baseline_days: int = 90, guard_days: int = 7, prior_volume: float = 500.0,
            min_expected_per_day: float = 0.0, arl_multiplier: Optional[Dict[str, float]] = None) -> Dict:
    """
    Run the CUSUMs over the whole history (sequentially: each day uses only earlier days) and return every alarm,
    plus the segments in alarm at `as_of`. `alerts` is a 0/1 Series aligned with `history` (model interventions).
    """
    as_of = pd.Timestamp(as_of or history["date"].max()).normalize()
    known = known_as_of(history, as_of)
    if alerts is not None:
        known = known.assign(alerts=alerts.reindex(known.index).fillna(0).astype(int))
    days = pd.date_range(known["date"].min(), as_of, freq="D")
    warm = baseline_days + guard_days
    out = {"as_of": as_of.date().isoformat(), "rho": rho, "baseline_days": baseline_days, "guard_days": guard_days,
           "false_alarms_per_family_per_month": false_alarms_per_family_per_month, "families": {}}
    for family in families or FAMILIES:
        cols = FAMILIES[family]
        if family == "typology":
            frame = known[known["fraud"] == 1]
            key = _segment_key(frame, cols)
            volume_frame, volume_key = known, None
        else:
            frame, key = known, _segment_key(known, cols)
        fam = {"segments": 0, "target_arl_days": None, "alarms": [], "in_alarm": []}
        for signal in SIGNALS[family]:
            if signal == "alerts" and "alerts" not in known.columns:
                continue
            if family == "typology":
                # typology: rate per transaction overall (the segment volume is the whole book)
                x = _matrix(frame, key, None, days)
                vol_total = known.groupby("date").size().reindex(days, fill_value=0).to_numpy(float)
                v = pd.DataFrame(np.repeat(vol_total[:, None], x.shape[1], axis=1), index=days, columns=x.columns)
            else:
                x = _matrix(frame, key, signal, days)
                v = _matrix(frame, key, None, days)[x.columns]
            X, V = x.to_numpy(float), v.to_numpy(float)
            n_seg = X.shape[1]
            fam["segments"] = max(fam["segments"], n_seg)
            n_signals = len([g for g in SIGNALS[family] if g != "alerts" or "alerts" in known.columns])
            target_arl = n_seg * n_signals * 30.0 / false_alarms_per_family_per_month   # budget shared by signals
            target_arl *= (arl_multiplier or {}).get(family, 1.0)     # historical calibration (estimation noise)
            fam["target_arl_days"] = round(target_arl, 0)
            cx = np.vstack([np.zeros(n_seg), np.cumsum(X, axis=0)])
            cv = np.vstack([np.zeros(n_seg), np.cumsum(V, axis=0)])
            S = np.zeros(n_seg)
            run_start = np.full(n_seg, -1)
            run_obs, run_exp = np.zeros(n_seg), np.zeros(n_seg)
            a = np.log(rho)
            for t in range(warm, len(days)):
                lo, hi = t - guard_days - baseline_days, t - guard_days
                xs, vs = cx[hi] - cx[lo], cv[hi] - cv[lo]
                if family == "typology":                       # prior: an equal share of the book's fraud rate
                    portfolio = xs.sum() / n_seg / max(vs[0], 1.0)
                else:
                    portfolio = xs.sum() / max(vs.sum(), 1.0)
                rate = (xs + prior_volume * portfolio) / (vs + prior_volume)
                E = V[t] * rate
                h = threshold(np.maximum(vs / baseline_days * rate, 1e-9), target_arl, rho)
                prev = S.copy()
                S = np.maximum(0.0, S + X[t] * a - E * (rho - 1))
                started = (prev == 0) & (S > 0)
                run_start[started] = t
                run_obs[started], run_exp[started] = 0.0, 0.0
                run_obs[S > 0] += X[t][S > 0]
                run_exp[S > 0] += E[S > 0]
                hit = (S >= h) & (vs / baseline_days * rate >= min_expected_per_day)
                for j in np.flatnonzero(hit):
                    event = {"segment": str(x.columns[j]), "signal": signal,
                             "started_on": days[max(run_start[j], 0)].date().isoformat(),
                             "alarmed_on": days[t].date().isoformat(), "observed": int(run_obs[j]),
                             "expected": round(float(run_exp[j]), 2),
                             "ratio": round(float(run_obs[j] / max(run_exp[j], 1e-9)), 2),
                             "cusum": round(float(S[j]), 2), "threshold": round(float(h[j]), 2)}
                    fam["alarms"].append(event)
                    if (as_of - days[t]).days <= guard_days:
                        fam["in_alarm"].append(event)
                S[hit] = 0.0
        out["families"][family] = fam
    return out


# ── evaluation with injected attacks ─────────────────────────────────────────
def _inject(history: pd.DataFrame, alerts: pd.Series, rows: pd.DataFrame, n: int, start: pd.Timestamp, days: int,
            intervention_rate: float, rng: np.random.Generator, overrides: Dict, amount_scale: float = 1.0):
    template = rows.sample(n, replace=True, random_state=int(rng.integers(1 << 30))).copy()
    template["date"] = [start + pd.Timedelta(days=int(d)) for d in rng.integers(0, days, n)]
    for k, val in overrides.items():
        template[k] = val
    template["fraud"] = 1
    template["confirmed_at"] = pd.NaT
    template["amount_usd"] *= amount_scale
    template["loss_usd"] = template["amount_usd"]
    template["transaction_id"] = [f"INJ-{start.date()}-{i}" for i in range(n)]
    new_alerts = pd.Series((rng.random(n) < intervention_rate).astype(int))
    h = pd.concat([history, template], ignore_index=True)
    a = pd.concat([alerts.reset_index(drop=True), new_alerts], ignore_index=True)
    return h, a


def scenarios(history: pd.DataFrame) -> List[Dict]:
    """Attacks to inject: chosen from the data so they exist in it; start dates late enough for a baseline."""
    last = history["date"].max()
    merchants = history["merchant_name"].value_counts()
    mid_merchant = merchants.index[len(merchants) // 2]
    return [
        {"name": "card testing at one merchant", "family": "merchant", "segment": mid_merchant,
         "start": last - pd.Timedelta(days=200), "days": 3, "n": 30, "overrides": {"merchant_name": mid_merchant},
         "amount_scale": 0.05, "filter": {"merchant_name": mid_merchant}},
        {"name": "authorised push payment surge (3x for 3 weeks)", "family": "typology",
         "segment": "Authorised Push Payment", "start": last - pd.Timedelta(days=150), "days": 21, "n": 14,
         "overrides": {"fraud_type": "Authorised Push Payment"}, "filter": {"fraud_type": "Authorised Push Payment"}},
        {"name": "corridor spike: Nigeria × USSD (5x for 2 weeks)", "family": "corridor", "segment": "Nigeria × USSD",
         "start": last - pd.Timedelta(days=100), "days": 14, "n": 6,
         "overrides": {"country": "Nigeria", "channel": "USSD"}, "filter": {"country": "Nigeria", "channel": "USSD"}},
        {"name": "merchant-category compromise: electronics (2x for a month)", "family": "merchant_category",
         "segment": "Electronics & Technology", "start": last - pd.Timedelta(days=60), "days": 30, "n": 18,
         "overrides": {"merchant_category": "Electronics & Technology"},
         "filter": {"merchant_category": "Electronics & Technology"}},
    ]


def calibrate(history: pd.DataFrame, alerts: Optional[pd.Series], false_alarms_per_family_per_month: float = 1.0,
              **kwargs) -> Dict[str, Dict]:
    """
    Per family, the smallest ARL multiplier whose false-alarm rate on the history is within budget. The table
    assumes each segment's baseline is known; estimating it from 90 days adds noise that this absorbs. It treats
    the history as attack-free, so calibrate on a period the fraud team has reviewed.
    """
    months = max((history["date"].max() - history["date"].min()).days - 97, 1) / 30.0
    remaining = set(FAMILIES)
    result = {}
    for m in MULTIPLIERS:
        if not remaining:
            break
        r = monitor(history, alerts, families=sorted(remaining), false_alarms_per_family_per_month=false_alarms_per_family_per_month,
                    arl_multiplier={f: m for f in remaining}, **kwargs)
        for f in list(remaining):
            rate = len(r["families"][f]["alarms"]) / months
            if rate <= false_alarms_per_family_per_month or m == MULTIPLIERS[-1]:
                result[f] = {"multiplier": m, "false_alarms_per_month": round(rate, 2)}
                remaining.discard(f)
    return result


def load_calibration() -> Optional[Dict[str, float]]:
    try:
        return {f: v["multiplier"] for f, v in json.loads(CALIBRATION.read_text())["families"].items()}
    except (OSError, ValueError, KeyError):
        return None


def evaluate(history: pd.DataFrame, alerts: pd.Series, seed: int = 3, **kwargs) -> Dict:
    rng = np.random.default_rng(seed)
    frauds = history[history["fraud"] == 1]
    intervention_rate = float(alerts[history["fraud"] == 1].mean()) if len(frauds) else 0.0
    months = max((history["date"].max() - history["date"].min()).days - 97, 1) / 30.0
    uncalibrated = monitor(history, alerts, **kwargs)
    cal = calibrate(history, alerts, **kwargs)
    multipliers = {f: v["multiplier"] for f, v in cal.items()}
    false_alarms = {f: {"segments": v["segments"],
                        "uncalibrated_per_month": round(len(v["alarms"]) / months, 2),
                        "arl_multiplier": cal[f]["multiplier"],
                        "calibrated_per_month": cal[f]["false_alarms_per_month"]}
                    for f, v in uncalibrated["families"].items()}
    h, a = history, alerts.reset_index(drop=True)
    plan = scenarios(history)
    for sc in plan:
        mask = np.ones(len(frauds), bool)
        for k, val in sc["filter"].items():
            mask &= (frauds[k] == val).to_numpy()
        pool = frauds[mask] if mask.any() else frauds
        h, a = _inject(h, a, pool, sc["n"], sc["start"], sc["days"], intervention_rate, rng, sc["overrides"],
                       sc.get("amount_scale", 1.0))
    attacked = monitor(h, a, arl_multiplier=multipliers, **kwargs)
    results = []
    for sc in plan:
        hits = [e for e in attacked["families"][sc["family"]]["alarms"]
                if e["segment"] == sc["segment"] and pd.Timestamp(e["alarmed_on"]) >= sc["start"]
                and pd.Timestamp(e["alarmed_on"]) <= sc["start"] + pd.Timedelta(days=sc["days"] + 30)]
        first = {}
        for e in hits:
            first.setdefault(e["signal"], e)
        results.append({"scenario": sc["name"], "segment": sc["segment"], "start": sc["start"].date().isoformat(),
                        "duration_days": sc["days"], "injected_frauds": sc["n"], "detected": bool(hits),
                        "delay_days_by_signal": {s: (pd.Timestamp(e["alarmed_on"]) - sc["start"]).days
                                                 for s, e in first.items()}})
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps({"calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                       "history": [str(history["date"].min().date()), str(history["date"].max().date())],
                                       "families": cal}, indent=2))
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_intervention_rate_on_fraud": round(intervention_rate, 3),
            "false_alarms": false_alarms, "months_monitored": round(months, 1),
            "injected_attacks": results,
            "note": "Synthetic data and injected attacks: capability evidence, not field performance. Thresholds are "
                    "calibrated on the clean history (treated as attack-free). Alert-signal delays assume injected "
                    "frauds are intervened on at the model's real rate on fraud."}


def _alerts_for(history: pd.DataFrame) -> pd.Series:
    """Model interventions (non-APPROVE under the live policy) for history rows, by transaction id."""
    from bti.operations.capacity import capacity_overrides
    from bti.operations.decisioning import _choose_actions, cost_model_for, step_up_available
    from bti.planning.replay import scored_window
    model_id, w = scored_window(window="all")
    cm = cost_model_for(None, capacity_overrides(model_id))
    can_step = np.array([step_up_available(c, t) for c, t in zip(w["channel"], w["transaction_type"])])
    acted = _choose_actions(w["p"].to_numpy(), w["amount_usd"].to_numpy(), can_step, cm) != "APPROVE"
    by_id = pd.Series(acted.astype(int), index=w["transaction_id"].to_numpy())
    return pd.Series(history["transaction_id"].map(by_id).fillna(0).astype(int).to_numpy(), index=history.index)


# ── live job ─────────────────────────────────────────────────────────────────
def run_daily(db, notify: bool = True, as_of: Optional[pd.Timestamp] = None) -> Dict:
    """Monitor the database history; record and dispatch alarms raised in the last guard window."""
    from bti.database.models import AuditLog, EarlyWarning, ScoreLog
    from bti.planning.history import from_database
    history = from_database(db)
    if history.empty:
        return {"status": "no history"}
    decisions = pd.read_sql_query(
        db.query(ScoreLog.transaction_id, ScoreLog.decision).filter(ScoreLog.is_shadow.is_(False)).statement,
        db.get_bind())
    alerts = None
    if not decisions.empty:
        acted = decisions.drop_duplicates("transaction_id", keep="last").set_index("transaction_id")["decision"] != "APPROVE"
        alerts = history["transaction_id"].map(acted.astype(int)).fillna(0).astype(int)
    result = monitor(history, alerts, as_of=as_of, arl_multiplier=load_calibration())
    new = []
    for family, fam in result["families"].items():
        for e in fam["in_alarm"]:
            exists = db.query(EarlyWarning).filter_by(family=family, segment=e["segment"], signal=e["signal"],
                                                      started_on=e["started_on"]).first()
            if exists:
                continue
            row = EarlyWarning(family=family, segment=e["segment"], signal=e["signal"], started_on=e["started_on"],
                               alarmed_on=e["alarmed_on"], observed=e["observed"], expected=e["expected"],
                               ratio=e["ratio"], cusum=e["cusum"], threshold=e["threshold"], status="open",
                               created_at=datetime.utcnow())
            db.add(row)
            db.add(AuditLog(ts=datetime.utcnow(), event_type="EARLY_WARNING", payload={"family": family, **e}))
            new.append({"family": family, **e})
    db.commit()
    if new and notify:
        try:
            from bti.alerts import AlertDispatcher
            AlertDispatcher()._post_webhook({"source": "BTI-EarlyWarning", "alarms": new})
        except Exception:
            pass
    return {"as_of": result["as_of"], "new_alarms": new,
            "in_alarm": {f: len(v["in_alarm"]) for f, v in result["families"].items()}}


def main() -> None:
    import argparse
    from bti.planning.history import load
    parser = argparse.ArgumentParser(description="Attack early warning")
    parser.add_argument("command", choices=["evaluate", "scan"])
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    history = load(args.data)
    alerts = _alerts_for(history)
    if args.command == "evaluate":
        r = evaluate(history, alerts)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(r, indent=2, default=str))
        print(json.dumps({k: r[k] for k in ("false_alarms", "injected_attacks")}, indent=1))
    else:
        r = monitor(history, alerts, arl_multiplier=load_calibration())
        print(json.dumps({f: v["in_alarm"] for f, v in r["families"].items()}, indent=1))


if __name__ == "__main__":
    main()
