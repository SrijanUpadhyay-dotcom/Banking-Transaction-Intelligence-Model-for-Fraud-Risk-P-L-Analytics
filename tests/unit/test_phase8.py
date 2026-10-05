# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 8: loss forecasting, staffing, attack early warning and the policy what-if simulator."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bti.planning import early_warning, forecast, staffing

CLEAN = Path("data/processed/banking_transactions_clean.csv")


def _synthetic_history(days=600, frauds_per_day=5.0, volume_per_day=200, delay_mean=None, seed=0,
                       countries=("A", "B")):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2023-01-01")
    for d in range(days):
        date = start + pd.Timedelta(days=d)
        n = rng.poisson(volume_per_day)
        fraud = rng.random(n) < frauds_per_day / volume_per_day
        for i in range(n):
            delay = None if delay_mean is None or not fraud[i] else int(rng.exponential(delay_mean))
            rows.append({"transaction_id": f"T{d}-{i}", "date": date, "country": countries[i % len(countries)],
                         "channel": "Mobile", "merchant_category": "Retail", "merchant_name": f"M{i % 10}",
                         "transaction_type": "Purchase", "customer_segment": "Retail", "fraud_type": "X" if fraud[i] else None,
                         "fraud": int(fraud[i]), "amount_usd": 100.0, "loss_usd": float(rng.lognormal(5, 1)) if fraud[i] else 0.0,
                         "confirmed_at": (date + pd.Timedelta(days=delay)) if delay is not None else pd.NaT,
                         "hour": int(rng.integers(0, 24))})
    return pd.DataFrame(rows)


def test_count_model_recovers_rate_and_weekly_pattern():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2023-01-01", periods=730, freq="D")
    rate = np.where(dates.dayofweek >= 5, 2.0, 6.0)
    m = forecast.CountModel().fit(dates, rng.poisson(rate).astype(float))
    future = pd.date_range("2025-01-01", periods=70, freq="D")
    sims = m.simulate(future, 3000, rng).mean(axis=0)
    weekend = future.dayofweek >= 5
    assert sims[weekend].mean() == pytest.approx(2.0, rel=0.15) and sims[~weekend].mean() == pytest.approx(6.0, rel=0.1)
    assert abs(m.describe()["trend_per_year_pct"]) < 10


def test_label_maturity_grosses_up_unconfirmed_recent_fraud():
    h = _synthetic_history(days=500, frauds_per_day=5.0, delay_mean=20)
    as_of = h["date"].max()
    f = forecast.completion_factors(h, as_of)
    assert f is not None and f[0] < f[30] < f[90] <= 1.0 and f[-1] == pytest.approx(1.0)
    known = forecast.known_as_of(h, as_of)
    recent = known[known["date"] > as_of - pd.Timedelta(days=14)]["fraud"].sum()
    assert recent < 0.7 * 5 * 14                                     # recent weeks look quieter than they are
    r = forecast.forecast(h, as_of, n_paths=2000, partitions=("country",))
    assert r["total"]["30d"]["frauds"]["p50"] == pytest.approx(150, rel=0.15)
    assert r["label_maturity"]["method"].startswith("completion")


def test_total_is_the_sum_of_country_paths_and_intervals_are_ordered():
    h = _synthetic_history(days=420, frauds_per_day=4.0)
    r = forecast.forecast(h, n_paths=2000, partitions=("country",))
    by = r["by_country"]
    assert r["total"]["30d"]["loss_usd"]["mean"] == pytest.approx(sum(v["30d"]["loss_usd"]["mean"] for v in by.values()),
                                                                  rel=0.01)
    t = r["total"]["90d"]["loss_usd"]
    assert t["p05"] <= t["p10"] <= t["p50"] <= t["p90"] <= t["p95"]


def test_severity_tail_extends_beyond_the_worst_loss_seen_but_is_capped():
    rng = np.random.default_rng(2)
    losses = np.concatenate([np.zeros(500), rng.pareto(1.5, 2000) * 1000])
    sev = forecast.Severity(losses)
    assert sev.threshold is not None
    draws = sev.extend(np.full(20000, losses.max()), rng)
    assert draws.max() > losses.max() and draws.max() <= sev.cap


def test_erlang_c_matches_the_textbook_value():
    assert staffing.erlang_c(3, 2.0) == pytest.approx(0.4444, abs=1e-3)
    assert staffing.agents_needed(0, 15, 60) == 0
    a, b = staffing.agents_needed(10, 15, 60), staffing.agents_needed(40, 15, 60)
    assert b > a >= 3 and staffing.service_level(b, 40, 15, 60) >= 0.9


def test_review_rate_proposal_warns_when_declines_fill_capacity():
    rates = {"rates_by_queue": {"urgent": 0.05, "high_value": 0.005, "standard": 0.005}, "provisional": True,
             "action_mix": {"DECLINE": 0.04}}
    aht = {q: {"minutes": 20.0} for q in rates["rates_by_queue"]}
    plan = {"transactions_per_day": {"p90": 30000}}
    small = staffing.recommend_review_rate(5, plan, rates, aht)
    assert small["proposed_max_review_rate"] == 0 and "Declines alone" in small["warning"]
    big = staffing.recommend_review_rate(400, plan, rates, aht)
    assert big["proposed_max_review_rate"] > 0 and big["warning"] is None and big["status"].startswith("proposal")


def test_cusum_threshold_rises_with_the_false_alarm_budget():
    h_short = early_warning.threshold(np.array([0.1, 1.0]), 300)
    h_long = early_warning.threshold(np.array([0.1, 1.0]), 3000)
    assert (h_long > h_short).all()


def test_early_warning_catches_a_doubling_and_stays_quiet_otherwise():
    h = _synthetic_history(days=400, frauds_per_day=3.0, volume_per_day=150, seed=4, countries=("A", "B", "C", "D"))
    alerts = pd.Series(np.zeros(len(h), int), index=h.index)
    quiet = early_warning.monitor(h, alerts, families=["merchant"])
    months = (400 - 97) / 30
    assert len(quiet["families"]["merchant"]["alarms"]) / months <= 2.0     # budget 1/month, Monte Carlo slack
    attack_start = h["date"].max() - pd.Timedelta(days=40)
    extra = h[(h["merchant_name"] == "M3") & (h["date"] >= attack_start)].copy()
    extra = extra.sample(len(extra) // 3, random_state=0)
    extra["fraud"], extra["transaction_id"] = 1, extra["transaction_id"] + "-x"
    attacked = pd.concat([h, extra], ignore_index=True)
    r = early_warning.monitor(attacked, None, families=["merchant"])
    hits = [e for e in r["families"]["merchant"]["alarms"] if e["segment"] == "M3"
            and pd.Timestamp(e["alarmed_on"]) >= attack_start]
    assert hits and hits[0]["ratio"] > 2


def test_whatif_rejects_unknown_inputs_and_an_empty_proposal_changes_nothing():
    from bti.modeling import registry
    from bti.planning.whatif import WhatIfError, simulate
    with pytest.raises(WhatIfError, match="unknown proposal keys"):
        simulate({"threshold": 0.5}, save=False)
    with pytest.raises(WhatIfError, match="cost-model fields"):
        simulate({"cost_overrides": {"bogus": 1}}, save=False)
    if not (registry.model_for_role("champion") or registry.model_for_role("challenger")) or not CLEAN.exists():
        pytest.skip("no registered model or processed data")
    same = simulate({}, n_boot=50, save=False)
    assert all(v["change"] == 0 for v in same["difference_with_95pct_ci"].values())
    stress = simulate({"cost_overrides": {"loss_given_fraud": 0.95}}, n_boot=50, save=False)
    assert stress["proposed_policy"]["residual_fraud_loss_usd"] > stress["current_policy"]["residual_fraud_loss_usd"]
    assert stress["status"].startswith("simulation only")
