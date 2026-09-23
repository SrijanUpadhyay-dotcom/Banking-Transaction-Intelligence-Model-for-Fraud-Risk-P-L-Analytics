# BTI Bank Readiness — Build Docket

Phased plan for deploying BTI in banks and multinationals, alongside and eventually
independent of an incumbent fraud platform such as SAS. Phases are in priority order.
Status: **Done** (in the codebase) · **In progress** · **Pending** (needs an external input) ·
**Extend** (partly built) · **New**.

All model results so far are on synthetic data and must be re-established on each bank's own
labelled history. Regulatory references are a starting map for compliance review, not legal advice.

**Running track (not code):** bank data onboarding — 12–24 months of transactions, confirmed fraud
outcomes with dates, and the incumbent's scores and decisions for the same transactions. It is the
only way to establish real lift over SAS.

---

## Phase 0 — Close out what is open

| # | Item | Status | Notes |
|---|---|---|---|
| 0.1 | Move `/score`, `/score/explain`, `/score/upload` and `/sas/enrich*` onto the v3 model | **Done** | One shared path (`bti/operations/scoring_service.py`): v3 score → expected-cost decision → shadow challenger → score log. Response shapes kept; `ml_lr_proba`, `ml_rf_proba`, `ml_iso_score` are now null; `risk_score` and post-event fields are accepted but ignored. |
| 0.2 | Approve v3 as champion | **Pending** | Needs a named approver who is not the developer. Until then the model is provisional and cannot auto-decline on any endpoint. `python -m bti.modeling.promote --model bti-v3-lgbm-20260923180905 --role champion --approver "<name, role>" --rationale "<validation reference>"` |
| 0.3 | Scheduled PSI / CSI drift job with alerting | **Done** | Weekly (Mon 06:00 UTC, `BTI_DRIFT_CHECK_CRON`), live and shadow traffic, alerts via webhook/email at investigate/escalate, history in the audit log. `POST /governance/drift/run`, `GET /governance/drift/history`, `GET /governance/monitoring/schedule`. Enable the scheduler on one worker only. |
| 0.4 | Latency and uptime reporting | **Done** | `GET /operations/service-metrics`: uptime, requests, 5xx rate and p50/p95/p99 per endpoint group; model scoring latency against the SLA from the score log. |

Also fixed in Phase 0: the SAS enrichment documentation claimed a model "trained on 590K IEEE-CIS
transactions (ROC-AUC 0.908)" — untrue, now corrected; IEEE-CIS style batches (`TransactionID`,
`TransactionAmt`) were silently dropped from batch enrichment — now normalised.

## Phase 1 — Model strength

| Item | Status | Notes |
|---|---|---|
| LightGBM and XGBoost challengers | **Done** | `bti/modeling/algorithms.py`: histogram GBM, LightGBM, XGBoost behind one interface — same monotonic constraints, early stopping on the most recent 15% of the training window, refit to the best tree count so SHAP explains exactly what scores (additivity-checked). Pinned in `requirements.txt`. |
| Velocity features, round two | **Done** | Extended feature set: merchant velocity 1h/24h, device velocity 24h, amount z-score vs the customer's own look-back history, just-below-threshold amounts and their 7-day repeats, hour-of-day deviation from the customer's usual time. Point-in-time and lineage-checked. Payee velocity and time since profile change need payee / security-event feeds the dataset does not have. |
| Impossible travel / geo-velocity | **Pending** | Needs a licensed IP-geolocation feed. |
| Time-series hyperparameter search | **Done** | `bti/modeling/tuning.py`: grid search on three rolling out-of-time folds inside the training window. |
| Challenger tournament | **Done** | `python -m bti.modeling.tournament`: trains candidates on identical data, registers all of them, selects on the calibration window (never the out-of-time test), requires beating the incumbent by 0.005 PR-AUC. |
| v3 training in the pipeline orchestrator | **Done** | The pipeline runs a tournament when the training data changes, rescores stored history, and dispatches alerts on v3 tiers. |
| Rescore stored history | **Done** | The exception queue, transaction filters and customer-risk analytics read stored scores that were still the legacy composite. `python -m bti.modeling.rescore` rewrites them with the scoring v3 model; legacy values remain in the ML-scored CSV. |

| Feature definitions versioned; novelty fix (v2) | **Done** | v1 treated "no prior history" as "new device", raising false positives for thin-history and new-to-bank customers. v2 marks novelty unknown without history. Every model records its feature version and is always scored with it (incumbent scores verified bit-identical). |

**Tournament results (synthetic data):**

1. *Tuned tournament, v1 features* — six candidates (histogram GBM, LightGBM, XGBoost × core, extended). None beat the
   incumbent by the 0.005 PR-AUC margin; four of six failed the fairness gate at the 10% stress budget. Tuning moved
   PR-AUC by ~0.002 while fold-to-fold variation was ±0.022 — gains within noise on this data.
2. *Remediation tournament, v2 features* — all six candidates passed every gate. **`bti-v3-lgbm-20260923180905`
   (LightGBM, core, v2) replaced the incumbent as challenger** under non-inferiority (calibration PR-AUC 0.8524 vs
   0.8507). Out-of-time: ROC-AUC 0.9551, PR-AUC 0.8856 — level with the incumbent (0.9581 / 0.8866) within noise.
   This was a fairness remediation, not an accuracy gain.

Neither the algorithm swap nor the extended features materially improved accuracy on the synthetic data, whose fraud
is generated from a few simple signals. Both are in place to be re-tested on bank data, where they are more likely
to matter.

**Open finding — Germany:** legitimate German customers are flagged ~1.23× the overall rate at the 10% stress budget
in every model trained (current challenger 1.228×, p=0.035) — statistically significant but below the 1.25
materiality threshold, so the gate passes. The v2 fix removed the larger disparities (Singapore 1.58–1.62×, Germany
1.54×, Private Banking 1.28×) but not this residual. Monitor quarterly; re-test on real German data before any
German deployment.

## Phase 2 — Run alongside SAS

| Item | Status | Scope |
|---|---|---|
| SAS score and decision ingestion | New | Batch and API adapters keyed on transaction ID. |
| Automated parallel-run report | New | Weekly incremental value detection at equal intervention rate, false declines, latency, drift. |
| Randomised traffic splitter | New | 5–10% of decisions to BTI with a significance test. |
| Fallback and reconciliation | New | SAS answers on BTI timeout; daily reconciliation of both decision logs. |

## Phase 3 — SR 11-7 completion

| Item | Status | Scope |
|---|---|---|
| Documentation pack | Done | `GET /governance/models/{id}/documentation`. |
| Benchmarking and sensitivity analysis | New | Challenger comparison, perturbation and stress tests in the pack. |
| Outcomes analysis on matured labels | Extend | Add live calibration and fairness, quarterly. Track the open Germany finding (Phase 1). |
| Validation workflow | New | Findings tracker, sign-off records, annual review schedule, model inventory in the database. |
| Immutable audit storage | Extend | Append-only (WORM) storage and retention policy. |

## Phase 4 — Feedback loop and continuous learning

| Item | Status | Scope |
|---|---|---|
| Confirmed-label feedback loop | Done | `POST /operations/labels`, 90-day maturity rule. |
| Case management feeding labels | New | Queues with SLAs; dispositions become labels automatically. |
| Continuous retraining | New | Scheduled and drift-triggered; new models auto-register as shadow challengers, promotion stays human. Per-event weight updates are not recommended — each is a model change a validator must approve. |
| Bounded daily recalibration | New | Refit calibration on matured labels within limits, logged as a minor change. |

## Phase 5 — Decision economics and rules

| Item | Status | Scope |
|---|---|---|
| False-positive cost model | Done | Expected-cost decisions, capacity and challenge budgets, guardrails. |
| Cost model v2 | Extend | Per-customer value from BTI's P&L data, measured step-up abandonment, segment friction. |
| Analyst rules engine | New | Versioned rules, simulation on history, champion/challenger rules. |
| Step-up orchestration | New | SMS / push / 3-D Secure adapters. |

## Phase 6 — Graph intelligence (GNN)

Needs PyTorch, which cannot be installed in the current environment — requires a network-policy
change or a separate training environment.

| Item | Status | Scope |
|---|---|---|
| Temporal entity graph | New | Point-in-time, so embeddings never see future links. |
| Graph embeddings without deep learning | New | Spectral / random-walk embeddings first. |
| GraphSAGE / temporal GNN | New | Inductive; mules and rings. |
| Synthetic ring generator | New | Current data has almost no shared devices. |

## Phase 7 — Streaming and scale

| Item | Status | Scope |
|---|---|---|
| Kafka ingestion with ISO 8583 / ISO 20022 adapters | New | |
| Online feature store | New | Redis, with parity tests against training features. |
| Inline latency | Extend | 70 ms p50 / 78 ms p95 with explanations today; target p99 < 50 ms, explanations async. |
| High availability and residency | New | Failover, DR targets, India in-country deployment. |

## Phase 8 — Forecasting and planning

| Item | Status | Scope |
|---|---|---|
| Fraud-loss forecasting | New | 30/60/90-day forecasts with intervals, by jurisdiction and channel. |
| Alert-volume and staffing forecast | New | Feeds rostering and the review-capacity setting. |
| Attack early warning | New | Change-point detection by typology, merchant, corridor. |
| Policy what-if simulator | New | Loss and friction impact of a policy change before making it. |

## Phase 9 — Scam and mule models

| Item | Status | Scope |
|---|---|---|
| Payee risk and APP-scam model | New | New payee, payee account age, Confirmation-of-Payee mismatch; UK reimbursement exposure. |
| Mule-account detection | New | Inbound-flow patterns plus Phase 6 graph signals. |

## Phase 10 — Certification

| Item | Status | Scope |
|---|---|---|
| SOC 2 Type II, ISO 27001, PCI DSS scope reduction, penetration test | New | Procurement gates. |

---

**On differentiation:** SAS sells network analytics, model management, rules and case management
as licensed modules. BTI's defensible differences are a transparent model with enforced lineage
and exact per-decision reasons, amount-aware expected-cost decisions fitted to each bank's
capacity, one policy engine across eight jurisdictions, the investigator copilot, and forecasting
tied to the decision engine. Check each bank's licence before claiming a specific gap.
