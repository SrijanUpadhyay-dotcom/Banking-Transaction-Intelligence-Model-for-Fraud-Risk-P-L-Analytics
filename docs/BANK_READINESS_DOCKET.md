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
| 0.2 | Approve v3 as champion | **Pending** | Needs a named approver who is not the developer. Until then the model is provisional and cannot auto-decline on any endpoint. `python -m bti.modeling.promote --model bti-v3-xgb-rn-20260925003533 --role champion --approver "<name, role>" --rationale "<validation reference>"` |
| 0.3 | Scheduled PSI / CSI drift job with alerting | **Done** | Weekly (Mon 06:00 UTC, `BTI_DRIFT_CHECK_CRON`), live and shadow traffic, alerts via webhook/email at investigate/escalate, history in the audit log. `POST /governance/drift/run`, `GET /governance/drift/history`, `GET /governance/monitoring/schedule`. Enable the scheduler on one worker only. |
| 0.4 | Latency and uptime reporting | **Done** | `GET /operations/service-metrics`: uptime, requests, 5xx rate and p50/p95/p99 per endpoint group; model scoring latency against the SLA from the score log. |

Also fixed in Phase 0: the SAS enrichment documentation claimed a model "trained on 590K IEEE-CIS
transactions (ROC-AUC 0.908)" — untrue, now corrected; IEEE-CIS style batches (`TransactionID`,
`TransactionAmt`) were silently dropped from batch enrichment — now normalised.

## Phase 1 — Model strength

| Item | Status | Notes |
|---|---|---|
| LightGBM and XGBoost challengers | **Done** | `bti/modeling/algorithms.py`: histogram GBM, LightGBM, XGBoost behind one interface — same monotonic constraints, early stopping on the most recent 15% of the training window, refit to the best tree count so SHAP explains exactly what scores (additivity-checked). Pinned in `requirements.txt`. |
| Velocity features, round two | **Done** | Extended feature set: merchant velocity 1h/24h, device velocity 24h, amount z-score vs the customer's own look-back history, just-below-threshold amounts and their 7-day repeats, hour-of-day deviation from the customer's usual time. Point-in-time and lineage-checked. |
| Impossible travel, payee velocity, time since security event | **Built — awaiting feeds** | See *Feed-dependent signals* below. |
| Time-series hyperparameter search | **Done** | `bti/modeling/tuning.py`: grid search on three rolling out-of-time folds inside the training window. |
| Challenger tournament | **Done** | `python -m bti.modeling.tournament`: trains candidates on identical data, registers all of them, selects on the calibration window (never the out-of-time test), requires beating the incumbent by 0.005 PR-AUC. |
| v3 training in the pipeline orchestrator | **Done** | The pipeline runs a tournament when the training data changes, rescores stored history, and dispatches alerts on v3 tiers. |
| Rescore stored history | **Done** | The exception queue, transaction filters and customer-risk analytics read stored scores that were still the legacy composite. `python -m bti.modeling.rescore` rewrites them with the scoring v3 model; legacy values remain in the ML-scored CSV. |
| Feature definitions versioned; novelty fix (v2) | **Done** | v1 treated "no prior history" as "new device", raising false positives for thin-history and new-to-bank customers. v2 marks novelty unknown without history. Every model records its feature version and is always scored with it (incumbent scores verified bit-identical). |
| Relative-amount feature sets | **Done** | `core-relative` / `extended-relative` drop absolute USD amount and balance, which track customer wealth (and so country and segment). Amounts are judged against the customer's own history; loss size is priced by the decision layer. |
| Fairness method corrected | **Done** | See *Fairness: the German finding* below. |
| Private Banking remediation | **Done** | See *Fairness: Private Banking* below. `balance_share_vs_own` added: amount-to-balance judged against the customer's own habit. |
| Verified remediation | **Done** | `--remediation-groups customer_segment=Corporate`: a replacement must carry no finding for the named groups and a lower worst-case ratio than the incumbent. |
| Post-registration notes | **Done** | Registered cards never change; re-assessments and corrections are appended to the registry and shown in the documentation pack. `python -m bti.modeling.reassess`. |

**Tournament results (synthetic data):**

1. *Tuned tournament, v1 features* — six candidates (histogram GBM, LightGBM, XGBoost × core, extended). None beat the
   incumbent by the 0.005 PR-AUC margin; four of six failed the fairness gate at the 10% stress budget. Tuning moved
   PR-AUC by ~0.002 while fold-to-fold variation was ±0.022 — gains within noise on this data.
2. *Remediation tournament, v2 features* — all six candidates passed every gate. `bti-v3-lgbm-20260923180905`
   (LightGBM, core, v2) replaced the incumbent as challenger under non-inferiority (calibration PR-AUC 0.8524 vs
   0.8507). Out-of-time: ROC-AUC 0.9551, PR-AUC 0.8856 — level with the incumbent (0.9581 / 0.8866) within noise.
3. *Remediation tournament, relative-amount features* — `bti-v3-xgb-r-20260923181450` (XGBoost, core-relative)
   took the challenger role: calibration PR-AUC 0.8519 vs 0.8524 (non-inferior), out-of-time ROC-AUC 0.9556,
   PR-AUC 0.8857. Removing absolute amounts cost no accuracy.
4. *Remediation tournament, Private Banking* — all six candidates passed every gate and the remediation check.
   **`bti-v3-xgb-rn-20260925003533` (XGBoost, core-relative without `amount_to_balance`) is the current challenger**:
   calibration PR-AUC 0.8518 vs 0.8519, out-of-time ROC-AUC 0.9563, PR-AUC 0.8857, ECE 0.003.

Neither the algorithm swap nor the extended features materially improved accuracy on the synthetic data, whose fraud
is generated from a few simple signals. Both are in place to be re-tested on bank data, where they are more likely
to matter.

### Fairness: the German finding (resolved)

**What was reported.** Every model flagged legitimate German customers about 1.23× the overall rate at the 10% stress
budget (p≈0.035). Tournament 3 was run to remediate it.

**What it was.** A false positive from multiple testing. The fairness test compared ~21 groups at three operating
points at α=0.05 with no correction, so a few "significant" groups were expected by chance alone. The German
disparity appeared only in the out-of-time window: 0.92–0.99× in the calibration window, 1.10× pooled, never
significant after correction. No model input differs for German customers (smallest KS p=0.36), and a permutation
test produced a ratio that large 23% of the time. The Student finding that drove the earlier monotone fix had the same
shape (1.34× out-of-time, 0.97× calibration, pooled q=0.44). Its constraints are kept on their own merits.

**The corrected method** (`bti/governance/fairness.py::fairness_assessment`). The calibration and out-of-time
windows are pooled, the p-values are Benjamini–Hochberg corrected across groups, and a group is a finding only when
three conditions all hold:

- the pooled FPR ratio is above 1.25
- q < 0.05
- the group's FPR is above the overall rate in both windows

A signal that shows up in only one window goes on a non-gating **watchlist**, which production monitoring re-tests.

**What the corrected method found instead.** Re-assessing all 21 registered models found a real, replicated
disparity: the absolute-amount models over-flag **Corporate and Private Banking** customers. The previous challenger
`lgbm-20260923180905` shows 1.27× and 1.32× (q=0.03), elevated in both windows. The mechanism is absolute USD amount
acting as a wealth proxy, which is the mechanism tournament 3 removed. The current challenger passes; Private Banking
was left on the watchlist (pooled 1.29×, q=0.13) and has since been remediated (below). The corrections are recorded as append-only registry notes on every
model. Six verdicts moved from review_required to pass, and two from pass to review_required.

### Fairness: Private Banking (remediated)

Private Banking was on the watchlist: 1.29× pooled at the 10% stress budget, above the norm in both windows, but
q=0.13 after correcting for 21 groups. That is plausibly real, but with ~1,000 legitimate customers it can't be
proven. A SHAP gap analysis found the cause:

- **One feature explains most of the gap.** `amount_to_balance` explains +0.21 of the +0.28 log-odds gap between
  legitimate Private Banking customers and everyone else.
- **Why the feature separates them.** A typical Private Banking payment is 18.7% of the balance, against 0.8% for
  other customers. The model learned that payments draining a large share of the balance look like account
  takeover.
- **How much is artefact.** The synthetic data scales Private Banking amounts but not their balances, so the size
  of the gap is partly an artefact of the generator.
- **Why it still matters.** The mechanism is real. On a real book it would hit any customer whose normal payments
  are a large share of their balance, such as students or people who spend their salary down.

Tournament 4 tested two fixes: dropping `amount_to_balance`, and replacing it with `balance_share_vs_own` (the same
ratio relative to the customer's own average). Each fix was tried with all three algorithms. Every candidate
lowered the Private Banking worst-case ratio, from 1.29× to between 0.99× and 1.16×, at no accuracy cost. The winner
was chosen by the standing rule (calibration PR-AUC within 0.005 of the incumbent, all gates, verified
improvement). It is the XGBoost model without the ratio: Private Banking 1.13× / 0.78× pooled at the 10% / 5%
budgets, and it is off the watchlist.

**Residual and trade-off:**

- **Residual.** Private Banking is still 1.35× in the calibration window alone at 10%. Re-test it on real labels.
- **Watchlist now.** The watchlist holds only single-window signals: Premium (calibration 1.43×, pooled 1.22×,
  q=0.93) and Nigeria (calibration 1.53×, pooled 1.10×).
- **Trade-off.** Without the raw ratio, a first-ever transaction that drains an account is no longer flagged by
  that feature. On bank data, re-test account takeover of new customers with `core-relative-own` as the
  alternative.

**Still true for Germany.** Real German customer data is needed before a German deployment. That is the bank data
onboarding track, not a model defect.

### Feed-dependent signals (built; activate when the bank supplies the feed)

The synthetic dataset cannot support these features. It picks `city` at random for each transaction, the IP
addresses are random, and it has no payees or security events. Impossible travel computed from those fields would
fire on noise. So the signals are built end to end, but they sit in no default feature set, and training refuses
them (`FeedUnavailableError`) unless the feed covers at least 1% of every window of the split.

| Feed | Fields / endpoint | Features (reason code) |
|---|---|---|
| Location | `latitude`, `longitude` on the transaction (terminal, device GPS, or IP geolocation resolved upstream) | `geo_distance_prev_km`, `geo_speed_kmh`, `impossible_travel` (>900 km/h over >300 km) — GEO_VELOCITY |
| Payee | `payee_id` on the transaction (tokenised beneficiary key) | `payee_new_for_customer`, `cust_new_payees_24h` — NEW_PAYEE; `payee_other_customer_txns_7d` — PAYEE_VELOCITY (mule pattern) |
| Security events | `POST /v3/security-events` (password reset, SIM swap / number port, contact change, device enrolment); training reads `security_events.csv` | `hours_since_security_event`, `hours_since_sim_swap`, `security_events_7d` — SECURITY_EVENT |

- **Point-in-time and live/batch parity.** All three are point-in-time, since only events strictly before the
  transaction count. The live scorer fetches payee history and security events from the database, and live and
  batch scores match, which a test trains a model on a labelled test feed to verify.
- **Schema.** The new columns are added to existing databases by an additive migration.
- **Checking feed status.** `GET /v3/feeds` shows which feeds are populated and whether the scoring model uses
  them.
- **Enabling a feed.** Once one flows, run `python -m bti.modeling.tournament --feature-sets extended-relative
  signals-relative`. The signals enter a model only if they earn their place against the incumbent.
- **What is not claimed.** No lift is claimed for these signals until they are measured on bank data.

## Phase 2 — Run alongside SAS

| Item | Status | Notes |
|---|---|---|
| SAS score and decision ingestion | **Done** | `POST /parallel/incumbent/decisions` (JSON), `POST /parallel/incumbent/upload` (CSV / Excel / Parquet with a column map), `python -m bti.parallel.incumbent --file …`. Vendor codes are mapped to APPROVE / STEP_UP / REVIEW / DECLINE: common codes are built in, and the bank adds its own under `parallel_run.decision_map`. Unknown codes are rejected row by row, never guessed. Records are append-only and the latest per transaction wins. `GET /parallel/incumbent/status` shows how well the two streams pair. |
| Automated parallel-run report | **Done** | `GET /parallel/report` and a weekly job (Mon 07:00 UTC) that stores to the audit log and alerts. See *How lift is measured* below. |
| Randomised traffic splitter | **Done — blocked on champion approval** | See *Traffic split* below. An experiment can be proposed today but cannot start: live customer decisions are never handed to a provisional model. |
| Fallback and reconciliation | **Done** | See *Fallback and reconciliation* below. |
| Capacity-constrained live decisions | **Done** | Found by the rehearsal: live decisions ignored analyst capacity and sent 20.6% of traffic to review. Capacity prices (review, step-up) are now fitted per model on the calibration window, stored with history, and used for every live and shadow decision. `python -m bti.operations.capacity`, `GET /operations/policy/capacity`. The current challenger holds review at 2.0% and step-up at 5.0% (out-of-time: 1.9% / 5.7%). While the model is provisional its declines become reviews, adding about 4% of traffic to review. |

**How lift is measured.** The report compares BTI with SAS on the same transactions.

- **The headline: equal intervention rate.** BTI's riskiest transactions, taken in the same number SAS
  intervened on, are compared with SAS's interventions on fraud caught (count and value) and on genuine
  customers disturbed. This removes the easy win of intervening more.
- **Significance.** McNemar's test on the frauds one system caught and the other missed, plus bootstrap 95%
  intervals on the detection-rate difference.
- **Verdict.** "BTI detects more" requires p < 0.05 and an interval above zero. At least 30 matured frauds
  are required first.
- **Also reported.** Decision agreement matrix, action rates, both systems at their actual decisions (false
  declines, hit rate), and latency for both.
- **Label maturity.** Only labels past the 90-day maturity window count.
- **Weekly views.** The job produces the last 7 days (operations), the cohort that matured this week
  (outcomes), and the total to date.
- **Alerts.** It alerts when SAS detects significantly more, when pairing falls below 95%, or when BTI's p99
  latency breaks the SLA.

**Traffic split.**

- **Assignment.** A salted hash of the customer puts each customer in the BTI arm with probability
  `bti_share`. It is deterministic and reproducible, and each customer always gets the same system.
- **Governance.**
  - One person proposes and a different approver starts it (four-eyes).
  - The share is capped at 10% (`parallel_run.max_bti_share`).
  - An approved champion must exist.
  - Only one experiment runs at a time, and stopping is immediate.
- **Analysis.** `GET /parallel/experiments/{id}/report` compares arms on matured outcomes: fraud loss in
  basis points of value, detection, false declines, and customer friction.
- **Statistics.**
  - Confidence intervals come from a bootstrap that resamples customers, because the customer is the unit of
    randomisation.
  - A sample-ratio-mismatch check invalidates a broken split.
  - The minimum detectable difference is reported.

**Fallback and reconciliation.** The bank's switch calls `POST /parallel/decide` with the transaction and
SAS's decision.

- **Routing.** BTI scores every call within a time budget (`parallel_run.bti_timeout_ms`, default 150 ms). In
  the BTI arm BTI's decision applies; otherwise SAS's does.
- **Fallback.** On a timeout or error, SAS's decision applies and the fallback is logged. Measured router
  latency on this container: p50 85 ms, p99 124 ms.
- **Daily reconciliation** (02:30 UTC; `POST /parallel/reconcile/run`) flags five kinds of break:
  - transactions routed but missing from the SAS feed
  - transactions SAS decided that BTI never scored
  - duplicate routings
  - a decision sent with the request that differs from SAS's own log
  - enforcement mismatches, where the bank executed a decision no system chose (critical)

  The fallback rate is reported alongside.

**Rehearsal on synthetic data (not SAS).** `python -m bti.parallel.simulate` runs the full report with a
simple pre-authorisation rules engine standing in for the incumbent. Its thresholds were fixed before any
results. At equal intervention (1,491 transactions each):

| | Detection rate (TDR) | Value detection rate (VDR) | Genuine customers disturbed |
|---|---|---|---|
| BTI | 0.904 | 0.989 | 929 |
| Rules stand-in | 0.770 | 0.935 | 1,012 |

McNemar p < 0.001. This proves the machinery, not lift. BTI was trained on the same generator, and a
five-rule stand-in is not SAS.

**What the bank pilot needs:**

- two to four weeks of SAS decisions (with executed decisions, if the switch logs them) alongside BTI
  scoring of the same traffic
- confirmed outcomes 90 days on
- an approved champion before any traffic split

## Phase 3 — SR 11-7 completion

| Item | Status | Scope |
|---|---|---|
| Documentation pack | Done | `GET /governance/models/{id}/documentation`. |
| Benchmarking and sensitivity analysis | New | Challenger comparison, perturbation and stress tests in the pack. |
| Outcomes analysis on matured labels | Extend | Add live calibration and the pooled fairness assessment on matured labels, quarterly; re-test watchlist groups (Private Banking) and, before a German deployment, German customers. |
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
