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
| 0.2 | Approve v3 as champion | **Pending** | Needs a named approver who is not the developer. Until then the model is provisional and cannot auto-decline on any endpoint. `python -m bti.modeling.promote --model bti-v3-lgbm-rn-20260925011359 --role champion --approver "<name, role>" --rationale "<validation reference>"` |
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
   `bti-v3-xgb-rn-20260925003533` (XGBoost, core-relative without `amount_to_balance`) took the challenger role:
   calibration PR-AUC 0.8518 vs 0.8519, out-of-time ROC-AUC 0.9563, PR-AUC 0.8857, ECE 0.003.
5. *Remediation tournament, missing inputs* (Phase 2 smoke run) — all six candidates, trained with missingness
   augmentation, passed every gate including the new missing-input gate. **`bti-v3-lgbm-rn-20260925011359`
   (LightGBM, core-relative without `amount_to_balance`) is the current challenger**: calibration PR-AUC 0.8506,
   out-of-time ROC-AUC 0.9557, PR-AUC 0.8845, ECE 0.003. Private Banking 1.13× / 0.93× pooled; fairness passes.

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
| Missing inputs handled safely | **Done** | Found in a live run of the API. `amount_vs_hist_avg` and `login_attempts` had never been missing in training, so a transaction without the customer's profile average or login count scored 0.97, and 100% of a genuine sample crossed p = 0.5. The fixes: models now train with single-feature missingness augmentation; a new `missing_input_robustness` gate fails any model where one missing input pushes more than 1% of genuine customers to p ≥ 0.5 (the new challenger's worst is 0.05%); and the API lists the inputs it scored as unknown. Models are warmed up at start-up: the cold first call took 437 ms and timed out to SAS. Live re-run: p99 51 ms, no fallbacks, sparse transactions approved at p 0.02. |
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

**Running state.** BTI is a service the bank runs. When the API process starts, it warms the models and its scheduler holds three jobs: weekly drift, the weekly parallel-run report, and daily reconciliation. On 2026-09-25 a live run on a scratch database exercised every endpoint: SAS feed, router, pairing status, report, weekly job, experiment (correctly refused without a champion), reconciliation, capacity policy and authentication. It surfaced the two defects above, both fixed and re-verified. Nothing runs permanently in this development environment; deployment on the bank's infrastructure is the pilot step.

**What the bank pilot needs:**

- two to four weeks of SAS decisions (with executed decisions, if the switch logs them) alongside BTI
  scoring of the same traffic
- confirmed outcomes 90 days on
- an approved champion before any traffic split

## Phase 3 — SR 11-7 completion

| Item | Status | Notes |
|---|---|---|
| Documentation pack | **Done** | `GET /governance/models/{id}/documentation`. New sections: 13 benchmarking, sensitivity and stress; 14 outcomes on matured labels; 15 independent validation (readiness, sign-offs, findings). |
| Benchmarking and sensitivity analysis | **Done** | `python -m bti.governance.benchmarking`, `POST/GET /governance/models/{id}/benchmark`. All on the out-of-time window. Stored under `models/registry/validation/<model>/`. See the results below. |
| Outcomes analysis on matured labels | **Done** | Quarterly job (1 Jan / Apr / Jul / Oct), `GET /governance/outcomes`, `POST /governance/outcomes/run`. See *Outcomes analysis* below. |
| Validation workflow | **Done** | Model inventory, findings tracker, independent sign-offs, review schedule and the champion gate. See *Validation workflow* below. |
| Immutable audit storage | **Done** | Hash chain, database guards, sealed daily archive and retention. See *Audit storage* below. |

**Benchmarking, sensitivity and stress testing.** The report covers:

- a logistic regression on the same features
- all alternatives registered on the same data
- the rules stand-in
- bootstrap intervals and month-by-month stability
- ±1 SD sensitivity with decision flips
- monotonicity sweeps
- seven stress scenarios: inflation / FX, spending spike, missing profile data, new-to-bank surge, channel shift,
  velocity surge, and a doubled fraud prior

It proposes findings from its own evidence.

**Outcomes analysis.** Live discrimination and calibration are compared with development: a decile table, ECE,
and calibration-in-the-large.

- **Fairness.** The pooled, corrected fairness test runs with the two halves of the quarter as the two windows,
  and development-watchlist groups are re-tested.
- **Data it relies on.** Segment, age band and country are recorded with each score for this purpose only; they
  are never model inputs.
- **Findings.** Degradation raises findings automatically, without duplicating an open one.

**Validation workflow.**

- **Tables.** `model_inventory` (synced from the registry, with lifecycle and next review), `validation_findings`
  and `validation_signoffs`.
- **Separation of roles.**
  - The developer cannot sign off their own model.
  - A finding's owner cannot close it, and closing needs evidence.
  - Risk acceptance needs someone other than both the owner and the developer. For high severity it is limited
    to 180 days.
- **Approval.** An approval is blocked while a high-severity finding is open, or if automated gates failed.
- **Periodic review.** A sign-off is valid for a year (Tier 1). A weekly governance check alerts on reviews due
  within 30 days or overdue, and on findings past their due date.
- **Champion gate.** `POST /governance/models/{id}/promote` and `python -m bti.modeling.promote` require all
  of the following:
  - passed automated gates
  - an in-date approving sign-off from a validator other than the developer
  - no open high-severity finding
  - an approver other than the developer

  `GET /governance/models/{id}/readiness` shows what is missing.
- **Seeding.** `python -m bti.governance.validation seed` raises the developer's known limitations plus the
  benchmark's proposed findings.

**Audit storage.**

- **Hash chain.** Every `audit_logs` row carries a sequence number and a SHA-256 over its content and the
  previous row's hash, assigned under a lock on the chain head.
- **Database guards.** Triggers on SQLite and PostgreSQL reject UPDATE and DELETE on sealed rows. Rows from
  before the chain are sealed once at start-up.
- **Verification.** `GET /governance/audit/verify` pinpoints the first altered, removed or reordered row, or a
  truncated tail. In testing it caught an edit made after the trigger had been dropped by a DBA-level user.
- **Daily archive.** Sealed daily segments are written as read-only JSONL plus a manifest (file SHA-256 and chain
  position); the job runs at 03:00 UTC. `GET /governance/audit/archive/{day}/verify` re-checks a segment
  against the database.
- **Retention.** Seven years (`audit.retention_days`, confirm with Compliance). Expired segments are reported,
  never deleted by code.
- **Production.** Put `audit.archive_dir` on write-once storage (S3 Object Lock in compliance mode, Azure
  immutable blobs, or a WORM appliance). A local read-only directory is tamper-evident, not tamper-proof.

**Validation results for the current challenger** (`bti-v3-lgbm-rn-20260925011359`, synthetic data,
out-of-time):

| Benchmark | ROC-AUC | PR-AUC |
|---|---|---|
| This model | 0.9557 | 0.8845 (95% CI 0.861–0.906) |
| Logistic regression, same features | 0.9532 | 0.8801 |
| Rules stand-in | 0.9021 | 0.6863 |

- **Monotonicity:** holds for every constrained feature.
- **Stress:** only missing profile data degrades performance. Detection falls from 44% to 30% at the 2% budget.
  Alerts drop rather than rise, so the failure is safe.
- **Findings raised:** 11 in the tracker.
  - **High (1):** synthetic-data-only evidence.
  - **Medium (6):** new-customer balance-draining gap; German data; illustrative cost figures; decisions
    concentrated on `amount_vs_hist_avg` (+1 SD flips 78% of decisions at the 10% budget); velocity features
    carrying no weight (a card-testing burst changes no decision); missing profile data.
  - **Low (4):** Private Banking residual; unvalidated feeds; gain over logistic regression within noise;
    probabilities not following a prior shift.
- **What this means for approval.** The high finding blocks champion approval, as it should. Closing it needs
  bank data. The alternative is a time-limited risk acceptance by someone other than the developer, for example a
  model risk committee accepting a limited pilot.

## Phase 4 — Feedback loop and continuous learning

| Item | Status | Notes |
|---|---|---|
| Confirmed-label feedback loop | **Done** | `POST /operations/labels`, 90-day maturity rule. |
| Case management feeding labels | **Done** | `/api/v1/cases`. See *Case management* below. |
| Analyst case workbench | **Done** | `GET /workbench`, served by the BTI API itself: same origin, no third-party requests. See *Case workbench* below. |
| Continuous retraining | **Done** | `python -m bti.modeling.retrain`, `GET /operations/retraining/triggers`, `POST /operations/retraining/run`. See *Continuous retraining* below. |
| Bounded daily recalibration | **Done** | `POST /operations/recalibration/run`, `/rollback`; daily job. See *Recalibration* below. |

**Case management.**

- **Which transactions open a case.** Every live REVIEW or DECLINE decision opens one, once per transaction, as
  do manual referrals such as a customer asking for human review of a decline. Declines must open cases: no money
  moves, so no chargeback will ever label them, and without a disposition the model learns nothing from its own
  declines.
- **Queues (`cases.queues`).**
  - `urgent`: probability ≥ 0.8 or expected loss ≥ $1,000; 60-minute SLA.
  - `high_value`: amount ≥ $10,000; 120-minute SLA.
  - `standard`: everything else; 8-hour SLA.

  Analysts pull the next case by queue, then expected loss, then age.
- **Dispositions become labels automatically.** `confirmed_fraud` records INVESTIGATOR_CONFIRMED and
  `confirmed_genuine` records INVESTIGATOR_CLEARED. Inconclusive and unreachable write no label and are left to
  the maturity rule. A later chargeback still overrides.
- **Maker-checker.** Clearing a case of $10,000 or more as genuine needs a second reviewer.
- **SLA and metrics.** An SLA check runs every 15 minutes and alerts once per breached case.
  `GET /cases/queues` reports open, breached, time to close, SLA attainment and fraud-confirmation rate per queue.

**Case workbench** (`/workbench`).

- **Queue health.** Open, unassigned and breached cases per queue, plus SLA attainment over 7 days.
- **Case list.** Live cases in urgency order, each with a ticking SLA clock (amber under 25% of the time left, red
  when breached). It filters by "mine" and by queue, and a button takes the next case.
- **Case detail.**
  - probability, amount and expected loss
  - reason codes in analyst language
  - decision guardrails
  - the customer's recent scored activity, with confirmed outcomes and earlier cases (`GET /cases/{id}/context`)
- **Actions.** Assign to me, "waiting for customer", and closing a case with an outcome. Fraud type and loss are
  asked for on confirmed fraud. The second-reviewer field appears when clearing $10,000 or more.
- **Safety.**
  - All data is rendered as text, never HTML.
  - The analyst ID and API key are kept in session storage only.
  - In production, put the page behind the bank's SSO and replace the key entry with the SSO identity.
- **Verified.** In a headless-browser run, "take next" handed over the most urgent case. Confirming fraud wrote a
  label. Clearing a $12,800 case was refused without a second reviewer, then accepted with one. No page errors in
  light or dark mode. A test syntax-checks the page's script on every run.
- **Relation to the old queue.** The older `/alerts` exception queue remains for the batch history only. The
  workbench is the live queue.

**Continuous retraining.**

- **When it runs.** A daily check retrains at most every 7 days, and only on one of these triggers:
  - the schedule (30 days)
  - an escalated drift check
  - a performance finding from outcomes analysis
  - 500 or more new confirmed labels
- **The extract.** It is the bank's latest extract with confirmed labels applied (the latest wins).
  Transactions younger than the 90-day maturity window are dropped unless explicitly labelled, so unreported
  fraud is not learned as genuine.
- **Selection.** The tournament now re-scores the incumbent on the new split, so every entrant is judged on the
  same rows. Before this fix, a data change compared candidates with the incumbent's old figures.
- **What a winner becomes.** A shadow challenger only. Champion promotion stays behind independent sign-off and
  four-eyes approval. Per-event online weight updates are deliberately not done.
- **Settings.** `modeling.feature_sets` now defaults to the relative-amount sets, because the absolute-amount
  sets failed fairness.

**Recalibration.**

- **The overlay.** A rank-preserving overlay p′ = sigmoid(α·logit(p) + β) on the model's own probability is fitted
  on matured labels. Ranking, reason codes and SHAP are unchanged.
- **Acceptance.** A candidate is accepted only if all of these hold:
  - it has at least 100 frauds and 1,000 genuine transactions
  - it lowers holdout ECE against both the current overlay and none
  - α is in [0.67, 1.5] and |β| ≤ 1: a bigger shift is a retraining problem
  - it changes at most 5% of decisions at the decline line and the reference threshold
- **Record keeping.** Overlays are versioned per model and applied by the scorer. Each score logs the model's
  pre-overlay probability and the overlay version, so a refit never compounds. Each change is logged as a minor
  change (audit chain plus registry note), and rollback is one call.

**Live run** (2026-09-28, copies of the database and registry). Nine scheduled jobs were registered.

- **Case loop.** A risky transfer (p 0.999) went to REVIEW and opened an urgent case with a 60-minute SLA.
  An analyst pulled it and confirmed fraud, which wrote an INVESTIGATOR_CONFIRMED label. The queue then showed
  100% SLA attainment.
- **Recalibration.** Correctly refused: no matured labels yet.
- **Forced retraining.** The extract dropped 6,224 immature transactions. The incumbent, re-scored on the same
  split, held its place: the best candidate was 0.8542 against 0.8529, short of the 0.005 margin. The
  7-day guard then blocked a re-run, and the audit chain was intact.

## Phase 5 — Decision economics and rules

| Item | Status | Notes |
|---|---|---|
| False-positive cost model | **Done** | Expected-cost decisions, capacity and challenge budgets, guardrails. |
| Cost model v2 | **Done — switch pending approval** | `bti.operations.cost_model`; see *Cost model v2* below. |
| Analyst rules engine | **Done** | `/api/v1/rules`; see *Rules engine* below. |
| Step-up orchestration | **Done — needs the bank's gateways** | `/api/v1/stepup`; see *Step-up* below. |
| Decision-level fairness | **Done** (new) | See *Decision-level fairness* below. |

**Rules engine.**

- **Authoring.** Rules are JSON condition trees (`all` / `any` / `not`, ten null-safe operators), never code.
  They use only lineage-allowed fields: model features, the model's probability, currency, and identifiers for
  merchant, payee, device and IP watchlists. Protected attributes, post-event and label-derived fields are refused
  with the reason. The legacy engine used customer segment (R13), the label-derived risk score (R17), and refund
  and chargeback history.
- **Versions.** Versions are immutable. At most one active (enforced) and one shadow (logged-only challenger)
  version per rule can be live at once.
- **Simulation.** Every version is simulated on out-of-time history against the model's own capacity-constrained
  decisions. The simulation reports hits, precision, incremental fraud caught (hits the model alone would have
  approved), genuine customers disturbed, and the fairness of the rule's hits.
- **Approval rules.**
  - The approver must differ from the author.
  - The simulation must be under 30 days old.
  - A DECLINE rule needs at least 50% historical precision.
  - A rule with a fairness finding may run only in shadow.
- **Live effect.** An active rule can only raise a decision, never lower one. STEP_UP becomes REVIEW where the
  channel cannot challenge. Every hit is recorded, and `GET /rules/{id}/performance` gives live precision per
  version on matured labels.
- **Starter library** (`bti.rules.library`, drafts), simulated:

  | Rule | Result on history | Status |
  |---|---|---|
  | New device with login anomaly | 99% precision, 11 extra frauds, 1 genuine disturbed | Approved in the live run |
  | Failed-authentication burst | 3.9% precision (5 frauds for 566 genuine) | The simulation shows it is a bad rule |
  | First transaction draining the balance | Corporate FPR finding | Blocked from active by the fairness gate |
  | Velocity burst | 0 hits on synthetic data | Still recommended: the model's velocity features carry no weight (Phase 3 finding) |
  | Repeated just-below-threshold amounts | 0 hits on synthetic data | Kept as a draft |

**Step-up.**

- **Methods.** 3-D Secure for card-not-present, push for mobile, SMS one-time code otherwise (`stepup.methods`).
- **SMS codes.** Stored as salted hashes, compared in constant time, 3 attempts, 5-minute expiry.
- **Callbacks.** Challenge IDs are unguessable. Push and 3-D Secure results must carry an HMAC-SHA256 signature.
- **Providers.** `log` (development only, refused in production) or `webhook`: a signed POST to the bank's SMS,
  push or 3DS gateway.
- **Send failures.** A failed send marks the challenge `send_failed` and routes the transaction to REVIEW.
- **Outcomes.** Measured per method (pass, fail, abandon, catch rate on labels). Outcomes are not written as
  labels, because passing an SMS code does not prove a customer genuine (SIM swap).
- **Expiry and issuing.** An expiry sweep runs every 5 minutes. Optional auto-issue on STEP_UP decisions
  (`stepup.auto_issue`).
- **3-D Secure fields.** The 3DS field names follow EMV 3DS 2.x. Map them to the bank's 3DS server during
  integration.

**Cost model v2.**

- **Customer value.** Per-customer annual net revenue from the bank's P&L data (point-in-time, annualised,
  refreshed daily), bounded to $75–$1,200. Unbounded, a Student ($1.8 a year here) would be about 700 times
  cheaper to decline than a Corporate client.
- **Step-up economics.** Friction and catch rate come from measured outcomes per method once 200 challenges have
  completed (priors until then). Capacity shadow prices stay as floors.
- **No segment pricing.** Segment is not priced (it is a protected proxy). This is a deliberate departure from
  the original "segment friction" wording.
- **Backtest.** `POST /operations/cost-model/backtest` replays v1 and v2 on history. v2 lowered realised cost by
  about $9.2k (0.5%) and added no decision disparity. Corrected in Phase 8: the first figure, $11.9k (0.6%),
  counted capacity shadow prices as money spent.
- **Default.** The default stays **v1** until the bank approves the switch (`decisioning.cost_model`).

**Decision-level fairness** (found in the backtest). Earlier fairness tests checked scores at thresholds; this
one checks the actual decisions.

- **The raw picture.** Genuine Corporate customers are intervened on 34% of the time against 0.5% for Students
  (raw ratio 4.1×), because expected-cost decisions weigh probability by amount.
- **The new test** (`decision_fairness`) standardises intervention rates for amount, separating amount (a
  legitimate risk factor) from group.
- **What it shows.** Corporate and Private Banking are intervened on *less* than their amounts explain (0.83×,
  0.90×). **Genuine Retail customers are intervened on 1.88× what their amounts explain** (q < 0.001), although
  raw they are intervened on least. A given payment is unusual relative to their own history.
- **Where it runs and how it is recorded.** Raised as a medium finding for a policy decision. The test now also
  runs quarterly on live decisions in the outcomes analysis.

**Live run** (copies of the DB and registry; 11 scheduled jobs).

- **Starter rules.** Created and simulated through the API. "New device with login anomaly" was approved by a
  second person. "First transaction draining the balance" was refused active by the fairness gate.
- **Live rule hit.** A transaction from a known customer on a never-seen device after 4 logins hit the rule. It
  was recorded, and the model had already chosen REVIEW. For a first-time customer the rule correctly did not
  fire: "new device" is unknown without history.
- **Step-up.** A push challenge was issued, and its signed callback passed it.
- **Cost model.** 7,655 customer values were refreshed, and the step-up rates showed "prior" (nothing measured yet).

## Phase 6 — Graph intelligence (GNN)

PyTorch 2.14 (CPU) installs from PyPI. Only `download.pytorch.org` is blocked by the network policy. It is optional
(`requirements-graph.txt`), and only `graph_sage_score` needs it.

| Item | Status | Notes |
|---|---|---|
| Temporal entity graph | **Done** | `bti.graph.temporal`; see *Entity graph* below. |
| Graph embeddings without deep learning | **Done** | `bti.graph.learned`; see *Learned graph features* below. |
| GraphSAGE / temporal GNN | **Done — optional** | `bti.graph.learned`; see *Learned graph features* below. |
| Synthetic ring generator | **Done** | `bti.graph.synthetic_rings`; see *Synthetic rings* below. |
| Live serving | **Done** | `bti.graph.snapshot`; see *Live serving* below. |

**Entity graph.**

- **Links.** Customers are linked through shared devices and IPs.
- **Point-in-time rules.** Transactions are processed day by day, with incremental union-find. Each
  transaction's features use the state at the start of its day and only fraud confirmed before that day, so
  its own outcome is never visible.
- **Features.** Seven features:
  - network size
  - confirmed fraud in the network
  - customers sharing this device or IP
  - confirmed fraud among them
  - payee senders
  - confirmed frauds paid to the payee
  - the payee's fraud share: a mule receives from a few senders, many later confirmed; a biller receives from
    thousands
- **Why payees don't merge components.** Payees stay out of the components; otherwise a popular biller joins
  its customers into one meaningless cluster.
- **Lineage.** A new source, "fraud confirmations made before the day", is allowed. The raw confirmation time is
  blocked like any label field.

**Learned graph features.**

- **Snapshots.** Weekly, point-in-time.
- **Spectral embedding.** Randomised SVD of the linked subgraph. The feature is a rotation-invariant similarity to
  confirmed fraudsters.
- **Temporal GraphSAGE.** Pure PyTorch, a bipartite mean aggregator, predicting "fraud in the next 30 days". It is
  trained on training-window snapshots with two-fold temporal cross-fitting inside that window, and applied
  inductively later. Its weights travel in the model artifact.

**Synthetic rings.**

- **What is injected.** Mule accounts sharing devices and IPs, victims paying mule payees with their own devices
  and typical amounts, cash-out through shared devices, and realistic confirmation delays.
- **Payees.** Every payment in the data gets a payee, including shared billers, so payee presence cannot give the
  rings away.
- **Purpose.** Capability testing only. The base data has almost no network: 50 devices shared by 2 customers,
  and no shared IPs.

**Live serving.** A nightly snapshot rebuilds the graph from transaction history with the same code as
training. Label confirmation times come from the labels table. The scorer looks the transaction up at scoring
time, a one-day lag in both training and serving. `GET /operations/graph/status`,
`POST /operations/graph/snapshot`; 12 scheduled jobs. A `graph-relative` model scored live from the snapshot at
82 ms p50 (95 ms p95) warm.

**Capability test** (`python -m bti.graph.capability`, LightGBM, out-of-time, 15 injected rings = 19% of fraud):

| Data | Feature set | OOT PR-AUC | Ring frauds caught @2% / @5% budget |
|---|---|---|---|
| with synthetic rings | `core-relative-nb` | 0.8774 | 37.9% / 81.8% |
| with synthetic rings | `graph-relative` | 0.8910 | 42.4% / 90.9% |
| with synthetic rings | `graph-full-relative` | 0.8902 | 42.4% / 92.4% |
| original | `core-relative-nb` | 0.8845 | — |
| original | `graph-relative` | 0.8847 | — |
| original | `graph-full-relative` | 0.8828 | — |

**Reading.**

- **The simple point-in-time graph features carry the value.** Ring detection at the 5% budget rises from 81.8% to
  90.9%, and there is no change on data without rings.
- **Spectral and GraphSAGE add almost nothing on top here.** One more ring fraud, and PR-AUC slightly lower, within
  noise. The rings are only uncovered after they finish, so there is little confirmed fraud for learned methods to
  propagate from.
- **Defaults.** `graph-relative` is added to the tournament sets. It wins only if it beats the incumbent by the
  standing margin. GraphSAGE stays available for a bank's real graph, which is far denser.
- **Caveats.** These are capability results on synthetic rings: 66 ring frauds out-of-time, so treat the size as
  rough. They are not lift on a real book.

## Phase 7 — Streaming and scale

Built and tested against a real Kafka 3.9.1 broker (KRaft) and Redis 7.0 on the development container. The HA,
DR and India deployment design is in [HA_DR_AND_RESIDENCY.md](HA_DR_AND_RESIDENCY.md).

| Item | Status | Notes |
|---|---|---|
| ISO 8583 / ISO 20022 adapters | **Done** | `bti.streaming.iso8583`, `bti.streaming.iso20022`; see *Adapters* below. |
| Kafka ingestion | **Done** | `bti.streaming.consumer`; see *Streaming service* below. |
| Online feature store | **Done** | Redis, `bti.streaming.feature_store`; parity proven, see *Feature store and parity* below. |
| Inline latency | **Done** | p99 12.8 ms end to end with explanations inline (target < 50 ms); see *Latency* below. |
| High availability and residency | **Done — design and guards; drills are bank-run** | Readiness probe, residency guard, store rebuild, RPO/RTO targets; see *HA, DR and residency* below. |

**Adapters.**

- **ISO 8583 (1987).** An ASCII dialect with hex bitmaps (primary and secondary) and LLVAR/LLLVAR fields. A bank
  passes its own switch's `Spec`.
  - The PAN (field 2, or the PAN inside track 2) is Luhn-checked and replaced by a keyed HMAC-SHA256 token.
    Only the last four digits are kept.
  - Track data, the PIN block and ICC data are dropped after parsing (PCI DSS: no sensitive authentication data
    after authorisation).
  - Mapping: amount with the currency exponent (JPY 0, KWD 3); local date and time with the year inferred
    across New Year; processing code to type and debit/credit; POS entry mode to channel and authorisation
    method; MCC to the trained merchant categories.
  - A shared terminal is not treated as a customer device.
  - Responses and reversals are recognised and skipped.
- **ISO 20022.** `pacs.008` and `pain.001`, any schema version, one transaction per `CdtTrfTxInf`.
  - XML is parsed with defusedxml: DTDs, entity expansion and external entities are rejected; size and count
    are capped.
  - Debtor and creditor accounts are tokenised with the same key, so a payee links across customers and
    messages. This feeds the payee and mule features.
  - Private names and remittance text are never kept. A creditor name is kept only for organisations.
  - Inbound direction is supported, for mule monitoring.
- **Tokenisation key.** `BTI_TOKEN_KEY`, from the bank's HSM or secret store. Without it tokenisation refuses to
  run, except in development.

**Streaming service** (`python -m bti.streaming.consumer run | topics | replay | rebuild-store | simulate`).

- **Topics.** Input topics carry JSON, ISO 8583 or ISO 20022. Decisions go to `bti.decisions`, keyed by customer.
  Failures go to `bti.dlq`.
- **Delivery.** At least once, with idempotent scoring. Offsets are committed only after decisions are flushed.
  A redelivered transaction is not rescored: its stored decision is republished with `duplicate: true`. So there
  is no double score, no duplicate case and no double feature-store write.
- **Failures.** Decode and validation failures are dead-lettered at once. Transient scoring failures are retried
  with backoff, then dead-lettered. DLQ records carry a hash and length, never a card or payment payload. Replay
  re-reads source offsets.
- **Store rebuild.** `rebuild-store --since` re-feeds the feature store from the retained topics without scoring.
  Live transactions are not written to the transactions table, so this covers the gap since the last
  extract.
- **Metrics.** Prometheus text at `:9308/metrics`, plus `/healthz`.
- **Live run.** 800 simulated transactions (600 JSON, 200 ISO 8583) went through the real broker: all scored, none
  dead-lettered. Per-message p50 6.7 ms, p99 20 ms. About 90 messages/s per consumer process; throughput scales
  with partitions and instances.

**Feature store and parity.**

- **Keys.** Redis sorted sets per customer, device, IP, merchant and payee, scored by event time. Reads stop
  strictly before the transaction's second; per-kind retention.
- **Online computation.** `bti.streaming.online_features` mirrors `build_features` in pure Python.
- **Feature version 3.** Per-customer exact window sums. Version 2 subtracted running totals over the whole
  frame, so values drifted by about 1e-8 with the row's position. Version 3 also returns unknown for degenerate
  cases: no usual hour, no spread in past amounts. Versions 1–2 stay bit-identical (verified on all 50,000 rows),
  so registered models reproduce their scores.
- **Parity results** (`python -m bti.streaming.parity`):
  - Version 3: 52 of 52 features exact (relative 1e-9) on 3,000 transactions, both in-memory and through real
    Redis.
  - Version 2: exact except five rounding-sensitive features.
- **Model certificates.** `parity certify` compares calibrated probabilities. The current challenger (version 2,
  which uses two of those five) is **certified**: 0 of 5,000 probabilities differ. That makes it eligible for the
  fast path.
- **Fallback.** If Redis is unreachable, the scorer falls back to the database path and `/readyz` reports
  degraded.
- **Skew fixed.** The database path could not see other customers' merchant rows, so merchant velocity was
  undercounted live. The store counts them.

**Latency.**

- **What changed.**
  - The store replaces the history query (42 ms) and the pandas feature build (33 ms).
  - A numpy model row (identical to `to_model_matrix`, tested) and a direct LightGBM booster call (identical to
    `predict_proba`, tested) replace about 20 ms of single-row pandas overhead.
- **Results** (`python -m bti.streaming.latency`, 1,000 transactions end to end, including the challenger shadow
  score, rules and database writes):

  | Path | p50 | p95 | p99 |
  |---|---|---|---|
  | Database history (Phase 6 path; now the fallback) | 58.8 ms | 75.8 ms | 86.8 ms |
  | Feature store, explanations inline (**default**) | 5.2 ms | 10.7 ms | 12.8 ms |
  | Feature store, explanations async | 10.3 ms | 17.4 ms | 20.7 ms |

- **Async explanations** (`scoring.explain_mode: async`).
  - The decision is returned first. SHAP reason codes fill the score log, case and audit log from a worker pool,
    and `GET /v3/explanations/{id}` serves them.
  - All 1,000 arrived, p99 16 ms after the response.
  - On this model it is *slower* end to end: the workers compete with scoring for the CPU and the database, and
    inline SHAP costs only about 2 ms. So inline stays the default; async is for heavier models.
- **Caveat.** These figures are from one process on a 4-CPU development container with SQLite and local Redis.
  Re-measure on the bank's platform.

**HA, DR and residency.**

- **Readiness.** `/readyz` checks the database, the scoring model and residency (critical: 503 if any fails). It
  also reports the feature store, graph snapshot, explanation backlog and tokenisation key (degraded only).
- **Residency guard** (`residency.jurisdiction`, `mode: warn | enforce`).
  - It lists every outbound endpoint: database, Redis, Kafka, SMTP, webhooks, and the copilot LLM. Each host is
    checked against the bank's attested in-country hosts and CIDRs.
  - In enforce mode the API and consumer refuse to start, and the copilot refuses calls. With `IN`, the public
    LLM endpoint is blocked unless routed through an in-region gateway.
- **Guide.** The guide sets RPO/RTO targets and maps the components, with regulatory notes for India: RBI payment-data
  localisation, the DPDP Act, CERT-In 6-hour reporting and 180-day logs, and TRAI DLT for OTP SMS. It gives a
  reference topology (Mumbai/Hyderabad, Pune/Chennai, Mumbai/Delhi) and a bank-run DR drill checklist.
- **What is not demonstrated here.** Multi-zone failover and cross-region RPO/RTO. The container is a single node,
  so these are targets until the bank's drills measure them.

Tests: 10 for features, parity, store and latency paths; 16 for adapters; 5 for Kafka on the real broker; 5 for
operations.

## Phase 8 — Forecasting and planning

All four pieces are in `bti.planning`, served under `/api/v1/planning`, with two scheduled jobs (14 in all): weekly
forecasts and daily early warning. Results below are on the synthetic book: 730 days, 2,487 frauds, about 70
transactions a day. They show the machinery works, not field accuracy.

| Item | Status | Notes |
|---|---|---|
| Fraud-loss forecasting | **Done** | `bti.planning.forecast`; see *Loss forecast* below. |
| Alert-volume and staffing forecast | **Done** | `bti.planning.staffing`; see *Staffing* below. |
| Attack early warning | **Done** | `bti.planning.early_warning`; see *Early warning* below. |
| Policy what-if simulator | **Done** | `bti.planning.whatif`; see *What-if* below. |

**Loss forecast** (`python -m bti.planning.forecast`, `GET /planning/forecast/loss`).

- **Frequency.** Daily confirmed-fraud counts per country and channel, from an over-dispersed Poisson model
  (day-of-week plus a shrunk trend), fitted on the trailing year.
- **Severity.**
  - Fraud losses are extremely heavy-tailed: the largest 1% of frauds carry 45% of the loss, and half are
    fully recovered.
  - Severity is drawn from the trailing year's losses, with a generalised Pareto tail above the 90th percentile.
    A future loss can then exceed the worst seen so far, capped at 3× it as a stand-in for transaction limits.
  - Each path also draws the coefficients and the mean loss level, so the intervals carry estimation error too.
- **Coherence.** The total is the path-by-path sum of the countries, so they add up.
- **Label maturity (IBNR).** With confirmation dates, recent days are grossed up by completion factors from the
  confirmation-delay distribution, and days under half complete are left out of the fit. Tested on synthetic
  delays: the 30-day forecast is within 15% of the truth, even though the last two weeks look 30%+ quieter. The
  synthetic book has no confirmation dates, so the report says history is treated as complete.
- **Forecast at 30 December 2024:**

  | Horizon | Median loss | 80% interval | Frauds (median) |
  |---|---|---|---|
  | 30 days | $1.31M | $0.58M – $2.87M | 107 |
  | 60 days | $2.82M | $1.59M – $5.25M | 217 |
  | 90 days | $4.41M | $2.64M – $7.45M | 327 |

- **Backtest** (10 monthly origins in 2024; each uses only what was known then):

  | Horizon | Total: 80% interval coverage | Total: median error, forecast vs naive | Countries pooled: 80% / 90% coverage |
  |---|---|---|---|
  | 30 days | 0.80 | 43% vs 47% | 0.78 / 0.84 |
  | 60 days | 0.80 | 39% vs 55% | 0.68 / 0.81 |
  | 90 days | 0.70 | 42% vs 58% | 0.59 / 0.84 |

- **Reading.** The total is reasonably calibrated and beats the naive trailing mean. Country intervals at
  60–90 days are too narrow at the 80% level (0.59–0.68), though the 90% level holds. Only 10 overlapping origins
  exist, so treat coverage as approximate.
- **What the backtest changed.** Two fixes came from the backtest. An all-history severity pool missed a rise in
  loss size, so the forecast now uses the trailing year. A severity bootstrap with no tail had 90-day coverage of
  only 0.4. Errors around 40% at portfolio level reflect how lumpy fraud loss is; they are not a modelling
  failure.

**Staffing** (`python -m bti.planning.staffing --analysts N [--scale k]`, `GET /planning/forecast/staffing`).

- **Volume and cases.** Volume paths are thinned into case paths. Every review and every decline opens a case,
  routed to queues as live, with rates from replaying the live policy on the out-of-time window.
- **Handling times.** From closed cases once a queue has 30. Until then, configured defaults (illustrative).
- **Staffing method.**
  - Workload FTE (case work ÷ occupancy ÷ productive hours).
  - Pooled Erlang C per hour, meeting the strictest SLA for 90% of cases at ≤ 85% occupancy.
  - 30% shrinkage.
- **Book.** 4 cases a day (P90: 7), a workload of 0.3 FTE, but 4.3 FTE for round-the-clock cover.
- **Projected to about 1M transactions a month** (`--scale 470`): 1,940 cases a day, 139 workload FTE, 141 Erlang
  FTE; plan to the P90 day at about 164.
- **Review capacity.** Given rostered analysts, staffing proposes a `max_review_rate`. It is only a proposal;
  applying it is a named capacity refit.
- **Finding.** The provisional model's declines (4.05% of transactions) become reviews, and with 1.88% reviews
  that is 5.9% case volume against the 2% review target. At the 1M scale, 25 analysts cannot clear the declines
  alone, and the proposal says so. The capacity fit limits reviews, not declines. Promoting an approved champion,
  or pricing declines, is the lever.

**Early warning** (`python -m bti.planning.early_warning evaluate`; daily job; `GET /planning/early-warning`).

- **What is watched.** Rate-based Poisson CUSUMs on two signals:
  - confirmed fraud, by typology (10), merchant category (20), merchant (98) and corridor (64, country ×
    channel)
  - model interventions, by merchant category, merchant and corridor. These are available the same day, before
    any dispute.
- **Baselines.** 90 days before a 7-day guard band, shrunk to the portfolio rate, so a rise in volume alone does not
  alarm.
- **Thresholds.**
  - Each segment's threshold comes from a simulated table, for the in-control run length implied by a budget of 1
    false alarm per family per month.
  - Calibrated on clean history, because estimating baselines adds noise: typology ×1.5 and corridor ×2. The
    false-alarm rates are then 1.0, 0.81, 0.38 and 0.90 a month.
- **Injected attacks** (all detected):

  | Attack | Detection delay |
  |---|---|
  | Card testing: 30 small frauds at one merchant over 3 days | same day, on both signals |
  | Authorised-push-payment surge, 3× for 3 weeks | 8 days |
  | Merchant-category compromise, 2× for a month | 6 days (alerts), 7 (fraud) |
  | Nigeria × USSD corridor spike, 5× for 2 weeks | 13 days, late; thin corridor |

- **Live.** Alarms go to `early_warnings` with an audit event and the webhook. They are acknowledged or closed
  through the API, and closing needs a note. On a copy of the live database the first run raised Card Not Present
  fraud (×4.0) and DoorDash (×3.0). A re-run added no duplicates.

**What-if** (`python -m bti.planning.whatif '{...}'`, `POST /planning/whatif`).

- **Method.** A proposal is replayed against the live policy on the labelled out-of-time window (12,500
  transactions, 181 days), with the production decision function. A proposal can change cost figures, capacity
  targets (refitted, nothing saved), step-up channels, the model, or provisional status.
- **Outputs.** Loss, genuine customers disturbed, realised cost, cases a day, analyst FTE and intervention ratios by
  segment and age band. Differences carry 95% intervals from a day-block bootstrap. Each report is saved with an id
  as change-request evidence.
- **Scenarios.**

  | Proposal | Realised cost (95% CI) | Customer impact |
  |---|---|---|
  | Double review capacity (2% → 4%) | −$39.7k (−$73.8k to −$11.1k) | 304 more genuine customers held; 1.7 more cases a day |
  | Step-up at POS and ATM | −$3.8k (CI spans 0) | 278 more genuine customers challenged |
  | Loss-given-fraud stress at 95% | Loss and cost rise as expected | — |

- **Accounting fix.** Decisions use the capacity shadow prices; realised cost uses the economic figures. Building
  this found that the Phase 5 cost-model backtest had counted the shadow prices ($107 per review instead of $8) as
  money spent. Corrected, cost model v2 still lowers realised cost: by $9.2k (0.5%), not the $11.9k (0.6%) reported
  before. The recommendation stands.

Tests: 9 unit tests (Phase 8) and 3 API tests.

## Phase 9 — Scam and mule models

All of this is in `bti.scams`, served under `/api/v1/scams`, with a daily mule scan (15 scheduled jobs in all).

**Governance.**

- **Model families.** Scam and mule models live in their own registry families (`models/registry_scam`,
  `models/registry_mule`), with the same rules as the fraud model: immutable entries, a validation gate,
  four-eyes promotion and append-only notes. Fraud-registry jobs never see them.
- **Mode.** Both models are registered as **challengers**, shadow only.
- **Lineage.** The new feeds go through the same lineage catalogue as everything else:
  - pre-authorisation: payee account opening date, Confirmation-of-Payee result, account opening date
  - identifiers: on-us payee customer, inbound counterparty
  - protected (reimbursement exposure only, never a model input): vulnerability flag
  - label-derived: scam typology

**Synthetic scams and mules** (`python -m bti.scams.synthetic`). A capability harness on a copy of the data, built
on the Phase 6 rings.

| Scams | Mule accounts |
|---|---|
| 1,177 scam payments among 29,646 payments | 128 accounts: 88 ring mules, 40 standalone |

- **Scam typologies.** Purchase, impersonation/safe-account, investment and romance scams, plus ring victims. All
  are paid from the victim's own device.
- **Benign lookalikes,** so no single signal gives the answer away:
  - large first payments to new payees, some to young accounts or with name near-misses
  - recurring rent and savings payments, 8% to young accounts
  - on-us payees that are ordinary customers
  - households sharing a device and home IP
  - pass-through salary accounts
  - 12% new customers
- **How they were found.** Each lookalike was added after a first model leaned on a synthetic artefact: repeat
  payments, on-us payees, shared devices and IPs. Mules and victims are drawn at random across segments and ages,
  so no group is labelled by construction.

**APP-scam model** (`python -m bti.scams.app_model`; challenger `bti-scam-lgbm-20261006103935`).

- **Features.** Point-in-time payee risk, under the same lineage gate:
  - new payee for the customer
  - payee account age
  - Confirmation-of-Payee no-match / close-match / unavailable
  - payee new to the bank, and days since anyone first paid it
  - distinct senders to the payee in 30 days, and payments in 24 hours
  - payee held at the bank
  - repeat and escalating payments
  - new payees in 7 days
  - amount against the customer's largest earlier payment
  - Phase 6 payee graph signals (day-lagged)
- **Population and split.** Outbound payments to a payee, split by time.
- **Out-of-time results** (295 scams among 5,661 payments), scam model vs transaction model:

  | | Scam model | Transaction model |
  |---|---|---|
  | PR-AUC | 0.63 | 0.08 |
  | Recall at 2% of payments | 32% (37% of scam value, 82% precision) | 3% |
  | Recall at 5% of payments | 59% (74% of value) | 8% |

- **Recall by typology at 2%.** Impersonation 57%, investment 30%, romance 9%, purchase 13%. Romance and purchase
  scams look like ordinary payments to a new payee, which is why the typologies differ so much.
- **Drivers.** Payee account age (41% of gain), payee held here, payee first seen.
- **Gates (all passed).** Lineage, leakage screen (no single feature at AUC ≥ 0.97), lift over the transaction
  model, calibration (ECE 0.012) and fairness.
- **Fairness method change.** The unit is now the **customer**, not the payment. A series of rent payments to a new
  landlord is one customer's experience, and counted per payment it overstated significance: an earlier
  synthetic draw gave Premium 2.96× and age 26–35 1.79× per payment. On that draw a remaining Premium signal did
  not reproduce across two other generator seeds. `fairness_assessment(clusters=…)` now supports any model.
- **Fairness result.** The final model passes, with Private Banking on the watchlist (one window). The development
  history is recorded as a registry note.

**UK reimbursement exposure and the scam overlay.**

- **The rules modelled.** The PSR mandatory reimbursement rules from 7 October 2024, configurable; compliance to
  confirm the rules in force as the PSR moves into the FCA:
  - Faster Payments and CHAPS between UK accounts
  - up to £85,000 per claim, split 50:50 between sending and receiving bank (100% when the payee is on-us)
  - an optional excess of up to £100, never for vulnerable customers
- **Not netted off.** The consumer-caution exception, so exposure is conservative.
- **Per payment.** Bank exposure and the customer's unreimbursed loss. Interventions are chosen by expected cost:
  none, a tailored warning, or hold and call. Effectiveness and friction figures are illustrative until measured.
- **Live overlay.**
  - It runs on every outbound payment inside `score_and_decide`, logs to the score log (four new columns), and
    returns `scam` in the v3 response.
  - Payments to an on-us payee under an open or confirmed mule alert are held for a call.
  - It is in **shadow mode by default**. A challenger-only scam model can never act, and the overlay never lowers
    a decision.
- **Smoke test.** £4,800 to a 13-day-old account with a name mismatch: probability 0.48, exposure £2,400,
  hold-and-call (logged, not applied). £80 to a biller: 0.004, no action.
- **Portfolio replay** (154 out-of-time days):
  - 305 scams, of which 41 are UK in-scope in 20 claims, are £147k of bank exposure if nothing is done
  - the overlay would make 6.3 calls a day and show 1,377 warnings
  - it would interrupt 801 genuine payments with calls and 1,309 with warnings
  - expected averted: £85k of bank exposure and £969k of customers' losses (most outside UK reimbursement
    scope), for £12k of intervention cost

**Mule-account detection** (`python -m bti.scams.mule`; challenger `bti-mule-lgbm-20261006104223`).

- **Unit.** Weekly account snapshots: 220,443 across 8,113 accounts.
- **Features.**
  - inbound credits, distinct and new senders
  - pass-through ratio, and the share of inflow sent on within two days (each outflow counted once)
  - cash or crypto share of outflow
  - new payees paid
  - account age
  - inbound payments already reported as fraud (sending-bank reports, confirmed before the snapshot)
  - Phase 6 network signals
- **Labels.** Positive only while a mule is active; snapshots before activation or after closure are left out.
- **Out-of-time, at 0.5% of active accounts per weekly scan** (11 alerts, 70% precision):

  | | Mules detected | Detected after activation (median) | Lead before uncovered (median) | PR-AUC |
  |---|---|---|---|---|
  | Model | 36 of 42 (86%): ring 31/34, standalone 5/8 | 8.5 days | 54 days | 0.98 |
  | Without network features | same 86% | 13 days | 49 days | 0.77 |
  | Rule (5+ senders, 70% pass-through) | 4 of 42 (10%) | — | — | — |

- **Reading.** The high PR-AUC reflects how visible ring structure is in synthetic data. The ablation shows inflow
  and outflow patterns alone still catch most mules.
- **Fairness.** Passes at the customer level; SME and 65+ are on the watchlist (one window each).
- **Live scan** (`run_scan`, daily job, `POST /scams/mule-alerts/run`).
  - It scores today's snapshot from the last 120 days of transactions and raises the budget as `mule_alerts`, with
    audit events and the webhook.
  - Analysts confirm or clear an alert with evidence (key required; decided once).
  - On a database copy: 1 June 2024 raised 11 alerts, 6 of them mules. 15 October 2024 raised 11, all mules. A
    re-run created no duplicates.
- **Schema.** The transactions table gained the scam and mule feed columns (additive migration).

**Bug found by a test** and fixed before results were recorded: fast-out matched one outflow against several
inflows.

**Still needed from the bank:**

- Confirmation-of-Payee responses and the payee-intelligence feed (payee account age).
- Inbound credit counterparties.
- Sending-bank scam reports.
- The vulnerability register.
- Measured warning and call effectiveness.
- Confirmed scam and mule outcomes to retrain on.

Tests: 8 unit tests (Phase 9) and 2 API tests.

## Phase 10 — Certification

**BTI is not certified, and no code can make it so.**
- SOC 2 is attested by an independent CPA firm over a 3–12-month observation period.
- ISO 27001 is certified by an accredited body.
- PCI DSS is validated by a QSA or a self-assessment.
- Penetration tests are done by independent testers.

All of them also cover the organisation that operates BTI. Phase 10 builds what the software contributes and fixes
what an auditor would have failed. The full mapping, scope and roadmap are in
[CERTIFICATION_READINESS.md](CERTIFICATION_READINESS.md).

| Item | Status | Notes |
|---|---|---|
| Access control fit for audit | **Done** | Principals, role table, identity binding, authenticated maker-checker; see below. |
| Hardening | **Done** | Security headers, docs off in production, 25 MB limit, rate limits, readiness detail only when authenticated, startup refusal of default secrets. |
| Model artifact integrity | **Done** | SHA-256 recorded at registration and verified before loading; 35 existing artifacts sealed (trust on first use, noted in each registry). |
| PCI DSS scope reduction | **Done — deployment by the bank** | See *Card data* below. |
| Security self-assessment and CI gate | **Done** | `python -m bti.security.assessment`: 12 pass, 2 open (per-deployment settings); see below. |
| Evidence pack | **Done** | `python -m bti.security.evidence build \| verify`: ten system artefacts and a SHA-256 manifest anchored in the audit log; `verify` caught a one-file alteration in testing. |
| SOC 2 / ISO 27001 mapping, threat model, pen-test scope | **Done — documents** | TSC CC1–CC9 / A1 / C1 / PI1 and an Annex A SoA draft, each marking what the software does and what the organisation must do; STRIDE threat model; penetration-test scope and rules of engagement. |
| SOC 2 Type II, ISO 27001 certificate, PCI validation, penetration test | **Pending — external** | Needs the operating organisation's ISMS (policies, risk assessment, HR, suppliers, incident response, MFA via SSO), an auditor and a tester. Realistic path about 9–12 months. |

**Access control** (`bti.security`).

- **What was found.** Before Phase 10, 94 API routes had no authentication beyond the three meant to be public:
  - 73 reads, including customer transactions, cases and per-customer risk
  - 21 writes, including scoring, model reload and alert edits

  Approvals carried a free-text approver behind one shared key, so four-eyes could not be enforced.
- **Principals.** Each person or system has its own key, stored only as SHA-256, with roles and immediate
  deactivation.
- **Role table.** Every one of the 144 routes is in the table; anything unlisted is admin-only. Customer-level data is
  for analysts and auditors; auditors cannot write.
- **Identity binding.** Approver, author, validator, reviewer, analyst and checker must equal the authenticated
  caller.
- **Maker-checker.** A high-value clearance needs the checker's own authenticated confirmation
  (`POST /cases/{id}/check`), and the workbench now works this way.
- **Legacy key.** The shared key is off by default, and production refuses to start with the default one.

**Card data.**

- **The tokenisation edge.** The ISO 8583 adapter is the only component that sees a PAN. Deployed inside the bank's
  CDE, it keeps the rest of BTI out of PCI scope.
- **Redaction.** Logs redact PANs and IBANs.
- **PAN discovery** (`bti.security.pan_scan`; Luhn, issuer prefixes, no decimals or hashes). It found none on the
  development system, nor on a database that processed 200 ISO 8583 card messages. An early run's 18 hits were
  decimals in reports and digits in audit hashes; the rule was tightened and those false-positive patterns are now
  in the tests.

**Self-assessment results.**

- **Pass (12):**
  - all 141 protected routes refuse anonymous callers
  - the auditor role is refused on all 68 write routes
  - injection and traversal payloads on every path parameter: no 5xx, no reflection
  - SAST (bandit) clean of medium and high findings, after fixing one high and accepting four justified items
  - deployable dependencies (101 resolved from requirements.txt) have no known vulnerabilities; the build
    container's own system packages are reported for information
  - no secrets, no card data
- **Open (2):** the tokenisation key and the residency jurisdiction, both per-deployment settings.
- **CI.** A new `security` job (bandit, pip-audit, self-assessment) gates the Docker build.

**What the bank or operator must still do:**

- Put human access behind SSO with MFA.
- Write the ISMS policies, risk assessment and SoA approval, internal audit and management review.
- Handle HR screening and training, and supplier contracts (cloud, SMS, LLM).
- Write an incident-response plan and test it.
- Run DR drills.
- Hold an HSM key ceremony for the tokenisation edge, and segmentation testing.
- Commission the external audits and penetration test.

Tests: 14 security tests (API access, identity binding, maker-checker, hardening, redaction, artifact integrity,
evidence pack, PAN discovery). The existing integration tests now authenticate.

---|---|---|
| SOC 2 Type II, ISO 27001, PCI DSS scope reduction, penetration test | New | Procurement gates. |

---

**On differentiation:** SAS sells network analytics, model management, rules and case management
as licensed modules. BTI's defensible differences are a transparent model with enforced lineage
and exact per-decision reasons, amount-aware expected-cost decisions fitted to each bank's
capacity, one policy engine across eight jurisdictions, the investigator copilot, and forecasting
tied to the decision engine. Check each bank's licence before claiming a specific gap.
