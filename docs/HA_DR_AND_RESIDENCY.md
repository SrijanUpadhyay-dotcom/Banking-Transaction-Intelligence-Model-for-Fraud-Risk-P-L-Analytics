# BTI — High Availability, Disaster Recovery and Data Residency

**Status.** This is the deployment design and the targets a bank should hold
BTI to. Every mechanism below exists in the codebase and is tested on one node:
- database fallback when the feature store is down
- idempotent stream processing
- topic replay and the store rebuild
- the readiness probe
- the residency guard

The multi-zone and multi-region figures are **targets**. They have not been
demonstrated: the development environment is a single container, and the
bank's platform team proves them in DR drills. Regulatory references are a
starting map for the bank's compliance and legal review, not legal advice.

---

## 1. What runs where

| Component | State | Role | Scaling and HA pattern |
|---|---|---|---|
| Scoring API (`api.main`) | Stateless | Inline decisions for the authorisation path | N replicas across ≥ 2 availability zones behind a load balancer that routes only to `/readyz` = 200 |
| Stream consumer (`bti.streaming.consumer run`) | Stateless (offsets in Kafka) | Kafka scoring: ISO 8583, ISO 20022, JSON | One consumer group; up to one instance per partition (6 by default). A failed instance's partitions rebalance to the others in seconds |
| PostgreSQL | **System of record** | Score log, cases, labels, rules, validation records, hash-chained audit log | Primary plus a synchronous standby in another zone; an asynchronous replica in the DR region |
| Redis feature store | Derived, rebuildable | Customer and entity event logs for online features | Primary plus a replica, with Sentinel or a managed failover. Append-only file (`appendfsync everysec`) on the replica |
| Kafka | Durable log | Input transactions, decisions, dead letters | 3 brokers across 3 zones; `replication.factor=3`, `min.insync.replicas=2`; producers use `acks=all` (BTI's do). MirrorMaker 2 to the DR region |
| Model registry and graph snapshot | Immutable files | Model artifacts, cards, parity certificates, nightly snapshot | Versioned object storage, replicated to the DR region. Artifacts never change after registration |
| Audit archive | Immutable | Daily sealed audit export, 7-year retention | Object storage with object lock (WORM), replicated in-country |

**Ordering.** The stream needs each customer's transactions in order.
Producers must key messages by customer, card or account, so that Kafka keeps
them on one partition.

---

## 2. Targets

| Scope | Event | RPO (data loss) | RTO (time to recover) | Mechanism |
|---|---|---|---|---|
| In-region | API or consumer instance lost | 0 | < 30 s | Load balancer stops routing on failed `/readyz`. Kafka rebalances partitions. Unacknowledged messages are redelivered and deduplicated by the idempotent scorer |
| In-region | Database primary lost | 0 | < 2 min | Synchronous standby promoted (managed Postgres multi-AZ, or Patroni) |
| In-region | Redis primary lost | ≤ 1 s of events | < 1 min | Replica promoted. **Scoring never stops:** while Redis is unreachable the scorer falls back to the database path (correct, slower: measured p99 87 ms vs 13 ms) |
| In-region | Redis lost entirely | Rebuilt | < 1 h (scales with history) | `feature_store backfill` from the transactions table, then `consumer rebuild-store --since <last extract>` from the retained input topics. Scoring continues on the database path meanwhile |
| In-region | Broker lost | 0 | Automatic | RF 3 / min ISR 2: the cluster keeps accepting writes with one broker down |
| Regional | Primary region lost | ≤ 5 min (async replication) | ≤ 60 min | Promote the DR database replica. Point consumers at the mirrored topics. Start API and consumer replicas from the same images. Rebuild or restore Redis as above |
| Any | Model artifact corrupted or deleted | 0 | Minutes | Immutable versioned artifacts, plus the registry index with its role history |

**The authorisation path must not wait on BTI.** The bank's switch or
authorisation host gives BTI a time budget, for example 50 ms. On a timeout
or error it falls back to its own rules or the incumbent (SAS). The Phase 2
router already implements fallback to the incumbent. BTI's own measured
budget, single node, 1,000 transactions end to end including the challenger
shadow score and database writes:

| Path | p50 | p95 | p99 |
|---|---|---|---|
| Database history (Phase 6 path; fallback) | 58.8 ms | 75.8 ms | 86.8 ms |
| Feature store, explanations inline (**default**) | 5.2 ms | 10.7 ms | 12.8 ms |
| Feature store, explanations async | 10.3 ms | 17.4 ms | 20.7 ms |

These come from `python -m bti.streaming.latency --n 1000`, in one process,
with SQLite and local Redis on a 4-CPU development container. Re-measure on
the target platform: network round trips to Redis and Postgres add to these.

---

## 3. Failure behaviour, as built

- **Feature store down.**
  - Reads fail fast: a 50 ms socket timeout, then the database path takes
    over.
  - Writes are best-effort and never fail a score.
  - `/readyz` reports `feature_store: degraded`; the instance stays ready.
  - After recovery, re-run `backfill` for the outage window. Transactions
    scored during the outage were not written to the store, so features
    undercount until they are backfilled.
- **Feature version and parity.**
  - The online path is used only where it is proven exact: feature version 3
    models, version 1–2 models that use none of the five rounding-sensitive
    features, or a model with a parity certificate
    (`python -m bti.streaming.parity certify`).
  - The current challenger is certified: identical probabilities on 5,000
    transactions.
- **Redelivery.**
  - A transaction that already has a champion score-log row is not scored
    again. Its stored decision is republished with `duplicate: true`.
  - No double-scoring, no duplicate case, no double write to the store.
- **Undecodable or invalid messages** go to `bti.dlq`. The record holds the
  error, the source offset, and a SHA-256 and length of the payload, never a
  card or payment payload.
- **Transient scoring failures** are retried with exponential backoff (3
  times by default), then dead-lettered. After the cause is fixed:
  `consumer replay --topic … --partition … --start … --end …`.
- **Missing tokenisation key.** The consumer stops rather than dead-lettering
  every card message.
- **Explanations.**
  - Inline by default (about 2 ms).
  - In async mode the decision never waits on SHAP. Reason codes reach the
    score log, the case and the audit log within milliseconds (measured p99
    16 ms after the response).
  - `/readyz` reports the backlog.

---

## 4. India in-country deployment

**Regulatory map** (for the bank's compliance team to confirm):
- **RBI, Storage of Payment System Data** (circular of 6 April 2018, with the
  2019 FAQs). The entire data relating to payment systems must be stored only
  in India. Where processing happens abroad, the data is to be deleted there
  and brought back to India within one business day or 24 hours, whichever is
  earlier. *Design consequence:* every BTI component that stores or processes
  transactions (API, consumers, Postgres, Redis, Kafka, object storage,
  backups, DR) runs in Indian regions. Nothing is processed abroad.
- **Digital Personal Data Protection Act, 2023.** Cross-border transfer is
  allowed except to countries the government restricts, but sectoral rules
  such as RBI's prevail where they are stricter. For payments, RBI
  localisation is the binding constraint.
- **CERT-In Directions (28 April 2022).** Report cyber incidents within 6
  hours. Keep ICT system logs for 180 days within Indian jurisdiction. BTI's
  audit log and 7-year sealed archive exceed the retention period. The
  application logs must also be shipped to an in-country log store.
- **TRAI DLT** for step-up SMS: OTP templates and sender IDs must be
  registered on the DLT platform through an Indian SMS provider.

**Reference topology.**

| | Primary | DR |
|---|---|---|
| AWS | Mumbai `ap-south-1` (3 AZs) | Hyderabad `ap-south-2` |
| Azure | Central India (Pune) | South India (Chennai) |
| Google Cloud | Mumbai `asia-south1` | Delhi `asia-south2` |
| On-premises | Bank's primary data centre | Bank's DR data centre (in India) |

**Residency guard.** Configure it in `config/settings.yaml`:

```yaml
residency:
  jurisdiction: "IN"
  mode: "enforce"
  in_country_hosts:
    - "*.ap-south-1.rds.amazonaws.com"
    - "*.ap-south-2.rds.amazonaws.com"
    - "*.bank.in.internal"
    - 10.0.0.0/8
```

The guard lists every outbound endpoint BTI is configured to use:
- database
- feature store
- Kafka brokers
- SMTP relay and alert webhook
- step-up webhook
- the copilot's LLM endpoint, when an API key is set

It checks each host against the bank's attested in-country list. In
`enforce` mode:
- the API and the stream consumer refuse to start if any endpoint is outside
  the list
- the copilot refuses calls
- `/readyz` returns 503 with the violations listed

The guard cannot see geography; it enforces the bank's own inventory.
Hostnames should resolve to in-country addresses, which the network team
verifies.

**The investigation copilot.**
- It sends case details to an LLM. With `jurisdiction: IN`, the public
  endpoint (`api.anthropic.com`) is outside the list, so the copilot is
  blocked.
- To use it in India, route through an approved in-region endpoint (set
  `ANTHROPIC_BASE_URL` to the bank's gateway on an attested host), or leave it
  disabled.
- Scoring does not depend on the copilot.

**Identifiers.**
- Card numbers and IBANs or account numbers are tokenised on arrival with a
  keyed HMAC (`BTI_TOKEN_KEY`).
- The key lives in the bank's HSM or secret store, in India.
- Rotating the key changes every token, so plan rotation as a re-keying of
  history.

---

## 5. DR drill checklist (bank-run)

1. Kill an API replica and a consumer during load. Confirm no lost or duplicate decisions: each `transaction_id` is decided once, and redeliveries are flagged `duplicate`.
2. Fail over the database primary. Confirm writes resume, and the audit chain verifies (`GET /api/v1/governance/audit/verify`).
3. Stop Redis. Confirm scoring continues on the database path with `/readyz` degraded. Restore it, backfill the window, and re-run `python -m bti.streaming.parity --redis`.
4. Wipe Redis. Run `backfill` then `rebuild-store --since <last extract>`, and time it against the RTO.
5. Promote the DR region. Confirm the residency guard passes there and `/readyz` returns 200. Replay the decisions topic gap if any.
6. Record the measured RPO and RTO against section 2 and file them with the validation record.
