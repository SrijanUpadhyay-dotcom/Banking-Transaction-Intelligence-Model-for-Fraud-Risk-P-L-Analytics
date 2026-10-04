# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Kafka streaming service: transactions in, decisions out.

    python -m bti.streaming.consumer run              # consume, score, publish (Ctrl-C to stop)
    python -m bti.streaming.consumer topics           # create the topics if missing
    python -m bti.streaming.consumer replay --topic T --partition P --start S --end E
    python -m bti.streaming.consumer simulate --n 500 --format json|iso8583
    python -m bti.streaming.consumer rebuild-store --since 2026-10-03T00:00:00+05:30

**Flow.**
1. Input topics carry JSON (the /v3/score schema), ISO 8583 or ISO 20022
   (`streaming.topics` maps topic to format).
2. Each message is decoded by its adapter. Card numbers and accounts are
   tokenised there.
3. It is scored by `score_and_decide`, the same path as the API: champion,
   decision, rules, shadow challenger, score log, cases.
4. The decision goes to `bti.decisions`, keyed by customer.

**Ordering.** Producers must key messages by customer, card or account. Kafka
then keeps each customer's transactions in order on one partition, and the
feature store sees a customer's earlier transaction before the later one.

**Delivery: at least once, with idempotent scoring.** Offsets are committed
only after the batch's decisions are flushed to the broker. A crash in between
redelivers the batch. A transaction that already has a champion score-log row
is not scored again: its stored decision is republished with
`"duplicate": true`. So redelivery never double-scores, double-writes the
feature store or opens a second case. Downstream consumers deduplicate on
`transaction_id`.

**Failures.**
- A message that cannot be decoded or validated goes to `bti.dlq` at once.
  This is a permanent failure.
- A scoring failure (for example the database is down) is retried with
  backoff, then dead-lettered.
- A missing tokenisation key stops the service rather than dead-lettering
  every card message.
- DLQ records carry the error, the source topic/partition/offset, and a
  SHA-256 and length of the payload. They never carry the payload of a card
  or payment message. Replay re-reads the source offsets once the cause is
  fixed.

**Responses and reversals.** ISO 8583 responses and reversals are recognised
and skipped, and counted.

**Metrics.** Prometheus text format at http://:9308/metrics, and /healthz.
"""

from __future__ import annotations

import hashlib
import json
import signal
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from bti.config import get_settings
from bti.logging_config import get_logger
from bti.streaming import iso8583, iso20022
from bti.streaming.tokenize import TokenisationError

log = get_logger("streaming.consumer")
SCHEMA_VERSION = 1
REQUIRED = ("transaction_id", "customer_id", "transaction_date", "transaction_amount", "currency")


class PermanentError(ValueError):
    """The message can never be processed as sent (decode or validation failure)."""


@dataclass
class Record:
    """The parts of a Kafka record the service needs (lets the handler run without a broker)."""
    topic: str
    partition: int
    offset: int
    key: Optional[bytes]
    value: bytes
    timestamp_ms: Optional[int] = None


# ── metrics ──────────────────────────────────────────────────────────────────
class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters: Dict[str, int] = defaultdict(int)
        self.latency = deque(maxlen=5000)
        self.last_poll = None

    def inc(self, name: str, n: int = 1) -> None:
        with self.lock:
            self.counters[name] += n

    def observe(self, ms: float) -> None:
        with self.lock:
            self.latency.append(ms)

    def snapshot(self) -> Dict:
        with self.lock:
            lat = np.asarray(self.latency) if self.latency else None
            return {"counters": dict(self.counters),
                    "latency_ms": None if lat is None else {q: round(float(np.percentile(lat, p)), 2)
                                                            for q, p in (("p50", 50), ("p95", 95), ("p99", 99))},
                    "last_poll": self.last_poll}

    def prometheus(self) -> str:
        snap = self.snapshot()
        lines = ["# TYPE bti_stream_events_total counter"]
        for name, value in sorted(snap["counters"].items()):
            lines.append(f'bti_stream_events_total{{event="{name}"}} {value}')
        if snap["latency_ms"]:
            lines.append("# TYPE bti_stream_score_latency_ms summary")
            for q, p in (("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")):
                lines.append(f'bti_stream_score_latency_ms{{quantile="{q}"}} {snap["latency_ms"][p]}')
        return "\n".join(lines) + "\n"


def serve_metrics(metrics: Metrics, port: int, stale_after_s: float = 60.0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/metrics":
                body, status, ctype = metrics.prometheus().encode(), 200, "text/plain; version=0.0.4"
            elif self.path == "/healthz":
                last = metrics.last_poll
                ok = last is not None and time.time() - last < stale_after_s
                body = json.dumps({"status": "ok" if ok else "stale", "last_poll": last}).encode()
                status, ctype = (200 if ok else 503), "application/json"
            else:
                body, status, ctype = b"not found", 404, "text/plain"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="bti-stream-metrics").start()
    return server


# ── decoding ─────────────────────────────────────────────────────────────────
def validate(txn: Dict) -> Dict:
    from bti.modeling.fx import supported_currencies
    missing = [k for k in REQUIRED if txn.get(k) in (None, "")]
    if missing:
        raise PermanentError(f"missing required fields: {', '.join(missing)}")
    try:
        amount = float(txn["transaction_amount"])
    except (TypeError, ValueError):
        raise PermanentError("transaction_amount is not a number")
    if not amount > 0:
        raise PermanentError("transaction_amount must be positive")
    if str(txn["currency"]).upper() not in supported_currencies():
        raise PermanentError(f"unsupported currency {txn['currency']}")
    try:
        datetime.strptime(str(txn["transaction_date"])[:10], "%Y-%m-%d")
    except ValueError:
        raise PermanentError("transaction_date must be YYYY-MM-DD")
    return txn


def decode(fmt: str, value: bytes, received_at: Optional[datetime] = None,
           customer_resolver: Optional[Callable[[str], Optional[str]]] = None) -> List[Dict]:
    """Transactions in one message; [] for a recognised message that is not scored (e.g. an 8583 response)."""
    try:
        if fmt == "json":
            body = json.loads(value)
            items = body if isinstance(body, list) else [body]
            if not all(isinstance(i, dict) for i in items):
                raise PermanentError("JSON message must be an object or a list of objects")
            return [validate(dict(i)) for i in items]
        if fmt == "iso8583":
            msg = iso8583.parse(value)
            if not msg.scoreable:
                return []
            return [validate(iso8583.to_transaction(msg, received_at, customer_resolver))]
        if fmt == "iso20022":
            return [validate(t) for t in iso20022.parse(value, customer_resolver=customer_resolver)]
    except (iso8583.Iso8583Error, iso20022.Iso20022Error, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PermanentError(f"{type(exc).__name__}: {exc}")
    raise PermanentError(f"unknown format {fmt!r}")


# ── service ──────────────────────────────────────────────────────────────────
class StreamService:
    def __init__(self, topics: Optional[Dict[str, str]] = None, bootstrap: Optional[str] = None,
                 group_id: Optional[str] = None, decisions_topic: Optional[str] = None,
                 dlq_topic: Optional[str] = None, session_factory=None,
                 customer_resolver: Optional[Callable[[str], Optional[str]]] = None,
                 enrich: Optional[Callable[[Dict], Dict]] = None, default_country: Optional[str] = None,
                 max_retries: Optional[int] = None, retry_backoff_s: float = 0.5, metrics: Optional[Metrics] = None):
        s = get_settings()
        self.topics = topics or dict(s.streaming_topics)
        self.bootstrap = bootstrap or s.streaming_bootstrap_servers
        self.group_id = group_id or s.streaming_group_id
        self.decisions_topic = decisions_topic or s.streaming_decisions_topic
        self.dlq_topic = dlq_topic or s.streaming_dlq_topic
        if session_factory is None:
            from bti.database.connection import SessionLocal
            session_factory = SessionLocal
        self.session_factory = session_factory
        self.customer_resolver, self.enrich = customer_resolver, enrich
        self.default_country = default_country if default_country is not None else s.streaming_default_country
        self.max_retries = s.streaming_max_retries if max_retries is None else max_retries
        self.retry_backoff_s = retry_backoff_s
        self.metrics = metrics or Metrics()
        self._stop = threading.Event()
        self.producer = None

    # ---- broker plumbing ----
    def _client_config(self) -> Dict:
        return {"bootstrap_servers": self.bootstrap.split(","),
                "security_protocol": get_settings().streaming_security_protocol}

    def make_producer(self):
        from kafka import KafkaProducer
        return KafkaProducer(**self._client_config(), acks="all", linger_ms=2, retries=5,
                             key_serializer=lambda k: k.encode() if isinstance(k, str) else k,
                             value_serializer=lambda v: json.dumps(v, default=str, separators=(",", ":")).encode())

    def make_consumer(self, group: bool = True):
        from kafka import KafkaConsumer
        cfg = dict(self._client_config(), enable_auto_commit=False, auto_offset_reset="earliest",
                   max_poll_records=200, consumer_timeout_ms=1000)
        if group:
            return KafkaConsumer(*self.topics, group_id=self.group_id, **cfg)
        return KafkaConsumer(group_id=None, **cfg)

    def ensure_topics(self, partitions: Optional[int] = None, replication: int = 1) -> List[str]:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
        admin = KafkaAdminClient(**self._client_config())
        try:
            existing = set(admin.list_topics())
            wanted = [t for t in list(self.topics) + [self.decisions_topic, self.dlq_topic] if t not in existing]
            if wanted:
                try:
                    admin.create_topics([NewTopic(t, partitions or get_settings().streaming_partitions, replication)
                                         for t in wanted])
                except TopicAlreadyExistsError:
                    pass
            return wanted
        finally:
            admin.close()

    # ---- per-message handling (no broker needed) ----
    def handle(self, record: Record) -> List[Dict]:
        """Outgoing messages for one input record: [(topic, key, value), ...] as dicts."""
        fmt = self.topics.get(record.topic, "json")
        source = {"topic": record.topic, "partition": record.partition, "offset": record.offset}
        received = (datetime.fromtimestamp(record.timestamp_ms / 1000, tz=timezone.utc)
                    if record.timestamp_ms else datetime.now(timezone.utc))
        self.metrics.inc("consumed")
        try:
            txns = decode(fmt, record.value, received, self.customer_resolver)
        except PermanentError as exc:
            self.metrics.inc("dead_lettered")
            return [self._dead_letter(record, fmt, "decode", exc)]
        except TokenisationError:
            raise                                                   # configuration error: stop, don't DLQ
        if not txns:
            self.metrics.inc("skipped_not_scoreable")
            return []
        out = []
        for txn in txns:
            if self.default_country and not txn.get("country"):
                txn["country"] = self.default_country
            if self.enrich:
                txn = self.enrich(txn)
            out.append(self._score_with_retry(txn, record, fmt, source))
        return out

    def _score_with_retry(self, txn: Dict, record: Record, fmt: str, source: Dict) -> Dict:
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._score(txn, source)
            except PermanentError as exc:
                last = exc
                break
            except Exception as exc:                                # transient: database, store, model load
                last = exc
                self.metrics.inc("retries")
                log.warning("Scoring failed; retrying", extra={"transaction_id": txn.get("transaction_id"),
                                                               "attempt": attempt + 1, "error": str(exc)})
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
        self.metrics.inc("dead_lettered")
        return self._dead_letter(record, fmt, "score", last, transaction_id=txn.get("transaction_id"))

    def _score(self, txn: Dict, source: Dict) -> Dict:
        from bti.database.models import ScoreLog
        from bti.operations.scoring_service import score_and_decide
        db = self.session_factory()
        try:
            tid = str(txn["transaction_id"])
            prior = (db.query(ScoreLog).filter(ScoreLog.transaction_id == tid, ScoreLog.is_shadow.is_(False))
                     .order_by(ScoreLog.id.asc()).first())
            if prior is not None:
                self.metrics.inc("duplicates")
                return self._decision_message({
                    "transaction_id": tid, "customer_id": prior.customer_id, "model_id": prior.model_id,
                    "fraud_probability": prior.fraud_probability, "score": prior.score, "decision": prior.decision,
                    "reason_codes": [r["code"] for r in (prior.reason_codes or [])], "duplicate": True,
                    "scored_at": prior.scored_at.isoformat() if prior.scored_at else None}, source)
            t0 = time.perf_counter()
            sd = score_and_decide(txn, db, explain=True)
            ms = (time.perf_counter() - t0) * 1000
            self.metrics.observe(ms)
            self.metrics.inc("scored")
            self.metrics.inc(f"decision_{sd.decision.action.lower()}")
            return self._decision_message({
                "transaction_id": sd.live.transaction_id, "customer_id": txn.get("customer_id"),
                "model_id": sd.live.model_id, "provisional": sd.live.provisional,
                "fraud_probability": sd.live.fraud_probability, "score": sd.live.score,
                "decision": sd.decision.action, "reason_codes": [r["code"] for r in sd.live.reason_codes],
                "explanation": sd.explanation, "feature_path": sd.live.feature_path,
                "shadow": sd.shadow, "step_up": sd.step_up, "latency_ms": round(ms, 2), "duplicate": False,
                "scored_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "source_format": txn.get("source_format", "json")}, source)
        finally:
            db.close()

    def _decision_message(self, body: Dict, source: Dict) -> Dict:
        return {"topic": self.decisions_topic, "key": body.get("customer_id") or body["transaction_id"],
                "value": {"schema_version": SCHEMA_VERSION, **body, "source": source}}

    def _dead_letter(self, record: Record, fmt: str, stage: str, exc: Exception,
                     transaction_id: Optional[str] = None) -> Dict:
        value = {"schema_version": SCHEMA_VERSION, "stage": stage, "format": fmt,
                 "error_type": type(exc).__name__, "error": str(exc)[:500],
                 "source": {"topic": record.topic, "partition": record.partition, "offset": record.offset,
                            "timestamp_ms": record.timestamp_ms},
                 "transaction_id": transaction_id,
                 "payload_sha256": hashlib.sha256(record.value).hexdigest(), "payload_bytes": len(record.value),
                 "dead_lettered_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}
        if fmt == "json" and get_settings().streaming_dlq_include_json_payload:
            value["payload"] = record.value.decode("utf-8", "replace")
        log.warning("Dead-lettered", extra={k: value[k] for k in ("stage", "format", "error_type", "transaction_id")})
        return {"topic": self.dlq_topic, "key": record.key, "value": value}

    def _publish(self, messages: Iterable[Dict]) -> None:
        for m in messages:
            self.producer.send(m["topic"], key=m["key"], value=m["value"])

    # ---- loops ----
    def run(self, max_records: Optional[int] = None, idle_exit_s: Optional[float] = None) -> Dict:
        """Consume until stopped (or `max_records` processed / idle for `idle_exit_s`, for tests and batch runs)."""
        consumer, self.producer = self.make_consumer(), self.make_producer()
        processed, idle_since = 0, time.time()
        try:
            while not self._stop.is_set():
                batch = consumer.poll(timeout_ms=500)
                self.metrics.last_poll = time.time()
                records = [r for rs in batch.values() for r in rs]
                if not records:
                    if idle_exit_s is not None and time.time() - idle_since > idle_exit_s:
                        break
                    continue
                idle_since = time.time()
                for r in records:
                    self._publish(self.handle(Record(r.topic, r.partition, r.offset, r.key, r.value, r.timestamp)))
                self.producer.flush()
                consumer.commit()                                   # only after decisions are on the broker
                processed += len(records)
                if max_records is not None and processed >= max_records:
                    break
        finally:
            try:
                self.producer.flush()
            finally:
                consumer.close()
                self.producer.close()
        return {"processed": processed, **self.metrics.snapshot()}

    def replay(self, topic: str, partition: int, start: int, end: Optional[int] = None) -> Dict:
        """Re-process source offsets [start, end] (e.g. after a DLQ cause is fixed). Idempotent."""
        from kafka import TopicPartition
        consumer, self.producer = self.make_consumer(group=False), self.make_producer()
        tp = TopicPartition(topic, partition)
        consumer.assign([tp])
        if end is None:
            end = consumer.end_offsets([tp])[tp] - 1
        consumer.seek(tp, start)
        processed = 0
        try:
            while processed <= end - start:
                batch = consumer.poll(timeout_ms=1000)
                records = [r for r in batch.get(tp, []) if r.offset <= end]
                if not batch:
                    break
                for r in records:
                    self._publish(self.handle(Record(r.topic, r.partition, r.offset, r.key, r.value, r.timestamp)))
                    processed += 1
                if any(r.offset >= end for r in batch.get(tp, [])):
                    break
            self.producer.flush()
        finally:
            consumer.close()
            self.producer.close()
        return {"replayed": processed, "topic": topic, "partition": partition, "start": start, "end": end,
                **self.metrics.snapshot()}

    def store_events(self, txns: List[Dict], store) -> int:
        """Write transactions to the feature store without scoring them (rebuild after a store loss)."""
        from bti.modeling.scorer import _epoch
        from bti.streaming import online_features
        from bti.streaming.feature_store import event_from_online
        lookback = store.lookback_s
        written = 0
        for txn in txns:
            ts = _epoch(txn)
            if ts is None:
                continue
            customer, entity, security = store.read(txn, ts)
            f = online_features.compute(txn, ts, customer, entity, security, lookback)
            store.write_events([event_from_online(txn, ts, f)])
            written += 1
        return written

    def rebuild_store(self, since: datetime, until: Optional[datetime] = None, store=None) -> Dict:
        """
        Re-feed the feature store from the retained input topics, from `since` (store-only: no scoring, no
        decisions, no score-log rows). Use after a total Redis loss: backfill from the transactions table first,
        then rebuild from the topics for the period the table does not yet hold. Writes are idempotent.
        """
        from kafka import TopicPartition
        from bti.streaming.feature_store import get_store
        store = store or get_store()
        if store is None:
            raise RuntimeError("No feature store configured (BTI_FEATURE_STORE_URL)")
        consumer = self.make_consumer(group=False)
        parts = [TopicPartition(t, p) for t in self.topics for p in (consumer.partitions_for_topic(t) or ())]
        consumer.assign(parts)
        starts = consumer.offsets_for_times({tp: int(since.timestamp() * 1000) for tp in parts})
        ends = consumer.end_offsets(parts)
        live = []
        for tp in parts:
            if starts.get(tp) is None:
                continue
            consumer.seek(tp, starts[tp].offset)
            live.append(tp)
        stop_ms = int(until.timestamp() * 1000) if until else None
        txns, skipped = [], 0
        try:
            remaining = {tp for tp in live if consumer.position(tp) < ends[tp]}
            while remaining:
                batch = consumer.poll(timeout_ms=1000)
                if not batch:
                    break
                for tp, records in batch.items():
                    for r in records:
                        if r.offset >= ends[tp] or (stop_ms and r.timestamp > stop_ms):
                            remaining.discard(tp)
                            continue
                        try:
                            received = datetime.fromtimestamp(r.timestamp / 1000, tz=timezone.utc)
                            txns.extend(decode(self.topics[tp.topic], r.value, received, self.customer_resolver))
                        except PermanentError:
                            skipped += 1
                    if consumer.position(tp) >= ends[tp]:
                        remaining.discard(tp)
        finally:
            consumer.close()
        from bti.modeling.scorer import _epoch
        txns.sort(key=lambda t: _epoch(t) or 0)                     # customer order across partitions and topics
        return {"since": since.isoformat(), "decoded": len(txns), "skipped_undecodable": skipped,
                "written": self.store_events(txns, store)}

    def stop(self, *_):
        self._stop.set()


# ── simulator ────────────────────────────────────────────────────────────────
def _test_pan(customer_id: str) -> str:
    """A Luhn-valid PAN in the 999999 test range (never issued), stable per customer."""
    body = "999999" + str(int(hashlib.sha256(customer_id.encode()).hexdigest(), 16))[:9]
    digits = [int(c) for c in body]
    total = sum(d if i % 2 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(reversed(digits)))
    return body + str((10 - total % 10) % 10)


def to_iso8583(txn: Dict) -> bytes:
    """An ISO 8583 0100 for a BTI transaction (simulation only)."""
    numeric = {v[0]: k for k, v in iso8583.CURRENCIES.items()}
    ccy = numeric[txn["currency"]]
    exp = iso8583.CURRENCIES[ccy][1]
    d, t = str(txn["transaction_date"])[:10], str(txn.get("transaction_time") or "12:00:00")[:8]
    hhmmss = t.replace(":", "")
    return iso8583.build("0100", {
        2: _test_pan(str(txn["customer_id"])), 3: "000000",
        4: str(round(float(txn["transaction_amount"]) * 10 ** exp)).zfill(12),
        7: d[5:7] + d[8:10] + hhmmss, 11: str(int(hashlib.sha256(str(txn["transaction_id"]).encode()).hexdigest(), 16) % 1_000_000).zfill(6),
        12: hhmmss, 13: d[5:7] + d[8:10], 18: "5411", 22: "051", 25: "00", 32: "999999",
        37: str(txn["transaction_id"])[-12:].rjust(12, "0"), 41: "SIMTERM1", 42: "SIMMERCHANT0001",
        43: str(txn.get("merchant_name") or "SIM MERCHANT")[:40].ljust(40), 49: ccy})


def simulate(n: int = 500, fmt: str = "json", topic: Optional[str] = None, data_path: Optional[str] = None,
             bootstrap: Optional[str] = None) -> Dict:
    import pandas as pd
    from bti.modeling.features import event_timestamps
    from bti.modeling.train import default_data_path
    service = StreamService(bootstrap=bootstrap)
    topic = topic or next(t for t, f in service.topics.items() if f == fmt)
    df = pd.read_csv(data_path or default_data_path(), low_memory=False)
    df = df.assign(_ts=event_timestamps(df)).sort_values("_ts").tail(n).drop(columns="_ts")
    from kafka import KafkaProducer
    producer = KafkaProducer(**service._client_config(), acks="all", linger_ms=5)
    cols = [c for c in df.columns if c not in ("fraud_flag", "fraud_type", "risk_score", "label_confirmed_at")]
    for rec in df[cols].to_dict("records"):
        rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        value = json.dumps(rec, default=str).encode() if fmt == "json" else to_iso8583(rec)
        key = str(rec["customer_id"]) if fmt == "json" else _test_pan(str(rec["customer_id"]))
        producer.send(topic, key=key.encode(), value=value)
    producer.flush()
    producer.close()
    return {"published": len(df), "topic": topic, "format": fmt}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BTI Kafka streaming service")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--no-metrics", action="store_true")
    sub.add_parser("topics")
    rp = sub.add_parser("replay")
    rp.add_argument("--topic", required=True)
    rp.add_argument("--partition", type=int, required=True)
    rp.add_argument("--start", type=int, required=True)
    rp.add_argument("--end", type=int, default=None)
    rb = sub.add_parser("rebuild-store", help="re-feed the feature store from the input topics (no scoring)")
    rb.add_argument("--since", required=True, help="ISO 8601 timestamp, e.g. 2026-10-03T00:00:00+05:30")
    rb.add_argument("--until", default=None)
    sp = sub.add_parser("simulate")
    sp.add_argument("--n", type=int, default=500)
    sp.add_argument("--format", choices=["json", "iso8583"], default="json")
    sp.add_argument("--topic", default=None)
    args = parser.parse_args()
    service = StreamService()
    if args.command == "topics":
        print({"created": service.ensure_topics()})
    elif args.command == "simulate":
        print(simulate(args.n, args.format, args.topic))
    elif args.command == "rebuild-store":
        until = datetime.fromisoformat(args.until) if args.until else None
        print(json.dumps(service.rebuild_store(datetime.fromisoformat(args.since), until), indent=2))
    elif args.command == "replay":
        print(json.dumps(service.replay(args.topic, args.partition, args.start, args.end), indent=2))
    else:
        from bti.operations.residency import enforce_at_startup
        from bti.operations.scoring_service import warm_up
        enforce_at_startup("stream-consumer")
        warm_up()
        if not args.no_metrics:
            serve_metrics(service.metrics, get_settings().streaming_metrics_port)
        signal.signal(signal.SIGTERM, service.stop)
        signal.signal(signal.SIGINT, service.stop)
        log.info("Streaming service started", extra={"topics": list(service.topics), "group": service.group_id})
        print(json.dumps(service.run(), indent=2, default=str))


if __name__ == "__main__":
    main()
