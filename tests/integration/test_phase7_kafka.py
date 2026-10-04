# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 7.2 on a real Kafka broker: score, publish, dead-letter, commit, and replay idempotently."""

import json
import os
import socket
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from bti.config import get_settings
from bti.streaming.consumer import Record, StreamService, decode, to_iso8583

BOOTSTRAP = os.environ.get("BTI_TEST_KAFKA", "localhost:9092")


def _broker_up() -> bool:
    host, port = BOOTSTRAP.split(",")[0].rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def _model() -> bool:
    from bti.modeling import registry
    return bool(registry.model_for_role("champion") or registry.model_for_role("challenger"))


def _txn(i, **over):
    t = {"transaction_id": f"K-{i}", "customer_id": f"C-K{i % 2}", "transaction_date": "2026-10-04",
         "transaction_time": f"10:{i:02d}:00", "transaction_amount": 120.0 + i, "currency": "GBP",
         "channel": "Mobile Banking", "transaction_type": "Transfer", "historical_average_transaction_amount": 90.0,
         "account_balance_before": 3000.0, "device_id": "DEV-K", "country": "GB"}
    t.update(over)
    return t


@pytest.fixture
def db_factory():
    from bti.database.models import Base
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "token_key", "integration-test-key")
    monkeypatch.setattr(s, "feature_store_url", "")
    monkeypatch.setattr(s, "explain_mode", "inline")


def test_handler_scores_dead_letters_and_is_idempotent_without_a_broker(db_factory):
    if not _model():
        pytest.skip("no registered model")
    svc = StreamService(topics={"in.json": "json", "in.8583": "iso8583"}, decisions_topic="out", dlq_topic="dlq",
                        session_factory=db_factory, max_retries=0)
    rec = Record("in.json", 0, 7, b"C-K1", json.dumps(_txn(1)).encode())
    first = svc.handle(rec)
    assert first[0]["topic"] == "out" and first[0]["value"]["duplicate"] is False
    again = svc.handle(rec)                                             # redelivery
    assert again[0]["value"]["duplicate"] is True
    assert again[0]["value"]["decision"] == first[0]["value"]["decision"]
    from bti.database.models import ScoreLog
    assert db_factory().query(ScoreLog).filter_by(transaction_id="K-1", is_shadow=False).count() == 1
    bad = svc.handle(Record("in.json", 0, 8, b"x", b'{"transaction_id": "K-9", "currency": "GBP"}'))
    assert bad[0]["topic"] == "dlq" and bad[0]["value"]["stage"] == "decode" and "payload" not in bad[0]["value"]
    card = svc.handle(Record("in.8583", 0, 1, b"k", to_iso8583(_txn(2))))
    assert card[0]["topic"] == "out" and card[0]["value"]["source_format"].startswith("ISO8583")
    resp = bytearray(to_iso8583(_txn(3)))
    resp[2] = ord("1")                                                  # 0100 -> 0110 response: skipped
    assert svc.handle(Record("in.8583", 0, 2, b"k", bytes(resp))) == []
    assert svc.metrics.snapshot()["counters"]["skipped_not_scoreable"] == 1


def test_transient_scoring_failures_are_retried_then_dead_lettered(db_factory, monkeypatch):
    svc = StreamService(topics={"in": "json"}, decisions_topic="out", dlq_topic="dlq", session_factory=db_factory,
                        max_retries=2, retry_backoff_s=0)
    calls = []

    def boom(txn, source):
        calls.append(1)
        raise ConnectionError("database unavailable")
    monkeypatch.setattr(svc, "_score", boom)
    out = svc.handle(Record("in", 0, 1, b"k", json.dumps(_txn(5)).encode()))
    assert len(calls) == 3 and out[0]["topic"] == "dlq" and out[0]["value"]["stage"] == "score"
    assert out[0]["value"]["transaction_id"] == "K-5"


def test_decode_rejects_unsupported_currency_and_bad_json():
    from bti.streaming.consumer import PermanentError
    with pytest.raises(PermanentError, match="currency"):
        decode("json", json.dumps(_txn(1, currency="XYZ")).encode())
    with pytest.raises(PermanentError):
        decode("json", b"{not json")


@pytest.mark.skipif(not _broker_up(), reason="no Kafka broker reachable")
def test_end_to_end_on_the_broker(db_factory):
    if not _model():
        pytest.skip("no registered model")
    from kafka import KafkaConsumer, KafkaProducer
    run = uuid.uuid4().hex[:8]
    topics = {f"bti.test.{run}.json": "json", f"bti.test.{run}.8583": "iso8583"}
    svc = StreamService(topics=topics, bootstrap=BOOTSTRAP, group_id=f"bti-test-{run}",
                        decisions_topic=f"bti.test.{run}.decisions", dlq_topic=f"bti.test.{run}.dlq",
                        session_factory=db_factory, max_retries=0)
    svc.ensure_topics(partitions=2)
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP, acks="all")
    json_topic, card_topic = list(topics)
    for i in range(4):
        producer.send(json_topic, key=f"C-K{i % 2}".encode(), value=json.dumps(_txn(i)).encode())
    producer.send(json_topic, key=b"bad", value=b'{"transaction_id": "K-BAD"}')
    producer.send(card_topic, key=b"card", value=to_iso8583(_txn(10)))
    producer.flush()
    result = svc.run(max_records=6, idle_exit_s=20)
    assert result["processed"] == 6

    def drain(topic):
        c = KafkaConsumer(topic, bootstrap_servers=BOOTSTRAP, auto_offset_reset="earliest", group_id=None,
                          consumer_timeout_ms=4000, value_deserializer=json.loads)
        msgs = [m.value for m in c]
        c.close()
        return msgs
    decisions, dlq = drain(svc.decisions_topic), drain(svc.dlq_topic)
    assert len(decisions) == 5 and len(dlq) == 1
    assert {d["transaction_id"] for d in decisions} >= {"K-0", "K-1", "K-2", "K-3"}
    assert all(d["duplicate"] is False for d in decisions) and dlq[0]["source"]["topic"] == json_topic

    # offsets were committed: a restarted service with the same group sees nothing new
    again = StreamService(topics=topics, bootstrap=BOOTSTRAP, group_id=f"bti-test-{run}",
                          decisions_topic=svc.decisions_topic, dlq_topic=svc.dlq_topic, session_factory=db_factory)
    assert again.run(idle_exit_s=5)["processed"] == 0

    # replaying the source offsets republishes the stored decisions, flagged duplicate, without rescoring
    from bti.database.models import ScoreLog
    before = db_factory().query(ScoreLog).count()
    for partition in (0, 1):
        svc.replay(json_topic, partition, 0)
    assert db_factory().query(ScoreLog).count() == before
    replayed = [d for d in drain(svc.decisions_topic) if d["duplicate"]]
    assert {d["transaction_id"] for d in replayed} == {"K-0", "K-1", "K-2", "K-3"}


@pytest.mark.skipif(not _broker_up(), reason="no Kafka broker reachable")
def test_rebuild_store_from_topics_without_scoring(db_factory):
    from datetime import datetime, timedelta, timezone
    from kafka import KafkaProducer
    from bti.streaming.parity import MemoryHistory

    class Store(MemoryHistory):
        lookback_s = 365 * 86400

        def __init__(self):
            super().__init__([])
            self.written = []

        def write_events(self, events, trim=False):
            for e in events:
                self.written.append(e)
                self.customer[e["customer"]].append(e)

    run = uuid.uuid4().hex[:8]
    topics = {f"bti.test.{run}.json": "json"}
    svc = StreamService(topics=topics, bootstrap=BOOTSTRAP, decisions_topic=f"bti.test.{run}.d",
                        dlq_topic=f"bti.test.{run}.q", session_factory=db_factory)
    svc.ensure_topics(partitions=2)
    since = datetime.now(timezone.utc) - timedelta(seconds=5)
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP, acks="all")
    for i in range(6):
        producer.send(list(topics)[0], key=f"C-K{i % 2}".encode(), value=json.dumps(_txn(i)).encode())
    producer.send(list(topics)[0], key=b"bad", value=b"not json")
    producer.flush()
    store = Store()
    r = svc.rebuild_store(since, store=store)
    assert r["decoded"] == 6 and r["written"] == 6 and r["skipped_undecodable"] == 1
    assert [e["txn"] for e in store.written] == [f"K-{i}" for i in range(6)]          # time order
    from bti.database.models import ScoreLog
    assert db_factory().query(ScoreLog).count() == 0                                  # nothing scored
