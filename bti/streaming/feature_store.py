# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Online feature store (Redis).

Live scoring used to load each customer's history from the database: about
40 ms, plus 30 ms of pandas feature computation. The store keeps compact event
logs in Redis, and `bti.streaming.online_features` computes the features from
them in microseconds. Both implementations share the feature definitions, and a
parity test holds them equal to the batch features used in training.

**Keys** (prefix `bti:fs:`). Each is a sorted set scored by event time
(epoch seconds):
- `c:<customer>` — the customer's own events: amount in USD, hour, device, IP,
  merchant, payee, near-threshold flag, balance share, location, new-payee
  flag, transaction ID
- `d:<device>`, `i:<ip>`, `m:<merchant>`, `p:<payee>` — `<txn>|<customer>`
  per event, for cross-customer counts. This includes merchant velocity, which
  the database path could not see.
- `s:<customer>` — security events (password reset, SIM swap…), once that feed
  is enabled

**Retention.** Customer, device and IP logs keep the 365-day look-back.
Merchant logs keep 1 day. Payee logs keep 7 days. Security logs keep 31 days.
Reads take only the window each feature needs, strictly before the
transaction's second.

**Writing.** Every scored transaction is written after its features are
computed, so it never counts towards itself. Backfill loads history from the
database or a file. If Redis is down, the scorer falls back to the database
path, so it is slower but still correct.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from bti.config import get_settings
from bti.logging_config import get_logger
from bti.modeling.features import DEFAULT_LOOKBACK_DAYS, build_features, event_timestamps

log = get_logger("streaming.feature_store")

RETENTION_S = {"c": DEFAULT_LOOKBACK_DAYS * 86400, "d": DEFAULT_LOOKBACK_DAYS * 86400,
               "i": DEFAULT_LOOKBACK_DAYS * 86400, "m": 86400, "p": 7 * 86400, "s": 31 * 86400}
ENTITY_FIELDS = (("d", "device_id", "device"), ("i", "ip_location", "ip"), ("m", "merchant_name", "merchant"),
                 ("p", "payee_id", "payee"))
_client = None


class FeatureStoreUnavailable(RuntimeError):
    pass


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v)
    return None if s in ("", "nan", "None", "<NA>") else s


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x            # full precision: JSON round-trips doubles exactly


class FeatureStore:
    def __init__(self, url: Optional[str] = None, prefix: str = "bti:fs:", lookback_days: int = DEFAULT_LOOKBACK_DAYS):
        import redis
        self.url = url or get_settings().feature_store_url
        if not self.url:
            raise FeatureStoreUnavailable("No feature store configured (BTI_FEATURE_STORE_URL)")
        self.r = redis.Redis.from_url(self.url, socket_timeout=0.05, socket_connect_timeout=0.2)
        self.prefix = prefix
        self.lookback_s = lookback_days * 86400

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False

    def key(self, kind: str, value: str) -> str:
        return f"{self.prefix}{kind}:{value}"

    # ── reads ──
    def read(self, txn: Dict, ts: int) -> Tuple[List[Dict], Dict[str, List[Tuple[int, str]]], Optional[List]]:
        p = self.r.pipeline(transaction=False)
        customer = str(txn.get("customer_id"))
        p.zrangebyscore(self.key("c", customer), ts - self.lookback_s, f"({ts}", withscores=False)
        wanted = []
        for kind, field, _ in ENTITY_FIELDS:
            value = _clean(txn.get(field))
            if value is not None:
                p.zrangebyscore(self.key(kind, value), ts - RETENTION_S[kind], f"({ts}", withscores=True)
                wanted.append(kind)
        p.get(self.prefix + "meta:security_feed")
        p.zrangebyscore(self.key("s", customer), ts - RETENTION_S["s"], f"({ts}", withscores=True)
        try:
            res = p.execute()
        except Exception as exc:
            raise FeatureStoreUnavailable(str(exc)) from exc
        customer_events = [dict(zip(_EVENT_KEYS, json.loads(m))) for m in res[0]]
        entity = {}
        name = {"d": "device", "i": "ip", "m": "merchant", "p": "payee"}
        for kind, rows in zip(wanted, res[1:1 + len(wanted)]):
            entity[name[kind]] = [(int(score), member.decode().split("|", 1)[1]) for member, score in rows]
        feed = res[1 + len(wanted)]
        security = [(int(score), member.decode().split("|", 1)[1]) for member, score in res[-1]] if feed else None
        return customer_events, entity, security

    # ── writes ──
    def write_events(self, events: Iterable[Dict], trim: bool = False) -> int:
        p = self.r.pipeline(transaction=False)
        n = 0
        for e in events:
            ts = int(e["ts"])
            p.zadd(self.key("c", e["customer"]), {json.dumps([e.get(k) for k in _EVENT_KEYS], separators=(",", ":")): ts})
            for kind, _, field in ENTITY_FIELDS:
                value = e.get(field)
                if value is not None:
                    p.zadd(self.key(kind, value), {f"{e['txn']}|{e['customer']}": ts})
                    if trim:
                        p.zremrangebyscore(self.key(kind, value), "-inf", f"({ts - RETENTION_S[kind]}")
            if trim:
                p.zremrangebyscore(self.key("c", e["customer"]), "-inf", f"({ts - RETENTION_S['c']}")
            n += 1
            if n % 2000 == 0:
                p.execute()
        p.execute()
        return n

    def write_security_events(self, events: Iterable[Tuple[str, int, str, str]]) -> int:
        """(customer, ts, event_type, event_id) rows; enables the security feed."""
        p = self.r.pipeline(transaction=False)
        n = 0
        for customer, ts, kind, event_id in events:
            p.zadd(self.key("s", customer), {f"{event_id}|{kind}": int(ts)})
            n += 1
        p.set(self.prefix + "meta:security_feed", 1)
        p.execute()
        return n

    def flush(self) -> int:
        keys = list(self.r.scan_iter(self.prefix + "*", count=5000))
        for i in range(0, len(keys), 5000):
            self.r.delete(*keys[i:i + 5000])
        return len(keys)


_EVENT_KEYS = ("ts", "usd", "hour", "device", "ip", "merchant", "payee", "near", "share", "lat", "lon", "new_payee",
               "txn", "customer")


def events_from_frame(df: pd.DataFrame, feature_version: int = 3) -> List[Dict]:
    """Store events for historical rows, with per-event flags computed by the batch feature code."""
    feats = build_features(df, feature_version=feature_version)
    ts = (event_timestamps(df) - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64").to_numpy()
    out = []
    cols = {c: (df[c].to_numpy(object) if c in df.columns else [None] * len(df))
            for c in ("customer_id", "device_id", "ip_location", "merchant_name", "payee_id", "latitude", "longitude",
                      "transaction_id")}
    usd, near = feats["amount_usd"].to_numpy(), feats["just_below_threshold"].to_numpy()
    share, newp, hour = feats["amount_to_balance"].to_numpy(), feats["payee_new_for_customer"].to_numpy(), \
        feats["txn_hour"].to_numpy()
    for k in range(len(df)):
        out.append({"ts": int(ts[k]), "usd": _f(usd[k]), "hour": int(hour[k]), "device": _clean(cols["device_id"][k]),
                    "ip": _clean(cols["ip_location"][k]), "merchant": _clean(cols["merchant_name"][k]),
                    "payee": _clean(cols["payee_id"][k]), "near": bool(near[k]), "share": _f(share[k]),
                    "lat": _f(cols["latitude"][k]), "lon": _f(cols["longitude"][k]), "new_payee": _f(newp[k]),
                    "txn": str(cols["transaction_id"][k]), "customer": str(cols["customer_id"][k])})
    return out


def event_from_online(txn: Dict, ts: int, f: Dict) -> Dict:
    """The store event for a live transaction, from the values the online computation already produced."""
    return {"ts": int(ts), "usd": _f(f["_usd"]), "hour": int(f["_hour"]), "device": _clean(txn.get("device_id")),
            "ip": _clean(txn.get("ip_location")), "merchant": _clean(txn.get("merchant_name")),
            "payee": _clean(txn.get("payee_id")), "near": bool(f["_near"]), "share": _f(f["_share"]),
            "lat": _f(txn.get("latitude")), "lon": _f(txn.get("longitude")), "new_payee": _f(f["_new_payee_flag"]),
            "txn": str(txn.get("transaction_id")), "customer": str(txn.get("customer_id"))}


def get_store() -> Optional[FeatureStore]:
    """The configured store, or None when none is configured."""
    global _client
    if not get_settings().feature_store_url:
        return None
    if _client is None:
        _client = FeatureStore()
    return _client


def reset_store() -> None:
    global _client
    _client = None


def backfill(source: str = "db", path: Optional[str] = None) -> Dict:
    from bti.modeling.train import default_data_path
    store = get_store()
    if store is None:
        raise FeatureStoreUnavailable("No feature store configured (BTI_FEATURE_STORE_URL)")
    if source == "db":
        from bti.database.connection import SessionLocal
        from bti.graph.snapshot import _history
        db = SessionLocal()
        try:
            df = _history(db)
            full = pd.read_sql_table("transactions", db.get_bind(), columns=[
                "transaction_id", "transaction_amount", "currency", "account_balance_before", "latitude", "longitude",
                "merchant_name"])
            df = df.merge(full, on="transaction_id", how="left")
        finally:
            db.close()
    else:
        df = pd.read_csv(path or default_data_path(), low_memory=False)
    n = store.write_events(events_from_frame(df))
    return {"events": n, "source": source}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Online feature store utilities")
    parser.add_argument("command", choices=["backfill", "flush", "ping"])
    parser.add_argument("--source", choices=["db", "csv"], default="db")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    store = get_store()
    if args.command == "ping":
        print("ok" if store and store.ping() else "unavailable")
    elif args.command == "flush":
        print(f"deleted {store.flush()} keys")
    else:
        print(backfill(args.source, args.path))


if __name__ == "__main__":
    main()
