# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Train/serve parity: online features against the batch features used in training.

For a sample of transactions, `parity_report` compares:
- the batch value from `build_features` over the whole frame, with
- the online value from `online_features.compute`, using only the store
  events strictly before each transaction.

The events come either from an in-memory history built from the same frame
(the default), or from a live Redis store already backfilled with that frame.

Run `python -m bti.streaming.parity` before turning on the store-backed fast
path, and after any feature change. It writes
outputs/streaming/parity_report.json. A feature passes when every sampled value
agrees to a relative 1e-9, or both are unknown.

**Model certificates.** `python -m bti.streaming.parity certify --model-id M`
scores a sample both ways with model M and compares the calibrated
probabilities. A feature version 1–2 model that uses a feature in
`online_features.V2_INEXACT` gets the fast path only with a certificate showing
no probability differs. The certificate is written to
outputs/streaming/certificates/<model>.json. Version 3 models need no
certificate, but certifying them is a useful end-to-end check.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bti.modeling.features import (
    CATEGORICAL_FEATURES, DEFAULT_LOOKBACK_DAYS, FEATURE_VERSION, NUMERIC_FEATURES, build_features, event_timestamps,
)
from bti.streaming import online_features
from bti.streaming.feature_store import ENTITY_FIELDS, events_from_frame

RTOL = 1e-9
OUT = Path("outputs/streaming/parity_report.json")
CERT_DIR = Path("outputs/streaming/certificates")


class MemoryHistory:
    """The store's read interface over an in-memory event list (no Redis needed)."""

    def __init__(self, events: List[Dict]):
        self.customer = defaultdict(list)
        self.entity = {name: defaultdict(list) for _, _, name in ENTITY_FIELDS}
        for e in events:
            self.customer[e["customer"]].append(e)
            for _, _, name in ENTITY_FIELDS:
                if e.get(name) is not None:
                    self.entity[name][e[name]].append((e["ts"], e["customer"]))

    def read(self, txn: Dict, ts: int):
        from bti.streaming.feature_store import _clean
        customer = [e for e in self.customer.get(str(txn.get("customer_id")), ()) if e["ts"] < ts]
        entity = {}
        for _, field, name in ENTITY_FIELDS:
            value = _clean(txn.get(field))
            if value is not None:
                entity[name] = [(t, c) for t, c in self.entity[name].get(value, ()) if t < ts]
        return customer, entity, None


def _same(a, b) -> bool:
    a_nan = a is None or (isinstance(a, float) and math.isnan(a))
    b_nan = b is None or (isinstance(b, float) and math.isnan(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return math.isclose(float(a), float(b), rel_tol=RTOL, abs_tol=1e-12)


def parity_report(df: pd.DataFrame, sample: int = 3000, feature_version: int = FEATURE_VERSION, seed: int = 0,
                  store=None, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Dict:
    """Per-feature mismatch counts between online and batch features on `sample` rows of `df`."""
    df = df.reset_index(drop=True)
    batch = build_features(df, lookback_days=lookback_days, feature_version=feature_version)
    ts = ((event_timestamps(df) - pd.Timestamp("1970-01-01")).dt.total_seconds()).astype("int64").to_numpy()
    history = store or MemoryHistory(events_from_frame(df, feature_version=feature_version))
    rows = np.random.default_rng(seed).choice(len(df), size=min(sample, len(df)), replace=False)
    cats = [c for c in CATEGORICAL_FEATURES if c in batch.columns]
    features = [c for c in NUMERIC_FEATURES + cats if c in batch.columns]
    mismatches: Dict[str, int] = {c: 0 for c in features}
    examples: Dict[str, Dict] = {}
    records = df.to_dict("records")
    for i in rows:
        txn = records[i]
        customer, entity, security = history.read(txn, int(ts[i]))
        f = online_features.compute(txn, int(ts[i]), customer, entity, security, lookback_days * 86400,
                                    feature_version=feature_version, categorical=cats)
        for c in features:
            b = batch.at[i, c]
            b = None if b is None or (not isinstance(b, str) and pd.isna(b)) else b
            if not _same(f.get(c), b):
                mismatches[c] += 1
                examples.setdefault(c, {"transaction_id": str(txn.get("transaction_id")), "online": f.get(c),
                                        "batch": b})
    failing = {c: n for c, n in mismatches.items() if n}
    return {"feature_version": feature_version, "rows": int(len(rows)), "features": len(features),
            "rtol": RTOL, "passing": len(features) - len(failing), "mismatches": failing,
            "examples": {c: {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in e.items()}
                         for c, e in examples.items()},
            "exact": not failing}


def online_frame(df: pd.DataFrame, rows: np.ndarray, feature_version: int, lookback_days: int,
                 store=None) -> pd.DataFrame:
    """Online feature rows (as the scorer builds them) for `rows` of `df`."""
    from bti.modeling.features import ALL_FEATURE_NAMES
    ts = ((event_timestamps(df) - pd.Timestamp("1970-01-01")).dt.total_seconds()).astype("int64").to_numpy()
    history = store or MemoryHistory(events_from_frame(df, feature_version=feature_version))
    records = df.to_dict("records")
    out = []
    for i in rows:
        customer, entity, security = history.read(records[i], int(ts[i]))
        f = online_features.compute(records[i], int(ts[i]), customer, entity, security, lookback_days * 86400,
                                    feature_version=feature_version, categorical=CATEGORICAL_FEATURES)
        out.append({c: f.get(c, np.nan) for c in ALL_FEATURE_NAMES})
    frame = pd.DataFrame(out, columns=ALL_FEATURE_NAMES)
    for c in CATEGORICAL_FEATURES:
        frame[c] = frame[c].astype("string")
    return frame


def certify_model(model_id: str, df: pd.DataFrame, sample: int = 5000, seed: int = 1, store=None) -> Dict:
    """Compare the model's calibrated probabilities on online vs batch features; write the certificate."""
    from datetime import datetime, timezone
    from bti.modeling import registry
    from bti.modeling.features import to_model_matrix
    art = registry.load_artifact(model_id)
    version, lookback, names = art.get("feature_version", 1), art["lookback_days"], art["feature_names"]
    df = df.reset_index(drop=True)
    rows = np.random.default_rng(seed).choice(len(df), size=min(sample, len(df)), replace=False)
    batch = build_features(df, lookback_days=lookback, feature_version=version).iloc[rows].reset_index(drop=True)
    online = online_frame(df, rows, version, lookback, store)

    def prob(F):
        return art["calibrator"].predict(art["estimator"].predict_proba(to_model_matrix(F, art["encodings"], names))[:, 1])
    po, pb = prob(online), prob(batch)
    diff = np.abs(po - pb)
    cert = {"model_id": model_id, "feature_version": version, "rows": int(len(rows)),
            "rows_with_different_probability": int((diff > 0).sum()), "max_abs_probability_difference": float(diff.max()),
            "inexact_features_used": sorted(online_features.V2_INEXACT & set(names)) if version < 3 else [],
            "certified": bool((diff == 0).all()),
            "certified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_rows": int(len(df)), "history": "redis" if store is not None else "in-memory"}
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    (CERT_DIR / f"{model_id}.json").write_text(json.dumps(cert, indent=2))
    _certified.cache_clear()
    return cert


def _cert_path(model_id: str) -> Path:
    return CERT_DIR / f"{model_id}.json"


@lru_cache(maxsize=64)
def _certified(model_id: str) -> bool:
    try:
        c = json.loads(_cert_path(model_id).read_text())
    except (OSError, ValueError):
        return False
    return bool(c.get("certified")) and c.get("model_id") == model_id


def online_allowed(model_id: str, art: Dict) -> bool:
    """Whether this model may be scored from the feature store."""
    if art.get("feature_version", 1) >= 3 or not (online_features.V2_INEXACT & set(art.get("feature_names", ()))):
        return True
    return _certified(model_id)


def main() -> None:
    import argparse
    from bti.modeling.train import default_data_path
    parser = argparse.ArgumentParser(description="Online/batch feature parity report")
    parser.add_argument("command", nargs="?", choices=["report", "certify"], default="report")
    parser.add_argument("--model-id", default=None, help="certify: the model (default: champion, else challenger)")
    parser.add_argument("--data", default=None)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--feature-version", type=int, default=FEATURE_VERSION)
    parser.add_argument("--redis", action="store_true", help="read history from the configured Redis store")
    args = parser.parse_args()
    df = pd.read_csv(args.data or default_data_path(), low_memory=False)
    store = None
    if args.redis:
        from bti.streaming.feature_store import get_store
        store = get_store()
    if args.command == "certify":
        from bti.modeling import registry
        model_id = args.model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
        print(json.dumps(certify_model(model_id, df, max(args.sample, 5000), store=store), indent=2))
        return
    report = parity_report(df, args.sample, args.feature_version, store=store)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("feature_version", "rows", "features", "passing", "mismatches")}))


if __name__ == "__main__":
    main()
