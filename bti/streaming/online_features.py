# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Online feature computation for one transaction from its history events.

This mirrors `bti.modeling.features.build_features` exactly, feature for
feature:
- the same windows, open on the left at t − w and closed before the
  transaction's own second
- the same novelty rule (unknown without prior history, feature version 2+)
- the same clipping and NaN handling, including the version 3 rules for
  degenerate cases (no spread in past amounts, no usual hour)

Version 3 batch features are reproduced to floating-point rounding (relative
error below 1e-9). Version 2 features match too, except five whose batch
values carry rounding noise that depends on the row's position in the frame
(the two amount sums, the amount z-score, balance share vs own and hour
deviation). `V2_INEXACT` lists them, so the scorer falls back to the batch
path for a version 2 model that uses any of them.

It works on small Python lists instead of a pandas frame, so it costs
microseconds rather than tens of milliseconds. `tests/unit/test_phase7.py`
and `bti.streaming.parity` check parity against the batch implementation on
thousands of transactions.

Inputs (from the feature store):
- `customer_events`: the customer's own events (ts, usd, hour, device, ip,
  merchant, payee, near, share, lat, lon, new_payee)
- `entity_events`: (ts, customer) pairs for the transaction's device, IP,
  merchant and payee
- `security_events`: (ts, type) for the customer, or None if there is no feed
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from bti.modeling.features import (
    IMPOSSIBLE_MIN_KM, IMPOSSIBLE_SPEED_KMH, NO_RECENT_EVENT_HOURS, SECURITY_LOOKBACK_S, STRUCTURING_THRESHOLDS_USD,
)
from bti.modeling.fx import RATES_TO_USD

NAN = float("nan")
V2_INEXACT = frozenset({"amount_zscore_customer", "balance_share_vs_own", "hour_deviation", "cust_amount_usd_24h",
                        "cust_amount_usd_7d"})
EVENT_FIELDS = ("ts", "usd", "hour", "device", "ip", "merchant", "payee", "near", "share", "lat", "lon",
                "new_payee", "txn")


def _present(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and str(v) != ""


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return NAN
    return f


def _usd(amount, currency) -> float:
    rate = RATES_TO_USD.get(str(currency).upper()) if currency is not None else None
    a = _num(amount)
    return a * rate if rate is not None and not math.isnan(a) else NAN


def near_threshold(usd: float) -> bool:
    return any(0.9 * t <= usd < t for t in STRUCTURING_THRESHOLDS_USD) if not math.isnan(usd) else False


def balance_share(amount, balance) -> float:
    a, b = _num(amount), _num(balance)
    if math.isnan(a) or math.isnan(b):
        return NAN
    return a / max(b, 1.0)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(max(a, 0.0), 1.0)))


def compute(txn: Dict, ts: int, customer_events: Sequence[Dict], entity_events: Dict[str, Sequence[Tuple[int, str]]],
            security_events: Optional[Sequence[Tuple[int, str]]], lookback_s: int, feature_version: int = 3,
            categorical: Sequence[str] = ()) -> Dict[str, object]:
    """Feature values for `txn` at epoch-second `ts`, from events strictly before `ts`."""
    customer = str(txn.get("customer_id"))
    amount, currency = txn.get("transaction_amount"), txn.get("currency")
    usd = _usd(amount, currency)
    hist_avg, balance = _num(txn.get("historical_average_transaction_amount")), _num(txn.get("account_balance_before"))
    a = _num(amount)
    hour = (ts // 3600) % 24
    weekday = ((ts // 86400) + 3) % 7                                   # 1970-01-01 was a Thursday (Mon=0)
    f: Dict[str, object] = {}
    f["amount_usd"] = usd
    f["log_amount_usd"] = math.log1p(max(usd, 0.0)) if not math.isnan(usd) else NAN
    f["amount_vs_hist_avg"] = a / hist_avg if hist_avg > 0 else NAN
    f["amount_to_balance"] = a / max(balance, 1.0) if not math.isnan(balance) else NAN
    f["balance_before_usd"] = _usd(balance, currency)
    f["failed_attempt_count"] = _num(txn.get("failed_attempt_count"))
    f["login_attempts"] = _num(txn.get("login_attempts"))
    f["txn_hour"] = float(hour)
    f["is_off_hours"] = float(hour < 6 or hour >= 22)
    f["is_weekend"] = float(weekday >= 5)
    f["is_debit"] = float(str(txn.get("debit_credit_flag")).lower() == "debit")

    def window(events, w):
        return [e for e in events if ts - w <= e["ts"] < ts]

    lb = window(customer_events, lookback_s)
    h1, h24, d7 = window(lb, 3600), window(lb, 86400), window(lb, 7 * 86400)
    f["cust_txn_count_1h"] = len(h1)
    f["cust_txn_count_24h"], f["cust_amount_usd_24h"] = len(h24), sum(_nz(e["usd"]) for e in h24)
    f["cust_txn_count_7d"], f["cust_amount_usd_7d"] = len(d7), sum(_nz(e["usd"]) for e in d7)
    earlier = [e["ts"] for e in customer_events if e["ts"] < ts]
    gap = ts - max(earlier) if earlier else NAN
    f["secs_since_last_txn"] = gap if earlier and gap <= lookback_s else NAN
    knowable = len(lb) > 0 if feature_version >= 2 else True

    def entity_counts(key, field):
        value = txn.get(field)
        if not _present(value):
            return False, 0, 0
        all_in = [c for t, c in entity_events.get(key, ()) if ts - lookback_s <= t < ts]
        mine = sum(1 for e in lb if e.get(field_map[field]) == str(value))
        return True, len(all_in), mine

    field_map = {"device_id": "device", "ip_location": "ip", "merchant_name": "merchant", "payee_id": "payee"}
    for field, key, new_col, shared_col in (("device_id", "device", "device_new_for_customer", "device_other_customer_txns"),
                                            ("ip_location", "ip", "ip_new_for_customer", "ip_other_customer_txns"),
                                            ("merchant_name", "merchant", "merchant_new_for_customer", None)):
        present, all_n, mine = entity_counts(key, field)
        f[new_col] = float(mine == 0) if present and knowable else NAN
        if shared_col:
            f[shared_col] = float(all_n - mine) if present else NAN

    merchant_present = _present(txn.get("merchant_name"))
    m = entity_events.get("merchant", ())
    f["merchant_txn_count_1h"] = sum(1 for t, _ in m if ts - 3600 <= t < ts) if merchant_present else NAN
    f["merchant_txn_count_24h"] = sum(1 for t, _ in m if ts - 86400 <= t < ts) if merchant_present else NAN
    f["device_txn_count_24h"] = (sum(1 for t, _ in entity_events.get("device", ()) if ts - 86400 <= t < ts)
                                 if _present(txn.get("device_id")) else NAN)

    n = len(lb)
    if n >= 2:
        s1 = sum(_nz(e["usd"]) for e in lb)
        s2 = sum(_nz(e["usd"]) ** 2 for e in lb)
        mean = s1 / n
        var = s2 / n - mean ** 2
        if feature_version >= 3 and not var > 1e-9 * (mean ** 2 + 1.0):
            var = 0.0
        std = math.sqrt(var) if var > 0 else 0.0
        f["amount_zscore_customer"] = (min(max((usd - mean) / std, -50.0), 50.0)
                                       if std > 0 and not math.isnan(usd) else NAN)
    else:
        f["amount_zscore_customer"] = NAN
    near = near_threshold(usd)
    f["just_below_threshold"] = float(near)
    f["cust_near_threshold_7d"] = float(sum(1 for e in d7 if e["near"]))

    if n >= 3:
        s_sin = sum(math.sin(2 * math.pi * e["hour"] / 24) for e in lb)
        s_cos = sum(math.cos(2 * math.pi * e["hour"] / 24) for e in lb)
        usual = math.atan2(s_sin, s_cos)
        angle = 2 * math.pi * hour / 24
        d = angle - usual
        defined = feature_version < 3 or math.hypot(s_sin, s_cos) > 1e-6 * n
        f["hour_deviation"] = abs(math.atan2(math.sin(d), math.cos(d))) * 24 / (2 * math.pi) if defined else NAN
    else:
        f["hour_deviation"] = NAN

    share = f["amount_to_balance"]
    if n >= 1 and not math.isnan(share):
        mean_share = sum(_nz(e["share"]) for e in lb) / n
        f["balance_share_vs_own"] = min(max(share / max(mean_share, 1e-6), 0.0), 1000.0)
    else:
        f["balance_share_vs_own"] = NAN

    lat, lon = _num(txn.get("latitude")), _num(txn.get("longitude"))
    located = [e for e in customer_events if e["ts"] < ts and not math.isnan(_num(e.get("lat")))
               and not math.isnan(_num(e.get("lon")))]
    if not math.isnan(lat) and not math.isnan(lon) and located:
        prev = max(located, key=lambda e: e["ts"])
        g = ts - prev["ts"]
        if g <= lookback_s:
            dist = _haversine_km(prev["lat"], prev["lon"], lat, lon)
            speed = dist / (max(g, 60.0) / 3600)
            f["geo_distance_prev_km"], f["geo_speed_kmh"] = dist, speed
            f["impossible_travel"] = float(speed > IMPOSSIBLE_SPEED_KMH and dist > IMPOSSIBLE_MIN_KM)
        else:
            f["geo_distance_prev_km"] = f["geo_speed_kmh"] = f["impossible_travel"] = NAN
    else:
        f["geo_distance_prev_km"] = f["geo_speed_kmh"] = f["impossible_travel"] = NAN

    payee = txn.get("payee_id")
    if _present(payee):
        mine_lb = sum(1 for e in lb if e.get("payee") == str(payee))
        new_payee = float(mine_lb == 0) if knowable else NAN
        f["payee_new_for_customer"] = new_payee
        f["cust_new_payees_24h"] = float(sum(_nz(e.get("new_payee")) for e in h24))
        mine_7d = sum(1 for e in d7 if e.get("payee") == str(payee))
        all_7d = sum(1 for t, _ in entity_events.get("payee", ()) if ts - 7 * 86400 <= t < ts)
        f["payee_other_customer_txns_7d"] = float(all_7d - mine_7d)
    else:
        f["payee_new_for_customer"] = f["cust_new_payees_24h"] = f["payee_other_customer_txns_7d"] = NAN

    if security_events is None:                         # no feed: unknown (an empty list means feed, no events)
        f["hours_since_security_event"] = f["hours_since_sim_swap"] = f["security_events_7d"] = NAN
    else:
        def since(kinds=None):
            before = [t for t, k in security_events if t < ts and (kinds is None or str(k).lower() in kinds)]
            if not before:
                return NO_RECENT_EVENT_HOURS
            age = ts - max(before)
            return age / 3600 if age <= SECURITY_LOOKBACK_S else NO_RECENT_EVENT_HOURS
        f["hours_since_security_event"] = since()
        f["hours_since_sim_swap"] = since({"sim_swap", "number_port"})
        f["security_events_7d"] = float(sum(1 for t, _ in security_events if ts - 7 * 86400 <= t < ts))

    for col in categorical:
        v = txn.get(col)
        f[col] = None if not _present(v) else str(v)
    f["_new_payee_flag"] = f.get("payee_new_for_customer")
    f["_near"] = near
    f["_share"] = share
    f["_usd"] = usd
    f["_hour"] = hour
    return f


def _nz(v) -> float:
    x = _num(v)
    return 0.0 if math.isnan(x) else x
