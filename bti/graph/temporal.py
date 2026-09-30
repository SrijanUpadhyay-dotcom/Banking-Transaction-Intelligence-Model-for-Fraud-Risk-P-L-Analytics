# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Point-in-time entity graph.

Customers are linked through the devices and IP addresses they use. Payees are
tracked separately: who has paid them, and how many confirmed frauds were paid
to them.

Transactions are processed in time order, one day at a time. A transaction's
graph features come from the state at the *start of its day*:
- edges from every earlier day
- fraud confirmations made before that day (each fraud carries its
  confirmation time; the transaction's own label is never visible, because it
  is always confirmed later)

The live service uses the same code. A nightly snapshot of the state is looked
up at scoring time, so training and serving see identical, one-day-lagged
information.

**Connected components.** They are built incrementally with union-find over
devices and IPs only. Payee links would join every customer of a popular
biller into one meaningless component. Mule networks show up instead through
shared devices and IPs, and through the payee signals.

**Features** (unknown = NaN when the transaction lacks the entity):
- `graph_component_customers`: customers in the customer's device/IP
  component
- `graph_component_known_fraud`: other customers in that component with a
  confirmed fraud
- `graph_shared_entity_customers`: most other customers seen on this
  transaction's device or IP
- `graph_shared_entity_known_fraud`: confirmed-fraud customers among them
- `graph_payee_senders`: other customers who have paid this payee
- `graph_payee_known_fraud`: confirmed frauds paid to this payee
- `graph_payee_known_fraud_share`: those frauds per customer who has paid the
  payee. A mule account receives from few senders, many of them later
  confirmed as fraud; a popular biller receives from thousands, with a few
  frauds by volume alone.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from bti.modeling.features import event_timestamps

GRAPH_BASE_FEATURES = ["graph_component_customers", "graph_component_known_fraud", "graph_shared_entity_customers",
                       "graph_shared_entity_known_fraud", "graph_payee_senders", "graph_payee_known_fraud",
                       "graph_payee_known_fraud_share"]
LINK_ENTITIES = (("device_id", "d:"), ("ip_location", "i:"))


def _present(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() not in ("", "nan", "None")


class TemporalGraph:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.comp_customers: Dict[str, int] = defaultdict(int)
        self.comp_known: Dict[str, int] = defaultdict(int)
        self.known: set = set()
        self.entity_customers: Dict[str, set] = defaultdict(set)
        self.entity_known: Dict[str, int] = defaultdict(int)
        self.customer_entities: Dict[str, set] = defaultdict(set)
        self.payee_senders: Dict[str, set] = defaultdict(set)
        self.payee_known: Dict[str, int] = defaultdict(int)
        self.pending: List = []            # (confirmed_at, seq, customer, payee)
        self._seq = 0

    # ── union-find ──
    def _find(self, x: str) -> str:
        root = x
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        while self.parent.get(x, x) != root:                      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        if self.comp_customers[ra] < self.comp_customers[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.comp_customers[ra] += self.comp_customers.pop(rb, 0)
        self.comp_known[ra] += self.comp_known.pop(rb, 0)

    def _customer_node(self, customer: str) -> str:
        node = "c:" + customer
        if node not in self.parent:
            self.parent[node] = node
            self.comp_customers[node] = 1
        return node

    # ── labels ──
    def schedule_label(self, confirmed_at: pd.Timestamp, customer: str, payee: Optional[str]) -> None:
        self._seq += 1
        heapq.heappush(self.pending, (confirmed_at, self._seq, customer, payee))

    def confirm_until(self, t: pd.Timestamp) -> int:
        n = 0
        while self.pending and self.pending[0][0] < t:
            _, _, customer, payee = heapq.heappop(self.pending)
            n += 1
            if _present(payee):
                self.payee_known[str(payee)] += 1
            if customer in self.known:
                continue
            self.known.add(customer)
            node = self._customer_node(customer)
            self.comp_known[self._find(node)] += 1
            for e in self.customer_entities[customer]:
                self.entity_known[e] += 1
        return n

    # ── features and edges ──
    def features(self, customer: str, device, ip, payee) -> Dict[str, float]:
        node = "c:" + customer
        if node in self.parent:
            root = self._find(node)
            comp_c, comp_k = self.comp_customers[root], self.comp_known[root] - (customer in self.known)
        else:
            comp_c, comp_k = 1, 0
        shared_c, shared_k, any_entity = 0, 0, False
        for (col, prefix), value in zip(LINK_ENTITIES, (device, ip)):
            if not _present(value):
                continue
            any_entity = True
            e = prefix + str(value)
            others = self.entity_customers.get(e, set()) - {customer}
            shared_c = max(shared_c, len(others))
            shared_k = max(shared_k, self.entity_known.get(e, 0) - (customer in self.known and customer in
                                                                     self.entity_customers.get(e, set())))
        has_payee = _present(payee)
        return {
            "graph_component_customers": float(comp_c),
            "graph_component_known_fraud": float(comp_k),
            "graph_shared_entity_customers": float(shared_c) if any_entity else np.nan,
            "graph_shared_entity_known_fraud": float(shared_k) if any_entity else np.nan,
            "graph_payee_senders": float(len(self.payee_senders.get(str(payee), set()) - {customer}))
            if has_payee else np.nan,
            "graph_payee_known_fraud": float(self.payee_known.get(str(payee), 0)) if has_payee else np.nan,
            "graph_payee_known_fraud_share": (self.payee_known.get(str(payee), 0)
                                              / len(self.payee_senders[str(payee)]))
            if has_payee and self.payee_senders.get(str(payee)) else np.nan,
        }

    def add(self, customer: str, device, ip, payee) -> None:
        node = self._customer_node(customer)
        for (col, prefix), value in zip(LINK_ENTITIES, (device, ip)):
            if not _present(value):
                continue
            e = prefix + str(value)
            if customer not in self.entity_customers[e]:
                self.entity_customers[e].add(customer)
                self.customer_entities[customer].add(e)
                if customer in self.known:
                    self.entity_known[e] += 1
            if e not in self.parent:
                self.parent[e] = e
            self._union(node, e)
        if _present(payee):
            self.payee_senders[str(payee)].add(customer)


def confirmation_series(df: pd.DataFrame, ts: pd.Series, delay_days: int) -> pd.Series:
    """When each fraud became known: the data's own confirmation time if present, else event time + a delay."""
    fraud = pd.to_numeric(df.get("fraud_flag", 0), errors="coerce").fillna(0).astype(int) == 1
    if "label_confirmed_at" in df.columns:
        confirmed = pd.to_datetime(df["label_confirmed_at"], errors="coerce")
        return confirmed.where(fraud & confirmed.notna(), ts + pd.Timedelta(days=delay_days)).where(fraud)
    return (ts + pd.Timedelta(days=delay_days)).where(fraud)


def graph_features(df: pd.DataFrame, delay_days: int = 30, graph: Optional[TemporalGraph] = None,
                   return_graph: bool = False):
    """Point-in-time graph features for every row of `df` (indexed like `df`)."""
    ts = event_timestamps(df)
    confirmed = confirmation_series(df, ts, delay_days)
    g = graph or TemporalGraph()
    cols = {c: (df[c].to_numpy(object) if c in df.columns else np.full(len(df), None, dtype=object))
            for c in ("customer_id", "device_id", "ip_location", "payee_id")}
    customers = cols["customer_id"].astype(str)
    days = ts.dt.floor("D").to_numpy()
    order = np.argsort(ts.to_numpy(), kind="stable")
    out = np.full((len(df), len(GRAPH_BASE_FEATURES)), np.nan)
    i = 0
    while i < len(order):
        day = days[order[i]]
        j = i
        while j < len(order) and days[order[j]] == day:
            j += 1
        idx = order[i:j]
        g.confirm_until(pd.Timestamp(day))
        for k in idx:
            f = g.features(customers[k], cols["device_id"][k], cols["ip_location"][k], cols["payee_id"][k])
            out[k] = [f[c] for c in GRAPH_BASE_FEATURES]
        for k in idx:
            g.add(customers[k], cols["device_id"][k], cols["ip_location"][k], cols["payee_id"][k])
            if pd.notna(confirmed.iloc[k]):
                g.schedule_label(confirmed.iloc[k], customers[k], cols["payee_id"][k])
        i = j
    frame = pd.DataFrame(out, columns=GRAPH_BASE_FEATURES, index=df.index)
    return (frame, g) if return_graph else frame
