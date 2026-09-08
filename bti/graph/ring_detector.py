"""
Graph / Network Analytics — Fraud Ring Detector
Phase 2 of BTI v2 (Fraud Intelligence Layer)

Builds a bipartite transaction graph where transactions are connected by
shared attributes (card, device, IP, address, email domain). Connected
components reveal coordinated fraud rings — groups of transactions linked
by common actors — which individual transaction scoring cannot detect.

Differentiator vs SAS: SAS scores each transaction in isolation.
This module finds the *network* of connected fraudulent actors, exposing
rings that coordinate across multiple cards, devices, and identities.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from bti.logging_config import get_logger

log = get_logger("graph.ring_detector")


# ── Entity types used as graph nodes ──────────────────────────────────────────
# Each is prefixed so the same value in different entity types doesn't collide.
# e.g. IP "1.2.3.4" → "ip:1.2.3.4", device "DEV-X" → "dev:DEV-X"

ENTITY_EXTRACTORS: Dict[str, str] = {
    "device_id":      "dev",
    "ip_location":    "ip",
    "card1":          "card",    # primary card token
    "card2":          "card2",   # secondary card identifier
    "addr1":          "addr",
    "P_emaildomain":  "email",
    "p_emaildomain":  "email",   # lowercase alias (BTI normalised)
}

# BTI synthetic model field names (map to same prefixes)
BTI_ENTITY_EXTRACTORS: Dict[str, str] = {
    "device_id":          "dev",
    "ip_location":        "ip",
    "merchant_name":      "merch",
    "customer_segment":   "seg",
}


# ── Risk signals used for ring scoring ────────────────────────────────────────
HIGH_RISK_MERCHANTS = {
    "Gaming & Gambling", "Crypto Exchanges",
    "Financial Services / Money Transfer", "Luxury Goods", "Online Marketplaces",
}


@dataclass
class RingMember:
    transaction_id:  str
    customer_id:     str
    amount:          float
    risk_score:      float          # final_risk_score from BTI scorer (0 if not scored)
    alert_tier:      str            # "LOW" / "MEDIUM" / "HIGH" / "VERY HIGH" / "CRITICAL"
    shared_entities: List[str]      # which entity nodes link this txn to the ring


@dataclass
class FraudRing:
    ring_id:            str
    size:               int                    # number of transactions in ring
    ring_risk_score:    float                  # 0–100
    ring_alert_tier:    str
    shared_entities:    List[str]              # entity nodes that tie the ring together
    entity_types:       List[str]              # human-readable types (Device, IP, Card, …)
    members:            List[RingMember]
    narrative:          str
    recommendation:     str
    avg_amount:         float
    total_amount:       float
    high_risk_count:    int                    # members with alert_tier HIGH+
    velocity_flag:      bool                   # ≥5 txns in this ring


@dataclass
class RingDetectionResult:
    total_transactions:  int
    transactions_in_rings: int
    rings_detected:      int
    rings_critical:      int
    rings_high:          int
    rings:               List[FraudRing]
    isolated_count:      int                  # transactions not in any ring
    summary_narrative:   str


# ── Graph builder ─────────────────────────────────────────────────────────────

def _entity_key(prefix: str, value: Any) -> Optional[str]:
    """Normalise and prefix an entity value. Returns None if value is empty/null."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("", "nan", "none", "unknown", "n/a", "0", "0.0"):
        return None
    return f"{prefix}:{v}"


def _extract_entities(txn: dict) -> List[str]:
    """Extract all linkable entity keys from a transaction dict."""
    entities: List[str] = []
    extractors = {**ENTITY_EXTRACTORS, **BTI_ENTITY_EXTRACTORS}
    for field_name, prefix in extractors.items():
        key = _entity_key(prefix, txn.get(field_name))
        if key and key not in entities:
            entities.append(key)
    return entities


def build_transaction_graph(transactions: List[dict]) -> nx.Graph:
    """
    Build a bipartite graph:
      - Transaction nodes: prefixed "txn:<transaction_id>"
      - Entity nodes:      prefixed "<type>:<value>"
      - Edges: transaction → entity (transaction uses this entity)

    Two transactions are in the same ring if they share ≥1 entity node,
    making them part of the same connected component.
    """
    G = nx.Graph()

    for txn in transactions:
        tid = str(txn.get("transaction_id") or txn.get("TransactionID") or "UNKNOWN")
        txn_node = f"txn:{tid}"
        G.add_node(txn_node, node_type="transaction", data=txn)

        entities = _extract_entities(txn)
        for ent in entities:
            G.add_node(ent, node_type="entity")
            G.add_edge(txn_node, ent)

    log.info(
        "Transaction graph built",
        extra={"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
               "transactions": len(transactions)},
    )
    return G


# ── Ring detector ─────────────────────────────────────────────────────────────

def _ring_id(nodes: Set[str]) -> str:
    """Stable short ID for a ring based on its transaction nodes."""
    txn_nodes = sorted(n for n in nodes if n.startswith("txn:"))
    h = hashlib.sha1("|".join(txn_nodes).encode()).hexdigest()[:8]
    return f"RING-{h.upper()}"


def _entity_type_label(entity_node: str) -> str:
    prefix = entity_node.split(":")[0]
    return {
        "dev":   "Device ID",
        "ip":    "IP Address",
        "card":  "Card Token",
        "card2": "Card Identifier",
        "addr":  "Address",
        "email": "Email Domain",
        "merch": "Merchant",
        "seg":   "Customer Segment",
    }.get(prefix, prefix.title())


def _score_ring(
    txn_nodes: List[str],
    entity_nodes: List[str],
    G: nx.Graph,
) -> Tuple[float, str]:
    """
    Compute a ring risk score (0–100) based on:
    - Size: more transactions = more suspicious
    - Shared entity density: more shared entities = tighter ring
    - High-risk entity types (device sharing > IP sharing > email)
    - Presence of high-risk merchant categories
    - Off-hours concentration
    - Velocity (many txns)
    """
    score = 0.0

    n = len(txn_nodes)
    # Size signal (up to 25 pts)
    score += min(n / 20 * 25, 25)

    # Entity density: how many unique entity types shared (up to 30 pts)
    entity_prefixes = {e.split(":")[0] for e in entity_nodes}
    high_value_prefixes = {"dev", "card", "card2", "ip"}
    shared_high = len(entity_prefixes & high_value_prefixes)
    score += min(shared_high * 10, 30)

    # Member risk signals (up to 30 pts)
    risk_sum = 0.0
    off_hours_count = 0
    high_risk_merchant_count = 0

    for txn_node in txn_nodes:
        txn_data = G.nodes[txn_node].get("data", {})
        risk = float(txn_data.get("final_risk_score") or txn_data.get("risk_score") or 0)
        risk_sum += risk

        time_str = str(txn_data.get("transaction_time") or "12:00")[:2]
        hour = int(time_str) if time_str.isdigit() else 12
        if hour < 6 or hour >= 22:
            off_hours_count += 1

        merch = str(txn_data.get("merchant_category") or "")
        if merch in HIGH_RISK_MERCHANTS:
            high_risk_merchant_count += 1

    avg_risk = risk_sum / max(n, 1)
    score += min(avg_risk / 100 * 20, 20)
    # Amplifier: all members are high-risk (most dangerous signal)
    if avg_risk >= 75:
        score += 15
    elif avg_risk >= 50:
        score += 7
    score += min(off_hours_count / max(n, 1) * 8, 8)
    score += min(high_risk_merchant_count / max(n, 1) * 7, 7)

    # Velocity bonus (up to 15 pts)
    if n >= 10:
        score += 15
    elif n >= 5:
        score += 8

    score = round(min(score, 100), 2)

    if score >= 80:
        tier = "CRITICAL"
    elif score >= 60:
        tier = "HIGH"
    elif score >= 40:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return score, tier


def _build_narrative(ring: "FraudRing") -> str:
    entity_desc = ", ".join(ring.entity_types[:3])
    lines = [
        f"Ring {ring.ring_id} contains {ring.size} linked transactions "
        f"connected via shared {entity_desc}.",
        f"Total exposure: ${ring.total_amount:,.2f} "
        f"(avg ${ring.avg_amount:,.2f} per transaction).",
    ]
    if ring.high_risk_count:
        lines.append(
            f"{ring.high_risk_count} of {ring.size} transactions are rated "
            f"HIGH risk or above — this ring shows coordinated fraud indicators."
        )
    if ring.velocity_flag:
        lines.append(
            "Velocity flag: 5+ transactions in this ring — consistent with "
            "automated or coordinated card-testing activity."
        )
    return " ".join(lines)


def _recommendation(tier: str, size: int) -> str:
    if tier == "CRITICAL":
        return (
            f"BLOCK all {size} transactions. Escalate to Fraud Operations immediately. "
            "Initiate card/account suspension for all entities in this ring."
        )
    if tier == "HIGH":
        return (
            f"HOLD all {size} transactions pending manual review. "
            "Flag all linked cards, devices, and IPs for enhanced monitoring."
        )
    if tier == "MEDIUM":
        return (
            "MONITOR ring activity. Apply step-up authentication to all linked accounts. "
            "Review transaction history for the past 30 days."
        )
    return "ALLOW with standard monitoring. Review if ring grows."


# ── Main entry point ──────────────────────────────────────────────────────────

def detect_fraud_rings(
    transactions: List[dict],
    min_ring_size: int = 2,
) -> RingDetectionResult:
    """
    Analyse a batch of transactions for fraud ring patterns.

    Parameters
    ----------
    transactions : list of transaction dicts. Each dict should have at minimum:
        transaction_id, customer_id, transaction_amount.
        Optionally: device_id, ip_location, card1, card2, addr1, P_emaildomain,
        merchant_category, transaction_time, risk_score / final_risk_score.

    min_ring_size : minimum number of transactions to qualify as a ring (default 2).

    Returns
    -------
    RingDetectionResult with all detected rings and summary statistics.
    """
    if not transactions:
        return RingDetectionResult(0, 0, 0, 0, 0, [], 0, "No transactions provided.")

    G = build_transaction_graph(transactions)

    # Find connected components and separate transaction nodes from entity nodes
    rings: List[FraudRing] = []
    isolated_count = 0
    transactions_in_rings = 0

    for component in nx.connected_components(G):
        txn_nodes   = [n for n in component if n.startswith("txn:")]
        entity_nodes = [n for n in component if not n.startswith("txn:")]

        if len(txn_nodes) < min_ring_size:
            isolated_count += len(txn_nodes)
            continue

        # Score the ring
        ring_score, ring_tier = _score_ring(txn_nodes, entity_nodes, G)

        # Build member list
        members: List[RingMember] = []
        for txn_node in txn_nodes:
            txn_data = G.nodes[txn_node].get("data", {})
            tid = str(txn_data.get("transaction_id") or txn_data.get("TransactionID") or txn_node)

            # Which entity nodes is this transaction connected to?
            shared = [
                n for n in G.neighbors(txn_node)
                if not n.startswith("txn:")
            ]

            members.append(RingMember(
                transaction_id  = tid,
                customer_id     = str(txn_data.get("customer_id") or "UNKNOWN"),
                amount          = float(txn_data.get("transaction_amount") or
                                        txn_data.get("TransactionAmt") or 0),
                risk_score      = float(txn_data.get("final_risk_score") or
                                        txn_data.get("risk_score") or 0),
                alert_tier      = str(txn_data.get("final_alert_tier") or "UNKNOWN"),
                shared_entities = shared,
            ))

        members.sort(key=lambda m: m.risk_score, reverse=True)

        amounts       = [m.amount for m in members]
        total_amount  = sum(amounts)
        avg_amount    = total_amount / max(len(amounts), 1)
        high_risk_count = sum(
            1 for m in members
            if m.alert_tier in ("HIGH", "VERY HIGH", "CRITICAL")
        )

        entity_type_labels = sorted({_entity_type_label(e) for e in entity_nodes})

        ring = FraudRing(
            ring_id          = _ring_id(set(txn_nodes)),
            size             = len(txn_nodes),
            ring_risk_score  = ring_score,
            ring_alert_tier  = ring_tier,
            shared_entities  = entity_nodes,
            entity_types     = entity_type_labels,
            members          = members,
            narrative        = "",          # filled below
            recommendation   = _recommendation(ring_tier, len(txn_nodes)),
            avg_amount       = round(avg_amount, 2),
            total_amount     = round(total_amount, 2),
            high_risk_count  = high_risk_count,
            velocity_flag    = len(txn_nodes) >= 5,
        )
        ring.narrative = _build_narrative(ring)
        rings.append(ring)
        transactions_in_rings += len(txn_nodes)

    rings.sort(key=lambda r: r.ring_risk_score, reverse=True)

    rings_critical = sum(1 for r in rings if r.ring_alert_tier == "CRITICAL")
    rings_high     = sum(1 for r in rings if r.ring_alert_tier == "HIGH")

    summary = (
        f"Analysed {len(transactions)} transactions. "
        f"Detected {len(rings)} fraud ring{'s' if len(rings) != 1 else ''} "
        f"involving {transactions_in_rings} linked transactions. "
        f"{rings_critical} CRITICAL, {rings_high} HIGH severity. "
        f"{isolated_count} transactions show no network links."
    )

    log.info("Ring detection complete", extra={
        "total": len(transactions), "rings": len(rings),
        "in_rings": transactions_in_rings, "isolated": isolated_count,
    })

    return RingDetectionResult(
        total_transactions     = len(transactions),
        transactions_in_rings  = transactions_in_rings,
        rings_detected         = len(rings),
        rings_critical         = rings_critical,
        rings_high             = rings_high,
        rings                  = rings,
        isolated_count         = isolated_count,
        summary_narrative      = summary,
    )
