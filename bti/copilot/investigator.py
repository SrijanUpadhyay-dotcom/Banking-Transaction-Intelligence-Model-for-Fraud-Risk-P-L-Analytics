"""
BTI Copilot — AI-Powered Fraud Investigation Assistant
Phase 3 of BTI v2 (Fraud Intelligence Layer)

Wraps Claude Opus 5 with a domain-specific fraud investigation system prompt.
Provides single-shot Q&A and multi-turn sessions so investigators can drill
into rings, transactions, and entity connections in plain English.

Differentiator vs SAS: SAS alerts investigators. This copilot *explains*
and guides the investigation — what the ring means, why it scored HIGH,
what to look at next, and how to write the case narrative.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bti.logging_config import get_logger

log = get_logger("copilot.investigator")

_SYSTEM_PROMPT = """You are BTI Copilot — an AI fraud investigation assistant embedded in the \
Banking Transaction Intelligence (BTI) platform.

Your role is to help fraud investigators understand, investigate, and act on:
- Fraud ring detections (groups of transactions linked by shared devices, IPs, cards, or addresses)
- Individual high-risk transaction alerts
- Network connections between cards, devices, IPs, and email domains
- Risk patterns and recommended next steps

When analyzing a fraud ring or transaction you have access to:
- Ring membership: which transactions share devices, IPs, card tokens, or addresses
- Risk scores (0–100) and alert tiers (LOW / MEDIUM / HIGH / VERY HIGH / CRITICAL)
- Transaction amounts, merchant categories, and timestamps
- Shared entity connections — the "threads" tying a fraud ring together

Respond with:
- **Actionable** next steps an investigator can take immediately
- **Concise** leading finding — fraud analysts are time-pressured
- **Evidence-grounded** specifics: cite amounts, IDs, entity values from the data provided
- **Risk-calibrated** language: distinguish confirmed signals from plausible-but-uncertain ones

Never fabricate transaction IDs, amounts, or entity values not present in the provided context.
If context is missing, say so and suggest what data to collect."""


# ── Context formatting helpers ────────────────────────────────────────────────

def _fmt_ring(ring: dict) -> str:
    """Format a ring dict (from graph/analyze response) as investigator context."""
    lines = [
        f"## Fraud Ring: {ring.get('ring_id', 'UNKNOWN')}",
        f"- Alert tier: {ring.get('ring_alert_tier', 'UNKNOWN')}  |  Score: {ring.get('ring_risk_score', 0):.1f}/100",
        f"- Size: {ring.get('size', 0)} linked transactions",
        f"- Total exposure: ${ring.get('total_amount', 0):,.2f}  |  Avg per txn: ${ring.get('avg_amount', 0):,.2f}",
        f"- High-risk members: {ring.get('high_risk_count', 0)} of {ring.get('size', 0)}",
        f"- Velocity flag: {'YES — 5+ transactions (card-testing pattern)' if ring.get('velocity_flag') else 'No'}",
        f"- Shared entity types: {', '.join(ring.get('entity_types', []))}",
        f"- Shared entities (graph links): {', '.join(ring.get('shared_entities', [])[:10])}",
        "",
        "### Ring members (sorted by risk score):",
    ]
    for m in ring.get("members", []):
        lines.append(
            f"  - TXN {m.get('transaction_id','?')} | Cust {m.get('customer_id','?')} | "
            f"${m.get('amount', 0):,.2f} | Risk {m.get('risk_score', 0):.0f} | {m.get('alert_tier','?')} | "
            f"Links: {', '.join(m.get('shared_entities', []))}"
        )
    if ring.get("narrative"):
        lines += ["", f"### Platform narrative:", ring["narrative"]]
    if ring.get("recommendation"):
        lines += ["", f"### Platform recommendation:", ring["recommendation"]]
    return "\n".join(lines)


def _fmt_transaction(txn: dict) -> str:
    """Format a single transaction dict as investigator context."""
    lines = [
        f"## Transaction: {txn.get('transaction_id', txn.get('TransactionID', 'UNKNOWN'))}",
        f"- Customer: {txn.get('customer_id', 'UNKNOWN')}",
        f"- Amount: ${float(txn.get('transaction_amount', txn.get('TransactionAmt', 0))):,.2f}",
        f"- Risk score: {float(txn.get('final_risk_score', txn.get('risk_score', 0))):.1f}  |  Tier: {txn.get('final_alert_tier', txn.get('alert_tier', 'UNKNOWN'))}",
    ]
    for key, label in [
        ("merchant_category", "Merchant category"),
        ("merchant_name", "Merchant name"),
        ("device_id", "Device ID"),
        ("ip_location", "IP address"),
        ("card1", "Card token (primary)"),
        ("card2", "Card token (secondary)"),
        ("addr1", "Billing address"),
        ("p_emaildomain", "Email domain"),
        ("transaction_time", "Transaction time"),
    ]:
        val = txn.get(key)
        if val and str(val).lower() not in ("nan", "none", "unknown", ""):
            lines.append(f"- {label}: {val}")
    return "\n".join(lines)


def _build_context_block(rings: list, transactions: list) -> str:
    parts = ["# Investigation Context\n"]
    if rings:
        parts.append(f"## {len(rings)} Fraud Ring(s) Under Investigation\n")
        for r in rings:
            parts.append(_fmt_ring(r))
            parts.append("")
    if transactions:
        parts.append(f"## {len(transactions)} Individual Transaction(s)\n")
        for t in transactions:
            parts.append(_fmt_transaction(t))
            parts.append("")
    return "\n".join(parts)


# ── Session store ─────────────────────────────────────────────────────────────

_SESSION_TTL_SECONDS = 1800  # 30 minutes idle timeout


@dataclass
class _Session:
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class _SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, _Session] = {}

    def get_or_create(self, session_id: Optional[str]) -> _Session:
        with self._lock:
            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
                s.last_used = time.time()
                return s
            s = _Session(session_id=session_id or str(uuid.uuid4()))
            self._sessions[s.session_id] = s
            return s

    def expire_old(self):
        cutoff = time.time() - _SESSION_TTL_SECONDS
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.last_used < cutoff]
            for sid in expired:
                del self._sessions[sid]
        if expired:
            log.info("Expired copilot sessions", extra={"count": len(expired)})


_store = _SessionStore()


# ── Core copilot ──────────────────────────────────────────────────────────────

@dataclass
class CopilotResponse:
    session_id: str
    answer: str
    model: str
    input_tokens: int
    output_tokens: int


def _get_client():
    """Return an Anthropic client. Raises RuntimeError if key not configured."""
    try:
        import anthropic  # noqa: F401 (already installed at module import time)
    except ImportError:
        raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it to your Anthropic API key to enable the AI Copilot."
        )
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def ask(
    question: str,
    rings: Optional[list] = None,
    transactions: Optional[list] = None,
    session_id: Optional[str] = None,
) -> CopilotResponse:
    """
    Ask a fraud investigation question, optionally with ring/transaction context.

    Parameters
    ----------
    question      : Natural-language question from the investigator.
    rings         : List of ring dicts (from /graph/analyze response).
    transactions  : List of transaction dicts (from /score or raw data).
    session_id    : If provided, continues an existing conversation session.
                    If None, a new session is created (session_id returned in response).

    Returns
    -------
    CopilotResponse with the answer and session_id for follow-up questions.
    """
    import anthropic as ant

    client = _get_client()
    session = _store.get_or_create(session_id)

    # Build context block only on first turn in a session, or when context provided
    context_text = None
    if rings or transactions:
        context_text = _build_context_block(rings or [], transactions or [])

    user_content = question
    if context_text:
        user_content = f"{context_text}\n\n---\n\n{question}"

    session.messages.append({"role": "user", "content": user_content})

    log.info("Copilot ask", extra={
        "session_id": session.session_id,
        "question_length": len(question),
        "has_rings": bool(rings),
        "has_transactions": bool(transactions),
    })

    try:
        with client.messages.stream(
            model="claude-opus-5",
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=session.messages,
            thinking={"type": "adaptive"},
        ) as stream:
            response = stream.get_final_message()

        answer_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        session.messages.append({"role": "assistant", "content": answer_text})
        session.last_used = time.time()

        _store.expire_old()

        log.info("Copilot response", extra={
            "session_id": session.session_id,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

        return CopilotResponse(
            session_id=session.session_id,
            answer=answer_text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    except ant.AuthenticationError:
        raise RuntimeError("Invalid ANTHROPIC_API_KEY. Check your API key and retry.")
    except ant.RateLimitError:
        raise RuntimeError("Anthropic API rate limit reached. Please retry in a moment.")
    except ant.APIStatusError as e:
        log.error("Anthropic API error", extra={"status": e.status_code, "message": str(e)})
        raise RuntimeError(f"Anthropic API error ({e.status_code}): {e.message}")


def summarize_ring(ring: dict) -> CopilotResponse:
    """
    Auto-generate a structured case note summary for a detected fraud ring.
    Suitable for pasting directly into a case management system.
    """
    question = (
        "Generate a structured fraud case note for this ring that an investigator "
        "can paste directly into a case management system. Include:\n"
        "1. One-line executive summary\n"
        "2. Key evidence (shared entities, risk scores, amounts)\n"
        "3. Fraud pattern assessment (what type of fraud does this look like?)\n"
        "4. Recommended immediate actions (numbered)\n"
        "5. Investigation checklist (what to verify next)\n"
        "Keep it under 400 words."
    )
    return ask(question, rings=[ring])
