"""
/api/v1/copilot — AI Copilot for Fraud Investigators
Phase 3 of BTI v2 (Fraud Intelligence Layer)

Provides natural-language Q&A so fraud investigators can ask questions
about rings, transactions, and risk signals in plain English and receive
actionable, evidence-grounded answers powered by Claude Opus 5.

Differentiator vs SAS: SAS generates alerts. This copilot *explains*
them — what the ring means, why it scored CRITICAL, what to look at next,
and how to write the case narrative.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from bti.copilot.investigator import ask, summarize_ring
from bti.logging_config import get_logger

router = APIRouter(prefix="/copilot", tags=["AI Copilot — Fraud Investigation"])
log = get_logger("api.copilot")


# ── Request / Response schemas ────────────────────────────────────────────────

class CopilotAskRequest(BaseModel):
    question: str = Field(
        ..., min_length=5,
        description="Natural-language question from the fraud investigator"
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID from a previous /copilot/ask response to continue the conversation. "
            "Omit to start a new session."
        )
    )
    rings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Fraud ring objects from /api/v1/graph/analyze to include as context"
    )
    transactions: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Individual transaction objects to include as context"
    )

    model_config = {"json_schema_extra": {"example": {
        "question": "Why is this ring rated CRITICAL, and which transaction should I investigate first?",
        "rings": [{
            "ring_id": "RING-34D8A7FB",
            "ring_alert_tier": "CRITICAL",
            "ring_risk_score": 87.5,
            "size": 5,
            "total_amount": 22400,
            "avg_amount": 4480,
            "high_risk_count": 4,
            "velocity_flag": True,
            "entity_types": ["Device ID", "IP Address"],
            "shared_entities": ["dev:DEV-X99", "ip:198.51.100.1"],
            "narrative": "Ring connected via shared device and IP. 5 transactions at crypto merchant.",
            "recommendation": "BLOCK all 5 transactions. Escalate to Fraud Operations.",
            "members": [
                {"transaction_id": "T001", "customer_id": "C001", "amount": 4500,
                 "risk_score": 92, "alert_tier": "CRITICAL",
                 "shared_entities": ["dev:DEV-X99", "ip:198.51.100.1"]},
            ]
        }]
    }}}


class CopilotSummarizeRequest(BaseModel):
    ring: Dict[str, Any] = Field(
        ...,
        description="Fraud ring object from /api/v1/graph/analyze to summarize into a case note"
    )


class CopilotResponse(BaseModel):
    session_id: str
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    tip: str = "Pass session_id in your next /copilot/ask request to continue this conversation."


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ask",
             summary="Ask a fraud investigation question (with optional ring/transaction context)",
             response_model=CopilotResponse)
def copilot_ask(body: CopilotAskRequest):
    """
    Submit a natural-language question to BTI Copilot and receive an actionable,
    evidence-grounded answer powered by Claude Opus 5.

    **Typical uses:**
    - "Why is this ring rated CRITICAL?"
    - "Which transaction in this ring should I investigate first?"
    - "What type of fraud does this pattern look like?"
    - "What other data should I collect to confirm this is fraud?"
    - "Draft a case note for this ring."

    **Multi-turn sessions:** include the `session_id` from the previous response
    to continue the conversation. The copilot remembers prior context for up to
    30 minutes of inactivity.

    **Providing context:** pass `rings` (from `/api/v1/graph/analyze`) and/or
    `transactions` (from `/api/v1/score` or raw data) so the copilot can give
    evidence-grounded answers. If no context is provided, it will answer based
    on general fraud investigation knowledge.
    """
    try:
        result = ask(
            question=body.question,
            rings=body.rings,
            transactions=body.transactions,
            session_id=body.session_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return CopilotResponse(
        session_id=result.session_id,
        answer=result.answer,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        tip="Pass session_id in your next /copilot/ask request to continue this conversation.",
    )


@router.post("/summarize/ring",
             summary="Generate a structured case note for a detected fraud ring",
             response_model=CopilotResponse)
def copilot_summarize_ring(body: CopilotSummarizeRequest):
    """
    Auto-generate a structured fraud case note for a detected ring, ready to
    paste into a case management system.

    The case note includes:
    - One-line executive summary
    - Key evidence (shared entities, risk scores, transaction amounts)
    - Fraud pattern assessment (what type of fraud does this resemble?)
    - Recommended immediate actions (numbered)
    - Investigation checklist (what to verify next)
    """
    try:
        result = summarize_ring(body.ring)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return CopilotResponse(
        session_id=result.session_id,
        answer=result.answer,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        tip="Pass session_id in your next /copilot/ask request to ask follow-up questions about this ring.",
    )


@router.get("/health",
            summary="AI Copilot module health check")
def copilot_health():
    """Confirms the AI Copilot module is loaded and ready."""
    api_key_configured = bool(__import__("os").environ.get("ANTHROPIC_API_KEY"))
    return {
        "status":            "ok" if api_key_configured else "degraded",
        "module":            "BTI Copilot — AI Fraud Investigation Assistant",
        "version":           "3.0.0",
        "engine":            "Claude Opus 5 (claude-opus-5)",
        "api_key_configured": api_key_configured,
        "capabilities": [
            "natural-language Q&A on fraud rings and transactions",
            "multi-turn investigation sessions (30-min TTL)",
            "automated case note generation",
            "fraud pattern classification",
            "investigation checklist generation",
            "evidence-grounded answers from ring/transaction context",
        ],
        "note": (
            "Set ANTHROPIC_API_KEY environment variable to enable the copilot."
            if not api_key_configured else
            "Copilot is ready to assist fraud investigators."
        ),
    }
