"""
One scoring path for every API: v3 score → expected-cost decision →
shadow challenger → score log.

/v3/score, /score, /score/explain, /score/upload and the SAS enrichment API
all call `score_and_decide`, so every decision is made by the same governed
model and lands in the same audit log that drift monitoring and
champion/challenger reporting read from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from bti.database.models import ScoreLog
from bti.jurisdiction.policies import JurisdictionPolicy, policy_for
from bti.logging_config import get_logger
from bti.modeling import registry
from bti.modeling.fx import to_usd
from bti.modeling.scorer import V3Score, scorer
from bti.operations.decisioning import Decision, decide

log = get_logger("operations.scoring_service")


@dataclass
class ScoredDecision:
    live: V3Score
    decision: Decision
    policy: Optional[JurisdictionPolicy]
    amount_usd: float
    shadow: Optional[dict]


def model_available() -> bool:
    try:
        scorer.resolve("champion")
        return True
    except registry.RegistryError:
        return False


def prepare_transaction(txn: dict) -> dict:
    """Fill fields the v3 feature builder needs; missing date means 'now' (real-time scoring)."""
    out = dict(txn)
    if not out.get("transaction_date"):
        out["transaction_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    time = str(out.get("transaction_time") or "12:00:00")
    out["transaction_time"] = time if time.count(":") == 2 else f"{time}:00"
    out["currency"] = (out.get("currency") or "USD").upper()
    if not out.get("customer_id"):
        # never pool unidentified customers into one shared history
        out["customer_id"] = f"UNIDENTIFIED-{out.get('transaction_id', 'UNKNOWN')}"
    return out


def _log(db: Session, s: V3Score, txn: dict, decision: Decision, jurisdiction: Optional[str],
         amount_usd: float, shadow: bool) -> None:
    db.add(ScoreLog(
        transaction_id=s.transaction_id, customer_id=txn.get("customer_id"), model_id=s.model_id,
        model_role=s.model_role, is_shadow=shadow, fraud_probability=s.fraud_probability, score=s.score,
        decision=decision.action, jurisdiction=jurisdiction, amount_usd=amount_usd,
        reason_codes=[{k: r[k] for k in ("code", "rank", "share_of_risk")} for r in s.reason_codes],
        features=s.features, guardrails=decision.guardrails_applied, latency_ms=s.latency_ms,
        scored_at=datetime.utcnow(),
    ))


def score_and_decide(txn: dict, db: Optional[Session] = None, explain: bool = True,
                     log_scores: bool = True) -> ScoredDecision:
    txn = prepare_transaction(txn)
    policy = policy_for(txn.get("country"))
    amount_usd = to_usd(float(txn.get("transaction_amount") or 0), txn["currency"])
    live = scorer.score(txn, db_session=db, explain=explain)
    decision = decide(live.fraud_probability, amount_usd, policy, txn.get("channel"), txn.get("transaction_type"),
                      provisional_model=live.provisional)
    iso = policy.iso2 if policy else None

    shadow = None
    challenger_id = registry.model_for_role("challenger")
    if challenger_id and challenger_id != live.model_id:
        sh = scorer.score(txn, db_session=db, role="challenger", explain=False)
        sh_decision = decide(sh.fraud_probability, amount_usd, policy, txn.get("channel"),
                             txn.get("transaction_type"), provisional_model=True)
        shadow = {"model_id": sh.model_id, "fraud_probability": sh.fraud_probability, "decision": sh_decision.action}
        if db is not None and log_scores:
            _log(db, sh, txn, sh_decision, iso, amount_usd, shadow=True)

    if db is not None and log_scores:
        try:
            _log(db, live, txn, decision, iso, amount_usd, shadow=False)
            db.commit()
        except Exception as exc:
            db.rollback()
            log.error("Score log write failed", extra={"transaction_id": live.transaction_id, "error": str(exc)})

    return ScoredDecision(live=live, decision=decision, policy=policy, amount_usd=amount_usd, shadow=shadow)
