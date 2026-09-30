"""
Step-up orchestration: challenge the customer when the decision is STEP_UP.

**Methods** (by transaction type, then channel; `stepup.methods`):
- `3ds`: card-not-present payments. BTI sends the issuer's 3-D Secure server an
  authentication request with a challenge mandate; the result (transStatus)
  comes back on the signed callback.
- `push`: approval in the bank's mobile app, through the bank's push service;
  approve or deny comes back on the signed callback.
- `sms_otp`: a six-digit one-time code sent through the bank's SMS gateway and
  entered by the customer.

**Security.**
- SMS codes are generated with `secrets` and stored only as a salted SHA-256
  hash. They are compared in constant time, with 3 attempts and a 5-minute
  expiry.
- Challenge IDs are unguessable tokens.
- Callbacks must carry an HMAC-SHA256 signature of the raw body made with
  `BTI_STEPUP_CALLBACK_SECRET`.
- Codes are never logged, except by the development `log` provider, which is
  refused in production.

**Providers.**
- `log` (development only).
- `webhook`: a signed POST to the bank's messaging, push or 3-D Secure gateway
  at `stepup.webhook_url`. A send failure marks the challenge `send_failed`, so
  the caller falls back to REVIEW.

The 3-D Secure field names follow EMV 3DS 2.x (threeDSRequestorChallengeInd,
transStatus); align the payload with the bank's 3DS server API during
integration.

**Outcomes and labels.** Challenges end `passed`, `failed`, `abandoned`
(expired unanswered) or `send_failed`. `stepup_stats` measures pass,
failure and abandonment rates per method, which cost model v2 uses.
Outcomes are not written as labels:
- a passed challenge does not prove the customer genuine (SIM-swap fraud
  passes SMS codes)
- a failed one does not prove fraud
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

from bti.config import get_settings
from bti.database.models import AuditLog, StepUpChallenge
from bti.logging_config import get_logger

log = get_logger("operations.stepup")

METHODS = ("sms_otp", "push", "3ds")
FINAL = ("passed", "failed", "abandoned", "send_failed")
TRANS_STATUS = {"Y": "passed", "A": "passed", "N": "failed", "R": "failed", "U": "failed", "C": None}


class StepUpError(ValueError):
    pass


def choose_method(channel: Optional[str], transaction_type: Optional[str]) -> Optional[str]:
    methods = get_settings().stepup_methods
    return methods.get(transaction_type) or methods.get(channel)


def _hash(salt: str, code: str) -> str:
    return hashlib.sha256(f"{salt}:{code}".encode()).hexdigest()


def _sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _row(c: StepUpChallenge) -> Dict:
    return {"challenge_id": c.id, "transaction_id": c.transaction_id, "method": c.method, "status": c.status,
            "created_at": c.created_at.isoformat(), "expires_at": c.expires_at.isoformat(),
            "attempts": c.attempts, "max_attempts": c.max_attempts, "provider": c.provider,
            "completed_at": c.completed_at.isoformat() if c.completed_at else None}


def _send(c: StepUpChallenge, message: Dict) -> Optional[str]:
    """Deliver through the configured provider; returns a provider reference or raises."""
    s = get_settings()
    if s.stepup_provider == "log":
        if s.environment == "production":
            raise StepUpError("The 'log' step-up provider is for development only; configure stepup.provider")
        log.warning("DEVELOPMENT step-up message (log provider)", extra={"challenge_id": c.id, "delivery": message})
        return f"log-{c.id[:8]}"
    if s.stepup_provider == "webhook":
        if not s.stepup_webhook_url or not s.stepup_webhook_secret:
            raise StepUpError("stepup.webhook_url and BTI_STEPUP_WEBHOOK_SECRET are required for the webhook provider")
        import requests
        body = json.dumps(message, separators=(",", ":")).encode()
        resp = requests.post(s.stepup_webhook_url, data=body, timeout=5,
                             headers={"Content-Type": "application/json",
                                      "X-BTI-Signature": _sign(s.stepup_webhook_secret, body)})
        resp.raise_for_status()
        return (resp.json() or {}).get("reference") if resp.content else None
    raise StepUpError(f"Unknown step-up provider {s.stepup_provider!r}")


def issue(db: Session, transaction_id: str, customer_id: Optional[str], channel: Optional[str],
          transaction_type: Optional[str], amount_usd: Optional[float] = None, currency: str = "USD",
          method: Optional[str] = None, callback_url: Optional[str] = None) -> Dict:
    s = get_settings()
    method = method or choose_method(channel, transaction_type)
    if method not in METHODS:
        raise StepUpError(f"No step-up method for channel {channel!r} / type {transaction_type!r}; "
                          f"configure stepup.methods or route to REVIEW")
    now = datetime.utcnow()
    c = StepUpChallenge(id=secrets.token_hex(16), transaction_id=str(transaction_id), customer_id=customer_id,
                        method=method, channel=channel, amount_usd=amount_usd, status="pending", created_at=now,
                        expires_at=now + timedelta(seconds=s.stepup_ttl_seconds), attempts=0,
                        max_attempts=s.stepup_max_attempts, provider=s.stepup_provider)
    base = {"challenge_id": c.id, "customer_id": customer_id, "transaction_id": c.transaction_id,
            "expires_at": c.expires_at.isoformat() + "Z", "callback_url": callback_url}
    if method == "sms_otp":
        code = f"{secrets.randbelow(10 ** 6):06d}"
        c.salt = secrets.token_hex(16)
        c.secret_hash = _hash(c.salt, code)
        message = {**base, "method": "sms_otp",
                   "text": f"Your code to approve this payment is {code}. It expires in "
                           f"{s.stepup_ttl_seconds // 60} minutes. Never share it — the bank will never ask for it."}
    elif method == "push":
        message = {**base, "method": "push", "prompt": "Approve this payment?",
                   "amount": round(amount_usd or 0, 2), "currency": "USD"}
    else:
        message = {**base, "method": "3ds", "messageCategory": "01", "threeDSRequestorChallengeInd": "04",
                   "purchaseAmount": round(amount_usd or 0, 2), "purchaseCurrency": currency,
                   "note": "Challenge mandated by the fraud decision; no TRA exemption applied."}
    try:
        c.provider_ref = _send(c, message)
    except Exception as exc:
        c.status, c.completed_at, c.outcome_detail = "send_failed", now, {"error": str(exc)[:300]}
        log.error("Step-up send failed", extra={"challenge_id": c.id, "error": str(exc)})
    db.add(c)
    db.add(AuditLog(ts=now, event_type="STEPUP_ISSUED", transaction_id=c.transaction_id,
                    payload={"challenge_id": c.id, "method": method, "status": c.status, "provider": c.provider}))
    db.commit()
    return _row(c)


def _get_open(db: Session, challenge_id: str) -> StepUpChallenge:
    c = db.get(StepUpChallenge, challenge_id)
    if c is None:
        raise StepUpError("Unknown challenge")
    if c.status == "pending" and datetime.utcnow() > c.expires_at:
        _finish(db, c, "abandoned", {"reason": "expired"})
    if c.status != "pending":
        raise StepUpError(f"Challenge is {c.status}")
    return c


def _finish(db: Session, c: StepUpChallenge, status: str, detail: Dict) -> Dict:
    c.status, c.completed_at, c.outcome_detail = status, datetime.utcnow(), detail
    db.add(AuditLog(ts=c.completed_at, event_type="STEPUP_COMPLETED", transaction_id=c.transaction_id,
                    payload={"challenge_id": c.id, "method": c.method, "status": status, **detail}))
    db.commit()
    return _row(c)


def verify_code(db: Session, challenge_id: str, code: str) -> Dict:
    c = _get_open(db, challenge_id)
    if c.method != "sms_otp":
        raise StepUpError(f"A {c.method} challenge is answered through its callback, not a code")
    c.attempts += 1
    ok = hmac.compare_digest(c.secret_hash or "", _hash(c.salt or "", str(code).strip()))
    if ok:
        return _finish(db, c, "passed", {"attempts": c.attempts})
    if c.attempts >= c.max_attempts:
        return _finish(db, c, "failed", {"attempts": c.attempts, "reason": "too many wrong codes"})
    db.commit()
    return {**_row(c), "detail": f"Wrong code; {c.max_attempts - c.attempts} attempt(s) left"}


def callback(db: Session, challenge_id: str, raw_body: bytes, signature: Optional[str]) -> Dict:
    secret = get_settings().stepup_callback_secret
    if not secret:
        raise StepUpError("Callbacks are disabled until BTI_STEPUP_CALLBACK_SECRET is configured")
    if not signature or not hmac.compare_digest(signature, _sign(secret, raw_body)):
        raise PermissionError("Invalid callback signature")
    body = json.loads(raw_body or b"{}")
    c = _get_open(db, challenge_id)
    if c.method == "3ds":
        status = TRANS_STATUS.get(str(body.get("transStatus", "")).upper(), "invalid")
        if status == "invalid":
            raise StepUpError("3-D Secure callback needs transStatus Y, A, N, R, U or C")
        if status is None:
            return {**_row(c), "detail": "Challenge in progress"}
        return _finish(db, c, status, {"transStatus": body["transStatus"],
                                       "transStatusReason": body.get("transStatusReason")})
    if c.method == "push":
        result = str(body.get("result", "")).lower()
        if result not in ("approved", "denied"):
            raise StepUpError("Push callback needs result 'approved' or 'denied'")
        return _finish(db, c, "passed" if result == "approved" else "failed", {"result": result})
    raise StepUpError("SMS challenges are answered with a code, not a callback")


def get(db: Session, challenge_id: str) -> Dict:
    c = db.get(StepUpChallenge, challenge_id)
    if c is None:
        raise StepUpError("Unknown challenge")
    return _row(c)


def expire(db: Session, now: Optional[datetime] = None) -> Dict:
    now = now or datetime.utcnow()
    stale = db.query(StepUpChallenge).filter(StepUpChallenge.status == "pending", StepUpChallenge.expires_at < now).all()
    for c in stale:
        c.status, c.completed_at, c.outcome_detail = "abandoned", now, {"reason": "expired"}
    if stale:
        db.add(AuditLog(ts=now, event_type="STEPUP_EXPIRED", payload={"challenges": [c.id for c in stale]}))
    db.commit()
    return {"abandoned": len(stale)}


def stepup_stats(db: Session, days: int = 30, now: Optional[datetime] = None) -> Dict:
    """Measured outcomes per method, the inputs cost model v2 needs. Fraud-after-pass uses matured labels."""
    from bti.operations.feedback import latest_labels
    now = now or datetime.utcnow()
    rows = db.query(StepUpChallenge).filter(StepUpChallenge.created_at >= now - timedelta(days=days)).all()
    frame = pd.DataFrame([{"transaction_id": c.transaction_id, "method": c.method, "status": c.status} for c in rows])
    out = {"window_days": days, "methods": {}}
    if frame.empty:
        return out
    labels = latest_labels(db, frame["transaction_id"].unique().tolist())
    fraud = set(labels.loc[labels["label"] == 1, "transaction_id"]) if not labels.empty else set()
    for method, g in frame.groupby("method"):
        done = g[g["status"].isin(("passed", "failed", "abandoned"))]
        passed, stopped = done[done["status"] == "passed"], done[done["status"] != "passed"]
        fraud_passed = int(passed["transaction_id"].isin(fraud).sum())
        fraud_stopped = int(stopped["transaction_id"].isin(fraud).sum())
        out["methods"][method] = {
            "issued": int(len(g)), "completed": int(len(done)),
            "send_failed": int((g["status"] == "send_failed").sum()),
            "passed": int(len(passed)), "failed": int((done["status"] == "failed").sum()),
            "abandoned": int((done["status"] == "abandoned").sum()),
            "abandonment_rate": round(float((done["status"] == "abandoned").mean()), 4) if len(done) else None,
            "pass_rate": round(float((done["status"] == "passed").mean()), 4) if len(done) else None,
            "fraud_passed": fraud_passed, "fraud_stopped": fraud_stopped,
            "catch_rate": round(fraud_stopped / (fraud_stopped + fraud_passed), 4)
            if fraud_stopped + fraud_passed else None,
        }
    return out
