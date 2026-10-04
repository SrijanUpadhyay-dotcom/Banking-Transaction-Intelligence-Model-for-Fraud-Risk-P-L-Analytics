# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
ISO 20022 credit transfers to BTI transactions.

**Messages.** Two message types are accepted, in any schema version (the
namespace version is read, not enforced):
- `pacs.008`: FI-to-FI customer credit transfer. These are interbank payments
  such as SWIFT CBPR+, SEPA SCT, India's NEFT/RTGS on ISO 20022, and FedNow.
- `pain.001`: customer credit-transfer initiation, from online or API banking
  and corporate channels.

One message can carry many transfers. `parse` returns one BTI transaction per
`CdtTrfTxInf`.

**Safe parsing.** XML comes from outside the bank's trust boundary, so:
- defusedxml rejects DTDs, entity expansion ("billion laughs") and external
  entities
- the message size and the number of transfers are capped

**Personal data.**
- Account identifiers (IBAN, or `Othr/Id`) are replaced by keyed tokens
  (`bti.streaming.tokenize`). The same account gets the same token as debtor
  or creditor, so a payee seen across customers links correctly.
- Names of private persons and remittance text are never kept.
- A creditor's name is kept as `merchant_name` only when the creditor is an
  organisation (`Cdtr/Id/OrgId`).

**Mapping** for outbound payments, where the bank's customer is the debtor:
- debtor account token → `customer_id`, via `customer_resolver` (map it to the
  bank's customer number)
- creditor account token → `payee_id`, which feeds the payee and mule features
- `InstdAmt`, else `IntrBkSttlmAmt` → `transaction_amount` and `currency`
- group header `CreDtTm` → `transaction_date` and `transaction_time` (the
  creation time; wall-clock time as sent)
- `pacs.008` → "Wire Transfer"; `pain.001` → "Transfer"
- debtor `PstlAdr/Ctry` → `country` (the jurisdiction policy)
- purpose code → `merchant_category` where it maps to a trained category

Inbound payments (`direction="inbound"`) swap the roles: the creditor is the
customer, the flag is Credit. Use this for mule-account monitoring of incoming
funds.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Callable, Dict, List, Optional
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from bti.streaming.tokenize import token

MAX_BYTES = 5 * 1024 * 1024
MAX_TRANSFERS = 10_000
SUPPORTED = ("pacs.008", "pain.001")
PURPOSE_CATEGORIES = {"TAXS": "Government & Taxes", "GOVT": "Government & Taxes", "VATX": "Government & Taxes",
                      "INSU": "Insurance", "LIFI": "Insurance", "ELEC": "Utilities & Telecom",
                      "PHON": "Utilities & Telecom", "UBIL": "Utilities & Telecom", "GASB": "Utilities & Telecom",
                      "WTER": "Utilities & Telecom", "STDY": "Education", "RENT": "Real Estate",
                      "REAL": "Real Estate", "HLTC": "Healthcare & Pharmacy", "MDCS": "Healthcare & Pharmacy"}


class Iso20022Error(ValueError):
    pass


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el: Optional[Element], *path: str) -> Optional[Element]:
    for name in path:
        if el is None:
            return None
        el = next((c for c in el if _local(c.tag) == name), None)
    return el


def _children(el: Element, name: str) -> List[Element]:
    return [c for c in el if _local(c.tag) == name]


def _text(el: Optional[Element], *path: str) -> Optional[str]:
    node = _child(el, *path) if path else el
    if node is None or node.text is None:
        return None
    return node.text.strip() or None


def _account_token(party_account: Optional[Element]) -> Optional[str]:
    if party_account is None:
        return None
    iban = _text(party_account, "Id", "IBAN")
    other = _text(party_account, "Id", "Othr", "Id")
    value = iban or other
    if value is None:
        return None
    if iban and not re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}", re.sub(r"\s", "", iban.upper())):
        raise Iso20022Error("malformed IBAN")
    return token("ACCT", value)


def _amount(tx: Element) -> (Decimal, str):
    for path in (("Amt", "InstdAmt"), ("InstdAmt",), ("IntrBkSttlmAmt",)):
        node = _child(tx, *path)
        if node is not None and node.text:
            try:
                value = Decimal(node.text.strip())
            except InvalidOperation:
                raise Iso20022Error(f"bad amount {node.text!r}")
            ccy = (node.get("Ccy") or "").upper()
            if not re.fullmatch(r"[A-Z]{3}", ccy):
                raise Iso20022Error("amount without a valid Ccy attribute")
            if value <= 0:
                raise Iso20022Error("amount must be positive")
            return value, ccy
    raise Iso20022Error("transfer has no amount")


def _created(text: Optional[str]) -> datetime:
    if not text:
        raise Iso20022Error("group header has no CreDtTm")
    t = text.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(t).replace(tzinfo=None)            # wall-clock as sent
    except ValueError:
        raise Iso20022Error(f"bad CreDtTm {text!r}")


def message_type(root: Element) -> str:
    ns = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""
    match = re.search(r"(pacs\.008|pain\.001)\.\d{3}\.\d{2}", ns)
    if match:
        return match.group(0)
    body = next(iter(root), None)
    name = _local(body.tag) if body is not None else ""
    return {"FIToFICstmrCdtTrf": "pacs.008", "CstmrCdtTrfInitn": "pain.001"}.get(name, name or "unknown")


def parse(document: bytes, direction: str = "outbound",
          customer_resolver: Optional[Callable[[str], Optional[str]]] = None) -> List[Dict]:
    """BTI transactions for every credit transfer in a pacs.008 or pain.001 document."""
    if isinstance(document, str):
        document = document.encode("utf-8")
    if len(document) > MAX_BYTES:
        raise Iso20022Error(f"document larger than {MAX_BYTES} bytes")
    if direction not in ("outbound", "inbound"):
        raise Iso20022Error("direction must be outbound or inbound")
    try:
        root = SafeET.fromstring(document, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except DefusedXmlException as exc:
        raise Iso20022Error(f"unsafe XML rejected: {type(exc).__name__}")
    except SafeET.ParseError as exc:
        raise Iso20022Error(f"malformed XML: {exc}")
    kind = message_type(root)
    if not kind.startswith(SUPPORTED):
        raise Iso20022Error(f"unsupported message type {kind}; expected pacs.008 or pain.001")
    body = next(iter(root), None)
    header = _child(body, "GrpHdr")
    msg_id = _text(header, "MsgId")
    created = _created(_text(header, "CreDtTm"))

    blocks = []                                    # (debtor, debtor account, transfers)
    if kind.startswith("pain.001"):
        for pmt in _children(body, "PmtInf"):
            blocks.append((_child(pmt, "Dbtr"), _child(pmt, "DbtrAcct"), _children(pmt, "CdtTrfTxInf")))
    else:
        for tx in _children(body, "CdtTrfTxInf"):
            blocks.append((_child(tx, "Dbtr"), _child(tx, "DbtrAcct"), [tx]))
    if sum(len(b[2]) for b in blocks) > MAX_TRANSFERS:
        raise Iso20022Error(f"more than {MAX_TRANSFERS} transfers in one message")

    out = []
    for debtor, debtor_acct, transfers in blocks:
        for i, tx in enumerate(transfers):
            debtor_tok = _account_token(debtor_acct if debtor_acct is not None else _child(tx, "DbtrAcct"))
            creditor_tok = _account_token(_child(tx, "CdtrAcct"))
            customer_tok, payee_tok = (debtor_tok, creditor_tok) if direction == "outbound" else \
                (creditor_tok, debtor_tok)
            if customer_tok is None:
                raise Iso20022Error(f"transfer {i + 1}: no {'debtor' if direction == 'outbound' else 'creditor'} "
                                    f"account")
            amount, ccy = _amount(tx)
            uetr = _text(tx, "PmtId", "UETR")
            e2e = _text(tx, "PmtId", "EndToEndId")
            tx_id = _text(tx, "PmtId", "TxId") or _text(tx, "PmtId", "InstrId")
            ref = uetr or "-".join(p for p in (msg_id, tx_id or (e2e if e2e != "NOTPROVIDED" else None) or str(i + 1))
                                   if p)
            creditor = _child(tx, "Cdtr")
            org = _child(creditor, "Id", "OrgId") is not None
            party = debtor if direction == "outbound" else creditor
            out.append({
                "transaction_id": f"20022-{ref}",
                "customer_id": (customer_resolver(customer_tok) if customer_resolver else None) or customer_tok,
                "account_token": customer_tok,
                "payee_id": payee_tok,
                "transaction_date": created.strftime("%Y-%m-%d"),
                "transaction_time": created.strftime("%H:%M:%S"),
                "transaction_amount": float(amount),
                "currency": ccy,
                "channel": "Internet Banking" if kind.startswith("pain.001") else None,
                "transaction_type": "Transfer" if kind.startswith("pain.001") else "Wire Transfer",
                "debit_credit_flag": "Debit" if direction == "outbound" else "Credit",
                "authorization_method": None,
                "merchant_category": PURPOSE_CATEGORIES.get(_text(tx, "Purp", "Cd") or ""),
                "merchant_name": (_text(creditor, "Nm") or "")[:60] or None if org and direction == "outbound" else None,
                "device_id": None,
                "ip_location": None,
                "country": _text(party, "PstlAdr", "Ctry"),
                "uetr": uetr,
                "end_to_end_id": e2e,
                "source_format": f"ISO20022/{kind}",
            })
    return out
