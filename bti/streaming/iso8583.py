# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
ISO 8583 (1987) card messages to BTI transactions.

**Dialect.** ISO 8583 is a family of dialects. Each card scheme and switch
fixes its own encoding and field formats. The default `ASCII_1987` spec
covers the common case:
- ASCII characters throughout
- the MTI as 4 digits
- bitmaps as 16 hex characters (a secondary bitmap when bit 1 is set)
- LLVAR/LLLVAR length prefixes counting the characters that follow
- binary fields (PIN block, ICC data) carried as hex text

A bank configures its own switch's dialect by passing a `Spec`. A field
outside the spec stops the parse, because a variable-length field cannot be
skipped safely without its definition.

**Card data.** Card data is protected on arrival (PCI DSS requirement 3):
- The PAN (field 2, or the PAN inside track 2 when field 2 is absent) is
  replaced by a keyed token (`bti.streaming.tokenize`); only the last four
  digits are kept.
- Track data (35, 45), the PIN block (52), the CVV result data and ICC data
  (55) are dropped after parsing. Only their presence is recorded, because
  sensitive authentication data must never be stored after authorisation.
- `Iso8583Message` never holds a clear PAN, so neither a log line nor a
  dead-letter record can leak one.

**Mapping** (`to_transaction`):

| ISO 8583 | BTI transaction |
|---|---|
| tokenised PAN | `customer_id`, via `customer_resolver`; default: the card token (map it to the bank's customer number so card history joins other channels) |
| 4 + 49 | `transaction_amount` in major units (currency exponent applied), `currency` |
| 12/13 local time/date (year inferred from receipt), else 7 | `transaction_date`, `transaction_time` |
| processing code (3) | `transaction_type`, `debit_credit_flag` |
| POS entry mode (22) and condition (25) | `channel`, `authorization_method` |
| 18 MCC | `merchant_category`; 43 gives `merchant_name` |

A shared terminal is not a customer device, so `device_id` is left unknown for
card-present transactions. Balance and historical average are not carried in
a request. Supply them with the consumer's enrichment hook (core-banking
profile cache), or the model scores them as unknown and says so.

Only authorisation and financial requests (0100, 0200) and advices
(0120, 0220) are scoreable. Responses (xx10) and reversals (04xx) are
recognised and skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Tuple, Union

from bti.streaming.tokenize import last4, luhn_ok, token

# field: (kind, length rule, max length, name). kind: n numeric, an/ans text, b binary-as-hex, z track data.
# length rule: an int for fixed fields, "LL" or "LLL" for variable ones.
FieldDef = Tuple[str, Union[int, str], int, str]


@dataclass(frozen=True)
class Spec:
    name: str
    fields: Dict[int, FieldDef]
    bitmap: str = "hex"            # hex | binary


ASCII_1987 = Spec("ASCII_1987", {
    2: ("n", "LL", 19, "pan"),
    3: ("n", 6, 6, "processing_code"),
    4: ("n", 12, 12, "amount_transaction"),
    5: ("n", 12, 12, "amount_settlement"),
    6: ("n", 12, 12, "amount_cardholder_billing"),
    7: ("n", 10, 10, "transmission_datetime"),
    11: ("n", 6, 6, "stan"),
    12: ("n", 6, 6, "local_time"),
    13: ("n", 4, 4, "local_date"),
    14: ("n", 4, 4, "expiry_date"),
    18: ("n", 4, 4, "merchant_category_code"),
    19: ("n", 3, 3, "acquirer_country"),
    22: ("n", 3, 3, "pos_entry_mode"),
    23: ("n", 3, 3, "card_sequence_number"),
    25: ("n", 2, 2, "pos_condition_code"),
    26: ("n", 2, 2, "pos_pin_capture_code"),
    32: ("n", "LL", 11, "acquirer_id"),
    33: ("n", "LL", 11, "forwarding_institution_id"),
    35: ("z", "LL", 37, "track2"),
    37: ("an", 12, 12, "retrieval_reference_number"),
    38: ("an", 6, 6, "authorization_id"),
    39: ("an", 2, 2, "response_code"),
    41: ("ans", 8, 8, "terminal_id"),
    42: ("ans", 15, 15, "merchant_id"),
    43: ("ans", 40, 40, "card_acceptor_name_location"),
    45: ("ans", "LL", 76, "track1"),
    48: ("ans", "LLL", 999, "additional_data_private"),
    49: ("n", 3, 3, "currency_transaction"),
    51: ("n", 3, 3, "currency_cardholder_billing"),
    52: ("b", 16, 16, "pin_block"),
    53: ("n", 16, 16, "security_control"),
    54: ("ans", "LLL", 120, "additional_amounts"),
    55: ("b", "LLL", 999, "icc_data"),
    60: ("ans", "LLL", 999, "reserved_national_60"),
    61: ("ans", "LLL", 999, "reserved_private_61"),
    62: ("ans", "LLL", 999, "reserved_private_62"),
    63: ("ans", "LLL", 999, "reserved_private_63"),
    90: ("n", 42, 42, "original_data_elements"),
    102: ("ans", "LL", 28, "account_id_1"),
    103: ("ans", "LL", 28, "account_id_2"),
    128: ("b", 16, 16, "mac"),
})

# Sensitive authentication data: dropped after parsing, presence recorded only.
SENSITIVE_DROP = {35, 45, 52, 55}
# Account identifiers: tokenised.
TOKENISED = {102: "ACCT", 103: "ACCT"}

# ISO 4217 numeric -> (alpha, minor-unit exponent)
CURRENCIES = {"840": ("USD", 2), "978": ("EUR", 2), "826": ("GBP", 2), "356": ("INR", 2), "702": ("SGD", 2),
              "344": ("HKD", 2), "784": ("AED", 2), "566": ("NGN", 2), "392": ("JPY", 0), "414": ("KWD", 3),
              "036": ("AUD", 2), "124": ("CAD", 2), "756": ("CHF", 2), "156": ("CNY", 2), "048": ("BHD", 3),
              "682": ("SAR", 2), "458": ("MYR", 2), "360": ("IDR", 2), "710": ("ZAR", 2), "404": ("KES", 2)}

# MCC -> BTI merchant category (the categories the model was trained on). Unmapped MCCs score as unknown.
_MCC_RANGES = (
    ((3000, 3299), "Travel & Airlines"), ((4511, 4511), "Travel & Airlines"), ((4722, 4722), "Travel & Airlines"),
    ((3501, 3999), "Hotels & Accommodation"), ((7011, 7011), "Hotels & Accommodation"),
    ((5411, 5411), "Grocery & Supermarkets"), ((5422, 5499), "Grocery & Supermarkets"),
    ((5541, 5542), "Fuel & Petroleum"), ((5983, 5983), "Fuel & Petroleum"),
    ((5811, 5814), "Restaurants & Food Delivery"),
    ((5045, 5045), "Electronics & Technology"), ((5732, 5734), "Electronics & Technology"),
    ((4829, 4829), "Financial Services / Money Transfer"), ((6010, 6012), "Financial Services / Money Transfer"),
    ((6050, 6050), "Financial Services / Money Transfer"), ((6051, 6051), "Crypto Exchanges"),
    ((7995, 7995), "Gaming & Gambling"), ((7800, 7802), "Gaming & Gambling"),
    ((4812, 4816), "Utilities & Telecom"), ((4899, 4900), "Utilities & Telecom"),
    ((5122, 5122), "Healthcare & Pharmacy"), ((5912, 5912), "Healthcare & Pharmacy"),
    ((8011, 8099), "Healthcare & Pharmacy"),
    ((8211, 8299), "Education"), ((5944, 5944), "Luxury Goods"), ((5094, 5094), "Luxury Goods"),
    ((4111, 4131), "Auto & Transportation"), ((5511, 5599), "Auto & Transportation"),
    ((7512, 7549), "Auto & Transportation"),
    ((5815, 5818), "Entertainment & Streaming"), ((7832, 7841), "Entertainment & Streaming"),
    ((6300, 6399), "Insurance"), ((9211, 9399), "Government & Taxes"), ((6513, 6513), "Real Estate"),
    ((5262, 5262), "Online Marketplaces"), ((5964, 5969), "Online Marketplaces"),
    ((5200, 5399), "Retail / General Merchandise"), ((5600, 5699), "Retail / General Merchandise"),
    ((5900, 5999), "Retail / General Merchandise"),
)


class Iso8583Error(ValueError):
    pass


@dataclass
class Iso8583Message:
    mti: str
    fields: Dict[int, str]                                  # clear fields, sensitive ones removed
    card_token: Optional[str] = None
    card_last4: str = ""
    present: Dict[int, bool] = field(default_factory=dict)  # sensitive fields seen, then dropped
    spec: str = ASCII_1987.name

    @property
    def message_class(self) -> str:
        return {"1": "authorization", "2": "financial", "4": "reversal", "8": "network"}.get(self.mti[1], "other")

    @property
    def function(self) -> str:
        return {"0": "request", "1": "response", "2": "advice", "3": "advice_response"}.get(self.mti[2], "other")

    @property
    def scoreable(self) -> bool:
        return self.message_class in ("authorization", "financial") and self.function in ("request", "advice")

    def named(self, spec: Spec = ASCII_1987) -> Dict[str, str]:
        return {spec.fields[k][3]: v for k, v in self.fields.items() if k in spec.fields}


def _bitmap_fields(bits: bytes) -> list:
    return [i * 8 + j + 1 for i, byte in enumerate(bits) for j in range(8) if byte & (0x80 >> j)]


def parse(message: Union[bytes, str], spec: Spec = ASCII_1987, max_length: int = 8192) -> Iso8583Message:
    """Parse one message; card data is tokenised or dropped before this returns."""
    raw = message.encode("latin-1") if isinstance(message, str) else bytes(message)
    if len(raw) > max_length:
        raise Iso8583Error(f"message longer than {max_length} bytes")
    if len(raw) < 4 + 8:
        raise Iso8583Error("message too short")
    mti = raw[:4].decode("ascii", "replace")
    if not re.fullmatch(r"[0-9]{4}", mti):
        raise Iso8583Error("MTI must be 4 digits")
    pos = 4

    def take_bitmap(at: int) -> Tuple[bytes, int]:
        if spec.bitmap == "binary":
            return raw[at:at + 8], at + 8
        text = raw[at:at + 16].decode("ascii", "replace")
        if not re.fullmatch(r"[0-9A-Fa-f]{16}", text):
            raise Iso8583Error("bitmap must be 16 hex characters")
        return bytes.fromhex(text), at + 16

    primary, pos = take_bitmap(pos)
    present = _bitmap_fields(primary)
    if 1 in present:
        secondary, pos = take_bitmap(pos)
        present = [f for f in present if f != 1] + [f + 64 for f in _bitmap_fields(secondary)]
    values: Dict[int, str] = {}
    for f in present:
        if f not in spec.fields:
            raise Iso8583Error(f"field {f} is present but not defined in spec {spec.name}")
        kind, rule, max_len, name = spec.fields[f]
        if rule in ("LL", "LLL"):
            width = len(rule)
            prefix = raw[pos:pos + width].decode("ascii", "replace")
            if not prefix.isdigit():
                raise Iso8583Error(f"field {f} ({name}): bad length prefix")
            length, pos = int(prefix), pos + width
            if length > max_len:
                raise Iso8583Error(f"field {f} ({name}): length {length} exceeds {max_len}")
        else:
            length = int(rule)
        chunk = raw[pos:pos + length]
        if len(chunk) != length:
            raise Iso8583Error(f"field {f} ({name}): message truncated")
        pos += length
        text = chunk.decode("latin-1")
        if kind == "n" and not text.isdigit():
            raise Iso8583Error(f"field {f} ({name}): expected digits")
        if kind == "b" and not re.fullmatch(r"[0-9A-Fa-f]*", text):
            raise Iso8583Error(f"field {f} ({name}): expected hex")
        values[f] = text
    if pos != len(raw):
        raise Iso8583Error(f"{len(raw) - pos} trailing bytes after the last field")

    msg = Iso8583Message(mti=mti, fields={}, spec=spec.name)
    pan = values.pop(2, None)
    if pan is None and 35 in values:                         # track 2: PAN '=' (or 'D') expiry ...
        pan = re.split(r"[=D]", values[35], maxsplit=1)[0]
    for f in SENSITIVE_DROP:
        if f in values:
            values.pop(f)
            msg.present[f] = True
    if pan:
        if not luhn_ok(pan):
            raise Iso8583Error("PAN fails the Luhn check")
        msg.card_token, msg.card_last4 = token("PAN", pan), last4(pan)
        del pan
    for f, kind in TOKENISED.items():
        if f in values:
            values[f] = token(kind, values[f])
    msg.fields = values
    return msg


def build(mti: str, fields: Dict[int, str], spec: Spec = ASCII_1987) -> bytes:
    """Encode a message (test fixtures, simulators, replay tooling). Values are given as transmitted text."""
    numbers = sorted(fields)
    secondary = any(n > 64 for n in numbers)
    bits = bytearray(16 if secondary else 8)
    if secondary:
        bits[0] |= 0x80
    for n in numbers:
        if n < 2 or n > 128 or n == 65:
            raise Iso8583Error(f"cannot encode field {n}")
        bits[(n - 1) // 8] |= 0x80 >> ((n - 1) % 8)
    if spec.bitmap == "binary":
        out = mti.encode() + bytes(bits)
    else:
        out = mti.encode() + bits.hex().upper().encode()
    for n in numbers:
        kind, rule, max_len, name = spec.fields[n]
        v = str(fields[n])
        if rule in ("LL", "LLL"):
            if len(v) > max_len:
                raise Iso8583Error(f"field {n} ({name}) longer than {max_len}")
            out += str(len(v)).zfill(len(rule)).encode() + v.encode("latin-1")
        else:
            if len(v) != int(rule):
                raise Iso8583Error(f"field {n} ({name}) must be {rule} characters")
            out += v.encode("latin-1")
    return out


def merchant_category(mcc: Optional[str]) -> Optional[str]:
    if not mcc or not mcc.isdigit():
        return None
    code = int(mcc)
    for (lo, hi), category in _MCC_RANGES:
        if lo <= code <= hi:
            return category
    return None


def _local_datetime(msg: Iso8583Message, received_at: datetime) -> datetime:
    f = msg.fields
    if 13 in f and 12 in f:
        month, day = int(f[13][:2]), int(f[13][2:])
        hh, mm, ss = int(f[12][:2]), int(f[12][2:4]), int(f[12][4:])
    elif 7 in f:
        month, day, hh, mm, ss = (int(f[7][i:i + 2]) for i in range(0, 10, 2))
    else:
        raise Iso8583Error("no transaction date: neither fields 12/13 nor 7")
    best = None
    for year in (received_at.year - 1, received_at.year, received_at.year + 1):  # MMDD carries no year
        try:
            candidate = datetime(year, month, day, hh, mm, ss)
        except ValueError:
            continue
        if best is None or abs(candidate - received_at) < abs(best - received_at):
            best = candidate
    if best is None:
        raise Iso8583Error("invalid transaction date/time")
    return best


def to_transaction(msg: Iso8583Message, received_at: Optional[datetime] = None,
                   customer_resolver: Optional[Callable[[str], Optional[str]]] = None) -> Dict:
    """The BTI transaction for a scoreable message (see the module docstring for the mapping)."""
    if not msg.scoreable:
        raise Iso8583Error(f"MTI {msg.mti} ({msg.message_class} {msg.function}) is not scored")
    if not msg.card_token:
        raise Iso8583Error("no card number (field 2 or track 2)")
    f = msg.fields
    if 4 not in f or 49 not in f:
        raise Iso8583Error("amount (4) and currency (49) are required")
    if f[49] not in CURRENCIES:
        raise Iso8583Error(f"unsupported currency code {f[49]}")
    currency, exponent = CURRENCIES[f[49]]
    received_at = (received_at or datetime.now(timezone.utc)).replace(tzinfo=None)
    when = _local_datetime(msg, received_at)
    processing = f.get(3, "000000")[:2]
    entry, condition = f.get(22, "000")[:2], f.get(25, "00")
    ecommerce = entry in ("81", "10") or condition == "59"
    if processing == "01":
        channel, ttype = "ATM", "Withdrawal"
    elif processing == "20":
        channel, ttype = ("Internet Banking" if ecommerce else "POS Terminal"), "Refund"
    else:
        channel, ttype = ("Internet Banking", "Card Not Present") if ecommerce else ("POS Terminal", "Purchase")
    if msg.present.get(52):
        auth = "PIN"
    elif entry in ("07", "91"):
        auth = "Contactless"
    elif entry in ("05", "95") and not ecommerce:
        auth = "Signature"
    else:
        auth = None
    customer = (customer_resolver(msg.card_token) if customer_resolver else None) or msg.card_token
    parts = [f.get(32), f.get(37), f.get(11), f.get(41)]
    name = f.get(43, "")[:25].strip() or None
    return {
        "transaction_id": "8583-" + "-".join(p.strip() for p in parts if p and p.strip()),
        "customer_id": customer,
        "card_token": msg.card_token,
        "card_last4": msg.card_last4,
        "transaction_date": when.strftime("%Y-%m-%d"),
        "transaction_time": when.strftime("%H:%M:%S"),
        "transaction_amount": int(f[4]) / (10 ** exponent),
        "currency": currency,
        "channel": channel,
        "transaction_type": ttype,
        "debit_credit_flag": "Credit" if processing == "20" else "Debit",
        "authorization_method": auth,
        "merchant_category": merchant_category(f.get(18)),
        "merchant_name": name,
        "device_id": None,
        "ip_location": None,
        "country": None,                       # the account's jurisdiction: set by the consumer, not the acquirer
        "acquirer_country": f.get(19),
        "source_format": f"ISO8583/{msg.spec}/{msg.mti}",
    }
