# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""Phase 7.1: ISO 8583 and ISO 20022 adapters, tokenisation and safe parsing."""

from datetime import datetime
from pathlib import Path

import pytest

from bti.config import get_settings
from bti.streaming import iso8583, iso20022
from bti.streaming.tokenize import TokenisationError, luhn_ok, token

FIX = Path("tests/fixtures/iso20022")
PAN = "4111111111111111"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(get_settings(), "token_key", "unit-test-key")


def _auth(over=None):
    fields = {2: PAN, 3: "000000", 4: "000000012550", 7: "1004101530", 11: "123456", 12: "101530", 13: "1004",
              18: "5411", 22: "051", 25: "00", 32: "123456", 37: "000000000001", 41: "TERM0001",
              42: "MERCHANT0000001", 43: "GREEN GROCER LTD         LONDON       GB", 49: "826", 52: "A1B2C3D4E5F60708"}
    fields.update(over or {})
    return {k: v for k, v in fields.items() if v is not None}


def test_tokens_are_keyed_stable_and_never_contain_the_number(monkeypatch):
    a = token("PAN", PAN)
    assert a == token("PAN", "4111 1111 1111 1111") and PAN not in a and a.startswith("PAN-")
    monkeypatch.setattr(get_settings(), "token_key", "another-key")
    assert token("PAN", PAN) != a                                  # without the key the token cannot be recomputed
    monkeypatch.setattr(get_settings(), "token_key", "")
    monkeypatch.setattr(get_settings(), "environment", "production")
    with pytest.raises(TokenisationError):
        token("PAN", PAN)
    assert luhn_ok(PAN) and not luhn_ok("4111111111111112")


def test_iso8583_round_trip_tokenises_pan_and_drops_sensitive_data():
    raw = iso8583.build("0100", _auth())
    msg = iso8583.parse(raw)
    assert msg.mti == "0100" and msg.scoreable and msg.card_last4 == "1111"
    assert 2 not in msg.fields and 52 not in msg.fields and msg.present == {52: True}
    assert PAN not in repr(msg) and PAN not in str(msg.fields)
    t = iso8583.to_transaction(msg, received_at=datetime(2026, 10, 4, 10, 20))
    assert t["customer_id"] == token("PAN", PAN) and PAN not in str(t)
    assert (t["transaction_amount"], t["currency"], t["transaction_date"], t["transaction_time"]) == \
        (125.5, "GBP", "2026-10-04", "10:15:30")
    assert (t["channel"], t["transaction_type"], t["authorization_method"], t["merchant_category"]) == \
        ("POS Terminal", "Purchase", "PIN", "Grocery & Supermarkets")


def test_iso8583_secondary_bitmap_track2_and_mappings():
    fields = _auth({2: None, 52: None, 22: "812", 25: "59", 3: "000000", 49: "392", 4: "000000012345",
                      35: PAN + "=29121011234567890", 102: "GB29NWBK60161331926819"})
    msg = iso8583.parse(iso8583.build("0200", fields))
    assert msg.card_token == token("PAN", PAN) and msg.present == {35: True}
    assert msg.fields[102] == token("ACCT", "GB29NWBK60161331926819")
    t = iso8583.to_transaction(msg, received_at=datetime(2026, 10, 4, 10, 20))
    assert t["transaction_amount"] == 12345 and t["currency"] == "JPY"            # exponent 0
    assert (t["channel"], t["transaction_type"], t["authorization_method"]) == ("Internet Banking", "Card Not Present",
                                                                                 None)
    atm = iso8583.to_transaction(iso8583.parse(iso8583.build("0200", _auth({3: "010000", 18: "6011"}))),
                                 received_at=datetime(2026, 10, 4))
    assert (atm["channel"], atm["transaction_type"]) == ("ATM", "Withdrawal")


def test_iso8583_year_is_inferred_across_new_year():
    msg = iso8583.parse(iso8583.build("0100", _auth({13: "1231", 12: "235959"})))
    assert iso8583.to_transaction(msg, received_at=datetime(2027, 1, 1, 0, 0, 5))["transaction_date"] == "2026-12-31"


@pytest.mark.parametrize("mutate, error", [
    (lambda r: r[:-3], "trailing|truncated"),
    (lambda r: r + b"XX", "trailing"),
    (lambda r: b"01A0" + r[4:], "MTI"),
    (lambda r: r[:4] + b"ZZZZ" + r[8:], "bitmap"),
])
def test_iso8583_rejects_malformed_messages(mutate, error):
    with pytest.raises(iso8583.Iso8583Error, match=error):
        iso8583.parse(mutate(iso8583.build("0100", _auth())))


def test_iso8583_rejects_bad_pan_unknown_fields_and_skips_responses():
    with pytest.raises(iso8583.Iso8583Error, match="Luhn"):
        iso8583.parse(iso8583.build("0100", _auth({2: "4111111111111112"})))
    raw = bytearray(iso8583.build("0100", _auth()))
    raw[4 + 2] = ord("F")                                           # turn on undefined fields 9-12 in the bitmap
    with pytest.raises(iso8583.Iso8583Error):
        iso8583.parse(bytes(raw))
    resp = iso8583.parse(iso8583.build("0110", _auth({39: "00"})))
    assert not resp.scoreable
    with pytest.raises(iso8583.Iso8583Error, match="not scored"):
        iso8583.to_transaction(resp)
    with pytest.raises(iso8583.Iso8583Error, match="currency"):
        iso8583.to_transaction(iso8583.parse(iso8583.build("0100", _auth({49: "999"}))))


def test_iso20022_pacs008_maps_each_transfer_and_keeps_no_personal_text():
    raw = (FIX / "pacs008_sample.xml").read_bytes()
    txns = iso20022.parse(raw)
    assert len(txns) == 2
    a, b = txns
    assert a["transaction_id"] == "20022-3f1c8a52-6c3e-4d6f-9b8e-0c6a1d2e7f41"
    assert (a["transaction_amount"], a["currency"], a["transaction_time"]) == (2450.0, "EUR", "09:41:07")
    assert a["customer_id"] == token("ACCT", "DE89370400440532013000") == b["customer_id"]
    assert a["merchant_name"] == "Example Utilities SA" and b["merchant_name"] is None   # organisation vs person
    assert a["merchant_category"] == "Utilities & Telecom" and a["country"] == "DE"
    assert b["transaction_id"] == "20022-BTIBANK20261004-0001-TX-0002"
    text = str(txns)
    for secret in ("DE89370400440532013000", "Private Person", "remittance"):
        assert secret not in text


def test_iso20022_pain001_and_payee_links_across_messages():
    pain = iso20022.parse((FIX / "pain001_sample.xml").read_bytes())[0]
    pacs = iso20022.parse((FIX / "pacs008_sample.xml").read_bytes())[1]
    assert pain["payee_id"] == pacs["payee_id"]                      # same creditor account, same token
    assert (pain["transaction_type"], pain["channel"], pain["merchant_category"]) == ("Transfer", "Internet Banking",
                                                                                      "Real Estate")
    inbound = iso20022.parse((FIX / "pain001_sample.xml").read_bytes(), direction="inbound")[0]
    assert inbound["customer_id"] == pain["payee_id"] and inbound["debit_credit_flag"] == "Credit"


@pytest.mark.parametrize("doc, error", [
    (b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;">]>'
     b'<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08"><x>&lol2;</x></Document>', "unsafe"),
    (b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "file:///etc/passwd">]><Document>&x;</Document>', "unsafe"),
    (b"<Document><unclosed></Document>", "malformed"),
    (b'<Document xmlns="urn:iso:std:iso:20022:tech:xsd:camt.053.001.08"><BkToCstmrStmt/></Document>', "unsupported"),
])
def test_iso20022_rejects_unsafe_malformed_and_unsupported_xml(doc, error):
    with pytest.raises(iso20022.Iso20022Error, match=error):
        iso20022.parse(doc)


def test_iso20022_rejects_bad_amounts_and_ibans():
    raw = (FIX / "pacs008_sample.xml").read_text()
    with pytest.raises(iso20022.Iso20022Error, match="positive"):
        iso20022.parse(raw.replace(">2450.00<", ">-5<"))
    with pytest.raises(iso20022.Iso20022Error, match="IBAN"):
        iso20022.parse(raw.replace("DE89370400440532013000", "NOT-AN-IBAN"))
    with pytest.raises(iso20022.Iso20022Error, match="Ccy"):
        iso20022.parse(raw.replace('Ccy="EUR">2450', 'Ccy="E">2450'))
