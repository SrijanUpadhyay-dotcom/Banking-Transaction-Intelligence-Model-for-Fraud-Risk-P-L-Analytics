"""
Feature governance and point-in-time feature engineering for the BTI v3 model.

Every source field is classified by *when it becomes known*. Only fields known
before the authorisation decision may feed the model; outcome fields (losses,
chargebacks), label-derived fields and protected / proxy attributes are
blocked, and every model feature declares its source lineage so the block is
enforced in code rather than by convention.

Behavioural features look strictly backwards (events before the current
timestamp, within a configurable look-back). Training and live scoring call the
same `build_features` function, so there is no train/serve skew.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bti.modeling.fx import series_to_usd


class Availability(str, Enum):
    PRE_AUTH      = "pre_authorization"   # known before the decision — allowed
    POST_EVENT    = "post_event"          # known only after settlement / dispute — blocked
    LABEL_DERIVED = "label_derived"       # generated from, or together with, the label — blocked
    PROTECTED     = "protected_or_proxy"  # protected attribute or proxy — blocked, monitored for fairness
    IDENTIFIER    = "identifier"          # keys — only used to link history


@dataclass(frozen=True)
class SourceField:
    name: str
    availability: Availability
    note: str


_A = Availability
SOURCE_FIELDS: Dict[str, SourceField] = {f.name: f for f in [
    SourceField("transaction_id", _A.IDENTIFIER, "Primary key"),
    SourceField("customer_id", _A.IDENTIFIER, "Links customer history"),
    SourceField("account_id", _A.IDENTIFIER, "Account key"),
    SourceField("device_id", _A.IDENTIFIER, "Links device history; never a direct input"),
    SourceField("ip_location", _A.IDENTIFIER, "Links IP history; never a direct input"),
    SourceField("merchant_name", _A.IDENTIFIER, "Links merchant history"),
    SourceField("payee_id", _A.IDENTIFIER, "Beneficiary account key (payments feed); links payee history"),
    SourceField("latitude", _A.PRE_AUTH, "Where the transaction happened: terminal location, device GPS, or IP "
                                         "geolocation resolved upstream (location feed)"),
    SourceField("longitude", _A.PRE_AUTH, "See latitude"),
    SourceField("security_event_time", _A.PRE_AUTH, "When a password reset, SIM swap, contact-detail change or "
                                                    "device enrolment was logged (security-event feed)"),
    SourceField("security_event_type", _A.PRE_AUTH, "Type of security event (security-event feed)"),
    SourceField("transaction_date", _A.PRE_AUTH, "Event date"),
    SourceField("transaction_time", _A.PRE_AUTH, "Event time"),
    SourceField("transaction_amount", _A.PRE_AUTH, "Requested amount, in account currency"),
    SourceField("currency", _A.PRE_AUTH, "Used for FX normalisation"),
    SourceField("transaction_type", _A.PRE_AUTH, "Requested transaction type"),
    SourceField("debit_credit_flag", _A.PRE_AUTH, "Direction of funds"),
    SourceField("channel", _A.PRE_AUTH, "Origination channel"),
    SourceField("branch_or_digital_flag", _A.PRE_AUTH, "Redundant with channel"),
    SourceField("merchant_category", _A.PRE_AUTH, "Merchant category"),
    SourceField("authorization_method", _A.PRE_AUTH, "Credential presented"),
    SourceField("account_balance_before", _A.PRE_AUTH, "Balance at request time"),
    SourceField("account_balance_after", _A.PRE_AUTH, "Arithmetic of balance and amount; redundant"),
    SourceField("failed_attempt_count", _A.PRE_AUTH, "Failed credential attempts in session"),
    SourceField("login_attempts", _A.PRE_AUTH, "Login attempts in session"),
    SourceField("historical_average_transaction_amount", _A.PRE_AUTH,
                "Customer profile average; must be maintained point-in-time in production"),
    SourceField("monthly_customer_transaction_count", _A.PRE_AUTH,
                "Profile field; replaced by point-in-time velocity features"),
    SourceField("customer_segment", _A.PROTECTED,
                "Contains nationality-adjacent segments (NRI / Diaspora); monitored, not modelled"),
    SourceField("customer_age_band", _A.PROTECTED, "Age is a protected characteristic"),
    SourceField("country", _A.PROTECTED, "Residence country — national-origin proxy; used for jurisdiction policy"),
    SourceField("geography", _A.PROTECTED, "Same as country"),
    SourceField("city", _A.PROTECTED, "Location proxy"),
    SourceField("transaction_status", _A.LABEL_DERIVED, "Takes the value 'Fraudulent' after investigation"),
    SourceField("reversal_flag", _A.POST_EVENT, "Set after the transaction is reversed"),
    SourceField("refund_flag", _A.POST_EVENT, "Set after a refund"),
    SourceField("chargeback_flag", _A.POST_EVENT, "Set weeks later when the cardholder disputes"),
    SourceField("fee_income", _A.POST_EVENT, "Settlement economics"),
    SourceField("interchange_income", _A.POST_EVENT, "Settlement economics"),
    SourceField("processing_cost", _A.POST_EVENT, "Settlement economics"),
    SourceField("chargeback_loss", _A.POST_EVENT, "Loss booked after dispute"),
    SourceField("refund_loss", _A.POST_EVENT, "Loss booked after refund"),
    SourceField("fraud_loss", _A.LABEL_DERIVED, "Non-zero only on confirmed fraud"),
    SourceField("net_revenue", _A.POST_EVENT, "Settlement economics"),
    SourceField("net_pnl_impact", _A.LABEL_DERIVED, "Includes fraud_loss"),
    SourceField("fraud_flag", _A.LABEL_DERIVED, "The label"),
    SourceField("fraud_type", _A.LABEL_DERIVED, "Label detail"),
    SourceField("risk_score", _A.LABEL_DERIVED,
                "Synthetic generator adds +60 when it creates a fraud (src/01_data_generation.py)"),
    SourceField("fraud_rule_score", _A.LABEL_DERIVED, "Legacy rules R06/R12/R17 read risk_score"),
    SourceField("rules_triggered", _A.LABEL_DERIVED, "Legacy rules R06/R12/R17 read risk_score"),
    SourceField("is_high_risk", _A.LABEL_DERIVED, "Threshold on risk_score"),
    SourceField("final_risk_score", _A.LABEL_DERIVED, "Legacy composite weights risk_score at 35%"),
]}


class Kind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: Kind
    sources: Tuple[str, ...]
    reason_code: str
    description: str


_N, _C = Kind.NUMERIC, Kind.CATEGORICAL
_AMT = ("transaction_amount", "currency")
_WHEN = ("transaction_date", "transaction_time")
_CUST = ("customer_id",) + _WHEN

CORE_FEATURES: List[FeatureSpec] = [
    FeatureSpec("amount_usd", _N, _AMT, "AMT_HIGH", "Amount in USD equivalent"),
    FeatureSpec("log_amount_usd", _N, _AMT, "AMT_HIGH", "log1p of USD amount"),
    FeatureSpec("amount_vs_hist_avg", _N, ("transaction_amount", "historical_average_transaction_amount"),
                "AMT_VS_HISTORY", "Amount ÷ customer's historical average (currency-invariant)"),
    FeatureSpec("amount_to_balance", _N, ("transaction_amount", "account_balance_before"),
                "AMT_VS_BALANCE", "Amount ÷ available balance"),
    FeatureSpec("balance_before_usd", _N, ("account_balance_before", "currency"),
                "AMT_VS_BALANCE", "Balance before the request, USD"),
    FeatureSpec("failed_attempt_count", _N, ("failed_attempt_count",), "AUTH_FAILURES",
                "Failed credential attempts"),
    FeatureSpec("login_attempts", _N, ("login_attempts",), "LOGIN_ANOMALY", "Login attempts in session"),
    FeatureSpec("txn_hour", _N, ("transaction_time",), "TIME_OF_DAY", "Hour of day"),
    FeatureSpec("is_off_hours", _N, ("transaction_time",), "TIME_OF_DAY", "00:00–05:59 or 22:00–23:59"),
    FeatureSpec("is_weekend", _N, ("transaction_date",), "TIME_OF_DAY", "Saturday or Sunday"),
    FeatureSpec("is_debit", _N, ("debit_credit_flag",), "TXN_TYPE", "Funds leaving the account"),
    FeatureSpec("cust_txn_count_1h", _N, _CUST, "VELOCITY", "Customer transactions in prior 1h"),
    FeatureSpec("cust_txn_count_24h", _N, _CUST, "VELOCITY", "Customer transactions in prior 24h"),
    FeatureSpec("cust_txn_count_7d", _N, _CUST, "VELOCITY", "Customer transactions in prior 7d"),
    FeatureSpec("cust_amount_usd_24h", _N, _CUST + _AMT, "VELOCITY", "Customer USD spend in prior 24h"),
    FeatureSpec("cust_amount_usd_7d", _N, _CUST + _AMT, "VELOCITY", "Customer USD spend in prior 7d"),
    FeatureSpec("secs_since_last_txn", _N, _CUST, "VELOCITY", "Seconds since customer's previous transaction"),
    FeatureSpec("device_new_for_customer", _N, _CUST + ("device_id",), "NEW_DEVICE",
                "Device not used by this customer within the look-back (unknown without prior history)"),
    FeatureSpec("ip_new_for_customer", _N, _CUST + ("ip_location",), "NEW_IP",
                "IP not used by this customer within the look-back (unknown without prior history)"),
    FeatureSpec("merchant_new_for_customer", _N, _CUST + ("merchant_name",), "NEW_MERCHANT",
                "First payment to this merchant within the look-back (unknown without prior history)"),
    FeatureSpec("device_other_customer_txns", _N, _CUST + ("device_id",), "SHARED_DEVICE",
                "Prior transactions on this device by other customers"),
    FeatureSpec("ip_other_customer_txns", _N, _CUST + ("ip_location",), "SHARED_DEVICE",
                "Prior transactions from this IP by other customers"),
    FeatureSpec("channel", _C, ("channel",), "CHANNEL_RISK", "Origination channel"),
    FeatureSpec("authorization_method", _C, ("authorization_method",), "CHANNEL_RISK", "Credential presented"),
    FeatureSpec("merchant_category", _C, ("merchant_category",), "MERCHANT_RISK", "Merchant category"),
    FeatureSpec("transaction_type", _C, ("transaction_type",), "TXN_TYPE", "Transaction type"),
]

# Velocity round two (Phase 1). Kept as a separate set so their value is measured, not assumed.
EXTENDED_FEATURES: List[FeatureSpec] = [
    FeatureSpec("merchant_txn_count_1h", _N, ("merchant_name",) + _WHEN, "MERCHANT_VELOCITY",
                "Transactions at this merchant (all customers) in prior 1h"),
    FeatureSpec("merchant_txn_count_24h", _N, ("merchant_name",) + _WHEN, "MERCHANT_VELOCITY",
                "Transactions at this merchant (all customers) in prior 24h"),
    FeatureSpec("device_txn_count_24h", _N, ("device_id",) + _WHEN, "DEVICE_VELOCITY",
                "Transactions on this device (all customers) in prior 24h"),
    FeatureSpec("amount_zscore_customer", _N, _CUST + _AMT, "AMT_VS_HISTORY",
                "Standard deviations from the customer's own amounts in the look-back"),
    FeatureSpec("just_below_threshold", _N, _AMT, "STRUCTURING",
                "USD amount within 10% below 1,000 / 3,000 / 5,000 / 10,000"),
    FeatureSpec("cust_near_threshold_7d", _N, _CUST + _AMT, "STRUCTURING",
                "Customer's just-below-threshold transactions in prior 7d"),
    FeatureSpec("hour_deviation", _N, _CUST, "TIME_OF_DAY",
                "Hours between this transaction and the customer's usual time of day"),
]

# Feed-dependent signals. They need data the synthetic dataset does not carry (transaction location,
# payee keys, the bank's security-event log), so they are in no default feature set and training
# refuses them unless the feed is actually populated (see FEEDS and feed_coverage).
_GEO = ("customer_id", "latitude", "longitude") + _WHEN
_PAYEE = ("customer_id", "payee_id") + _WHEN
_SEC = ("customer_id", "security_event_time", "security_event_type") + _WHEN
SIGNAL_FEATURES: List[FeatureSpec] = [
    FeatureSpec("geo_distance_prev_km", _N, _GEO, "GEO_VELOCITY",
                "Kilometres from the customer's previous located transaction"),
    FeatureSpec("geo_speed_kmh", _N, _GEO, "GEO_VELOCITY",
                "Implied travel speed from the customer's previous located transaction"),
    FeatureSpec("impossible_travel", _N, _GEO, "GEO_VELOCITY",
                "Implied speed above 900 km/h over more than 300 km"),
    FeatureSpec("payee_new_for_customer", _N, _PAYEE, "NEW_PAYEE",
                "Customer has not paid this payee in the look-back"),
    FeatureSpec("cust_new_payees_24h", _N, _PAYEE, "NEW_PAYEE",
                "Payments to payees new for the customer in prior 24h"),
    FeatureSpec("payee_other_customer_txns_7d", _N, _PAYEE, "PAYEE_VELOCITY",
                "Payments to this payee from other customers in prior 7d (mule pattern)"),
    FeatureSpec("hours_since_security_event", _N, _SEC, "SECURITY_EVENT",
                "Hours since the customer's last password reset, SIM swap, contact change or device enrolment "
                "(within 30 days)"),
    FeatureSpec("hours_since_sim_swap", _N, _SEC, "SECURITY_EVENT",
                "Hours since the customer's last SIM swap or number port (within 30 days)"),
    FeatureSpec("security_events_7d", _N, _SEC, "SECURITY_EVENT", "Security events for the customer in prior 7d"),
]
FEEDS: Dict[str, Dict] = {
    "location": {"features": ["geo_distance_prev_km", "geo_speed_kmh", "impossible_travel"],
                 "fields": ["latitude", "longitude"],
                 "source": "Card-present terminal coordinates, mobile device GPS, or an IP-geolocation service "
                           "resolved before scoring"},
    "payee": {"features": ["payee_new_for_customer", "cust_new_payees_24h", "payee_other_customer_txns_7d"],
              "fields": ["payee_id"],
              "source": "Beneficiary account identifier from the payments system (sort code + account, IBAN, UPI "
                        "VPA, or a tokenised equivalent)"},
    "security_events": {"features": ["hours_since_security_event", "hours_since_sim_swap", "security_events_7d"],
                        "fields": ["security_event_time", "security_event_type"],
                        "source": "The bank's identity / security log: password resets, SIM swaps and number "
                                  "ports (mobile-operator API), contact-detail changes, device enrolments"},
}
SECURITY_EVENT_TYPES = ("password_reset", "sim_swap", "number_port", "phone_change", "email_change", "address_change",
                        "device_enrolment", "limit_increase")
IMPOSSIBLE_SPEED_KMH = 900.0
IMPOSSIBLE_MIN_KM = 300.0
SECURITY_LOOKBACK_S = 30 * 86_400
NO_RECENT_EVENT_HOURS = 24.0 * 31

MODEL_FEATURES: List[FeatureSpec] = CORE_FEATURES
ALL_FEATURES: List[FeatureSpec] = CORE_FEATURES + EXTENDED_FEATURES + SIGNAL_FEATURES
FEATURE_NAMES: List[str] = [f.name for f in CORE_FEATURES]
EXTENDED_NAMES: List[str] = [f.name for f in CORE_FEATURES + EXTENDED_FEATURES]
SIGNAL_NAMES: List[str] = [f.name for f in SIGNAL_FEATURES]
ALL_FEATURE_NAMES: List[str] = [f.name for f in ALL_FEATURES]
# Absolute size (USD amount, USD balance) tracks customer wealth, which varies by country and segment.
# The "relative" sets judge amounts only against the customer's own behaviour; loss severity is
# handled by the expected-cost decision layer (probability × amount), not by the probability model.
ABSOLUTE_AMOUNT_FEATURES: List[str] = ["amount_usd", "log_amount_usd", "balance_before_usd"]
FEATURE_SETS: Dict[str, List[str]] = {
    "core": FEATURE_NAMES,
    "extended": EXTENDED_NAMES,
    "core-relative": [n for n in FEATURE_NAMES if n not in ABSOLUTE_AMOUNT_FEATURES],
    "extended-relative": [n for n in EXTENDED_NAMES if n not in ABSOLUTE_AMOUNT_FEATURES],
    "signals-relative": [n for n in EXTENDED_NAMES if n not in ABSOLUTE_AMOUNT_FEATURES] + SIGNAL_NAMES,
}
FEATURE_SET_SUFFIX: Dict[str, str] = {"core": "", "extended": "-x", "core-relative": "-r", "extended-relative": "-xr",
                                      "signals-relative": "-sr"}
CATEGORICAL_FEATURES: List[str] = [f.name for f in ALL_FEATURES if f.kind is Kind.CATEGORICAL]
NUMERIC_FEATURES: List[str] = [f.name for f in ALL_FEATURES if f.kind is Kind.NUMERIC]
FEATURE_BY_NAME: Dict[str, FeatureSpec] = {f.name: f for f in ALL_FEATURES}
STRUCTURING_THRESHOLDS_USD = (1_000, 3_000, 5_000, 10_000)
PROTECTED_ATTRIBUTES: List[str] = [n for n, f in SOURCE_FIELDS.items() if f.availability is _A.PROTECTED]

DEFAULT_LOOKBACK_DAYS = 365

# Feature-definition versions. Every model records the version it was trained on
# and is always scored with that version, so a definition change never silently
# alters what a registered model sees.
#   1 — novelty flags are 1 when the customer has no prior transactions
#   2 — novelty flags are unknown (NaN) when the customer has no prior
#       transactions; v1 conflated "never seen this customer" with "new device",
#       penalising thin-history and new-to-bank customers
FEATURE_VERSION = 2
_ALLOWED = {_A.PRE_AUTH, _A.IDENTIFIER}


class FeatureGovernanceError(RuntimeError):
    pass


def feature_specs(feature_set: str) -> List[FeatureSpec]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
    names = set(FEATURE_SETS[feature_set])
    return [f for f in ALL_FEATURES if f.name in names]


def assert_feature_lineage(features: Iterable[FeatureSpec] = ALL_FEATURES) -> None:
    """Raise if any model feature is built from a field not known pre-authorisation."""
    problems = []
    for f in features:
        for src in f.sources:
            field = SOURCE_FIELDS.get(src)
            if field is None:
                problems.append(f"{f.name}: source '{src}' is not in the governance catalogue")
            elif field.availability not in _ALLOWED:
                problems.append(f"{f.name}: source '{src}' is {field.availability.value}")
    if problems:
        raise FeatureGovernanceError("; ".join(problems))


assert_feature_lineage()


# ── Point-in-time window arithmetic ────────────────────────────────────────────

_SPAN = np.int64(10**10)  # > any epoch-seconds value in range, keeps groups disjoint


def _group_ids(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    keys = df[cols].astype("string").fillna("\x00missing")
    return keys.groupby(cols, sort=False).ngroup().to_numpy(np.int64)


def _prior_window(
    gid: np.ndarray, ts: np.ndarray, window_s: int, values: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Count (and optionally sum) events in the same group with ts in [t - window, t)."""
    order = np.lexsort((ts, gid))
    key = gid[order] * _SPAN + ts[order]
    start = np.searchsorted(key, key - window_s, side="left")
    end = np.searchsorted(key, key, side="left")
    count = np.empty(len(ts), dtype=np.int64)
    count[order] = end - start
    total = None
    if values is not None:
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values[order]))])
        total = np.empty(len(ts), dtype=float)
        total[order] = cs[end] - cs[start]
    return count, total


def _seconds_since_prior(gid: np.ndarray, ts: np.ndarray, lookback_s: int) -> np.ndarray:
    order = np.lexsort((ts, gid))
    g, t = gid[order], ts[order]
    key = g * _SPAN + t
    end = np.searchsorted(key, key, side="left")
    group_start = np.searchsorted(key, g * _SPAN, side="left")
    prev = end - 1
    has_prev = prev >= group_start
    gap = np.full(len(t), np.nan)
    gap[has_prev] = t[has_prev] - t[prev[has_prev]]
    gap[gap > lookback_s] = np.nan
    out = np.empty(len(t))
    out[order] = gap
    return out


def event_timestamps(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df["transaction_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    time = df.get("transaction_time", pd.Series("12:00:00", index=df.index)).fillna("12:00:00").astype(str)
    return pd.to_datetime(date + " " + time.str.slice(0, 8), errors="coerce")


def build_features(df: pd.DataFrame, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                   feature_version: int = FEATURE_VERSION,
                   security_events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute model features for every row of `df` using only information that
    precedes each row's timestamp. Returns a frame indexed like `df`.

    `security_events` (customer_id, event_time, event_type) is the bank's
    security-event log; without it the security-event features are unknown.
    """
    if feature_version not in (1, 2):
        raise ValueError(f"Unknown feature_version {feature_version}")
    df = df.copy()
    for col in ("device_id", "ip_location", "merchant_name", "customer_id", "currency", "payee_id",
                "latitude", "longitude",
                "historical_average_transaction_amount", "account_balance_before",
                "failed_attempt_count", "login_attempts", "debit_credit_flag", *CATEGORICAL_FEATURES):
        if col not in df.columns:
            df[col] = np.nan

    ts_dt = event_timestamps(df)
    if ts_dt.isna().any():
        raise ValueError(f"{int(ts_dt.isna().sum())} rows have an unparseable transaction_date/transaction_time")
    ts = (ts_dt - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy().astype(np.int64)
    lookback_s = int(lookback_days) * 86_400

    out = pd.DataFrame(index=df.index)
    amount = pd.to_numeric(df["transaction_amount"], errors="coerce")
    amount_usd = series_to_usd(amount, df["currency"])
    hist_avg = pd.to_numeric(df["historical_average_transaction_amount"], errors="coerce")
    balance = pd.to_numeric(df["account_balance_before"], errors="coerce")

    out["amount_usd"] = amount_usd
    out["log_amount_usd"] = np.log1p(amount_usd.clip(lower=0))
    out["amount_vs_hist_avg"] = (amount / hist_avg.where(hist_avg > 0)).astype(float)
    out["amount_to_balance"] = (amount / balance.clip(lower=1.0)).astype(float)
    out["balance_before_usd"] = series_to_usd(balance, df["currency"])
    out["failed_attempt_count"] = pd.to_numeric(df["failed_attempt_count"], errors="coerce")
    out["login_attempts"] = pd.to_numeric(df["login_attempts"], errors="coerce")
    hour = ts_dt.dt.hour
    out["txn_hour"] = hour.astype(float)
    out["is_off_hours"] = ((hour < 6) | (hour >= 22)).astype(float)
    out["is_weekend"] = (ts_dt.dt.dayofweek >= 5).astype(float)
    out["is_debit"] = (df["debit_credit_flag"].astype(str).str.lower() == "debit").astype(float)

    cust = _group_ids(df, ["customer_id"])
    usd = amount_usd.to_numpy(float)
    out["cust_txn_count_1h"], _ = _prior_window(cust, ts, 3_600)
    out["cust_txn_count_24h"], out["cust_amount_usd_24h"] = _prior_window(cust, ts, 86_400, usd)
    out["cust_txn_count_7d"], out["cust_amount_usd_7d"] = _prior_window(cust, ts, 7 * 86_400, usd)
    out["secs_since_last_txn"] = _seconds_since_prior(cust, ts, lookback_s)
    cust_prior, _ = _prior_window(cust, ts, lookback_s)
    knowable = (cust_prior > 0) if feature_version >= 2 else np.ones(len(ts), dtype=bool)

    for entity, new_col, shared_col in (
        ("device_id", "device_new_for_customer", "device_other_customer_txns"),
        ("ip_location", "ip_new_for_customer", "ip_other_customer_txns"),
        ("merchant_name", "merchant_new_for_customer", None),
    ):
        present = df[entity].notna() & (df[entity].astype(str) != "")
        cust_entity, _ = _prior_window(_group_ids(df, ["customer_id", entity]), ts, lookback_s)
        out[new_col] = np.where(present & knowable, (cust_entity == 0).astype(float), np.nan)
        if shared_col:
            entity_all, _ = _prior_window(_group_ids(df, [entity]), ts, lookback_s)
            out[shared_col] = np.where(present, (entity_all - cust_entity).astype(float), np.nan)

    # ── Velocity round two ────────────────────────────────────────────────────
    merchant_present = df["merchant_name"].notna() & (df["merchant_name"].astype(str) != "")
    merch = _group_ids(df, ["merchant_name"])
    m1h, _ = _prior_window(merch, ts, 3_600)
    m24h, _ = _prior_window(merch, ts, 86_400)
    out["merchant_txn_count_1h"] = np.where(merchant_present, m1h, np.nan)
    out["merchant_txn_count_24h"] = np.where(merchant_present, m24h, np.nan)

    device_present = df["device_id"].notna() & (df["device_id"].astype(str) != "")
    d24h, _ = _prior_window(_group_ids(df, ["device_id"]), ts, 86_400)
    out["device_txn_count_24h"] = np.where(device_present, d24h, np.nan)

    usd_clean = np.nan_to_num(usd)
    n_lb, s1 = _prior_window(cust, ts, lookback_s, usd_clean)
    _, s2 = _prior_window(cust, ts, lookback_s, usd_clean ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n_lb
        std = np.sqrt(np.clip(s2 / n_lb - mean ** 2, 0, None))
        z = (usd - mean) / std
    out["amount_zscore_customer"] = np.where((n_lb >= 2) & (std > 0), np.clip(z, -50, 50), np.nan)

    near = np.zeros(len(usd), dtype=bool)
    for t in STRUCTURING_THRESHOLDS_USD:
        near |= (usd >= 0.9 * t) & (usd < t)
    out["just_below_threshold"] = near.astype(float)
    _, near_7d = _prior_window(cust, ts, 7 * 86_400, near.astype(float))
    out["cust_near_threshold_7d"] = near_7d

    angle = 2 * np.pi * hour.to_numpy(float) / 24
    n_h, s_sin = _prior_window(cust, ts, lookback_s, np.sin(angle))
    _, s_cos = _prior_window(cust, ts, lookback_s, np.cos(angle))
    usual = np.arctan2(s_sin, s_cos)
    gap = np.abs(np.angle(np.exp(1j * (angle - usual)))) * 24 / (2 * np.pi)
    out["hour_deviation"] = np.where(n_h >= 3, gap, np.nan)

    # ── Feed-dependent signals (unknown when the feed is absent) ─────────────
    lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(float)
    dist, gap_s = _previous_located(cust, ts, lat, lon, lookback_s)
    out["geo_distance_prev_km"] = dist
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = dist / (np.maximum(gap_s, 60.0) / 3_600)
    out["geo_speed_kmh"] = speed
    out["impossible_travel"] = np.where(np.isnan(dist), np.nan,
                                        ((speed > IMPOSSIBLE_SPEED_KMH) & (dist > IMPOSSIBLE_MIN_KM)).astype(float))

    payee_present = (df["payee_id"].notna() & (df["payee_id"].astype(str) != "")).to_numpy()
    cust_payee, _ = _prior_window(_group_ids(df, ["customer_id", "payee_id"]), ts, lookback_s)
    new_payee = np.where(payee_present & knowable, (cust_payee == 0).astype(float), np.nan)
    out["payee_new_for_customer"] = new_payee
    _, new_24h = _prior_window(cust, ts, 86_400, np.nan_to_num(new_payee))
    out["cust_new_payees_24h"] = np.where(payee_present, new_24h, np.nan)
    payee_7d, _ = _prior_window(_group_ids(df, ["payee_id"]), ts, 7 * 86_400)
    cust_payee_7d, _ = _prior_window(_group_ids(df, ["customer_id", "payee_id"]), ts, 7 * 86_400)
    out["payee_other_customer_txns_7d"] = np.where(payee_present, (payee_7d - cust_payee_7d).astype(float), np.nan)

    since_any, since_sim, count_7d = _security_event_features(df["customer_id"], ts, security_events)
    out["hours_since_security_event"] = since_any
    out["hours_since_sim_swap"] = since_sim
    out["security_events_7d"] = count_7d

    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype("string")

    return out[ALL_FEATURE_NAMES]


def feed_coverage(features: pd.DataFrame) -> Dict[str, float]:
    """Share of rows on which each feed's features are known (0.0 means the feed is absent)."""
    return {feed: round(float(features[spec["features"]].notna().any(axis=1).mean()), 4) if len(features) else 0.0
            for feed, spec in FEEDS.items()}


def feeds_required(feature_names: Iterable[str]) -> List[str]:
    names = set(feature_names)
    return [feed for feed, spec in FEEDS.items() if names & set(spec["features"])]


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _previous_located(gid: np.ndarray, ts: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                      lookback_s: int) -> Tuple[np.ndarray, np.ndarray]:
    """Distance (km) and gap (s) to the same customer's latest earlier transaction that has a location."""
    dist, gap = np.full(len(ts), np.nan), np.full(len(ts), np.nan)
    located = np.flatnonzero(~np.isnan(lat) & ~np.isnan(lon))
    if len(located) < 2:
        return dist, gap
    order = located[np.lexsort((ts[located], gid[located]))]
    key = gid[order] * _SPAN + ts[order]
    prev = np.searchsorted(key, key, side="left") - 1            # strictly earlier timestamp
    ok = (prev >= 0) & (gid[order[np.maximum(prev, 0)]] == gid[order])
    cur, before = order[ok], order[prev[ok]]
    g = (ts[cur] - ts[before]).astype(float)
    recent = g <= lookback_s
    cur, before, g = cur[recent], before[recent], g[recent]
    dist[cur] = _haversine_km(lat[before], lon[before], lat[cur], lon[cur])
    gap[cur] = g
    return dist, gap


def _security_event_features(customers: pd.Series, ts: np.ndarray, events: Optional[pd.DataFrame]):
    """
    Hours since the last (any / SIM-swap) security event strictly before each transaction, and the count in
    the prior 7 days. Unknown (NaN) without a feed; with a feed, "nothing in the last 30 days" is encoded as
    NO_RECENT_EVENT_HOURS so it is distinguishable from a missing feed.
    """
    n = len(ts)
    unknown = (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    if events is None or len(events) == 0:
        return unknown
    ev_dt = pd.to_datetime(events["event_time"], errors="coerce")
    keep = ev_dt.notna().to_numpy()
    if not keep.any():
        return unknown
    ev_ts = (ev_dt[keep] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy().astype(np.int64)
    ev_cust = events["customer_id"].astype("string").to_numpy()[keep]
    ev_type = events["event_type"].astype("string").str.lower().to_numpy()[keep]
    txn_cust = customers.astype("string").to_numpy()
    codes = {c: i for i, c in enumerate(pd.unique(np.concatenate([txn_cust, ev_cust])))}
    t_gid = np.array([codes[c] for c in txn_cust], dtype=np.int64)
    e_gid = np.array([codes[c] for c in ev_cust], dtype=np.int64)
    q = t_gid * _SPAN + ts

    def hours_since_last(mask):
        hours = np.full(n, NO_RECENT_EVENT_HOURS)
        if not mask.any():
            return hours
        key = np.sort(e_gid[mask] * _SPAN + ev_ts[mask])
        i = np.searchsorted(key, q, side="left") - 1               # strictly before the transaction
        ok = (i >= 0) & (key[np.maximum(i, 0)] // _SPAN == t_gid)
        age = ts[ok] - key[i[ok]] % _SPAN
        hours[np.flatnonzero(ok)] = np.where(age <= SECURITY_LOOKBACK_S, age / 3_600, NO_RECENT_EVENT_HOURS)
        return hours

    key = np.sort(e_gid * _SPAN + ev_ts)
    count_7d = (np.searchsorted(key, q, side="left") - np.searchsorted(key, q - 7 * 86_400, side="left")).astype(float)
    return (hours_since_last(np.ones(len(ev_ts), dtype=bool)),
            hours_since_last(np.isin(ev_type, ["sim_swap", "number_port"])), count_7d)


# Categorical fields enter the model as their smoothed historical fraud rate.
# Native tree categorical splits are avoided on purpose: SHAP's TreeExplainer
# does not reproduce HistGradientBoosting categorical bitset splits (additivity
# fails silently), which would make every reason code wrong.

def fit_category_encodings(features: pd.DataFrame, y: np.ndarray, smoothing: float = 50.0) -> Dict[str, Dict]:
    y = np.asarray(y, dtype=float)
    prior = float(y.mean())
    encodings: Dict[str, Dict] = {}
    for col in CATEGORICAL_FEATURES:
        s = features[col].astype("string").fillna("(missing)")
        stats = pd.DataFrame({"k": s.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        rates = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
        encodings[col] = {"prior": prior, "smoothing": smoothing,
                          "rates": {str(k): round(float(v), 6) for k, v in rates.items()},
                          "counts": {str(k): int(v) for k, v in stats["count"].items()}}
    return encodings


def out_of_fold_category_encoding(features: pd.DataFrame, y: np.ndarray, smoothing: float = 50.0,
                                  n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Encode training rows with mappings learned on the other folds, so no row sees its own label."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(features))
    out = pd.DataFrame(index=features.index, columns=CATEGORICAL_FEATURES, dtype=float)
    y = np.asarray(y)
    for k in range(n_folds):
        enc = fit_category_encodings(features[fold != k], y[fold != k], smoothing)
        out.loc[fold == k, CATEGORICAL_FEATURES] = _apply_encodings(features[fold == k], enc).to_numpy()
    return out.astype(float)


def _apply_encodings(features: pd.DataFrame, encodings: Dict[str, Dict]) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)
    for col in CATEGORICAL_FEATURES:
        enc = encodings[col]
        s = features[col].astype("string").fillna("(missing)")
        out[col] = s.map(enc["rates"]).astype(float).fillna(enc["prior"])
    return out


def to_model_matrix(features: pd.DataFrame, encodings: Dict[str, Dict],
                    feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """All-numeric model input: numeric features as-is, categoricals as learned fraud rates."""
    names = list(feature_names or FEATURE_NAMES)
    X = features[names].copy()
    for col in names:
        if col not in CATEGORICAL_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    cats = [c for c in CATEGORICAL_FEATURES if c in names]
    X[cats] = _apply_encodings(features, encodings)[cats]
    return X.astype(float)


def leakage_audit(df: pd.DataFrame, label_col: str = "fraud_flag", threshold: float = 0.97) -> List[dict]:
    """
    Single-feature separability screen. Any numeric column that alone separates
    the label above `threshold` AUC is almost certainly leakage.
    """
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return []
    findings = []
    for col in df.columns:
        if col == label_col or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        x = df[col].astype(float).fillna(df[col].median() if df[col].notna().any() else 0)
        if x.nunique() < 2:
            continue
        auc = roc_auc_score(y, x)
        auc = max(auc, 1 - auc)
        field = SOURCE_FIELDS.get(col)
        findings.append({
            "column": col,
            "single_feature_auc": round(float(auc), 4),
            "catalogued_as": field.availability.value if field else "uncatalogued",
            "suspected_leak": bool(auc >= threshold),
        })
    return sorted(findings, key=lambda r: -r["single_feature_auc"])
