"""
Currency normalisation to a single reporting currency (USD).

Amount-based features must be comparable across the book: INR 10,000 and
USD 10,000 are not the same exposure. The reference table below holds
indicative 2024 year-end rates so the model can train reproducibly; a
production deployment overrides them with the bank's official daily rate
feed via `fx.rates_to_usd` in config/settings.yaml.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bti.config import CONFIG_FILE
from bti.logging_config import get_logger

log = get_logger("modeling.fx")

REFERENCE_RATES_TO_USD: Dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.04,
    "GBP": 1.25,
    "INR": 0.0117,
    "SGD": 0.733,
    "HKD": 0.1287,
    "AED": 0.2723,
    "NGN": 0.00065,
}
RATES_AS_OF = "2024-12-31 (indicative)"


def _configured_rates() -> Dict[str, float]:
    try:
        import yaml
        if CONFIG_FILE.exists():
            cfg = yaml.safe_load(CONFIG_FILE.read_text()) or {}
            override = (cfg.get("fx") or {}).get("rates_to_usd") or {}
            return {**REFERENCE_RATES_TO_USD, **{k.upper(): float(v) for k, v in override.items()}}
    except Exception as exc:
        log.warning(f"Could not read FX overrides, using reference rates: {exc}")
    return dict(REFERENCE_RATES_TO_USD)


RATES_TO_USD: Dict[str, float] = _configured_rates()


def rate_to_usd(currency: Optional[str]) -> float:
    """Rate for one unit of `currency` in USD; NaN when the currency is unknown."""
    if not currency:
        return float("nan")
    return RATES_TO_USD.get(str(currency).upper(), float("nan"))


def to_usd(amount: Optional[float], currency: Optional[str]) -> float:
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return float("nan")
    return float(amount) * rate_to_usd(currency)


def series_to_usd(amounts: pd.Series, currencies: pd.Series) -> pd.Series:
    rates = currencies.astype(str).str.upper().map(RATES_TO_USD).astype(float)
    unknown = rates.isna() & currencies.notna()
    if unknown.any():
        log.warning("Unknown currencies — amounts left as NaN",
                    extra={"currencies": sorted(currencies[unknown].astype(str).unique().tolist())[:10]})
    return pd.to_numeric(amounts, errors="coerce") * rates


def supported_currencies() -> list[str]:
    return sorted(k for k, v in RATES_TO_USD.items() if v and not np.isnan(v))
