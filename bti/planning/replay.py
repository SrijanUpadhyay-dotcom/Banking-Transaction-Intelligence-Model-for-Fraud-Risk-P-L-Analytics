# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
A labelled, scored window for planning: the model's calibrated probabilities on
its out-of-time window, with amounts, channels, labels and monitoring
attributes.

The staffing forecast uses it for intervention rates, and the what-if
simulator replays decision policies on it. Probabilities come from the
registered model exactly as scored live (same features, encodings and
calibrator). The recalibration overlay is not applied.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd


@lru_cache(maxsize=4)
def scored_window(model_id: Optional[str] = None, data_path: Optional[str] = None, window: str = "out_of_time"):
    from bti.modeling import registry
    from bti.modeling.reassess import model_probabilities
    from bti.modeling.train import prepare
    model_id = model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
    if not model_id:
        raise registry.RegistryError("No registered model to replay")
    version = registry.load_artifact(model_id).get("feature_version", 1)
    data = prepare(data_path, feature_version=version)
    rows = {"out_of_time": data.te, "calibration": data.ca, "all": np.arange(len(data.df))}[window]
    p = model_probabilities(model_id, data)[rows]
    df = data.df.iloc[rows]
    frame = pd.DataFrame({
        "transaction_id": df["transaction_id"].astype(str).to_numpy(),
        "date": pd.to_datetime(df["transaction_date"]).dt.normalize().to_numpy(),
        "p": p, "y": data.y[rows], "amount_usd": data.features["amount_usd"].to_numpy(float)[rows],
        "channel": df["channel"].to_numpy(), "transaction_type": df["transaction_type"].to_numpy(),
        "country": df["country"].to_numpy(), "customer_segment": df.get("customer_segment", pd.Series(index=df.index)).to_numpy(),
        "customer_age_band": df.get("customer_age_band", pd.Series(index=df.index)).to_numpy(),
    })
    return model_id, frame
