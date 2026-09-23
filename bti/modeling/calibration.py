"""Platt (sigmoid) calibration on the logit of the raw score — strictly monotone, so ranking is preserved."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

_CLIP = 1e-7


class PlattCalibrator:
    def __init__(self):
        self.slope: float = 1.0
        self.intercept: float = 0.0

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        z = logit(np.clip(np.asarray(raw, dtype=float), _CLIP, 1 - _CLIP)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y, dtype=int))
        self.slope, self.intercept = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        z = logit(np.clip(np.asarray(raw, dtype=float), _CLIP, 1 - _CLIP))
        return expit(self.slope * z + self.intercept)

    def describe(self) -> dict:
        return {"method": "platt_sigmoid_on_logit", "slope": round(self.slope, 6),
                "intercept": round(self.intercept, 6)}
