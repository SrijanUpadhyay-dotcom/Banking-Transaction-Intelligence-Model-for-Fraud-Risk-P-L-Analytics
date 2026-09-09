"""
BTI Model Lift Analytics

Computes lift table, KS statistic, Gini coefficient, and incremental lift
from the saved scored dataset. Reconstructs the held-out test split using
the same parameters as the training script (test_size=0.25, random_state=42).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from bti.config import get_settings
from bti.logging_config import get_logger

log = get_logger("analytics.lift")

_SCORE_COL    = "rf_fraud_proba"
_LABEL_COL    = "fraud_flag"
_TEST_SIZE    = 0.25
_RANDOM_STATE = 42
_N_DECILES    = 10


@dataclass
class DecileBucket:
    decile: int
    score_min: float
    score_max: float
    n_transactions: int
    n_fraud: int
    fraud_rate: float
    cum_fraud_captured: float
    lift: float
    precision: float


@dataclass
class LiftReport:
    n_test: int
    n_fraud_test: int
    fraud_rate_baseline: float
    roc_auc: float
    gini: float
    ks_stat: float
    ks_decile: int
    deciles: List[DecileBucket]
    computed_at: str


@dataclass
class IncrementalLiftReport:
    n_records: int
    n_fraud: int
    bti_roc_auc: float
    baseline_roc_auc: float
    incremental_auc: float
    bti_ks: float
    baseline_ks: float
    bti_top_decile_lift: float
    baseline_top_decile_lift: float
    incremental_top_decile_lift: float
    bti_top_decile_precision: float
    baseline_top_decile_precision: float
    computed_at: str


def _scored_csv_path() -> Path:
    settings = get_settings()
    return Path(settings.processed_data_dir) / "banking_transactions_ml_scored.csv"


def _load_test_set() -> tuple[np.ndarray, np.ndarray]:
    path = _scored_csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Scored dataset not found at {path}. "
            "Run the ML pipeline (POST /api/v1/pipeline/run) first."
        )
    df = pd.read_csv(path, usecols=[_SCORE_COL, _LABEL_COL])
    df[_SCORE_COL] = pd.to_numeric(df[_SCORE_COL], errors="coerce").clip(0, 1).fillna(0)
    df[_LABEL_COL] = pd.to_numeric(df[_LABEL_COL], errors="coerce").fillna(0).astype(int)

    indices = np.arange(len(df))
    y_all = df[_LABEL_COL].values
    _, test_idx = train_test_split(
        indices, test_size=_TEST_SIZE, stratify=y_all, random_state=_RANDOM_STATE
    )
    return df[_SCORE_COL].values[test_idx], y_all[test_idx]


def _ks_stat(y_score: np.ndarray, y_true: np.ndarray) -> tuple[float, int]:
    order = np.argsort(-y_score)
    ys = y_true[order]
    n_pos = max(int(ys.sum()), 1)
    n_neg = max(len(ys) - n_pos, 1)
    tpr = np.cumsum(ys) / n_pos
    fpr = np.cumsum(1 - ys) / n_neg
    ks_vals = np.abs(tpr - fpr)
    ks_idx = int(np.argmax(ks_vals))
    ks_decile = max(1, int(np.ceil((ks_idx + 1) / len(ys) * _N_DECILES)))
    return float(ks_vals[ks_idx]), ks_decile


def compute_lift() -> LiftReport:
    t0 = time.time()
    y_score, y_true = _load_test_set()

    n_test = len(y_true)
    n_fraud = int(y_true.sum())
    if n_fraud == 0:
        raise ValueError("No fraud cases found in test set.")

    auc = float(roc_auc_score(y_true, y_score))
    gini = 2 * auc - 1
    ks, ks_decile = _ks_stat(y_score, y_true)

    order = np.argsort(-y_score)
    ys_sorted = y_true[order]
    ss_sorted = y_score[order]

    bucket = n_test // _N_DECILES
    cum_fraud = 0
    deciles: List[DecileBucket] = []

    for i in range(_N_DECILES):
        start = i * bucket
        end = start + bucket if i < _N_DECILES - 1 else n_test
        by = ys_sorted[start:end]
        bs = ss_sorted[start:end]
        n_in = len(by)
        nf = int(by.sum())
        cum_fraud += nf
        cum_cap = cum_fraud / n_fraud
        frac = (i + 1) / _N_DECILES
        lift = cum_cap / frac if frac else 0
        deciles.append(DecileBucket(
            decile=i + 1,
            score_min=round(float(bs.min()), 4),
            score_max=round(float(bs.max()), 4),
            n_transactions=n_in,
            n_fraud=nf,
            fraud_rate=round(nf / n_in if n_in else 0, 4),
            cum_fraud_captured=round(cum_cap, 4),
            lift=round(lift, 3),
            precision=round(nf / n_in if n_in else 0, 4),
        ))

    log.info(
        "Lift computed",
        extra={"n_test": n_test, "n_fraud": n_fraud, "auc": round(auc, 4),
               "ks": round(ks, 4), "ms": round((time.time() - t0) * 1000)},
    )

    return LiftReport(
        n_test=n_test,
        n_fraud_test=n_fraud,
        fraud_rate_baseline=round(n_fraud / n_test, 4),
        roc_auc=round(auc, 4),
        gini=round(gini, 4),
        ks_stat=round(ks, 4),
        ks_decile=ks_decile,
        deciles=deciles,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def compare_with_baseline(
    records: list[dict],
    baseline_score_field: str = "sas_score",
    label_field: str = "label",
) -> IncrementalLiftReport:
    """
    Compare BTI lift vs. a baseline on the same labeled records.
    Each record must have: {label: 0|1, sas_score: 0-1, bti_score?: 0-1}
    If bti_score is absent, the record is scored live via RealTimeScorer.
    """
    from bti.scoring.realtime import RealTimeScorer
    scorer = RealTimeScorer()

    y_true_list, bti_list, base_list = [], [], []
    for rec in records:
        label = int(rec.get(label_field, 0))
        base  = float(rec.get(baseline_score_field, 0))
        bti   = float(rec.get("bti_score", -1))
        if bti < 0:
            try:
                result = scorer.score(rec)
                bti = result.final_risk_score / 100.0
            except Exception:
                bti = base  # graceful fallback
        y_true_list.append(label)
        bti_list.append(min(max(bti, 0.0), 1.0))
        base_list.append(min(max(base, 0.0), 1.0))

    y_true = np.array(y_true_list)
    bti_scores  = np.array(bti_list)
    base_scores = np.array(base_list)

    n_fraud = int(y_true.sum())
    if n_fraud == 0:
        raise ValueError("No positive (fraud) labels found. Each record needs label=1 for fraud.")

    bti_auc  = float(roc_auc_score(y_true, bti_scores))
    base_auc = float(roc_auc_score(y_true, base_scores))
    bti_ks,  _ = _ks_stat(bti_scores, y_true)
    base_ks, _ = _ks_stat(base_scores, y_true)

    def top_decile(scores: np.ndarray) -> tuple[float, float]:
        n = len(scores)
        top_n = max(1, n // 10)
        idx = np.argsort(-scores)[:top_n]
        captured = y_true[idx].sum() / n_fraud
        precision = float(y_true[idx].mean())
        lift = round(captured / 0.1, 3)
        return lift, round(precision, 4)

    bti_lift,  bti_prec  = top_decile(bti_scores)
    base_lift, base_prec = top_decile(base_scores)

    return IncrementalLiftReport(
        n_records=len(y_true),
        n_fraud=n_fraud,
        bti_roc_auc=round(bti_auc, 4),
        baseline_roc_auc=round(base_auc, 4),
        incremental_auc=round(bti_auc - base_auc, 4),
        bti_ks=round(bti_ks, 4),
        baseline_ks=round(base_ks, 4),
        bti_top_decile_lift=bti_lift,
        baseline_top_decile_lift=base_lift,
        incremental_top_decile_lift=round(bti_lift - base_lift, 3),
        bti_top_decile_precision=bti_prec,
        baseline_top_decile_precision=base_prec,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
