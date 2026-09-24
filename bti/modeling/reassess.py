"""
Re-assess registered models under the current fairness method.

Registered cards are immutable. When the fairness method changes, every model
is re-tested on the data it was trained on, using the stored artifact, and the
outcome is appended to the registry as a note, including where the new verdict
differs from the one recorded at registration.

Usage:
  python -m bti.modeling.reassess --author "Srijan Upadhyay"           # report only
  python -m bti.modeling.reassess --author "Srijan Upadhyay" --record  # also append registry notes
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from bti.modeling import registry
from bti.modeling.features import FEATURE_NAMES, to_model_matrix
from bti.modeling.train import TrainingData, assess_fairness, prepare


def model_probabilities(model_id: str, data: TrainingData):
    art = registry.load_artifact(model_id)
    X = to_model_matrix(data.features, art["encodings"], art.get("feature_names", FEATURE_NAMES))
    return art["calibrator"].predict(art["estimator"].predict_proba(X)[:, 1])


def reassess_fairness(model_ids: Optional[List[str]] = None, data_path=None) -> List[Dict]:
    model_ids = model_ids or [m["model_id"] for m in registry.read_index()["models"]]
    prepared: Dict[int, TrainingData] = {}
    out = []
    for model_id in model_ids:
        card = registry.load_card(model_id)
        version = registry.load_artifact(model_id).get("feature_version", 1)
        if version not in prepared:
            prepared[version] = prepare(data_path, feature_version=version)
        data = prepared[version]
        if card.get("data", {}).get("sha256") not in (None, data.sha256):
            out.append({"model_id": model_id, "skipped": "trained on different data"})
            continue
        fairness = assess_fairness(data, model_probabilities(model_id, data))
        out.append({"model_id": model_id, "registered_status": card["fairness"]["status"],
                    "status": fairness["status"], "findings": fairness["findings"],
                    "watchlist": fairness["watchlist"], "method": fairness["method"]})
    return out


def _describe(r: Dict) -> str:
    def items(rows):
        return "; ".join(sorted({f"{f['attribute']}={f['group']} pooled {f['pooled_fpr_ratio']}x "
                                 f"(q {f['pooled_q_value']})" for f in rows})) or "none"
    changed = "" if r["status"] == r["registered_status"] else \
        f" This differs from the registered verdict ({r['registered_status']})."
    return (f"Fairness re-assessed under the pooled, corrected method: {r['status']}.{changed} "
            f"Findings: {items(r['findings'])}. Watchlist: {items(r['watchlist'])}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-assess registered models under the current fairness method")
    parser.add_argument("--author", required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--record", action="store_true", help="Append the outcome to the registry as notes")
    args = parser.parse_args()
    for r in reassess_fairness(args.models):
        if r.get("skipped"):
            print(f"{r['model_id']:32s} skipped: {r['skipped']}")
            continue
        print(f"{r['model_id']:32s} registered {r['registered_status']:16s} now {r['status']}")
        print(f"    {_describe(r)}")
        if args.record:
            registry.add_note(r["model_id"], args.author, "Fairness re-assessment", _describe(r),
                              {"status": r["status"], "findings": r["findings"], "watchlist": r["watchlist"],
                               "method": r["method"]})


if __name__ == "__main__":
    main()
