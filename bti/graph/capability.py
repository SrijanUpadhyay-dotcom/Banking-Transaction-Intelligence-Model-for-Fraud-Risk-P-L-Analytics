# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Graph capability test: can the graph features see mule rings, and do they do no harm without them?

LightGBM candidates (not registered) are trained on the same split with three feature sets:
- the current set
- plus point-in-time graph features
- plus spectral and GraphSAGE features

They are run on (a) the data with injected synthetic rings and (b) the original data. The test reports
out-of-time PR-AUC and the share of *ring* frauds caught at 2% and 5% alert budgets.

Synthetic rings measure capability, not lift on a real book.

Usage:
  python -m bti.graph.capability      # writes outputs/graph/capability_test.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from bti.config import get_settings
from bti.graph.learned import torch_available
from bti.graph.synthetic_rings import NOTICE, write
from bti.modeling.train import default_data_path, prepare, train_candidate

SETS = ("core-relative-nb", "graph-relative", "graph-full-relative")


def run(algorithm: str = "lightgbm") -> Dict:
    s = get_settings()
    ring = write(pd.read_csv(default_data_path(), low_memory=False), Path(s.processed_data_dir) / "graph")
    results: Dict = {"notice": NOTICE, "rings": {k: ring[k] for k in ("rings", "ring_rows", "ring_fraud_rows",
                                                                     "ring_fraud_share_of_all_fraud")},
                     "algorithm": algorithm, "torch": torch_available(), "runs": []}
    for label, path in (("with_synthetic_rings", ring["path"]), ("original", None)):
        data = prepare(path)
        for fs in SETS:
            if fs == "graph-full-relative" and not torch_available():
                continue
            card = train_candidate(data, algorithm, fs, developer="graph-capability-test", register=False,
                                   auto_challenger=False)
            m = card["performance"]["metrics"]["out_of_time"]
            cap = card.get("synthetic_ring_capability")
            results["runs"].append({
                "data": label, "feature_set": fs, "gates": card["validation"]["status"],
                "failed_gates": [g["gate"] for g in card["validation"]["gates"] if not g["passed"]],
                "oot_roc_auc": m["roc_auc"], "oot_pr_auc": m["pr_auc"],
                "ring_fraud_caught_at_2pct": cap[0] if cap else None,
                "ring_fraud_caught_at_5pct": cap[1] if cap else None,
                "ring_frauds_out_of_time": cap[2] if cap else None,
                "top_features": [f["feature"] for f in card.get("explainability", {}).get("global_importance", [])[:6]],
            })
    out = Path(s.outputs_dir) / "graph" / "capability_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    results["path"] = str(out)
    return results


def main() -> None:
    r = run()
    print(r["notice"])
    for x in r["runs"]:
        ring = (f" | ring frauds caught {x['ring_fraud_caught_at_2pct']:.1%} @2%, {x['ring_fraud_caught_at_5pct']:.1%} @5% "
                f"(n={x['ring_frauds_out_of_time']})") if x["ring_fraud_caught_at_2pct"] is not None else ""
        print(f"{x['data']:21s} {x['feature_set']:20s} {x['gates']:6s} PR {x['oot_pr_auc']:.4f} ROC {x['oot_roc_auc']:.4f}{ring}")
    print("Written to", r["path"])


if __name__ == "__main__":
    main()
