"""
Challenger tournament: train several candidate models on identical data and
split, register every one of them (the alternatives considered are part of the
model-risk record), and assign the challenger role to a new model only if it
beats the incumbent.

Selection uses the calibration window — never the out-of-time test window —
so the reported out-of-time figures remain an unbiased estimate. A candidate
must pass every validation gate and beat the incumbent's calibration-window
PR-AUC by `min_gain` to take the challenger slot. Champion promotion remains a
separate, human, four-eyes decision.

Remediation mode (`--remediation "<finding>"`) is for replacing an incumbent
with a known defect: the incumbent cannot win, and the best new candidate that
passes every gate replaces it if it is non-inferior (calibration PR-AUC no more
than `min_gain` below the incumbent's). The finding is recorded with the role
change.

Usage:
  python -m bti.modeling.tournament --developer "Srijan Upadhyay" \\
      --algorithms hgb lightgbm xgboost --feature-sets core extended --tune
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Dict, List

from bti.logging_config import get_logger
from bti.modeling import algorithms, registry
from bti.modeling.train import MONOTONE_INCREASING, prepare, train_candidate
from bti.modeling.tuning import tune as tune_candidate

log = get_logger("modeling.tournament")

DEFAULT_MIN_GAIN = 0.005


def _summary(card: Dict, incumbent: bool = False) -> Dict:
    m = card["performance"]["metrics"]
    return {
        "model_id": card["model_id"],
        "incumbent": incumbent,
        "algorithm": card["methodology"].get("algorithm_key", "hgb"),
        "feature_set": card["features"].get("feature_set", "core"),
        "validation": card["validation"]["status"],
        "failed_gates": [g["gate"] for g in card["validation"]["gates"] if not g["passed"]],
        "calibration_pr_auc": m["calibration"].get("pr_auc"),
        "calibration_roc_auc": m["calibration"].get("roc_auc"),
        "oot_pr_auc": m["out_of_time"].get("pr_auc"),
        "oot_roc_auc": m["out_of_time"].get("roc_auc"),
        "oot_ks": m["out_of_time"].get("ks"),
        "oot_ece": m["out_of_time"].get("ece"),
        "tuned": bool(card["methodology"].get("tuning")),
    }


def run_tournament(developer: str, algorithm_names: List[str], feature_sets: List[str], tune: bool = False,
                   data_path=None, min_gain: float = DEFAULT_MIN_GAIN, assign: bool = True,
                   remediation: str = "") -> Dict:
    t0 = time.time()
    data = prepare(data_path)
    incumbent_id = registry.model_for_role("challenger") or registry.model_for_role("champion")
    entrants: List[Dict] = []
    if incumbent_id:
        entrants.append(_summary(registry.load_card(incumbent_id), incumbent=True))

    for algorithm in algorithm_names:
        for feature_set in feature_sets:
            tuning = tune_candidate(data, algorithm, feature_set, MONOTONE_INCREASING) if tune else None
            params = tuning["best_params"] if tuning else None
            if tuning:
                tuning = {k: v for k, v in tuning.items() if k != "results"} | {"top_results": tuning["results"][:5]}
            card = train_candidate(data, algorithm, feature_set, params, developer, register=True,
                                   tuning=tuning, auto_challenger=False)
            entrants.append(_summary(card))
            log.info("Candidate registered", extra={"model_id": card["model_id"],
                                                     "validation": card["validation"]["status"]})

    eligible = [e for e in entrants if e["validation"] == "passed" and not (remediation and e["incumbent"])]
    best = max(eligible, key=lambda e: e["calibration_pr_auc"], default=None)
    incumbent = next((e for e in entrants if e["incumbent"]), None)

    if remediation and best is not None and incumbent is not None:
        floor = incumbent["calibration_pr_auc"] - min_gain
        if best["calibration_pr_auc"] >= floor:
            decision = {"outcome": "remediation_replacement", "challenger": best["model_id"],
                        "previous": incumbent_id, "finding": remediation,
                        "reason": f"{best['model_id']} is non-inferior (calibration PR-AUC {best['calibration_pr_auc']}"
                                  f" vs incumbent {incumbent['calibration_pr_auc']}, floor {round(floor, 4)}) and "
                                  f"passes every gate"}
            if assign:
                registry.assign_role(best["model_id"], "challenger", approver="bti.modeling.tournament",
                                     rationale=f"Remediation of finding: {remediation}. {decision['reason']}. "
                                               f"Selected on the calibration window.")
        else:
            decision = {"outcome": "remediation_blocked", "challenger": incumbent_id, "finding": remediation,
                        "reason": f"Best fix {best['model_id']} ({best['calibration_pr_auc']}) is below the "
                                  f"non-inferiority floor {round(floor, 4)}; escalate for a risk decision"}
    elif best is None:
        decision = {"outcome": "no_eligible_candidate", "challenger": incumbent_id}
    elif best["incumbent"]:
        decision = {"outcome": "incumbent_retained", "challenger": incumbent_id,
                    "reason": f"No candidate beat the incumbent's calibration PR-AUC "
                              f"({incumbent['calibration_pr_auc']}) by {min_gain}"}
    elif incumbent is not None and best["calibration_pr_auc"] < incumbent["calibration_pr_auc"] + min_gain:
        decision = {"outcome": "incumbent_retained", "challenger": incumbent_id,
                    "reason": f"Best candidate {best['model_id']} ({best['calibration_pr_auc']}) did not beat the "
                              f"incumbent ({incumbent['calibration_pr_auc']}) by the required {min_gain}"}
    else:
        decision = {"outcome": "new_challenger", "challenger": best["model_id"], "previous": incumbent_id,
                    "reason": f"{best['model_id']} calibration PR-AUC {best['calibration_pr_auc']} vs incumbent "
                              f"{incumbent['calibration_pr_auc'] if incumbent else 'none'}"}
        if assign:
            registry.assign_role(best["model_id"], "challenger", approver="bti.modeling.tournament",
                                 rationale=f"Tournament winner: {decision['reason']}. Selected on the calibration "
                                           f"window; out-of-time figures unused for selection.")

    ranked_cal = [e["model_id"] for e in sorted(eligible, key=lambda e: -e["calibration_pr_auc"])]
    ranked_oot = [e["model_id"] for e in sorted(eligible, key=lambda e: -(e["oot_pr_auc"] or 0))]
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "developer": developer,
        "selection_rule": (f"Remediation: incumbent excluded; best passing candidate replaces it if non-inferior "
                           f"(within {min_gain} calibration PR-AUC)." if remediation else
                           f"Passed all validation gates; highest calibration-window PR-AUC; must beat the "
                           f"incumbent by {min_gain}.") + " Out-of-time window reported, never used to select.",
        "remediation_finding": remediation or None,
        "tuned": tune,
        "entrants": sorted(entrants, key=lambda e: -(e["calibration_pr_auc"] or 0)),
        "decision": decision,
        "rank_agreement": {"calibration_winner": ranked_cal[0] if ranked_cal else None,
                           "out_of_time_winner": ranked_oot[0] if ranked_oot else None,
                           "agree": bool(ranked_cal and ranked_oot and ranked_cal[0] == ranked_oot[0])},
        "seconds": round(time.time() - t0, 1),
    }
    folder = registry.registry_dir() / "tournaments"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"tournament-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candidate models and select a challenger")
    parser.add_argument("--developer", required=True)
    parser.add_argument("--algorithms", nargs="+", choices=algorithms.ALGORITHMS, default=list(algorithms.ALGORITHMS))
    parser.add_argument("--feature-sets", nargs="+", choices=["core", "extended"], default=["core", "extended"])
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN)
    parser.add_argument("--remediation", default="", help="Finding being remediated; switches to non-inferiority")
    args = parser.parse_args()
    r = run_tournament(args.developer, args.algorithms, args.feature_sets, args.tune, min_gain=args.min_gain,
                       remediation=args.remediation)
    print(f"{'model':32s} {'algo/features':20s} {'gates':10s} {'cal PR':>7s} {'OOT PR':>7s} {'OOT ROC':>8s} {'ECE':>7s}")
    for e in r["entrants"]:
        tag = " (incumbent)" if e["incumbent"] else ""
        print(f"{e['model_id']:32s} {e['algorithm'] + '/' + e['feature_set']:20s} "
              f"{'PASS' if e['validation'] == 'passed' else ','.join(e['failed_gates'])[:10]:10s} "
              f"{e['calibration_pr_auc']:7.4f} {e['oot_pr_auc']:7.4f} {e['oot_roc_auc']:8.4f} {e['oot_ece']:7.4f}{tag}")
    print(f"\nDecision: {r['decision']['outcome']} → challenger {r['decision']['challenger']}")
    if r["decision"].get("reason"):
        print(f"  {r['decision']['reason']}")
    print(f"Calibration and out-of-time winners agree: {r['rank_agreement']['agree']}")
    print(f"Report: {r['report_path']}  ({r['seconds']}s)")


if __name__ == "__main__":
    main()
