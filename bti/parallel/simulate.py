"""
Rehearsal of the parallel-run report on the synthetic data. This is NOT a
comparison with SAS.

There is no SAS output for the synthetic transactions. To exercise the report
end to end, a simple rules engine stands in for the incumbent. It reads
pre-authorisation fields only (failed and excess logins, amount against the
customer's own average, new device or IP, night-time, large amounts), and its
thresholds were fixed before looking at any results. BTI was trained on data
from the same generator, so its advantage here is expected and says nothing
about a real incumbent.

What the rehearsal shows is the shape of the evidence a bank pilot produces,
and that every part of the report computes. Real lift can only be measured on a
bank's own traffic, with the bank's incumbent decisions and matured labels.

Usage:
  python -m bti.parallel.simulate            # prints a summary, writes outputs/parallel_run/rehearsal.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from bti.config import get_settings
from bti.jurisdiction.policies import policy_for
from bti.modeling import registry
from bti.operations.capacity import capacity_overrides
from bti.operations.decisioning import decide
from bti.parallel.report import compare_systems

STAND_IN = "RULES-STAND-IN"
NOTICE = ("Rehearsal on synthetic data against a simple rules stand-in, not SAS. BTI was trained on the same "
          "generator, so its advantage here is expected and is not evidence of lift over any real incumbent.")


def rules_stand_in(features: pd.DataFrame) -> pd.DataFrame:
    """A deliberately ordinary pre-authorisation rules engine. Thresholds fixed in advance, never tuned."""
    f = features
    points = (40 * (f["failed_attempt_count"].fillna(0) >= 3) + 25 * (f["login_attempts"].fillna(0) >= 4)
              + 30 * (f["amount_vs_hist_avg"].fillna(0) >= 5) + 20 * (f["device_new_for_customer"] == 1)
              + 15 * (f["ip_new_for_customer"] == 1) + 10 * (f["is_off_hours"] == 1)
              + 15 * (f["amount_usd"].fillna(0) >= 5_000))
    decision = np.select([points >= 70, points >= 50, points >= 35], ["DECLINE", "REVIEW", "STEP_UP"], "APPROVE")
    return pd.DataFrame({"incumbent_score": points.astype(float), "incumbent_decision": decision}, index=f.index)


def rehearse(model_id: str | None = None, bootstrap_reps: int = 500) -> Dict:
    from bti.modeling.reassess import model_probabilities
    from bti.modeling.train import prepare

    model_id = model_id or registry.model_for_role("champion") or registry.model_for_role("challenger")
    data = prepare()
    p = model_probabilities(model_id, data)
    te = data.te
    df, feats = data.df.loc[te], data.features.loc[te]
    provisional = registry.model_for_role("champion") != model_id
    capacity = capacity_overrides(model_id)
    decisions = [decide(float(pi), float(a), policy_for(c), ch, tt, provisional_model=provisional,
                        cost_overrides=capacity).action
                 for pi, a, c, ch, tt in zip(p[te], feats["amount_usd"].fillna(0), df["country"], df["channel"],
                                             df["transaction_type"])]
    frame = pd.DataFrame({"transaction_id": df["transaction_id"].to_numpy(), "bti_probability": p[te],
                          "bti_decision": decisions, "amount_usd": feats["amount_usd"].to_numpy(),
                          "label": data.y[te].astype(float)})
    frame = pd.concat([frame, rules_stand_in(feats).reset_index(drop=True)], axis=1)
    result = compare_systems(frame, STAND_IN, bootstrap_reps=bootstrap_reps)
    result.update({"notice": NOTICE, "bti_model": model_id, "bti_provisional": provisional,
                   "bti_capacity_policy": capacity,
                   "window": "synthetic out-of-time test window (last 25% by time)"})
    return result


def main() -> None:
    r = rehearse()
    out = Path(get_settings().outputs_dir) / "parallel_run" / "rehearsal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, default=str))
    eq = r.get("at_equal_intervention_rate", {})
    print(r["notice"], "\n")
    print(f"Paired transactions {r['paired_transactions']:,}; matured frauds {r['labels']['fraud']:,}")
    print(f"Decision agreement {r['operational']['decision_agreement']:.1%}; "
          f"intervention agreement {r['operational']['intervention_agreement']:.1%}")
    if eq:
        print(f"At equal intervention rate ({eq['incumbent']['interventions']:,} each):")
        print(f"  {STAND_IN}: TDR {eq['incumbent']['tdr']}, VDR {eq['incumbent']['vdr']}, "
              f"genuine disturbed {eq['incumbent']['genuine_intervened']:,}")
        print(f"  BTI:            TDR {eq['bti']['tdr']}, VDR {eq['bti']['vdr']}, "
              f"genuine disturbed {eq['bti']['genuine_intervened']:,}")
        print(f"  TDR difference {eq['tdr_difference']} (95% CI {eq['tdr_difference_ci95']}); "
              f"McNemar p {r['fraud_overlap']['mcnemar_p_value']}; verdict {r['verdict']}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
