"""
Render a registry card as a model documentation pack structured for
SR 11-7 / PRA SS1/23 review. Sign-off fields are left for the bank's model
owner and independent validator.
"""

from __future__ import annotations

from typing import Dict, List


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.2f}%"


def _num(x, nd=4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _table(headers: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_model_card(card: Dict, index: Dict) -> str:
    mid = card["model_id"]
    perf = card["performance"]
    oot = perf["metrics"]["out_of_time"]
    split = card["data"]["split"]
    roles = [r for r in ("champion", "challenger") if index.get(r) == mid]
    history = [h for h in index.get("history", []) if h["model_id"] == mid]
    s = []
    s.append(f"# Model documentation — {mid}\n")
    s.append(f"**Family:** {card['model_family']}  \n**Registered:** {card['created_at']}  \n"
             f"**Current role:** {', '.join(roles) or 'none'}  \n"
             f"**Automated validation:** {card['validation']['status'].upper()}\n")

    s.append("## 1. Ownership and approval\n")
    own = card["ownership"]
    s.append(_table(["Role", "Name", "Sign-off date"], [
        ["Model developer", own.get("developer") or "", ""],
        ["Model owner (business)", own.get("business_owner") or "_to be assigned_", ""],
        ["Independent validator", own.get("validator") or "_to be assigned_", ""],
        ["Model risk committee", "", ""],
    ]))
    s.append("")

    s.append("## 2. Purpose and use\n")
    s.append(card["purpose"]["intended_use"] + "\n")
    s.append(f"**Decision type:** {card['purpose']['decision_type']}\n")
    s.append("**Out of scope:**\n" + "\n".join(f"- {x}" for x in card["purpose"]["out_of_scope"]) + "\n")

    s.append("## 3. Regulatory mapping\n")
    s.append(_table(["Framework", "Relevance"], [[r["framework"], r["scope"]] for r in card["regulatory_mapping"]]))
    s.append("")

    s.append("## 4. Data\n")
    d = card["data"]
    s.append(_table(["Item", "Value"], [
        ["Source", f"`{d['path']}`"], ["SHA-256", f"`{d['sha256']}`"], ["Rows", f"{d['rows']:,}"],
        ["Fraud rate", _pct(d["fraud_rate"])], ["Window", f"{d['window'][0]} → {d['window'][1]}"],
        ["Reporting currency", f"{d['fx']['reporting_currency']} (rates {d['fx']['rates_as_of']})"],
    ]))
    s.append("\n**Out-of-time split** (never random):\n")
    s.append(_table(["Window", "From", "To", "Rows", "Fraud"], [
        ["Train", "start", split["train"]["to"], f"{split['train']['rows']:,}", split["train"]["fraud"]],
        ["Calibration", split["calibration"]["from"], split["calibration"]["to"],
         f"{split['calibration']['rows']:,}", split["calibration"]["fraud"]],
        ["Out-of-time test", split["out_of_time_test"]["from"], "end",
         f"{split['out_of_time_test']['rows']:,}", split["out_of_time_test"]["fraud"]],
    ]))
    s.append("")

    s.append("## 5. Features and exclusions\n")
    f = card["features"]
    s.append(f"{len(f['model_features'])} model features, all derived from fields known before the "
             f"authorisation decision; look-back {f['lookback_days']} days. Lineage is enforced in code.\n")
    imp = {r["feature"]: r["share"] for r in f.get("global_importance", [])}
    s.append(_table(["Feature", "Reason code", "Sources", "Importance"], [
        [m["name"], m["reason_code"], ", ".join(m["sources"]), _pct(imp.get(m["name"]))]
        for m in sorted(f["model_features"], key=lambda m: -imp.get(m["name"], 0))
    ]))
    s.append("\n**Excluded source fields:**\n")
    s.append(_table(["Field", "Classification", "Reason"],
                    [[e["field"], e["classification"], e["reason"]] for e in f["excluded_source_fields"]]))
    leaks = [r for r in f["source_leakage_audit"] if r["suspected_leak"]]
    if leaks:
        s.append("\n**Leakage screen on source data** (single-feature AUC ≥ 0.97 — all excluded):\n")
        s.append(_table(["Column", "Single-feature AUC", "Classification"],
                        [[r["column"], r["single_feature_auc"], r["catalogued_as"]] for r in leaks]))
    s.append(f"\nMonotone-increasing constraints: {', '.join(f['monotone_increasing'])}.\n")

    s.append("## 6. Methodology\n")
    m = card["methodology"]
    cal = m["calibration"]
    s.append(f"- Algorithm: {m['algorithm']} ({m['iterations_used']} boosting iterations)\n"
             f"- Calibration: {cal['method'] if isinstance(cal, dict) else cal}\n"
             f"- Categorical encoding: {m.get('categorical_encoding', '—')}\n"
             f"- Threshold policy: {m['threshold_policy']} (threshold {m['reference_threshold']:.4f})\n")

    s.append("## 7. Performance (out-of-time)\n")
    s.append(_table(["Metric", "Train", "Calibration", "Out-of-time"], [
        [k, _num(perf["metrics"]["train"].get(k)), _num(perf["metrics"]["calibration"].get(k)), _num(oot.get(k))]
        for k in ("roc_auc", "pr_auc", "ks", "gini", "brier", "ece")
    ]))
    s.append("\n**Operating points (out-of-time):**\n")
    s.append(_table(["Alert budget", "Precision", "TDR", "VDR", "ADR", "FP : TP"], [
        [_pct(b["alert_budget"]), _pct(b["precision"]), _pct(b["tdr"]), _pct(b.get("vdr")), _pct(b.get("adr")),
         b["false_positive_ratio"]] for b in perf["alert_budgets"]
    ]))
    s.append("\n**Performance by country (reference threshold):**\n")
    s.append(_table(["Country", "n", "Fraud rate", "ROC-AUC", "Alert rate"], [
        [r["segment"], f"{r['n']:,}", _pct(r["fraud_rate"]), _num(r["roc_auc"]), _pct(r["alert_rate"])]
        for r in perf["segments"].get("country", [])
    ]))
    legacy = perf.get("legacy_benchmark") or {}
    if legacy:
        s.append(f"\n**Legacy benchmark:** ROC-AUC {legacy.get('roc_auc')} — {legacy.get('note')}\n")

    s.append("## 8. Fairness\n")
    fair = card["fairness"]
    s.append(f"Status: **{fair['status'].upper()}**. {fair.get('note', '')}\n")
    for name, run in fair.get("operating_points", {}).items():
        worst = []
        for a in run["attributes"]:
            ratios = [g["fpr_ratio"] for g in a["groups"] if g["fpr_ratio"] is not None]
            worst.append(f"{a['attribute']} max FPR ratio {max(ratios):.2f}" if ratios else f"{a['attribute']} n/a")
        s.append(f"- {name}: overall FPR {_pct(run['overall_fpr'])}; " + "; ".join(worst))
    s.append("")
    remediation = card["validation"].get("remediation_log") or []
    if remediation:
        s.append("**Remediation log:**\n")
        for r in remediation:
            s.append(f"- *Finding:* {r['finding']}\n  *Root cause:* {r['root_cause']}\n"
                     f"  *Remediation:* {r['remediation']}\n  *Re-test:* {r['retest']}")
        s.append("")

    s.append("## 9. Validation gates\n")
    s.append(_table(["Gate", "Result", "Detail"],
                    [[g["gate"], "PASS" if g["passed"] else "FAIL", g["detail"]] for g in card["validation"]["gates"]]))
    s.append("")

    s.append("## 10. Limitations and assumptions\n")
    s.append("\n".join(f"- {x}" for x in card["limitations"]) + "\n")

    s.append("## 11. Ongoing monitoring plan\n")
    s.append("\n".join(f"- **{k.replace('_', ' ').title()}:** {v}" for k, v in card["monitoring"]["plan"].items()) + "\n")

    s.append("## 12. Change and approval history\n")
    s.append(_table(["When", "Role", "Approver", "Rationale"],
                    [[h["at"], h["role"], h["approver"], h["rationale"]] for h in history]) if history
             else "_No role assignments yet._")
    s.append(f"\n\n---\nProvenance: git `{card['provenance'].get('git_sha')}`, Python {card['provenance']['python']}, "
             f"scikit-learn {card['provenance']['sklearn']}, pandas {card['provenance']['pandas']}.")
    return "\n".join(s) + "\n"
