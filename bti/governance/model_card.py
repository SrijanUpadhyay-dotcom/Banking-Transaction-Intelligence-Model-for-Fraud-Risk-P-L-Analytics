"""
Render a registry card as a model documentation pack structured for
SR 11-7 / PRA SS1/23 review. Sections 13–15 carry the validation evidence:
benchmarking, sensitivity and stress tests; outcomes analysis on matured
labels; and the independent validation record (sign-offs, findings, readiness).
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.2f}%"


def _num(x, nd=4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _table(headers: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_model_card(card: Dict, index: Dict, validation: Optional[Dict] = None) -> str:
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
    s.append(f"Status: **{fair['status'].upper()}**. {fair.get('method') or fair.get('note', '')}\n")
    for f in fair.get("findings", []):
        if "pooled_fpr_ratio" in f:
            s.append(f"- **Finding** ({f['operating_point']}): {f['attribute']} = {f['group']}, pooled FPR ratio "
                     f"{f['pooled_fpr_ratio']} (q {f['pooled_q_value']}), by window {f['window_ratios']}")
        else:
            s.append(f"- **Finding** ({f['operating_point']}): {f['attribute']} = {f['group']}, FPR ratio "
                     f"{f['fpr_ratio']} (p {f.get('fpr_p_value')})")
    for f in fair.get("watchlist", []):
        where = f" in {f['window']}" if f.get("window") else ""
        s.append(f"- Watchlist ({f['operating_point']}): {f['attribute']} = {f['group']}, FPR ratio "
                 f"{f.get('fpr_ratio', f.get('pooled_fpr_ratio'))}{where}; pooled {f.get('pooled_fpr_ratio')} "
                 f"(q {f.get('pooled_q_value')}) — {f['reason']}")
    for name, run in fair.get("operating_points", {}).items():
        run = run.get("pooled", run)
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
                     f"  *Remediation:* {r['remediation']}\n  *Re-test:* {r['retest']}"
                     + (f"\n  *Re-assessment:* {r['reassessment']}" if r.get("reassessment") else ""))
        s.append("")
    notes = [n for n in index.get("notes", []) if n["model_id"] == card["model_id"]]
    if notes:
        s.append("**Post-registration notes** (appended to the registry; the card above is unchanged):\n")
        for n in notes:
            s.append(f"- {n['at'][:10]} — *{n['subject']}* ({n['author']}): {n['detail']}")
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
    s.append("")
    s.extend(_validation_sections(validation or {}))
    s.append(f"\n\n---\nProvenance: git `{card['provenance'].get('git_sha')}`, Python {card['provenance']['python']}, "
             f"scikit-learn {card['provenance']['sklearn']}, pandas {card['provenance']['pandas']}.")
    return "\n".join(s) + "\n"


def _validation_sections(v: Dict) -> List[str]:
    s: List[str] = []
    s.append("## 13. Benchmarking, sensitivity and stress testing\n")
    b = v.get("benchmark")
    if not b:
        s.append("_No benchmark report yet: POST /governance/models/{id}/benchmark._\n")
    else:
        bm = b["benchmarks"]
        s.append(f"Out-of-time window; generated {b['generated_at'][:19]}.\n")
        s.append(_table(["Model", "ROC-AUC", "PR-AUC", "Detail"], [
            ["This model", bm["model"]["roc_auc"], bm["model"]["pr_auc"],
             f"PR-AUC 95% CI {bm['model']['pr_auc_ci95']}"],
            ["Logistic regression", bm["logistic_regression"]["roc_auc"], bm["logistic_regression"]["pr_auc"],
             bm["logistic_regression"]["specification"]],
            ["Rules stand-in", bm["rules_stand_in"]["roc_auc"], bm["rules_stand_in"]["pr_auc"],
             "pre-authorisation rules"],
        ]))
        s.append(f"\n{bm['note']}\n")
        s.append("**Temporal stability (out-of-time months):**\n")
        s.append(_table(["Month", "n", "Fraud", "ROC-AUC", "PR-AUC", "ECE"],
                        [[m["month"], m["n"], m["fraud"], m["roc_auc"], m["pr_auc"], m["ece"]]
                         for m in b["temporal_stability"]]))
        s.append("\n**Sensitivity (±1 SD, top features):**\n")
        s.append(_table(["Feature", "Mean |Δp| (+1 SD)", "Decision flips at 10% budget (+1 SD)"],
                        [[f["feature"], f["plus_1sd"]["mean_abs_change"], _pct(f["plus_1sd"]["flips_at_10pct_threshold"])]
                         for f in b["sensitivity"]["features"][:8]]))
        s.append(f"\n**Monotonicity:** {'holds for every constrained feature' if b['monotonicity']['all_monotone'] else 'VIOLATED'}"
                 f" ({len(b['monotonicity']['features'])} features swept).\n")
        s.append("**Stress scenarios (2% alert-budget threshold fixed on the calibration window):**\n")
        s.append(_table(["Scenario", "Alert rate", "Detection", "ROC-AUC", "Flags"],
                        [[r["scenario"], _pct(r["at_2pct_threshold"]["alert_rate"]),
                          _pct(r["at_2pct_threshold"]["tdr"]), r.get("roc_auc", "—"), "; ".join(r.get("flags", [])) or "—"]
                         for r in b["stress"]]))
        s.append("")
    s.append("## 14. Outcomes analysis on matured labels\n")
    o = v.get("outcomes")
    if not o:
        s.append("_No outcomes analysis on matured production labels yet (quarterly; needs 90-day-old outcomes)._\n")
    else:
        live, dev = o.get("live", {}), o.get("development_out_of_time", {})
        s.append(f"Latest run: {o.get('window', '')}; {o.get('matured', 0)} matured transactions, {o.get('fraud', 0)} frauds; "
                 f"status {o.get('status')}.\n")
        if o.get("status") == "ok":
            s.append(_table(["Metric", "Live (matured)", "Development"],
                            [[k, live.get(k), dev.get(k)] for k in ("roc_auc", "pr_auc", "ks", "ece")]))
            s.append("\n" + ("\n".join(f"- **{f['severity']}**: {f['title']} — {f['detail']}" for f in o.get("flags", []))
                             or "No degradation beyond tolerance.") + "\n")
    s.append("## 15. Independent validation\n")
    r = v.get("readiness")
    if r:
        s.append(f"**Ready for champion:** {'YES' if r['ready'] else 'NO'}\n")
        s.append(_table(["Check", "Result", "Detail"],
                        [[c["check"], "PASS" if c["passed"] else "FAIL", c["detail"]] for c in r["checks"]]))
        if r.get("conditions"):
            s.append(f"\n**Conditions of approval:** {r['conditions']}")
        s.append("")
    signoffs = v.get("signoffs") or []
    s.append("**Sign-offs:**\n")
    s.append(_table(["Signed", "Validator", "Role", "Decision", "Valid until", "Scope"],
                    [[x["signed_at"][:10], x["validator"], x.get("validator_role") or "—", x["decision"],
                      x["valid_until"][:10], x["scope"]] for x in signoffs]) if signoffs else "_None recorded._")
    findings = v.get("findings") or []
    s.append("\n**Findings:**\n")
    s.append(_table(["#", "Severity", "Status", "Title", "Owner", "Due", "Source"],
                    [[f["id"], f["severity"], f["status"] + (" (overdue)" if f.get("overdue") else ""), f["title"],
                      f["owner"], f["due_date"][:10], f["source"]] for f in findings]) if findings else "_None recorded._")
    s.append("")
    return s
