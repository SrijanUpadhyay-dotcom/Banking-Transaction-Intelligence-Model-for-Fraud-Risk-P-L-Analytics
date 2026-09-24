"""Fairness method: multiple-testing correction, pooled windows, watchlist, remediation acceptance, registry notes."""

import numpy as np
import pytest

from bti.governance.fairness import benjamini_hochberg, fairness_assessment, fairness_report
from bti.modeling import registry
from bti.modeling.tournament import remediation_check


def test_benjamini_hochberg_matches_reference_values():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5, 0.002])
    assert q == pytest.approx([0.025, 0.05, 0.05, 0.5, 0.01])
    assert benjamini_hochberg([]) == []


def test_correction_removes_chance_findings_across_many_groups():
    # 30 groups with identical true FPR: uncorrected testing finds something most of the time.
    rng = np.random.default_rng(3)
    n = 30 * 400
    groups = np.repeat([f"g{i}" for i in range(30)], 400)
    y = np.zeros(n, dtype=int)
    flagged = rng.random(n) < 0.05
    report = fairness_report(y, flagged, {"segment": groups})
    assert report["tests"] == 30
    assert report["status"] == "pass"


def _windowed(disparity_in):
    """Two windows of 6,000 legitimate customers; group B is flagged 2.4x as often in the listed windows."""
    rng = np.random.default_rng(11)
    n = 12000
    window = np.repeat(["w1", "w2"], 6000)
    group = np.tile(np.repeat(["A", "B", "C"], 2000), 2)
    rate = np.where(np.isin(window, disparity_in) & (group == "B"), 0.12, 0.05)
    p = np.where(rng.random(n) < rate, 0.9, 0.1)
    return np.zeros(n, dtype=int), p, {"w1": window == "w1", "w2": window == "w2"}, {"segment": group}


def test_disparity_present_in_every_window_fails_the_gate():
    y, p, windows, groups = _windowed(["w1", "w2"])
    result = fairness_assessment(y, p, {"op": 0.5}, windows, groups)
    assert result["status"] == "review_required"
    [finding] = result["findings"]
    assert finding["group"] == "B" and finding["pooled_q_value"] < 0.05
    assert all(r > 1 for r in finding["window_ratios"].values())


def test_disparity_in_one_window_only_goes_to_the_watchlist():
    y, p, windows, groups = _windowed(["w2"])
    result = fairness_assessment(y, p, {"op": 0.5}, windows, groups)
    assert result["status"] == "pass" and result["findings"] == []
    assert any(w["group"] == "B" for w in result["watchlist"])


def _fair(ratio, finding=False):
    run = {"pooled": {"attributes": [{"attribute": "customer_segment",
                                      "groups": [{"group": "Corporate", "fpr_ratio": ratio}]}]}}
    return {"operating_points": {"op": run},
            "findings": [{"attribute": "customer_segment", "group": "Corporate"}] if finding else []}


def test_remediation_check_requires_verified_improvement():
    incumbent = _fair(1.30, finding=True)
    assert remediation_check(_fair(1.10), incumbent, ["customer_segment=Corporate"])["passed"]
    assert not remediation_check(_fair(1.35), incumbent, ["customer_segment=Corporate"])["passed"]
    assert not remediation_check(_fair(1.10, finding=True), incumbent, ["customer_segment=Corporate"])["passed"]
    assert not remediation_check(_fair(1.10), incumbent, ["customer_segment=Unknown"])["passed"]


def test_registry_notes_are_append_only_and_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("BTI_MODEL_REGISTRY_DIR", str(tmp_path))
    registry.clear_cache()
    registry.save_model("m1", {"x": 1}, {"model_id": "m1", "ownership": {"developer": "dev"},
                                         "validation": {"status": "passed"}})
    registry.add_note("m1", "reviewer", "Fairness re-assessment", "Verdict changed after the method was corrected")
    registry.add_note("m1", "reviewer", "Second note", "Another post-registration observation")
    assert [n["subject"] for n in registry.notes_for("m1")] == ["Fairness re-assessment", "Second note"]
    with pytest.raises(registry.RegistryError):
        registry.add_note("unknown", "reviewer", "x", "a long enough detail")
    with pytest.raises(registry.RegistryError):
        registry.add_note("m1", "", "x", "a long enough detail")
    registry.clear_cache()
