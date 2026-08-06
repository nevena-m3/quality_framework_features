
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper1_qc_reviewed.qadd_v420_final import (
    ACCEPTANCE_TOKEN,
    ANALYSIS_FEATURES,
    FINAL_MEASUREMENT_VERSION,
    combined_figure_index,
    corrected_hum_summary,
    final_decisions_frame,
    final_gate_summary_frame,
    final_registry_frame,
    ten_domain_dashboard_frame,
)

def _sample_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "measurement_version": "qadd-v4.2.0-candidate",
                "analysis_eligible": True,
                "publication_status": "candidate",
            }
            for feature in ANALYSIS_FEATURES
        ]
    )

def test_final_decisions_cover_all_features():
    decisions = final_decisions_frame()
    assert list(decisions["feature"]) == list(ANALYSIS_FEATURES)
    assert decisions["final_decision"].str.startswith("RETAIN_").all()
    assert not decisions["standalone_gate_allowed"].any()
    assert decisions["composite_use_prohibited"].all()

def test_final_registry_updates_version_and_roles():
    registry = final_registry_frame(_sample_registry())
    assert set(registry["measurement_version"]) == {FINAL_MEASUREMENT_VERSION}
    assert set(registry["publication_status"]) == {"scientifically_accepted_pending_freeze"}
    assert not registry["standalone_gate_allowed"].any()
    assert not registry["family_scalar_allowed"].any()
    assert registry["composite_use_prohibited"].all()

def test_ten_domain_dashboard_has_exact_domains_and_conditional_labels():
    dashboard = ten_domain_dashboard_frame()
    assert len(dashboard) == 10
    assert set(dashboard["status"]).issubset({"PASS", "CONDITIONAL"})
    assert set(dashboard.loc[dashboard["status"] == "CONDITIONAL", "domain"]) == {
        "Discriminant validity",
        "Reliability and robustness",
    }

def test_gate_summary_marks_g9_explicit_na_and_g10_pass():
    gates = final_gate_summary_frame()
    assert gates.loc[gates["gate"] == "G9", "status"].item() == "N/A"
    assert gates.loc[gates["gate"] == "G10", "status"].item() == "PASS"
    assert not gates["status"].isin(["FAIL", "PENDING"]).any()

def test_corrected_hum_summary_counts_only_eligible_winners():
    table = pd.DataFrame(
        {
            "qadd_mains_hum_null_calibration_status": [
                "applied_exact_count",
                "applied_exact_count",
                "not_applicable_insufficient_support",
                "applied_exact_count",
            ],
            "qadd_mains_hum_joint_evidence_above_null": [True, False, False, False],
            "qadd_mains_hum_winner_hz": [50.0, 60.0, 60.0, 60.0],
        }
    )
    summary = corrected_hum_summary(table).iloc[0]
    assert summary["eligible_recordings"] == 3
    assert summary["winner_50_count_eligible"] == 1
    assert summary["winner_60_count_eligible"] == 2
    assert summary["winner_count_sum_check"] == 3
    assert bool(summary["winner_counts_match_eligible"])

def test_acceptance_token_is_version_specific():
    assert ACCEPTANCE_TOKEN == "ACCEPT_QADD_V420"

def test_figure_index_requires_complete_artifact_bundle(tmp_path: Path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "galleries").mkdir()
    stems = [
        "A_construct_response",
        "B_hum_discriminant_specificity",
        "C_transformation_contract",
        "D_support_and_availability",
        "E_support_boundary_sensitivity",
        "F_empirical_distributions",
        "H_reliability_redundancy_weighting",
        "J_ml_handoff_contract",
    ]
    for stem in stems:
        base = tmp_path / "figures" / stem
        for suffix in (".png", ".svg", ".pdf", ".source.csv", ".caption.md", ".provenance.json"):
            base.with_suffix(suffix).write_text("x", encoding="utf-8")
    gallery_id = "example"
    base = tmp_path / "galleries" / f"G_signal_example_{gallery_id}"
    for suffix in (".png", ".svg", ".pdf", ".source.csv", ".caption.md", ".provenance.json"):
        base.with_suffix(suffix).write_text("x", encoding="utf-8")
    pd.DataFrame(
        [{"logical_recording_id": gallery_id, "selection_reason": "example"}]
    ).to_csv(tmp_path / "galleries" / "qadd_v420_gallery_index.csv", index=False)
    index = combined_figure_index(tmp_path)
    assert set(index["panel"]) == set("ABCDEFGHJI")
    assert (index["panel"] == "G").sum() == 1
    assert index.loc[index["panel"] == "I", "purpose"].item() == "no retained event detector"

def test_feature_roles_preserve_nonordinal_and_mixed_claims():
    decisions = final_decisions_frame().set_index("feature")
    assert decisions.loc["qadd_pause_spectral_flatness", "final_decision"] == "RETAIN_SECONDARY_NONORDINAL"
    assert decisions.loc["qadd_speech_pause_level_contrast_db", "final_decision"] == "RETAIN_SECONDARY_MIXED_NONINDEPENDENT"
    assert decisions.loc["qadd_mains_hum_comb_score_db", "final_decision"] == "RETAIN_TARGETED_CONDITIONAL"

def test_no_family_scalar_or_standalone_gate_is_approved():
    decisions = final_decisions_frame()
    assert not decisions["standalone_gate_allowed"].any()
    assert decisions["composite_use_prohibited"].all()
