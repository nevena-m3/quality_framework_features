from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper1_qc_reviewed.qchan_v400_final import (
    ACCEPTANCE_TOKEN,
    ANALYSIS_FEATURES,
    FINAL_FEATURE_DEFINITIONS,
    analysis_values_equal,
    final_checklist_frame,
    final_decisions_frame,
    reference_bootstrap_rank_stability,
    repeated_recording_bootstrap_ci,
    standardized_figure_index,
    ten_domain_dashboard_frame,
)


def test_final_feature_roles_are_explicit_and_no_scalar() -> None:
    decisions = final_decisions_frame()
    assert decisions["feature"].tolist() == list(ANALYSIS_FEATURES)
    assert decisions["final_decision"].tolist() == [
        "RETAIN_PRIMARY_NONORDINAL",
        "RETAIN_PRIMARY_ONE_SIDED",
        "RETAIN_SECONDARY_NONINDEPENDENT",
        "RETAIN_EXPLORATORY_PHENOTYPE_SENSITIVE",
    ]
    assert not decisions["standalone_gate_allowed"].any()
    assert not decisions["family_scalar_allowed"].any()
    assert decisions["default_manuscript_inclusion"].tolist() == [
        True,
        True,
        True,
        False,
    ]


def test_dashboard_has_ten_domains_and_conditional_scope() -> None:
    dashboard = ten_domain_dashboard_frame()
    assert len(dashboard) == 10
    assert set(dashboard["status"]).issubset(
        {"PASS", "CONDITIONAL", "PASS_WITH_QUALIFICATION"}
    )
    assert dashboard.loc[
        dashboard["domain"].eq("Discriminant validity"), "status"
    ].item() == "CONDITIONAL"


def test_final_checklist_resolves_pending_and_preserves_g5_g9() -> None:
    source = pd.DataFrame(
        [
            {"item_id": "G2.3", "gate": "G2", "status": "EVIDENCE_COMPLETE_PENDING_REVIEW", "evidence": "x"},
            {"item_id": "G5.4", "gate": "G5", "status": "CONDITIONAL", "evidence": "x"},
            {"item_id": "G9.1", "gate": "G9", "status": np.nan, "evidence": "x"},
            {"item_id": "G10.1", "gate": "G10", "status": "PENDING", "evidence": "x"},
        ]
    )
    final = final_checklist_frame(source)
    assert final.set_index("item_id").at["G2.3", "status"] == "PASS"
    assert final.set_index("item_id").at["G5.4", "status"] == "CONDITIONAL"
    assert final.set_index("item_id").at["G9.1", "status"] == "N/A"
    assert final.set_index("item_id").at["G10.1", "status"] == "PASS"


def test_analysis_values_equal_is_exact_for_features_and_signed_precursors() -> None:
    columns = {
        "logical_recording_id": ["a", "b"],
        "qchan_ltas_distance_db": [1.0, 2.0],
        "qchan_rolloff95_deficit_hz": [0.0, 4.0],
        "qchan_highband_ratio_deficit": [0.0, 0.2],
        "qchan_tilt_steepening_db_per_oct": [0.1, 0.0],
        "qchan_rolloff95_signed_difference_hz": [-1.0, 4.0],
        "qchan_highband_ratio_signed_difference": [-0.1, 0.2],
        "qchan_tilt_signed_difference_db_per_oct": [0.1, -0.2],
    }
    left = pd.DataFrame(columns)
    right = left.iloc[::-1].reset_index(drop=True)
    assert analysis_values_equal(left, right)
    right.loc[right["logical_recording_id"].eq("a"), "qchan_ltas_distance_db"] += 1e-12
    assert not analysis_values_equal(left, right)


def _synthetic_repeat_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for subject in range(18):
        base = rng.normal()
        for repeat in range(2):
            row = {
                "SubjectID": f"S{subject:02d}",
                "recording_date_analysis": f"2026-01-{1 + repeat:02d}",
                "logical_recording_id": f"S{subject:02d}_{repeat}",
            }
            for index, feature in enumerate(ANALYSIS_FEATURES):
                row[feature] = base * (index + 1) + rng.normal(scale=0.15)
            rows.append(row)
    return pd.DataFrame(rows)


def test_repeated_recording_bootstrap_ci_is_participant_based_and_bounded() -> None:
    result = repeated_recording_bootstrap_ci(
        _synthetic_repeat_frame(), iterations=100, seed=11
    )
    assert len(result) == 4
    assert result["paired_subject_count"].eq(18).all()
    assert result["bootstrap_unit"].eq("participant pair").all()
    assert result["bootstrap_iterations"].eq(100).all()
    assert (
        result["spearman_bootstrap_p025"]
        <= result["first_second_spearman"]
    ).all()
    assert (
        result["first_second_spearman"]
        <= result["spearman_bootstrap_p975"]
    ).all()


def test_reference_bootstrap_rank_stability_summarizes_iterationwise_ranks() -> None:
    rows = []
    for feature in ANALYSIS_FEATURES:
        for iteration in range(5):
            for target in range(8):
                rows.append(
                    {
                        "comparison": "subject_bootstrap",
                        "feature": feature,
                        "iteration": iteration,
                        "baseline_value": float(target),
                        "variant_value": float(target) + 0.01 * iteration,
                    }
                )
    summary = reference_bootstrap_rank_stability(pd.DataFrame(rows))
    assert len(summary) == 4
    assert summary["iterations"].eq(5).all()
    assert np.allclose(summary["median_spearman_rho"], 1.0)


def test_completed_candidate_has_22_applicable_figure_bundles() -> None:
    project_root = Path(__file__).resolve().parents[1]
    candidate = (
        project_root
        / "outputs reviewed"
        / "channel_device"
        / "qchan-v4.0.0-candidate"
    )
    if not candidate.exists():
        pytest.skip("Completed QCHAN cohort candidate is not present")
    index = standardized_figure_index(candidate)
    assert (index["panel"] != "I").sum() == 22
    assert (index["panel"] == "G").sum() >= 8
    assert index.loc[index["panel"].eq("I"), "selection_reason"].item() == "no retained event detector"
