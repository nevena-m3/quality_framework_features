from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper1_qc_reviewed.qrev_v400_final import (
    ACCEPTANCE_TOKEN,
    ANALYSIS_FEATURES,
    FINAL_MEASUREMENT_VERSION,
    analysis_values_equal,
    corrected_horizon_long,
    final_decisions_frame,
    final_gate_summary_frame,
    sensitivity_summary,
    ten_domain_dashboard_frame,
)


def test_acceptance_token_is_version_specific():
    assert ACCEPTANCE_TOKEN == "ACCEPT_QREV_V400"
    assert FINAL_MEASUREMENT_VERSION == "qrev-v4.0.0"


def test_final_decisions_cover_exact_feature_set():
    decisions = final_decisions_frame()
    assert tuple(decisions["feature"]) == ANALYSIS_FEATURES
    assert decisions.set_index("feature").loc[
        "qrev_tail_excess_100ms_db", "final_decision"
    ] == "RETAIN_PRIMARY_CONDITIONAL"
    assert decisions.set_index("feature").loc[
        "qrev_tail_persistence_median_sec", "final_decision"
    ] == "RETAIN_SECONDARY_CONDITIONAL_NONINDEPENDENT"
    assert decisions.set_index("feature").loc[
        "qrev_downward_decay_rate_db_per_sec", "final_decision"
    ] == "RETAIN_EXPLORATORY_CONDITIONAL"
    assert decisions.set_index("feature").loc[
        "qrev_srmr_norm", "final_decision"
    ] == "RETAIN_ESTABLISHED_COMPARATOR"


def test_decay_is_not_default_manuscript_feature():
    decisions = final_decisions_frame().set_index("feature")
    assert not bool(decisions.loc[
        "qrev_downward_decay_rate_db_per_sec", "default_manuscript_inclusion"
    ])


def test_no_scalar_or_standalone_gate_is_authorized():
    decisions = final_decisions_frame()
    assert not decisions["standalone_gate_allowed"].any()
    assert not decisions["family_scalar_allowed"].any()
    assert decisions["composite_use_prohibited"].all()


def test_dashboard_and_gate_template_are_complete():
    assert len(ten_domain_dashboard_frame()) == 10
    gates = final_gate_summary_frame()
    assert gates["gate"].tolist() == [f"G{i}" for i in range(1, 11)]
    assert gates.loc[gates["gate"].eq("G9"), "status"].item() == "N/A"


def test_shorter_horizon_is_derived_from_first_return_or_censor():
    boundary = pd.DataFrame({
        "logical_recording_id": ["a", "a", "b", "b", "c"],
        "persistence_eligible": [True, True, True, True, True],
        "tail_persistence_sec": [0.10, 0.60, 0.45, 0.55, 0.60],
    })
    recording = pd.DataFrame({
        "logical_recording_id": ["a", "b", "c"],
        "qrev_tail_persistence_median_sec": [0.35, 0.50, np.nan],
    })
    result = corrected_horizon_long(
        boundary, recording, ["a", "b", "c"], horizons_sec=(0.4,), minimum_boundary_count=2
    ).set_index("logical_recording_id")
    assert np.isclose(result.loc["a", "variant_value"], 0.25)
    assert np.isclose(result.loc["b", "variant_value"], 0.40)
    assert result.loc["b", "variant_status"] == "right_censored_at_horizon"
    assert not bool(result.loc["c", "variant_available"])


def test_corrected_horizon_summary_retains_available_recordings():
    boundary = pd.DataFrame({
        "logical_recording_id": ["a", "a", "b", "b"],
        "persistence_eligible": [True] * 4,
        "tail_persistence_sec": [0.1, 0.6, 0.2, 0.5],
    })
    recording = pd.DataFrame({
        "logical_recording_id": ["a", "b"],
        "qrev_tail_persistence_median_sec": [0.35, 0.35],
    })
    long = corrected_horizon_long(boundary, recording, ["a", "b"], horizons_sec=(0.4,))
    summary = sensitivity_summary(long).iloc[0]
    assert summary["baseline_available_n"] == 2
    assert summary["variant_available_n"] == 2
    assert summary["availability_agreement_fraction"] == 1.0


def test_analysis_equivalence_accepts_version_metadata_change():
    left = pd.DataFrame({
        "logical_recording_id": ["a", "b"],
        **{feature: [1.0, np.nan] for feature in ANALYSIS_FEATURES},
        "qrev_measurement_version": ["candidate", "candidate"],
    })
    right = left.copy()
    right["qrev_measurement_version"] = "final"
    assert analysis_values_equal(left, right)


def test_analysis_equivalence_rejects_numerical_change():
    left = pd.DataFrame({
        "logical_recording_id": ["a"],
        **{feature: [1.0] for feature in ANALYSIS_FEATURES},
    })
    right = left.copy()
    right.loc[0, "qrev_srmr_norm"] = 1.01
    assert not analysis_values_equal(left, right)


def test_support_language_does_not_claim_calibrated_precision():
    decisions = final_decisions_frame()
    combined = " ".join(decisions.astype(str).to_numpy().ravel()).lower()
    assert "precision tier" not in combined
