from pathlib import Path

import numpy as np
import pandas as pd

from paper1_qc_reviewed.qadd_v420_cohort import (
    CANONICAL_PAUSE_VIEW,
    CANONICAL_PRIMARY_VIEW,
    CANONICAL_SPEECH_VIEW,
    as_bool,
    attach_interval_provenance,
    canonical_interval_contract,
    canonical_interval_subset,
    conservative_nonincreasing_thresholds,
    observed_hum_support_grid,
    hum_null_calibration_grid,
    hum_null_window_pool,
    icc1_balanced_first_two,
    model_interface_frame,
    participant_balanced_resampling,
    repeated_recording_persistence,
    select_hum_null_reference,
)


def _interval_fixture():
    decisions = pd.DataFrame(
        {
            "logical_recording_id": ["a", "b"],
            "segmentation_analysis_eligible": [True, "True"],
        }
    )
    rows = []
    for recording_id in ["a", "b"]:
        for view in [CANONICAL_PRIMARY_VIEW, CANONICAL_SPEECH_VIEW]:
            rows.append(
                {
                    "logical_recording_id": recording_id,
                    "view": view,
                    "profile": "primary",
                    "interval_index": 0,
                    "start_sec": 0.2,
                    "end_sec": 1.0,
                    "decision": "KEEP",
                    "segmentation_analysis_eligible": True,
                }
            )
    rows.append(
        {
            "logical_recording_id": "a",
            "view": CANONICAL_PAUSE_VIEW,
            "profile": "primary",
            "interval_index": 0,
            "start_sec": 1.2,
            "end_sec": 1.8,
            "decision": "KEEP",
            "segmentation_analysis_eligible": True,
        }
    )
    return decisions, pd.DataFrame(rows)


def test_boolean_coercion_is_explicit():
    observed = as_bool(pd.Series([True, False, "True", "False", "1", "0"]))
    assert observed.tolist() == [True, False, True, False, True, False]


def test_canonical_contract_allows_pause_absence_but_not_speech_absence():
    decisions, intervals = _interval_fixture()
    tables, summary = canonical_interval_contract(decisions, intervals)
    assert summary["contract_pass"].all()
    assert set(tables) == {
        CANONICAL_PRIMARY_VIEW,
        CANONICAL_SPEECH_VIEW,
        CANONICAL_PAUSE_VIEW,
    }
    assert summary.loc[
        summary["view"].eq(CANONICAL_PAUSE_VIEW),
        "missing_eligible_recording_count",
    ].iloc[0] == 1


def test_canonical_subset_rejects_duplicate_identity():
    _, intervals = _interval_fixture()
    duplicate = pd.concat(
        [
            intervals,
            intervals.loc[
                intervals["view"].eq(CANONICAL_SPEECH_VIEW)
                & intervals["logical_recording_id"].eq("a")
            ],
        ],
        ignore_index=True,
    )
    try:
        canonical_interval_subset(duplicate, view=CANONICAL_SPEECH_VIEW)
    except ValueError as exc:
        assert "duplicated" in str(exc)
    else:
        raise AssertionError("Expected duplicate-identity failure")


def test_interval_provenance_maps_local_to_frozen_identity():
    canonical = pd.DataFrame(
        {
            "logical_recording_id": ["a", "a"],
            "interval_index": [4, 9],
            "start_sec": [1.0, 3.0],
            "end_sec": [2.0, 4.0],
        }
    )
    ledger = pd.DataFrame(
        {
            "logical_recording_id": ["a", "a"],
            "region": ["pause", "pause"],
            "interval_index": [0, 1],
        }
    )
    observed = attach_interval_provenance(
        ledger,
        region="pause",
        canonical_intervals=canonical,
        canonical_view=CANONICAL_PAUSE_VIEW,
    )
    assert observed["frozen_interval_index"].tolist() == [4, 9]
    assert observed["frozen_view"].eq(CANONICAL_PAUSE_VIEW).all()


def test_hum_null_grid_is_count_aware_and_reference_is_conservative():
    pool = hum_null_window_pool(pool_size=300, seed=18)
    grid = hum_null_calibration_grid(
        pool, support_counts=[2, 4, 8], iterations=400, seed=19
    )
    assert grid["window_count"].tolist() == [2, 4, 8]
    selected_count, threshold = select_hum_null_reference(7, grid)
    assert selected_count == 4
    assert np.isfinite(threshold)
    assert observed_hum_support_grid([0, 1, 2, 4, 4, 13]) == [2, 4, 13]


def test_hum_threshold_monotonic_adjustment_is_conservative():
    raw = np.array([3.0, 2.4, 2.6, 1.8])
    adjusted = conservative_nonincreasing_thresholds(raw)
    assert np.all(np.diff(adjusted) <= 0)
    assert np.all(adjusted >= raw)
    assert np.allclose(adjusted, [3.0, 2.6, 2.6, 1.8])


def test_icc_and_persistence_return_finite_for_consistent_repeats():
    frame = pd.DataFrame(
        {
            "SubjectID": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "recording_date_analysis": pd.date_range("2024-01-01", periods=8),
            "logical_recording_id": [f"r{i}" for i in range(8)],
            "qadd_pause_ac_level_dbfs_median": [-40, -39.8, -35, -35.1, -50, -49.8, -45, -45.2],
        }
    )
    icc = icc1_balanced_first_two(
        frame,
        subject_column="SubjectID",
        date_column="recording_date_analysis",
        feature="qadd_pause_ac_level_dbfs_median",
    )
    assert icc["subject_count"] == 4
    assert icc["icc1"] > 0.9
    persistence = repeated_recording_persistence(
        frame,
        subject_column="SubjectID",
        date_column="recording_date_analysis",
        features=["qadd_pause_ac_level_dbfs_median"],
    )
    assert persistence["paired_subject_count"].iloc[0] == 4
    assert persistence["first_second_spearman"].iloc[0] > 0.9


def test_participant_balancing_samples_one_recording_per_subject():
    frame = pd.DataFrame(
        {
            "SubjectID": ["a", "a", "a", "b", "b"],
            "qadd_pause_ac_level_dbfs_median": [-40, -30, -20, -50, -48],
        }
    )
    result = participant_balanced_resampling(
        frame,
        subject_column="SubjectID",
        features=["qadd_pause_ac_level_dbfs_median"],
        iterations=20,
        seed=20,
    )
    assert result["participant_count"].eq(2).all()
    assert result["available_participant_count"].eq(2).all()


def test_model_interface_never_imputes_missing_value():
    feature = "qadd_pause_ac_level_dbfs_median"
    frame = pd.DataFrame(
        {
            "logical_recording_id": ["a", "b"],
            "qadd_measurement_version": ["qadd-v4.2.0-candidate"] * 2,
            "qadd_signal_view": ["analysis"] * 2,
            "qadd_family_status": ["primary_available", "unavailable"],
            feature: [-40.0, np.nan],
            f"{feature}_status": ["ok_high", "insufficient_support"],
            "qadd_pause_level_support_tier": ["high", "unavailable"],
            "qadd_pause_level_iqr_db": [2.0, np.nan],
            "qadd_pause_level_iqr_db_status": ["ok_high", "insufficient_support"],
            "qadd_pause_dispersion_support_tier": ["high", "unavailable"],
            "qadd_speech_pause_level_contrast_db": [20.0, np.nan],
            "qadd_speech_pause_level_contrast_db_status": ["ok_high", "insufficient_support"],
            "qadd_speech_pause_contrast_support_tier": ["high", "unavailable"],
            "qadd_pause_spectral_flatness": [0.3, np.nan],
            "qadd_pause_spectral_flatness_status": ["ok_high", "insufficient_support"],
            "qadd_flatness_support_tier": ["high", "unavailable"],
            "qadd_mains_hum_comb_score_db": [1.0, np.nan],
            "qadd_mains_hum_comb_score_db_status": ["ok_high", "insufficient_support"],
            "qadd_hum_support_tier": ["high", "unavailable"],
        }
    )
    output = model_interface_frame(frame)
    assert np.isnan(output.loc[1, feature])
    assert not bool(output.loc[1, f"{feature}__available"])
    assert output["qadd_standalone_reject_allowed"].eq(False).all()
