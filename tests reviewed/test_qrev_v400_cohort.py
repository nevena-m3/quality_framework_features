from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper1_qc_reviewed.qrev_v400 import SpeechInterval
from paper1_qc_reviewed.qrev_v400_cohort import (
    ANALYSIS_FEATURES,
    analysis_waveform_from_audio_views,
    as_bool,
    canonical_interval_contract,
    canonical_interval_subset,
    delete_one_boundary_grid,
    summarize_delete_one,
    deterministic_stratified_sample,
    empirical_feature_summary,
    intervals_for_recording,
    model_interface_frame,
    participant_balanced_resampling,
    participant_balanced_summary,
    policy_values,
    repeated_recording_persistence,
    resolve_media_path,
    shift_primary_offsets,
    support_policy_availability,
)


def _decisions():
    return pd.DataFrame(
        {
            "logical_recording_id": ["r1", "r2"],
            "segmentation_analysis_eligible": [True, True],
        }
    )


def _intervals():
    rows = []
    for recording_id in ["r1", "r2"]:
        for index, (start, end) in enumerate([(0.2, 1.0), (1.5, 2.4), (3.0, 3.8)]):
            rows.append(
                {
                    "logical_recording_id": recording_id,
                    "view": "primary_speech",
                    "profile": "primary",
                    "interval_index": index,
                    "start_sec": start,
                    "end_sec": end,
                    "decision": "KEEP",
                    "segmentation_analysis_eligible": True,
                }
            )
            rows.append(
                {
                    "logical_recording_id": recording_id,
                    "view": "strict_speech",
                    "profile": "primary",
                    "interval_index": index,
                    "start_sec": start + 0.05,
                    "end_sec": end - 0.05,
                    "decision": "KEEP",
                    "segmentation_analysis_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def _recording_table():
    return pd.DataFrame(
        {
            "logical_recording_id": ["r1", "r2", "r3", "r4"],
            "SubjectID": ["s1", "s1", "s2", "s2"],
            "recording_date_analysis": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
            "qrev_tail_excess_100ms_db": [2.0, 2.5, 4.0, 4.5],
            "qrev_tail_persistence_median_sec": [0.2, 0.3, 0.6, 0.6],
            "qrev_downward_decay_rate_db_per_sec": [10.0, 11.0, np.nan, 9.0],
            "qrev_srmr_norm": [4.0, 3.8, 3.0, 3.2],
            "qrev_tail_excess_100ms_db_raw_estimate": [2.0, 2.5, 4.0, 4.5],
            "qrev_tail_persistence_median_sec_raw_estimate": [0.2, 0.3, 0.6, 0.6],
            "qrev_downward_decay_rate_db_per_sec_raw_estimate": [10.0, 11.0, np.nan, 9.0],
            "qrev_tail_valid_boundary_count": [2, 3, 4, 5],
            "qrev_persistence_valid_boundary_count": [2, 3, 4, 5],
            "qrev_decay_valid_boundary_count": [2, 3, 1, 5],
            "qrev_tail_valid_pause_support_sec": [2.0, 3.0, 4.0, 5.0],
            "qrev_persistence_valid_pause_support_sec": [2.0, 3.0, 4.0, 5.0],
            "qrev_decay_valid_pause_support_sec": [2.0, 3.0, 1.0, 5.0],
            "qrev_persistence_recording_median_censored": [False, False, True, True],
            "qrev_persistence_right_censored_fraction": [0.0, 0.0, 1.0, 0.8],
            "qrev_internal_boundary_count": [2, 3, 4, 5],
            "qrev_srmr_primary_task_span_sec": [5.0] * 4,
            "qrev_srmr_strict_speech_support_sec": [4.0] * 4,
            "qrev_srmr_estimated_working_set_mb": [10.0] * 4,
            "qrev_measurement_version": ["qrev-v4.0.0-candidate"] * 4,
            "qrev_signal_view": ["mono_globally_dc_removed_16k_analysis"] * 4,
            "qrev_boundary_source_view": ["primary_speech"] * 4,
            "qrev_boundary_source_profile": ["primary"] * 4,
            "qrev_srmr_variant": ["pinned"] * 4,
            "qrev_srmr_upstream_commit": ["commit"] * 4,
            "qrev_family_status": ["all_primary_available"] * 4,
            "qrev_tail_excess_100ms_db_status": ["measured"] * 4,
            "qrev_tail_persistence_median_sec_status": ["measured", "measured", "right_censored_at_horizon", "right_censored_at_horizon"],
            "qrev_downward_decay_rate_db_per_sec_status": ["measured", "measured", "no_valid_downward_decay", "measured"],
            "qrev_srmr_norm_status": ["measured"] * 4,
            "qrev_tail_excess_100ms_db_support_tier": ["minimum", "minimum", "moderate", "moderate"],
            "qrev_tail_persistence_median_sec_support_tier": ["minimum", "minimum", "moderate", "moderate"],
            "qrev_downward_decay_rate_db_per_sec_support_tier": ["minimum", "minimum", "unavailable", "moderate"],
            "qrev_srmr_norm_support_tier": ["minimum"] * 4,
        }
    )


def test_as_bool_handles_strings_and_native_values():
    result = as_bool(pd.Series([True, False, "yes", "0", "TRUE", "n"]))
    assert result.tolist() == [True, False, True, False, True, False]


def test_canonical_contract_pairs_primary_and_strict():
    tables, summary, pair = canonical_interval_contract(_decisions(), _intervals())
    assert summary["contract_pass"].all()
    assert len(tables["primary_speech"]) == 6
    assert len(tables["strict_speech"]) == 6
    assert pair["pair_complete"].all()
    assert pair["strict_inside_primary"].all()
    assert np.allclose(pair["start_erosion_sec"], 0.05)
    assert np.allclose(pair["end_erosion_sec"], 0.05)


def test_canonical_subset_rejects_overlapping_intervals():
    intervals = _intervals()
    mask = (
        intervals["logical_recording_id"].eq("r1")
        & intervals["view"].eq("primary_speech")
        & intervals["interval_index"].eq(1)
    )
    intervals.loc[mask, "start_sec"] = 0.9
    with pytest.raises(ValueError, match="overlap"):
        canonical_interval_subset(intervals, view="primary_speech")


def test_intervals_for_recording_preserves_frozen_identity():
    primary = canonical_interval_subset(_intervals(), view="primary_speech")
    intervals, table = intervals_for_recording(primary, "r1")
    assert len(intervals) == 3
    assert intervals[0].interval_id == "r1:primary_speech:primary:00000"
    assert intervals[0].view == "primary_speech"
    assert table["frozen_interval_id"].is_unique


def test_shift_primary_offsets_preserves_ids_and_gap():
    intervals = [
        SpeechInterval(0.0, 1.0, "a", 0),
        SpeechInterval(1.2, 2.0, "b", 1),
    ]
    shifted = shift_primary_offsets(intervals, 500.0, separation_sec=0.03)
    assert shifted[0].interval_id == "a"
    assert shifted[0].end_sec <= shifted[1].start_sec - 0.03 + 1e-12
    earlier = shift_primary_offsets(intervals, -100.0)
    assert np.isclose(earlier[0].end_sec, 0.9)


def test_policy_values_preserve_raw_estimates_and_apply_minimum_count():
    table = _recording_table()
    p4 = policy_values(table, minimum_boundary_count=4)
    assert p4["qrev_tail_excess_100ms_db__available"].tolist() == [False, False, True, True]
    assert np.isnan(p4.loc[0, "qrev_tail_excess_100ms_db"])
    assert p4.loc[2, "qrev_tail_excess_100ms_db"] == 4.0
    assert p4["qrev_srmr_norm__available"].all()


def test_support_policy_availability_is_monotone():
    summary = support_policy_availability(_recording_table())
    tail = summary.loc[summary["feature"].eq("qrev_tail_excess_100ms_db")]
    assert tail.sort_values("minimum_boundary_count")["available_n"].tolist() == [4, 3, 2]


def test_delete_one_grid_tracks_value_and_availability_changes():
    ledger = pd.DataFrame(
        {
            "logical_recording_id": ["r1"] * 3,
            "tail_eligible": [True] * 3,
            "tail_excess_100ms_db": [1.0, 2.0, 7.0],
            "persistence_eligible": [True] * 3,
            "tail_persistence_sec": [0.1, 0.2, 0.6],
            "decay_eligible": [True] * 3,
            "downward_decay_rate_db_per_sec": [5.0, 10.0, 15.0],
        }
    )
    grid = delete_one_boundary_grid(ledger, policies=[2, 3])
    assert len(grid) == 18
    assert grid.loc[grid["minimum_boundary_count"].eq(3), "availability_retained"].eq(False).all()
    summary = summarize_delete_one(grid)
    assert set(summary["minimum_boundary_count"]) == {2, 3}


def test_deterministic_stratified_sample_is_repeatable():
    frame = pd.DataFrame(
        {
            "logical_recording_id": [f"r{i}" for i in range(20)],
            "support": [i % 3 for i in range(20)],
        }
    )
    one = deterministic_stratified_sample(frame, maximum_rows=8, stratum_columns=["support"])
    two = deterministic_stratified_sample(frame, maximum_rows=8, stratum_columns=["support"])
    assert one["logical_recording_id"].tolist() == two["logical_recording_id"].tolist()
    assert len(one) == 8


def test_repeated_recording_persistence_separates_censored_subset():
    result = repeated_recording_persistence(
        _recording_table(),
        subject_column="SubjectID",
        date_column="recording_date_analysis",
    )
    assert set(result["feature"]) == set(ANALYSIS_FEATURES)
    persistence = result.loc[result["feature"].eq("qrev_tail_persistence_median_sec")].iloc[0]
    assert persistence["paired_subject_count"] == 2
    assert persistence["paired_uncensored_subject_count"] == 1


def test_participant_balancing_and_summary_are_deterministic():
    one = participant_balanced_resampling(
        _recording_table(), subject_column="SubjectID", iterations=20, seed=17
    )
    two = participant_balanced_resampling(
        _recording_table(), subject_column="SubjectID", iterations=20, seed=17
    )
    pd.testing.assert_frame_equal(one, two)
    summary = participant_balanced_summary(one)
    assert set(summary["feature"]) == set(ANALYSIS_FEATURES)
    assert (summary["iterations"] == 20).all()


def test_empirical_summary_and_model_interface_keep_missingness_explicit():
    table = _recording_table()
    summary = empirical_feature_summary(table)
    decay = summary.loc[summary["feature"].eq("qrev_downward_decay_rate_db_per_sec")].iloc[0]
    assert decay["available_n"] == 3
    interface = model_interface_frame(table)
    assert not interface.loc[2, "qrev_downward_decay_rate_db_per_sec__available"]
    assert interface.loc[2, "qrev_downward_decay_rate_db_per_sec__missing_reason"] == "no_valid_downward_decay"
    assert not interface["qrev_family_scalar_available"].any()
    assert not interface["qrev_standalone_reject_allowed"].any()


def test_resolve_media_path_uses_override(tmp_path: Path):
    target = tmp_path / "s1" / "audio.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abc")
    raw = r"C:\old\Bamboo_passage_only\s1\audio.wav"
    assert resolve_media_path(raw, media_root_override=tmp_path) == target


def test_analysis_waveform_from_audio_views_uses_canonical_analysis_16k():
    class DummyViews:
        analysis_16k = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        analysis = np.array([99.0], dtype=np.float32)

    waveform = analysis_waveform_from_audio_views(DummyViews())
    assert waveform.dtype == np.float64
    assert waveform.ndim == 1
    assert np.allclose(waveform, [0.25, -0.5, 0.75])


def test_analysis_waveform_from_audio_views_rejects_noncanonical_alias():
    class LegacyAliasOnly:
        analysis = np.array([0.0, 1.0], dtype=np.float32)

    with pytest.raises(AttributeError, match="analysis_16k"):
        analysis_waveform_from_audio_views(LegacyAliasOnly())
