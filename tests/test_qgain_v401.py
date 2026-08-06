from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper1_qc_reviewed.qgain_v401 import (  # noqa: E402
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    MEASUREMENT_VERSION,
    TimeInterval,
    ac_rms_dbfs,
    apply_gain_db,
    apply_level_envelope_db,
    extract_qgain,
    feature_registry_frame,
    guarded_speech_intervals,
    measurement_long_frame,
    model_interface_frame,
    permutation_drift_audit,
    select_canonical_speech_intervals,
    CANONICAL_SPEECH_VIEW,
    CANONICAL_SEGMENTATION_PROFILE,
)

FS = 16_000


def carrier(duration_sec: float = 18.0, amplitude: float = 0.03) -> np.ndarray:
    time = np.arange(round(duration_sec * FS)) / FS
    return amplitude * (
        np.sin(2 * np.pi * 173 * time)
        + 0.55 * np.sin(2 * np.pi * 421 * time + 0.3)
        + 0.25 * np.sin(2 * np.pi * 911 * time + 0.7)
    )


def intervals_for_duration(duration_sec: float = 18.0) -> list[TimeInterval]:
    third = duration_sec / 3
    return [
        TimeInterval(0.0, third - 0.05),
        TimeInterval(third + 0.05, 2 * third - 0.05),
        TimeInterval(2 * third + 0.05, duration_sec),
    ]


def extract(waveform: np.ndarray, intervals=None):
    return extract_qgain(
        waveform,
        FS,
        strict_speech=intervals or intervals_for_duration(len(waveform) / FS),
        logical_recording_id="test",
    )


def test_registry_is_four_feature_non_gating_contract():
    registry = feature_registry_frame()
    assert MEASUREMENT_VERSION == "qgain-v4.0.1-candidate"
    assert tuple(registry["feature"]) == ANALYSIS_FEATURES
    assert len(registry) == 4
    assert not registry["standalone_gate_allowed"].any()
    assert registry["composite_use_prohibited"].all()
    assert set(registry["quality_direction"]) == {"nonordinal"}


def test_guard_is_applied_after_merge():
    assert guarded_speech_intervals(
        [TimeInterval(0, 1), TimeInterval(0.8, 2)], 3.0
    ) == [TimeInterval(0.2, 1.8)]


def test_ac_rms_is_dc_invariant_and_marks_subfloor_nonzero_frames():
    alternating = np.tile(np.array([-0.1, 0.1]), 100)
    level, at_floor, rms = ac_rms_dbfs(alternating)
    shifted, shifted_floor, shifted_rms = ac_rms_dbfs(alternating + 0.3)
    assert level == pytest.approx(-20.0, abs=1e-12)
    assert shifted == pytest.approx(level, abs=1e-12)
    assert shifted_rms == pytest.approx(rms, abs=1e-12)
    assert not at_floor and not shifted_floor
    tiny = np.tile(np.array([-1e-8, 1e-8]), 100)
    tiny_level, tiny_floor, _ = ac_rms_dbfs(tiny, floor_db=-120.0)
    assert tiny_level == -120.0
    assert tiny_floor


def test_uniform_gain_equivariance_and_invariance():
    baseline = extract(carrier()).recording
    shifted = extract(apply_gain_db(carrier(), 6.0)).recording
    assert shifted["qgain_typical_speech_level_dbfs"] - baseline[
        "qgain_typical_speech_level_dbfs"
    ] == pytest.approx(6.0, abs=1e-10)
    for feature in ANALYSIS_FEATURES[1:]:
        assert shifted[feature] == pytest.approx(baseline[feature], abs=1e-10)


def test_polarity_invariance():
    baseline = extract(carrier()).recording
    inverted = extract(-carrier()).recording
    for feature in ANALYSIS_FEATURES:
        assert inverted[feature] == pytest.approx(baseline[feature], abs=1e-12)


def test_framewise_ac_rms_features_are_dc_offset_invariant():
    baseline = extract(carrier()).recording
    shifted = extract(carrier() + 0.25).recording
    for feature in ANALYSIS_FEATURES:
        assert shifted[feature] == pytest.approx(baseline[feature], abs=1e-12)


def test_time_shift_invariance_when_intervals_shift_with_signal():
    waveform = carrier(18.0)
    pad = np.zeros(FS)
    shifted_waveform = np.concatenate([pad, waveform])
    base = extract(waveform).recording
    shifted_intervals = [
        TimeInterval(item.start_sec + 1.0, item.end_sec + 1.0)
        for item in intervals_for_duration(18.0)
    ]
    shifted = extract(shifted_waveform, shifted_intervals).recording
    for feature in ANALYSIS_FEATURES:
        assert shifted[feature] == pytest.approx(base[feature], abs=1e-10)


def test_within_iqr_orders_amplitude_modulation():
    waveform = carrier()
    time = np.arange(len(waveform)) / FS
    values = []
    for dose in (0.0, 2.0, 4.0, 8.0):
        result = extract(
            apply_level_envelope_db(waveform, dose * np.sin(2 * np.pi * 0.7 * time))
        ).recording
        values.append(result["qgain_within_segment_iqr_db"])
    assert values == sorted(values)


def test_between_segment_mad_uses_explicit_normal_consistency_factor():
    waveform = carrier()
    envelope = np.zeros(len(waveform))
    envelope[int(6 * FS) : int(12 * FS)] = 6.0
    envelope[int(12 * FS) :] = 12.0
    result = extract(apply_level_envelope_db(waveform, envelope)).recording
    expected = DEFAULT_PARAMETERS.mad_normal_consistency_factor * 6.0
    assert result["qgain_between_segment_mad_db"] == pytest.approx(expected, abs=0.08)
    assert result["qgain_between_segment_mad_unscaled_db"] == pytest.approx(6.0, abs=0.08)


def test_drift_recovers_linear_db_ramp():
    waveform = carrier()
    time = np.arange(len(waveform)) / FS
    target = 18.0
    result = extract(
        apply_level_envelope_db(waveform, target * time / 60.0)
    ).recording
    assert result["qgain_abs_drift_db_per_min"] == pytest.approx(target, abs=0.2)
    assert result["qgain_signed_drift_db_per_min"] > 0


def test_floor_contamination_is_missing_not_zero():
    waveform = carrier()
    waveform[int(2 * FS) : int(3 * FS)] = 0.0
    result = extract(waveform).recording
    assert result["qgain_floor_contaminated"]
    for feature in ANALYSIS_FEATURES:
        assert np.isnan(result[feature])
        assert result[f"{feature}_status"] == "floor_contaminated"
        assert not result[f"{feature}_available"]


def test_short_support_is_unavailable():
    waveform = carrier(0.9)
    result = extract(waveform, [TimeInterval(0, 0.9)]).recording
    assert np.isnan(result["qgain_typical_speech_level_dbfs"])
    assert result["qgain_typical_speech_level_dbfs_status"] == "insufficient_support"


def test_amplitude_normalized_input_is_rejected():
    with pytest.raises(ValueError, match="normalization"):
        extract_qgain(
            carrier(),
            FS,
            strict_speech=intervals_for_duration(),
            signal_provenance={"amplitude_normalization_applied": True},
        )


def test_long_and_ml_exports_preserve_missingness_and_forbid_gates():
    row = extract(carrier()).recording
    long = measurement_long_frame(row)
    assert len(long) == 4
    assert long["available"].all()
    assert not long["standalone_gate_allowed"].any()
    ml = model_interface_frame(row)
    assert ml.loc[0, "qgain_decision_threshold_status"] == "not_calibrated"
    assert not bool(ml.loc[0, "qgain_standalone_reject_allowed"])


def test_drift_permutation_audit_is_reproducible():
    result = extract(carrier())
    first = permutation_drift_audit(result.segment_ledger, iterations=50, seed=7)
    second = permutation_drift_audit(result.segment_ledger, iterations=50, seed=7)
    assert first == second


def test_deterministic_ledgers_and_values():
    first = extract(carrier())
    second = extract(carrier())
    pd.testing.assert_frame_equal(first.frame_ledger, second.frame_ledger)
    pd.testing.assert_frame_equal(first.segment_ledger, second.segment_ledger)
    assert first.recording == second.recording


def test_canonical_interval_selector_never_pools_views_or_profiles():
    frame = pd.DataFrame([
        {"logical_recording_id":"r1","view":"primary_speech","profile":"primary","interval_index":0,"start_sec":0.0,"end_sec":2.0,"decision":"KEEP","segmentation_analysis_eligible":True},
        {"logical_recording_id":"r1","view":"strict_speech","profile":"permissive","interval_index":0,"start_sec":0.1,"end_sec":1.9,"decision":"KEEP","segmentation_analysis_eligible":True},
        {"logical_recording_id":"r1","view":"strict_speech","profile":"primary","interval_index":0,"start_sec":0.2,"end_sec":1.8,"decision":"KEEP","segmentation_analysis_eligible":True},
        {"logical_recording_id":"r1","view":"strict_speech","profile":"conservative","interval_index":0,"start_sec":0.3,"end_sec":1.7,"decision":"KEEP","segmentation_analysis_eligible":True},
    ])
    selected = select_canonical_speech_intervals(frame, "r1")
    assert CANONICAL_SPEECH_VIEW == "strict_speech"
    assert CANONICAL_SEGMENTATION_PROFILE == "primary"
    assert selected == [TimeInterval(0.2, 1.8)]


def test_canonical_interval_selector_rejects_duplicate_interval_identity():
    frame = pd.DataFrame([
        {"logical_recording_id":"r1","view":"strict_speech","profile":"primary","interval_index":0,"start_sec":0.2,"end_sec":1.0,"decision":"KEEP","segmentation_analysis_eligible":True},
        {"logical_recording_id":"r1","view":"strict_speech","profile":"primary","interval_index":0,"start_sec":1.2,"end_sec":2.0,"decision":"KEEP","segmentation_analysis_eligible":True},
    ])
    with pytest.raises(ValueError, match="Duplicate canonical interval_index"):
        select_canonical_speech_intervals(frame, "r1")
