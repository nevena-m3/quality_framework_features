from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from paper1_qc.qgain import (
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    MEASUREMENT_VERSION,
    TimeInterval,
    apply_gain_db,
    apply_level_envelope_db,
    extract_qgain,
    feature_registry_frame,
    guarded_speech_intervals,
    poisson_rate_interval,
)

FS = 16_000


def carrier(duration_sec: float = 18.0, amplitude: float = 0.03) -> np.ndarray:
    time = np.arange(round(duration_sec * FS)) / FS
    # A deterministic multitone avoids stochastic test tolerance and provides
    # non-zero AC energy in every frame.
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


def extract(
    waveform: np.ndarray,
    intervals: list[TimeInterval] | None = None,
    *,
    parameters=DEFAULT_PARAMETERS,
):
    return extract_qgain(
        waveform,
        FS,
        strict_speech=intervals or intervals_for_duration(len(waveform) / FS),
        logical_recording_id="test",
        parameters=parameters,
    )


def test_registry_has_exactly_four_prespecified_analysis_features():
    registry = feature_registry_frame()
    assert MEASUREMENT_VERSION == "qgain-v3.1.0"
    assert tuple(registry["name"]) == ANALYSIS_FEATURES
    assert len(registry) == 4
    assert not registry["name"].str.contains(
        "crest|composite|score|step|transition"
    ).any()


def test_guard_is_applied_after_interval_merge():
    guarded = guarded_speech_intervals([TimeInterval(0, 1), TimeInterval(0.8, 2)], 3.0)
    assert guarded == [TimeInterval(0.2, 1.8)]


def test_typical_level_is_gain_equivariant_and_dynamics_are_invariant():
    waveform = carrier()
    baseline = extract(waveform).recording
    shifted = extract(apply_gain_db(waveform, 6.0)).recording
    assert shifted["qgain_typical_speech_level_dbfs"] - baseline[
        "qgain_typical_speech_level_dbfs"
    ] == pytest.approx(6.0, abs=1e-10)
    for feature in (
        "qgain_within_segment_iqr_db",
        "qgain_between_segment_mad_db",
        "qgain_abs_drift_db_per_min",
    ):
        assert shifted[feature] == pytest.approx(baseline[feature], abs=1e-10)


def test_within_segment_iqr_increases_with_amplitude_modulation():
    waveform = carrier()
    time = np.arange(len(waveform)) / FS
    unmodulated = extract(waveform).recording
    modulated = extract(
        apply_level_envelope_db(waveform, 8.0 * np.sin(2 * np.pi * 0.7 * time))
    ).recording
    assert modulated["qgain_within_segment_iqr_db"] > (
        unmodulated["qgain_within_segment_iqr_db"] + 4.0
    )


def test_between_segment_mad_recovers_segment_specific_offsets():
    waveform = carrier()
    intervals = intervals_for_duration()
    envelope = np.zeros(len(waveform))
    envelope[int(6 * FS) : int(12 * FS)] = 6.0
    envelope[int(12 * FS) :] = 12.0
    result = extract(apply_level_envelope_db(waveform, envelope), intervals).recording
    # Segment medians are separated by approximately 6 dB. MAD scale is
    # 1.4826 * 6 = 8.8956 dB.
    assert result["qgain_between_segment_mad_db"] == pytest.approx(8.8956, abs=0.08)


def test_between_segment_measure_requires_three_usable_segments():
    waveform = carrier(8.0)
    result = extract(waveform, [TimeInterval(0, 3.95), TimeInterval(4.05, 8)]).recording
    assert np.isnan(result["qgain_between_segment_mad_db"])
    assert np.isfinite(result["qgain_between_segment_mad_db_raw_estimate"])
    assert result["qgain_between_segment_mad_db_status"] == "insufficient_support"


def test_theil_sen_drift_recovers_linear_db_ramp():
    waveform = carrier()
    time = np.arange(len(waveform)) / FS
    target_db_per_min = 18.0
    ramp_db = target_db_per_min * time / 60.0
    result = extract(apply_level_envelope_db(waveform, ramp_db)).recording
    assert result["qgain_abs_drift_db_per_min"] == pytest.approx(target_db_per_min, abs=0.2)
    assert result["qgain_signed_drift_db_per_min"] > 0


def test_rejected_transition_detector_is_audit_only():
    waveform = carrier()
    no_step = extract(waveform).recording
    envelope = np.zeros(len(waveform))
    envelope[int(9 * FS) :] = 9.0
    step = extract(apply_level_envelope_db(waveform, envelope)).recording
    time = np.arange(len(waveform)) / FS
    modulation = extract(
        apply_level_envelope_db(waveform, 1.5 * np.sin(2 * np.pi * 1.1 * time))
    ).recording
    assert no_step["qgain_exploratory_local_transition_count"] == 0
    assert modulation["qgain_exploratory_local_transition_count"] == 0
    assert step["qgain_exploratory_local_transition_count"] >= 1
    assert "qgain_sustained_step_rate_per_min" not in step
    assert (
        step["qgain_exploratory_local_transition_status"]
        == "rejected_v3_0_false_positive_burden_not_analysis"
    )


def test_exploratory_transition_dose_grid_is_reproducible():
    waveform = carrier()
    counts = []
    for dose in (0.0, 3.0, 6.0, 9.0, 12.0):
        envelope = np.zeros(len(waveform))
        envelope[int(9 * FS) :] = dose
        counts.append(
            extract(apply_level_envelope_db(waveform, envelope)).recording[
                "qgain_exploratory_local_transition_count"
            ]
        )
    assert counts[:2] == [0, 0]
    assert counts[-1] >= 1
    assert counts == sorted(counts)


def test_floor_mixture_is_explicitly_censored():
    waveform = carrier()
    waveform[int(2 * FS) : int(3 * FS)] = 0.0
    result = extract(waveform).recording
    assert result["qgain_floor_censored"]
    for feature in ANALYSIS_FEATURES:
        assert np.isnan(result[feature])
        assert result[f"{feature}_status"] == "floor_censored"


def test_short_support_is_unavailable_not_zero():
    waveform = carrier(0.9)
    result = extract(waveform, [TimeInterval(0, 0.9)]).recording
    assert np.isnan(result["qgain_typical_speech_level_dbfs"])
    assert result["qgain_typical_speech_level_dbfs_status"] == "insufficient_support"


def test_poisson_interval_is_finite_and_contains_rate():
    low, high = poisson_rate_interval(3, 30.0)
    rate = 6.0
    assert 0 <= low < rate < high
    zero_low, zero_high = poisson_rate_interval(0, 30.0)
    assert zero_low == 0
    assert zero_high > 0


def test_ledgers_reconstruct_recording_estimators():
    result = extract(carrier())
    frame = result.frame_ledger.loc[result.frame_ledger["valid_level_frame"]]
    assert np.median(frame["ac_rms_dbfs"]) == pytest.approx(
        result.recording["qgain_typical_speech_level_dbfs"], abs=1e-12
    )
    segments = result.segment_ledger.loc[result.segment_ledger["usable_segment"]]
    values = segments["segment_level_median_dbfs"].to_numpy(float)
    reconstructed = 1.4826 * np.median(np.abs(values - np.median(values)))
    assert reconstructed == pytest.approx(
        result.recording["qgain_between_segment_mad_db"], abs=1e-12
    )


def test_extraction_is_deterministic():
    first = extract(carrier())
    second = extract(carrier())
    pd.testing.assert_frame_equal(first.frame_ledger, second.frame_ledger)
    pd.testing.assert_frame_equal(first.segment_ledger, second.segment_ledger)
    pd.testing.assert_frame_equal(first.event_ledger, second.event_ledger)
    assert first.recording == second.recording


def test_analysis_row_contains_no_scalar_family_score():
    result = extract(carrier()).recording
    forbidden = {"qgain_score", "qgain_composite", "qgain_burden"}
    assert forbidden.isdisjoint(result)
    assert "qgain_sustained_step_rate_per_min" not in ANALYSIS_FEATURES
    assert "qgain_sustained_step_rate_per_min" not in result


def test_sample_rate_and_mono_contracts_are_enforced():
    with pytest.raises(ValueError, match="16000"):
        extract_qgain(
            np.zeros(8_000),
            8_000,
            strict_speech=[TimeInterval(0, 1)],
        )
    with pytest.raises(ValueError, match="mono"):
        extract_qgain(
            np.zeros((FS, 2)),
            FS,
            strict_speech=[TimeInterval(0, 1)],
        )


def test_parameter_change_is_explicit_not_global_mutation():
    changed = replace(DEFAULT_PARAMETERS, step_minimum_amplitude_db=20.0)
    waveform = carrier()
    envelope = np.zeros(len(waveform))
    envelope[int(9 * FS) :] = 9.0
    result = extract(apply_level_envelope_db(waveform, envelope), parameters=changed).recording
    assert result["qgain_exploratory_local_transition_count"] == 0
