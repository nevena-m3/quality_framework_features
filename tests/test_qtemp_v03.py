from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
import pandas as pd
import pytest

from paper1_qc.qtemp import (
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    MEASUREMENT_VERSION,
    TimeInterval,
    apply_gain_step,
    extract_qtemp,
    feature_registry_frame,
    hard_clip,
    inject_consecutive_duplicate,
    inject_dropout,
    inject_impulse,
    inject_splice_delete,
    inject_splice_replace,
    match_events_to_truth,
    match_events_to_truth_points,
    poisson_rate_interval,
    reconstruct_recording_features,
)

FS = 16_000


def carrier(duration_sec: float = 8.0, amplitude: float = 0.04) -> np.ndarray:
    time_axis = np.arange(int(duration_sec * FS)) / FS
    envelope = 0.8 + 0.15 * np.sin(2 * np.pi * 0.7 * time_axis)
    return amplitude * envelope * (
        np.sin(2 * np.pi * 173 * time_axis + 0.1)
        + 0.55 * np.sin(2 * np.pi * 421 * time_axis + 0.7)
        + 0.25 * np.sin(2 * np.pi * 911 * time_axis + 1.4)
    )


def stochastic_carrier(duration_sec: float = 8.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    excitation = rng.normal(0, 1, int(duration_sec * FS))
    waveform = np.zeros_like(excitation)
    for index in range(3, len(waveform)):
        waveform[index] = (
            0.78 * waveform[index - 1]
            - 0.31 * waveform[index - 2]
            + 0.12 * waveform[index - 3]
            + 0.08 * excitation[index]
        )
    waveform /= max(np.max(np.abs(waveform)), 1e-12)
    return 0.08 * waveform


def test_registry_contract_is_exactly_five_features():
    registry = feature_registry_frame()
    assert MEASUREMENT_VERSION == "qtemp-v0.3.0-measurement-development"
    assert tuple(registry["name"]) == ANALYSIS_FEATURES
    assert not registry["name"].str.contains("score|composite|burden_index").any()
    assert registry["signal_view"].str.contains("native decoded").all()


def test_measured_zero_and_unavailable_are_distinct():
    clean = extract_qtemp(carrier(), FS, logical_recording_id="clean")
    missing = extract_qtemp(
        carrier(), FS, logical_recording_id="missing", native_source_confirmed=False
    )
    assert clean.recording["qtemp_status"] == "measured"
    assert all(clean.recording[name] == 0 for name in ANALYSIS_FEATURES)
    assert missing.recording["qtemp_status"] == "unavailable_native_source"
    assert all(np.isnan(missing.recording[name]) for name in ANALYSIS_FEATURES)


def test_event_ledgers_have_separate_semantics():
    perturbed = inject_dropout(carrier(), FS, 3.0, 40.0, mode="zero")
    result = extract_qtemp(perturbed, FS, logical_recording_id="ledger")
    assert "initial_disposition" in result.candidate_ledger
    assert "disposition" in result.disposition_ledger
    assert set(result.event_ledger["disposition"]) == {"accepted"}
    assert len(result.event_ledger) <= len(result.disposition_ledger)


@pytest.mark.parametrize("duration_ms", [10.0, 20.0, 40.0, 80.0, 160.0])
def test_bracketed_exact_zero_dropout_is_recovered(duration_ms: float):
    perturbed = inject_dropout(carrier(), FS, 3.0, duration_ms, mode="zero")
    result = extract_qtemp(
        perturbed,
        FS,
        logical_recording_id="dropout",
        enabled_event_types=("dropout",),
    )
    match = match_events_to_truth(
        result.event_ledger,
        event_type="dropout",
        truth_start_sec=3.0,
        truth_end_sec=3.0 + duration_ms / 1000.0,
        tolerance_ms=2.0,
    )
    assert match["detected"]
    assert abs(match["start_error_ms"]) <= 1.0
    assert abs(match["end_error_ms"]) <= 1.0


def test_constant_low_information_dropout_is_recovered():
    perturbed = inject_dropout(
        carrier(), FS, 3.0, 50.0, mode="constant", constant_value=1e-5
    )
    result = extract_qtemp(
        perturbed,
        FS,
        logical_recording_id="constant",
        enabled_event_types=("dropout",),
    )
    accepted = result.event_ledger.query("event_type == 'dropout'")
    assert len(accepted) == 1
    assert accepted.iloc[0]["duration_sec"] >= 0.045


def test_edge_zero_and_attenuation_are_not_accepted_as_dropouts():
    edge = carrier()
    edge[: int(0.5 * FS)] = 0.0
    attenuated = inject_dropout(
        carrier(), FS, 3.0, 40.0, mode="attenuated", attenuation_db=-12
    )
    assert extract_qtemp(
        edge, FS, enabled_event_types=("dropout",)
    ).recording["qtemp_dropout_accepted_event_count"] == 0
    assert extract_qtemp(
        attenuated, FS, enabled_event_types=("dropout",)
    ).recording["qtemp_dropout_accepted_event_count"] == 0


def test_native_channels_are_not_averaged_before_detection():
    stereo = np.column_stack([carrier(), carrier()])
    stereo = inject_dropout(stereo, FS, 3.0, 40.0, mode="zero", channel=1)
    result = extract_qtemp(
        stereo,
        FS,
        logical_recording_id="stereo",
        enabled_event_types=("dropout",),
    )
    accepted_channels = result.disposition_ledger.query(
        "event_type == 'dropout' and disposition == 'accepted'"
    )["channel_index"].unique()
    assert accepted_channels.tolist() == [1]
    assert result.recording["qtemp_dropout_accepted_event_count"] == 1
    assert result.event_ledger.iloc[0]["channels_detected"] == "1"


@pytest.mark.parametrize("duration_ms", [20.0, 40.0, 80.0, 160.0])
def test_exact_consecutive_duplicate_is_detected_at_non_grid_alignment(duration_ms: float):
    perturbed = inject_consecutive_duplicate(carrier(), FS, 2.013, duration_ms)
    result = extract_qtemp(
        perturbed,
        FS,
        logical_recording_id="duplicate",
        enabled_event_types=("frozen_audio",),
    )
    accepted = result.event_ledger.query("event_type == 'frozen_audio'")
    assert len(accepted) == 1
    assert accepted.iloc[0]["duration_sec"] >= (duration_ms - 3.0) / 1000.0


def test_small_near_exact_duplicate_is_detected_but_large_perturbation_is_not():
    base = carrier()
    small = inject_consecutive_duplicate(
        base, FS, 2.013, 80.0, perturbation_sd=2e-5, random_seed=1
    )
    large = inject_consecutive_duplicate(
        base, FS, 2.013, 80.0, perturbation_sd=2e-3, random_seed=1
    )
    small_result = extract_qtemp(
        small, FS, enabled_event_types=("frozen_audio",)
    )
    large_result = extract_qtemp(
        large, FS, enabled_event_types=("frozen_audio",)
    )
    assert small_result.recording["qtemp_frozen_audio_accepted_event_count"] >= 1
    assert large_result.recording["qtemp_frozen_audio_accepted_event_count"] == 0


def test_stationary_tone_and_harmonic_carrier_are_not_accepted_as_duplicate():
    time_axis = np.arange(8 * FS) / FS
    tone = 0.05 * np.sin(2 * np.pi * 200 * time_axis)
    for waveform in [tone, carrier()]:
        result = extract_qtemp(
            waveform,
            FS,
            logical_recording_id="periodic",
            enabled_event_types=("frozen_audio",),
        )
        assert result.recording["qtemp_frozen_audio_accepted_event_count"] == 0


def test_abrupt_source_replacement_recovers_both_join_boundaries():
    waveform = carrier()
    donor = stochastic_carrier(seed=99)
    length = int(round(0.060 * FS))
    target = int(round(3.0 * FS))
    replacement = donor[int(4.0 * FS): int(4.0 * FS) + length].copy()
    replacement *= np.sqrt(np.mean(waveform[target:target + length] ** 2)) / max(
        np.sqrt(np.mean(replacement**2)), 1e-12
    )
    spliced, boundaries = inject_splice_replace(
        waveform, FS, 3.0, replacement
    )
    result = extract_qtemp(
        spliced,
        FS,
        logical_recording_id="source-switch",
        enabled_event_types=("splice",),
    )
    match = match_events_to_truth_points(
        result.event_ledger,
        event_type="splice",
        truth_times_sec=boundaries,
        tolerance_ms=10.0,
    )
    assert match["boundary_recall"] == pytest.approx(1.0)
    assert match["maximum_abs_error_ms"] <= 10.0


def test_smooth_deletion_is_characterized_without_being_forced_positive():
    waveform = stochastic_carrier()
    spliced, join_sec = inject_splice_delete(waveform, FS, 3.0, 40.0)
    result = extract_qtemp(
        spliced,
        FS,
        logical_recording_id="smooth-delete",
        enabled_event_types=("splice",),
    )
    # No-reference smooth joins may be unidentifiable. The contract requires a
    # valid measured result, not artificial detection.
    assert result.recording["qtemp_status"] == "measured"
    assert np.isfinite(result.recording["qtemp_splice_discontinuity_rate_per_min"])
    assert join_sec == pytest.approx(3.0)


def test_gain_step_impulse_and_clean_signal_are_not_accepted_as_splice():
    controls = [
        stochastic_carrier(),
        apply_gain_step(stochastic_carrier(), FS, 3.0, 9.0),
        inject_impulse(stochastic_carrier(), FS, 3.0, 0.8),
    ]
    for waveform in controls:
        result = extract_qtemp(
            waveform, FS, enabled_event_types=("splice",)
        )
        assert result.recording["qtemp_splice_accepted_event_count"] == 0


def test_clipping_guard_blocks_splice_candidate():
    waveform = stochastic_carrier()
    clipped = hard_clip(waveform * 10, 0.25)
    result = extract_qtemp(
        clipped,
        FS,
        logical_recording_id="clipping",
        clipping_event_intervals=[TimeInterval(2.5, 3.5)],
        enabled_event_types=("splice",),
    )
    accepted = result.event_ledger.query("event_type == 'splice'") if len(result.event_ledger) else result.event_ledger
    assert not (
        (accepted["start_sec"] >= 2.5) & (accepted["start_sec"] <= 3.5)
    ).any() if len(accepted) else True


def test_splice_speech_boundary_guard_is_applied():
    waveform = carrier()
    donor = stochastic_carrier(seed=99)
    replacement = donor[int(4.0 * FS): int(4.06 * FS)].copy()
    replacement *= np.sqrt(np.mean(waveform[int(3.0 * FS):int(3.06 * FS)] ** 2)) / max(
        np.sqrt(np.mean(replacement**2)), 1e-12
    )
    spliced, boundaries = inject_splice_replace(waveform, FS, 3.0, replacement)
    result = extract_qtemp(
        spliced,
        FS,
        logical_recording_id="boundary",
        speech_intervals=[
            TimeInterval(1.0, boundaries[0]),
            TimeInterval(boundaries[0], 7.0),
        ],
        enabled_event_types=("splice",),
    )
    accepted = result.event_ledger.query("event_type == 'splice'") if len(result.event_ledger) else result.event_ledger
    near = accepted.loc[np.abs(accepted["start_sec"] - boundaries[0]) <= 0.010] if len(accepted) else accepted
    assert len(near) == 0


def test_splice_at_dropout_boundaries_is_arbitrated():
    perturbed = inject_dropout(stochastic_carrier(), FS, 3.0, 40.0, mode="zero")
    parameters = replace(
        DEFAULT_PARAMETERS,
        splice_derivative_z_accept=5.0,
        splice_prediction_z_accept=2.5,
        splice_derivative_z_indeterminate=4.0,
        splice_prediction_z_indeterminate=2.0,
    )
    result = extract_qtemp(
        perturbed, FS, logical_recording_id="arbitration", parameters=parameters
    )
    splice = result.disposition_ledger.query("event_type == 'splice'")
    near = splice.loc[(splice["start_sec"] >= 2.990) & (splice["start_sec"] <= 3.050)]
    assert len(near) == 0 or not near["disposition"].eq("accepted").any()


def test_recording_features_reconstruct_exactly_from_accepted_event_ledger():
    waveform = inject_dropout(carrier(), FS, 3.0, 40.0, mode="zero")
    result = extract_qtemp(waveform, FS, logical_recording_id="reconstruct")
    reconstructed = reconstruct_recording_features(
        result.event_ledger, result.recording["qtemp_eligible_duration_sec"]
    )
    for feature in ANALYSIS_FEATURES:
        assert abs(result.recording[feature] - reconstructed[feature]) < 1e-12


def test_rate_interval_contains_point_estimate():
    low, high = poisson_rate_interval(3, 30.0)
    point = 3 * 60 / 30.0
    assert low <= point <= high


def test_extraction_is_deterministic():
    waveform = inject_consecutive_duplicate(
        inject_dropout(carrier(), FS, 3.0, 40.0, mode="zero"),
        FS,
        2.013,
        80.0,
    )
    first = extract_qtemp(waveform, FS, logical_recording_id="deterministic")
    second = extract_qtemp(waveform, FS, logical_recording_id="deterministic")
    assert first.recording == second.recording
    pd.testing.assert_frame_equal(first.candidate_ledger, second.candidate_ledger)
    pd.testing.assert_frame_equal(first.disposition_ledger, second.disposition_ledger)
    pd.testing.assert_frame_equal(first.event_ledger, second.event_ledger)


def test_detector_specific_execution_matches_full_run_for_isolated_dropout():
    waveform = stochastic_carrier()
    injected = inject_dropout(waveform, FS, 3.0, 40.0, mode="zero")
    full = extract_qtemp(injected, FS, logical_recording_id="full")
    dropout_only = extract_qtemp(
        injected,
        FS,
        logical_recording_id="dropout-only",
        enabled_event_types=("dropout",),
    )
    assert full.recording["qtemp_dropout_duration_fraction"] == pytest.approx(
        dropout_only.recording["qtemp_dropout_duration_fraction"]
    )
    assert dropout_only.recording["qtemp_frozen_audio_accepted_event_count"] == 0
    assert dropout_only.recording["qtemp_splice_accepted_event_count"] == 0
    assert dropout_only.recording["qtemp_enabled_event_types"] == "dropout"


def test_all_detector_runtime_is_bounded_on_short_signal():
    waveform = inject_consecutive_duplicate(
        inject_dropout(carrier(), FS, 3.0, 40.0, mode="zero"),
        FS,
        2.013,
        80.0,
    )
    started = time.perf_counter()
    extract_qtemp(waveform, FS, logical_recording_id="runtime")
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0
