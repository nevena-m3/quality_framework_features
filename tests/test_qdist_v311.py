from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from paper1_qc.qdist import (
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    MEASUREMENT_VERSION,
    NativeSignalProvenance,
    QDISTParameters,
    TimeInterval,
    apply_hard_clip,
    apply_soft_clip,
    extract_qdist,
    feature_registry_frame,
    poisson_rate_interval,
    quantize_pcm,
    reconstruct_qdist_features,
)


def speech_like_carrier(
    fs: int = 16_000,
    duration_sec: float = 5.0,
    seed: int = 31,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(round(fs * duration_sec)) / fs
    fundamental = 135.0 + 18.0 * np.sin(2 * np.pi * 0.37 * time)
    phase = 2 * np.pi * np.cumsum(fundamental) / fs
    source = (
        0.62 * np.sin(phase)
        + 0.24 * np.sin(2 * phase + 0.2)
        + 0.12 * np.sin(3 * phase - 0.4)
    )
    envelope = 0.40 + 0.55 * np.square(np.sin(2 * np.pi * 0.72 * time))
    noise = 0.004 * rng.standard_normal(len(time))
    waveform = envelope * source + noise
    waveform /= max(np.max(np.abs(waveform)), 1e-12)
    return 0.86 * waveform


def hard_clip_bursts(
    waveform: np.ndarray,
    fs: int,
    bursts: list[tuple[float, float]],
    limit: float = 0.46,
) -> np.ndarray:
    output = np.asarray(waveform, dtype=np.float64).copy()
    for start_sec, end_sec in bursts:
        start = round(start_sec * fs)
        end = round(end_sec * fs)
        output[start:end] = apply_hard_clip(output[start:end], limit)
    return output


def pcm16_provenance(**kwargs) -> NativeSignalProvenance:
    return NativeSignalProvenance(
        codec_name="pcm_s16le",
        sample_format="s16",
        bits_per_raw_sample=16,
        **kwargs,
    )


def test_registry_is_exactly_three_hard_clipping_views():
    registry = feature_registry_frame()
    assert MEASUREMENT_VERSION == "qdist-v3.1.1"
    assert tuple(registry["name"]) == ANALYSIS_FEATURES
    assert len(registry) == 3
    assert not registry["name"].str.contains(
        "thd|soft|compression|agc|perceptual|composite|score",
        case=False,
        regex=True,
    ).any()
    assert set(registry["source_ledger"]) == {
        "accepted plateau ledger",
        "merged episode ledger",
    }


def test_clean_speech_like_carrier_is_valid_zero():
    fs = 16_000
    extraction = extract_qdist(
        speech_like_carrier(fs),
        fs,
        provenance=pcm16_provenance(),
    )
    assert extraction.recording["qdist_status"] == "available_no_events"
    assert extraction.recording["qdist_available"] is True
    assert extraction.recording["qdist_hard_clip_event_count"] == 0
    for feature in ANALYSIS_FEATURES:
        assert extraction.recording[feature] == 0.0
        assert extraction.recording[f"{feature}_status"] == "available_no_events"
    assert extraction.accepted_plateau_ledger.empty
    assert extraction.episode_ledger.empty


def test_hard_clipping_burst_is_detected_and_ledgers_are_populated():
    fs = 16_000
    clean = speech_like_carrier(fs)
    clipped = hard_clip_bursts(clean, fs, [(1.0, 1.7)], limit=0.42)
    extraction = extract_qdist(
        clipped,
        fs,
        logical_recording_id="clip",
        provenance=pcm16_provenance(),
    )
    row = extraction.recording
    assert row["qdist_status"] == "available_events"
    assert row["qdist_accepted_plateau_count"] > 0
    assert row["qdist_hard_clip_event_count"] >= 1
    assert row["qdist_hard_clipped_frame_fraction"] > 0
    assert row["qdist_hard_clip_event_rate_per_min"] > 0
    assert row["qdist_hard_clipped_sample_fraction"] > 0
    assert extraction.candidate_ledger["accepted"].any()
    assert set(extraction.accepted_plateau_ledger["rejection_reason"]) == {"accepted"}
    assert extraction.episode_ledger["constituent_candidate_ids"].str.len().gt(0).all()


def test_two_clipping_bursts_separated_beyond_merge_gap_make_two_episodes():
    fs = 16_000
    clean = speech_like_carrier(fs, duration_sec=6.0)
    clipped = hard_clip_bursts(
        clean,
        fs,
        [(1.0, 1.35), (3.0, 3.35)],
        limit=0.40,
    )
    extraction = extract_qdist(
        clipped,
        fs,
        provenance=pcm16_provenance(),
    )
    assert extraction.recording["qdist_hard_clip_event_count"] == 2
    assert len(extraction.episode_ledger) == 2


def test_episode_merge_rule_is_explicit_and_changes_only_event_view():
    fs = 16_000
    clean = speech_like_carrier(fs, duration_sec=5.0)
    # 15-ms separation: merged at 20 ms but separated at 5 ms.
    clipped = hard_clip_bursts(
        clean,
        fs,
        [(1.0, 1.20), (1.215, 1.42)],
        limit=0.40,
    )
    default = extract_qdist(clipped, fs, provenance=pcm16_provenance())
    strict = extract_qdist(
        clipped,
        fs,
        provenance=pcm16_provenance(),
        parameters=replace(DEFAULT_PARAMETERS, episode_merge_gap_ms=5.0),
    )
    assert strict.recording["qdist_hard_clip_event_count"] >= default.recording[
        "qdist_hard_clip_event_count"
    ]
    assert strict.recording["qdist_hard_clipped_sample_fraction"] == pytest.approx(
        default.recording["qdist_hard_clipped_sample_fraction"], abs=1e-15
    )
    assert strict.recording["qdist_hard_clipped_frame_fraction"] == pytest.approx(
        default.recording["qdist_hard_clipped_frame_fraction"], abs=1e-15
    )


def test_recording_features_reconstruct_exactly_from_saved_ledgers():
    fs = 16_000
    clean = speech_like_carrier(fs)
    clipped = hard_clip_bursts(clean, fs, [(0.8, 1.5), (3.1, 3.5)], limit=0.43)
    extraction = extract_qdist(clipped, fs, provenance=pcm16_provenance())
    row = extraction.recording
    reconstructed = reconstruct_qdist_features(
        extraction.accepted_plateau_ledger,
        extraction.episode_ledger,
        finite_channel_sample_count=row["qdist_finite_channel_sample_count"],
        finite_time_sample_count=row["qdist_finite_time_sample_count"],
        finite_exposure_sec=row["qdist_finite_exposure_sec"],
        frame_length_samples=row["qdist_frame_length_samples"],
        complete_frame_count=row["qdist_complete_frame_count"],
    )
    for feature in ANALYSIS_FEATURES:
        assert reconstructed[feature] == pytest.approx(row[feature], abs=1e-15)


def test_multichannel_burden_uses_channel_samples_not_max_channel():
    fs = 16_000
    clean = speech_like_carrier(fs)
    clipped = hard_clip_bursts(clean, fs, [(1.0, 2.0)], limit=0.42)
    stereo = np.column_stack([clipped, clean])
    mono = extract_qdist(clipped, fs, provenance=pcm16_provenance())
    multi = extract_qdist(stereo, fs, provenance=pcm16_provenance())
    assert multi.recording["qdist_affected_channel_count"] == 1
    assert multi.recording["qdist_hard_clipped_sample_fraction"] == pytest.approx(
        0.5 * mono.recording["qdist_hard_clipped_sample_fraction"],
        rel=1e-12,
    )
    assert multi.recording["qdist_hard_clipped_frame_fraction"] == pytest.approx(
        mono.recording["qdist_hard_clipped_frame_fraction"],
        abs=1e-15,
    )


def test_polarity_inversion_preserves_all_three_analysis_features():
    fs = 16_000
    clipped = hard_clip_bursts(
        speech_like_carrier(fs), fs, [(1.0, 2.0)], limit=0.43
    )
    positive = extract_qdist(clipped, fs, provenance=pcm16_provenance())
    inverted = extract_qdist(-clipped, fs, provenance=pcm16_provenance())
    for feature in ANALYSIS_FEATURES:
        assert inverted.recording[feature] == pytest.approx(
            positive.recording[feature], rel=1e-12, abs=1e-15
        )


def test_post_clipping_attenuation_preserves_detection_below_full_scale():
    fs = 16_000
    clipped = hard_clip_bursts(
        speech_like_carrier(fs), fs, [(1.0, 2.0)], limit=0.44
    )
    baseline = extract_qdist(clipped, fs, provenance=NativeSignalProvenance())
    attenuated = extract_qdist(0.25 * clipped, fs, provenance=NativeSignalProvenance())
    assert attenuated.recording["qdist_status"] == "available_events"
    for feature in ANALYSIS_FEATURES:
        assert attenuated.recording[feature] == pytest.approx(
            baseline.recording[feature], rel=1e-12, abs=1e-15
        )
    assert attenuated.recording["qdist_near_fullscale_channel_sample_fraction"] == 0.0


def test_soft_clipping_is_not_required_positive():
    fs = 16_000
    clean = speech_like_carrier(fs)
    softened = apply_soft_clip(clean, drive=3.0)
    extraction = extract_qdist(softened, fs, provenance=NativeSignalProvenance())
    assert extraction.recording["qdist_status"] == "available_no_events"
    assert extraction.recording["qdist_hard_clip_event_count"] == 0


def test_coarse_quantization_of_clean_audio_does_not_create_clipping_events():
    fs = 16_000
    clean = speech_like_carrier(fs)
    for bits in (8, 12, 16, 24):
        quantized = quantize_pcm(clean, bits)
        extraction = extract_qdist(
            quantized,
            fs,
            provenance=NativeSignalProvenance(
                codec_name=f"pcm_s{bits}le",
                sample_format=f"s{bits}",
                bits_per_raw_sample=bits,
            ),
        )
        assert extraction.recording["qdist_hard_clip_event_count"] == 0, bits


def test_sine_tone_is_negative_and_square_like_waveform_is_rejected_as_ambiguous():
    fs = 16_000
    time = np.arange(5 * fs) / fs
    sine = 0.9 * np.sin(2 * np.pi * 173 * time)
    square = 0.7 * np.sign(np.sin(2 * np.pi * 173 * time))
    sine_result = extract_qdist(sine, fs, provenance=pcm16_provenance())
    square_result = extract_qdist(square, fs, provenance=pcm16_provenance())
    assert sine_result.recording["qdist_hard_clip_event_count"] == 0
    assert square_result.recording["qdist_hard_clip_event_count"] == 0
    if len(square_result.candidate_ledger):
        assert square_result.candidate_ledger["rejection_reason"].str.contains(
            "square_like_two_level_ambiguity"
        ).any()


def test_task_span_is_continuous_and_excludes_leading_trailing_audio():
    fs = 16_000
    clean = speech_like_carrier(fs, duration_sec=6.0)
    clipped = hard_clip_bursts(clean, fs, [(0.2, 0.7), (2.0, 2.8)], limit=0.4)
    extraction = extract_qdist(
        clipped,
        fs,
        task_span=TimeInterval(1.0, 5.0),
        provenance=pcm16_provenance(),
    )
    assert extraction.recording["qdist_task_span_start_sample_native"] == fs
    assert extraction.recording["qdist_task_span_end_sample_native_exclusive"] == 5 * fs
    assert extraction.accepted_plateau_ledger["start_sec_native"].ge(1.0).all()
    assert extraction.recording["qdist_hard_clip_event_count"] >= 1
    assert extraction.episode_ledger["start_sec_native"].ge(1.0).all()


def test_provenance_and_support_failures_are_missing_not_zero():
    fs = 16_000
    waveform = speech_like_carrier(fs)
    unverified = extract_qdist(
        waveform,
        fs,
        provenance=NativeSignalProvenance(native_view_verified=False),
    )
    assert unverified.recording["qdist_status"] == "unavailable_native_view_not_verified"
    preprocessed = extract_qdist(
        waveform,
        fs,
        provenance=NativeSignalProvenance(known_preprocessing_applied=True),
    )
    assert preprocessed.recording["qdist_status"] == "unavailable_preprocessed_source"
    short = extract_qdist(waveform[: fs], fs, provenance=pcm16_provenance())
    assert short.recording["qdist_status"] == "indeterminate_insufficient_support"
    for result in (unverified, preprocessed, short):
        for feature in ANALYSIS_FEATURES:
            assert np.isnan(result.recording[feature])


def test_nonfinite_fraction_contract_is_enforced_without_converting_to_zero():
    fs = 16_000
    waveform = speech_like_carrier(fs)
    waveform[: 1000] = np.nan
    result = extract_qdist(waveform, fs, provenance=pcm16_provenance())
    assert result.recording["qdist_status"] == "indeterminate_nonfinite_support"
    assert all(np.isnan(result.recording[feature]) for feature in ANALYSIS_FEATURES)


def test_poisson_interval_is_finite_and_contains_observed_rate():
    low, high = poisson_rate_interval(3, 120.0)
    observed = 3 * 60 / 120
    assert 0 <= low < observed < high
    zero_low, zero_high = poisson_rate_interval(0, 120.0)
    assert zero_low == 0
    assert zero_high > 0


def test_extraction_is_deterministic_including_all_ledgers():
    fs = 16_000
    clipped = hard_clip_bursts(
        speech_like_carrier(fs), fs, [(1.0, 2.0)], limit=0.42
    )
    first = extract_qdist(clipped, fs, provenance=pcm16_provenance())
    second = extract_qdist(clipped, fs, provenance=pcm16_provenance())
    assert first.recording == second.recording
    pd.testing.assert_frame_equal(first.candidate_ledger, second.candidate_ledger)
    pd.testing.assert_frame_equal(
        first.accepted_plateau_ledger, second.accepted_plateau_ledger
    )
    pd.testing.assert_frame_equal(first.episode_ledger, second.episode_ledger)
    pd.testing.assert_frame_equal(first.edge_ledger, second.edge_ledger)


def test_input_geometry_and_sample_rate_contracts():
    with pytest.raises(ValueError, match="positive integer"):
        extract_qdist(np.zeros(10), 0)
    with pytest.raises(ValueError, match="samples x channels"):
        extract_qdist(np.zeros((2, 3, 4)), 16_000)


def test_low_level_local_plateau_is_rejected_by_recording_relative_floor():
    fs = 16_000
    waveform = speech_like_carrier(fs)
    waveform[20_000:20_004] = 0.002
    result = extract_qdist(waveform, fs, provenance=pcm16_provenance())
    assert result.recording["qdist_hard_clip_event_count"] == 0
    assert result.recording["qdist_flat_run_count_all_amplitudes"] >= 1
    if len(result.candidate_ledger):
        assert not np.isclose(result.candidate_ledger["candidate_level"], 0.002, atol=1e-8).any()


def test_candidate_prefilter_keeps_clean_candidate_ledger_compact():
    fs = 16_000
    clean = quantize_pcm(speech_like_carrier(fs), 16)
    result = extract_qdist(clean, fs, provenance=pcm16_provenance())
    assert len(result.candidate_ledger) < 100
    assert result.recording["qdist_flat_run_prefilter_reduction_fraction"] >= 0.0


def test_local_gain_state_hard_clip_above_recording_floor_remains_detectable():
    fs = 16_000
    clean = speech_like_carrier(fs)
    clipped = hard_clip_bursts(clean, fs, [(1.0, 1.7)], limit=0.42)
    result = extract_qdist(clipped, fs, provenance=pcm16_provenance())
    assert result.recording["qdist_status"] == "available_events"
    assert result.recording["qdist_hard_clip_event_count"] >= 1
    assert result.accepted_plateau_ledger["recording_magnitude_pass"].all()
    assert result.accepted_plateau_ledger["local_magnitude_pass"].all()


def test_high_energy_hard_clip_grid_has_high_sample_precision_and_recall():
    fs = 16_000
    carrier = speech_like_carrier(fs, duration_sec=6.0, seed=71)
    duration_sec = 0.10
    window_samples = round(duration_sec * fs)
    above = (np.abs(carrier) > 0.72).astype(np.int64)
    cumulative = np.concatenate([[0], np.cumsum(above)])
    counts = cumulative[window_samples:] - cumulative[:-window_samples]
    margin = round(0.25 * fs)
    search = counts[margin:len(counts) - margin]
    start = margin + int(np.argmax(search))
    end = start + window_samples
    transformed = carrier.copy()
    transformed[start:end] = apply_hard_clip(transformed[start:end], 0.52)
    extraction = extract_qdist(transformed, fs, provenance=NativeSignalProvenance())

    truth = np.zeros(len(carrier), dtype=bool)
    truth[start:end] = np.abs(carrier[start:end]) > 0.52
    predicted = np.zeros(len(carrier), dtype=bool)
    for row in extraction.accepted_plateau_ledger.itertuples(index=False):
        predicted[int(row.start_sample_task):int(row.end_sample_task_exclusive)] = True

    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    assert truth.sum() > 0
    assert extraction.recording["qdist_hard_clip_event_count"] > 0
    assert precision >= 0.99
    assert recall >= 0.85



def test_low_gain_quantized_smooth_extremum_is_not_hard_clipping():
    """A quiet quantized sinusoidal trough inside a louder recording is not clipping."""
    fs = 44_100
    loud = speech_like_carrier(fs=fs, duration_sec=2.0, seed=311)
    time = np.arange(round(2.0 * fs)) / fs
    quiet = 0.0175 * np.sin(2 * np.pi * 173.0 * time)
    waveform = quantize_pcm(np.concatenate([loud, quiet]), 16)
    extraction = extract_qdist(
        waveform,
        fs,
        provenance=pcm16_provenance(),
    )
    assert extraction.recording["qdist_hard_clip_event_count"] == 0
    assert extraction.accepted_plateau_ledger.empty


def test_repeated_low_level_saturation_path_recovers_known_hard_clipping():
    fs = 16_000
    clean = speech_like_carrier(fs, duration_sec=6.0, seed=71)
    # A genuine hard limit below the strong recording-edge floor must remain
    # detectable when it creates extensive repeated plateau support.
    clipped = hard_clip_bursts(clean, fs, [(2.0, 2.50)], limit=0.32)
    extraction = extract_qdist(
        clipped,
        fs,
        provenance=pcm16_provenance(),
    )
    assert extraction.recording["qdist_hard_clip_event_count"] > 0
    accepted = extraction.accepted_plateau_ledger
    assert len(accepted) > 0
    assert accepted["low_level_repeated_edge_pass"].astype(bool).any()
    assert accepted["magnitude_path"].eq(
        "repeated_low_level_saturation"
    ).any()


def test_isolated_low_level_quantized_natural_extremum_is_not_accepted():
    fs = 16_000
    time_axis = np.arange(round(2.0 * fs)) / fs
    quiet_extremum = quantize_pcm(
        0.0175 * np.sin(2 * np.pi * 173.0 * time_axis),
        16,
    )
    waveform = np.concatenate(
        [
            speech_like_carrier(fs, duration_sec=2.0, seed=311),
            quiet_extremum,
        ]
    )
    extraction = extract_qdist(
        waveform,
        fs,
        provenance=pcm16_provenance(),
    )
    assert extraction.recording["qdist_hard_clip_event_count"] == 0
    if len(extraction.candidate_ledger):
        assert not extraction.candidate_ledger[
            "low_level_repeated_edge_pass"
        ].fillna(False).astype(bool).any()
