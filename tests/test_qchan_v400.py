from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper1_qc_reviewed.qchan_v400 import (
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    MEASUREMENT_VERSION,
    PREFLIGHT_REVISION,
    RecordingSpectrum,
    TimeInterval,
    analysis_waveform_from_audio_views,
    apply_gain_db,
    broad_notch_filter,
    build_subject_balanced_loso_references,
    compute_reference_relative_features,
    extract_controlled_features,
    extract_recording_spectrum,
    feature_registry_frame,
    full_span_interval,
    lowpass_filter,
    reference_from_recording_spectrum,
    reference_vintage_sha256,
    smooth_high_shelf,
    synthetic_speech_like,
)

FS = DEFAULT_PARAMETERS.analysis_sample_rate_hz


def baseline_bundle():
    waveform = synthetic_speech_like(duration_sec=12.0, sample_rate_hz=FS)
    spectrum = extract_recording_spectrum(
        waveform,
        FS,
        strict_speech=full_span_interval(waveform, FS),
        logical_recording_id="baseline",
    )
    reference = reference_from_recording_spectrum(spectrum)
    return waveform, spectrum, reference


def feature_vector(row):
    return np.array([float(row[name]) for name in ANALYSIS_FEATURES])


def test_version_and_registry_are_exactly_four_features():
    assert MEASUREMENT_VERSION == "qchan-v4.0.0"
    assert PREFLIGHT_REVISION == "qchan-v4.0.0-preflight-r1"
    assert tuple(feature_registry_frame()["name"]) == ANALYSIS_FEATURES
    assert len(ANALYSIS_FEATURES) == 4
    assert not any("scalar" in name.lower() for name in ANALYSIS_FEATURES)


def test_audio_views_helper_requires_analysis_16k():
    class Good:
        analysis_16k = np.array([0.0, 1.0, -1.0])
        analysis = np.array([99.0])

    class Bad:
        analysis = np.array([0.0])

    assert np.array_equal(
        analysis_waveform_from_audio_views(Good()),
        np.array([0.0, 1.0, -1.0]),
    )
    try:
        analysis_waveform_from_audio_views(Bad())
    except AttributeError as exc:
        assert "analysis_16k" in str(exc)
    else:
        raise AssertionError("Historical AudioViews alias was accepted")


def test_gain_polarity_and_dc_are_feature_invariant():
    waveform, _, reference = baseline_bundle()
    baseline = extract_controlled_features(
        waveform,
        reference=reference,
        logical_recording_id="baseline_target",
    )
    variants = [
        apply_gain_db(waveform, -12.0),
        apply_gain_db(waveform, 9.0),
        -waveform,
        waveform + 0.25,
    ]
    expected = feature_vector(baseline)
    for index, variant in enumerate(variants):
        observed = feature_vector(
            extract_controlled_features(
                variant,
                reference=reference,
                logical_recording_id=f"variant_{index}",
            )
        )
        assert np.allclose(observed, expected, atol=1e-10, rtol=1e-10)


def test_common_time_shift_with_shifted_interval_is_invariant():
    waveform, _, reference = baseline_bundle()
    pad_n = int(0.75 * FS)
    shifted = np.pad(waveform, (pad_n, 0))
    interval = [
        TimeInterval(
            -DEFAULT_PARAMETERS.speech_boundary_guard_ms / 1000 + 0.75,
            0.75 + len(waveform) / FS
            + DEFAULT_PARAMETERS.speech_boundary_guard_ms / 1000,
        )
    ]
    spectrum = extract_recording_spectrum(
        shifted,
        FS,
        strict_speech=interval,
        logical_recording_id="shifted",
    )
    observed = compute_reference_relative_features(spectrum, reference)
    assert np.allclose(feature_vector(observed), 0.0, atol=1e-10)


def test_lowpass_dose_orders_bandwidth_features():
    waveform, _, reference = baseline_bundle()
    cutoffs = [7500.0, 6500.0, 5500.0, 4500.0, 4000.0, 3400.0]
    rows = []
    for cutoff in cutoffs:
        transformed = lowpass_filter(waveform, FS, cutoff)
        rows.append(
            extract_controlled_features(
                transformed,
                reference=reference,
                logical_recording_id=f"lp_{int(cutoff)}",
            )
        )
    ltas = np.array([row["qchan_ltas_distance_db"] for row in rows])
    rolloff = np.array([row["qchan_rolloff95_deficit_hz"] for row in rows])
    highband = np.array([row["qchan_highband_ratio_deficit"] for row in rows])
    assert np.all(np.diff(ltas) >= -1e-8)
    assert np.all(np.diff(rolloff) >= -1e-8)
    assert np.all(np.diff(highband) >= -1e-8)
    assert ltas[-1] > ltas[1]
    assert rolloff[-1] > rolloff[1]
    assert highband[-1] > highband[1]


def test_two_sided_shelves_and_notch_increase_nonordinal_ltas_distance():
    waveform, _, reference = baseline_bundle()
    negative = extract_controlled_features(
        smooth_high_shelf(waveform, FS, -12.0),
        reference=reference,
        logical_recording_id="shelf_negative",
    )
    positive = extract_controlled_features(
        smooth_high_shelf(waveform, FS, 12.0),
        reference=reference,
        logical_recording_id="shelf_positive",
    )
    notch = extract_controlled_features(
        broad_notch_filter(waveform, FS, depth_db=-18.0),
        reference=reference,
        logical_recording_id="notch",
    )
    assert negative["qchan_ltas_distance_db"] > 1.0
    assert positive["qchan_ltas_distance_db"] > 1.0
    assert notch["qchan_ltas_distance_db"] > 1.0
    assert negative["qchan_rolloff95_deficit_hz"] >= 0
    assert positive["qchan_rolloff95_deficit_hz"] == 0


def test_one_sided_features_keep_signed_precursors():
    waveform, _, reference = baseline_bundle()
    boosted = extract_controlled_features(
        smooth_high_shelf(waveform, FS, 12.0),
        reference=reference,
        logical_recording_id="boosted",
    )
    assert boosted["qchan_rolloff95_deficit_hz"] == 0
    assert boosted["qchan_rolloff95_signed_difference_hz"] < 0
    assert boosted["qchan_highband_ratio_deficit"] == 0
    assert boosted["qchan_highband_ratio_signed_difference"] < 0


def test_insufficient_support_is_missing_not_zero():
    waveform = synthetic_speech_like(duration_sec=2.0, sample_rate_hz=FS)
    spectrum = extract_recording_spectrum(
        waveform,
        FS,
        strict_speech=full_span_interval(waveform, FS),
        logical_recording_id="short",
    )
    assert spectrum.status == "insufficient_strict_speech_support"
    baseline, _, reference = baseline_bundle()
    result = compute_reference_relative_features(spectrum, reference)
    for feature in ANALYSIS_FEATURES:
        assert np.isnan(result[feature])
        assert result[f"{feature}_status"] != "measured"


def _synthetic_reference_records():
    base = synthetic_speech_like(duration_sec=6.0, sample_rate_hz=FS)
    spectra = {}
    rows = []
    for subject_index in range(6):
        for recording_index in range(2):
            gain = 1.0 + 0.01 * subject_index + 0.005 * recording_index
            recording_id = f"s{subject_index}_r{recording_index}"
            spectrum = extract_recording_spectrum(
                base * gain,
                FS,
                strict_speech=full_span_interval(base, FS),
                logical_recording_id=recording_id,
            )
            spectra[recording_id] = spectrum
            rows.append({
                "logical_recording_id": recording_id,
                "subject_id": f"s{subject_index}",
                "task_stratum": "bamboo",
            })
    return spectra, pd.DataFrame(rows)


def test_loso_reference_excludes_target_subject_and_is_subject_balanced():
    spectra, metadata = _synthetic_reference_records()
    references = build_subject_balanced_loso_references(spectra, metadata)
    target = metadata.iloc[0]
    reference = references[target.logical_recording_id]
    assert reference.status == "measured"
    assert target.subject_id not in reference.member_subject_ids
    assert reference.subject_count == 5
    assert reference.recording_count == len(metadata) - 2
    assert len(set(reference.member_subject_ids)) == reference.subject_count


def test_reference_is_unavailable_without_minimum_other_subjects():
    spectra, metadata = _synthetic_reference_records()
    limited = metadata.loc[metadata["subject_id"].isin(["s0", "s1", "s2", "s3", "s4"])].copy()
    limited_spectra = {key: spectra[key] for key in limited.logical_recording_id}
    references = build_subject_balanced_loso_references(limited_spectra, limited)
    first = references[limited.iloc[0].logical_recording_id]
    assert first.status == "reference_unavailable"


def test_reference_vintage_changes_with_membership():
    spectra, metadata = _synthetic_reference_records()
    first = reference_vintage_sha256(spectra, metadata)
    reduced = metadata.iloc[:-1].copy()
    second = reference_vintage_sha256(
        {key: spectra[key] for key in reduced.logical_recording_id}, reduced
    )
    assert first != second


def test_source_bandwidth_limitation_is_retained_as_metadata():
    waveform, _, reference = baseline_bundle()
    result = extract_controlled_features(
        waveform,
        reference=reference,
        logical_recording_id="limited_source",
        source_sample_rate_hz=8000,
    )
    assert result["qchan_source_sample_rate_hz"] == 8000
    assert result["qchan_source_nyquist_hz"] == 4000
    assert result["qchan_source_bandwidth_limited"] is True


def test_floor_parameter_is_part_of_measurement_identity_and_changes_scale():
    waveform, _, _ = baseline_bundle()
    baseline_spectrum = extract_recording_spectrum(
        waveform,
        FS,
        strict_speech=full_span_interval(waveform, FS),
        logical_recording_id="baseline",
    )
    transformed = lowpass_filter(waveform, FS, 3400.0)
    values = []
    for floor_db in (-100.0, -80.0, -60.0):
        params = replace(DEFAULT_PARAMETERS, relative_psd_floor_db=floor_db)
        reference = reference_from_recording_spectrum(
            baseline_spectrum, parameters=params
        )
        observation = extract_recording_spectrum(
            transformed,
            FS,
            strict_speech=full_span_interval(transformed, FS),
            logical_recording_id=f"lp_{floor_db}",
            parameters=params,
        )
        row = compute_reference_relative_features(
            observation, reference, parameters=params
        )
        values.append(row["qchan_ltas_distance_db"])
    assert len(set(round(value, 6) for value in values)) == 3
    assert values[0] > values[1] > values[2]


def test_frames_do_not_cross_separated_segments():
    base = synthetic_speech_like(duration_sec=8.0, sample_rate_hz=FS)
    intervals = [TimeInterval(0.0, 3.0), TimeInterval(5.0, 8.0)]
    altered = base.copy()
    altered[int(3.0 * FS):int(5.0 * FS)] = 100.0
    first = extract_recording_spectrum(
        base,
        FS,
        strict_speech=intervals,
        logical_recording_id="base",
    )
    second = extract_recording_spectrum(
        altered,
        FS,
        strict_speech=intervals,
        logical_recording_id="altered_gap",
    )
    assert np.array_equal(first.normalized_psd_per_hz, second.normalized_psd_per_hz)


def test_source_rate_roundtrip_shows_expected_bandwidth_ordering():
    waveform, _, reference = baseline_bundle()
    source_rates = [16000, 12000, 8000]
    deficits = []
    for rate in source_rates:
        if rate == FS:
            roundtrip = waveform
        else:
            divisor = np.gcd(FS, rate)
            down = signal.resample_poly(waveform, rate // divisor, FS // divisor)
            divisor2 = np.gcd(rate, FS)
            roundtrip = signal.resample_poly(down, FS // divisor2, rate // divisor2)
            roundtrip = roundtrip[: len(waveform)]
            if len(roundtrip) < len(waveform):
                roundtrip = np.pad(roundtrip, (0, len(waveform) - len(roundtrip)))
        row = extract_controlled_features(
            roundtrip,
            reference=reference,
            logical_recording_id=f"source_{rate}",
            source_sample_rate_hz=rate,
        )
        deficits.append(row["qchan_rolloff95_deficit_hz"])
    assert deficits[0] <= deficits[1] <= deficits[2]


def test_high_frequency_noise_can_mask_bandwidth_deficit():
    waveform, _, reference = baseline_bundle()
    lowpassed = lowpass_filter(waveform, FS, 3400.0)
    rng = np.random.default_rng(7)
    noise = rng.normal(size=len(waveform))
    sos = signal.butter(4, [4500 / (FS / 2), 7400 / (FS / 2)], btype="band", output="sos")
    high_noise = signal.sosfilt(sos, noise)
    high_noise *= 0.08 * np.std(lowpassed) / max(np.std(high_noise), 1e-12)
    plain = extract_controlled_features(
        lowpassed,
        reference=reference,
        logical_recording_id="plain_lp",
    )
    masked = extract_controlled_features(
        lowpassed + high_noise,
        reference=reference,
        logical_recording_id="masked_lp",
    )
    assert masked["qchan_highband_ratio_deficit"] < plain["qchan_highband_ratio_deficit"]
    assert masked["qchan_rolloff95_deficit_hz"] < plain["qchan_rolloff95_deficit_hz"]



def test_saved_psd_grid_supports_declared_smoothing_sensitivity_variants():
    waveform, observation, reference = baseline_bundle()
    for octave_fraction in [1, 2, 3]:
        parameters = replace(
            DEFAULT_PARAMETERS,
            octave_fraction=octave_fraction,
        )
        values = compute_reference_relative_features(
            observation,
            reference,
            parameters,
        )
        assert all(
            np.isfinite(values[feature])
            for feature in ANALYSIS_FEATURES
        )

    # A one-sixth-octave grid beginning at 100 Hz is not resolvable on the
    # frozen n_fft=2048 / 16-kHz PSD grid. It must not be used as a cohort
    # sensitivity condition without recomputing spectra at higher resolution.
    with pytest.raises(ValueError, match="non-finite bands"):
        compute_reference_relative_features(
            observation,
            reference,
            replace(DEFAULT_PARAMETERS, octave_fraction=6),
        )
