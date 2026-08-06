from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from paper1_qc.qchan import (
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    MEASUREMENT_VERSION,
    PRIMARY_FEATURES,
    SECONDARY_FEATURES,
    ReferenceSpectrum,
    TimeInterval,
    apply_gain_db,
    broad_notch_filter,
    build_subject_balanced_loso_references,
    compute_reference_relative_features,
    extract_recording_spectrum,
    feature_registry_frame,
    full_span_interval,
    highband_ratio,
    lowpass_filter,
    reference_vintage_sha256,
    smooth_high_shelf,
    spectral_descriptors,
    spectral_rolloff_hz,
    spectral_tilt_db_per_octave,
    synthetic_speech_like,
)


FS = DEFAULT_PARAMETERS.analysis_sample_rate_hz


def spectrum(y: np.ndarray, recording_id: str, source_rate: int = 48_000):
    return extract_recording_spectrum(
        y,
        FS,
        strict_speech=full_span_interval(y, FS),
        logical_recording_id=recording_id,
        source_sample_rate_hz=source_rate,
    )


def direct_reference(observation, reference_observation):
    return ReferenceSpectrum(
        reference_key="fixture",
        task_stratum="bamboo",
        excluded_subject_id="target",
        frequencies_hz=reference_observation.frequencies_hz,
        normalized_psd_per_hz=reference_observation.normalized_psd_per_hz,
        status="measured",
        member_recording_ids=("reference",),
        member_subject_ids=("reference_subject",),
        recording_count=1,
        subject_count=1,
        reference_sha256=reference_observation.spectrum_sha256,
        reference_vintage_sha256="fixture-vintage",
    )


@pytest.fixture(scope="module")
def baseline():
    return synthetic_speech_like(duration_sec=8.0, sample_rate_hz=FS)


def test_registry_is_exact_four_feature_profile():
    registry = feature_registry_frame()
    assert tuple(registry["name"]) == ANALYSIS_FEATURES
    assert set(PRIMARY_FEATURES).isdisjoint(SECONDARY_FEATURES)
    assert set(PRIMARY_FEATURES) | set(SECONDARY_FEATURES) == set(ANALYSIS_FEATURES)
    assert not registry["name"].str.contains("score|composite|burden").any()
    assert MEASUREMENT_VERSION == "qchan-v3.0.1"


def test_gain_normalization_is_numerically_invariant(baseline):
    original = spectrum(baseline, "original")
    for gain_db in (-18.0, -6.0, 6.0, 12.0):
        changed = spectrum(apply_gain_db(baseline, gain_db), f"gain_{gain_db}")
        np.testing.assert_allclose(
            original.normalized_psd_per_hz,
            changed.normalized_psd_per_hz,
            rtol=1e-12,
            atol=1e-15,
        )
        assert spectral_descriptors(
            original.frequencies_hz,
            original.normalized_psd_per_hz,
        ) == pytest.approx(
            spectral_descriptors(
                changed.frequencies_hz,
                changed.normalized_psd_per_hz,
            ),
            abs=1e-9,
        )


def test_polarity_invariance(baseline):
    positive = spectrum(baseline, "positive")
    negative = spectrum(-baseline, "negative")
    np.testing.assert_allclose(
        positive.normalized_psd_per_hz,
        negative.normalized_psd_per_hz,
        rtol=0,
        atol=1e-15,
    )


def test_identity_condition_is_zero(baseline):
    observed = spectrum(baseline, "observed")
    result = compute_reference_relative_features(
        observed, direct_reference(observed, observed)
    )
    for feature in ANALYSIS_FEATURES:
        assert result[feature] == pytest.approx(0.0, abs=1e-12)
        assert result[f"{feature}_status"] == "measured"


def test_lowpass_dose_orders_bandwidth_features(baseline):
    reference_observation = spectrum(baseline, "reference")
    reference = direct_reference(reference_observation, reference_observation)
    cutoffs = [7200.0, 6500.0, 5500.0, 4500.0, 3500.0]
    rows = []
    for cutoff in cutoffs:
        observed = spectrum(
            lowpass_filter(baseline, FS, cutoff), f"lowpass_{cutoff}"
        )
        rows.append(
            compute_reference_relative_features(observed, reference)
        )
    for feature in (
        "qchan_ltas_distance_db",
        "qchan_rolloff95_deficit_hz",
        "qchan_highband_ratio_deficit",
    ):
        values = np.asarray([row[feature] for row in rows])
        assert np.all(np.diff(values) >= -1e-8), (feature, values)
        assert values[-1] > values[0]


def test_two_sided_shelf_has_correct_orientation(baseline):
    reference_observation = spectrum(baseline, "reference")
    reference = direct_reference(reference_observation, reference_observation)
    attenuation = compute_reference_relative_features(
        spectrum(smooth_high_shelf(baseline, FS, -12.0), "attenuation"),
        reference,
    )
    boost = compute_reference_relative_features(
        spectrum(smooth_high_shelf(baseline, FS, 12.0), "boost"),
        reference,
    )
    assert attenuation["qchan_ltas_distance_db"] > 0.5
    assert boost["qchan_ltas_distance_db"] > 0.5
    assert attenuation["qchan_highband_ratio_deficit"] > 0
    assert attenuation["qchan_rolloff95_deficit_hz"] > 0
    assert attenuation["qchan_tilt_steepening_db_per_oct"] > 0
    assert boost["qchan_highband_ratio_deficit"] == pytest.approx(0.0)
    assert boost["qchan_rolloff95_deficit_hz"] == pytest.approx(0.0)
    assert boost["qchan_tilt_steepening_db_per_oct"] == pytest.approx(0.0)


def test_broad_notch_is_detected_by_nonordinal_distance(baseline):
    reference_observation = spectrum(baseline, "reference")
    reference = direct_reference(reference_observation, reference_observation)
    notched = spectrum(broad_notch_filter(baseline, FS), "notched")
    result = compute_reference_relative_features(notched, reference)
    assert result["qchan_ltas_distance_db"] > 0.5


def test_source_bandwidth_is_audited_not_reconstructed(baseline):
    observed = spectrum(baseline, "limited_source", source_rate=12_000)
    assert observed.status == "measured"
    assert observed.source_nyquist_hz == 6000
    assert observed.source_bandwidth_limited


def test_guard_and_support_contract():
    y = synthetic_speech_like(duration_sec=3.2, sample_rate_hz=FS)
    result = extract_recording_spectrum(
        y,
        FS,
        strict_speech=[TimeInterval(0.0, 3.2)],
        logical_recording_id="short_after_guards",
        source_sample_rate_hz=48_000,
    )
    assert result.status == "insufficient_strict_speech_support"
    assert np.isnan(result.normalized_psd_per_hz).all()


def reference_fixture(baseline, duplicate_subject_zero: bool = False):
    spectra = {}
    metadata = []
    subject_count = 7
    for subject_index in range(subject_count):
        repeats = 8 if duplicate_subject_zero and subject_index == 0 else 2
        for repeat in range(repeats):
            recording_id = f"s{subject_index}_r{repeat}"
            shaped = smooth_high_shelf(
                baseline,
                FS,
                gain_db=(subject_index - 3) * 0.25,
            )
            spectra[recording_id] = spectrum(shaped, recording_id)
            metadata.append({
                "logical_recording_id": recording_id,
                "subject_id": f"s{subject_index}",
                "task_stratum": "bamboo",
            })
    return spectra, pd.DataFrame(metadata)


def test_reference_is_leave_one_subject_out(baseline):
    spectra, metadata = reference_fixture(baseline)
    references = build_subject_balanced_loso_references(spectra, metadata)
    target = references["s0_r0"]
    assert target.status == "measured"
    assert "s0" not in target.member_subject_ids
    assert not any(member.startswith("s0_") for member in target.member_recording_ids)
    assert target.subject_count == 6
    assert target.recording_count == 12
    assert references["s0_r0"].reference_sha256 == references["s0_r1"].reference_sha256


def test_subject_balancing_prevents_repeat_overweighting(baseline):
    spectra_a, metadata_a = reference_fixture(baseline, False)
    spectra_b, metadata_b = reference_fixture(baseline, True)
    reference_a = build_subject_balanced_loso_references(
        spectra_a, metadata_a
    )["s6_r0"]
    reference_b = build_subject_balanced_loso_references(
        spectra_b, metadata_b
    )["s6_r0"]
    np.testing.assert_allclose(
        reference_a.normalized_psd_per_hz,
        reference_b.normalized_psd_per_hz,
        atol=1e-15,
        rtol=1e-12,
    )


def test_no_cross_task_or_global_fallback(baseline):
    spectra, metadata = reference_fixture(baseline)
    metadata.loc[
        ~metadata["subject_id"].eq("s0"), "task_stratum"
    ] = "other_task"
    references = build_subject_balanced_loso_references(spectra, metadata)
    assert references["s0_r0"].status == "reference_unavailable"
    result = compute_reference_relative_features(
        spectra["s0_r0"], references["s0_r0"]
    )
    assert result["qchan_family_status"] == "reference_unavailable"
    assert all(np.isnan(result[feature]) for feature in ANALYSIS_FEATURES)


def test_reference_requires_full_band_members(baseline):
    spectra, metadata = reference_fixture(baseline)
    for recording_id in list(spectra):
        if not recording_id.startswith("s0_"):
            spectra[recording_id] = replace(
                spectra[recording_id],
                source_sample_rate_hz=12_000,
                source_nyquist_hz=6000.0,
                source_bandwidth_limited=True,
            )
    references = build_subject_balanced_loso_references(spectra, metadata)
    assert references["s0_r0"].status == "reference_unavailable"


def test_reference_vintage_changes_with_membership(baseline):
    spectra, metadata = reference_fixture(baseline)
    first = reference_vintage_sha256(spectra, metadata)
    reduced_metadata = metadata.loc[
        ~metadata["logical_recording_id"].eq("s6_r1")
    ]
    second = reference_vintage_sha256(spectra, reduced_metadata)
    assert first != second


def test_common_mode_identity_blind_spot_is_explicit(baseline):
    filtered = lowpass_filter(baseline, FS, 4000)
    observation = spectrum(filtered, "common_mode_target")
    reference_observation = spectrum(filtered, "common_mode_reference")
    result = compute_reference_relative_features(
        observation, direct_reference(observation, reference_observation)
    )
    for feature in ANALYSIS_FEATURES:
        assert result[feature] == pytest.approx(0.0, abs=1e-12)
    assert spectral_rolloff_hz(
        observation.frequencies_hz,
        observation.normalized_psd_per_hz,
    ) < spectral_rolloff_hz(
        spectrum(baseline, "unfiltered").frequencies_hz,
        spectrum(baseline, "unfiltered").normalized_psd_per_hz,
    )


def test_descriptor_ranges(baseline):
    measured = spectrum(baseline, "measured")
    ratio = highband_ratio(
        measured.frequencies_hz, measured.normalized_psd_per_hz
    )
    rolloff = spectral_rolloff_hz(
        measured.frequencies_hz, measured.normalized_psd_per_hz
    )
    tilt = spectral_tilt_db_per_octave(
        measured.frequencies_hz, measured.normalized_psd_per_hz
    )
    assert 0 <= ratio <= 1
    assert DEFAULT_PARAMETERS.analysis_low_hz <= rolloff <= DEFAULT_PARAMETERS.analysis_high_hz
    assert np.isfinite(tilt)


def test_deterministic_extraction(baseline):
    first = spectrum(baseline, "same")
    second = spectrum(baseline.copy(), "same")
    assert first.spectrum_sha256 == second.spectrum_sha256
    np.testing.assert_array_equal(
        first.normalized_psd_per_hz,
        second.normalized_psd_per_hz,
    )


def test_digital_zero_has_specific_status_after_sufficient_time_support():
    y = np.zeros(int(4.0 * FS), dtype=np.float64)
    result = extract_recording_spectrum(
        y,
        FS,
        strict_speech=[TimeInterval(0.0, 4.0)],
        logical_recording_id="digital_zero",
        source_sample_rate_hz=48_000,
    )
    assert result.status == "digital_zero_speech"
    assert result.valid_frame_count == 0
    assert np.isnan(result.normalized_psd_per_hz).all()


def test_nonpositive_source_rate_is_rejected(baseline):
    with pytest.raises(ValueError, match="Source sample rate must be positive"):
        extract_recording_spectrum(
            baseline,
            FS,
            strict_speech=full_span_interval(baseline, FS),
            logical_recording_id="invalid_source_rate",
            source_sample_rate_hz=0,
        )


def test_reference_metadata_rejects_missing_or_blank_identity(baseline):
    measured = spectrum(baseline, "r1")
    spectra = {"r1": measured, "r2": spectrum(baseline, "r2")}
    missing = pd.DataFrame([
        {"logical_recording_id": "r1", "subject_id": pd.NA, "task_stratum": "bamboo"},
        {"logical_recording_id": "r2", "subject_id": "s2", "task_stratum": "bamboo"},
    ])
    with pytest.raises(ValueError, match="missing identity"):
        build_subject_balanced_loso_references(spectra, missing)
    blank = missing.copy()
    blank.loc[0, "subject_id"] = "   "
    with pytest.raises(ValueError, match="blank identity"):
        build_subject_balanced_loso_references(spectra, blank)


def test_signed_audit_differences_are_retained(baseline):
    reference_observation = spectrum(baseline, "reference")
    reference = direct_reference(reference_observation, reference_observation)
    attenuated = spectrum(smooth_high_shelf(baseline, FS, -12.0), "attenuated")
    result = compute_reference_relative_features(attenuated, reference)
    assert result["qchan_rolloff95_signed_difference_hz"] >= 0
    assert result["qchan_highband_ratio_signed_difference"] >= 0
    assert result["qchan_tilt_signed_difference_db_per_oct"] >= 0
    assert result["qchan_rolloff95_deficit_hz"] == pytest.approx(
        result["qchan_rolloff95_signed_difference_hz"]
    )


def test_registry_contains_full_measurement_governance_fields():
    registry = feature_registry_frame()
    required = {
        "name", "display_name", "subdomain", "role", "unit", "estimand",
        "orientation", "claim_boundary", "minimum_support",
        "known_confounds", "evidence_class",
    }
    assert required.issubset(registry.columns)
    assert registry["evidence_class"].eq(
        "study-specific reference-relative estimator"
    ).all()
