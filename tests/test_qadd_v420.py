import numpy as np

from paper1_qc_reviewed.qadd_v420 import (
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    MEASUREMENT_VERSION,
    QADDParameters,
    TimeInterval,
    ac_rms_measurement,
    apply_hum_null_calibration,
    cluster_delete_one_diagnostics,
    compare_reconstruction,
    erode_intervals,
    extract_qadd,
    guarded_internal_pauses,
    intersect_intervals,
    summarize_cluster_deletion,
    _raw_feature_estimates,
)

SR = 16_000


def _speech_pause_fixture(
    *,
    pause_noise_sd: float = 0.002,
    hum_hz: float | None = None,
    hum_amplitude: float = 0.0,
    seed: int = 71,
):
    duration = 16.0
    time = np.arange(int(duration * SR)) / SR
    rng = np.random.default_rng(seed)
    primary_speech = [
        TimeInterval(0.0, 2.0),
        TimeInterval(3.0, 5.0),
        TimeInterval(6.0, 8.0),
        TimeInterval(9.0, 11.0),
        TimeInterval(12.0, 16.0),
    ]
    strict_pauses = [
        TimeInterval(2.0, 3.0),
        TimeInterval(5.0, 6.0),
        TimeInterval(8.0, 9.0),
        TimeInterval(11.0, 12.0),
    ]
    waveform = rng.normal(0.0, pause_noise_sd, time.size)
    speech_mask = np.zeros(time.size, dtype=bool)
    for interval in primary_speech:
        speech_mask[
            round(interval.start_sec * SR) : round(interval.end_sec * SR)
        ] = True
    waveform[speech_mask] += 0.05 * np.sin(2 * np.pi * 180.0 * time[speech_mask])
    if hum_hz is not None and hum_amplitude > 0:
        hum = np.zeros_like(waveform)
        for harmonic, relative in [(1, 1.0), (2, 0.6), (3, 0.35), (4, 0.2)]:
            hum += (
                hum_amplitude
                * relative
                * np.sin(2 * np.pi * hum_hz * harmonic * time)
            )
        pause_mask = ~speech_mask
        waveform[pause_mask] += hum[pause_mask]
    return waveform, primary_speech, strict_pauses


def _extract(waveform, speech, pauses, **kwargs):
    return extract_qadd(
        waveform,
        SR,
        primary_speech=speech,
        strict_speech=speech,
        strict_internal_nonspeech=pauses,
        logical_recording_id="synthetic",
        **kwargs,
    )


def test_ac_rms_matches_sine_theory():
    assert MEASUREMENT_VERSION == "qadd-v4.2.0-candidate"
    time = np.arange(int(0.03 * SR)) / SR
    amplitude = 0.1
    sine = amplitude * np.sin(2 * np.pi * 1000.0 * time)
    observed_db, observed_rms, at_floor, exact_zero = ac_rms_measurement(sine)
    assert np.isclose(observed_rms, amplitude / np.sqrt(2), atol=1e-12)
    assert np.isclose(observed_db, 20 * np.log10(amplitude / np.sqrt(2)), atol=1e-12)
    assert not at_floor
    assert not exact_zero


def test_frozen_frequency_parameters_match_measurement_contract():
    assert DEFAULT_PARAMETERS.flatness_low_hz == 80.0
    assert DEFAULT_PARAMETERS.flatness_high_hz == 7000.0
    assert DEFAULT_PARAMETERS.hum_window_ms == 500.0
    assert DEFAULT_PARAMETERS.hum_tone_half_width_hz == 2.0
    assert DEFAULT_PARAMETERS.hum_sideband_inner_hz > DEFAULT_PARAMETERS.hum_tone_half_width_hz


def test_interval_guard_and_intersection_are_exact():
    speech = [
        TimeInterval(0.0, 1.0),
        TimeInterval(2.0, 3.0),
        TimeInterval(4.0, 5.0),
    ]
    strict = [TimeInterval(1.1, 1.7), TimeInterval(3.25, 3.9)]
    pauses = guarded_internal_pauses(
        speech,
        5.0,
        strict_nonspeech=strict,
        guard_ms=200.0,
        minimum_residual_ms=100.0,
    )
    assert pauses == [TimeInterval(1.2, 1.7), TimeInterval(3.25, 3.8)]
    assert intersect_intervals(
        [TimeInterval(0.0, 2.0)], [TimeInterval(1.0, 3.0)]
    ) == [TimeInterval(1.0, 2.0)]


def test_noise_dose_raises_pause_level_and_reduces_contrast():
    quiet, speech, pauses = _speech_pause_fixture(pause_noise_sd=0.0005, seed=72)
    noisy, _, _ = _speech_pause_fixture(pause_noise_sd=0.01, seed=72)
    quiet_result = _extract(quiet, speech, pauses).recording
    noisy_result = _extract(noisy, speech, pauses).recording
    assert (
        noisy_result["qadd_pause_ac_level_dbfs_median"]
        > quiet_result["qadd_pause_ac_level_dbfs_median"] + 20.0
    )
    assert (
        noisy_result["qadd_speech_pause_level_contrast_db"]
        < quiet_result["qadd_speech_pause_level_contrast_db"] - 20.0
    )


def test_global_gain_has_prespecified_equivariance_and_invariance():
    waveform, speech, pauses = _speech_pause_fixture(seed=73)
    low = _extract(waveform * 0.5, speech, pauses).recording
    high = _extract(waveform * 2.0, speech, pauses).recording
    expected_shift = 20.0 * np.log10(4.0)
    assert np.isclose(
        high["qadd_pause_ac_level_dbfs_median"]
        - low["qadd_pause_ac_level_dbfs_median"],
        expected_shift,
        atol=1e-10,
    )
    for feature in [
        "qadd_pause_level_iqr_db",
        "qadd_speech_pause_level_contrast_db",
        "qadd_pause_spectral_flatness",
        "qadd_mains_hum_comb_score_db",
    ]:
        assert np.isclose(high[feature], low[feature], atol=1e-10)


def test_flatness_is_higher_for_broadband_than_tonal_pause_content():
    broadband, speech, pauses = _speech_pause_fixture(pause_noise_sd=0.01, seed=74)
    tonal, _, _ = _speech_pause_fixture(
        pause_noise_sd=0.0001,
        hum_hz=437.0,
        hum_amplitude=0.02,
        seed=74,
    )
    broadband_result = _extract(broadband, speech, pauses).recording
    tonal_result = _extract(tonal, speech, pauses).recording
    assert (
        broadband_result["qadd_pause_spectral_flatness"]
        > tonal_result["qadd_pause_spectral_flatness"] + 0.25
    )


def test_hum_comb_responds_to_50hz_harmonics_and_identifies_winner():
    baseline, speech, pauses = _speech_pause_fixture(pause_noise_sd=0.002, seed=75)
    hum, _, _ = _speech_pause_fixture(
        pause_noise_sd=0.002,
        hum_hz=50.0,
        hum_amplitude=0.02,
        seed=75,
    )
    baseline_result = _extract(baseline, speech, pauses).recording
    hum_extraction = _extract(hum, speech, pauses)
    hum_result = hum_extraction.recording
    assert (
        hum_result["qadd_mains_hum_comb_score_db"]
        > baseline_result["qadd_mains_hum_comb_score_db"] + 15.0
    )
    assert hum_result["qadd_mains_hum_winner_hz"] == 50.0
    assert hum_result["qadd_mains_hum_supported_harmonic_count_median"] >= 3
    hum_windows = hum_extraction.spectral_ledger.loc[
        hum_extraction.spectral_ledger["window_kind"].eq("hum")
        & hum_extraction.spectral_ledger["valid_acoustic_window"].astype(bool)
    ]
    assert hum_windows["hum_evaluated_harmonic_count"].eq(4).all()


def test_off_grid_tone_does_not_mimic_mains_comb():
    mains, speech, pauses = _speech_pause_fixture(
        pause_noise_sd=0.002,
        hum_hz=60.0,
        hum_amplitude=0.02,
        seed=76,
    )
    off_grid, _, _ = _speech_pause_fixture(
        pause_noise_sd=0.002,
        hum_hz=53.0,
        hum_amplitude=0.02,
        seed=76,
    )
    mains_result = _extract(mains, speech, pauses).recording
    off_grid_result = _extract(off_grid, speech, pauses).recording
    assert (
        mains_result["qadd_mains_hum_comb_score_db"]
        > off_grid_result["qadd_mains_hum_comb_score_db"] + 10.0
    )


def test_floor_censoring_blocks_analysis_value_but_preserves_raw_estimate():
    waveform, speech, pauses = _speech_pause_fixture(seed=77)
    # One of four one-second frozen pauses becomes digital zero.  After the
    # 200-ms guards this produces 25% censored pause frames, above the frozen
    # calibrated ceiling of 2%.
    zero_pause = pauses[0]
    waveform[
        int(zero_pause.start_sec * SR) : int(zero_pause.end_sec * SR)
    ] = 0.0
    result = _extract(waveform, speech, pauses).recording
    assert result["qadd_pause_at_floor_frame_fraction"] > 0.10
    assert result["qadd_pause_ac_level_dbfs_median_status"] == "floor_censored"
    assert np.isnan(result["qadd_pause_ac_level_dbfs_median"])
    assert np.isfinite(result["qadd_pause_ac_level_dbfs_median_raw_estimate"])
    assert result["qadd_family_status"] == "floor_censored"
    # Spectral estimands already exclude floor-only windows.  Their availability
    # is therefore governed by their own valid-window support, not the level
    # estimator's floor-censoring threshold.
    assert result["qadd_pause_spectral_flatness_status"].startswith("ok_")
    assert result["qadd_mains_hum_comb_score_db_status"].startswith("ok_")
    assert np.isfinite(result["qadd_pause_spectral_flatness"])
    assert np.isfinite(result["qadd_mains_hum_comb_score_db"])
    assert result["qadd_flatness_at_floor_window_fraction"] > 0
    assert result["qadd_hum_at_floor_window_fraction"] > 0


def test_insufficient_support_is_feature_specific():
    rng = np.random.default_rng(78)
    waveform = rng.normal(0.0, 0.002, 5 * SR)
    speech = [TimeInterval(0.0, 2.0), TimeInterval(2.55, 5.0)]
    pause = [TimeInterval(2.0, 2.55)]
    result = _extract(waveform, speech, pause).recording
    assert result["qadd_pause_ac_level_dbfs_median_status"] == "insufficient_support"
    assert result["qadd_pause_level_iqr_db_status"] == "insufficient_support"
    assert np.isnan(result["qadd_pause_ac_level_dbfs_median"])
    assert np.isnan(result["qadd_pause_level_iqr_db"])


def test_saved_ledgers_reconstruct_every_raw_estimate():
    waveform, speech, pauses = _speech_pause_fixture(seed=79)
    extraction = _extract(waveform, speech, pauses)
    comparison = compare_reconstruction(extraction)
    assert comparison["pass"].all()
    assert set(ANALYSIS_FEATURES).issubset(extraction.recording)
    assert extraction.frame_ledger["logical_recording_id"].eq("synthetic").all()
    assert extraction.spectral_ledger["logical_recording_id"].eq("synthetic").all()


def test_cluster_deletion_covers_every_feature_and_summarizes_by_recording():
    waveform, speech, pauses = _speech_pause_fixture(seed=82)
    extraction = _extract(waveform, speech, pauses)
    diagnostics = cluster_delete_one_diagnostics(
        extraction.frame_ledger, extraction.spectral_ledger
    )
    assert set(diagnostics["feature"]) == set(ANALYSIS_FEATURES)
    assert diagnostics["omitted_interval"].nunique() == len(pauses)
    assert diagnostics.groupby("omitted_interval")["feature"].size().eq(
        len(ANALYSIS_FEATURES)
    ).all()
    summary = summarize_cluster_deletion(diagnostics)
    assert len(summary) == len(ANALYSIS_FEATURES)
    assert summary["logical_recording_id"].eq("synthetic").all()
    assert summary["delete_one_max_absolute_change"].ge(0).all()


def test_support_classes_do_not_claim_empirical_robustness():
    waveform, speech, pauses = _speech_pause_fixture(seed=83)
    recording = _extract(waveform, speech, pauses).recording
    support_columns = [
        "qadd_pause_level_support_tier",
        "qadd_pause_dispersion_support_tier",
        "qadd_speech_pause_contrast_support_tier",
        "qadd_flatness_support_tier",
        "qadd_hum_support_tier",
    ]
    allowed = {"minimum", "moderate", "high", "unavailable"}
    assert all(recording[column] in allowed for column in support_columns)
    assert all("robust" not in recording[column] for column in support_columns)


def test_boundary_audit_uses_genuine_additional_erosion():
    waveform, speech, strict_pauses = _speech_pause_fixture(seed=84)
    reference_pauses = guarded_internal_pauses(
        speech,
        len(waveform) / SR,
        strict_nonspeech=strict_pauses,
    )
    eroded_pauses = erode_intervals(
        reference_pauses,
        len(waveform) / SR,
        guard_ms=100.0,
        minimum_ms=DEFAULT_PARAMETERS.minimum_residual_pause_ms,
    )
    assert eroded_pauses != reference_pauses
    assert sum(item.duration_sec for item in eroded_pauses) < sum(
        item.duration_sec for item in reference_pauses
    )
    reference = extract_qadd(
        waveform,
        SR,
        primary_speech=speech,
        strict_speech=speech,
        strict_internal_nonspeech=reference_pauses,
        pause_intervals_are_guarded=True,
        logical_recording_id="synthetic",
    ).recording
    eroded = extract_qadd(
        waveform,
        SR,
        primary_speech=speech,
        strict_speech=speech,
        strict_internal_nonspeech=eroded_pauses,
        pause_intervals_are_guarded=True,
        logical_recording_id="synthetic",
    ).recording
    assert (
        eroded["qadd_pause_effective_nonfloor_support_sec"]
        < reference["qadd_pause_effective_nonfloor_support_sec"]
    )
    assert all(
        f"{feature}_raw_estimate" in reference and f"{feature}_raw_estimate" in eroded
        for feature in ANALYSIS_FEATURES
    )


def test_nondefault_floor_limit_is_explicit_and_effective():
    waveform, speech, pauses = _speech_pause_fixture(seed=80)
    waveform[int(pauses[0].start_sec * SR) : int(pauses[0].end_sec * SR)] = 0.0
    permissive = QADDParameters(
        maximum_floor_censored_fraction=0.30,
        random_seed=DEFAULT_PARAMETERS.random_seed,
    )
    result = _extract(waveform, speech, pauses, parameters=permissive).recording
    assert result["qadd_pause_ac_level_dbfs_median_status"].startswith("ok_")
    assert np.isfinite(result["qadd_pause_ac_level_dbfs_median"])


def test_hum_null_calibration_preserves_raw_feature_and_adds_audit_companions():
    waveform, speech, pauses = _speech_pause_fixture(
        pause_noise_sd=0.002,
        hum_hz=50.0,
        hum_amplitude=0.02,
        seed=81,
    )
    result = _extract(waveform, speech, pauses).recording
    calibrated = apply_hum_null_calibration(result, null_p95_db=3.5)
    assert calibrated["qadd_mains_hum_comb_score_db"] == result["qadd_mains_hum_comb_score_db"]
    assert calibrated["qadd_mains_hum_null_p95_db"] == 3.5
    assert np.isclose(
        calibrated["qadd_mains_hum_excess_over_null_p95_db"],
        result["qadd_mains_hum_comb_score_db_raw_estimate"] - 3.5,
    )
    assert calibrated["qadd_mains_hum_null_calibration_status"] == "applied"
    assert calibrated["qadd_mains_hum_joint_evidence_above_null"]



def test_canonical_strict_speech_can_be_used_without_second_erosion():
    waveform, primary, pauses = _speech_pause_fixture(seed=91)
    canonical_strict = [TimeInterval(item.start_sec + 0.05, item.end_sec - 0.05) for item in primary]
    direct = extract_qadd(
        waveform,
        SR,
        primary_speech=primary,
        strict_speech=canonical_strict,
        strict_internal_nonspeech=pauses,
        speech_intervals_are_guarded=True,
        logical_recording_id="direct",
    )
    duplicated = extract_qadd(
        waveform,
        SR,
        primary_speech=primary,
        strict_speech=canonical_strict,
        strict_internal_nonspeech=pauses,
        speech_intervals_are_guarded=False,
        logical_recording_id="duplicated",
    )
    direct_speech = direct.frame_ledger.loc[direct.frame_ledger["region"].eq("speech")]
    duplicated_speech = duplicated.frame_ledger.loc[duplicated.frame_ledger["region"].eq("speech")]
    assert len(direct_speech) > len(duplicated_speech)
    assert direct.recording["qadd_speech_source"] == "provided_guarded_strict_speech"
    assert direct.recording["qadd_speech_guard_applied_ms"] == 0.0
    assert duplicated.recording["qadd_speech_guard_applied_ms"] == DEFAULT_PARAMETERS.speech_guard_ms


def test_recording_winner_uses_frequency_specific_harmonic_support():
    import pandas as pd
    frame = pd.DataFrame(
        {
            "region": ["pause", "speech"],
            "rms_dbfs": [-50.0, -20.0],
            "at_computational_floor": [False, False],
        }
    )
    spectral = pd.DataFrame(
        {
            "window_kind": ["hum", "hum", "flatness"],
            "valid_acoustic_window": [True, True, True],
            "spectral_flatness": [np.nan, np.nan, 0.2],
            "hum_score_50_db": [10.0, 8.0, np.nan],
            "hum_score_60_db": [1.0, 20.0, np.nan],
            # Per-window winner counts are intentionally misleading.
            "hum_supported_harmonic_count": [4, 4, np.nan],
            "hum_supported_harmonic_count_50": [4, 3, np.nan],
            "hum_supported_harmonic_count_60": [0, 4, np.nan],
        }
    )
    raw = _raw_feature_estimates(frame, spectral)
    # Median 50-Hz score = 9, median 60-Hz score = 10.5, so 60 Hz wins.
    assert raw["qadd_mains_hum_winner_hz"] == 60.0
    # Correct winner-specific median is median([0, 4]) = 2, not the mixed [4, 4] = 4.
    assert raw["qadd_mains_hum_supported_harmonic_count_median"] == 2.0
