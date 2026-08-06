from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat

from paper1_qc.qrev import (
    ANALYSIS_FEATURES,
    BROADLY_AVAILABLE_COMPARATOR_FEATURES,
    CONDITIONAL_BOUNDARY_FEATURES,
    DEFAULT_PARAMETERS,
    MEASUREMENT_VERSION,
    SRMR_PINNED_REGRESSION_VALUE,
    SRMR_UPSTREAM_COMMIT,
    TimeInterval,
    ac_rms_dbfs,
    apply_gain_db,
    boundary_envelope_trace,
    compute_srmr_norm,
    estimate_srmr_working_set_mb,
    extract_qrev,
    feature_registry_frame,
    internal_pause_boundaries,
    merge_intervals,
)

FS = 16_000
FIXTURES = Path(__file__).parent / "fixtures" / "srmrpy"


def synthetic_recording(
    *,
    speech_count: int = 5,
    speech_sec: float = 0.8,
    pause_sec: float = 1.2,
    tail_level: float = 0.0,
    tail_tau_sec: float = 0.10,
    persistence_sec: float | None = None,
    floor_level: float = 2e-5,
    seed: int = 12,
) -> tuple[np.ndarray, list[TimeInterval]]:
    rng = np.random.default_rng(seed)
    total = speech_count * speech_sec + (speech_count - 1) * pause_sec
    waveform = floor_level * rng.standard_normal(round(total * FS))
    intervals = []
    cursor = 0.0
    for index in range(speech_count):
        start = cursor
        end = start + speech_sec
        intervals.append(TimeInterval(start, end))
        left = round(start * FS)
        right = round(end * FS)
        time = np.arange(right - left) / FS
        waveform[left:right] += 0.05 * (
            np.sin(2 * np.pi * 173 * time)
            + 0.5 * np.sin(2 * np.pi * 421 * time + 0.3)
            + 0.2 * np.sin(2 * np.pi * 911 * time + 0.7)
        )
        if index < speech_count - 1 and tail_level > 0:
            pause_left = right
            pause_right = round((end + pause_sec) * FS)
            relative = np.arange(pause_right - pause_left) / FS
            carrier = rng.standard_normal(len(relative))
            if persistence_sec is None:
                envelope = tail_level * np.exp(-relative / tail_tau_sec)
            else:
                envelope = tail_level * (relative < persistence_sec)
            waveform[pause_left:pause_right] += envelope * carrier
        cursor = end + (pause_sec if index < speech_count - 1 else 0.0)
    return waveform, intervals


def extract(waveform: np.ndarray, intervals: list[TimeInterval], **kwargs):
    return extract_qrev(
        waveform,
        FS,
        strict_speech=intervals,
        logical_recording_id="test",
        compute_srmr=False,
        **kwargs,
    )


def test_registry_is_exactly_four_features_and_has_no_scalar_score():
    registry = feature_registry_frame()
    assert MEASUREMENT_VERSION == "qrev-v3.1.0"
    assert tuple(registry["name"]) == ANALYSIS_FEATURES
    assert len(registry) == 4
    assert not registry["name"].str.contains(
        "rt60|edt|c50|c80|d50|drr|sti|composite|burden|score",
        case=False,
        regex=True,
    ).any()


def test_registry_distinguishes_conditional_estimators_from_comparator():
    registry = feature_registry_frame().set_index("name")
    assert tuple(CONDITIONAL_BOUNDARY_FEATURES) == ANALYSIS_FEATURES[:3]
    assert tuple(BROADLY_AVAILABLE_COMPARATOR_FEATURES) == (
        "qrev_srmr_norm",
    )
    assert registry.loc[
        list(CONDITIONAL_BOUNDARY_FEATURES), "role"
    ].str.contains("conditional").all()
    assert (
        registry.loc["qrev_srmr_norm", "role"]
        == "broadly available established comparator"
    )
    assert not registry.index.str.contains(
        "echo_delay|echo_detector|echo_identity",
        case=False,
        regex=True,
    ).any()


def test_intervals_are_clipped_merged_and_create_internal_boundaries():
    merged = merge_intervals(
        [TimeInterval(-1, 1), TimeInterval(0.8, 2), TimeInterval(3, 4)],
        3.5,
    )
    assert merged == [TimeInterval(0, 2), TimeInterval(3, 3.5)]
    boundaries = internal_pause_boundaries(merged, 3.5)
    assert len(boundaries) == 1
    assert boundaries[0]["pause_duration_sec"] == pytest.approx(1.0)


def test_ac_rms_removes_dc_and_identifies_digital_floor():
    time = np.arange(FS) / FS
    sine = 0.1 * np.sin(2 * np.pi * 200 * time)
    baseline, baseline_floor = ac_rms_dbfs(sine)
    shifted, shifted_floor = ac_rms_dbfs(sine + 0.5)
    assert shifted == pytest.approx(baseline, abs=1e-12)
    assert not baseline_floor and not shifted_floor
    zero, at_floor = ac_rms_dbfs(np.zeros(500))
    assert at_floor and zero == DEFAULT_PARAMETERS.dbfs_floor_db


def test_frame_windows_are_strictly_contained_after_offset():
    waveform, intervals = synthetic_recording()
    boundary = internal_pause_boundaries(intervals, len(waveform) / FS)[0]
    trace = boundary_envelope_trace(
        waveform,
        FS,
        boundary["speech_offset_sec"],
        boundary["pause_end_sec"],
    )
    assert (trace["relative_start_sec"] >= -1e-12).all()
    assert (trace["relative_end_sec"] <= 1.0 + 1e-12).all()
    assert trace.iloc[0]["relative_start_sec"] == pytest.approx(0.0)
    assert trace.iloc[0]["relative_end_sec"] == pytest.approx(0.03)


def test_signed_tail_excess_is_not_clipped_at_zero():
    waveform, intervals = synthetic_recording(floor_level=1e-5)
    rng = np.random.default_rng(44)
    for boundary in internal_pause_boundaries(intervals, len(waveform) / FS):
        left = round((boundary["speech_offset_sec"] + 0.70) * FS)
        right = round((boundary["speech_offset_sec"] + 1.00) * FS)
        waveform[left:right] += 0.003 * rng.standard_normal(right - left)
    result = extract(waveform, intervals)
    assert result.recording["qrev_tail_excess_100ms_db"] < 0


def test_tail_and_persistence_are_gain_invariant():
    waveform, intervals = synthetic_recording(
        tail_level=0.008,
        persistence_sec=0.20,
    )
    baseline = extract(waveform, intervals).recording
    shifted = extract(apply_gain_db(waveform, 9.0), intervals).recording
    for feature in (
        "qrev_tail_excess_100ms_db",
        "qrev_tail_persistence_median_sec",
        "qrev_downward_decay_rate_db_per_sec",
    ):
        left = baseline[feature]
        right = shifted[feature]
        if np.isnan(left):
            assert np.isnan(right)
        else:
            assert right == pytest.approx(left, abs=1e-9)


def test_persistence_recovers_controlled_plateau_duration():
    durations = [0.08, 0.16, 0.24]
    measured = []
    for duration in durations:
        waveform, intervals = synthetic_recording(
            tail_level=0.01,
            persistence_sec=duration,
            seed=33,
        )
        measured.append(
            extract(waveform, intervals).recording[
                "qrev_tail_persistence_median_sec"
            ]
        )
    assert measured == sorted(measured)
    assert np.max(np.abs(np.asarray(measured) - np.asarray(durations))) <= 0.04


def test_downward_decay_rate_recovers_exponential_envelope():
    tau = 0.12
    waveform, intervals = synthetic_recording(
        tail_level=0.02,
        tail_tau_sec=tau,
        floor_level=2e-6,
        seed=50,
    )
    result = extract(waveform, intervals).recording
    expected = 20.0 / np.log(10.0) / tau
    assert result["qrev_downward_decay_rate_db_per_sec"] == pytest.approx(
        expected,
        rel=0.16,
    )


def test_nondecaying_tail_is_not_reinterpreted_as_zero_decay():
    waveform, intervals = synthetic_recording(
        tail_level=0.01,
        persistence_sec=1.0,
    )
    result = extract(waveform, intervals).recording
    assert np.isnan(result["qrev_downward_decay_rate_db_per_sec"])
    assert result["qrev_downward_decay_rate_db_per_sec_status"] == (
        "no_valid_downward_decay"
    )


def test_short_pause_support_is_unavailable_not_zero():
    waveform, intervals = synthetic_recording(pause_sec=0.50)
    result = extract(waveform, intervals).recording
    for feature in ANALYSIS_FEATURES[:3]:
        assert np.isnan(result[feature])
        assert result[f"{feature}_status"] in {
            "insufficient_support",
            "no_valid_downward_decay",
        }


def test_boundary_ledger_reconstructs_recording_estimators():
    waveform, intervals = synthetic_recording(
        tail_level=0.012,
        persistence_sec=0.18,
    )
    result = extract(waveform, intervals)
    ledger = result.boundary_ledger
    tail = ledger.loc[ledger["tail_eligible"].astype(bool), "tail_excess_100ms_db"]
    persistence = ledger.loc[
        ledger["persistence_eligible"].astype(bool),
        "tail_persistence_sec",
    ]
    assert np.median(tail) == pytest.approx(
        result.recording["qrev_tail_excess_100ms_db"],
        abs=1e-12,
    )
    assert np.median(persistence) == pytest.approx(
        result.recording["qrev_tail_persistence_median_sec"],
        abs=1e-12,
    )
    for feature in CONDITIONAL_BOUNDARY_FEATURES:
        if np.isfinite(result.recording[feature]):
            assert result.recording[
                f"{feature}_support_tier"
            ] == "minimum"


def test_extraction_is_deterministic():
    waveform, intervals = synthetic_recording(
        tail_level=0.01,
        persistence_sec=0.15,
    )
    first = extract(waveform, intervals)
    second = extract(waveform, intervals)
    assert first.recording == second.recording
    pd.testing.assert_frame_equal(first.boundary_ledger, second.boundary_ledger)


def test_srmr_pinned_python3_runtime_regression_is_exact():
    pytest.importorskip("gammatone")
    sample = loadmat(FIXTURES / "test.mat")["s"][:, 0]
    historical = loadmat(FIXTURES / "correct_ratios.mat")["correct_ratios"][0, 2]
    assert SRMR_UPSTREAM_COMMIT == "fee009779cef96bed34db3a7e31d10f3ad1ea133"
    observed = compute_srmr_norm(sample, FS)
    assert observed == pytest.approx(
        SRMR_PINNED_REGRESSION_VALUE,
        rel=1e-10,
        abs=1e-12,
    )
    # The upstream repository's 2014 fixture predates the Python-3-compatible
    # Gammatone release. The discrepancy is intentionally explicit and is
    # reported by the notebook rather than silently waived.
    assert abs(observed - float(historical)) > 0.20


def test_srmr_is_gain_invariant():
    pytest.importorskip("gammatone")
    sample = loadmat(FIXTURES / "test.mat")["s"][:, 0]
    baseline = compute_srmr_norm(sample, FS)
    shifted = compute_srmr_norm(apply_gain_db(sample, -12.0), FS)
    assert shifted == pytest.approx(baseline, rel=1e-10, abs=1e-12)


def test_srmr_memory_preflight_can_censor_without_truncation():
    parameters = replace(DEFAULT_PARAMETERS, maximum_srmr_estimated_memory_mb=0.01)
    waveform, intervals = synthetic_recording()
    result = extract_qrev(
        waveform,
        FS,
        strict_speech=intervals,
        logical_recording_id="test",
        parameters=parameters,
        compute_srmr=True,
    ).recording
    assert estimate_srmr_working_set_mb(len(waveform), parameters) > 0.01
    assert np.isnan(result["qrev_srmr_norm"])
    assert result["qrev_srmr_norm_status"] == "resource_limit"


def test_analysis_row_has_no_scalar_family_score_or_forbidden_room_parameter():
    waveform, intervals = synthetic_recording()
    recording = extract(waveform, intervals).recording
    names = {name.lower() for name in recording}
    assert "qrev_score" not in names
    assert "qrev_composite" not in names
    forbidden_components = {"rt60", "edt", "c50", "c80", "d50", "drr", "sti"}
    assert not any(
        forbidden_components.intersection(name.split("_"))
        for name in names
    )


def test_sample_rate_mono_and_finite_contracts_are_enforced():
    with pytest.raises(ValueError, match="16000"):
        extract_qrev(
            np.zeros(8_000),
            8_000,
            strict_speech=[TimeInterval(0, 1)],
        )
    with pytest.raises(ValueError, match="mono"):
        extract_qrev(
            np.zeros((FS, 2)),
            FS,
            strict_speech=[TimeInterval(0, 1)],
        )
    with pytest.raises(ValueError, match="non-finite"):
        extract_qrev(
            np.array([0.0, np.nan]),
            FS,
            strict_speech=[],
        )
