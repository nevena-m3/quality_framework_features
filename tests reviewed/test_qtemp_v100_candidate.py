from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for source in [ROOT / "src", ROOT / "src reviewed"]:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from paper1_qc_reviewed import qtemp_v100_candidate as qtemp


def carrier(fs: int = 16_000, duration_sec: float = 6.0) -> np.ndarray:
    t = np.arange(round(fs * duration_sec)) / fs
    envelope = 0.75 + 0.2 * np.sin(2 * np.pi * 0.73 * t)
    return 0.05 * envelope * (
        np.sin(2 * np.pi * 173 * t + 0.1)
        + 0.45 * np.sin(2 * np.pi * 421 * t + 0.7)
        + 0.18 * np.sin(2 * np.pi * 911 * t + 1.2)
    )


def test_registry_and_export_are_exactly_four_features():
    registry = qtemp.feature_registry_frame()
    assert tuple(registry["name"]) == qtemp.ANALYSIS_FEATURES
    assert not registry["name"].str.contains("splice").any()
    result = qtemp.extract_qtemp(carrier(), 16_000)
    assert not any("splice" in key for key in result.recording)
    assert result.recording["qtemp_publication_ready"] is False


@pytest.mark.parametrize("duration_ms, expected", [(20.0, 0), (30.0, 0), (40.0, 1), (50.0, 1), (80.0, 1)])
def test_final_duplicate_scope_is_enforced_in_extraction(duration_ms, expected):
    waveform = qtemp.inject_consecutive_duplicate(carrier(), 16_000, 2.013, duration_ms)
    result = qtemp.extract_qtemp(
        waveform,
        16_000,
        enabled_event_types=("frozen_audio",),
    )
    assert int(result.recording["qtemp_frozen_audio_accepted_event_count"] > 0) == expected


def test_subfinal_duplicate_parameter_is_refused():
    invalid = replace(qtemp.FINAL_PARAMETERS, duplicate_min_sequence_ms=18.0)
    with pytest.raises(ValueError, match=">=37.5 ms"):
        qtemp.extract_qtemp(carrier(), 16_000, parameters=invalid)


@pytest.mark.parametrize("fs", [8_000, 16_000, 24_000, 44_100, 48_000])
@pytest.mark.parametrize("start_sec", [2.0, 2.013])
def test_40ms_truth_scope_is_stable_across_native_rates_and_alignment(fs, start_sec):
    waveform = qtemp.inject_consecutive_duplicate(carrier(fs=fs), fs, start_sec, 40.0)
    result = qtemp.extract_qtemp(
        waveform, fs, enabled_event_types=("frozen_audio",)
    )
    assert result.recording["qtemp_frozen_audio_accepted_event_count"] == 1


def test_splice_cannot_be_reenabled():
    with pytest.raises(ValueError, match="Non-retained"):
        qtemp.extract_qtemp(carrier(), 16_000, enabled_event_types=("splice",))


def test_zero_missing_and_same_ledger_reconstruction():
    clean = qtemp.extract_qtemp(carrier(), 16_000)
    unavailable = qtemp.extract_qtemp(
        carrier(), 16_000, native_source_confirmed=False
    )
    assert all(clean.recording[name] == 0 for name in qtemp.ANALYSIS_FEATURES)
    assert all(np.isnan(unavailable.recording[name]) for name in qtemp.ANALYSIS_FEATURES)
    injected = qtemp.inject_dropout(carrier(), 16_000, 2.0, 40.0, mode="zero")
    result = qtemp.extract_qtemp(injected, 16_000, enabled_event_types=("dropout",))
    rebuilt = qtemp.reconstruct_recording_features(
        result.event_ledger, result.recording["qtemp_eligible_duration_sec"]
    )
    for name in qtemp.ANALYSIS_FEATURES:
        assert rebuilt[name] == pytest.approx(result.recording[name], abs=1e-15)


def test_exact_one_second_post_guard_exposure_is_available():
    result = qtemp.extract_qtemp(carrier(duration_sec=1.2), 16_000)
    assert result.recording["qtemp_eligible_duration_sec"] == pytest.approx(1.0)
    assert result.recording["qtemp_status"] == "measured"


@pytest.mark.parametrize("gain", [0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("polarity", [1.0, -1.0])
def test_dropout_gain_and_polarity_contract(gain, polarity):
    waveform = qtemp.inject_dropout(carrier(), 16_000, 2.0, 40.0, mode="zero")
    result = qtemp.extract_qtemp(
        polarity * gain * waveform,
        16_000,
        enabled_event_types=("dropout",),
    )
    assert result.recording["qtemp_dropout_accepted_event_count"] == 1
    assert result.recording["qtemp_dropout_duration_fraction"] == pytest.approx(
        0.04 / 5.8, abs=1e-15
    )


def test_periodic_controls_have_no_accepted_frozen_audio():
    fs = 16_000
    t = np.arange(6 * fs) / fs
    controls = [
        0.05 * np.sin(2 * np.pi * 120 * t),
        0.05 * sum(np.sin(2 * np.pi * 100 * harmonic * t) / harmonic for harmonic in range(1, 8)),
        carrier(),
    ]
    for waveform in controls:
        result = qtemp.extract_qtemp(
            waveform, fs, enabled_event_types=("frozen_audio",)
        )
        assert result.recording["qtemp_frozen_audio_accepted_event_count"] == 0
