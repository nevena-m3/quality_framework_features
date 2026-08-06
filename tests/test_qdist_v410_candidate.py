from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
for source_path in [ROOT / "src", ROOT / "src"]:
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from paper1_qc import qdist_v410_candidate as qdist
from paper1_qc_reviewed.qdist_v400 import synthetic_speech_like


def analyze(waveform: np.ndarray, fs: int, recording_id: str):
    pcm = qdist.quantize_pcm(np.asarray(waveform, dtype=float), 16)
    provenance = qdist.NativeSignalProvenance(
        native_view_verified=True,
        known_preprocessing_applied=False,
        codec_name="pcm_s16le",
        sample_format="s16",
        bits_per_raw_sample=16,
    )
    return qdist.extract_qdist(
        pcm,
        fs,
        logical_recording_id=recording_id,
        provenance=provenance,
    )


def carrier(fs: int = 48_000, duration_sec: float = 4.0, seed: int = 5103):
    return synthetic_speech_like(
        sample_rate_hz=fs,
        duration_sec=duration_sec,
        seed=seed,
    )


def one_sided_clip(
    waveform: np.ndarray,
    limit: float,
    polarity: str,
) -> np.ndarray:
    output = np.asarray(waveform, dtype=float).copy()
    if polarity == "positive":
        output[output > limit] = limit
    elif polarity == "negative":
        output[output < -limit] = -limit
    else:
        raise ValueError(polarity)
    return output


def features(result) -> dict[str, float]:
    return {
        feature: float(result.recording[feature])
        for feature in qdist.ANALYSIS_FEATURES
    }


def test_candidate_identity_and_feature_roles():
    assert qdist.MEASUREMENT_VERSION == "qdist-v4.1.0-candidate"
    assert qdist.PRIMARY_FEATURES == ("qdist_hard_clipped_sample_fraction",)
    assert qdist.SECONDARY_FEATURES == ("qdist_hard_clip_event_rate_per_min",)
    assert qdist.CONDITIONAL_FEATURES == ("qdist_hard_clipped_frame_fraction",)


@pytest.mark.parametrize("fs", [8_000, 16_000, 22_050, 44_100, 48_000])
@pytest.mark.parametrize("polarity", ["positive", "negative"])
def test_moderate_one_sided_clipping_is_detected(fs, polarity):
    clipped = one_sided_clip(carrier(fs=fs), 0.60, polarity)
    result = analyze(clipped, fs, f"{polarity}_{fs}")
    assert result.recording["qdist_hard_clip_event_count"] > 0
    assert result.recording["qdist_hard_clipped_sample_fraction"] > 0


def test_one_sided_polarity_inversion_is_exact():
    clipped = one_sided_clip(carrier(), 0.60, "negative")
    negative = analyze(clipped, 48_000, "negative")
    inverted = analyze(-clipped, 48_000, "inverted")
    for feature in qdist.ANALYSIS_FEATURES:
        assert np.isclose(
            negative.recording[feature],
            inverted.recording[feature],
            rtol=0,
            atol=1e-15,
        )


def test_local_prominence_uses_same_polarity_context():
    clipped = one_sided_clip(carrier(), 0.60, "negative")
    result = analyze(clipped, 48_000, "negative_context")
    assert len(result.accepted_plateau_ledger) > 0
    required = {
        "local_context_peak_abs",
        "local_same_polarity_context_peak_abs",
        "candidate_to_context_ratio",
    }
    assert required.issubset(result.candidate_ledger.columns)
    accepted = result.accepted_plateau_ledger
    assert accepted["local_magnitude_pass"].astype(bool).all()


def test_low_level_repeated_saturation_path_is_exercised():
    waveform = carrier(duration_sec=6.0, seed=6201)
    midpoint = len(waveform) // 2
    waveform[:midpoint] *= 0.60
    waveform[:midpoint] = np.clip(waveform[:midpoint], -0.30, 0.30)
    result = analyze(waveform, 48_000, "low_level_state")
    assert len(result.accepted_plateau_ledger) > 0
    assert set(result.accepted_plateau_ledger["magnitude_path"]) == {
        "repeated_low_level_saturation"
    }


def test_subframe_shift_preserves_direct_burden_and_event_rate_but_not_grid_view():
    waveform = carrier(duration_sec=6.0, seed=4101)
    clipped = np.clip(waveform, -0.65, 0.65)
    shift = 261
    moved = np.concatenate([np.zeros(shift), clipped[:-shift]])
    baseline = analyze(clipped, 48_000, "frame_base")
    translated = analyze(moved, 48_000, "frame_shift")
    for feature in [
        "qdist_hard_clip_event_rate_per_min",
        "qdist_hard_clipped_sample_fraction",
    ]:
        assert np.isclose(
            baseline.recording[feature],
            translated.recording[feature],
            rtol=0,
            atol=1e-15,
        )
    assert not np.isclose(
        baseline.recording["qdist_hard_clipped_frame_fraction"],
        translated.recording["qdist_hard_clipped_frame_fraction"],
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("fs", [8_000, 16_000, 48_000])
@pytest.mark.parametrize("frequency", [80, 137, 440, 1_000])
def test_periodic_nonclipping_controls_remain_negative(fs, frequency):
    time = np.arange(4 * fs) / fs
    controls = [
        0.995 * np.sin(2 * np.pi * frequency * time),
        0.95 * signal.sawtooth(2 * np.pi * frequency * time, width=0.5),
        0.80 * signal.square(2 * np.pi * frequency * time),
    ]
    for index, waveform in enumerate(controls):
        result = analyze(waveform, fs, f"periodic_{fs}_{frequency}_{index}")
        assert result.recording["qdist_hard_clip_event_count"] == 0


def test_registry_has_detector_specific_scientific_lineage():
    registry = qdist.feature_registry_frame()
    references = " ".join(registry["supporting_references"].astype(str))
    assert "Hansen et al. (2021)" in references
    assert "Laguna & Lerch (2016)" in references
    assert "Eaton & Naylor (2013, 2014)" in references
