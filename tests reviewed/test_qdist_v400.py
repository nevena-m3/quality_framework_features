from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for source_path in [ROOT / "src", ROOT / "src reviewed"]:
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from paper1_qc_reviewed import qdist_v400 as reviewed


def source(duration=4.0, seed=991):
    return reviewed.synthetic_speech_like(
        sample_rate_hz=48_000,
        duration_sec=duration,
        seed=seed,
    )


def summary(result):
    return reviewed._result_summary(result)


def test_exact_feature_registry():
    assert reviewed.legacy_contract()["features_exact"]


def test_legacy_version_is_pinned():
    assert reviewed.legacy_contract()["version_exact"]


def test_production_api_does_not_require_test_fixture_exports():
    contract = reviewed.legacy_contract()
    assert contract["missing_symbols"] == []
    assert contract["fixture_helpers_owned_by_reviewed_preflight"]
    assert contract["resolved_helper_api"]["synthetic_speech_like"] == "reviewed_local"
    assert contract["resolved_helper_api"]["hard_clip"] == "reviewed_local"
    assert contract["resolved_helper_api"]["soft_clip_tanh"] == "reviewed_local"


def test_quantizer_adapter_returns_pcm_and_positive_step():
    waveform = source()
    pcm, step = reviewed.quantize_fixture(waveform, 16)
    assert pcm.shape == waveform.shape
    assert np.isfinite(pcm).all()
    assert step == pytest.approx(1.0 / 32768.0)


def test_clean_speech_is_not_positive():
    result, _, _ = reviewed.analyze_pcm(source(), 48_000, "CLEAN")
    assert not reviewed.is_positive(result)


def test_hard_clipping_is_positive():
    clipped, _ = reviewed.hard_clip(source(), 0.60)
    result, _, _ = reviewed.analyze_pcm(clipped, 48_000, "CLIPPED")
    assert reviewed.is_positive(result)


def test_polarity_equivariance():
    clipped, _ = reviewed.hard_clip(source(), 0.60)
    result, _, _ = reviewed.analyze_pcm(clipped, 48_000, "BASE")
    inverted, _, _ = reviewed.analyze_pcm(-clipped, 48_000, "INV")
    for feature in reviewed.ANALYSIS_FEATURES:
        assert np.isclose(
            summary(result)[feature],
            summary(inverted)[feature],
            rtol=0,
            atol=1e-12,
        )


def test_common_frame_aligned_time_shift_invariance():
    clipped, _ = reviewed.hard_clip(source(), 0.60)
    shift = reviewed.detector_frame_length_samples(48_000)
    shifted = np.concatenate([np.zeros(shift), clipped[:-shift]])
    result, _, _ = reviewed.analyze_pcm(clipped, 48_000, "BASE")
    moved, _, _ = reviewed.analyze_pcm(shifted, 48_000, "SHIFT")
    for feature in reviewed.ANALYSIS_FEATURES:
        assert np.isclose(
            summary(result)[feature],
            summary(moved)[feature],
            rtol=0,
            atol=1e-12,
        )


def test_exact_mono_ledger_reconstruction():
    clipped, _ = reviewed.hard_clip(source(), 0.60)
    result, pcm, _ = reviewed.analyze_pcm(clipped, 48_000, "RECON")
    rebuilt = reviewed.reconstruct_mono(result, len(pcm), 48_000)
    for feature in reviewed.ANALYSIS_FEATURES:
        assert np.isclose(
            summary(result)[feature],
            rebuilt[feature],
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("gain", [0.25, 0.50, 0.75, 1.00])
def test_post_clip_attenuation_preserves_event_presence(gain):
    clipped, _ = reviewed.hard_clip(source(), 0.60)
    result, _, _ = reviewed.analyze_pcm(clipped * gain, 48_000, f"GAIN_{gain}")
    assert reviewed.is_positive(result)


def test_short_support_is_not_positive():
    clipped, _ = reviewed.hard_clip(source(duration=1.0), 0.60)
    result, _, _ = reviewed.analyze_pcm(clipped, 48_000, "SHORT")
    assert not reviewed.is_positive(result)


@pytest.mark.parametrize("bits", [8, 10, 12])
def test_coarse_pcm_does_not_create_positive(bits):
    result, _, _ = reviewed.analyze_pcm(source(), 48_000, f"Q{bits}", bit_depth=bits)
    assert not reviewed.is_positive(result)


@pytest.mark.parametrize("bits", [16, 24])
def test_supported_pcm_detects_hard_clipping(bits):
    clipped, _ = reviewed.hard_clip(source(), 0.60)
    result, _, _ = reviewed.analyze_pcm(
        clipped,
        48_000,
        f"Q{bits}_CLIP",
        bit_depth=bits,
    )
    assert reviewed.is_positive(result)


@pytest.mark.parametrize("drive", [1.0, 2.0, 4.0])
def test_moderate_smooth_saturation_not_promoted(drive):
    waveform = reviewed.soft_clip_tanh(source(), drive)
    result, _, _ = reviewed.analyze_pcm(waveform, 48_000, f"SOFT_{drive}")
    assert not reviewed.is_positive(result)
