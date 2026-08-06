from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from paper1_qc.external_validation import (
    DEFAULT_COMPARATOR_SPECS,
    comparator_registry_frame,
    discover_project_root,
    native_peak_audit,
    _parse_float,
)


def test_comparator_registry_is_unique_and_directional():
    frame = comparator_registry_frame()
    assert len(frame) == len(DEFAULT_COMPARATOR_SPECS)
    assert not frame[
        ["feature", "comparator", "comparator_column"]
    ].duplicated().any()
    assert set(frame["expected_direction"]) <= {"positive", "negative"}
    assert frame["claim_limit"].str.len().gt(10).all()


def test_parse_float_reads_ffmpeg_style_value():
    text = "RMS level dB: -23.125\nPeak count: 48"
    assert _parse_float(text, r"RMS level dB:\s*([-+0-9.eE]+)") == -23.125
    assert _parse_float(text, r"Peak count:\s*([-+0-9.eE]+)") == 48.0
    assert np.isnan(_parse_float(text, r"Not present:\s*([-+0-9.eE]+)"))


def test_native_peak_audit(tmp_path: Path):
    sample_rate = 16_000
    waveform = np.zeros(sample_rate, dtype=np.float64)
    waveform[100:110] = 1.0
    path = tmp_path / "test.wav"
    sf.write(path, waveform, sample_rate, subtype="FLOAT")

    result = native_peak_audit(path, near_fullscale=0.999)
    assert result["native_sample_rate_hz"] == sample_rate
    assert result["native_peak_abs"] == 1.0
    assert result["native_near_fullscale_fraction"] == 10 / sample_rate
    assert result["native_exact_peak_fraction"] == 10 / sample_rate


def test_discover_project_root(tmp_path: Path):
    root = tmp_path / "repo"
    nested = root / "notebooks" / "06_external_validation"
    nested.mkdir(parents=True)
    (root / "src").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert discover_project_root(nested) == root.resolve()
