import numpy as np

from paper1_qc_reviewed.qtemp_revised import extract_qtemp_revised

FS = 16000


def _signal(duration: float = 3.0) -> np.ndarray:
    time = np.arange(round(FS * duration)) / FS
    return 0.08 * np.sin(2 * np.pi * 211 * time)


def test_clean_signal_has_no_dropout() -> None:
    result = extract_qtemp_revised(_signal(), FS)
    assert result.recording["qtemp_revised_status"] == "measured"
    assert result.recording["qtemp_revised_dropout_count"] == 0


def test_bracketed_zero_run_is_detected() -> None:
    signal = _signal()
    signal[FS : FS + round(0.08 * FS)] = 0.0
    result = extract_qtemp_revised(signal, FS)
    assert result.recording["qtemp_revised_dropout_count"] >= 1
    assert result.recording["qtemp_revised_dropout_duration_fraction"] > 0


def test_abrupt_sample_jump_is_detected() -> None:
    signal = _signal()
    signal[FS:] += 0.5
    result = extract_qtemp_revised(signal, FS)
    assert result.recording["qtemp_revised_glitch_count"] >= 1
    assert result.recording["qtemp_revised_glitch_peak_robust_z"] > 6


def test_edge_silence_is_not_a_dropout() -> None:
    signal = _signal()
    signal[: round(0.2 * FS)] = 0.0
    result = extract_qtemp_revised(signal, FS)
    assert result.recording["qtemp_revised_dropout_count"] == 0
