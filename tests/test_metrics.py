import numpy as np
from scipy import signal

from paper1_qc.metrics import (
    additive_interference_metrics,
    channel_device_metrics,
    gain_dynamics_metrics,
    nonlinear_distortion_metrics,
    reverberation_tail_metrics,
    temporal_discontinuity_metrics,
)
from paper1_qc.segmentation import Interval


SR = 16000


def test_additive_metrics_track_noise_level_and_snr():
    rng = np.random.default_rng(4)
    duration = 12
    speech_intervals = [Interval(0.5 + 1.5 * i, 1.5 + 1.5 * i) for i in range(7)]
    noise_intervals = [Interval(1.7 + 1.5 * i, 2.0 + 1.5 * i) for i in range(6)]
    time = np.arange(duration * SR) / SR
    speech = 0.1 * np.sin(2 * np.pi * 180 * time)
    speech_mask = np.zeros(len(time), dtype=bool)
    for interval in speech_intervals:
        speech_mask[int(interval.start_sec * SR) : int(interval.end_sec * SR)] = True

    quiet = rng.normal(0, 0.001, len(time))
    loud = rng.normal(0, 0.01, len(time))
    quiet[speech_mask] += speech[speech_mask]
    loud[speech_mask] += speech[speech_mask]
    quiet_result = additive_interference_metrics(
        quiet,
        SR,
        strict_speech=speech_intervals,
        strict_internal_nonspeech=noise_intervals,
    )
    loud_result = additive_interference_metrics(
        loud,
        SR,
        strict_speech=speech_intervals,
        strict_internal_nonspeech=noise_intervals,
    )
    assert loud_result["qadd_nonspeech_level_dbfs"] > quiet_result["qadd_nonspeech_level_dbfs"] + 15
    assert loud_result["qadd_snr_proxy_db"] < quiet_result["qadd_snr_proxy_db"] - 15


def test_hum_prominence_increases_for_60hz_tone():
    rng = np.random.default_rng(5)
    duration = 8
    time = np.arange(duration * SR) / SR
    speech_intervals = [Interval(0.0, 3.5), Interval(4.5, 8.0)]
    noise_intervals = [Interval(3.5, 4.5)]
    baseline = rng.normal(0, 0.002, len(time))
    hum = baseline + 0.02 * np.sin(2 * np.pi * 60 * time)
    baseline_result = additive_interference_metrics(
        baseline, SR, strict_speech=speech_intervals, strict_internal_nonspeech=noise_intervals
    )
    hum_result = additive_interference_metrics(
        hum, SR, strict_speech=speech_intervals, strict_internal_nonspeech=noise_intervals
    )
    assert hum_result["qadd_hum_prominence_db"] > baseline_result["qadd_hum_prominence_db"] + 10


def test_additive_pause_metrics_do_not_require_three_seconds_of_speech():
    rng = np.random.default_rng(41)
    waveform = rng.normal(0, 0.002, 5 * SR)
    result = additive_interference_metrics(
        waveform,
        SR,
        strict_speech=[Interval(0.0, 2.0)],
        strict_internal_nonspeech=[Interval(2.5, 3.5)],
    )
    assert result["qadd_status"] == "partial_support"
    assert result["qadd_nonspeech_level_dbfs_status"] == "ok"
    assert np.isfinite(result["qadd_nonspeech_level_dbfs"])
    assert result["qadd_nonspeech_variability_db_status"] == "ok"
    assert np.isfinite(result["qadd_nonspeech_variability_db"])
    assert result["qadd_transient_rate_per_min_status"] == "ok"
    assert np.isfinite(result["qadd_transient_rate_per_min"])
    assert result["qadd_snr_proxy_db_status"] == "insufficient_speech_duration"
    assert np.isnan(result["qadd_snr_proxy_db"])


def test_additive_spectral_status_matches_continuous_pause_support():
    rng = np.random.default_rng(42)
    waveform = rng.normal(0, 0.002, 6 * SR)
    result = additive_interference_metrics(
        waveform,
        SR,
        strict_speech=[Interval(0.0, 3.5)],
        strict_internal_nonspeech=[
            Interval(4.0, 4.2),
            Interval(4.5, 4.7),
            Interval(5.0, 5.2),
        ],
    )
    assert result["qadd_status"] == "partial_support"
    assert result["qadd_nonspeech_level_dbfs_status"] == "ok"
    assert result["qadd_spectral_flatness_status"] == "insufficient_spectral_support"
    assert result["qadd_hum_prominence_db_status"] == (
        "insufficient_contiguous_spectral_support"
    )
    assert np.isnan(result["qadd_spectral_flatness"])
    assert np.isnan(result["qadd_hum_prominence_db"])


def test_additive_transients_are_counted_separately_across_pauses():
    rng = np.random.default_rng(43)
    waveform = rng.normal(0, 0.001, 5 * SR)
    pauses = [Interval(1.0, 1.5), Interval(3.0, 3.5)]
    waveform[int(1.43 * SR) : int(1.50 * SR)] += rng.normal(
        0, 0.02, int(0.07 * SR)
    )
    waveform[int(3.00 * SR) : int(3.07 * SR)] += rng.normal(
        0, 0.02, int(0.07 * SR)
    )
    result = additive_interference_metrics(
        waveform,
        SR,
        strict_speech=[
            Interval(0.0, 1.0),
            Interval(1.5, 3.0),
            Interval(3.5, 5.0),
        ],
        strict_internal_nonspeech=pauses,
    )
    # Two events over one second of supported nonspeech = 120 events/min.
    assert result["qadd_transient_rate_per_min"] == 120.0


def test_additive_relative_metrics_are_invariant_to_global_gain():
    rng = np.random.default_rng(44)
    waveform = rng.normal(0, 0.002, 6 * SR)
    time = np.arange(len(waveform)) / SR
    speech = [Interval(0.0, 3.5), Interval(4.8, 6.0)]
    pause = [Interval(3.5, 4.8)]
    speech_mask = np.zeros(len(waveform), dtype=bool)
    for interval in speech:
        speech_mask[
            int(interval.start_sec * SR) : int(interval.end_sec * SR)
        ] = True
    waveform[speech_mask] += 0.05 * np.sin(
        2 * np.pi * 180 * time[speech_mask]
    )
    low = additive_interference_metrics(
        waveform * 0.5,
        SR,
        strict_speech=speech,
        strict_internal_nonspeech=pause,
    )
    high = additive_interference_metrics(
        waveform * 2.0,
        SR,
        strict_speech=speech,
        strict_internal_nonspeech=pause,
    )
    assert np.isclose(
        high["qadd_nonspeech_level_dbfs"]
        - low["qadd_nonspeech_level_dbfs"],
        20 * np.log10(4),
        atol=1e-8,
    )
    for feature in [
        "qadd_snr_proxy_db",
        "qadd_nonspeech_variability_db",
        "qadd_hum_prominence_db",
        "qadd_transient_rate_per_min",
        "qadd_spectral_flatness",
    ]:
        assert np.isclose(high[feature], low[feature], atol=1e-8)


def test_gain_drift_reports_db_per_minute():
    time = np.arange(20 * SR) / SR
    # A 20-dB amplitude rise over 20 seconds corresponds to about 60 dB/min.
    envelope = 10 ** ((20 * time / 20) / 20)
    waveform = 0.005 * envelope * np.sin(2 * np.pi * 180 * time)
    result = gain_dynamics_metrics(waveform, SR, strict_speech=[Interval(0, 20)])
    assert 50 <= result["qgain_abs_drift_db_per_min"] <= 70


def test_channel_bandwidth_proxy_responds_to_lowpass():
    rng = np.random.default_rng(6)
    wide = rng.normal(0, 0.02, 6 * SR)
    sos = signal.butter(8, 1200, btype="lowpass", fs=SR, output="sos")
    narrow = signal.sosfilt(sos, wide)
    intervals = [Interval(0, 6)]
    wide_result = channel_device_metrics(wide, SR, strict_speech=intervals)
    narrow_result = channel_device_metrics(narrow, SR, strict_speech=intervals)
    assert narrow_result["qchan_effective_bandwidth_hz"] < wide_result["qchan_effective_bandwidth_hz"]
    assert narrow_result["qchan_highband_ratio"] < wide_result["qchan_highband_ratio"]


def test_hard_clipping_detector_separates_clean_and_clipped_sine():
    time = np.arange(5 * SR) / SR
    clean = 0.7 * np.sin(2 * np.pi * 213 * time)
    clipped = np.clip(1.5 * np.sin(2 * np.pi * 213 * time), -0.8, 0.8)
    intervals = [Interval(0, 5)]
    clean_result = nonlinear_distortion_metrics(clean[:, None], SR, strict_speech=intervals)
    clipped_result = nonlinear_distortion_metrics(clipped[:, None], SR, strict_speech=intervals)
    assert clean_result["qdist_hard_clip_sample_fraction"] == 0
    assert clipped_result["qdist_hard_clip_sample_fraction"] > 0.05
    assert clipped_result["qdist_clip_event_rate_per_min"] > 0


def test_dropout_events_use_original_contiguous_intervals():
    time = np.arange(6 * SR) / SR
    waveform = 0.05 * np.sin(2 * np.pi * 190 * time)
    waveform[int(1.0 * SR) : int(1.05 * SR)] = 0
    waveform[int(3.0 * SR) : int(3.08 * SR)] = 0
    result = temporal_discontinuity_metrics(
        waveform, SR, strict_speech=[Interval(0, 6)]
    )
    assert result["qtemp_zero_dropout_fraction"] > 0.015
    assert 15 <= result["qtemp_zero_dropout_rate_per_min"] <= 25


def _speech_with_tails(tail_amplitude: float, decay_rate: float):
    rng = np.random.default_rng(1)
    waveform = rng.normal(0, 1e-4, 7 * SR)
    intervals = [Interval(0.2, 1.0), Interval(2.0, 2.8), Interval(3.8, 4.6), Interval(5.6, 6.4)]
    for interval in intervals:
        start = int(interval.start_sec * SR)
        end = int(interval.end_sec * SR)
        local_t = np.arange(end - start) / SR
        waveform[start:end] += 0.1 * np.sin(2 * np.pi * 180 * local_t)
        tail_end = min(len(waveform), end + int(0.5 * SR))
        tail_t = np.arange(tail_end - end) / SR
        waveform[end:tail_end] += tail_amplitude * np.exp(-decay_rate * tail_t) * np.sin(
            2 * np.pi * 180 * tail_t
        )
    return waveform, intervals


def test_reverberation_tail_proxy_increases_for_slow_tail():
    dry, intervals = _speech_with_tails(0.005, 25)
    wet, _ = _speech_with_tails(0.05, 5)
    dry_result = reverberation_tail_metrics(dry, SR, primary_speech=intervals)
    wet_result = reverberation_tail_metrics(wet, SR, primary_speech=intervals)
    assert dry_result["qrev_status"] == "ok"
    assert wet_result["qrev_tail_excess_db"] > dry_result["qrev_tail_excess_db"]
    assert wet_result["qrev_decay_time_proxy_sec"] >= dry_result["qrev_decay_time_proxy_sec"]
