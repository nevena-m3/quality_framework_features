from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import signal, stats

from .media import AudioViews
from .segmentation import Interval


EPS = 1e-12
PSD_TINY = np.finfo(np.float64).tiny


def dbfs_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    rms = np.sqrt(np.mean(values**2, dtype=np.float64))
    return float(20 * np.log10(max(rms, EPS)))


def _interval_slices(
    waveform: np.ndarray, sample_rate: int, intervals: Iterable[Interval], min_samples: int = 1
) -> list[np.ndarray]:
    clips = []
    for item in intervals:
        start = max(0, int(round(item.start_sec * sample_rate)))
        end = min(len(waveform), int(round(item.end_sec * sample_rate)))
        if end - start >= min_samples:
            clips.append(np.asarray(waveform[start:end]))
    return clips


def _total_duration(intervals: Iterable[Interval]) -> float:
    return float(sum(item.duration_sec for item in intervals))


def _eligible_interval_support(
    intervals: Iterable[Interval], *, minimum_clip_sec: float
) -> tuple[list[Interval], float]:
    """Return intervals that meet a continuous-duration requirement and their exposure."""
    eligible = [
        item for item in intervals if item.duration_sec + EPS >= minimum_clip_sec
    ]
    return eligible, _total_duration(eligible)


def _frame_levels(
    waveform: np.ndarray,
    sample_rate: int,
    intervals: Iterable[Interval],
    *,
    frame_ms: float = 30,
    hop_ms: float = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_len = max(1, int(round(frame_ms * sample_rate / 1000)))
    hop = max(1, int(round(hop_ms * sample_rate / 1000)))
    times: list[float] = []
    levels: list[float] = []
    segment_ids: list[int] = []
    for segment_id, item in enumerate(intervals):
        start = max(0, int(round(item.start_sec * sample_rate)))
        end = min(len(waveform), int(round(item.end_sec * sample_rate)))
        for frame_start in range(start, end - frame_len + 1, hop):
            frame = waveform[frame_start : frame_start + frame_len]
            times.append((frame_start + frame_len / 2) / sample_rate)
            levels.append(dbfs_rms(frame))
            segment_ids.append(segment_id)
    return np.asarray(times), np.asarray(levels), np.asarray(segment_ids, dtype=int)


def _count_runs(mask: np.ndarray, minimum_length: int = 1) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= minimum_length]


def _mean_interval_psd(
    waveform: np.ndarray,
    sample_rate: int,
    intervals: Iterable[Interval],
    *,
    minimum_clip_sec: float = 0.25,
    maximum_nperseg_sec: float = 2.0,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    minimum = int(round(minimum_clip_sec * sample_rate))
    psds = []
    weights = []
    nfft = max(4096, int(round(maximum_nperseg_sec * sample_rate)))
    common_freqs = np.fft.rfftfreq(nfft, d=1 / sample_rate)
    for clip in _interval_slices(waveform, sample_rate, intervals, min_samples=minimum):
        nperseg = min(len(clip), max(256, int(round(maximum_nperseg_sec * sample_rate))))
        freqs, psd = signal.welch(
            clip,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            nfft=nfft,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
        )
        psds.append(np.interp(common_freqs, freqs, psd))
        weights.append(len(clip))
    if not psds:
        return None, None
    return common_freqs, np.average(np.vstack(psds), axis=0, weights=np.asarray(weights))


def _spectral_flatness_ratio(psd: np.ndarray) -> float:
    """Return a gain-invariant geometric/arithmetic PSD ratio."""
    power = np.asarray(psd, dtype=np.float64)
    if power.size == 0 or not np.isfinite(power).all():
        return np.nan
    mean_power = float(np.mean(power))
    if mean_power <= PSD_TINY:
        return np.nan
    safe_power = np.maximum(power, PSD_TINY)
    return float(np.exp(np.mean(np.log(safe_power))) / mean_power)


def _power_ratio_db(numerator: float, denominator: float) -> float:
    """Return a gain-invariant power ratio in dB, or NaN for a silent reference."""
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if denominator <= PSD_TINY:
        return np.nan
    return float(10 * np.log10(max(numerator, PSD_TINY) / denominator))


def additive_interference_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    strict_speech: list[Interval],
    strict_internal_nonspeech: list[Interval],
) -> dict[str, float | str]:
    speech_sec = _total_duration(strict_speech)
    nonspeech_sec = _total_duration(strict_internal_nonspeech)
    _, speech_levels, _ = _frame_levels(waveform, sample_rate, strict_speech)
    _, noise_levels, noise_segment_ids = _frame_levels(
        waveform, sample_rate, strict_internal_nonspeech
    )
    flatness_intervals, flatness_support_sec = _eligible_interval_support(
        strict_internal_nonspeech, minimum_clip_sec=0.25
    )
    hum_intervals, hum_support_sec = _eligible_interval_support(
        strict_internal_nonspeech, minimum_clip_sec=1.0
    )
    nonspeech_durations = [
        item.duration_sec for item in strict_internal_nonspeech
    ]
    result: dict[str, float | str] = {
        "qadd_status": "insufficient_support",
        "qadd_speech_support_sec": speech_sec,
        "qadd_nonspeech_support_sec": nonspeech_sec,
        "qadd_speech_frame_count": int(len(speech_levels)),
        "qadd_nonspeech_frame_count": int(len(noise_levels)),
        "qadd_nonspeech_interval_count": int(len(strict_internal_nonspeech)),
        "qadd_max_nonspeech_interval_sec": (
            float(max(nonspeech_durations)) if nonspeech_durations else 0.0
        ),
        "qadd_flatness_spectral_support_sec": flatness_support_sec,
        "qadd_hum_spectral_support_sec": hum_support_sec,
        "qadd_nonspeech_level_dbfs": np.nan,
        "qadd_nonspeech_level_dbfs_status": "insufficient_nonspeech_support",
        "qadd_snr_proxy_db": np.nan,
        "qadd_snr_proxy_db_status": "insufficient_nonspeech_support",
        "qadd_nonspeech_variability_db": np.nan,
        "qadd_nonspeech_variability_db_status": "insufficient_nonspeech_support",
        "qadd_hum_prominence_db": np.nan,
        "qadd_hum_prominence_db_status": "insufficient_contiguous_spectral_support",
        "qadd_transient_rate_per_min": np.nan,
        "qadd_transient_rate_per_min_status": "insufficient_nonspeech_support",
        "qadd_spectral_flatness": np.nan,
        "qadd_spectral_flatness_status": "insufficient_spectral_support",
    }

    nonspeech_support_ok = nonspeech_sec >= 0.5 and len(noise_levels) >= 20
    if nonspeech_support_ok:
        result["qadd_nonspeech_level_dbfs"] = float(np.median(noise_levels))
        result["qadd_nonspeech_level_dbfs_status"] = "ok"
        result["qadd_nonspeech_variability_db"] = float(
            np.subtract(*np.percentile(noise_levels, [75, 25]))
        )
        result["qadd_nonspeech_variability_db_status"] = "ok"

        median = float(np.median(noise_levels))
        mad = float(np.median(np.abs(noise_levels - median)))
        transient = noise_levels >= median + max(12.0, 6 * 1.4826 * mad)
        transient_events = sum(
            len(_count_runs(transient[noise_segment_ids == segment_id]))
            for segment_id in np.unique(noise_segment_ids)
        )
        result["qadd_transient_rate_per_min"] = float(
            transient_events / nonspeech_sec * 60
        )
        result["qadd_transient_rate_per_min_status"] = "ok"
    elif nonspeech_sec >= 0.5:
        frame_status = "insufficient_nonspeech_frames"
        result["qadd_nonspeech_level_dbfs_status"] = frame_status
        result["qadd_nonspeech_variability_db_status"] = frame_status
        result["qadd_transient_rate_per_min_status"] = frame_status

    speech_support_ok = speech_sec >= 3.0 and len(speech_levels) >= 100
    if nonspeech_support_ok and speech_support_ok:
        result["qadd_snr_proxy_db"] = float(
            np.median(speech_levels) - np.median(noise_levels)
        )
        result["qadd_snr_proxy_db_status"] = "ok"
    elif nonspeech_support_ok:
        result["qadd_snr_proxy_db_status"] = (
            "insufficient_speech_duration"
            if speech_sec < 3.0
            else "insufficient_speech_frames"
        )

    if flatness_support_sec >= 1.0:
        freqs, psd = _mean_interval_psd(
            waveform,
            sample_rate,
            flatness_intervals,
            minimum_clip_sec=0.25,
        )
        if freqs is not None:
            valid = (freqs >= 20) & (freqs <= min(1000, sample_rate / 2))
            if valid.any():
                flatness = _spectral_flatness_ratio(psd[valid])
                if np.isfinite(flatness):
                    result["qadd_spectral_flatness"] = flatness
                    result["qadd_spectral_flatness_status"] = "ok"
                else:
                    result["qadd_spectral_flatness_status"] = "silent_spectral_support"
            else:
                result["qadd_spectral_flatness_status"] = "unsupported_frequency_range"
        else:
            result["qadd_spectral_flatness_status"] = "spectral_computation_failed"

    if hum_support_sec >= 1.0:
        freqs, psd = _mean_interval_psd(
            waveform,
            sample_rate,
            hum_intervals,
            minimum_clip_sec=1.0,
        )
        candidate_prominence = []
        if freqs is not None:
            for fundamental in (50.0, 60.0):
                harmonic_db = []
                for harmonic in range(1, 5):
                    center = fundamental * harmonic
                    if center + 12 >= sample_rate / 2:
                        continue
                    tone = (freqs >= center - 1.0) & (freqs <= center + 1.0)
                    reference = (
                        ((freqs >= center - 12) & (freqs <= center - 4))
                        | ((freqs >= center + 4) & (freqs <= center + 12))
                    )
                    if tone.any() and reference.any():
                        prominence = _power_ratio_db(
                            float(np.mean(psd[tone])),
                            float(np.median(psd[reference])),
                        )
                        if np.isfinite(prominence):
                            harmonic_db.append(prominence)
                if harmonic_db:
                    candidate_prominence.append(float(np.mean(harmonic_db)))
        if candidate_prominence:
            result["qadd_hum_prominence_db"] = float(max(candidate_prominence))
            result["qadd_hum_prominence_db_status"] = "ok"
        else:
            result["qadd_hum_prominence_db_status"] = "spectral_computation_failed"

    primary_status_fields = [
        "qadd_nonspeech_level_dbfs_status",
        "qadd_snr_proxy_db_status",
        "qadd_nonspeech_variability_db_status",
        "qadd_hum_prominence_db_status",
        "qadd_transient_rate_per_min_status",
    ]
    primary_ok = sum(result[field] == "ok" for field in primary_status_fields)
    if primary_ok == len(primary_status_fields):
        result["qadd_status"] = "ok"
    elif primary_ok:
        result["qadd_status"] = "partial_support"
    return result


def rest_reference_metrics(waveform: np.ndarray, sample_rate: int) -> dict[str, float | str]:
    """Describe the full Rest task without applying speech VAD.

    Rest is a contextual session reference, not guaranteed silence or a calibrated noise floor.
    The first/last 0.5 seconds are guarded when duration permits.
    """
    duration = len(waveform) / sample_rate
    interval = Interval(0.5, duration - 0.5) if duration >= 2 else Interval(0, duration)
    _, levels, _ = _frame_levels(waveform, sample_rate, [interval])
    result: dict[str, float | str] = {
        "restref_status": "ok",
        "restref_support_sec": interval.duration_sec,
        "restref_level_dbfs": np.nan,
        "restref_level_iqr_db": np.nan,
        "restref_transient_rate_per_min": np.nan,
        "restref_hum_prominence_db": np.nan,
        "restref_spectral_flatness": np.nan,
    }
    if interval.duration_sec < 1 or len(levels) < 20:
        result["restref_status"] = "insufficient_support"
        return result
    median = float(np.median(levels))
    mad = float(np.median(np.abs(levels - median)))
    result["restref_level_dbfs"] = median
    result["restref_level_iqr_db"] = float(np.subtract(*np.percentile(levels, [75, 25])))
    transient = levels >= median + max(12.0, 6 * 1.4826 * mad)
    result["restref_transient_rate_per_min"] = float(
        len(_count_runs(transient)) / interval.duration_sec * 60
    )
    freqs, psd = _mean_interval_psd(waveform, sample_rate, [interval])
    if freqs is None:
        return result
    valid = (freqs >= 20) & (freqs <= min(1000, sample_rate / 2))
    flatness = _spectral_flatness_ratio(psd[valid])
    if np.isfinite(flatness):
        result["restref_spectral_flatness"] = flatness
    prominence = []
    for fundamental in (50.0, 60.0):
        values = []
        for harmonic in range(1, 5):
            center = fundamental * harmonic
            tone = (freqs >= center - 1) & (freqs <= center + 1)
            reference = (
                ((freqs >= center - 12) & (freqs <= center - 4))
                | ((freqs >= center + 4) & (freqs <= center + 12))
            )
            if tone.any() and reference.any():
                candidate = _power_ratio_db(
                    float(np.mean(psd[tone])),
                    float(np.median(psd[reference])),
                )
                if np.isfinite(candidate):
                    values.append(candidate)
        if values:
            prominence.append(float(np.mean(values)))
    if prominence:
        result["restref_hum_prominence_db"] = max(prominence)
    return result


def gain_dynamics_metrics(
    waveform: np.ndarray, sample_rate: int, *, strict_speech: list[Interval]
) -> dict[str, float | str]:
    speech_sec = _total_duration(strict_speech)
    times, levels, segment_ids = _frame_levels(waveform, sample_rate, strict_speech)
    result: dict[str, float | str] = {
        "qgain_status": "ok",
        "qgain_speech_support_sec": speech_sec,
        "qgain_active_level_dbfs": np.nan,
        "qgain_level_iqr_db": np.nan,
        "qgain_segment_sd_db": np.nan,
        "qgain_abs_drift_db_per_min": np.nan,
        "qgain_step_rate_per_min": np.nan,
        "qgain_crest_factor_db": np.nan,
    }
    if speech_sec < 3 or len(levels) < 100:
        result["qgain_status"] = "insufficient_support"
        return result
    result["qgain_active_level_dbfs"] = float(np.median(levels))
    result["qgain_level_iqr_db"] = float(np.subtract(*np.percentile(levels, [75, 25])))

    segment_medians = np.asarray([np.median(levels[segment_ids == index]) for index in np.unique(segment_ids)])
    if len(segment_medians) >= 3:
        result["qgain_segment_sd_db"] = float(np.std(segment_medians, ddof=1))

    if times[-1] - times[0] >= 10:
        if len(times) > 500:
            selected = np.linspace(0, len(times) - 1, 500).round().astype(int)
        else:
            selected = np.arange(len(times))
        slope = stats.theilslopes(levels[selected], times[selected]).slope
        result["qgain_abs_drift_db_per_min"] = float(abs(slope) * 60)

    changes = []
    for segment_id in np.unique(segment_ids):
        local = levels[segment_ids == segment_id]
        if len(local) >= 2:
            changes.extend(np.abs(np.diff(local)))
    result["qgain_step_rate_per_min"] = float(np.sum(np.asarray(changes) >= 12.0) / speech_sec * 60)

    clips = _interval_slices(waveform, sample_rate, strict_speech)
    if clips:
        peak = max(float(np.max(np.abs(clip))) for clip in clips)
        total_sum_squares = sum(float(np.sum(clip.astype(np.float64) ** 2)) for clip in clips)
        total_n = sum(len(clip) for clip in clips)
        rms = np.sqrt(total_sum_squares / total_n)
        if peak > 0 and rms > 0:
            result["qgain_crest_factor_db"] = float(20 * np.log10(peak / rms))
    return result


def reverberation_tail_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    primary_speech: list[Interval],
    tail_window_sec: float = 0.5,
    minimum_offsets: int = 3,
) -> dict[str, float | str | int]:
    result: dict[str, float | str | int] = {
        "qrev_status": "ok",
        "qrev_valid_offset_count": 0,
        "qrev_tail_excess_db": np.nan,
        "qrev_decay_time_proxy_sec": np.nan,
        "qrev_decay_slope_db_per_sec": np.nan,
        "qrev_srmr": np.nan,
    }
    tails = []
    decay_times = []
    slopes = []
    frame_len = max(1, int(round(0.02 * sample_rate)))
    hop = max(1, int(round(0.01 * sample_rate)))
    for left, right in zip(primary_speech[:-1], primary_speech[1:]):
        pause = right.start_sec - left.end_sec
        if pause < tail_window_sec:
            continue
        start = int(round(left.end_sec * sample_rate))
        end = min(len(waveform), start + int(round(tail_window_sec * sample_rate)))
        clip = waveform[start:end]
        if len(clip) < frame_len:
            continue
        levels = np.asarray(
            [dbfs_rms(clip[index : index + frame_len]) for index in range(0, len(clip) - frame_len + 1, hop)]
        )
        times = (np.arange(len(levels)) * hop + frame_len / 2) / sample_rate
        early_mask = (times >= 0.03) & (times <= 0.10)
        # Estimate the pause floor at the end of the entire inter-speech pause,
        # not at the end of the 500-ms tail window. Otherwise slow tails raise
        # their own reference floor and paradoxically appear less reverberant.
        floor_start = max(left.end_sec, right.start_sec - 0.15)
        floor_end = right.start_sec
        _, floor_levels, _ = _frame_levels(
            waveform,
            sample_rate,
            [Interval(floor_start, floor_end)],
            frame_ms=20,
            hop_ms=10,
        )
        if len(floor_levels) < 3 or early_mask.sum() < 3:
            continue
        floor = float(np.median(floor_levels))
        early = float(np.median(levels[early_mask]))
        tails.append(max(0.0, early - floor))

        within_floor = levels <= floor + 3.0
        decay = tail_window_sec
        for idx in range(0, len(within_floor) - 2):
            if bool(np.all(within_floor[idx : idx + 3])):
                decay = float(times[idx])
                break
        decay_times.append(decay)

        slope_mask = (times >= 0.03) & (times <= 0.30) & (levels > floor + 1.0)
        if slope_mask.sum() >= 5:
            slopes.append(float(stats.theilslopes(levels[slope_mask], times[slope_mask]).slope))

    result["qrev_valid_offset_count"] = len(tails)
    if len(tails) < minimum_offsets:
        result["qrev_status"] = "insufficient_boundary_support"
        return result
    result["qrev_tail_excess_db"] = float(np.median(tails))
    result["qrev_decay_time_proxy_sec"] = float(np.median(decay_times))
    if slopes:
        result["qrev_decay_slope_db_per_sec"] = float(np.median(slopes))
    return result


def optional_srmr(waveform: np.ndarray, sample_rate: int) -> tuple[float, str]:
    """Compute SRMR when the optional validated dependency stack is available."""
    if sample_rate not in {8000, 16000}:
        return np.nan, "unsupported_sample_rate"
    try:
        import torch
        from torchmetrics.functional.audio.srmr import (
            speech_reverberation_modulation_energy_ratio,
        )

        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
        value = speech_reverberation_modulation_energy_ratio(
            preds=tensor,
            fs=sample_rate,
            fast=False,
            norm=False,
        )
        numeric = float(np.asarray(value.detach().cpu(), dtype=float).mean())
        return (numeric, "ok") if np.isfinite(numeric) else (np.nan, "nonfinite")
    except ImportError:
        return np.nan, "optional_dependency_unavailable"
    except Exception as exc:
        return np.nan, f"failed:{type(exc).__name__}"


def channel_device_metrics(
    waveform: np.ndarray, sample_rate: int, *, strict_speech: list[Interval]
) -> dict[str, float | str]:
    result: dict[str, float | str] = {
        "qchan_status": "ok",
        "qchan_effective_bandwidth_hz": np.nan,
        "qchan_highband_ratio": np.nan,
        "qchan_spectral_tilt_db_per_oct": np.nan,
    }
    if _total_duration(strict_speech) < 3:
        result["qchan_status"] = "insufficient_support"
        return result
    freqs, psd = _mean_interval_psd(waveform, sample_rate, strict_speech)
    if freqs is None:
        result["qchan_status"] = "insufficient_support"
        return result
    valid = (freqs >= 100) & (freqs <= sample_rate / 2)
    valid_freqs = freqs[valid]
    valid_psd = psd[valid]
    cumulative = np.cumsum(valid_psd)
    if cumulative[-1] > 0:
        index = int(np.searchsorted(cumulative, 0.99 * cumulative[-1]))
        result["qchan_effective_bandwidth_hz"] = float(valid_freqs[min(index, len(valid_freqs) - 1)])
        high = valid_freqs >= 3000
        result["qchan_highband_ratio"] = float(np.sum(valid_psd[high]) / np.sum(valid_psd))
    tilt = (valid_freqs >= 300) & (valid_freqs <= min(7000, sample_rate / 2 - 1))
    if tilt.sum() >= 20:
        x = np.log2(valid_freqs[tilt] / 300)
        y = 10 * np.log10(valid_psd[tilt] + EPS)
        # Subsample to keep Theil-Sen tractable while retaining the entire frequency span.
        indices = np.linspace(0, len(x) - 1, min(300, len(x))).round().astype(int)
        result["qchan_spectral_tilt_db_per_oct"] = float(stats.theilslopes(y[indices], x[indices]).slope)
    return result


def _hard_clip_mask(samples: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return np.zeros(0, dtype=bool), []
    max_abs = float(np.max(np.abs(samples)))
    if max_abs < 0.20:
        return np.zeros(len(samples), dtype=bool), []
    edge = 0.995 * max_abs
    tolerance = max(2e-7, 2e-4 * max_abs)
    near_edge = np.abs(samples) >= edge
    flat = np.r_[False, np.abs(np.diff(samples)) <= tolerance]
    candidates = near_edge & flat
    runs = _count_runs(candidates, minimum_length=3)
    mask = np.zeros(len(samples), dtype=bool)
    for start, end in runs:
        mask[max(0, start - 1) : end] = True
    return mask, runs


def nonlinear_distortion_metrics(
    native: np.ndarray,
    sample_rate: int,
    *,
    strict_speech: list[Interval],
) -> dict[str, float | str | int]:
    speech_sec = _total_duration(strict_speech)
    result: dict[str, float | str | int] = {
        "qdist_status": "ok",
        "qdist_speech_support_sec": speech_sec,
        "qdist_hard_clip_sample_fraction": np.nan,
        "qdist_clip_event_rate_per_min": np.nan,
        "qdist_clipped_frame_fraction": np.nan,
        "qdist_near_fullscale_fraction": np.nan,
        "qdist_edge_histogram_spike": np.nan,
    }
    if speech_sec < 3:
        result["qdist_status"] = "insufficient_support"
        return result
    native_2d = native[:, None] if native.ndim == 1 else native
    channel_results = []
    for channel in range(native_2d.shape[1]):
        waveform = native_2d[:, channel]
        clips = _interval_slices(waveform, sample_rate, strict_speech)
        if not clips:
            continue
        pooled = np.concatenate(clips)
        clip_masks_and_events = [_hard_clip_mask(clip) for clip in clips]
        clipped_sample_count = sum(int(mask.sum()) for mask, _ in clip_masks_and_events)
        event_count = sum(len(events) for _, events in clip_masks_and_events)
        near_full = float(np.mean(np.abs(pooled) >= 0.98))

        abs_values = np.abs(pooled)
        maximum = float(np.max(abs_values))
        if maximum > 0:
            hist, _ = np.histogram(abs_values, bins=np.linspace(0, maximum, 101))
            reference = np.median(hist[-6:-1]) + 1
            edge_spike = float(hist[-1] / reference)
        else:
            edge_spike = 0.0

        frame_len = max(1, int(round(0.03 * sample_rate)))
        clipped_frames = []
        for clip in clips:
            for start in range(0, len(clip) - frame_len + 1, frame_len):
                frame_mask, _ = _hard_clip_mask(clip[start : start + frame_len])
                clipped_frames.append(bool(frame_mask.any()))
        channel_results.append(
            {
                "fraction": float(clipped_sample_count / len(pooled)),
                "events": event_count,
                "frames": float(np.mean(clipped_frames)) if clipped_frames else np.nan,
                "near": near_full,
                "spike": edge_spike,
            }
        )
    if not channel_results:
        result["qdist_status"] = "insufficient_support"
        return result
    result["qdist_hard_clip_sample_fraction"] = max(item["fraction"] for item in channel_results)
    result["qdist_clip_event_rate_per_min"] = max(item["events"] for item in channel_results) / speech_sec * 60
    result["qdist_clipped_frame_fraction"] = np.nanmax([item["frames"] for item in channel_results])
    result["qdist_near_fullscale_fraction"] = max(item["near"] for item in channel_results)
    result["qdist_edge_histogram_spike"] = max(item["spike"] for item in channel_results)
    return result


def temporal_discontinuity_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    strict_speech: list[Interval],
) -> dict[str, float | str]:
    speech_sec = _total_duration(strict_speech)
    result: dict[str, float | str] = {
        "qtemp_status": "ok",
        "qtemp_speech_support_sec": speech_sec,
        "qtemp_zero_dropout_fraction": np.nan,
        "qtemp_zero_dropout_rate_per_min": np.nan,
        "qtemp_duplicate_window_rate_per_min": np.nan,
        "qtemp_energy_jump_rate_per_min": np.nan,
        "qtemp_continuity_break_rate_per_min": np.nan,
    }
    clips = _interval_slices(waveform, sample_rate, strict_speech)
    if speech_sec < 3 or not clips:
        result["qtemp_status"] = "insufficient_support"
        return result

    total_samples = sum(len(clip) for clip in clips)
    pooled_rms = np.sqrt(sum(float(np.sum(clip.astype(np.float64) ** 2)) for clip in clips) / total_samples)
    zero_threshold = max(1e-7, min(1e-5, pooled_rms * 1e-3))
    minimum_zero = max(1, int(round(0.010 * sample_rate)))
    zero_samples = 0
    zero_events = 0
    duplicate_events = 0
    energy_jumps = 0
    continuity_breaks = 0
    window = max(16, int(round(0.020 * sample_rate)))
    frame = max(16, int(round(0.030 * sample_rate)))

    for clip in clips:
        zero_runs = _count_runs(np.abs(clip) <= zero_threshold, minimum_length=minimum_zero)
        zero_events += len(zero_runs)
        zero_samples += sum(end - start for start, end in zero_runs)

        in_duplicate_run = False
        for start in range(0, len(clip) - 2 * window + 1, window):
            first = clip[start : start + window]
            second = clip[start + window : start + 2 * window]
            rms1 = np.sqrt(np.mean(first.astype(float) ** 2))
            rms2 = np.sqrt(np.mean(second.astype(float) ** 2))
            if min(rms1, rms2) < max(1e-5, pooled_rms * 0.02):
                in_duplicate_run = False
                continue
            nrmse = np.sqrt(np.mean((first - second).astype(float) ** 2)) / (rms1 + rms2 + EPS)
            corr = np.corrcoef(first, second)[0, 1] if np.std(first) > EPS and np.std(second) > EPS else 0
            duplicate = bool(nrmse <= 1e-4 and np.isfinite(corr) and corr >= 0.999999)
            if duplicate and not in_duplicate_run:
                duplicate_events += 1
            in_duplicate_run = duplicate

        levels = np.asarray(
            [dbfs_rms(clip[start : start + frame]) for start in range(0, len(clip) - frame + 1, frame)]
        )
        if len(levels) >= 2:
            energy_jumps += int(np.sum(np.abs(np.diff(levels)) >= 18.0))

        if len(clip) >= 20:
            differences = np.abs(np.diff(clip.astype(float)))
            median = np.median(differences)
            mad = np.median(np.abs(differences - median))
            if mad > EPS:
                robust_z = (differences - median) / (1.4826 * mad)
                continuity_breaks += int(np.sum(robust_z >= 25.0))

    result["qtemp_zero_dropout_fraction"] = float(zero_samples / total_samples)
    result["qtemp_zero_dropout_rate_per_min"] = float(zero_events / speech_sec * 60)
    result["qtemp_duplicate_window_rate_per_min"] = float(duplicate_events / speech_sec * 60)
    result["qtemp_energy_jump_rate_per_min"] = float(energy_jumps / speech_sec * 60)
    result["qtemp_continuity_break_rate_per_min"] = float(continuity_breaks / speech_sec * 60)
    return result


def extract_all_metrics(audio: AudioViews, views: dict[str, list[Interval]]) -> dict:
    """Extract all families while preserving family-specific support/status fields."""
    strict_speech = views["strict_speech"]
    strict_nonspeech = views["strict_internal_nonspeech"]
    output = {
        "native_sample_rate_hz": audio.sample_rate_native,
        "native_channels": int(audio.native.shape[1]),
        "native_codec": audio.probe.get("codec_name"),
        "decode_warning": audio.decode_stderr,
    }
    output.update(
        additive_interference_metrics(
            audio.analysis_16k,
            16000,
            strict_speech=strict_speech,
            strict_internal_nonspeech=strict_nonspeech,
        )
    )
    output.update(gain_dynamics_metrics(audio.analysis_16k, 16000, strict_speech=strict_speech))
    reverb = reverberation_tail_metrics(
        audio.analysis_16k, 16000, primary_speech=views["primary_speech"]
    )
    reverb["qrev_srmr"], reverb["qrev_srmr_status"] = optional_srmr(audio.analysis_16k, 16000)
    output.update(reverb)
    output.update(
        channel_device_metrics(
            audio.mono_native, audio.sample_rate_native, strict_speech=strict_speech
        )
    )
    output.update(
        nonlinear_distortion_metrics(
            audio.native, audio.sample_rate_native, strict_speech=strict_speech
        )
    )
    output.update(
        temporal_discontinuity_metrics(
            audio.mono_native, audio.sample_rate_native, strict_speech=strict_speech
        )
    )
    return output
