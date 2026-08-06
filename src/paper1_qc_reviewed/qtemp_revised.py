"""Experimental waveform anomaly features for the QTEMP revised family.

This module preserves the original QTEMP implementation.  It implements a
transparent derivative detector for abrupt sample discontinuities and a
short-time RMS detector for bracketed digital-silence/energy-collapse runs.
The outputs describe decoded-waveform anomalies, not their transport cause.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MEASUREMENT_VERSION = "qtemp-revised-v0.1.0-experimental"

ANALYSIS_FEATURES = (
    "qtemp_revised_glitch_rate_per_min",
    "qtemp_revised_glitch_peak_robust_z",
    "qtemp_revised_dropout_event_rate_per_min",
    "qtemp_revised_dropout_duration_fraction",
)


@dataclass(frozen=True)
class QTEMPRevisedParameters:
    derivative_robust_z: float = 6.0
    glitch_refractory_ms: float = 2.0
    edge_guard_ms: float = 100.0
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    dropout_min_ms: float = 20.0
    dropout_max_ms: float = 1000.0
    dropout_absolute_rms: float = 1e-4
    dropout_relative_rms: float = 0.02
    active_context_rms: float = 5e-4
    context_ms: float = 100.0
    digital_zero_fraction: float = 0.98
    digital_zero_abs: float = 1e-7
    minimum_duration_sec: float = 1.0
    max_glitch_events_in_ledger: int = 200


DEFAULT_PARAMETERS = QTEMPRevisedParameters()


@dataclass
class QTEMPRevisedExtraction:
    recording: dict
    event_ledger: pd.DataFrame


def _float_channels(waveform: np.ndarray) -> np.ndarray:
    raw = np.asarray(waveform)
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.ndim != 2:
        raise ValueError("waveform must be samples or samples-by-channels")
    values = raw.astype(np.float64, copy=False)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        values = values / float(max(abs(info.min), abs(info.max)))
    return values


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True))


def _merge_times(times: Iterable[float], refractory_sec: float) -> list[float]:
    ordered = sorted(float(value) for value in times)
    merged: list[float] = []
    for value in ordered:
        if not merged or value - merged[-1] > refractory_sec:
            merged.append(value)
        elif value < merged[-1]:
            merged[-1] = value
    return merged


def extract_qtemp_revised(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    logical_recording_id: str | None = None,
    parameters: QTEMPRevisedParameters = DEFAULT_PARAMETERS,
) -> QTEMPRevisedExtraction:
    """Extract experimental glitch and bracketed-dropout measurements."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = _float_channels(waveform)
    duration_sec = values.shape[0] / float(sample_rate)
    base = {
        "logical_recording_id": logical_recording_id,
        "qtemp_revised_measurement_version": MEASUREMENT_VERSION,
        "qtemp_revised_sample_rate_hz": int(sample_rate),
        "qtemp_revised_channel_count": int(values.shape[1]),
        "qtemp_revised_duration_sec": duration_sec,
    }
    if duration_sec < parameters.minimum_duration_sec or not np.isfinite(values).all():
        recording = {**base, "qtemp_revised_status": "unavailable"}
        recording.update({name: np.nan for name in ANALYSIS_FEATURES})
        return QTEMPRevisedExtraction(recording, pd.DataFrame())

    guard = max(1, round(parameters.edge_guard_ms * sample_rate / 1000.0))
    start, stop = guard, values.shape[0] - guard
    if stop <= start:
        recording = {**base, "qtemp_revised_status": "unavailable"}
        recording.update({name: np.nan for name in ANALYSIS_FEATURES})
        return QTEMPRevisedExtraction(recording, pd.DataFrame())

    eligible_sec = (stop - start) / float(sample_rate)
    events: list[dict] = []
    glitch_candidates: list[dict] = []
    peak_z = 0.0

    for channel_index in range(values.shape[1]):
        channel = values[:, channel_index]
        differences = np.diff(channel)
        core = differences[start : max(start, stop - 1)]
        center = float(np.median(core))
        mad = float(np.median(np.abs(core - center)))
        scale = max(1e-12, 1.4826 * mad)
        robust_z = np.abs((core - center) / scale)
        peak_z = max(peak_z, float(np.max(robust_z, initial=0.0)))
        indices = np.flatnonzero(robust_z > parameters.derivative_robust_z) + start
        for index in indices:
            time_sec = (index + 1) / float(sample_rate)
            glitch_candidates.append({
                "event_type": "glitch",
                "channel_index": channel_index,
                "start_sec": time_sec,
                "end_sec": time_sec,
                "duration_sec": 0.0,
                "score": float(abs((differences[index] - center) / scale)),
                "reason": "sample_difference_above_robust_threshold",
            })

    refractory_sec = parameters.glitch_refractory_ms / 1000.0
    merged_glitches: list[dict] = []
    for candidate in sorted(glitch_candidates, key=lambda row: row["start_sec"]):
        if not merged_glitches or candidate["start_sec"] - merged_glitches[-1]["start_sec"] > refractory_sec:
            merged_glitches.append(candidate)
        elif candidate["score"] > merged_glitches[-1]["score"]:
            merged_glitches[-1] = candidate

    # Retain the full event count as the feature, but bound the review ledger.
    # A permissive derivative threshold can otherwise create multi-gigabyte
    # ledgers on ordinary speech. The highest-scoring events are most useful for
    # listening review and the truncation flag makes the sampling explicit.
    reviewed_glitches = sorted(
        merged_glitches, key=lambda row: row["score"], reverse=True
    )[: parameters.max_glitch_events_in_ledger]
    events.extend(reviewed_glitches)

    # Dropout detection uses the maximum channel RMS so one silent channel in a
    # valid stereo recording cannot create a false recording-level dropout.
    frame_n = max(1, round(parameters.frame_ms * sample_rate / 1000.0))
    hop_n = max(1, round(parameters.hop_ms * sample_rate / 1000.0))
    starts = np.arange(start, max(start, stop - frame_n + 1), hop_n, dtype=int)
    squared_cumsum = np.vstack(
        [np.zeros(values.shape[1]), np.cumsum(values * values, axis=0)]
    )
    zero_cumsum = np.vstack([
        np.zeros(values.shape[1]),
        np.cumsum(np.abs(values) <= parameters.digital_zero_abs, axis=0),
    ])
    frame_energy = squared_cumsum[starts + frame_n] - squared_cumsum[starts]
    frame_zeros = zero_cumsum[starts + frame_n] - zero_cumsum[starts]
    rms = np.max(np.sqrt(frame_energy / frame_n), axis=1)
    zero_fraction = np.min(frame_zeros / frame_n, axis=1)

    positive_rms = rms[rms > parameters.dropout_absolute_rms]
    baseline = float(np.median(positive_rms)) if positive_rms.size else 0.0
    dropout_threshold = max(
        parameters.dropout_absolute_rms,
        parameters.dropout_relative_rms * baseline,
    )
    candidates = (rms <= dropout_threshold) | (zero_fraction >= parameters.digital_zero_fraction)
    context_frames = max(1, round(parameters.context_ms / parameters.hop_ms))
    dropout_events: list[dict] = []
    for run_start, run_end in _runs(candidates):
        event_start = starts[run_start] / float(sample_rate)
        event_end = (starts[run_end - 1] + frame_n) / float(sample_rate)
        event_ms = (event_end - event_start) * 1000.0
        left = rms[max(0, run_start - context_frames) : run_start]
        right = rms[run_end : min(len(rms), run_end + context_frames)]
        bracketed = (
            left.size > 0
            and right.size > 0
            and float(np.max(left)) >= parameters.active_context_rms
            and float(np.max(right)) >= parameters.active_context_rms
        )
        if not bracketed or not (parameters.dropout_min_ms <= event_ms <= parameters.dropout_max_ms):
            continue
        row = {
            "event_type": "dropout",
            "channel_index": -1,
            "start_sec": event_start,
            "end_sec": event_end,
            "duration_sec": event_end - event_start,
            "score": float(dropout_threshold / max(1e-12, np.max(rms[run_start:run_end]))),
            "reason": "bracketed_short_time_rms_collapse",
        }
        dropout_events.append(row)
        events.append(row)

    dropout_duration = sum(row["duration_sec"] for row in dropout_events)
    recording = {
        **base,
        "qtemp_revised_status": "measured",
        "qtemp_revised_eligible_duration_sec": eligible_sec,
        "qtemp_revised_glitch_count": len(merged_glitches),
        "qtemp_revised_glitch_ledger_count": len(reviewed_glitches),
        "qtemp_revised_glitch_ledger_truncated": len(merged_glitches) > len(reviewed_glitches),
        "qtemp_revised_dropout_count": len(dropout_events),
        "qtemp_revised_dropout_rms_threshold": dropout_threshold,
        "qtemp_revised_glitch_rate_per_min": len(merged_glitches) * 60.0 / eligible_sec,
        "qtemp_revised_glitch_peak_robust_z": peak_z,
        "qtemp_revised_dropout_event_rate_per_min": len(dropout_events) * 60.0 / eligible_sec,
        "qtemp_revised_dropout_duration_fraction": dropout_duration / eligible_sec,
    }
    ledger = pd.DataFrame(events).sort_values(
        ["start_sec", "event_type"], ignore_index=True
    ) if events else pd.DataFrame(columns=[
        "event_type", "channel_index", "start_sec", "end_sec",
        "duration_sec", "score", "reason",
    ])
    return QTEMPRevisedExtraction(recording, ledger)


def parameter_frame(parameters: QTEMPRevisedParameters = DEFAULT_PARAMETERS) -> pd.DataFrame:
    return pd.DataFrame([asdict(parameters)])
