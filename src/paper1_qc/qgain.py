"""QGAIN v3.1: recorded-level and level-dynamics measurements.

The module contains the authoritative estimators.  The companion notebook calls
these functions; it does not duplicate feature algorithms.

QGAIN describes recorded level behaviour. It does not identify automatic gain
control, microphone motion, vocal effort, or any other unique cause.

QGAIN v3.1 contains four analysis features. The v3.0 local transition detector
is retained only to reproduce the negative validation result that led to its
exclusion. Its outputs are explicitly exploratory and must not be used as
recording-quality features.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import log10, sqrt

import numpy as np
import pandas as pd
from scipy import stats

MEASUREMENT_VERSION = "qgain-v3.1.0"

ANALYSIS_FEATURES = (
    "qgain_typical_speech_level_dbfs",
    "qgain_within_segment_iqr_db",
    "qgain_between_segment_mad_db",
    "qgain_abs_drift_db_per_min",
)

PRIMARY_FEATURES = (
    "qgain_within_segment_iqr_db",
    "qgain_between_segment_mad_db",
    "qgain_abs_drift_db_per_min",
)


@dataclass(frozen=True)
class TimeInterval:
    """Half-open interval in original recording seconds."""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


@dataclass(frozen=True)
class QGAINParameters:
    """Frozen QGAIN v3 engineering and support parameters."""

    analysis_sample_rate_hz: int = 16_000
    frame_length_ms: float = 40.0
    frame_hop_ms: float = 10.0
    speech_edge_guard_ms: float = 200.0
    dbfs_floor_db: float = -120.0
    maximum_floor_frame_fraction: float = 0.02
    minimum_speech_support_sec: float = 1.0
    minimum_valid_frame_count: int = 25
    minimum_segment_frame_count: int = 5
    between_minimum_segment_count: int = 3
    drift_minimum_segment_count: int = 3
    drift_minimum_span_sec: float = 10.0
    step_context_ms: float = 250.0
    step_minimum_amplitude_db: float = 6.0
    step_minimum_standardized_effect: float = 3.0
    step_refractory_ms: float = 750.0
    moderate_speech_support_sec: float = 3.0
    high_speech_support_sec: float = 5.0
    moderate_segment_count: int = 5
    high_segment_count: int = 8
    moderate_drift_span_sec: float = 20.0
    high_drift_span_sec: float = 30.0
    random_seed: int = 20260729

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMETERS = QGAINParameters()


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    display_name: str
    subdomain: str
    role: str
    unit: str
    estimand: str
    orientation: str
    claim_boundary: str
    minimum_support: str
    known_confounds: str


FEATURE_DEFINITIONS = (
    FeatureDefinition(
        "qgain_typical_speech_level_dbfs",
        "Typical strict-speech operating level",
        "operating level",
        "contextual",
        "dBFS",
        "Median AC-RMS level of guarded strict-speech frames.",
        "Higher is closer to digital full scale; not ordinal quality.",
        "Not ITU-T P.56 active speech level and not calibrated vocal intensity.",
        "At least 1.0 s guarded speech and 25 non-floor frames.",
        "Vocal intensity, dysarthria, distance, task, device gain, and AGC.",
    ),
    FeatureDefinition(
        "qgain_within_segment_iqr_db",
        "Within-segment centered level IQR",
        "short-term dynamics",
        "primary",
        "dB",
        "IQR after subtracting each segment's median frame level.",
        "Higher indicates greater within-segment level dispersion.",
        "Compatible with gain modulation but not specific to AGC.",
        "At least 1.0 s guarded speech, 25 frames, and one usable segment.",
        "Prosody, phonetic composition, respiratory control, dysarthria, segmentation.",
    ),
    FeatureDefinition(
        "qgain_between_segment_mad_db",
        "Between-segment level MAD",
        "segment dynamics",
        "primary",
        "dB",
        "1.4826 times MAD of usable segment-median levels.",
        "Higher indicates larger segment-level shifts.",
        "Within-recording variability; not a pure device-gain estimate.",
        "At least three usable speech segments.",
        "Sentence content, fatigue, posture, distance, task, unequal segment duration.",
    ),
    FeatureDefinition(
        "qgain_abs_drift_db_per_min",
        "Absolute robust level drift",
        "slow dynamics",
        "primary",
        "dB/min",
        "Absolute Theil-Sen slope of segment level against original time.",
        "Higher indicates faster gradual level change.",
        "Compatible with changing gain or distance but not source-identifiable.",
        "At least three segments spanning at least 10 s.",
        "Fatigue, intentional loudness, respiration, posture, and task order.",
    ),
)


@dataclass
class QGAINExtraction:
    recording: dict
    frame_ledger: pd.DataFrame
    segment_ledger: pd.DataFrame
    event_ledger: pd.DataFrame


def feature_registry_frame() -> pd.DataFrame:
    """Return the immutable, one-row-per-analysis-feature registry."""

    return pd.DataFrame([asdict(item) for item in FEATURE_DEFINITIONS])


def merge_intervals(intervals: Iterable[TimeInterval]) -> list[TimeInterval]:
    valid = sorted(
        (
            TimeInterval(float(item.start_sec), float(item.end_sec))
            for item in intervals
            if np.isfinite(item.start_sec)
            and np.isfinite(item.end_sec)
            and item.end_sec > item.start_sec
        ),
        key=lambda item: (item.start_sec, item.end_sec),
    )
    merged: list[TimeInterval] = []
    for item in valid:
        if not merged or item.start_sec > merged[-1].end_sec:
            merged.append(item)
        else:
            merged[-1] = TimeInterval(merged[-1].start_sec, max(merged[-1].end_sec, item.end_sec))
    return merged


def guarded_speech_intervals(
    intervals: Iterable[TimeInterval],
    duration_sec: float,
    *,
    parameters: QGAINParameters = DEFAULT_PARAMETERS,
    guard_ms: float | None = None,
) -> list[TimeInterval]:
    """Merge, clip, and symmetrically erode every speech interval."""

    guard_sec = (parameters.speech_edge_guard_ms if guard_ms is None else float(guard_ms)) / 1000.0
    output = []
    for item in merge_intervals(intervals):
        start = max(0.0, item.start_sec + guard_sec)
        end = min(float(duration_sec), item.end_sec - guard_sec)
        if end > start:
            output.append(TimeInterval(start, end))
    return output


def ac_rms_dbfs(samples: np.ndarray, floor_db: float = -120.0) -> tuple[float, bool]:
    """Return frame AC-RMS dBFS and whether it is at the computational floor."""

    values = np.asarray(samples, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        return np.nan, False
    centered = values - float(np.mean(values))
    rms = float(sqrt(float(np.mean(centered * centered))))
    if rms <= 0.0 or not np.isfinite(rms):
        return float(floor_db), True
    return float(max(floor_db, 20.0 * log10(rms))), False


def _support_tier(
    feature: str,
    *,
    speech_sec: float,
    frame_count: int,
    segment_count: int,
    span_sec: float,
    parameters: QGAINParameters,
) -> str:
    if feature == "qgain_between_segment_mad_db":
        if segment_count >= parameters.high_segment_count:
            return "high"
        if segment_count >= parameters.moderate_segment_count:
            return "moderate"
        if segment_count >= parameters.between_minimum_segment_count:
            return "minimum"
        return "unavailable"
    if feature == "qgain_abs_drift_db_per_min":
        if (
            segment_count >= parameters.high_segment_count
            and span_sec >= parameters.high_drift_span_sec
        ):
            return "high"
        if (
            segment_count >= parameters.moderate_segment_count
            and span_sec >= parameters.moderate_drift_span_sec
        ):
            return "moderate"
        if (
            segment_count >= parameters.drift_minimum_segment_count
            and span_sec >= parameters.drift_minimum_span_sec
        ):
            return "minimum"
        return "unavailable"
    if (
        speech_sec >= parameters.high_speech_support_sec
        and frame_count >= 200
        and segment_count >= 3
    ):
        return "high"
    if speech_sec >= parameters.moderate_speech_support_sec and frame_count >= 100:
        return "moderate"
    if (
        speech_sec >= parameters.minimum_speech_support_sec
        and frame_count >= parameters.minimum_valid_frame_count
    ):
        return "minimum"
    return "unavailable"


def _frame_speech(
    waveform: np.ndarray,
    fs: int,
    intervals: list[TimeInterval],
    logical_recording_id: str,
    parameters: QGAINParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_length = round(parameters.frame_length_ms * fs / 1000.0)
    hop = round(parameters.frame_hop_ms * fs / 1000.0)
    frame_rows: list[dict] = []
    segment_rows: list[dict] = []
    for segment_id, interval in enumerate(intervals):
        left = max(0, round(interval.start_sec * fs))
        right = min(len(waveform), round(interval.end_sec * fs))
        local_levels = []
        local_floor_count = 0
        if right - left >= frame_length:
            for frame_left in range(left, right - frame_length + 1, hop):
                frame_right = frame_left + frame_length
                level, at_floor = ac_rms_dbfs(
                    waveform[frame_left:frame_right], parameters.dbfs_floor_db
                )
                valid = bool(np.isfinite(level) and not at_floor)
                frame_rows.append(
                    {
                        "logical_recording_id": logical_recording_id,
                        "segment_id": int(segment_id),
                        "segment_start_sec": interval.start_sec,
                        "segment_end_sec": interval.end_sec,
                        "frame_start_sec": frame_left / fs,
                        "frame_end_sec": frame_right / fs,
                        "frame_mid_sec": (frame_left + frame_right) / (2.0 * fs),
                        "ac_rms_dbfs": level,
                        "at_digital_floor": at_floor,
                        "valid_level_frame": valid,
                    }
                )
                if at_floor:
                    local_floor_count += 1
                if valid:
                    local_levels.append(level)
        levels = np.asarray(local_levels, dtype=float)
        segment_rows.append(
            {
                "logical_recording_id": logical_recording_id,
                "segment_id": int(segment_id),
                "segment_start_sec": interval.start_sec,
                "segment_end_sec": interval.end_sec,
                "segment_mid_sec": (interval.start_sec + interval.end_sec) / 2.0,
                "guarded_segment_duration_sec": interval.duration_sec,
                "frame_count_total": int(
                    sum(row["segment_id"] == segment_id for row in frame_rows)
                ),
                "frame_count_nonfloor": len(levels),
                "floor_frame_count": int(local_floor_count),
                "segment_level_median_dbfs": (
                    float(np.median(levels))
                    if len(levels) >= parameters.minimum_segment_frame_count
                    else np.nan
                ),
                "segment_level_iqr_db": (
                    float(np.quantile(levels, 0.75) - np.quantile(levels, 0.25))
                    if len(levels) >= parameters.minimum_segment_frame_count
                    else np.nan
                ),
                "usable_segment": bool(len(levels) >= parameters.minimum_segment_frame_count),
            }
        )
    return pd.DataFrame(frame_rows), pd.DataFrame(segment_rows)


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def detect_exploratory_local_transitions(
    frame_ledger: pd.DataFrame,
    *,
    logical_recording_id: str,
    parameters: QGAINParameters = DEFAULT_PARAMETERS,
) -> tuple[pd.DataFrame, float]:
    """Detect local context-median transitions for audit only.

    Each candidate must have complete pre/post contexts, exceed both the
    absolute 6-dB engineering threshold and an adaptive local-variation
    threshold, and survive non-maximum suppression. Real-speech validation in
    QGAIN v3.0 showed an unacceptable false-positive burden from phonetic and
    prosodic transitions. The detector is therefore excluded from ANALYSIS_FEATURES.
    """

    context_frames = max(1, round(parameters.step_context_ms / parameters.frame_hop_ms))
    refractory_frames = max(1, round(parameters.step_refractory_ms / parameters.frame_hop_ms))
    candidate_rows: list[dict] = []
    evaluated_centers = 0
    for segment_id, local in frame_ledger.groupby("segment_id", sort=True):
        local = local.loc[local["valid_level_frame"].astype(bool)].sort_values("frame_mid_sec")
        local = local.reset_index(drop=True)
        levels = pd.to_numeric(local["ac_rms_dbfs"], errors="coerce").to_numpy(float)
        if len(levels) < 2 * context_frames:
            continue
        segment_candidates = []
        for split in range(context_frames, len(levels) - context_frames + 1):
            pre = levels[split - context_frames : split]
            post = levels[split : split + context_frames]
            if len(pre) != context_frames or len(post) != context_frames:
                continue
            evaluated_centers += 1
            pre_median = float(np.median(pre))
            post_median = float(np.median(post))
            delta = post_median - pre_median
            residual = np.concatenate([pre - pre_median, post - post_median])
            local_scale = max(0.25, _robust_scale(residual))
            adaptive_threshold = max(
                parameters.step_minimum_amplitude_db,
                parameters.step_minimum_standardized_effect * local_scale,
            )
            if abs(delta) < adaptive_threshold:
                continue
            segment_candidates.append(
                {
                    "logical_recording_id": logical_recording_id,
                    "segment_id": int(segment_id),
                    "candidate_index": int(split),
                    "event_time_sec": float(local.iloc[split]["frame_start_sec"]),
                    "pre_context_start_sec": float(
                        local.iloc[split - context_frames]["frame_start_sec"]
                    ),
                    "post_context_end_sec": float(
                        local.iloc[split + context_frames - 1]["frame_end_sec"]
                    ),
                    "pre_level_dbfs": pre_median,
                    "post_level_dbfs": post_median,
                    "step_delta_db": delta,
                    "absolute_step_db": abs(delta),
                    "local_residual_scale_db": local_scale,
                    "adaptive_threshold_db": adaptive_threshold,
                    "standardized_effect": abs(delta) / local_scale,
                }
            )
        accepted: list[dict] = []
        for candidate in sorted(
            segment_candidates, key=lambda row: row["absolute_step_db"], reverse=True
        ):
            if all(
                abs(candidate["candidate_index"] - prior["candidate_index"]) >= refractory_frames
                for prior in accepted
            ):
                accepted.append(candidate)
        for event_index, candidate in enumerate(
            sorted(accepted, key=lambda row: row["event_time_sec"])
        ):
            candidate["event_id"] = (
                f"{logical_recording_id}:segment-{int(segment_id):04d}:event-{event_index:04d}"
            )
            candidate["event_status"] = "exploratory_rejected_not_analysis"
            candidate_rows.append(candidate)
    exposure_sec = evaluated_centers * parameters.frame_hop_ms / 1000.0
    columns = [
        "logical_recording_id",
        "event_id",
        "segment_id",
        "event_time_sec",
        "pre_context_start_sec",
        "post_context_end_sec",
        "pre_level_dbfs",
        "post_level_dbfs",
        "step_delta_db",
        "absolute_step_db",
        "local_residual_scale_db",
        "adaptive_threshold_db",
        "standardized_effect",
        "event_status",
    ]
    return pd.DataFrame(candidate_rows, columns=columns), float(exposure_sec)


def poisson_rate_interval(
    count: int, exposure_sec: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Exact central Poisson count interval converted to events/minute."""

    if exposure_sec <= 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence
    lower_count = 0.0 if count == 0 else 0.5 * stats.chi2.ppf(alpha / 2, 2 * count)
    upper_count = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (count + 1))
    factor = 60.0 / exposure_sec
    return float(lower_count * factor), float(upper_count * factor)


def extract_qgain(
    waveform: np.ndarray,
    fs: int,
    *,
    strict_speech: Iterable[TimeInterval],
    logical_recording_id: str = "recording",
    parameters: QGAINParameters = DEFAULT_PARAMETERS,
    guard_ms: float | None = None,
) -> QGAINExtraction:
    """Extract the four QGAIN v3.1 features and supporting audit ledgers."""

    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("QGAIN requires a mono analysis waveform.")
    if int(fs) != parameters.analysis_sample_rate_hz:
        raise ValueError(
            f"QGAIN requires {parameters.analysis_sample_rate_hz} Hz analysis audio; got {fs}."
        )
    if not np.isfinite(samples).all():
        raise ValueError("Waveform contains non-finite samples.")
    duration_sec = len(samples) / float(fs)
    guarded = guarded_speech_intervals(
        strict_speech,
        duration_sec,
        parameters=parameters,
        guard_ms=guard_ms,
    )
    speech_support_sec = float(sum(item.duration_sec for item in guarded))
    frames, segments = _frame_speech(samples, fs, guarded, logical_recording_id, parameters)
    if len(frames):
        valid_frames = frames.loc[frames["valid_level_frame"].astype(bool)].copy()
        frame_count_total = len(frames)
        frame_count_nonfloor = len(valid_frames)
        floor_fraction = float(frames["at_digital_floor"].astype(bool).mean())
    else:
        valid_frames = pd.DataFrame(columns=frames.columns)
        frame_count_total = 0
        frame_count_nonfloor = 0
        floor_fraction = np.nan
    usable_segments = (
        segments.loc[segments["usable_segment"].astype(bool)].copy()
        if len(segments)
        else pd.DataFrame(columns=segments.columns)
    )
    segment_levels = pd.to_numeric(
        usable_segments.get("segment_level_median_dbfs"), errors="coerce"
    ).dropna()
    segment_times = pd.to_numeric(usable_segments.get("segment_mid_sec"), errors="coerce").loc[
        segment_levels.index
    ]
    segment_count = len(segment_levels)
    span_sec = float(segment_times.max() - segment_times.min()) if segment_count >= 2 else 0.0
    floor_censored = bool(
        np.isfinite(floor_fraction) and floor_fraction > parameters.maximum_floor_frame_fraction
    )
    basic_support = bool(
        speech_support_sec >= parameters.minimum_speech_support_sec
        and frame_count_nonfloor >= parameters.minimum_valid_frame_count
        and segment_count >= 1
        and not floor_censored
    )

    raw_typical = np.nan
    raw_within = np.nan
    if frame_count_nonfloor:
        raw_typical = float(np.median(valid_frames["ac_rms_dbfs"]))
        centered = []
        for _, local in valid_frames.groupby("segment_id"):
            levels = pd.to_numeric(local["ac_rms_dbfs"], errors="coerce").dropna()
            if len(levels):
                centered.extend((levels - levels.median()).tolist())
        if centered:
            raw_within = float(np.quantile(centered, 0.75) - np.quantile(centered, 0.25))

    raw_between = np.nan
    if segment_count >= 2:
        raw_between = _robust_scale(segment_levels.to_numpy(float))

    signed_drift = np.nan
    raw_abs_drift = np.nan
    drift_ci_low = np.nan
    drift_ci_high = np.nan
    if (
        segment_count >= parameters.drift_minimum_segment_count
        and span_sec >= parameters.drift_minimum_span_sec
    ):
        slope = stats.theilslopes(
            segment_levels.to_numpy(float), segment_times.to_numpy(float), alpha=0.95
        )
        signed_drift = float(slope.slope * 60.0)
        raw_abs_drift = abs(signed_drift)
        drift_ci_low = float(slope.low_slope * 60.0)
        drift_ci_high = float(slope.high_slope * 60.0)

    event_ledger, transition_exposure_sec = detect_exploratory_local_transitions(
        valid_frames,
        logical_recording_id=logical_recording_id,
        parameters=parameters,
    )
    transition_count = len(event_ledger)
    exploratory_transition_rate = (
        float(transition_count * 60.0 / transition_exposure_sec)
        if transition_exposure_sec > 0
        else np.nan
    )
    transition_ci_low, transition_ci_high = poisson_rate_interval(
        transition_count, transition_exposure_sec
    )

    values = {
        "qgain_typical_speech_level_dbfs": raw_typical if basic_support else np.nan,
        "qgain_within_segment_iqr_db": raw_within if basic_support else np.nan,
        "qgain_between_segment_mad_db": (
            raw_between
            if segment_count >= parameters.between_minimum_segment_count and not floor_censored
            else np.nan
        ),
        "qgain_abs_drift_db_per_min": (raw_abs_drift if not floor_censored else np.nan),
    }
    recording = {
        "logical_recording_id": logical_recording_id,
        "qgain_measurement_version": MEASUREMENT_VERSION,
        "qgain_signal_view": "mono_dc_preserved_16k_analysis; framewise_AC_RMS",
        "qgain_speech_source": "frozen_strict_speech",
        "qgain_speech_edge_guard_ms": (
            parameters.speech_edge_guard_ms if guard_ms is None else float(guard_ms)
        ),
        "qgain_guarded_speech_interval_count": len(guarded),
        "qgain_guarded_speech_support_sec": speech_support_sec,
        "qgain_frame_count_total": frame_count_total,
        "qgain_frame_count_nonfloor": frame_count_nonfloor,
        "qgain_floor_frame_fraction": floor_fraction,
        "qgain_floor_censored": floor_censored,
        "qgain_usable_segment_count": segment_count,
        "qgain_original_time_span_sec": span_sec,
        "qgain_signed_drift_db_per_min": signed_drift,
        "qgain_signed_drift_ci95_low_db_per_min": drift_ci_low,
        "qgain_signed_drift_ci95_high_db_per_min": drift_ci_high,
        "qgain_exploratory_local_transition_count": transition_count,
        "qgain_exploratory_local_transition_exposure_sec": transition_exposure_sec,
        "qgain_exploratory_local_transition_rate_per_min": exploratory_transition_rate,
        "qgain_exploratory_local_transition_ci95_low_per_min": transition_ci_low,
        "qgain_exploratory_local_transition_ci95_high_per_min": transition_ci_high,
        "qgain_exploratory_local_transition_status": (
            "rejected_v3_0_false_positive_burden_not_analysis"
        ),
        "qgain_typical_speech_level_dbfs_raw_estimate": raw_typical,
        "qgain_within_segment_iqr_db_raw_estimate": raw_within,
        "qgain_between_segment_mad_db_raw_estimate": raw_between,
        "qgain_abs_drift_db_per_min_raw_estimate": raw_abs_drift,
        **values,
    }
    for feature in ANALYSIS_FEATURES:
        tier = _support_tier(
            feature,
            speech_sec=speech_support_sec,
            frame_count=frame_count_nonfloor,
            segment_count=segment_count,
            span_sec=span_sec,
            parameters=parameters,
        )
        value = values[feature]
        if floor_censored:
            status = "floor_censored"
        elif np.isfinite(value):
            status = "measured"
        else:
            status = "insufficient_support"
        recording[f"{feature}_support_tier"] = tier
        recording[f"{feature}_status"] = status
    primary_available = sum(np.isfinite(values[name]) for name in PRIMARY_FEATURES)
    recording["qgain_primary_available_count"] = int(primary_available)
    recording["qgain_primary_analysis_eligible"] = bool(primary_available > 0)
    if primary_available == len(PRIMARY_FEATURES):
        recording["qgain_family_status"] = "all_primary_available"
    elif primary_available:
        recording["qgain_family_status"] = "partial_primary_available"
    elif floor_censored:
        recording["qgain_family_status"] = "floor_censored"
    else:
        recording["qgain_family_status"] = "unavailable"
    return QGAINExtraction(recording, frames, segments, event_ledger)


def apply_gain_db(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply unclipped digital gain for deterministic controls."""

    return np.asarray(waveform, dtype=np.float64) * (10.0 ** (float(gain_db) / 20.0))


def apply_level_envelope_db(waveform: np.ndarray, envelope_db: np.ndarray | float) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.float64)
    envelope = np.asarray(envelope_db, dtype=np.float64)
    if envelope.ndim == 0:
        envelope = np.full(len(values), float(envelope))
    if len(envelope) != len(values):
        raise ValueError("Envelope must have one dB value per waveform sample.")
    return values * np.power(10.0, envelope / 20.0)
