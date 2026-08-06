"""QGAIN v4.0 reviewed candidate: recorded level and level dynamics.

The authoritative estimators live here. The notebook orchestrates validation,
cohort extraction, figures, and freeze governance without reimplementing the
algorithms.

Claim boundary
--------------
These features describe observable recorded-level behaviour. They do not
identify automatic gain control, microphone motion, speaker effort, respiratory
change, dysarthria, or another unique physical cause from a no-reference signal.
They are non-ordinal measurement-context variables. No feature is an approved
standalone reject/accept gate for downstream biomarkers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import log10, sqrt

import numpy as np
import pandas as pd
from scipy import stats

MEASUREMENT_VERSION = "qgain-v4.0.0-candidate"
FAMILY = "QGAIN"
FAMILY_DISPLAY_NAME = "Recorded level and level dynamics"

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
    moderate_speech_support_sec: float = 3.0
    high_speech_support_sec: float = 5.0
    moderate_segment_count: int = 5
    high_segment_count: int = 8
    moderate_drift_span_sec: float = 20.0
    high_drift_span_sec: float = 30.0
    mad_normal_consistency_factor: float = 1.482602218505602
    drift_permutation_iterations: int = 1000
    nominal_full_scale: float = 1.0
    full_scale_tolerance: float = 1e-6
    random_seed: int = 20260801

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMETERS = QGAINParameters()


@dataclass(frozen=True)
class FeatureDefinition:
    feature: str
    display_name: str
    subdomain: str
    measurement_role: str
    interpretation_class: str
    unit: str
    maturity: str
    estimator: str
    estimand: str
    orientation: str
    quality_direction: str
    claim_limit: str
    minimum_support: str
    known_confounds: str
    phenotype_confounding_risk: str
    ml_role: str
    standalone_gate_allowed: bool
    composite_use_prohibited: bool
    missing_value_behavior: str


FEATURE_DEFINITIONS = (
    FeatureDefinition(
        feature="qgain_typical_speech_level_dbfs",
        display_name="Typical guarded strict-speech operating level",
        subdomain="operating level",
        measurement_role="contextual",
        interpretation_class="mixed acquisition/physiology context",
        unit="analysis-view dBFS",
        maturity="established RMS/dB primitive; study-specific robust aggregation",
        estimator="median of 40-ms framewise AC-RMS dBFS; 10-ms hop; 200-ms edge guard",
        estimand="typical non-floor strict-speech level in the defined 16-kHz mono analysis view",
        orientation="higher is closer to nominal digital full scale",
        quality_direction="nonordinal",
        claim_limit="not calibrated SPL, not vocal intensity, and not ITU-T P.56 active speech level",
        minimum_support=">=1.0 s guarded speech, >=25 non-floor frames, >=1 usable segment",
        known_confounds="speaker effort, dysarthria, respiration, distance, device gain, AGC, downmix, task",
        phenotype_confounding_risk="high",
        ml_role="measurement-context covariate; candidate input to biomarker-specific reliability model",
        standalone_gate_allowed=False,
        composite_use_prohibited=True,
        missing_value_behavior="NaN with explicit status/support; never imputed by extractor",
    ),
    FeatureDefinition(
        feature="qgain_within_segment_iqr_db",
        display_name="Within-segment centered level IQR",
        subdomain="short-term recorded-level dynamics",
        measurement_role="primary",
        interpretation_class="mixed acquisition/physiology descriptor",
        unit="dB",
        maturity="established IQR primitive; study-specific support and aggregation",
        estimator="subtract each usable segment median, pool centered frame levels, compute Q75-Q25",
        estimand="short-term recorded-level dispersion after removal of segment offsets",
        orientation="higher means greater within-segment level dispersion",
        quality_direction="nonordinal",
        claim_limit="compatible with gain modulation but not specific to AGC or acquisition instability",
        minimum_support=">=1.0 s guarded speech, >=25 non-floor frames, >=1 usable segment",
        known_confounds="prosody, phonetic composition, respiration, dysarthria, segmentation",
        phenotype_confounding_risk="high",
        ml_role="candidate biomarker-reliability covariate; no standalone reject rule",
        standalone_gate_allowed=False,
        composite_use_prohibited=True,
        missing_value_behavior="NaN with explicit status/support; never imputed by extractor",
    ),
    FeatureDefinition(
        feature="qgain_between_segment_mad_db",
        display_name="Between-segment normal-consistent MAD scale",
        subdomain="segment-level recorded-level dynamics",
        measurement_role="primary",
        interpretation_class="mixed acquisition/physiology descriptor",
        unit="dB",
        maturity="established MAD primitive; study-specific segment aggregation",
        estimator="1/Phi^-1(0.75) times MAD of usable segment-median levels",
        estimand="robust segment-to-segment dispersion in typical recorded level",
        orientation="higher means larger segment-level shifts",
        quality_direction="nonordinal",
        claim_limit="not an unbiased standard deviation estimate and not a pure device-gain estimate",
        minimum_support=">=3 usable segments; segment count and span reported",
        known_confounds="phrase content, fatigue, posture, distance, task structure, segment definition",
        phenotype_confounding_risk="high",
        ml_role="candidate biomarker-reliability covariate; low-support cases require mask/tier",
        standalone_gate_allowed=False,
        composite_use_prohibited=True,
        missing_value_behavior="NaN with explicit status/support; never imputed by extractor",
    ),
    FeatureDefinition(
        feature="qgain_abs_drift_db_per_min",
        display_name="Absolute robust recorded-level drift",
        subdomain="slow recorded-level dynamics",
        measurement_role="primary",
        interpretation_class="mixed acquisition/physiology descriptor",
        unit="dB/min",
        maturity="published Theil-Sen slope primitive; study-specific segment/time aggregation",
        estimator="absolute Theil-Sen slope of usable segment medians against original recording time",
        estimand="magnitude of monotonic recorded-level change over the observed task span",
        orientation="higher means faster absolute recorded-level drift",
        quality_direction="nonordinal",
        claim_limit="cannot distinguish device gain/distance change from fatigue, effort, or respiratory decline",
        minimum_support=">=3 usable segments spanning >=10 s; precision tier is mandatory",
        known_confounds="fatigue, intentional loudness, respiration, posture, task order, segmentation",
        phenotype_confounding_risk="very_high",
        ml_role="candidate reliability/context covariate only; not acquisition-specific",
        standalone_gate_allowed=False,
        composite_use_prohibited=True,
        missing_value_behavior="NaN with explicit status/support; never imputed by extractor",
    ),
)


@dataclass
class QGAINExtraction:
    recording: dict
    frame_ledger: pd.DataFrame
    segment_ledger: pd.DataFrame


def feature_registry_frame() -> pd.DataFrame:
    rows = []
    for definition in FEATURE_DEFINITIONS:
        row = asdict(definition)
        row.update(
            {
                "family": FAMILY,
                "family_display_name": FAMILY_DISPLAY_NAME,
                "measurement_version": MEASUREMENT_VERSION,
                "analysis_eligible": True,
                "publication_status": "candidate_pending_review",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


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
            merged[-1] = TimeInterval(
                merged[-1].start_sec, max(merged[-1].end_sec, item.end_sec)
            )
    return merged


def guarded_speech_intervals(
    intervals: Iterable[TimeInterval],
    duration_sec: float,
    *,
    parameters: QGAINParameters = DEFAULT_PARAMETERS,
    guard_ms: float | None = None,
) -> list[TimeInterval]:
    guard_sec = (
        parameters.speech_edge_guard_ms if guard_ms is None else float(guard_ms)
    ) / 1000.0
    output: list[TimeInterval] = []
    for item in merge_intervals(intervals):
        start = max(0.0, item.start_sec + guard_sec)
        end = min(float(duration_sec), item.end_sec - guard_sec)
        if end > start:
            output.append(TimeInterval(start, end))
    return output


def ac_rms_dbfs(
    samples: np.ndarray, floor_db: float = -120.0
) -> tuple[float, bool, float]:
    """Return frame AC-RMS dBFS, floor status, and linear AC-RMS.

    A frame is at the computational floor when its AC-RMS is less than or equal
    to the linear amplitude represented by ``floor_db``. This corrects the v3.1
    behavior in which sub-floor nonzero frames were clamped but not marked as
    floor affected.
    """

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        return np.nan, False, np.nan
    centered = values - float(np.mean(values, dtype=np.float64))
    rms = float(sqrt(float(np.mean(centered * centered, dtype=np.float64))))
    linear_floor = 10.0 ** (float(floor_db) / 20.0)
    if not np.isfinite(rms) or rms <= linear_floor:
        return float(floor_db), True, rms
    return float(20.0 * log10(rms)), False, rms


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
        and segment_count >= 1
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
        local_rows: list[dict] = []
        if right - left >= frame_length:
            for frame_left in range(left, right - frame_length + 1, hop):
                frame_right = frame_left + frame_length
                level, at_floor, linear_rms = ac_rms_dbfs(
                    waveform[frame_left:frame_right], parameters.dbfs_floor_db
                )
                local_rows.append(
                    {
                        "logical_recording_id": logical_recording_id,
                        "segment_id": int(segment_id),
                        "segment_start_sec": interval.start_sec,
                        "segment_end_sec": interval.end_sec,
                        "frame_start_sec": frame_left / fs,
                        "frame_end_sec": frame_right / fs,
                        "frame_mid_sec": (frame_left + frame_right) / (2.0 * fs),
                        "ac_rms_linear": linear_rms,
                        "ac_rms_dbfs": level,
                        "at_computational_floor": bool(at_floor),
                        "valid_level_frame": bool(np.isfinite(level) and not at_floor),
                    }
                )
        frame_rows.extend(local_rows)
        local = pd.DataFrame(local_rows)
        if len(local):
            valid_levels = pd.to_numeric(
                local.loc[local["valid_level_frame"], "ac_rms_dbfs"], errors="coerce"
            ).dropna().to_numpy(float)
            floor_count = int(local["at_computational_floor"].sum())
        else:
            valid_levels = np.asarray([], dtype=float)
            floor_count = 0
        usable = len(valid_levels) >= parameters.minimum_segment_frame_count
        segment_rows.append(
            {
                "logical_recording_id": logical_recording_id,
                "segment_id": int(segment_id),
                "segment_start_sec": interval.start_sec,
                "segment_end_sec": interval.end_sec,
                "segment_mid_sec": (interval.start_sec + interval.end_sec) / 2.0,
                "guarded_segment_duration_sec": interval.duration_sec,
                "frame_count_total": int(len(local_rows)),
                "frame_count_nonfloor": int(len(valid_levels)),
                "floor_frame_count": floor_count,
                "segment_level_median_dbfs": (
                    float(np.median(valid_levels)) if usable else np.nan
                ),
                "segment_level_iqr_db": (
                    float(np.quantile(valid_levels, 0.75) - np.quantile(valid_levels, 0.25))
                    if usable
                    else np.nan
                ),
                "usable_segment": bool(usable),
            }
        )
    frame_columns = [
        "logical_recording_id",
        "segment_id",
        "segment_start_sec",
        "segment_end_sec",
        "frame_start_sec",
        "frame_end_sec",
        "frame_mid_sec",
        "ac_rms_linear",
        "ac_rms_dbfs",
        "at_computational_floor",
        "valid_level_frame",
    ]
    segment_columns = [
        "logical_recording_id",
        "segment_id",
        "segment_start_sec",
        "segment_end_sec",
        "segment_mid_sec",
        "guarded_segment_duration_sec",
        "frame_count_total",
        "frame_count_nonfloor",
        "floor_frame_count",
        "segment_level_median_dbfs",
        "segment_level_iqr_db",
        "usable_segment",
    ]
    return (
        pd.DataFrame(frame_rows, columns=frame_columns),
        pd.DataFrame(segment_rows, columns=segment_columns),
    )


def _mad(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return np.nan
    center = float(np.median(array))
    return float(np.median(np.abs(array - center)))


def _normal_consistent_mad(values: Sequence[float], factor: float) -> float:
    raw = _mad(values)
    return float(factor * raw) if np.isfinite(raw) else np.nan


def _default_signal_provenance() -> dict:
    return {
        "signal_view": "mono_global_dc_removed_resampled_16k; framewise_mean_removed_AC_RMS",
        "source_decode": "ffmpeg_f32le_or_equivalent",
        "amplitude_normalization_applied": False,
        "denoising_applied": False,
        "dynamic_range_processing_applied": False,
        "native_sample_rate_hz": np.nan,
        "native_channels": np.nan,
        "codec_name": None,
    }


def extract_qgain(
    waveform: np.ndarray,
    fs: int,
    *,
    strict_speech: Iterable[TimeInterval],
    logical_recording_id: str = "recording",
    parameters: QGAINParameters = DEFAULT_PARAMETERS,
    guard_ms: float | None = None,
    signal_provenance: Mapping[str, object] | None = None,
) -> QGAINExtraction:
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("QGAIN requires a mono analysis waveform.")
    if int(fs) != parameters.analysis_sample_rate_hz:
        raise ValueError(
            f"QGAIN requires {parameters.analysis_sample_rate_hz} Hz analysis audio; got {fs}."
        )
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Waveform is empty or contains non-finite samples.")

    provenance = _default_signal_provenance()
    if signal_provenance:
        provenance.update(dict(signal_provenance))
    if bool(provenance.get("amplitude_normalization_applied", False)):
        raise ValueError("QGAIN is invalid after amplitude normalization.")
    if bool(provenance.get("dynamic_range_processing_applied", False)):
        raise ValueError("QGAIN is invalid after untracked dynamic-range processing.")

    duration_sec = len(samples) / float(fs)
    guarded = guarded_speech_intervals(
        strict_speech, duration_sec, parameters=parameters, guard_ms=guard_ms
    )
    speech_support_sec = float(sum(item.duration_sec for item in guarded))
    frames, segments = _frame_speech(
        samples, fs, guarded, logical_recording_id, parameters
    )

    if len(frames):
        valid_frames = frames.loc[frames["valid_level_frame"]].copy()
        frame_count_total = int(len(frames))
        frame_count_nonfloor = int(len(valid_frames))
        floor_fraction = float(frames["at_computational_floor"].mean())
    else:
        valid_frames = pd.DataFrame(columns=frames.columns)
        frame_count_total = 0
        frame_count_nonfloor = 0
        floor_fraction = np.nan

    usable_segments = (
        segments.loc[segments["usable_segment"]].copy()
        if len(segments)
        else pd.DataFrame(columns=segments.columns)
    )
    usable_ids = set(pd.to_numeric(usable_segments.get("segment_id"), errors="coerce").dropna().astype(int))
    within_frames = (
        valid_frames.loc[valid_frames["segment_id"].isin(usable_ids)].copy()
        if len(valid_frames)
        else pd.DataFrame(columns=valid_frames.columns)
    )
    segment_levels = pd.to_numeric(
        usable_segments.get("segment_level_median_dbfs"), errors="coerce"
    ).dropna()
    segment_times = pd.to_numeric(
        usable_segments.get("segment_mid_sec"), errors="coerce"
    ).loc[segment_levels.index]
    segment_count = int(len(segment_levels))
    span_sec = (
        float(segment_times.max() - segment_times.min()) if segment_count >= 2 else 0.0
    )
    floor_contaminated = bool(
        np.isfinite(floor_fraction)
        and floor_fraction > parameters.maximum_floor_frame_fraction
    )

    basic_support = bool(
        speech_support_sec >= parameters.minimum_speech_support_sec
        and frame_count_nonfloor >= parameters.minimum_valid_frame_count
        and segment_count >= 1
        and not floor_contaminated
    )

    raw_typical = (
        float(np.median(pd.to_numeric(valid_frames["ac_rms_dbfs"], errors="coerce").dropna()))
        if frame_count_nonfloor
        else np.nan
    )
    centered: list[float] = []
    if len(within_frames):
        for _, local in within_frames.groupby("segment_id", sort=True):
            levels = pd.to_numeric(local["ac_rms_dbfs"], errors="coerce").dropna()
            if len(levels):
                centered.extend((levels - levels.median()).tolist())
    raw_within = (
        float(np.quantile(centered, 0.75) - np.quantile(centered, 0.25))
        if centered
        else np.nan
    )

    raw_mad_unscaled = _mad(segment_levels.to_numpy(float)) if segment_count >= 2 else np.nan
    raw_between = (
        _normal_consistent_mad(
            segment_levels.to_numpy(float), parameters.mad_normal_consistency_factor
        )
        if segment_count >= 2
        else np.nan
    )

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

    segment_balanced_typical = (
        float(np.median(segment_levels.to_numpy(float))) if segment_count else np.nan
    )
    segment_iqrs = pd.to_numeric(
        usable_segments.get("segment_level_iqr_db"), errors="coerce"
    ).dropna()
    segment_balanced_within = (
        float(np.median(segment_iqrs.to_numpy(float))) if len(segment_iqrs) else np.nan
    )

    values = {
        "qgain_typical_speech_level_dbfs": raw_typical if basic_support else np.nan,
        "qgain_within_segment_iqr_db": raw_within if basic_support else np.nan,
        "qgain_between_segment_mad_db": (
            raw_between
            if segment_count >= parameters.between_minimum_segment_count
            and not floor_contaminated
            else np.nan
        ),
        "qgain_abs_drift_db_per_min": (
            raw_abs_drift if not floor_contaminated else np.nan
        ),
    }

    peak_abs = float(np.max(np.abs(samples)))
    over_nominal = np.abs(samples) > (
        parameters.nominal_full_scale + parameters.full_scale_tolerance
    )
    over_nominal_fraction = float(np.mean(over_nominal))

    recording = {
        "logical_recording_id": str(logical_recording_id),
        "qgain_measurement_version": MEASUREMENT_VERSION,
        "qgain_family_display_name": FAMILY_DISPLAY_NAME,
        "qgain_signal_view": str(provenance["signal_view"]),
        "qgain_source_decode": str(provenance["source_decode"]),
        "qgain_amplitude_normalization_applied": bool(
            provenance["amplitude_normalization_applied"]
        ),
        "qgain_denoising_applied": bool(provenance["denoising_applied"]),
        "qgain_dynamic_range_processing_applied": bool(
            provenance["dynamic_range_processing_applied"]
        ),
        "qgain_native_sample_rate_hz": provenance.get("native_sample_rate_hz"),
        "qgain_native_channels": provenance.get("native_channels"),
        "qgain_codec_name": provenance.get("codec_name"),
        "qgain_speech_source": "frozen_strict_speech",
        "qgain_speech_edge_guard_ms": (
            parameters.speech_edge_guard_ms if guard_ms is None else float(guard_ms)
        ),
        "qgain_guarded_speech_interval_count": int(len(guarded)),
        "qgain_guarded_speech_support_sec": speech_support_sec,
        "qgain_frame_count_total": frame_count_total,
        "qgain_frame_count_nonfloor": frame_count_nonfloor,
        "qgain_floor_frame_fraction": floor_fraction,
        "qgain_floor_contaminated": floor_contaminated,
        "qgain_usable_segment_count": segment_count,
        "qgain_original_time_span_sec": span_sec,
        "qgain_analysis_peak_abs": peak_abs,
        "qgain_samples_over_nominal_full_scale_fraction": over_nominal_fraction,
        "qgain_nominal_full_scale_exceeded": bool(over_nominal.any()),
        "qgain_signed_drift_db_per_min": signed_drift,
        "qgain_signed_drift_ci95_low_db_per_min": drift_ci_low,
        "qgain_signed_drift_ci95_high_db_per_min": drift_ci_high,
        "qgain_theilsen_pair_count": int(segment_count * (segment_count - 1) // 2),
        "qgain_typical_speech_level_dbfs_raw_estimate": raw_typical,
        "qgain_within_segment_iqr_db_raw_estimate": raw_within,
        "qgain_between_segment_mad_db_raw_estimate": raw_between,
        "qgain_between_segment_mad_unscaled_db": raw_mad_unscaled,
        "qgain_abs_drift_db_per_min_raw_estimate": raw_abs_drift,
        "qgain_typical_level_segment_balanced_dbfs": segment_balanced_typical,
        "qgain_typical_level_segment_balance_delta_db": (
            raw_typical - segment_balanced_typical
            if np.isfinite(raw_typical) and np.isfinite(segment_balanced_typical)
            else np.nan
        ),
        "qgain_within_iqr_segment_median_db": segment_balanced_within,
        "qgain_within_iqr_segment_balance_delta_db": (
            raw_within - segment_balanced_within
            if np.isfinite(raw_within) and np.isfinite(segment_balanced_within)
            else np.nan
        ),
        "qgain_decision_threshold_status": "not_calibrated",
        "qgain_standalone_reject_allowed": False,
        "qgain_ml_use_status": "candidate_measurement_context_and_biomarker_reliability_covariates",
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
        if floor_contaminated:
            status = "floor_contaminated"
        elif np.isfinite(value):
            status = "measured"
        else:
            status = "insufficient_support"
        recording[f"{feature}_available"] = bool(np.isfinite(value))
        recording[f"{feature}_support_tier"] = tier
        recording[f"{feature}_status"] = status

    primary_available = sum(np.isfinite(values[name]) for name in PRIMARY_FEATURES)
    recording["qgain_primary_available_count"] = int(primary_available)
    recording["qgain_primary_analysis_eligible"] = bool(primary_available > 0)
    if primary_available == len(PRIMARY_FEATURES):
        family_status = "all_primary_available"
    elif primary_available:
        family_status = "partial_primary_available"
    elif floor_contaminated:
        family_status = "floor_contaminated"
    else:
        family_status = "unavailable"
    recording["qgain_family_measurement_status"] = family_status
    return QGAINExtraction(recording=recording, frame_ledger=frames, segment_ledger=segments)


def measurement_long_frame(
    rows: Mapping[str, object] | pd.DataFrame,
    *,
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert wide QGAIN measurements to one row per recording-feature.

    No imputation is performed. A legitimate measured zero remains available;
    unavailable measurements remain NaN with an explicit status.
    """

    wide = pd.DataFrame([rows]) if isinstance(rows, Mapping) else rows.copy()
    registry = feature_registry_frame() if registry is None else registry.copy()
    definitions = registry.set_index("feature").to_dict(orient="index")
    output: list[dict] = []
    for _, row in wide.iterrows():
        for feature in ANALYSIS_FEATURES:
            value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
            meta = definitions[feature]
            output.append(
                {
                    "logical_recording_id": str(row.get("logical_recording_id")),
                    "family": FAMILY,
                    "feature": feature,
                    "value": value,
                    "unit": meta["unit"],
                    "available": bool(row.get(f"{feature}_available", np.isfinite(value))),
                    "measurement_status": row.get(f"{feature}_status"),
                    "support_tier": row.get(f"{feature}_support_tier"),
                    "measurement_version": row.get(
                        "qgain_measurement_version", MEASUREMENT_VERSION
                    ),
                    "signal_view": row.get("qgain_signal_view"),
                    "phenotype_confounding_risk": meta[
                        "phenotype_confounding_risk"
                    ],
                    "quality_direction": meta["quality_direction"],
                    "standalone_gate_allowed": bool(meta["standalone_gate_allowed"]),
                    "ml_role": meta["ml_role"],
                }
            )
    return pd.DataFrame(output)


def model_interface_frame(rows: Mapping[str, object] | pd.DataFrame) -> pd.DataFrame:
    """Return a wide, non-imputed ML interface with masks and support metadata."""

    wide = pd.DataFrame([rows]) if isinstance(rows, Mapping) else rows.copy()
    columns = ["logical_recording_id", "qgain_measurement_version", "qgain_signal_view"]
    for feature in ANALYSIS_FEATURES:
        columns.extend(
            [
                feature,
                f"{feature}_available",
                f"{feature}_status",
                f"{feature}_support_tier",
            ]
        )
    columns.extend(
        [
            "qgain_guarded_speech_support_sec",
            "qgain_usable_segment_count",
            "qgain_original_time_span_sec",
            "qgain_floor_frame_fraction",
            "qgain_floor_contaminated",
            "qgain_family_measurement_status",
            "qgain_decision_threshold_status",
            "qgain_standalone_reject_allowed",
            "qgain_ml_use_status",
        ]
    )
    return wide.reindex(columns=columns)


def permutation_drift_audit(
    segment_ledger: pd.DataFrame,
    *,
    iterations: int = DEFAULT_PARAMETERS.drift_permutation_iterations,
    seed: int = DEFAULT_PARAMETERS.random_seed,
) -> dict:
    """Time-order permutation audit for absolute Theil-Sen drift.

    This is an audit of ordered trend evidence, not a feature and not evidence
    that the trend is acquisition-caused.
    """

    local = segment_ledger.loc[segment_ledger["usable_segment"]].copy()
    levels = pd.to_numeric(local["segment_level_median_dbfs"], errors="coerce")
    times = pd.to_numeric(local["segment_mid_sec"], errors="coerce")
    valid = levels.notna() & times.notna()
    levels = levels.loc[valid].to_numpy(float)
    times = times.loc[valid].to_numpy(float)
    if len(levels) < 3 or float(np.max(times) - np.min(times)) <= 0:
        return {
            "observed_abs_drift_db_per_min": np.nan,
            "permutation_pvalue": np.nan,
            "permutation_iterations": int(iterations),
            "segment_count": int(len(levels)),
        }
    observed = abs(float(stats.theilslopes(levels, times).slope * 60.0))
    rng = np.random.default_rng(seed)
    null = np.empty(iterations, dtype=float)
    for index in range(iterations):
        shuffled = rng.permutation(levels)
        null[index] = abs(float(stats.theilslopes(shuffled, times).slope * 60.0))
    pvalue = float((1 + np.count_nonzero(null >= observed)) / (iterations + 1))
    return {
        "observed_abs_drift_db_per_min": observed,
        "permutation_pvalue": pvalue,
        "permutation_iterations": int(iterations),
        "segment_count": int(len(levels)),
        "null_median_db_per_min": float(np.median(null)),
        "null_p95_db_per_min": float(np.quantile(null, 0.95)),
    }


def apply_gain_db(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    return np.asarray(waveform, dtype=np.float64) * (10.0 ** (float(gain_db) / 20.0))


def apply_level_envelope_db(
    waveform: np.ndarray, envelope_db: np.ndarray | float
) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.float64)
    envelope = np.asarray(envelope_db, dtype=np.float64)
    if envelope.ndim == 0:
        envelope = np.full(len(values), float(envelope))
    if len(envelope) != len(values):
        raise ValueError("Envelope must have one dB value per waveform sample.")
    return values * np.power(10.0, envelope / 20.0)
