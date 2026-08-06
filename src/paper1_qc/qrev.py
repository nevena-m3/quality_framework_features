"""QREV v3.1: no-reference residual-tail and modulation measurements.

The module is the authoritative implementation for the QREV notebook. It
describes observable post-speech persistence and modulation smearing compatible
with reverberation or echo. It does not estimate RT60, EDT, C50/C80, D50, DRR,
STI, or a room impulse response. It also does not detect or separately quantify
discrete-delay echo; echo is one possible cause of a residual-tail response.

The three boundary estimators use only natural internal speech-to-pause
boundaries. The SRMR comparator uses the original task span from first speech
onset to last speech offset, preserving internal pauses. Extraction is
recording-at-a-time and retains only a compact boundary ledger. Natural-boundary
features are conditional measurements: gradual, breathy, or uncertain speech
offsets and insufficient long-pause support can make them unavailable. Whether
availability is associated with ALS severity is a downstream scientific
question, not an assumption made by this estimator.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from math import ceil, floor, log10, sqrt

import numpy as np
import pandas as pd
from scipy import stats

MEASUREMENT_VERSION = "qrev-v3.1.0"
SRMR_UPSTREAM_COMMIT = "fee009779cef96bed34db3a7e31d10f3ad1ea133"
SRMR_VARIANT = "SRMRpy normalized-fast; norm=True; fast=True; max_cf=30"
SRMR_GAMMATONE_VERSION = "1.0.3"
SRMR_PINNED_REGRESSION_VALUE = 3.7158141034373164

ANALYSIS_FEATURES = (
    "qrev_tail_excess_100ms_db",
    "qrev_tail_persistence_median_sec",
    "qrev_downward_decay_rate_db_per_sec",
    "qrev_srmr_norm",
)

PRIMARY_FEATURES = (
    "qrev_tail_excess_100ms_db",
    "qrev_tail_persistence_median_sec",
)

CONDITIONAL_BOUNDARY_FEATURES = (
    "qrev_tail_excess_100ms_db",
    "qrev_tail_persistence_median_sec",
    "qrev_downward_decay_rate_db_per_sec",
)

BROADLY_AVAILABLE_COMPARATOR_FEATURES = ("qrev_srmr_norm",)


@dataclass(frozen=True)
class TimeInterval:
    """Half-open interval in original recording seconds."""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


@dataclass(frozen=True)
class QREVParameters:
    """QREV v3.1 engineering and support parameters."""

    analysis_sample_rate_hz: int = 16_000
    frame_length_ms: float = 30.0
    frame_hop_ms: float = 10.0
    dbfs_floor_db: float = -120.0
    maximum_floor_frame_fraction: float = 0.10
    maximum_floor_iqr_db: float = 12.0

    early_tail_start_ms: float = 0.0
    early_tail_end_ms: float = 100.0
    decay_start_ms: float = 0.0
    decay_end_ms: float = 300.0
    floor_start_ms: float = 700.0
    floor_end_ms: float = 1000.0
    persistence_horizon_ms: float = 1000.0
    persistence_threshold_db: float = 3.0
    persistence_consecutive_frames: int = 3
    minimum_early_frame_count: int = 5
    minimum_floor_frame_count: int = 20
    minimum_decay_frame_count: int = 20
    minimum_persistence_frame_count: int = 80
    minimum_decay_dynamic_range_db: float = 3.0

    minimum_tail_boundary_count: int = 4
    minimum_tail_pause_support_sec: float = 2.0
    minimum_persistence_boundary_count: int = 4
    minimum_decay_boundary_count: int = 4
    moderate_boundary_count: int = 6
    high_boundary_count: int = 8
    moderate_pause_support_sec: float = 3.0
    high_pause_support_sec: float = 4.0

    minimum_srmr_speech_support_sec: float = 3.0
    minimum_srmr_task_span_sec: float = 3.0
    srmr_n_cochlear_filters: int = 23
    srmr_low_frequency_hz: float = 125.0
    srmr_min_modulation_cf_hz: float = 4.0
    srmr_max_modulation_cf_hz: float = 30.0
    srmr_normalized: bool = True
    srmr_fast: bool = True
    maximum_srmr_estimated_memory_mb: float = 512.0

    random_seed: int = 20260729

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMETERS = QREVParameters()


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    display_name: str
    subdomain: str
    role: str
    unit: str
    maturity: str
    estimand: str
    orientation: str
    claim_boundary: str
    minimum_support: str
    known_confounds: str


FEATURE_DEFINITIONS = (
    FeatureDefinition(
        "qrev_tail_excess_100ms_db",
        "Early post-offset tail excess",
        "residual magnitude",
        "primary conditional",
        "dB",
        "study-specific residual-tail proxy",
        "Median signed difference between the first 100-ms post-offset level and "
        "an independent 700-1000-ms local pause baseline.",
        "Higher indicates stronger early residual energy above the late-pause baseline.",
        "Conditional blind residual-tail proxy; not RT60, DRR, an "
        "impulse-response estimate, or a discrete-delay echo detector.",
        "At least four valid boundaries and 2.0 s eligible pause support.",
        "Breath, noise change, echo, speech leakage, offset error, and floor instability.",
    ),
    FeatureDefinition(
        "qrev_tail_persistence_median_sec",
        "Bounded tail persistence",
        "residual persistence",
        "primary conditional",
        "s",
        "study-specific censored persistence estimator",
        "Median time until the frame envelope first remains within 3 dB of the "
        "local floor for three frames, right-censored at 1.0 s.",
        "Higher indicates longer observable above-floor persistence.",
        "Conditional feature, not reverberation time; a ceiling value is a "
        "censored lower bound and echo identity is unresolved.",
        "At least four valid full-horizon boundaries.",
        "Echo, breath, changing noise, pause length, floor variability, and offset error.",
    ),
    FeatureDefinition(
        "qrev_downward_decay_rate_db_per_sec",
        "Conditional downward tail-decay rate",
        "decay shape",
        "secondary conditional",
        "dB/s",
        "study-specific robust slope",
        "Median magnitude of a negative Theil-Sen slope during the first 300 ms "
        "when robust dynamic range is at least 3 dB.",
        "Lower positive magnitude indicates slower valid downward decay.",
        "Conditional feature, not a Schroeder decay or an RT estimate; "
        "unavailable is not zero and discrete-delay echo is not identified.",
        "At least four eligible downward-decay boundaries.",
        "Nonsmooth echo, floor crossing, breath, insufficient range, and smoothing.",
    ),
    FeatureDefinition(
        "qrev_srmr_norm",
        "Normalized speech-to-reverberation modulation energy ratio",
        "modulation smearing",
        "broadly available established comparator",
        "ratio",
        "published no-reference metric; pinned implementation",
        "Normalized-fast SRMRpy ratio over the natural task span, with internal "
        "pauses preserved.",
        "Typically lower indicates more reverberation-related modulation smearing.",
        "No-reference reverberation-sensitive comparator; not a direct RT60 measure "
        "and not specific to room reverberation.",
        "At least 3.0 s frozen speech and a 3.0-s task span.",
        "Additive noise, codec, speech content, bandwidth, pauses, and variant choice.",
    ),
)


@dataclass
class QREVExtraction:
    recording: dict
    boundary_ledger: pd.DataFrame


def feature_registry_frame() -> pd.DataFrame:
    """Return the immutable, one-row-per-analysis-feature registry."""

    return pd.DataFrame([asdict(item) for item in FEATURE_DEFINITIONS])


def merge_intervals(
    intervals: Iterable[TimeInterval],
    duration_sec: float | None = None,
) -> list[TimeInterval]:
    """Validate, optionally clip, sort, and merge half-open intervals."""

    clean: list[TimeInterval] = []
    for item in intervals:
        start = float(item.start_sec)
        end = float(item.end_sec)
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        if duration_sec is not None:
            start = min(max(start, 0.0), float(duration_sec))
            end = min(max(end, 0.0), float(duration_sec))
        if end > start:
            clean.append(TimeInterval(start, end))
    clean.sort(key=lambda item: (item.start_sec, item.end_sec))
    merged: list[TimeInterval] = []
    for item in clean:
        if not merged or item.start_sec > merged[-1].end_sec:
            merged.append(item)
        else:
            merged[-1] = TimeInterval(
                merged[-1].start_sec,
                max(merged[-1].end_sec, item.end_sec),
            )
    return merged


def internal_pause_boundaries(
    strict_speech: Iterable[TimeInterval],
    duration_sec: float,
) -> list[dict]:
    """Return natural internal speech-offset boundaries and following pauses."""

    speech = merge_intervals(strict_speech, duration_sec)
    rows = []
    for boundary_index, (left, right) in enumerate(zip(speech[:-1], speech[1:])):
        if right.start_sec <= left.end_sec:
            continue
        rows.append(
            {
                "boundary_index": int(boundary_index),
                "previous_speech_start_sec": left.start_sec,
                "speech_offset_sec": left.end_sec,
                "pause_start_sec": left.end_sec,
                "pause_end_sec": right.start_sec,
                "next_speech_onset_sec": right.start_sec,
                "pause_duration_sec": right.start_sec - left.end_sec,
            }
        )
    return rows


def ac_rms_dbfs(samples: np.ndarray, floor_db: float = -120.0) -> tuple[float, bool]:
    """Return AC-RMS dBFS and whether the frame reached the digital floor."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        return np.nan, False
    centered = values - float(np.mean(values))
    rms = float(sqrt(float(np.mean(centered * centered))))
    if rms <= 0.0 or not np.isfinite(rms):
        return float(floor_db), True
    return float(max(floor_db, 20.0 * log10(rms))), False


def _frame_levels(
    waveform: np.ndarray,
    fs: int,
    window_start_sec: float,
    window_end_sec: float,
    parameters: QREVParameters,
) -> pd.DataFrame:
    """Frame a window using strict sample containment.

    Every returned frame satisfies ``start >= window_start`` and
    ``end <= window_end``. No midpoint-only inclusion is used.
    """

    frame_n = round(parameters.frame_length_ms * fs / 1000.0)
    hop_n = round(parameters.frame_hop_ms * fs / 1000.0)
    first = max(0, int(ceil(window_start_sec * fs - 1e-9)))
    final_sample = min(len(waveform), int(floor(window_end_sec * fs + 1e-9)))
    final_start = final_sample - frame_n
    columns = (
        "frame_start_sec",
        "frame_end_sec",
        "frame_mid_sec",
        "ac_rms_dbfs",
        "at_digital_floor",
    )
    if final_start < first:
        return pd.DataFrame(columns=columns)
    starts = np.arange(first, final_start + 1, hop_n, dtype=np.int64)
    rows = []
    for start in starts:
        end = int(start + frame_n)
        level, at_floor = ac_rms_dbfs(
            waveform[int(start) : end],
            parameters.dbfs_floor_db,
        )
        rows.append(
            {
                "frame_start_sec": float(start / fs),
                "frame_end_sec": float(end / fs),
                "frame_mid_sec": float((start + frame_n / 2.0) / fs),
                "ac_rms_dbfs": level,
                "at_digital_floor": bool(at_floor),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def boundary_envelope_trace(
    waveform: np.ndarray,
    fs: int,
    speech_offset_sec: float,
    pause_end_sec: float,
    *,
    parameters: QREVParameters = DEFAULT_PARAMETERS,
) -> pd.DataFrame:
    """Create an on-demand post-offset envelope trace for figures and audit."""

    end = min(
        float(pause_end_sec),
        float(speech_offset_sec) + parameters.persistence_horizon_ms / 1000.0,
    )
    trace = _frame_levels(waveform, fs, float(speech_offset_sec), end, parameters)
    if len(trace):
        trace["relative_start_sec"] = trace["frame_start_sec"] - float(speech_offset_sec)
        trace["relative_end_sec"] = trace["frame_end_sec"] - float(speech_offset_sec)
        trace["relative_mid_sec"] = trace["frame_mid_sec"] - float(speech_offset_sec)
    return trace


def _select_relative_frames(
    trace: pd.DataFrame,
    start_sec: float,
    end_sec: float,
) -> pd.DataFrame:
    if trace.empty:
        return trace.copy()
    return trace.loc[
        trace["relative_start_sec"].ge(float(start_sec) - 1e-12)
        & trace["relative_end_sec"].le(float(end_sec) + 1e-12)
    ].copy()


def _nonfloor_levels(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)
    mask = ~frame["at_digital_floor"].astype(bool)
    values = pd.to_numeric(frame.loc[mask, "ac_rms_dbfs"], errors="coerce")
    return values.loc[np.isfinite(values)].to_numpy(float)


def _persistence(
    trace: pd.DataFrame,
    floor_dbfs: float,
    parameters: QREVParameters,
) -> tuple[float, bool]:
    """Return first stable within-floor time and right-censor indicator."""

    horizon = parameters.persistence_horizon_ms / 1000.0
    local = _select_relative_frames(trace, 0.0, horizon)
    if len(local) < parameters.minimum_persistence_frame_count:
        return np.nan, False
    within = (
        pd.to_numeric(local["ac_rms_dbfs"], errors="coerce").to_numpy(float)
        <= float(floor_dbfs) + parameters.persistence_threshold_db
    )
    run = 0
    for index, flag in enumerate(within):
        run = run + 1 if bool(flag) else 0
        if run >= parameters.persistence_consecutive_frames:
            first = index - parameters.persistence_consecutive_frames + 1
            return float(local.iloc[first]["relative_mid_sec"]), False
    return float(horizon), True


def _empty_boundary_row(boundary: dict, logical_recording_id: str) -> dict:
    return {
        "logical_recording_id": logical_recording_id,
        "boundary_id": (
            f"{logical_recording_id}:boundary-{int(boundary['boundary_index']):04d}"
        ),
        **boundary,
        "tail_eligible": False,
        "persistence_eligible": False,
        "decay_eligible": False,
        "exclusion_reason": "",
        "early_frame_count": 0,
        "floor_frame_count": 0,
        "persistence_frame_count": 0,
        "decay_frame_count": 0,
        "floor_frame_fraction": np.nan,
        "floor_dbfs": np.nan,
        "floor_iqr_db": np.nan,
        "floor_stable": False,
        "early_level_dbfs": np.nan,
        "tail_excess_100ms_db": np.nan,
        "tail_persistence_sec": np.nan,
        "tail_persistence_right_censored": False,
        "signed_decay_slope_db_per_sec": np.nan,
        "decay_dynamic_range_db": np.nan,
        "downward_decay_rate_db_per_sec": np.nan,
        "nondecreasing_decay": False,
    }


def measure_boundary(
    waveform: np.ndarray,
    fs: int,
    boundary: dict,
    *,
    logical_recording_id: str,
    parameters: QREVParameters = DEFAULT_PARAMETERS,
) -> dict:
    """Measure one natural speech-to-pause boundary."""

    row = _empty_boundary_row(boundary, logical_recording_id)
    pause_duration = float(boundary["pause_duration_sec"])
    floor_end_sec = parameters.floor_end_ms / 1000.0
    horizon_sec = parameters.persistence_horizon_ms / 1000.0
    if pause_duration < floor_end_sec:
        row["exclusion_reason"] = "pause_shorter_than_floor_window"
        return row

    offset = float(boundary["speech_offset_sec"])
    trace = boundary_envelope_trace(
        waveform,
        fs,
        offset,
        float(boundary["pause_end_sec"]),
        parameters=parameters,
    )
    early = _select_relative_frames(
        trace,
        parameters.early_tail_start_ms / 1000.0,
        parameters.early_tail_end_ms / 1000.0,
    )
    floor_frames = _select_relative_frames(
        trace,
        parameters.floor_start_ms / 1000.0,
        parameters.floor_end_ms / 1000.0,
    )
    decay = _select_relative_frames(
        trace,
        parameters.decay_start_ms / 1000.0,
        parameters.decay_end_ms / 1000.0,
    )
    early_levels = _nonfloor_levels(early)
    floor_levels = _nonfloor_levels(floor_frames)
    floor_fraction = (
        float(floor_frames["at_digital_floor"].astype(bool).mean())
        if len(floor_frames)
        else np.nan
    )
    row.update(
        {
            "early_frame_count": int(len(early_levels)),
            "floor_frame_count": int(len(floor_levels)),
            "persistence_frame_count": int(len(trace)),
            "decay_frame_count": int(len(_nonfloor_levels(decay))),
            "floor_frame_fraction": floor_fraction,
        }
    )
    if len(early_levels) < parameters.minimum_early_frame_count:
        row["exclusion_reason"] = "insufficient_early_frames"
        return row
    if len(floor_levels) < parameters.minimum_floor_frame_count:
        row["exclusion_reason"] = "insufficient_nonfloor_floor_frames"
        return row
    if (
        np.isfinite(floor_fraction)
        and floor_fraction > parameters.maximum_floor_frame_fraction
    ):
        row["exclusion_reason"] = "digital_floor_censored"
        return row

    floor_dbfs = float(np.median(floor_levels))
    floor_iqr = float(np.quantile(floor_levels, 0.75) - np.quantile(floor_levels, 0.25))
    floor_stable = bool(floor_iqr <= parameters.maximum_floor_iqr_db)
    row.update(
        {
            "floor_dbfs": floor_dbfs,
            "floor_iqr_db": floor_iqr,
            "floor_stable": floor_stable,
            "early_level_dbfs": float(np.median(early_levels)),
        }
    )
    if not floor_stable:
        row["exclusion_reason"] = "unstable_late_pause_floor"
        return row

    row["tail_eligible"] = True
    row["tail_excess_100ms_db"] = (
        row["early_level_dbfs"] - row["floor_dbfs"]
    )

    decay_levels = _nonfloor_levels(decay)
    if len(decay_levels) >= parameters.minimum_decay_frame_count:
        decay_valid = decay.loc[~decay["at_digital_floor"].astype(bool)].copy()
        decay_valid = decay_valid.loc[
            np.isfinite(pd.to_numeric(decay_valid["ac_rms_dbfs"], errors="coerce"))
        ]
        times = decay_valid["relative_mid_sec"].to_numpy(float)
        levels = decay_valid["ac_rms_dbfs"].to_numpy(float)
        dynamic_range = float(np.quantile(levels, 0.90) - np.quantile(levels, 0.10))
        slope = float(stats.theilslopes(levels, times, alpha=0.95).slope)
        row["signed_decay_slope_db_per_sec"] = slope
        row["decay_dynamic_range_db"] = dynamic_range
        row["nondecreasing_decay"] = bool(slope >= 0.0)
        if slope < 0.0 and dynamic_range >= parameters.minimum_decay_dynamic_range_db:
            row["decay_eligible"] = True
            row["downward_decay_rate_db_per_sec"] = -slope

    if pause_duration >= horizon_sec:
        value, censored = _persistence(trace, floor_dbfs, parameters)
        if np.isfinite(value):
            row["persistence_eligible"] = True
            row["tail_persistence_sec"] = value
            row["tail_persistence_right_censored"] = bool(censored)
    return row


def _boundary_support_tier(
    count: int,
    pause_support_sec: float,
    parameters: QREVParameters,
) -> str:
    if (
        count >= parameters.high_boundary_count
        and pause_support_sec >= parameters.high_pause_support_sec
    ):
        return "high"
    if (
        count >= parameters.moderate_boundary_count
        and pause_support_sec >= parameters.moderate_pause_support_sec
    ):
        return "moderate"
    if (
        count >= parameters.minimum_tail_boundary_count
        and pause_support_sec >= parameters.minimum_tail_pause_support_sec
    ):
        return "minimum"
    return "unavailable"


def estimate_srmr_working_set_mb(sample_count: int, parameters: QREVParameters) -> float:
    """Conservative working-set estimate for normalized-fast SRMRpy.

    The estimate covers the waveform, 23-channel 400-Hz gammatonegram,
    one eight-channel modulation-filter output, the modulation-energy tensor,
    and a 2.5x implementation/FFT safety factor.
    """

    samples = max(0, int(sample_count))
    gt_steps = int(ceil(samples / 40.0))
    modulation_frames = max(
        1,
        int(ceil((gt_steps - ceil(0.256 * 400.0)) / ceil(0.064 * 400.0))) + 1,
    )
    float64_values = (
        samples
        + parameters.srmr_n_cochlear_filters * gt_steps
        + 8 * gt_steps
        + parameters.srmr_n_cochlear_filters * 8 * modulation_frames
    )
    return float(float64_values * 8.0 * 2.5 / (1024.0**2))


def compute_srmr_norm(
    waveform: np.ndarray,
    fs: int,
    *,
    parameters: QREVParameters = DEFAULT_PARAMETERS,
) -> float:
    """Compute the pinned normalized-fast SRMRpy reference implementation."""

    if int(fs) != parameters.analysis_sample_rate_hz:
        raise ValueError(
            f"SRMR requires {parameters.analysis_sample_rate_hz} Hz; got {fs}."
        )
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("SRMR requires a mono waveform.")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("SRMR waveform is empty or non-finite.")
    try:
        installed_gammatone = version("Gammatone")
    except PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            "QREV normalized SRMR requires Gammatone 1.0.3."
        ) from exc
    if installed_gammatone != SRMR_GAMMATONE_VERSION:
        raise RuntimeError(
            "QREV normalized SRMR requires exactly Gammatone "
            f"{SRMR_GAMMATONE_VERSION}; found {installed_gammatone}."
        )
    from ._vendor.srmrpy import srmr

    score, energy = srmr(
        values,
        int(fs),
        n_cochlear_filters=parameters.srmr_n_cochlear_filters,
        low_freq=parameters.srmr_low_frequency_hz,
        min_cf=parameters.srmr_min_modulation_cf_hz,
        max_cf=parameters.srmr_max_modulation_cf_hz,
        fast=parameters.srmr_fast,
        norm=parameters.srmr_normalized,
    )
    result = float(score)
    del energy
    if not np.isfinite(result) or result <= 0:
        raise RuntimeError(f"SRMR returned an invalid ratio: {result!r}")
    return result


def extract_qrev(
    waveform: np.ndarray,
    fs: int,
    *,
    strict_speech: Iterable[TimeInterval],
    logical_recording_id: str = "recording",
    parameters: QREVParameters = DEFAULT_PARAMETERS,
    compute_srmr: bool = True,
) -> QREVExtraction:
    """Extract the four QREV v3 analysis features and compact audit ledger."""

    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("QREV requires a mono analysis waveform.")
    if int(fs) != parameters.analysis_sample_rate_hz:
        raise ValueError(
            f"QREV requires {parameters.analysis_sample_rate_hz} Hz analysis audio; got {fs}."
        )
    if not np.isfinite(samples).all():
        raise ValueError("Waveform contains non-finite samples.")
    duration_sec = len(samples) / float(fs)
    speech = merge_intervals(strict_speech, duration_sec)
    speech_support_sec = float(sum(item.duration_sec for item in speech))
    boundaries = internal_pause_boundaries(speech, duration_sec)
    boundary_rows = [
        measure_boundary(
            samples,
            fs,
            boundary,
            logical_recording_id=logical_recording_id,
            parameters=parameters,
        )
        for boundary in boundaries
    ]
    ledger = pd.DataFrame(boundary_rows)

    if ledger.empty:
        tail_valid = pd.DataFrame()
        persistence_valid = pd.DataFrame()
        decay_valid = pd.DataFrame()
    else:
        tail_valid = ledger.loc[ledger["tail_eligible"].astype(bool)].copy()
        persistence_valid = ledger.loc[
            ledger["persistence_eligible"].astype(bool)
        ].copy()
        decay_valid = ledger.loc[ledger["decay_eligible"].astype(bool)].copy()

    tail_count = len(tail_valid)
    persistence_count = len(persistence_valid)
    decay_count = len(decay_valid)
    tail_pause_support = (
        float(tail_valid["pause_duration_sec"].sum()) if tail_count else 0.0
    )
    persistence_observation_support = (
        persistence_count * parameters.persistence_horizon_ms / 1000.0
    )
    tail_raw = (
        float(np.median(tail_valid["tail_excess_100ms_db"]))
        if tail_count
        else np.nan
    )
    persistence_raw = (
        float(np.median(persistence_valid["tail_persistence_sec"]))
        if persistence_count
        else np.nan
    )
    persistence_censored_fraction = (
        float(
            persistence_valid["tail_persistence_right_censored"].astype(bool).mean()
        )
        if persistence_count
        else np.nan
    )
    persistence_median_censored = bool(
        np.isfinite(persistence_raw)
        and np.isclose(
            persistence_raw,
            parameters.persistence_horizon_ms / 1000.0,
        )
    )
    decay_raw = (
        float(np.median(decay_valid["downward_decay_rate_db_per_sec"]))
        if decay_count
        else np.nan
    )

    tail_available = bool(
        tail_count >= parameters.minimum_tail_boundary_count
        and tail_pause_support >= parameters.minimum_tail_pause_support_sec
    )
    persistence_available = bool(
        persistence_count >= parameters.minimum_persistence_boundary_count
    )
    decay_available = bool(decay_count >= parameters.minimum_decay_boundary_count)

    srmr_value = np.nan
    srmr_status = "not_requested"
    task_start = speech[0].start_sec if speech else np.nan
    task_end = speech[-1].end_sec if speech else np.nan
    task_span_sec = (
        float(task_end - task_start)
        if np.isfinite(task_start) and np.isfinite(task_end)
        else 0.0
    )
    task_left = max(0, int(floor(task_start * fs))) if speech else 0
    task_right = min(len(samples), int(ceil(task_end * fs))) if speech else 0
    srmr_sample_count = max(0, task_right - task_left)
    srmr_memory_mb = estimate_srmr_working_set_mb(srmr_sample_count, parameters)
    if compute_srmr:
        if (
            speech_support_sec < parameters.minimum_srmr_speech_support_sec
            or task_span_sec < parameters.minimum_srmr_task_span_sec
        ):
            srmr_status = "insufficient_support"
        elif srmr_memory_mb > parameters.maximum_srmr_estimated_memory_mb:
            srmr_status = "resource_limit"
        else:
            try:
                srmr_value = compute_srmr_norm(
                    samples[task_left:task_right],
                    fs,
                    parameters=parameters,
                )
                srmr_status = "measured"
            except ModuleNotFoundError:
                srmr_status = "dependency_unavailable"
            except (ValueError, RuntimeError, FloatingPointError):
                srmr_status = "computation_failed"

    values = {
        "qrev_tail_excess_100ms_db": tail_raw if tail_available else np.nan,
        "qrev_tail_persistence_median_sec": (
            persistence_raw if persistence_available else np.nan
        ),
        "qrev_downward_decay_rate_db_per_sec": (
            decay_raw if decay_available else np.nan
        ),
        "qrev_srmr_norm": srmr_value,
    }
    tail_tier = _boundary_support_tier(
        tail_count,
        tail_pause_support,
        parameters,
    )
    persistence_tier = _boundary_support_tier(
        persistence_count,
        persistence_observation_support,
        parameters,
    )
    decay_tier = _boundary_support_tier(
        decay_count,
        tail_pause_support,
        parameters,
    )
    recording = {
        "logical_recording_id": logical_recording_id,
        "qrev_measurement_version": MEASUREMENT_VERSION,
        "qrev_signal_view": "mono_dc_removed_16k_analysis; framewise_AC_RMS",
        "qrev_speech_source": "frozen_strict_speech",
        "qrev_internal_boundary_count": len(boundaries),
        "qrev_tail_valid_boundary_count": tail_count,
        "qrev_tail_valid_pause_support_sec": tail_pause_support,
        "qrev_persistence_valid_boundary_count": persistence_count,
        "qrev_persistence_observation_support_sec": persistence_observation_support,
        "qrev_persistence_right_censored_fraction": persistence_censored_fraction,
        "qrev_persistence_recording_median_censored": persistence_median_censored,
        "qrev_decay_valid_boundary_count": decay_count,
        "qrev_nondecreasing_decay_boundary_fraction": (
            float(ledger["nondecreasing_decay"].astype(bool).mean())
            if len(ledger)
            else np.nan
        ),
        "qrev_floor_unstable_boundary_fraction": (
            float(
                (
                    ledger["exclusion_reason"].astype(str)
                    == "unstable_late_pause_floor"
                ).mean()
            )
            if len(ledger)
            else np.nan
        ),
        "qrev_srmr_variant": SRMR_VARIANT,
        "qrev_srmr_upstream_commit": SRMR_UPSTREAM_COMMIT,
        "qrev_srmr_task_span_sec": task_span_sec,
        "qrev_srmr_strict_speech_support_sec": speech_support_sec,
        "qrev_srmr_estimated_working_set_mb": srmr_memory_mb,
        "qrev_tail_excess_100ms_db_raw_estimate": tail_raw,
        "qrev_tail_persistence_median_sec_raw_estimate": persistence_raw,
        "qrev_downward_decay_rate_db_per_sec_raw_estimate": decay_raw,
        **values,
    }
    statuses = {
        "qrev_tail_excess_100ms_db": (
            "measured" if tail_available else "insufficient_support"
        ),
        "qrev_tail_persistence_median_sec": (
            "right_censored_at_horizon"
            if persistence_available and persistence_median_censored
            else "measured"
            if persistence_available
            else "insufficient_support"
        ),
        "qrev_downward_decay_rate_db_per_sec": (
            "measured"
            if decay_available
            else "no_valid_downward_decay"
            if tail_count >= parameters.minimum_tail_boundary_count
            else "insufficient_support"
        ),
        "qrev_srmr_norm": srmr_status,
    }
    tiers = {
        "qrev_tail_excess_100ms_db": tail_tier,
        "qrev_tail_persistence_median_sec": persistence_tier,
        "qrev_downward_decay_rate_db_per_sec": decay_tier,
        "qrev_srmr_norm": (
            "minimum" if srmr_status == "measured" else "unavailable"
        ),
    }
    for feature in ANALYSIS_FEATURES:
        recording[f"{feature}_status"] = statuses[feature]
        recording[f"{feature}_support_tier"] = tiers[feature]

    primary_available = sum(np.isfinite(values[name]) for name in PRIMARY_FEATURES)
    recording["qrev_primary_available_count"] = int(primary_available)
    recording["qrev_primary_analysis_eligible"] = bool(primary_available > 0)
    if primary_available == len(PRIMARY_FEATURES):
        recording["qrev_family_status"] = "all_primary_available"
    elif primary_available:
        recording["qrev_family_status"] = "partial_primary_available"
    else:
        recording["qrev_family_status"] = "primary_unavailable"
    return QREVExtraction(recording=recording, boundary_ledger=ledger)


def apply_gain_db(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply unclipped digital gain for deterministic controls."""

    return np.asarray(waveform, dtype=np.float64) * (
        10.0 ** (float(gain_db) / 20.0)
    )
