"""QDIST v3.1.1: native-waveform hard-clipping and saturation evidence.

QDIST is deliberately narrower than the observation model's general nonlinear
operator.  It detects time-domain plateaus whose local morphology and
polarity-specific edge distribution are jointly compatible with hard clipping
or saturation.  It does not estimate total harmonic distortion,
intermodulation distortion, soft clipping, dynamic-range compression,
limiting, automatic gain control, codec distortion, or perceptual distortion.

The canonical input is the first native-rate decoded audio stream with all
channels preserved.  Resampling, channel averaging, amplitude normalization,
filtering, denoising, interpolation, DC removal, and codec re-encoding are not
part of the canonical measurement path.  Frozen segmentation may define the
continuous task span, but speech intervals are never concatenated.

The three analysis features are reconstructable views of one accepted plateau
system.  Candidate, accepted-plateau, and merged-episode ledgers are returned so
that every recording-level value and every rejection can be audited.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

MEASUREMENT_VERSION = "qdist-v3.1.1"

ANALYSIS_FEATURES = (
    "qdist_hard_clipped_frame_fraction",
    "qdist_hard_clip_event_rate_per_min",
    "qdist_hard_clipped_sample_fraction",
)

PRIMARY_FEATURES = ANALYSIS_FEATURES[:2]
SECONDARY_FEATURES = ANALYSIS_FEATURES[2:]


@dataclass(frozen=True, order=True)
class TimeInterval:
    """Half-open interval in original/native recording seconds."""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


@dataclass(frozen=True)
class NativeSignalProvenance:
    """Metadata required to interpret the decoded native-rate waveform.

    The extractor accepts direct arrays for deterministic validation, so most
    fields are optional.  Production extraction should populate them from
    FFprobe/FFmpeg and source-file provenance.
    """

    native_view_verified: bool = True
    known_preprocessing_applied: bool = False
    codec_name: str | None = None
    sample_format: str | None = None
    bits_per_raw_sample: int | None = None
    container_format: str | None = None
    channel_layout: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    decoded_sha256: str | None = None
    decoder: str | None = None
    decoder_version: str | None = None
    decode_arguments: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QDISTParameters:
    """Prespecified engineering and minimum-support parameters for QDIST v3.1.1.

    These are candidate-release operating parameters.  The validation notebook
    must characterize their neighborhood before a family freeze.  A support
    tier describes exposure only; it is not a claim of empirical robustness.
    """

    measurement_version: str = MEASUREMENT_VERSION

    # Exposure and aggregation.
    minimum_task_span_sec: float = 3.0
    minimum_finite_fraction: float = 1.0
    frame_length_ms: float = 30.0
    minimum_complete_frame_count: int = 100
    standard_exposure_sec: float = 10.0
    high_exposure_sec: float = 30.0

    # Candidate plateau geometry.
    minimum_plateau_samples: int = 4
    minimum_singleton_plateau_samples: int = 7
    maximum_plateau_duration_ms: float = 10.0
    maximum_plateau_unique_levels: int = 3
    absolute_flat_tolerance: float = 2.0e-7
    relative_flat_tolerance: float = 5.0e-5
    integer_flat_tolerance_steps: float = 0.25
    floating_cluster_tolerance_fraction: float = 1.0e-4
    integer_cluster_tolerance_steps: float = 0.75

    # Magnitude and local morphology. Candidate generation uses a permissive
    # recording-relative floor. Final acceptance then follows one of two
    # prespecified paths: (i) a strong recording-edge path, or (ii) a lower-level
    # path that requires extensive repeated edge support. Both paths also require
    # local prominence. This preserves sensitivity to genuine saturation in a
    # lower gain state while rejecting isolated quantized natural extrema.
    robust_peak_quantile: float = 0.999
    candidate_generation_minimum_edge_to_robust_peak_ratio: float = 0.25
    minimum_edge_to_robust_peak_ratio: float = 0.45
    low_level_minimum_cluster_candidates: int = 4
    low_level_minimum_cluster_plateau_samples: int = 24
    low_level_minimum_edge_zone_samples: int = 24
    minimum_edge_to_local_peak_ratio: float = 0.90
    context_ms: float = 4.0
    minimum_context_peak_ratio: float = 0.50
    minimum_transition_relative_to_edge: float = 5.0e-4
    minimum_transition_tolerance_multiples: float = 4.0
    minimum_transition_quantization_steps: float = 2.0
    maximum_plateau_slope_tolerance_multiples: float = 1.0

    # Polarity-specific edge evidence is evaluated in local neighborhoods
    # around each level cluster. This permits transient clipping at one gain
    # state even when other recording regions legitimately exceed that level.
    edge_evidence_context_ms: float = 4.0
    edge_evidence_max_component_gap_ms: float = 50.0
    minimum_edge_zone_samples: int = 8
    minimum_level_cluster_samples: int = 8
    edge_zone_tolerance_multiples: float = 1.5
    interior_shell_inner_multiples: float = 1.5
    interior_shell_outer_multiples: float = 4.5
    minimum_edge_to_interior_ratio: float = 2.0
    minimum_edge_excess_samples: int = 4
    maximum_beyond_edge_fraction: float = 1.0e-6
    maximum_beyond_edge_samples: int = 3

    # Extra conservatism for coarse quantization.
    coarse_quantization_bits: int = 12
    coarse_minimum_cluster_candidates: int = 2
    coarse_minimum_singleton_samples: int = 12
    coarse_minimum_edge_to_interior_ratio: float = 3.0

    # Ambiguous two-level/square-like streams are not interpreted as clipping.
    square_like_edge_fraction: float = 0.80
    square_like_maximum_unique_levels: int = 16

    # Episode construction and diagnostics.
    episode_merge_gap_ms: float = 20.0
    near_full_scale_threshold: float = 0.99
    poisson_confidence: float = 0.95
    random_seed: int = 20260731

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def parameter_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_PARAMETERS = QDISTParameters()


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
    source_ledger: str
    supporting_references: str


FEATURE_DEFINITIONS = (
    FeatureDefinition(
        "qdist_hard_clipped_frame_fraction",
        "Hard-clipped frame fraction",
        "time-localized clipping prevalence",
        "primary",
        "fraction [0,1]",
        "literature-informed engineering detector; study-specific aggregation",
        "Fraction of non-overlapping native task-span frames intersecting at least "
        "one accepted clipping plateau on any channel.",
        "Higher indicates a larger fraction of task-span frames containing accepted "
        "hard-clipping morphology.",
        "Hard-clipping/saturation evidence only; not total nonlinear or perceptual distortion.",
        "At least 3 s finite task-span exposure and 100 complete 30-ms frames.",
        "Quantization, naturally flat extrema, square-like signals, codec smearing, "
        "postprocessing, and task-span errors.",
        "accepted plateau ledger",
        "Li et al. (2014); Patel et al. (2018); Goldsack et al. (2020)",
    ),
    FeatureDefinition(
        "qdist_hard_clip_event_rate_per_min",
        "Hard-clipping episode rate",
        "clipping episode frequency",
        "primary event metric",
        "events/min",
        "literature-informed engineering detector; study-specific aggregation",
        "Number of merged accepted clipping episodes per minute of finite native "
        "task-span exposure.",
        "Higher indicates more distinct accepted clipping episodes per unit exposure.",
        "Episode grouping depends on the frozen merge rule; not an independent detector.",
        "At least 3 s finite task-span exposure; event count and exposure are retained.",
        "Merge-gap choice, fragmented codec-smeared plateaus, quantization, and source content.",
        "merged episode ledger",
        "Li et al. (2014); Patel et al. (2018); Goldsack et al. (2020)",
    ),
    FeatureDefinition(
        "qdist_hard_clipped_sample_fraction",
        "Hard-clipped channel-sample fraction",
        "clipped waveform burden",
        "secondary",
        "fraction [0,1]",
        "literature-informed engineering detector; study-specific aggregation",
        "Fraction of finite eligible native channel-samples covered by accepted clipping plateaus.",
        "Higher indicates greater accepted plateau support across channels.",
        "Channel-sample burden, not affected-time fraction and not an independent detector.",
        "At least 3 s finite task-span exposure and finite fraction at least 0.999.",
        "Channel count, quantization, codec smearing, and event-boundary tolerance.",
        "accepted plateau ledger",
        "Li et al. (2014); Patel et al. (2018); Goldsack et al. (2020)",
    ),
)


CANDIDATE_COLUMNS = [
    "logical_recording_id",
    "candidate_id",
    "channel_index",
    "polarity",
    "cluster_id",
    "start_sample_task",
    "end_sample_task_exclusive",
    "start_sample_native",
    "end_sample_native_exclusive",
    "start_sec_native",
    "end_sec_native",
    "duration_sec",
    "sample_count",
    "candidate_level",
    "candidate_abs_level",
    "unique_level_count",
    "plateau_range",
    "median_abs_first_difference",
    "plateau_slope_per_sample",
    "entry_signed_transition",
    "exit_signed_transition",
    "transition_threshold",
    "pre_context_peak_abs",
    "post_context_peak_abs",
    "local_context_peak_abs",
    "candidate_to_context_ratio",
    "channel_robust_peak_abs",
    "candidate_to_robust_peak_ratio",
    "flat_tolerance",
    "cluster_tolerance",
    "quantization_step",
    "inferred_bits_per_sample",
    "cluster_candidate_count",
    "cluster_plateau_sample_count",
    "edge_zone_sample_count",
    "interior_shell_sample_count",
    "edge_to_interior_ratio",
    "edge_excess_samples",
    "beyond_edge_sample_count",
    "beyond_edge_fraction",
    "allowed_beyond_edge_samples",
    "edge_zone_fraction",
    "morphology_pass",
    "duration_pass",
    "recording_magnitude_pass",
    "strong_recording_magnitude_pass",
    "low_level_repeated_edge_pass",
    "magnitude_path",
    "local_magnitude_pass",
    "magnitude_pass",
    "context_pass",
    "transition_pass",
    "cluster_support_pass",
    "edge_support_pass",
    "edge_ratio_pass",
    "edge_excess_pass",
    "terminal_edge_pass",
    "quantization_guard_pass",
    "square_like_guard_pass",
    "accepted",
    "rejection_reason",
    "measurement_version",
    "parameter_hash",
]

EPISODE_COLUMNS = [
    "logical_recording_id",
    "episode_id",
    "start_sample_task",
    "end_sample_task_exclusive",
    "start_sample_native",
    "end_sample_native_exclusive",
    "start_sec_native",
    "end_sec_native",
    "duration_sec",
    "plateau_count",
    "constituent_candidate_ids",
    "channel_indices",
    "polarity_composition",
    "channel_sample_count",
    "any_channel_time_sample_count",
    "intersected_frame_count",
    "maximum_internal_merge_gap_samples",
    "merge_gap_ms",
    "measurement_version",
    "parameter_hash",
]


@dataclass
class QDISTExtraction:
    """Recording-level output and reconstructable QDIST audit ledgers."""

    recording: dict[str, Any]
    candidate_ledger: pd.DataFrame
    accepted_plateau_ledger: pd.DataFrame
    episode_ledger: pd.DataFrame
    edge_ledger: pd.DataFrame


def feature_registry_frame() -> pd.DataFrame:
    """Return the immutable one-row-per-analysis-feature registry."""

    frame = pd.DataFrame([asdict(item) for item in FEATURE_DEFINITIONS])
    if tuple(frame["name"]) != ANALYSIS_FEATURES:
        raise RuntimeError("QDIST registry and ANALYSIS_FEATURES are inconsistent")
    return frame


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_COLUMNS)


def _empty_episodes() -> pd.DataFrame:
    return pd.DataFrame(columns=EPISODE_COLUMNS)


def _as_2d(waveform: np.ndarray) -> np.ndarray:
    samples = np.asarray(waveform)
    if samples.ndim == 1:
        samples = samples[:, None]
    elif samples.ndim != 2:
        raise ValueError("QDIST waveform must have shape samples or samples x channels.")
    if samples.shape[0] == 0 or samples.shape[1] == 0:
        return np.empty((0, max(1, samples.shape[1] if samples.ndim == 2 else 1)))
    return samples.astype(np.float64, copy=False)


def _normalise_task_span(
    duration_sec: float,
    task_span: TimeInterval | None,
) -> TimeInterval:
    if task_span is None:
        return TimeInterval(0.0, float(duration_sec))
    start = float(task_span.start_sec)
    end = float(task_span.end_sec)
    if not np.isfinite(start) or not np.isfinite(end):
        return TimeInterval(0.0, 0.0)
    return TimeInterval(
        min(max(start, 0.0), float(duration_sec)),
        min(max(end, 0.0), float(duration_sec)),
    )


def _task_span_indices(span: TimeInterval, fs: int, sample_count: int) -> tuple[int, int]:
    start = int(np.floor(span.start_sec * fs + 1e-12))
    end = int(np.ceil(span.end_sec * fs - 1e-12))
    start = min(max(start, 0), sample_count)
    end = min(max(end, 0), sample_count)
    return start, max(start, end)


def _codec_is_lossy(codec_name: str | None) -> bool:
    if not codec_name:
        return False
    codec = codec_name.lower()
    lossless = (
        codec.startswith("pcm_")
        or codec in {"flac", "alac", "wavpack", "ape"}
    )
    return not lossless


def infer_integer_bits(provenance: NativeSignalProvenance | None) -> int | None:
    """Infer exact PCM bit depth only when source metadata supports it."""

    if provenance is None or _codec_is_lossy(provenance.codec_name):
        return None
    raw = provenance.bits_per_raw_sample
    if raw is not None:
        try:
            bits = int(raw)
            if 2 <= bits <= 64:
                return bits
        except (TypeError, ValueError):
            pass
    text = f"{provenance.codec_name or ''} {provenance.sample_format or ''}".lower()
    for bits in (64, 32, 24, 16, 8):
        if str(bits) in text and ("pcm" in text or text.strip().startswith(("s", "u"))):
            return bits
    return None


def quantization_step_from_bits(bits: int | None) -> float:
    """Return the FFmpeg normalized signed-PCM code step when bit depth is known."""

    if bits is None:
        return np.nan
    return float(2.0 ** (1 - int(bits)))


def decoded_waveform_sha256(waveform: np.ndarray) -> str:
    """Stable hash of shape, dtype-normalized values, and finite-mask geometry."""

    samples = _as_2d(waveform)
    canonical = np.ascontiguousarray(samples.astype("<f8", copy=False))
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def poisson_rate_interval(
    count: int,
    exposure_sec: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact central Poisson interval converted to events per minute."""

    if not np.isfinite(exposure_sec) or exposure_sec <= 0:
        return np.nan, np.nan
    alpha = 1.0 - float(confidence)
    lower_count = 0.0 if count == 0 else 0.5 * stats.chi2.ppf(alpha / 2, 2 * count)
    upper_count = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (count + 1))
    factor = 60.0 / float(exposure_sec)
    return float(lower_count * factor), float(upper_count * factor)


def _support_tier(duration_sec: float, parameters: QDISTParameters) -> str:
    if duration_sec < parameters.minimum_task_span_sec:
        return "insufficient"
    if duration_sec >= parameters.high_exposure_sec:
        return "high"
    if duration_sec >= parameters.standard_exposure_sec:
        return "standard"
    return "low"


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs where a one-dimensional Boolean mask is true."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        return []
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _count_true_runs(mask: np.ndarray) -> int:
    """Count half-open true runs without materialising Python interval objects."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        return 0
    return int(values[0]) + int(np.count_nonzero(~values[:-1] & values[1:]))


def _channel_tolerances(
    values: np.ndarray,
    quantization_step: float,
    parameters: QDISTParameters,
) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.nan, np.nan, np.nan
    robust_peak = float(
        np.quantile(np.abs(finite), parameters.robust_peak_quantile)
    )
    floating_tolerance = max(
        parameters.absolute_flat_tolerance,
        robust_peak * parameters.relative_flat_tolerance,
    )
    if np.isfinite(quantization_step) and quantization_step > 0:
        flat_tolerance = max(
            parameters.absolute_flat_tolerance,
            quantization_step * parameters.integer_flat_tolerance_steps,
        )
        cluster_tolerance = max(
            flat_tolerance,
            quantization_step * parameters.integer_cluster_tolerance_steps,
        )
    else:
        flat_tolerance = floating_tolerance
        cluster_tolerance = max(
            flat_tolerance,
            robust_peak * parameters.floating_cluster_tolerance_fraction,
        )
    return float(robust_peak), float(flat_tolerance), float(cluster_tolerance)


def _candidate_rows_for_channel(
    values: np.ndarray,
    *,
    channel_index: int,
    fs: int,
    task_start_sample: int,
    logical_recording_id: str,
    quantization_step: float,
    inferred_bits: int | None,
    parameters: QDISTParameters,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    finite = np.isfinite(values)
    robust_peak, flat_tolerance, cluster_tolerance = _channel_tolerances(
        values, quantization_step, parameters
    )
    diagnostics = {
        "logical_recording_id": logical_recording_id,
        "channel_index": int(channel_index),
        "inferred_bits_per_sample": inferred_bits,
        "quantization_step": quantization_step,
        "channel_robust_peak_abs": robust_peak,
        "flat_tolerance": flat_tolerance,
        "cluster_tolerance": cluster_tolerance,
        "finite_sample_count": int(finite.sum()),
        "near_full_scale_fraction": (
            float(np.mean(np.abs(values[finite]) >= parameters.near_full_scale_threshold))
            if finite.any()
            else np.nan
        ),
    }
    if not finite.any() or not np.isfinite(robust_peak) or robust_peak <= 0:
        return [], diagnostics

    # Candidate generation is intentionally restricted to the recording-level
    # plausible amplitude region. Earlier versions enumerated every locally
    # flat run and then allowed local prominence alone to satisfy the magnitude
    # gate. That created >1 million low-level quantisation candidates in the
    # cohort and admitted implausible near-zero plateaus. QDIST v3.1.1 requires
    # both a recording-relative plausibility floor and local edge prominence.
    pair_flat_all = (
        finite[:-1]
        & finite[1:]
        & (np.abs(np.diff(values)) <= flat_tolerance)
        & (np.signbit(values[:-1]) == np.signbit(values[1:]))
        & (np.abs(values[:-1]) > flat_tolerance)
        & (np.abs(values[1:]) > flat_tolerance)
    )
    recording_edge_mask = finite & (
        np.abs(values)
        >= (
            parameters.candidate_generation_minimum_edge_to_robust_peak_ratio
            * robust_peak
        )
    )
    pair_flat = (
        pair_flat_all
        & recording_edge_mask[:-1]
        & recording_edge_mask[1:]
    )
    diagnostics.update({
        "flat_run_count_all_amplitudes": _count_true_runs(pair_flat_all),
        "flat_run_count_at_recording_edge": _count_true_runs(pair_flat),
        "flat_pair_count_all_amplitudes": int(pair_flat_all.sum()),
        "flat_pair_count_at_recording_edge": int(pair_flat.sum()),
    })
    context_samples = max(1, int(round(parameters.context_ms * fs / 1000.0)))
    maximum_plateau_samples = max(
        parameters.minimum_plateau_samples,
        int(np.ceil(parameters.maximum_plateau_duration_ms * fs / 1000.0)),
    )
    rows: list[dict[str, Any]] = []
    for pair_start, pair_end in _true_runs(pair_flat):
        start = int(pair_start)
        end = int(pair_end + 1)  # pair-run -> sample-run, half-open
        sample_count = end - start
        if sample_count < parameters.minimum_plateau_samples:
            continue
        segment = values[start:end]
        if not np.isfinite(segment).all():
            continue
        candidate_level = float(np.median(segment))
        polarity = 1 if candidate_level > 0 else -1
        abs_level = abs(candidate_level)
        pre_start = max(0, start - context_samples)
        post_end = min(len(values), end + context_samples)
        pre = values[pre_start:start]
        post = values[end:post_end]
        pre_finite = pre[np.isfinite(pre)]
        post_finite = post[np.isfinite(post)]
        pre_peak = float(np.max(np.abs(pre_finite))) if pre_finite.size else np.nan
        post_peak = float(np.max(np.abs(post_finite))) if post_finite.size else np.nan
        context_peak = (
            float(np.nanmax([pre_peak, post_peak]))
            if np.isfinite(pre_peak) or np.isfinite(post_peak)
            else np.nan
        )
        entry = (
            float(polarity * (values[start] - values[start - 1]))
            if start > 0 and np.isfinite(values[start - 1])
            else np.nan
        )
        exit_ = (
            float(polarity * (values[end - 1] - values[end]))
            if end < len(values) and np.isfinite(values[end])
            else np.nan
        )
        transition_threshold = max(
            parameters.minimum_transition_relative_to_edge * abs_level,
            parameters.minimum_transition_tolerance_multiples * flat_tolerance,
            (
                parameters.minimum_transition_quantization_steps * quantization_step
                if np.isfinite(quantization_step)
                else 0.0
            ),
        )
        plateau_range = float(np.max(segment) - np.min(segment))
        first_differences = np.diff(segment)
        median_abs_diff = (
            float(np.median(np.abs(first_differences)))
            if first_differences.size
            else 0.0
        )
        plateau_slope = (
            float((segment[-1] - segment[0]) / max(sample_count - 1, 1))
        )
        unique_levels = int(np.unique(segment).size)
        morphology_pass = bool(
            plateau_range <= 2.0 * flat_tolerance
            and unique_levels <= parameters.maximum_plateau_unique_levels
            and abs(plateau_slope)
            <= parameters.maximum_plateau_slope_tolerance_multiples * flat_tolerance
        )
        duration_pass = bool(sample_count <= maximum_plateau_samples)
        magnitude_ratio = abs_level / robust_peak if robust_peak > 0 else np.nan
        local_magnitude_ratio = (
            abs_level / context_peak
            if np.isfinite(context_peak) and context_peak > 0
            else np.nan
        )
        recording_magnitude_pass = bool(
            np.isfinite(magnitude_ratio)
            and magnitude_ratio
            >= parameters.candidate_generation_minimum_edge_to_robust_peak_ratio
        )
        local_magnitude_pass = bool(
            np.isfinite(local_magnitude_ratio)
            and local_magnitude_ratio >= parameters.minimum_edge_to_local_peak_ratio
        )
        magnitude_pass = bool(recording_magnitude_pass and local_magnitude_pass)
        context_pass = bool(
            np.isfinite(pre_peak)
            and np.isfinite(post_peak)
            and pre_peak >= parameters.minimum_context_peak_ratio * abs_level
            and post_peak >= parameters.minimum_context_peak_ratio * abs_level
        )
        transition_pass = bool(
            np.isfinite(entry)
            and np.isfinite(exit_)
            and entry >= transition_threshold
            and exit_ >= transition_threshold
        )
        rows.append(
            {
                "logical_recording_id": logical_recording_id,
                "channel_index": int(channel_index),
                "polarity": int(polarity),
                "start_sample_task": start,
                "end_sample_task_exclusive": end,
                "start_sample_native": int(task_start_sample + start),
                "end_sample_native_exclusive": int(task_start_sample + end),
                "start_sec_native": float((task_start_sample + start) / fs),
                "end_sec_native": float((task_start_sample + end) / fs),
                "duration_sec": float(sample_count / fs),
                "sample_count": int(sample_count),
                "candidate_level": candidate_level,
                "candidate_abs_level": abs_level,
                "unique_level_count": unique_levels,
                "plateau_range": plateau_range,
                "median_abs_first_difference": median_abs_diff,
                "plateau_slope_per_sample": plateau_slope,
                "entry_signed_transition": entry,
                "exit_signed_transition": exit_,
                "transition_threshold": float(transition_threshold),
                "pre_context_peak_abs": pre_peak,
                "post_context_peak_abs": post_peak,
                "local_context_peak_abs": context_peak,
                "candidate_to_context_ratio": local_magnitude_ratio,
                "channel_robust_peak_abs": robust_peak,
                "candidate_to_robust_peak_ratio": magnitude_ratio,
                "flat_tolerance": flat_tolerance,
                "cluster_tolerance": cluster_tolerance,
                "quantization_step": quantization_step,
                "inferred_bits_per_sample": inferred_bits,
                "morphology_pass": morphology_pass,
                "duration_pass": duration_pass,
                "recording_magnitude_pass": recording_magnitude_pass,
                "local_magnitude_pass": local_magnitude_pass,
                "magnitude_pass": magnitude_pass,
                "context_pass": context_pass,
                "transition_pass": transition_pass,
            }
        )
    return rows, diagnostics


def _assign_level_clusters(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        rows = rows.copy()
        rows["cluster_id"] = pd.Series(dtype="string")
        return rows
    output = rows.copy()
    output["cluster_id"] = ""
    for (channel, polarity), index in output.groupby(
        ["channel_index", "polarity"], sort=True
    ).groups.items():
        local = output.loc[index].sort_values("candidate_level")
        cluster_number = 0
        previous_level: float | None = None
        previous_tolerance: float | None = None
        for row_index, row in local.iterrows():
            level = float(row["candidate_level"])
            tolerance = float(row["cluster_tolerance"])
            if (
                previous_level is None
                or abs(level - previous_level) > max(tolerance, previous_tolerance or 0.0)
            ):
                cluster_number += 1
            cluster_id = f"ch{int(channel):02d}_{'pos' if int(polarity) > 0 else 'neg'}_{cluster_number:04d}"
            output.at[row_index, "cluster_id"] = cluster_id
            previous_level = level
            previous_tolerance = tolerance
    return output


def _edge_evidence_for_cluster(
    values: np.ndarray,
    cluster: pd.DataFrame,
    fs: int,
    parameters: QDISTParameters,
) -> dict[str, Any]:
    context_samples = max(1, int(round(parameters.edge_evidence_context_ms * fs / 1000.0)))
    maximum_gap = max(
        0,
        int(round(parameters.edge_evidence_max_component_gap_ms * fs / 1000.0)),
    )
    ordered = cluster.sort_values(
        ["start_sample_task", "end_sample_task_exclusive"]
    )
    components: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    current_end = -1
    for row in ordered.itertuples(index=False):
        interval = (
            int(row.start_sample_task),
            int(row.end_sample_task_exclusive),
        )
        if current and interval[0] - current_end > maximum_gap:
            components.append(current)
            current = []
        current.append(interval)
        current_end = max(current_end, interval[1]) if current_end >= 0 else interval[1]
    if current:
        components.append(current)
    windows: list[tuple[int, int]] = []
    for component in components:
        if len(component) >= 2:
            left = min(item[0] for item in component)
            right = max(item[1] for item in component)
        else:
            left = max(0, component[0][0] - context_samples)
            right = min(len(values), component[0][1] + context_samples)
        windows.append((left, right))
    merged_windows: list[tuple[int, int]] = []
    for left, right in sorted(windows):
        if not merged_windows or left > merged_windows[-1][1]:
            merged_windows.append((left, right))
        else:
            merged_windows[-1] = (merged_windows[-1][0], max(merged_windows[-1][1], right))
    local_chunks = [
        values[left:right][np.isfinite(values[left:right])]
        for left, right in merged_windows
        if right > left
    ]
    finite = (
        np.concatenate([chunk for chunk in local_chunks if chunk.size])
        if any(chunk.size for chunk in local_chunks)
        else np.empty(0, dtype=float)
    )
    polarity = int(cluster["polarity"].iloc[0])
    weights = cluster["sample_count"].to_numpy(float)
    levels = cluster["candidate_level"].to_numpy(float)
    edge_level = float(np.average(levels, weights=weights))
    tolerance = float(np.max(cluster["cluster_tolerance"].to_numpy(float)))
    zone = max(
        parameters.absolute_flat_tolerance,
        parameters.edge_zone_tolerance_multiples * tolerance,
    )
    signed = polarity * finite
    signed_edge = polarity * edge_level
    edge_mask = np.abs(signed - signed_edge) <= zone
    inner_low = signed_edge - parameters.interior_shell_outer_multiples * zone
    inner_high = signed_edge - parameters.interior_shell_inner_multiples * zone
    interior_mask = (signed >= inner_low) & (signed < inner_high)
    beyond_mask = signed > signed_edge + zone
    edge_count = int(edge_mask.sum())
    interior_count = int(interior_mask.sum())
    beyond_count = int(beyond_mask.sum())
    ratio = float(edge_count / max(interior_count, 1))
    edge_excess = int(edge_count - interior_count)
    allowed_beyond = max(
        int(parameters.maximum_beyond_edge_samples),
        int(ceil(parameters.maximum_beyond_edge_fraction * len(finite))),
    )
    return {
        "edge_level": edge_level,
        "edge_zone_width": zone,
        "cluster_candidate_count": int(len(cluster)),
        "cluster_plateau_sample_count": int(cluster["sample_count"].sum()),
        "edge_zone_sample_count": edge_count,
        "interior_shell_sample_count": interior_count,
        "edge_to_interior_ratio": ratio,
        "edge_excess_samples": edge_excess,
        "beyond_edge_sample_count": beyond_count,
        "beyond_edge_fraction": float(beyond_count / len(finite)) if len(finite) else np.nan,
        "allowed_beyond_edge_samples": int(allowed_beyond),
        "edge_zone_fraction": float(edge_count / len(finite)) if len(finite) else np.nan,
    }


def _square_like_channels(
    task_samples: np.ndarray,
    edge_ledger: pd.DataFrame,
    parameters: QDISTParameters,
) -> dict[int, bool]:
    result: dict[int, bool] = {}
    for channel in range(task_samples.shape[1]):
        values = task_samples[:, channel]
        finite = values[np.isfinite(values)]
        local_edges = edge_ledger.loc[edge_ledger["channel_index"].eq(channel)]
        if not len(finite) or local_edges.empty:
            result[channel] = False
            continue
        covered = np.zeros(len(finite), dtype=bool)
        for row in local_edges.itertuples(index=False):
            covered |= np.abs(finite - float(row.edge_level)) <= float(row.edge_zone_width)
        # Rounding only supports an ambiguity diagnostic; it never creates candidates.
        scale = max(float(np.quantile(np.abs(finite), 0.999)), 1e-12)
        rounded = np.round(finite / max(scale * 1e-5, 1e-12)).astype(np.int64)
        unique_count = int(np.unique(rounded).size)
        result[channel] = bool(
            covered.mean() >= parameters.square_like_edge_fraction
            and unique_count <= parameters.square_like_maximum_unique_levels
        )
    return result


def _finalise_candidate_acceptance(
    task_samples: np.ndarray,
    candidates: pd.DataFrame,
    *,
    fs: int,
    parameters: QDISTParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return _empty_candidates(), pd.DataFrame(
            columns=[
                "logical_recording_id",
                "channel_index",
                "polarity",
                "cluster_id",
                "edge_level",
                "edge_zone_width",
                "cluster_candidate_count",
                "cluster_plateau_sample_count",
                "edge_zone_sample_count",
                "interior_shell_sample_count",
                "edge_to_interior_ratio",
                "edge_excess_samples",
                "beyond_edge_sample_count",
                "beyond_edge_fraction",
                "allowed_beyond_edge_samples",
                "edge_zone_fraction",
                "edge_support_pass",
                "edge_ratio_pass",
                "edge_excess_pass",
                "terminal_edge_pass",
                "measurement_version",
                "parameter_hash",
            ]
        )
    candidates = _assign_level_clusters(candidates)
    edge_rows: list[dict[str, Any]] = []
    parameter_hash = parameters.parameter_hash()
    for (channel, cluster_id), index in candidates.groupby(
        ["channel_index", "cluster_id"], sort=True
    ).groups.items():
        cluster = candidates.loc[index]
        evidence = _edge_evidence_for_cluster(
            task_samples[:, int(channel)], cluster, fs, parameters
        )
        edge_support_pass = bool(
            evidence["edge_zone_sample_count"] >= parameters.minimum_edge_zone_samples
        )
        ratio_threshold = parameters.minimum_edge_to_interior_ratio
        bits = cluster["inferred_bits_per_sample"].iloc[0]
        if pd.notna(bits) and int(bits) <= parameters.coarse_quantization_bits:
            ratio_threshold = parameters.coarse_minimum_edge_to_interior_ratio
        edge_rows.append(
            {
                "logical_recording_id": cluster["logical_recording_id"].iloc[0],
                "channel_index": int(channel),
                "polarity": int(cluster["polarity"].iloc[0]),
                "cluster_id": cluster_id,
                **evidence,
                "edge_support_pass": edge_support_pass,
                "edge_ratio_pass": bool(
                    evidence["edge_to_interior_ratio"] >= ratio_threshold
                ),
                "edge_excess_pass": bool(
                    evidence["edge_excess_samples"] >= parameters.minimum_edge_excess_samples
                ),
                "terminal_edge_pass": bool(
                    evidence["beyond_edge_sample_count"]
                    <= evidence["allowed_beyond_edge_samples"]
                ),
                "measurement_version": MEASUREMENT_VERSION,
                "parameter_hash": parameter_hash,
            }
        )
    edge_ledger = pd.DataFrame(edge_rows)
    candidates = candidates.merge(
        edge_ledger[
            [
                "channel_index",
                "cluster_id",
                "cluster_candidate_count",
                "cluster_plateau_sample_count",
                "edge_zone_sample_count",
                "interior_shell_sample_count",
                "edge_to_interior_ratio",
                "edge_excess_samples",
                "beyond_edge_sample_count",
                "beyond_edge_fraction",
                "allowed_beyond_edge_samples",
                "edge_zone_fraction",
                "edge_support_pass",
                "edge_ratio_pass",
                "edge_excess_pass",
                "terminal_edge_pass",
            ]
        ],
        on=["channel_index", "cluster_id"],
        how="left",
        validate="many_to_one",
    )
    square_like = _square_like_channels(task_samples, edge_ledger, parameters)
    cluster_support = []
    quantization_guard = []
    square_guard = []
    for row in candidates.itertuples(index=False):
        multiple_support = (
            int(row.cluster_candidate_count) >= 2
            and int(row.cluster_plateau_sample_count)
            >= parameters.minimum_level_cluster_samples
        )
        singleton_support = (
            int(row.cluster_candidate_count) == 1
            and int(row.sample_count) >= parameters.minimum_singleton_plateau_samples
        )
        cluster_support.append(bool(multiple_support or singleton_support))
        bits = row.inferred_bits_per_sample
        if pd.notna(bits) and int(bits) <= parameters.coarse_quantization_bits:
            quantization_guard.append(
                bool(
                    int(row.cluster_candidate_count)
                    >= parameters.coarse_minimum_cluster_candidates
                    or int(row.sample_count)
                    >= parameters.coarse_minimum_singleton_samples
                )
            )
        else:
            quantization_guard.append(True)
        square_guard.append(not square_like.get(int(row.channel_index), False))
    candidates["cluster_support_pass"] = cluster_support
    candidates["quantization_guard_pass"] = quantization_guard
    candidates["square_like_guard_pass"] = square_guard

    strong_recording_magnitude_pass = (
        pd.to_numeric(
            candidates["candidate_to_robust_peak_ratio"],
            errors="coerce",
        )
        >= parameters.minimum_edge_to_robust_peak_ratio
    )
    low_level_repeated_edge_pass = (
        (
            pd.to_numeric(
                candidates["candidate_to_robust_peak_ratio"],
                errors="coerce",
            )
            >= parameters.candidate_generation_minimum_edge_to_robust_peak_ratio
        )
        & (
            pd.to_numeric(
                candidates["cluster_candidate_count"],
                errors="coerce",
            )
            >= parameters.low_level_minimum_cluster_candidates
        )
        & (
            pd.to_numeric(
                candidates["cluster_plateau_sample_count"],
                errors="coerce",
            )
            >= parameters.low_level_minimum_cluster_plateau_samples
        )
        & (
            pd.to_numeric(
                candidates["edge_zone_sample_count"],
                errors="coerce",
            )
            >= parameters.low_level_minimum_edge_zone_samples
        )
    )

    candidates["strong_recording_magnitude_pass"] = (
        strong_recording_magnitude_pass.fillna(False).astype(bool)
    )
    candidates["low_level_repeated_edge_pass"] = (
        low_level_repeated_edge_pass.fillna(False).astype(bool)
    )
    candidates["recording_magnitude_pass"] = (
        candidates["strong_recording_magnitude_pass"]
        | candidates["low_level_repeated_edge_pass"]
    )
    candidates["magnitude_pass"] = (
        candidates["recording_magnitude_pass"].astype(bool)
        & candidates["local_magnitude_pass"].fillna(False).astype(bool)
    )
    candidates["magnitude_path"] = np.select(
        [
            candidates["strong_recording_magnitude_pass"],
            candidates["low_level_repeated_edge_pass"],
        ],
        [
            "strong_recording_edge",
            "repeated_low_level_saturation",
        ],
        default="rejected",
    )

    required = [
        "morphology_pass",
        "duration_pass",
        "recording_magnitude_pass",
        "local_magnitude_pass",
        "magnitude_pass",
        "context_pass",
        "transition_pass",
        "cluster_support_pass",
        "edge_support_pass",
        "edge_ratio_pass",
        "edge_excess_pass",
        "terminal_edge_pass",
        "quantization_guard_pass",
        "square_like_guard_pass",
    ]
    candidates["accepted"] = candidates[required].fillna(False).astype(bool).all(axis=1)
    reason_order = [
        ("morphology_pass", "plateau_morphology"),
        ("duration_pass", "excessive_flat_run_duration"),
        ("magnitude_pass", "insufficient_relative_edge_magnitude"),
        ("context_pass", "insufficient_bilateral_active_context"),
        ("transition_pass", "weak_or_nondirectional_plateau_edges"),
        ("cluster_support_pass", "insufficient_repeated_edge_support"),
        ("edge_support_pass", "insufficient_edge_zone_support"),
        ("edge_ratio_pass", "edge_not_concentrated_vs_interior"),
        ("edge_excess_pass", "no_edge_occupancy_excess"),
        ("terminal_edge_pass", "meaningful_support_beyond_proposed_edge"),
        ("quantization_guard_pass", "coarse_quantization_ambiguity"),
        ("square_like_guard_pass", "square_like_two_level_ambiguity"),
    ]
    rejection_reasons = []
    for row in candidates.itertuples(index=False):
        failed = [label for field, label in reason_order if not bool(getattr(row, field))]
        rejection_reasons.append("accepted" if not failed else "|".join(failed))
    candidates["rejection_reason"] = rejection_reasons
    candidates = candidates.sort_values(
        ["channel_index", "start_sample_task", "end_sample_task_exclusive"]
    ).reset_index(drop=True)
    candidates["candidate_id"] = [
        f"{str(candidates.loc[i, 'logical_recording_id'])}__cand_{i:06d}"
        for i in range(len(candidates))
    ]
    candidates["measurement_version"] = MEASUREMENT_VERSION
    candidates["parameter_hash"] = parameter_hash
    for column in CANDIDATE_COLUMNS:
        if column not in candidates:
            candidates[column] = pd.NA
    return candidates[CANDIDATE_COLUMNS], edge_ledger


def _interval_union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    clean = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    total = 0
    start, end = clean[0]
    for next_start, next_end in clean[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return int(total + end - start)


def _intersected_frame_indices(
    intervals: list[tuple[int, int]],
    frame_length_samples: int,
    complete_frame_count: int,
) -> set[int]:
    frames: set[int] = set()
    for start, end in intervals:
        if end <= start or complete_frame_count <= 0:
            continue
        first = max(0, int(start) // frame_length_samples)
        last = min(
            complete_frame_count - 1,
            (int(end) - 1) // frame_length_samples,
        )
        if last >= first:
            frames.update(range(first, last + 1))
    return frames


def _build_episode_ledger(
    accepted: pd.DataFrame,
    *,
    logical_recording_id: str,
    fs: int,
    task_start_sample: int,
    frame_length_samples: int,
    complete_frame_count: int,
    parameters: QDISTParameters,
) -> pd.DataFrame:
    if accepted.empty:
        return _empty_episodes()
    rows = accepted.sort_values(
        ["start_sample_task", "end_sample_task_exclusive", "channel_index"]
    ).to_dict(orient="records")
    merge_gap = max(0, int(round(parameters.episode_merge_gap_ms * fs / 1000.0)))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [rows[0]]
    current_end = int(rows[0]["end_sample_task_exclusive"])
    for row in rows[1:]:
        start = int(row["start_sample_task"])
        if start - current_end <= merge_gap:
            current.append(row)
            current_end = max(current_end, int(row["end_sample_task_exclusive"]))
        else:
            groups.append(current)
            current = [row]
            current_end = int(row["end_sample_task_exclusive"])
    groups.append(current)

    parameter_hash = parameters.parameter_hash()
    episodes: list[dict[str, Any]] = []
    for episode_index, group in enumerate(groups):
        start = min(int(item["start_sample_task"]) for item in group)
        end = max(int(item["end_sample_task_exclusive"]) for item in group)
        by_channel: dict[int, list[tuple[int, int]]] = {}
        for item in group:
            by_channel.setdefault(int(item["channel_index"]), []).append(
                (int(item["start_sample_task"]), int(item["end_sample_task_exclusive"]))
            )
        channel_sample_count = sum(
            _interval_union_length(intervals) for intervals in by_channel.values()
        )
        any_channel_count = _interval_union_length(
            [interval for intervals in by_channel.values() for interval in intervals]
        )
        frames = _intersected_frame_indices(
            [(start, end)], frame_length_samples, complete_frame_count
        )
        sorted_group = sorted(group, key=lambda item: int(item["start_sample_task"]))
        internal_gaps = [
            max(
                0,
                int(right["start_sample_task"])
                - int(left["end_sample_task_exclusive"]),
            )
            for left, right in zip(sorted_group[:-1], sorted_group[1:])
        ]
        polarities = sorted({int(item["polarity"]) for item in group})
        episodes.append(
            {
                "logical_recording_id": logical_recording_id,
                "episode_id": f"{logical_recording_id}__episode_{episode_index:05d}",
                "start_sample_task": start,
                "end_sample_task_exclusive": end,
                "start_sample_native": int(task_start_sample + start),
                "end_sample_native_exclusive": int(task_start_sample + end),
                "start_sec_native": float((task_start_sample + start) / fs),
                "end_sec_native": float((task_start_sample + end) / fs),
                "duration_sec": float((end - start) / fs),
                "plateau_count": int(len(group)),
                "constituent_candidate_ids": "|".join(
                    str(item["candidate_id"]) for item in group
                ),
                "channel_indices": "|".join(
                    str(value) for value in sorted(by_channel)
                ),
                "polarity_composition": (
                    "both" if len(polarities) > 1 else ("positive" if polarities[0] > 0 else "negative")
                ),
                "channel_sample_count": int(channel_sample_count),
                "any_channel_time_sample_count": int(any_channel_count),
                "intersected_frame_count": int(len(frames)),
                "maximum_internal_merge_gap_samples": int(max(internal_gaps, default=0)),
                "merge_gap_ms": float(parameters.episode_merge_gap_ms),
                "measurement_version": MEASUREMENT_VERSION,
                "parameter_hash": parameter_hash,
            }
        )
    return pd.DataFrame(episodes, columns=EPISODE_COLUMNS)


def reconstruct_qdist_features(
    accepted_plateau_ledger: pd.DataFrame,
    episode_ledger: pd.DataFrame,
    *,
    finite_channel_sample_count: int,
    finite_time_sample_count: int,
    finite_exposure_sec: float,
    frame_length_samples: int,
    complete_frame_count: int,
) -> dict[str, float]:
    """Independently reconstruct the three analysis features from saved ledgers."""

    if finite_channel_sample_count <= 0 or finite_exposure_sec <= 0:
        return {feature: np.nan for feature in ANALYSIS_FEATURES}
    accepted = accepted_plateau_ledger.copy()
    channel_intervals: dict[int, list[tuple[int, int]]] = {}
    all_intervals: list[tuple[int, int]] = []
    if len(accepted):
        for row in accepted.itertuples(index=False):
            interval = (
                int(row.start_sample_task),
                int(row.end_sample_task_exclusive),
            )
            channel_intervals.setdefault(int(row.channel_index), []).append(interval)
            all_intervals.append(interval)
    clipped_channel_samples = sum(
        _interval_union_length(intervals) for intervals in channel_intervals.values()
    )
    frame_indices = _intersected_frame_indices(
        all_intervals,
        frame_length_samples,
        complete_frame_count,
    )
    sample_fraction = clipped_channel_samples / finite_channel_sample_count
    frame_fraction = (
        len(frame_indices) / complete_frame_count if complete_frame_count > 0 else np.nan
    )
    event_rate = len(episode_ledger) * 60.0 / finite_exposure_sec
    return {
        "qdist_hard_clipped_frame_fraction": float(frame_fraction),
        "qdist_hard_clip_event_rate_per_min": float(event_rate),
        "qdist_hard_clipped_sample_fraction": float(sample_fraction),
    }


def _base_recording(
    *,
    logical_recording_id: str,
    fs: int,
    channels: int,
    task_span: TimeInterval,
    task_start_sample: int,
    task_end_sample: int,
    finite_fraction: float,
    finite_channel_sample_count: int,
    finite_time_sample_count: int,
    complete_frame_count: int,
    frame_length_samples: int,
    provenance: NativeSignalProvenance,
    parameters: QDISTParameters,
) -> dict[str, Any]:
    parameter_hash = parameters.parameter_hash()
    duration_sec = max(0.0, (task_end_sample - task_start_sample) / fs)
    recording: dict[str, Any] = {
        "logical_recording_id": logical_recording_id,
        "qdist_measurement_version": MEASUREMENT_VERSION,
        "qdist_parameter_hash": parameter_hash,
        "qdist_signal_view": "native_rate_multichannel_first_decoded_audio_stream_no_transform",
        "qdist_task_span_start_sec": task_span.start_sec,
        "qdist_task_span_end_sec": task_span.end_sec,
        "qdist_task_span_start_sample_native": task_start_sample,
        "qdist_task_span_end_sample_native_exclusive": task_end_sample,
        "qdist_task_span_duration_sec": duration_sec,
        "qdist_native_sample_rate_hz": int(fs),
        "qdist_native_channel_count": int(channels),
        "qdist_finite_fraction": float(finite_fraction),
        "qdist_finite_channel_sample_count": int(finite_channel_sample_count),
        "qdist_finite_time_sample_count": int(finite_time_sample_count),
        "qdist_frame_length_ms": float(parameters.frame_length_ms),
        "qdist_frame_length_samples": int(frame_length_samples),
        "qdist_complete_frame_count": int(complete_frame_count),
        "qdist_episode_merge_gap_ms": float(parameters.episode_merge_gap_ms),
        "qdist_support_tier": _support_tier(duration_sec, parameters),
        "qdist_status": "indeterminate",
        "qdist_available": False,
        "qdist_accepted_plateau_count": 0,
        "qdist_hard_clip_event_count": 0,
        "qdist_affected_channel_count": 0,
        "qdist_any_channel_affected_time_fraction": np.nan,
        "qdist_near_fullscale_channel_sample_fraction": np.nan,
        "qdist_hard_clip_event_rate_ci95_low_per_min": np.nan,
        "qdist_hard_clip_event_rate_ci95_high_per_min": np.nan,
        "qdist_native_view_verified": bool(provenance.native_view_verified),
        "qdist_known_preprocessing_applied": bool(provenance.known_preprocessing_applied),
        "qdist_codec_name": provenance.codec_name,
        "qdist_sample_format": provenance.sample_format,
        "qdist_bits_per_raw_sample": provenance.bits_per_raw_sample,
        "qdist_container_format": provenance.container_format,
        "qdist_channel_layout": provenance.channel_layout,
        "qdist_source_path": provenance.source_path,
        "qdist_source_sha256": provenance.source_sha256,
        "qdist_decoded_sha256": provenance.decoded_sha256,
        "qdist_decoder": provenance.decoder,
        "qdist_decoder_version": provenance.decoder_version,
        "qdist_decode_arguments": provenance.decode_arguments,
    }
    for feature in ANALYSIS_FEATURES:
        recording[feature] = np.nan
        recording[f"{feature}_status"] = "indeterminate"
    return recording


def _unavailable_extraction(
    recording: dict[str, Any],
    status: str,
) -> QDISTExtraction:
    recording = dict(recording)
    recording["qdist_status"] = status
    recording["qdist_available"] = False
    for feature in ANALYSIS_FEATURES:
        recording[feature] = np.nan
        recording[f"{feature}_status"] = status
    return QDISTExtraction(
        recording=recording,
        candidate_ledger=_empty_candidates(),
        accepted_plateau_ledger=_empty_candidates(),
        episode_ledger=_empty_episodes(),
        edge_ledger=pd.DataFrame(),
    )


def extract_qdist(
    waveform: np.ndarray,
    fs: int,
    *,
    task_span: TimeInterval | None = None,
    logical_recording_id: str = "recording",
    provenance: NativeSignalProvenance | None = None,
    parameters: QDISTParameters = DEFAULT_PARAMETERS,
) -> QDISTExtraction:
    """Extract QDIST hard-clipping features and all reconstructable ledgers.

    Parameters
    ----------
    waveform:
        Native-rate decoded samples with shape ``samples`` or
        ``samples x channels``.  No channel reduction is performed.
    fs:
        Native sampling rate in Hz.
    task_span:
        Continuous natural task span.  Internal pauses remain in the waveform.
        The full waveform is used when omitted.
    provenance:
        Native-view verification and source/decode metadata.

    Notes
    -----
    The function is deterministic and label-free.  It never uses clinical or
    human-QC outcomes.  Candidate generation is permissive; acceptance is a
    conjunctive morphology, edge-concentration, terminality, context, and
    quantization decision.
    """

    if not isinstance(fs, (int, np.integer)) or int(fs) <= 0:
        raise ValueError("QDIST requires a positive integer native sample rate.")
    fs = int(fs)
    samples = _as_2d(waveform)
    provenance = provenance or NativeSignalProvenance()
    duration_sec = len(samples) / fs if len(samples) else 0.0
    span = _normalise_task_span(duration_sec, task_span)
    start, end = _task_span_indices(span, fs, len(samples))
    task = samples[start:end]
    finite = np.isfinite(task)
    total_channel_samples = int(task.size)
    finite_channel_samples = int(finite.sum())
    finite_fraction = (
        finite_channel_samples / total_channel_samples if total_channel_samples else 0.0
    )
    finite_time_samples = int(np.isfinite(task).all(axis=1).sum()) if len(task) else 0
    frame_length_samples = max(1, int(round(parameters.frame_length_ms * fs / 1000.0)))
    complete_frames = int(len(task) // frame_length_samples)
    recording = _base_recording(
        logical_recording_id=str(logical_recording_id),
        fs=fs,
        channels=task.shape[1] if task.ndim == 2 else 1,
        task_span=span,
        task_start_sample=start,
        task_end_sample=end,
        finite_fraction=finite_fraction,
        finite_channel_sample_count=finite_channel_samples,
        finite_time_sample_count=finite_time_samples,
        complete_frame_count=complete_frames,
        frame_length_samples=frame_length_samples,
        provenance=provenance,
        parameters=parameters,
    )
    recording["qdist_decoded_sha256"] = (
        provenance.decoded_sha256 or decoded_waveform_sha256(samples)
    )

    if not provenance.native_view_verified:
        return _unavailable_extraction(recording, "unavailable_native_view_not_verified")
    if provenance.known_preprocessing_applied:
        return _unavailable_extraction(recording, "unavailable_preprocessed_source")
    if total_channel_samples == 0 or finite_channel_samples == 0:
        return _unavailable_extraction(recording, "unavailable_no_finite_exposure")
    if finite_fraction < parameters.minimum_finite_fraction:
        return _unavailable_extraction(recording, "indeterminate_nonfinite_support")
    task_duration_sec = len(task) / fs
    if (
        task_duration_sec < parameters.minimum_task_span_sec
        or complete_frames < parameters.minimum_complete_frame_count
    ):
        return _unavailable_extraction(recording, "indeterminate_insufficient_support")

    inferred_bits = infer_integer_bits(provenance)
    quantization_step = quantization_step_from_bits(inferred_bits)
    candidate_rows: list[dict[str, Any]] = []
    edge_diagnostics: list[dict[str, Any]] = []
    for channel_index in range(task.shape[1]):
        rows, diagnostics = _candidate_rows_for_channel(
            task[:, channel_index],
            channel_index=channel_index,
            fs=fs,
            task_start_sample=start,
            logical_recording_id=str(logical_recording_id),
            quantization_step=quantization_step,
            inferred_bits=inferred_bits,
            parameters=parameters,
        )
        candidate_rows.extend(rows)
        edge_diagnostics.append(diagnostics)
    raw_candidates = pd.DataFrame(candidate_rows)
    if raw_candidates.empty:
        candidates = _empty_candidates()
        edge_ledger = pd.DataFrame(edge_diagnostics)
    else:
        candidates, edge_ledger = _finalise_candidate_acceptance(
            task,
            raw_candidates,
            fs=fs,
            parameters=parameters,
        )
        if edge_diagnostics:
            channel_diagnostics = pd.DataFrame(edge_diagnostics)
            edge_ledger = edge_ledger.merge(
                channel_diagnostics,
                on=["logical_recording_id", "channel_index"],
                how="left",
                validate="many_to_one",
                suffixes=("", "_channel"),
            )
    accepted = candidates.loc[candidates["accepted"].astype(bool)].copy()
    episode_ledger = _build_episode_ledger(
        accepted,
        logical_recording_id=str(logical_recording_id),
        fs=fs,
        task_start_sample=start,
        frame_length_samples=frame_length_samples,
        complete_frame_count=complete_frames,
        parameters=parameters,
    )
    finite_exposure_sec = finite_time_samples / fs
    reconstructed = reconstruct_qdist_features(
        accepted,
        episode_ledger,
        finite_channel_sample_count=finite_channel_samples,
        finite_time_sample_count=finite_time_samples,
        finite_exposure_sec=finite_exposure_sec,
        frame_length_samples=frame_length_samples,
        complete_frame_count=complete_frames,
    )
    event_count = int(len(episode_ledger))
    event_ci_low, event_ci_high = poisson_rate_interval(
        event_count,
        finite_exposure_sec,
        confidence=parameters.poisson_confidence,
    )
    channel_intervals = {
        int(channel): [
            (int(row.start_sample_task), int(row.end_sample_task_exclusive))
            for row in local.itertuples(index=False)
        ]
        for channel, local in accepted.groupby("channel_index")
    }
    any_channel_intervals = [
        interval for intervals in channel_intervals.values() for interval in intervals
    ]
    near_fullscale = (
        float(np.mean(np.abs(task[finite]) >= parameters.near_full_scale_threshold))
        if finite.any()
        else np.nan
    )
    status = "available_events" if event_count else "available_no_events"
    recording.update(reconstructed)
    recording.update(
        {
            "qdist_status": status,
            "qdist_available": True,
            "qdist_accepted_plateau_count": int(len(accepted)),
            "qdist_hard_clip_event_count": event_count,
            "qdist_affected_channel_count": int(len(channel_intervals)),
            "qdist_any_channel_affected_time_fraction": (
                _interval_union_length(any_channel_intervals) / finite_time_samples
                if finite_time_samples > 0
                else np.nan
            ),
            "qdist_near_fullscale_channel_sample_fraction": near_fullscale,
            "qdist_hard_clip_event_rate_ci95_low_per_min": event_ci_low,
            "qdist_hard_clip_event_rate_ci95_high_per_min": event_ci_high,
            "qdist_candidate_plateau_count": int(len(candidates)),
            "qdist_rejected_candidate_count": int((~candidates["accepted"].astype(bool)).sum())
            if len(candidates)
            else 0,
            "qdist_flat_run_count_all_amplitudes": int(
                sum(int(item.get("flat_run_count_all_amplitudes", 0)) for item in edge_diagnostics)
            ),
            "qdist_flat_run_count_at_recording_edge": int(
                sum(int(item.get("flat_run_count_at_recording_edge", 0)) for item in edge_diagnostics)
            ),
            "qdist_flat_run_prefilter_reduction_fraction": (
                1.0
                - sum(int(item.get("flat_run_count_at_recording_edge", 0)) for item in edge_diagnostics)
                / max(sum(int(item.get("flat_run_count_all_amplitudes", 0)) for item in edge_diagnostics), 1)
            ),
            "qdist_finite_exposure_sec": float(finite_exposure_sec),
            "qdist_inferred_integer_bits": inferred_bits,
            "qdist_quantization_step": quantization_step,
        }
    )
    for feature in ANALYSIS_FEATURES:
        recording[f"{feature}_status"] = status
    return QDISTExtraction(
        recording=recording,
        candidate_ledger=candidates,
        accepted_plateau_ledger=accepted.reset_index(drop=True),
        episode_ledger=episode_ledger,
        edge_ledger=edge_ledger.reset_index(drop=True),
    )


def apply_hard_clip(
    waveform: np.ndarray,
    positive_limit: float,
    negative_limit: float | None = None,
) -> np.ndarray:
    """Apply deterministic hard clipping for construct-recovery controls."""

    positive_limit = float(positive_limit)
    negative_limit = -positive_limit if negative_limit is None else float(negative_limit)
    if not np.isfinite(positive_limit) or not np.isfinite(negative_limit):
        raise ValueError("Clipping limits must be finite.")
    if negative_limit >= positive_limit:
        raise ValueError("negative_limit must be less than positive_limit.")
    return np.clip(np.asarray(waveform, dtype=np.float64), negative_limit, positive_limit)


def apply_soft_clip(waveform: np.ndarray, drive: float = 2.0) -> np.ndarray:
    """Smooth tanh saturation used only as a QDIST scope/discriminant control."""

    drive = float(drive)
    if not np.isfinite(drive) or drive <= 0:
        raise ValueError("drive must be positive and finite.")
    samples = np.asarray(waveform, dtype=np.float64)
    normalizer = np.tanh(drive)
    return np.tanh(drive * samples) / normalizer


def quantize_pcm(waveform: np.ndarray, bits: int) -> np.ndarray:
    """Quantize to a normalized signed PCM lattice without clipping the input silently."""

    bits = int(bits)
    if bits < 2 or bits > 32:
        raise ValueError("bits must be between 2 and 32.")
    samples = np.asarray(waveform, dtype=np.float64)
    maximum_code = 2 ** (bits - 1) - 1
    minimum_code = -(2 ** (bits - 1))
    codes = np.rint(samples * (2 ** (bits - 1)))
    codes = np.clip(codes, minimum_code, maximum_code)
    return codes / float(2 ** (bits - 1))


def with_relaxed_support(
    parameters: QDISTParameters = DEFAULT_PARAMETERS,
) -> QDISTParameters:
    """Small-array test helper; never use for cohort extraction."""

    return replace(
        parameters,
        minimum_task_span_sec=0.0,
        minimum_complete_frame_count=0,
    )
