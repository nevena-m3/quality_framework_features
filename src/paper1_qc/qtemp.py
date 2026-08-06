"""QTEMP v0.3 measurement-development implementation.

QTEMP measures observable temporal continuity violations in a *native decoded*
audio stream. It does not identify packet loss, browser failure, buffering,
transport failure, or any other unique upstream cause.

The implementation is intentionally event-ledger first:

* native channels are inspected independently;
* all generated candidates are preserved;
* candidate disposition is explicit (accepted / indeterminate / rejected);
* accepted channel events are collapsed into a recording-level event ledger;
* the five analysis features are exact summaries of that accepted event ledger;
* measured zero is distinct from unavailable.

This is a governed measurement-development release with optimized native-stream detectors. Thresholds remain
engineering parameters until the notebook's synthetic, real-speech injection,
signal-chain, empirical, and blinded-review evidence is complete.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from math import ceil, floor, sqrt
import json
from typing import Literal

import numpy as np
import pandas as pd
from scipy import linalg, signal, stats

MEASUREMENT_VERSION = "qtemp-v0.3.0-measurement-development"

ANALYSIS_FEATURES = (
    "qtemp_dropout_duration_fraction",
    "qtemp_dropout_event_rate_per_min",
    "qtemp_frozen_audio_duration_fraction",
    "qtemp_frozen_audio_event_rate_per_min",
    "qtemp_splice_discontinuity_rate_per_min",
)

PRIMARY_FEATURES = (
    "qtemp_dropout_duration_fraction",
    "qtemp_dropout_event_rate_per_min",
    "qtemp_frozen_audio_duration_fraction",
)

EVENT_TYPES = ("dropout", "frozen_audio", "splice")
DISPOSITIONS = ("accepted", "indeterminate", "rejected")


@dataclass(frozen=True)
class TimeInterval:
    """Half-open interval in original recording seconds."""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


@dataclass(frozen=True)
class QTEMPParameters:
    """Versioned engineering parameters for measurement development."""

    # Common native-stream exposure contract.
    edge_guard_ms: float = 100.0
    minimum_eligible_duration_sec: float = 1.0
    cross_channel_merge_ms: float = 2.0
    within_type_merge_ms: float = 3.0

    # Bracketed dropout-like runs.
    dropout_min_duration_ms: float = 10.0
    dropout_max_duration_ms: float = 1000.0
    dropout_constant_max_duration_ms: float = 250.0
    dropout_merge_gap_ms: float = 3.0
    dropout_context_ms: float = 40.0
    dropout_zero_abs_threshold: float = 1e-7
    dropout_constant_diff_abs_threshold: float = 2e-7
    dropout_constant_std_abs_threshold: float = 2e-6
    dropout_constant_peak_abs_threshold: float = 2e-4
    dropout_constant_std_context_fraction: float = 0.02
    dropout_constant_peak_context_fraction: float = 0.25
    dropout_context_min_abs_ac_rms: float = 2e-5
    dropout_context_min_interval_fraction: float = 0.03
    dropout_context_run_ac_ratio_accept: float = 10.0
    dropout_context_run_ac_ratio_indeterminate: float = 5.0
    dropout_boundary_review_guard_ms: float = 30.0

    # Consecutive duplicated/frozen multiframe sequences.
    duplicate_frame_ms: float = 4.0
    duplicate_hop_ms: float = 2.0
    duplicate_lags_ms: tuple[float, ...] = (
        10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
        60.0, 80.0, 100.0, 120.0, 160.0, 200.0,
    )
    duplicate_min_sequence_ms: float = 18.0
    duplicate_max_sequence_ms: float = 500.0
    duplicate_min_ac_rms_abs: float = 2e-5
    duplicate_min_interval_rms_fraction: float = 0.03
    duplicate_residual_accept: float = 0.003
    duplicate_residual_indeterminate: float = 0.015
    duplicate_cosine_accept: float = 0.9995
    duplicate_cosine_indeterminate: float = 0.995
    duplicate_spectral_cosine_accept: float = 0.995
    duplicate_boundary_novelty_accept: float = 0.03
    duplicate_boundary_novelty_indeterminate: float = 0.01
    duplicate_periodicity_lag_max_ms: float = 20.0
    duplicate_periodicity_similarity: float = 0.96
    duplicate_low_entropy_threshold: float = 0.18
    duplicate_merge_gap_ms: float = 5.0

    # Abrupt splice-like joins.
    splice_context_ms: float = 30.0
    splice_ar_order: int = 12
    splice_edge_guard_ms: float = 35.0
    splice_local_baseline_ms: float = 600.0
    splice_lpc_block_ms: float = 900.0
    splice_lpc_overlap_fraction: float = 0.50
    splice_derivative_z_accept: float = 9.0
    splice_derivative_z_indeterminate: float = 6.0
    splice_prediction_z_accept: float = 3.7
    splice_prediction_z_indeterminate: float = 3.0
    splice_min_context_ac_rms_abs: float = 2e-5
    splice_min_interval_rms_fraction: float = 0.03
    splice_refractory_ms: float = 8.0
    splice_clipping_edge_guard_ms: float = 3.0
    splice_speech_boundary_guard_ms: float = 25.0
    splice_qtemp_event_guard_ms: float = 3.0
    splice_impulse_return_ratio: float = 0.005
    splice_context_spectral_cosine_accept_max: float = 0.90
    splice_context_spectral_cosine_indeterminate_max: float = 0.97
    splice_level_step_review_db: float = 4.0
    splice_max_candidates_per_min: int = 80

    random_seed: int = 20260731

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMETERS = QTEMPParameters()


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    measurement_version: str
    display_name: str
    subdomain: str
    role: str
    unit: str
    formula: str
    estimand: str
    signal_view: str
    signal_region: str
    orientation: str
    mathematical_range: str
    minimum_support: str
    support_field: str
    status_field: str
    positive_control: str
    negative_control: str
    maturity: str
    known_confounds: str
    claim_boundary: str
    analysis_eligibility: str
    composite_use: str


FEATURE_DEFINITIONS = (
    FeatureDefinition(
        name="qtemp_dropout_duration_fraction",
        measurement_version=MEASUREMENT_VERSION,
        display_name="Bracketed dropout-like duration fraction",
        subdomain="missing/held decoded support",
        role="candidate primary",
        unit="fraction",
        formula="sum accepted recording-level dropout durations / eligible native-stream duration",
        estimand="Fraction of observable task-stream time occupied by accepted bracketed dropout-like runs.",
        signal_view="native decoded channels before resampling, mono conversion, filtering, interpolation, or normalization",
        signal_region="frozen task-analysis interval with symmetric native-stream edge guard",
        orientation="higher = greater accepted dropout-like time burden",
        mathematical_range="[0, 1]",
        minimum_support="successful native decode/provenance and >=1 s eligible exposure",
        support_field="qtemp_eligible_duration_sec",
        status_field="qtemp_dropout_duration_fraction_status",
        positive_control="injected exact-zero and constant-low-information runs with active bilateral context",
        negative_control="unmodified speech, natural pause/closure, weak speech, and edge silence",
        maturity="study-specific event detector",
        known_confounds="true silence, stop closure, muted microphone, digital floor, VAD response, decoder concealment",
        claim_boundary="observable dropout-like gaps; not packet-loss fraction or transport-mechanism identification",
        analysis_eligibility="candidate; retain only after synthetic, real-speech, and adjudication gates",
        composite_use="prohibited until independent dimensionality/weighting justification",
    ),
    FeatureDefinition(
        name="qtemp_dropout_event_rate_per_min",
        measurement_version=MEASUREMENT_VERSION,
        display_name="Bracketed dropout-like event rate",
        subdomain="missing/held decoded support",
        role="candidate primary event metric",
        unit="events/min",
        formula="accepted recording-level dropout event count * 60 / eligible native-stream seconds",
        estimand="Frequency of discrete accepted dropout-like events.",
        signal_view="native decoded channels before transformations",
        signal_region="frozen task-analysis interval with symmetric native-stream edge guard",
        orientation="higher = more frequent accepted dropout-like events",
        mathematical_range="[0, +infinity)",
        minimum_support="successful native decode/provenance and >=1 s eligible exposure",
        support_field="qtemp_eligible_duration_sec",
        status_field="qtemp_dropout_event_rate_per_min_status",
        positive_control="injected event-count and merge-rule recovery",
        negative_control="natural closures and long silence without qualifying bilateral evidence",
        maturity="study-specific event aggregation",
        known_confounds="event fragmentation/merging, articulation, exposure length, edge effects",
        claim_boundary="same-ledger event-frequency view; not independent evidence from duration burden",
        analysis_eligibility="candidate; sparse count/rate analysis may replace continuous analysis",
        composite_use="prohibited",
    ),
    FeatureDefinition(
        name="qtemp_frozen_audio_duration_fraction",
        measurement_version=MEASUREMENT_VERSION,
        display_name="Near-exact consecutive decoded-repetition duration fraction",
        subdomain="near-exact consecutive decoded repetition",
        role="provisional primary",
        unit="fraction",
        formula="union duration of accepted recording-level duplicated targets / eligible native-stream duration",
        estimand="Time burden of accepted near-exact repeated decoded waveform support.",
        signal_view="native decoded channels before transformations",
        signal_region="frozen task-analysis interval with symmetric native-stream edge guard",
        orientation="higher = greater accepted repeated-stream burden",
        mathematical_range="[0, 1]",
        minimum_support="successful native decode/provenance, non-silent support, and >=1 s eligible exposure",
        support_field="qtemp_eligible_duration_sec",
        status_field="qtemp_frozen_audio_duration_fraction_status",
        positive_control="exact and perturbed duplicate injection over lag/duration grids",
        negative_control="periodic voiced speech, sustained vowels, tones, and repeated linguistic material",
        maturity="high-risk study-specific event detector",
        known_confounds="periodic voicing, tones, low-entropy signals, codec/quantization behavior",
        claim_boundary="near-exact decoded repetition evidence; not all freezes or packet-loss concealment",
        analysis_eligibility="provisional; periodic-speech specificity is blocking",
        composite_use="prohibited",
    ),
    FeatureDefinition(
        name="qtemp_frozen_audio_event_rate_per_min",
        measurement_version=MEASUREMENT_VERSION,
        display_name="Near-exact consecutive decoded-repetition event rate",
        subdomain="near-exact consecutive decoded repetition",
        role="provisional secondary event metric",
        unit="events/min",
        formula="accepted recording-level duplicated event count * 60 / eligible native-stream seconds",
        estimand="Frequency of discrete accepted repeated-support events.",
        signal_view="native decoded channels before transformations",
        signal_region="frozen task-analysis interval with symmetric native-stream edge guard",
        orientation="higher = more accepted repeated-support events",
        mathematical_range="[0, +infinity)",
        minimum_support="successful native decode/provenance and >=1 s eligible exposure",
        support_field="qtemp_eligible_duration_sec",
        status_field="qtemp_frozen_audio_event_rate_per_min_status",
        positive_control="duplicate event-count and event-grouping recovery",
        negative_control="periodic speech and tones",
        maturity="study-specific aggregation of a high-risk detector",
        known_confounds="grouping/merge rules, periodic speech, short exposure, cross-channel overlap",
        claim_boundary="secondary same-ledger frequency view; grouping is algorithm-dependent",
        analysis_eligibility="provisional secondary",
        composite_use="prohibited",
    ),
    FeatureDefinition(
        name="qtemp_splice_discontinuity_rate_per_min",
        measurement_version=MEASUREMENT_VERSION,
        display_name="Abrupt splice-like discontinuity rate",
        subdomain="localized sample continuity",
        role="provisional secondary event metric",
        unit="events/min",
        formula="accepted recording-level splice-like event count * 60 / eligible native-stream seconds",
        estimand="Frequency of localized bilateral prediction failures after competing-event exclusions.",
        signal_view="native decoded channels before transformations",
        signal_region="frozen task-analysis interval excluding edge, speech-boundary, clipping, and accepted-QTEMP guards",
        orientation="higher = more accepted localized abrupt-join evidence",
        mathematical_range="[0, +infinity)",
        minimum_support="successful native decode/provenance, >=1 s exposure, and bilateral active context",
        support_field="qtemp_eligible_duration_sec",
        status_field="qtemp_splice_discontinuity_rate_per_min_status",
        positive_control="finite source replacement or insertion with observable bilateral context mismatch",
        negative_control="plosive/click, clipping edge, gain-step-only, and speech onset/offset",
        maturity="high-risk study-specific engineering detector",
        known_confounds="plosives, clicks, mouth sounds, clipping edges, gain steps, edits, speech boundaries",
        claim_boundary="nonspecific strong bilateral context-mismatch evidence; smooth or phase-compatible deletions may be unidentifiable, and the feature is not proof of deleted audio or packet loss",
        analysis_eligibility="provisional secondary; may be audit-only or dropped",
        composite_use="prohibited",
    ),
)


@dataclass
class QTEMPExtraction:
    recording: dict
    candidate_ledger: pd.DataFrame
    disposition_ledger: pd.DataFrame
    event_ledger: pd.DataFrame
    exposure_ledger: pd.DataFrame


def feature_registry_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in FEATURE_DEFINITIONS])


def _channels_float(waveform: np.ndarray) -> np.ndarray:
    """Return samples x channels without mono averaging or DC removal."""

    raw = np.asarray(waveform)
    if raw.ndim == 1:
        raw = raw[:, None]
    elif raw.ndim != 2:
        raise ValueError("waveform must be samples or samples-by-channels")
    values = raw.astype(np.float64, copy=False)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        values = values / scale
    return values


def merge_intervals(intervals: Iterable[TimeInterval]) -> list[TimeInterval]:
    valid = sorted(
        (
            TimeInterval(float(item.start_sec), float(item.end_sec))
            for item in intervals
            if np.isfinite(item.start_sec)
            and np.isfinite(item.end_sec)
            and float(item.end_sec) > float(item.start_sec)
        ),
        key=lambda item: (item.start_sec, item.end_sec),
    )
    merged: list[TimeInterval] = []
    for item in valid:
        if not merged or item.start_sec > merged[-1].end_sec:
            merged.append(item)
        else:
            merged[-1] = TimeInterval(
                merged[-1].start_sec,
                max(merged[-1].end_sec, item.end_sec),
            )
    return merged


def eligible_intervals(
    duration_sec: float,
    analysis_intervals: Iterable[TimeInterval] | None = None,
    *,
    parameters: QTEMPParameters = DEFAULT_PARAMETERS,
) -> list[TimeInterval]:
    source = (
        list(analysis_intervals)
        if analysis_intervals is not None
        else [TimeInterval(0.0, float(duration_sec))]
    )
    guard = parameters.edge_guard_ms / 1000.0
    output: list[TimeInterval] = []
    for item in merge_intervals(source):
        start = max(0.0, float(item.start_sec) + guard)
        end = min(float(duration_sec), float(item.end_sec) - guard)
        if end > start:
            output.append(TimeInterval(start, end))
    return output


def _interval_samples(interval: TimeInterval, fs: int, n_samples: int) -> tuple[int, int]:
    start = int(floor(interval.start_sec * fs))
    end = int(ceil(interval.end_sec * fs))
    return max(0, min(n_samples, start)), max(0, min(n_samples, end))


def _rms(values: np.ndarray, *, ac: bool = False) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        return np.nan
    if ac:
        x = x - float(np.mean(x))
    return float(sqrt(float(np.mean(x * x))))


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    return median, max(1e-15, 1.4826 * mad)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return []
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _interval_overlap(start: float, end: float, intervals: Sequence[TimeInterval]) -> float:
    return float(
        sum(max(0.0, min(end, item.end_sec) - max(start, item.start_sec)) for item in intervals)
    )


def _distance_to_boundaries(time_sec: float, intervals: Sequence[TimeInterval]) -> float:
    boundaries = [value for item in intervals for value in (item.start_sec, item.end_sec)]
    if not boundaries:
        return np.inf
    return float(min(abs(time_sec - value) for value in boundaries))


def _candidate_id(event_type: str, channel_index: int, index: int) -> str:
    return f"{event_type}-ch{channel_index:02d}-{index:07d}"


def _merge_raw_events(events: list[dict], max_gap_samples: int) -> list[dict]:
    if not events:
        return []
    ordered = sorted(events, key=lambda row: (row["start_sample"], row["end_sample"]))
    merged = [dict(ordered[0])]
    for event in ordered[1:]:
        previous = merged[-1]
        compatible = (
            event.get("event_type") == previous.get("event_type")
            and event.get("event_subtype") == previous.get("event_subtype")
            and event.get("channel_index") == previous.get("channel_index")
            and event["start_sample"] - previous["end_sample"] <= max_gap_samples
        )
        if compatible:
            previous["end_sample"] = max(previous["end_sample"], event["end_sample"])
            previous["candidate_count_merged"] = int(previous.get("candidate_count_merged", 1)) + int(
                event.get("candidate_count_merged", 1)
            )
        else:
            merged.append(dict(event))
    return merged


def _event_context_metrics(
    x: np.ndarray,
    start: int,
    end: int,
    context_n: int,
    interval_start: int,
    interval_end: int,
) -> dict:
    left = x[max(interval_start, start - context_n) : start]
    right = x[end : min(interval_end, end + context_n)]
    run = x[start:end]
    return {
        "left_context_sample_count": int(left.size),
        "right_context_sample_count": int(right.size),
        "left_context_rms": _rms(left),
        "right_context_rms": _rms(right),
        "left_context_ac_rms": _rms(left, ac=True),
        "right_context_ac_rms": _rms(right, ac=True),
        "run_rms": _rms(run),
        "run_ac_rms": _rms(run, ac=True),
        "run_std": float(np.std(run)) if run.size else np.nan,
        "run_peak_abs": float(np.max(np.abs(run))) if run.size else np.nan,
    }


def detect_dropout_candidates(
    waveform: np.ndarray,
    sample_rate_hz: int,
    intervals: Iterable[TimeInterval],
    *,
    speech_intervals: Iterable[TimeInterval] | None = None,
    parameters: QTEMPParameters = DEFAULT_PARAMETERS,
) -> list[dict]:
    """Generate and classify bracketed dropout-like candidates per native channel."""

    values = _channels_float(waveform)
    fs = int(sample_rate_hz)
    min_n = max(1, round(parameters.dropout_min_duration_ms * fs / 1000.0))
    max_n = max(min_n, round(parameters.dropout_max_duration_ms * fs / 1000.0))
    constant_max_n = max(min_n, round(parameters.dropout_constant_max_duration_ms * fs / 1000.0))
    context_n = max(1, round(parameters.dropout_context_ms * fs / 1000.0))
    merge_n = max(0, round(parameters.dropout_merge_gap_ms * fs / 1000.0))
    speech = merge_intervals(speech_intervals or [])
    output: list[dict] = []

    for channel_index in range(values.shape[1]):
        x = values[:, channel_index]
        raw_candidates: list[dict] = []
        interval_geometry: dict[int, tuple[int, int, float]] = {}
        for interval_index, interval in enumerate(intervals):
            lo, hi = _interval_samples(interval, fs, len(x))
            segment = x[lo:hi]
            interval_ac_rms = _rms(segment, ac=True)
            interval_geometry[interval_index] = (lo, hi, interval_ac_rms)
            finite = np.isfinite(segment)

            exact_mask = finite & (np.abs(segment) <= parameters.dropout_zero_abs_threshold)
            for start, end in _runs(exact_mask):
                if min_n <= end - start <= max_n:
                    raw_candidates.append(
                        {
                            "event_type": "dropout",
                            "event_subtype": "near_zero_run",
                            "channel_index": channel_index,
                            "interval_index": interval_index,
                            "start_sample": lo + start,
                            "end_sample": lo + end,
                            "candidate_count_merged": 1,
                        }
                    )

            if segment.size >= min_n + 1:
                stable_steps = finite[1:] & finite[:-1] & (
                    np.abs(np.diff(segment)) <= parameters.dropout_constant_diff_abs_threshold
                )
                for diff_start, diff_end in _runs(stable_steps):
                    start = diff_start
                    end = diff_end + 1
                    duration_n = end - start
                    if min_n <= duration_n <= constant_max_n:
                        block = segment[start:end]
                        if np.all(np.abs(block) <= parameters.dropout_zero_abs_threshold):
                            continue
                        raw_candidates.append(
                            {
                                "event_type": "dropout",
                                "event_subtype": "constant_low_information_run",
                                "channel_index": channel_index,
                                "interval_index": interval_index,
                                "start_sample": lo + start,
                                "end_sample": lo + end,
                                "candidate_count_merged": 1,
                            }
                        )

        # Preserve every generated run in the candidate ledger. Event grouping is
        # performed only after classification, so a short active gap never causes
        # the merged span itself to fail the low-information test.
        for local_index, event in enumerate(raw_candidates):
            start = int(event["start_sample"])
            end = int(event["end_sample"])
            lo, hi, interval_ac_rms = interval_geometry[int(event["interval_index"])]
            metrics = _event_context_metrics(x, start, end, context_n, lo, hi)
            full_context = (
                metrics["left_context_sample_count"] == context_n
                and metrics["right_context_sample_count"] == context_n
            )
            context_floor = max(
                parameters.dropout_context_min_abs_ac_rms,
                parameters.dropout_context_min_interval_fraction * max(interval_ac_rms, 0.0),
            )
            bilateral_active = bool(
                full_context
                and np.isfinite(metrics["left_context_ac_rms"])
                and np.isfinite(metrics["right_context_ac_rms"])
                and min(metrics["left_context_ac_rms"], metrics["right_context_ac_rms"])
                >= context_floor
            )
            context_ac = min(metrics["left_context_ac_rms"], metrics["right_context_ac_rms"])
            ac_ratio = (
                context_ac / max(metrics["run_ac_rms"], 1e-15)
                if bilateral_active and np.isfinite(metrics["run_ac_rms"])
                else np.nan
            )
            context_raw = min(metrics["left_context_rms"], metrics["right_context_rms"])
            peak_fraction = (
                metrics["run_peak_abs"] / max(context_raw, 1e-15)
                if bilateral_active and np.isfinite(metrics["run_peak_abs"])
                else np.nan
            )
            std_ceiling = max(
                parameters.dropout_constant_std_abs_threshold,
                parameters.dropout_constant_std_context_fraction * max(context_ac, 0.0),
            )
            peak_ceiling = max(
                parameters.dropout_constant_peak_abs_threshold,
                parameters.dropout_constant_peak_context_fraction * max(context_raw, 0.0),
            )
            is_exact = event["event_subtype"] == "near_zero_run"
            low_information = bool(
                is_exact
                or (
                    np.isfinite(metrics["run_std"])
                    and np.isfinite(metrics["run_peak_abs"])
                    and metrics["run_std"] <= std_ceiling
                    and metrics["run_peak_abs"] <= peak_ceiling
                )
            )
            boundary_distance = min(
                _distance_to_boundaries(start / fs, speech),
                _distance_to_boundaries(end / fs, speech),
            )
            near_speech_boundary = boundary_distance <= parameters.dropout_boundary_review_guard_ms / 1000.0

            accepted = bool(
                bilateral_active
                and low_information
                and np.isfinite(ac_ratio)
                and ac_ratio >= parameters.dropout_context_run_ac_ratio_accept
            )
            indeterminate = bool(
                not accepted
                and bilateral_active
                and low_information
                and np.isfinite(ac_ratio)
                and ac_ratio >= parameters.dropout_context_run_ac_ratio_indeterminate
            )
            if accepted:
                disposition = "accepted"
                reason = ""
            elif indeterminate:
                disposition = "indeterminate"
                reason = "borderline_context_run_contrast"
            elif not bilateral_active:
                disposition = "rejected"
                reason = "insufficient_bilateral_active_context"
            elif not low_information:
                disposition = "rejected"
                reason = "run_not_sufficiently_low_information"
            else:
                disposition = "rejected"
                reason = "insufficient_context_run_contrast"

            output.append(
                {
                    **event,
                    **metrics,
                    "candidate_id": _candidate_id("dropout", channel_index, local_index),
                    "start_sec": start / fs,
                    "end_sec": end / fs,
                    "duration_sec": (end - start) / fs,
                    "support_duration_sec": (end - start) / fs,
                    "span_duration_sec": (end - start) / fs,
                    "support_intervals_json": json.dumps([[start / fs, end / fs]]),
                    "interval_ac_rms": interval_ac_rms,
                    "context_activity_floor": context_floor,
                    "context_run_ac_ratio": ac_ratio,
                    "run_peak_context_fraction": peak_fraction,
                    "constant_std_ceiling": std_ceiling,
                    "constant_peak_ceiling": peak_ceiling,
                    "nearest_speech_boundary_distance_sec": boundary_distance,
                    "near_speech_boundary": near_speech_boundary,
                    "score": float(ac_ratio) if np.isfinite(ac_ratio) else np.nan,
                    "initial_disposition": disposition,
                    "initial_reason": reason,
                }
            )
    return output


def _normalized_residual(a: np.ndarray, b: np.ndarray) -> float:
    denominator = sqrt(float(np.sum(a * a) + np.sum(b * b))) + 1e-15
    return float(np.linalg.norm(b - a) / denominator)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return np.nan
    return float(np.dot(a, b) / denominator)


def _spectral_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8 or b.size != a.size:
        return np.nan
    window = np.hanning(a.size)
    pa = np.log(np.abs(np.fft.rfft(a * window)) ** 2 + 1e-20)
    pb = np.log(np.abs(np.fft.rfft(b * window)) ** 2 + 1e-20)
    pa -= float(np.mean(pa))
    pb -= float(np.mean(pb))
    return _cosine_similarity(pa, pb)


def _spectral_entropy(x: np.ndarray) -> float:
    if x.size < 8:
        return np.nan
    power = np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(x.size))) ** 2
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    probabilities = power / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-20)))
    return entropy / max(np.log(len(probabilities)), 1e-15)


def _periodicity_similarity(x: np.ndarray, fs: int, max_lag_ms: float) -> tuple[float, float]:
    centered = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    if centered.size < 16 or _rms(centered) <= 0:
        return np.nan, np.nan
    minimum_lag = max(1, int(round(fs / 500.0)))
    maximum_lag = min(centered.size // 2, int(round(max_lag_ms * fs / 1000.0)))
    if maximum_lag < minimum_lag:
        return np.nan, np.nan
    best_similarity = -np.inf
    best_lag = np.nan
    for lag in range(minimum_lag, maximum_lag + 1):
        similarity = _cosine_similarity(centered[lag:], centered[:-lag])
        if np.isfinite(similarity) and similarity > best_similarity:
            best_similarity = similarity
            best_lag = lag / fs
    return float(best_similarity), float(best_lag)



def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Return all contiguous window sums in O(n)."""

    x = np.asarray(values, dtype=np.float64)
    if window <= 0 or x.size < window:
        return np.empty(0, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    return cumulative[window:] - cumulative[:-window]


def detect_duplicate_candidates(
    waveform: np.ndarray,
    sample_rate_hz: int,
    intervals: Iterable[TimeInterval],
    *,
    parameters: QTEMPParameters = DEFAULT_PARAMETERS,
) -> list[dict]:
    """Detect near-exact *consecutive* decoded repetition per native channel.

    This detector is intentionally narrow. It does not claim to detect arbitrary
    packet-loss concealment or codec-smeared freezes. Screening uses cumulative
    window statistics rather than materializing a three-dimensional window tensor,
    which substantially reduces runtime and memory use on long recordings.
    """

    values = _channels_float(waveform)
    fs = int(sample_rate_hz)
    frame_n = max(8, round(parameters.duplicate_frame_ms * fs / 1000.0))
    hop_n = max(1, round(parameters.duplicate_hop_ms * fs / 1000.0))
    minimum_n = max(frame_n, round(parameters.duplicate_min_sequence_ms * fs / 1000.0))
    maximum_n = max(minimum_n, round(parameters.duplicate_max_sequence_ms * fs / 1000.0))
    merge_n = max(0, round(parameters.duplicate_merge_gap_ms * fs / 1000.0))
    output: list[dict] = []

    for channel_index in range(values.shape[1]):
        x = values[:, channel_index]
        raw: list[dict] = []

        for interval_index, interval in enumerate(intervals):
            lo, hi = _interval_samples(interval, fs, len(x))
            segment = np.asarray(x[lo:hi], dtype=np.float64)
            if segment.size < 2 * minimum_n:
                continue

            interval_ac_rms = _rms(segment, ac=True)
            activity_floor = max(
                parameters.duplicate_min_ac_rms_abs,
                parameters.duplicate_min_interval_rms_fraction * max(interval_ac_rms, 0.0),
            )

            for lag_ms in parameters.duplicate_lags_ms:
                lag_n = max(frame_n, int(round(lag_ms * fs / 1000.0)))
                comparable_n = segment.size - lag_n
                if comparable_n < minimum_n:
                    continue

                source = segment[:comparable_n]
                target = segment[lag_n:]
                finite = np.isfinite(source) & np.isfinite(target)

                source_safe = np.where(finite, source, 0.0)
                target_safe = np.where(finite, target, 0.0)
                difference = target_safe - source_safe

                diff2 = difference * difference
                energy = source_safe * source_safe + target_safe * target_safe
                dot = source_safe * target_safe
                source2 = source_safe * source_safe
                target2 = target_safe * target_safe
                source_sum = source_safe
                target_sum = target_safe
                finite_float = finite.astype(np.float64)

                diff2_sum = _rolling_sum(diff2, frame_n)
                energy_sum = _rolling_sum(energy, frame_n)
                dot_sum = _rolling_sum(dot, frame_n)
                source2_sum = _rolling_sum(source2, frame_n)
                target2_sum = _rolling_sum(target2, frame_n)
                source_sum_window = _rolling_sum(source_sum, frame_n)
                target_sum_window = _rolling_sum(target_sum, frame_n)
                finite_count = _rolling_sum(finite_float, frame_n)

                if diff2_sum.size == 0:
                    continue

                starts = np.arange(0, diff2_sum.size, hop_n, dtype=int)
                residual = np.sqrt(np.maximum(diff2_sum[starts], 0.0)) / (
                    np.sqrt(np.maximum(energy_sum[starts], 0.0)) + 1e-15
                )
                cosine_denominator = np.sqrt(
                    np.maximum(source2_sum[starts], 0.0)
                    * np.maximum(target2_sum[starts], 0.0)
                )
                cosine = np.full(starts.size, np.nan, dtype=np.float64)
                valid_cosine = cosine_denominator > 0
                cosine[valid_cosine] = dot_sum[starts][valid_cosine] / cosine_denominator[valid_cosine]

                source_variance = np.maximum(
                    source2_sum[starts] / frame_n
                    - (source_sum_window[starts] / frame_n) ** 2,
                    0.0,
                )
                target_variance = np.maximum(
                    target2_sum[starts] / frame_n
                    - (target_sum_window[starts] / frame_n) ** 2,
                    0.0,
                )
                source_ac = np.sqrt(source_variance)
                target_ac = np.sqrt(target_variance)

                match = (
                    (finite_count[starts] == frame_n)
                    & (source_ac >= activity_floor)
                    & (target_ac >= activity_floor)
                    & np.isfinite(residual)
                    & np.isfinite(cosine)
                    & (residual <= parameters.duplicate_residual_indeterminate)
                    & (cosine >= parameters.duplicate_cosine_indeterminate)
                )

                for run_start, run_end in _runs(match):
                    target_start_local = int(starts[run_start] + lag_n)
                    target_end_local = int(starts[run_end - 1] + lag_n + frame_n)
                    sequence_n = target_end_local - target_start_local
                    # A single consecutive-copy event cannot use source support that
                    # overlaps its target. Longer freeze trains are represented as
                    # adjacent events and merged at recording level.
                    sequence_n = min(sequence_n, lag_n, maximum_n)
                    if sequence_n < minimum_n:
                        continue

                    target_end_local = target_start_local + sequence_n
                    source_start_local = target_start_local - lag_n
                    source_end_local = source_start_local + sequence_n
                    if source_start_local < 0 or target_end_local > segment.size:
                        continue

                    source_sequence = segment[source_start_local:source_end_local]
                    target_sequence = segment[target_start_local:target_end_local]
                    if (
                        source_sequence.size != target_sequence.size
                        or source_sequence.size < minimum_n
                        or not np.isfinite(source_sequence).all()
                        or not np.isfinite(target_sequence).all()
                    ):
                        continue

                    residual_full = _normalized_residual(source_sequence, target_sequence)
                    cosine_full = _cosine_similarity(source_sequence, target_sequence)
                    spectral_cosine = _spectral_cosine(source_sequence, target_sequence)
                    source_ac_rms = _rms(source_sequence, ac=True)
                    target_ac_rms = _rms(target_sequence, ac=True)
                    entropy = _spectral_entropy(np.r_[source_sequence, target_sequence])

                    absolute_source_start = lo + source_start_local
                    absolute_source_end = lo + source_end_local
                    absolute_target_start = lo + target_start_local
                    absolute_target_end = lo + target_end_local

                    periodicity_context_start = max(lo, absolute_source_start - round(0.08 * fs))
                    periodicity_context_end = min(hi, absolute_target_end + round(0.08 * fs))
                    periodicity_similarity, periodicity_lag_sec = _periodicity_similarity(
                        x[periodicity_context_start:periodicity_context_end],
                        fs,
                        parameters.duplicate_periodicity_lag_max_ms,
                    )

                    before = x[
                        max(lo, absolute_source_start - sequence_n):absolute_source_start
                    ]
                    after = x[
                        absolute_target_end:min(hi, absolute_target_end + sequence_n)
                    ]
                    novelty_values: list[float] = []
                    if before.size == source_sequence.size:
                        novelty_values.append(_normalized_residual(before, source_sequence))
                    if after.size == target_sequence.size:
                        novelty_values.append(_normalized_residual(target_sequence, after))
                    boundary_novelty = min(novelty_values) if novelty_values else np.nan

                    periodicity_guard = bool(
                        (
                            np.isfinite(periodicity_similarity)
                            and periodicity_similarity
                            >= parameters.duplicate_periodicity_similarity
                        )
                        or (
                            np.isfinite(entropy)
                            and entropy <= parameters.duplicate_low_entropy_threshold
                        )
                    )

                    strict = bool(
                        residual_full <= parameters.duplicate_residual_accept
                        and cosine_full >= parameters.duplicate_cosine_accept
                        and np.isfinite(boundary_novelty)
                        and boundary_novelty
                        >= parameters.duplicate_boundary_novelty_accept
                        and not periodicity_guard
                    )
                    borderline = bool(
                        not strict
                        and residual_full
                        <= parameters.duplicate_residual_indeterminate
                        and cosine_full
                        >= parameters.duplicate_cosine_indeterminate
                        and (
                            not np.isfinite(boundary_novelty)
                            or boundary_novelty
                            >= parameters.duplicate_boundary_novelty_indeterminate
                        )
                    )

                    if strict:
                        disposition = "accepted"
                        reason = ""
                    elif borderline:
                        disposition = "indeterminate"
                        reason = (
                            "periodicity_or_low_boundary_novelty"
                            if periodicity_guard
                            else "borderline_near_exact_repetition"
                        )
                    else:
                        disposition = "rejected"
                        reason = "insufficient_near_exact_repetition_evidence"

                    raw.append(
                        {
                            "event_type": "frozen_audio",
                            "event_subtype": "consecutive_near_exact_decoded_repetition",
                            "channel_index": channel_index,
                            "interval_index": interval_index,
                            "source_start_sample": absolute_source_start,
                            "source_end_sample": absolute_source_end,
                            "start_sample": absolute_target_start,
                            "end_sample": absolute_target_end,
                            "source_start_sec": absolute_source_start / fs,
                            "source_end_sec": absolute_source_end / fs,
                            "start_sec": absolute_target_start / fs,
                            "end_sec": absolute_target_end / fs,
                            "duration_sec": sequence_n / fs,
                            "support_duration_sec": sequence_n / fs,
                            "span_duration_sec": sequence_n / fs,
                            "support_intervals_json": json.dumps(
                                [[absolute_target_start / fs, absolute_target_end / fs]]
                            ),
                            "lag_samples": lag_n,
                            "lag_sec": lag_n / fs,
                            "lag_ms": lag_ms,
                            "matched_frame_count": int(run_end - run_start),
                            "normalized_residual": residual_full,
                            "cosine_similarity": cosine_full,
                            "spectral_cosine_similarity": spectral_cosine,
                            "source_ac_rms": source_ac_rms,
                            "target_ac_rms": target_ac_rms,
                            "activity_floor": activity_floor,
                            "boundary_novelty": boundary_novelty,
                            "spectral_entropy": entropy,
                            "periodicity_similarity": periodicity_similarity,
                            "periodicity_lag_sec": periodicity_lag_sec,
                            "periodicity_guard_triggered": periodicity_guard,
                            "score": float(-np.log10(max(residual_full, 1e-15))),
                            "initial_disposition": disposition,
                            "initial_reason": reason,
                            "candidate_count_merged": 1,
                        }
                    )

        # Resolve overlapping lag hypotheses by final disposition, residual,
        # duration, and boundary novelty. Keep rejected rows only when no stronger
        # hypothesis covers the same target support.
        raw.sort(
            key=lambda row: (
                row["start_sample"],
                row["end_sample"],
                DISPOSITIONS.index(row["initial_disposition"]),
                row["normalized_residual"],
                -row["duration_sec"],
            )
        )
        selected: list[dict] = []
        for event in raw:
            overlapping = [
                index
                for index, previous in enumerate(selected)
                if event["start_sample"] < previous["end_sample"]
                and previous["start_sample"] < event["end_sample"]
            ]
            if not overlapping:
                selected.append(event)
                continue
            event_key = (
                DISPOSITIONS.index(event["initial_disposition"]),
                event["normalized_residual"],
                -event["duration_sec"],
                -float(event.get("boundary_novelty", -np.inf)),
            )
            best_index = overlapping[0]
            previous = selected[best_index]
            previous_key = (
                DISPOSITIONS.index(previous["initial_disposition"]),
                previous["normalized_residual"],
                -previous["duration_sec"],
                -float(previous.get("boundary_novelty", -np.inf)),
            )
            if event_key < previous_key:
                selected[best_index] = event

        selected = _merge_raw_events(selected, merge_n)
        for local_index, event in enumerate(selected):
            event["candidate_id"] = _candidate_id(
                "frozen_audio", channel_index, local_index
            )
            output.append(event)

    return output



def _fit_lpc_yule_walker(values: np.ndarray, order: int) -> np.ndarray | None:
    """Fit a stable low-order AR predictor using Yule–Walker equations."""

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < max(8 * order, order + 32):
        return None
    centered = x - float(np.mean(x))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-20:
        return None
    autocorrelation = np.array(
        [
            float(np.dot(centered[lag:], centered[: centered.size - lag]))
            / centered.size
            for lag in range(order + 1)
        ],
        dtype=np.float64,
    )
    ridge = max(1e-12, 1e-6 * autocorrelation[0])
    autocorrelation[0] += ridge
    try:
        coefficients = linalg.solve_toeplitz(
            (autocorrelation[:-1], autocorrelation[:-1]),
            autocorrelation[1:],
            check_finite=False,
        )
    except (linalg.LinAlgError, ValueError):
        return None
    if not np.isfinite(coefficients).all():
        return None
    # Unstable LPC fits create huge residuals unrelated to continuity. A simple
    # coefficient-norm guard is conservative and deterministic.
    if float(np.sum(np.abs(coefficients))) > 8.0:
        return None
    return np.asarray(coefficients, dtype=np.float64)



def _local_ar_predict_next(
    context: np.ndarray,
    order: int,
) -> tuple[float, float]:
    """Predict the next sample and estimate robust in-context residual scale."""

    x = np.asarray(context, dtype=np.float64)
    if x.size < max(8 * order, order + 32) or not np.isfinite(x).all():
        return np.nan, np.nan
    coefficients = _fit_lpc_yule_walker(x, order)
    if coefficients is None:
        return np.nan, np.nan
    centered = x - float(np.mean(x))
    residual = _prediction_residual(x, coefficients)
    finite_residual = residual[np.isfinite(residual)]
    if finite_residual.size < 16:
        return np.nan, np.nan
    _, scale = _robust_location_scale(finite_residual)
    prediction = float(
        np.dot(centered[-order:][::-1], coefficients)
        + float(np.mean(x))
    )
    return prediction, scale


def _prediction_residual(values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """One-step forward residual aligned to the input samples."""

    x = np.asarray(values, dtype=np.float64)
    centered = x - float(np.mean(x))
    numerator = np.concatenate(([1.0], -np.asarray(coefficients, dtype=np.float64)))
    residual = signal.lfilter(numerator, [1.0], centered)
    residual[: len(coefficients)] = np.nan
    return residual


def _robust_positive_z(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    finite = absolute[np.isfinite(absolute)]
    if finite.size < 16:
        return np.full(absolute.shape, np.nan, dtype=np.float64)
    median, scale = _robust_location_scale(finite)
    return (absolute - median) / max(scale, 1e-15)


def _blockwise_continuity_scores(
    x: np.ndarray,
    lo: int,
    hi: int,
    fs: int,
    *,
    parameters: QTEMPParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute derivative and bilateral LPC residual z-scores once per block.

    Overlapping blocks reduce nonstationarity while avoiding one least-squares
    fit per candidate. Scores are combined by maximum evidence across blocks.
    """

    length = len(x)
    derivative_z = np.full(length, np.nan, dtype=np.float64)
    forward_z = np.full(length, np.nan, dtype=np.float64)
    backward_z = np.full(length, np.nan, dtype=np.float64)

    block_n = max(
        8 * parameters.splice_ar_order,
        int(round(parameters.splice_lpc_block_ms * fs / 1000.0)),
    )
    overlap = min(
        0.90,
        max(0.0, float(parameters.splice_lpc_overlap_fraction)),
    )
    hop_n = max(1, int(round(block_n * (1.0 - overlap))))

    starts = list(range(lo, max(lo + 1, hi - block_n + 1), hop_n))
    final_start = max(lo, hi - block_n)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    for start in sorted(set(starts)):
        end = min(hi, start + block_n)
        block = np.asarray(x[start:end], dtype=np.float64)
        if block.size < max(8 * parameters.splice_ar_order, 64):
            continue
        if not np.isfinite(block).all():
            continue

        coefficients_forward = _fit_lpc_yule_walker(
            block, parameters.splice_ar_order
        )
        coefficients_backward = _fit_lpc_yule_walker(
            block[::-1], parameters.splice_ar_order
        )
        if coefficients_forward is None or coefficients_backward is None:
            continue

        forward_residual = _prediction_residual(
            block, coefficients_forward
        )
        backward_residual = _prediction_residual(
            block[::-1], coefficients_backward
        )[::-1]
        local_forward_z = _robust_positive_z(forward_residual)
        local_backward_z = _robust_positive_z(backward_residual)

        local_derivative = np.abs(np.diff(block, prepend=block[0]))
        local_derivative_z = _robust_positive_z(local_derivative)

        target = slice(start, end)
        for destination, local in (
            (derivative_z, local_derivative_z),
            (forward_z, local_forward_z),
            (backward_z, local_backward_z),
        ):
            existing = destination[target]
            valid_local = np.isfinite(local)
            combined = existing.copy()
            only_local = valid_local & ~np.isfinite(existing)
            both = valid_local & np.isfinite(existing)
            combined[only_local] = local[only_local]
            combined[both] = np.maximum(existing[both], local[both])
            destination[target] = combined

    return derivative_z, forward_z, backward_z


def _nonmaximum_candidates(
    score: np.ndarray,
    candidate_mask: np.ndarray,
    lo: int,
    hi: int,
    refractory_n: int,
    maximum: int,
) -> list[int]:
    indices = np.flatnonzero(candidate_mask[lo:hi]) + lo
    if indices.size == 0:
        return []
    ordered = indices[np.argsort(score[indices])[::-1]]
    selected: list[int] = []
    for index in ordered:
        if all(abs(int(index) - previous) >= refractory_n for previous in selected):
            selected.append(int(index))
            if len(selected) >= maximum:
                break
    return sorted(selected)


def _near_any_interval(
    time_sec: float,
    intervals: Sequence[TimeInterval],
    guard_sec: float,
) -> bool:
    return any(
        item.start_sec - guard_sec <= time_sec <= item.end_sec + guard_sec
        for item in intervals
    )


def detect_splice_candidates(
    waveform: np.ndarray,
    sample_rate_hz: int,
    intervals: Iterable[TimeInterval],
    *,
    speech_intervals: Iterable[TimeInterval] | None = None,
    clipping_event_intervals: Iterable[TimeInterval] | None = None,
    parameters: QTEMPParameters = DEFAULT_PARAMETERS,
) -> list[dict]:
    """Detect abrupt bilateral prediction mismatches in the native stream.

    Scope is deliberately narrower than arbitrary editing or sample deletion:
    a temporally smooth join can be unidentifiable from no-reference audio and is
    characterized as a known false negative rather than forced into this feature.
    """

    values = _channels_float(waveform)
    fs = int(sample_rate_hz)
    context_n = max(
        3 * parameters.splice_ar_order + 2,
        round(parameters.splice_context_ms * fs / 1000.0),
    )
    edge_n = max(
        context_n,
        round(parameters.splice_edge_guard_ms * fs / 1000.0),
    )
    refractory_n = max(
        1, round(parameters.splice_refractory_ms * fs / 1000.0)
    )
    clipping = merge_intervals(clipping_event_intervals or [])
    # Preserve every frozen speech interval boundary. Adjacent intervals may
    # encode a genuine VAD onset/offset transition; merging them would erase
    # precisely the boundary needed by the splice false-positive guard.
    speech = sorted(
        [
            TimeInterval(float(item.start_sec), float(item.end_sec))
            for item in (speech_intervals or [])
            if np.isfinite(item.start_sec)
            and np.isfinite(item.end_sec)
            and float(item.end_sec) > float(item.start_sec)
        ],
        key=lambda item: (item.start_sec, item.end_sec),
    )
    clip_guard = parameters.splice_clipping_edge_guard_ms / 1000.0
    boundary_guard = parameters.splice_speech_boundary_guard_ms / 1000.0
    output: list[dict] = []

    for channel_index in range(values.shape[1]):
        x = values[:, channel_index]
        local_index = 0

        for interval_index, interval in enumerate(intervals):
            raw_lo, raw_hi = _interval_samples(interval, fs, len(x))
            lo = raw_lo + edge_n
            hi = raw_hi - edge_n
            if hi <= lo:
                continue

            interval_ac_rms = _rms(x[raw_lo:raw_hi], ac=True)
            activity_floor = max(
                parameters.splice_min_context_ac_rms_abs,
                parameters.splice_min_interval_rms_fraction
                * max(interval_ac_rms, 0.0),
            )

            derivative_z, forward_z, backward_z = _blockwise_continuity_scores(
                x, raw_lo, raw_hi, fs, parameters=parameters
            )
            bilateral_z = np.minimum(forward_z, backward_z)
            threshold_ratio = np.minimum(
                derivative_z / max(parameters.splice_derivative_z_indeterminate, 1e-15),
                bilateral_z / max(parameters.splice_prediction_z_indeterminate, 1e-15),
            )
            candidate_mask = (
                np.isfinite(derivative_z)
                & np.isfinite(bilateral_z)
                & (derivative_z >= parameters.splice_derivative_z_indeterminate)
                & (bilateral_z >= parameters.splice_prediction_z_indeterminate)
            )
            exposure_min = max((hi - lo) / fs / 60.0, 1e-6)
            maximum = max(
                1,
                int(
                    ceil(
                        parameters.splice_max_candidates_per_min
                        * exposure_min
                    )
                ),
            )
            samples = _nonmaximum_candidates(
                threshold_ratio,
                candidate_mask,
                lo,
                hi,
                refractory_n,
                maximum,
            )

            for sample in samples:
                left = x[sample - context_n:sample]
                right = x[sample:sample + context_n]
                left_ac = _rms(left, ac=True)
                right_ac = _rms(right, ac=True)
                if min(left_ac, right_ac) < activity_floor:
                    continue

                # Confirm the fast shortlist with independent local models on
                # each side. This rejects transient samples inside a changed
                # segment because both local models then describe the same source.
                left_prediction, left_scale = _local_ar_predict_next(
                    left, parameters.splice_ar_order
                )
                right_prediction, right_scale = _local_ar_predict_next(
                    right[::-1], parameters.splice_ar_order
                )
                left_error = (
                    abs(float(x[sample]) - left_prediction)
                    if np.isfinite(left_prediction)
                    else np.nan
                )
                right_error = (
                    abs(float(x[sample - 1]) - right_prediction)
                    if np.isfinite(right_prediction)
                    else np.nan
                )
                left_prediction_z = (
                    left_error / left_scale
                    if np.isfinite(left_scale) and left_scale > 0
                    else np.nan
                )
                right_prediction_z = (
                    right_error / right_scale
                    if np.isfinite(right_scale) and right_scale > 0
                    else np.nan
                )
                prediction_z = (
                    float(
                        np.sqrt(
                            max(left_prediction_z, 0.0)
                            * max(right_prediction_z, 0.0)
                        )
                    )
                    if np.isfinite([left_prediction_z, right_prediction_z]).all()
                    else np.nan
                )
                context_spectral_cosine = _spectral_cosine(left, right)
                # Exclude a short guard around the candidate when comparing
                # persistent left/right context. A one-sample impulse otherwise
                # contaminates the right spectrum and can mimic a source switch.
                spectral_guard_n = max(2, round(0.003 * fs))
                stable_left = left[:-spectral_guard_n] if left.size > 2 * spectral_guard_n else left
                stable_right = right[spectral_guard_n:] if right.size > 2 * spectral_guard_n else right
                guarded_context_spectral_cosine = _spectral_cosine(
                    stable_left, stable_right
                )
                local_derivative_z = float(derivative_z[sample])
                time_sec = sample / fs

                clipping_guard = _near_any_interval(
                    time_sec, clipping, clip_guard
                )
                boundary_distance = _distance_to_boundaries(
                    time_sec, speech
                )
                speech_boundary_guard = boundary_distance <= boundary_guard

                jump = abs(float(x[sample]) - float(x[sample - 1]))
                return_errors: list[float] = []
                for offset in (1, 2):
                    if sample + offset < len(x):
                        return_errors.append(
                            abs(
                                float(x[sample + offset])
                                - float(x[sample - 1])
                            )
                        )
                impulse_return_ratio = (
                    min(return_errors) / max(jump, 1e-15)
                    if return_errors
                    else np.nan
                )
                # A one-sample additive impulse returns almost immediately to
                # the pre-event trajectory. A genuine finite source replacement
                # can also have a small return ratio by chance, so the primary
                # guard is very strict; a looser 2% guard is used only for an
                # exceptionally large derivative outlier. This protects QADD
                # arbitration without suppressing observable source joins.
                impulse_guard = bool(
                    np.isfinite(impulse_return_ratio)
                    and (
                        impulse_return_ratio
                        <= parameters.splice_impulse_return_ratio
                        or (
                            impulse_return_ratio <= 0.02
                            and local_derivative_z >= 50.0
                        )
                    )
                )

                level_window = max(8, round(0.05 * fs))
                left_level = _rms(
                    x[max(raw_lo, sample - level_window):sample],
                    ac=True,
                )
                right_level = _rms(
                    x[sample:min(raw_hi, sample + level_window)],
                    ac=True,
                )
                level_shift_db = (
                    20.0
                    * np.log10(
                        max(right_level, 1e-15)
                        / max(left_level, 1e-15)
                    )
                    if np.isfinite(left_level)
                    and np.isfinite(right_level)
                    else np.nan
                )
                qgain_competing = bool(
                    np.isfinite(level_shift_db)
                    and abs(level_shift_db)
                    >= parameters.splice_level_step_review_db
                )

                strict = bool(
                    local_derivative_z
                    >= parameters.splice_derivative_z_accept
                    and prediction_z
                    >= parameters.splice_prediction_z_accept
                    and np.isfinite(guarded_context_spectral_cosine)
                    and guarded_context_spectral_cosine
                    <= parameters.splice_context_spectral_cosine_accept_max
                    and not clipping_guard
                    and not speech_boundary_guard
                    and not impulse_guard
                    and not qgain_competing
                )
                borderline = bool(
                    not strict
                    and local_derivative_z
                    >= parameters.splice_derivative_z_indeterminate
                    and prediction_z
                    >= parameters.splice_prediction_z_indeterminate
                    and (
                        not np.isfinite(guarded_context_spectral_cosine)
                        or guarded_context_spectral_cosine
                        <= parameters.splice_context_spectral_cosine_indeterminate_max
                    )
                    and not clipping_guard
                    and not speech_boundary_guard
                    and not impulse_guard
                )

                if strict:
                    disposition = "accepted"
                    reason = ""
                elif clipping_guard:
                    disposition = "rejected"
                    reason = "qdist_clipping_edge_guard"
                elif speech_boundary_guard:
                    disposition = "rejected"
                    reason = "speech_onset_offset_guard"
                elif impulse_guard:
                    disposition = "rejected"
                    reason = "qadd_impulse_like_guard"
                elif borderline:
                    disposition = "indeterminate"
                    reason = (
                        "qgain_level_step_competing"
                        if qgain_competing
                        else "borderline_abrupt_join_evidence"
                    )
                else:
                    disposition = "rejected"
                    reason = "insufficient_bilateral_prediction_mismatch"

                score = float(
                    min(
                        local_derivative_z
                        / max(parameters.splice_derivative_z_accept, 1e-15),
                        prediction_z
                        / max(parameters.splice_prediction_z_accept, 1e-15),
                    )
                )
                output.append(
                    {
                        "event_type": "splice",
                        "event_subtype": "abrupt_bilateral_prediction_mismatch",
                        "candidate_id": _candidate_id(
                            "splice", channel_index, local_index
                        ),
                        "channel_index": channel_index,
                        "interval_index": interval_index,
                        "start_sample": sample,
                        "end_sample": sample + 1,
                        "start_sec": time_sec,
                        "end_sec": (sample + 1) / fs,
                        "duration_sec": 1.0 / fs,
                        "support_duration_sec": 1.0 / fs,
                        "span_duration_sec": 1.0 / fs,
                        "support_intervals_json": json.dumps(
                            [[time_sec, (sample + 1) / fs]]
                        ),
                        "interval_ac_rms": interval_ac_rms,
                        "activity_floor": activity_floor,
                        "left_context_ac_rms": left_ac,
                        "right_context_ac_rms": right_ac,
                        "derivative_z": local_derivative_z,
                        "left_prediction_error_z": left_prediction_z,
                        "right_prediction_error_z": right_prediction_z,
                        "prediction_z": prediction_z,
                        "context_spectral_cosine": context_spectral_cosine,
                        "guarded_context_spectral_cosine": guarded_context_spectral_cosine,
                        "nearest_speech_boundary_distance_sec": boundary_distance,
                        "speech_boundary_guard": speech_boundary_guard,
                        "qdist_clipping_edge_guard": clipping_guard,
                        "impulse_return_ratio": impulse_return_ratio,
                        "qadd_impulse_like_guard": impulse_guard,
                        "left_context_level_ac_rms": left_level,
                        "right_context_level_ac_rms": right_level,
                        "level_shift_db": level_shift_db,
                        "qgain_level_step_competing": qgain_competing,
                        "score": score,
                        "initial_disposition": disposition,
                        "initial_reason": reason,
                        "candidate_count_merged": 1,
                    }
                )
                local_index += 1

    return output


def arbitrate_candidates(
    candidates: pd.DataFrame,
    *,
    parameters: QTEMPParameters = DEFAULT_PARAMETERS,
) -> pd.DataFrame:
    """Apply within-QTEMP precedence and preserve final disposition reasons."""

    if candidates.empty:
        return candidates.copy()
    frame = candidates.copy()
    frame["disposition"] = frame["initial_disposition"].astype(str)
    frame["disposition_reason"] = frame["initial_reason"].fillna("").astype(str)
    frame["arbitrated"] = False
    guard = parameters.splice_qtemp_event_guard_ms / 1000.0

    accepted_duration = frame.loc[
        frame["event_type"].isin(["dropout", "frozen_audio"])
        & frame["disposition"].eq("accepted")
    ]
    duration_intervals_by_channel: dict[int, list[TimeInterval]] = {}
    for channel, local in accepted_duration.groupby("channel_index"):
        duration_intervals_by_channel[int(channel)] = [
            TimeInterval(float(row.start_sec), float(row.end_sec))
            for row in local.itertuples(index=False)
        ]

    splice_mask = frame["event_type"].eq("splice") & frame["disposition"].isin(["accepted", "indeterminate"])
    for index in frame.index[splice_mask]:
        row = frame.loc[index]
        local_intervals = duration_intervals_by_channel.get(int(row["channel_index"]), [])
        if _near_any_interval(float(row["start_sec"]), local_intervals, guard):
            frame.at[index, "disposition"] = "rejected"
            frame.at[index, "disposition_reason"] = "overlap_with_accepted_dropout_or_duplicate"
            frame.at[index, "arbitrated"] = True

    # Dropout takes precedence over overlapping duplicate targets.
    accepted_dropouts = frame.loc[
        frame["event_type"].eq("dropout") & frame["disposition"].eq("accepted")
    ]
    dropout_by_channel: dict[int, list[TimeInterval]] = {}
    for channel, local in accepted_dropouts.groupby("channel_index"):
        dropout_by_channel[int(channel)] = [
            TimeInterval(float(row.start_sec), float(row.end_sec))
            for row in local.itertuples(index=False)
        ]
    duplicate_mask = frame["event_type"].eq("frozen_audio") & frame["disposition"].isin(["accepted", "indeterminate"])
    for index in frame.index[duplicate_mask]:
        row = frame.loc[index]
        local_intervals = dropout_by_channel.get(int(row["channel_index"]), [])
        if _interval_overlap(float(row["start_sec"]), float(row["end_sec"]), local_intervals) > 0:
            frame.at[index, "disposition"] = "rejected"
            frame.at[index, "disposition_reason"] = "overlap_with_accepted_dropout"
            frame.at[index, "arbitrated"] = True
    return frame


def _support_intervals_from_candidate(row: pd.Series) -> list[TimeInterval]:
    raw = row.get("support_intervals_json", "")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            intervals = [TimeInterval(float(item[0]), float(item[1])) for item in parsed]
            return [item for item in intervals if item.duration_sec > 0]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    start = float(row.get("start_sec", np.nan))
    end = float(row.get("end_sec", np.nan))
    return [TimeInterval(start, end)] if np.isfinite(start) and np.isfinite(end) and end > start else []


def _merge_recording_events(
    disposition_ledger: pd.DataFrame,
    *,
    parameters: QTEMPParameters = DEFAULT_PARAMETERS,
) -> pd.DataFrame:
    """Collapse accepted channel candidates into recording-level event unions.

    Candidate rows remain untouched. Accepted candidates are grouped with a
    versioned type-specific temporal tolerance. Duration burden is the union of
    observed support intervals, not the enclosing span, so an active gap used
    only for event-count grouping is never counted as corrupted time.
    """

    if disposition_ledger.empty:
        return pd.DataFrame()
    accepted = disposition_ledger.loc[disposition_ledger["disposition"].eq("accepted")].copy()
    if accepted.empty:
        return pd.DataFrame()
    cross_sec = parameters.cross_channel_merge_ms / 1000.0
    type_merge_sec = {
        "dropout": max(cross_sec, parameters.dropout_merge_gap_ms / 1000.0),
        "frozen_audio": max(cross_sec, parameters.duplicate_merge_gap_ms / 1000.0),
        "splice": max(cross_sec, parameters.splice_refractory_ms / 1000.0 + 0.002),
    }
    rows: list[dict] = []
    for event_type, local in accepted.groupby("event_type", sort=False):
        local = local.sort_values(["start_sec", "end_sec", "channel_index"])
        tolerance = type_merge_sec.get(str(event_type), cross_sec)
        groups: list[list[pd.Series]] = []
        for _, row in local.iterrows():
            if not groups:
                groups.append([row])
                continue
            previous_rows = groups[-1]
            group_start = min(float(item["start_sec"]) for item in previous_rows)
            group_end = max(float(item["end_sec"]) for item in previous_rows)
            if event_type == "splice":
                compatible = abs(float(row["start_sec"]) - group_start) <= tolerance
            else:
                compatible = (
                    float(row["start_sec"]) <= group_end + tolerance
                    and float(row["end_sec"]) >= group_start - tolerance
                )
            if compatible:
                previous_rows.append(row)
            else:
                groups.append([row])

        for group in groups:
            start_sec = min(float(item["start_sec"]) for item in group)
            end_sec = max(float(item["end_sec"]) for item in group)
            channels = sorted({int(item["channel_index"]) for item in group})
            best = max(
                group,
                key=lambda item: (
                    float(item.get("score", -np.inf))
                    if np.isfinite(item.get("score", np.nan))
                    else -np.inf
                ),
            )
            support_intervals = []
            for item in group:
                support_intervals.extend(_support_intervals_from_candidate(item))
            support_union = merge_intervals(support_intervals)
            support_duration = float(sum(item.duration_sec for item in support_union))
            row = {
                "event_type": event_type,
                "event_subtype": str(best.get("event_subtype", "")),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "span_duration_sec": max(0.0, end_sec - start_sec),
                "support_duration_sec": support_duration,
                "duration_sec": support_duration,
                "support_intervals_json": json.dumps(
                    [[item.start_sec, item.end_sec] for item in support_union]
                ),
                "channels_detected": "|".join(str(value) for value in channels),
                "channel_count": len(channels),
                "channel_candidate_ids": "|".join(str(item["candidate_id"]) for item in group),
                "channel_candidate_count": len(group),
                "maximum_score": max(float(item.get("score", np.nan)) for item in group),
                "source_start_sec": best.get("source_start_sec", np.nan),
                "source_end_sec": best.get("source_end_sec", np.nan),
                "lag_sec": best.get("lag_sec", np.nan),
            }
            rows.append(row)
    event_ledger = pd.DataFrame(rows).sort_values(["start_sec", "event_type"]).reset_index(drop=True)
    event_ledger.insert(0, "event_id", [f"event-{index:07d}" for index in range(len(event_ledger))])
    return event_ledger

def poisson_rate_interval(
    count: int,
    exposure_sec: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact Poisson interval in events/minute."""

    if exposure_sec <= 0 or count < 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence
    low_count = 0.0 if count == 0 else 0.5 * stats.chi2.ppf(alpha / 2.0, 2 * count)
    high_count = 0.5 * stats.chi2.ppf(1.0 - alpha / 2.0, 2 * (count + 1))
    factor = 60.0 / exposure_sec
    return float(low_count * factor), float(high_count * factor)


def reconstruct_recording_features(
    event_ledger: pd.DataFrame,
    eligible_duration_sec: float,
) -> dict:
    if eligible_duration_sec <= 0:
        return {feature: np.nan for feature in ANALYSIS_FEATURES}
    accepted = event_ledger.copy()
    if "disposition" in accepted:
        accepted = accepted.loc[accepted["disposition"].eq("accepted")]

    def local(event_type: str) -> pd.DataFrame:
        if accepted.empty or "event_type" not in accepted:
            return accepted.iloc[0:0]
        return accepted.loc[accepted["event_type"].eq(event_type)]

    dropout = local("dropout")
    frozen = local("frozen_audio")
    splice = local("splice")
    return {
        "qtemp_dropout_duration_fraction": float(dropout["duration_sec"].sum() / eligible_duration_sec) if len(dropout) else 0.0,
        "qtemp_dropout_event_rate_per_min": float(len(dropout) * 60.0 / eligible_duration_sec),
        "qtemp_frozen_audio_duration_fraction": float(frozen["duration_sec"].sum() / eligible_duration_sec) if len(frozen) else 0.0,
        "qtemp_frozen_audio_event_rate_per_min": float(len(frozen) * 60.0 / eligible_duration_sec),
        "qtemp_splice_discontinuity_rate_per_min": float(len(splice) * 60.0 / eligible_duration_sec),
    }


def extract_qtemp(
    waveform: np.ndarray,
    sample_rate_hz: int,
    *,
    analysis_intervals: Iterable[TimeInterval] | None = None,
    speech_intervals: Iterable[TimeInterval] | None = None,
    logical_recording_id: str = "",
    native_source_confirmed: bool = True,
    preprocessing_provenance_ok: bool = True,
    clipping_event_intervals: Iterable[TimeInterval] | None = None,
    parameters: QTEMPParameters = DEFAULT_PARAMETERS,
    enabled_event_types: Sequence[str] | None = None,
) -> QTEMPExtraction:
    """Extract candidate, disposition, accepted-event, exposure, and feature ledgers.

    ``enabled_event_types`` is an analytical-validation control that permits a
    detector-specific run (for example, only ``("dropout",)`` during a
    dropout dose grid). Production cohort extraction should omit it so all
    three detectors and the within-QTEMP arbitration layer are executed.
    """

    enabled = tuple(EVENT_TYPES if enabled_event_types is None else enabled_event_types)
    unknown = sorted(set(enabled).difference(EVENT_TYPES))
    if unknown:
        raise ValueError(f"Unsupported QTEMP event type(s): {unknown}")
    if len(set(enabled)) != len(enabled):
        raise ValueError("enabled_event_types must not contain duplicates")

    values = _channels_float(waveform)
    fs = int(sample_rate_hz)
    duration_sec = values.shape[0] / fs if fs > 0 else 0.0
    intervals = eligible_intervals(duration_sec, analysis_intervals, parameters=parameters)
    # Retain the original frozen speech boundaries for splice arbitration.
    # Detector-specific functions may merge support where appropriate.
    speech = sorted(
        [
            TimeInterval(float(item.start_sec), float(item.end_sec))
            for item in (speech_intervals or [])
            if np.isfinite(item.start_sec)
            and np.isfinite(item.end_sec)
            and float(item.end_sec) > float(item.start_sec)
        ],
        key=lambda item: (item.start_sec, item.end_sec),
    )

    exposure_rows = []
    for interval_index, interval in enumerate(intervals):
        start, end = _interval_samples(interval, fs, values.shape[0])
        exposure_rows.append(
            {
                "logical_recording_id": str(logical_recording_id),
                "qtemp_measurement_version": MEASUREMENT_VERSION,
                "interval_index": interval_index,
                "start_sec": interval.start_sec,
                "end_sec": interval.end_sec,
                "duration_sec": interval.duration_sec,
                "sample_count": end - start,
                "finite_sample_count_all_channels": int(np.isfinite(values[start:end]).all(axis=1).sum()),
                "native_channel_count": values.shape[1],
            }
        )
    exposure_ledger = pd.DataFrame(exposure_rows)
    eligible_duration_sec = float(exposure_ledger["duration_sec"].sum()) if len(exposure_ledger) else 0.0

    base = {
        "logical_recording_id": str(logical_recording_id),
        "qtemp_measurement_version": MEASUREMENT_VERSION,
        "qtemp_native_sample_rate_hz": fs,
        "qtemp_native_channel_count": int(values.shape[1]),
        "qtemp_native_duration_sec": duration_sec,
        "qtemp_eligible_duration_sec": eligible_duration_sec,
        "qtemp_eligible_interval_count": int(len(intervals)),
        "qtemp_native_source_confirmed": bool(native_source_confirmed),
        "qtemp_preprocessing_provenance_ok": bool(preprocessing_provenance_ok),
        "qtemp_channel_aggregation": "per-channel detection; temporally coincident accepted events collapsed by union",
        "qtemp_enabled_event_types": "|".join(enabled),
    }

    unavailable_reason = ""
    if not native_source_confirmed:
        unavailable_reason = "unavailable_native_source"
    elif not preprocessing_provenance_ok:
        unavailable_reason = "unavailable_preprocessing_provenance"
    elif fs <= 0 or values.shape[0] == 0:
        unavailable_reason = "decode_failure_or_empty_waveform"
    elif not np.isfinite(values).all():
        unavailable_reason = "nonfinite_native_waveform"
    elif eligible_duration_sec < parameters.minimum_eligible_duration_sec:
        unavailable_reason = "insufficient_exposure"

    if unavailable_reason:
        recording = {**base, "qtemp_status": unavailable_reason}
        for feature in ANALYSIS_FEATURES:
            recording[feature] = np.nan
            recording[f"{feature}_status"] = unavailable_reason
        empty = pd.DataFrame()
        return QTEMPExtraction(recording, empty, empty, empty, exposure_ledger)

    dropout = (
        detect_dropout_candidates(
            values,
            fs,
            intervals,
            speech_intervals=speech,
            parameters=parameters,
        )
        if "dropout" in enabled
        else []
    )
    duplicate = (
        detect_duplicate_candidates(values, fs, intervals, parameters=parameters)
        if "frozen_audio" in enabled
        else []
    )
    splice = (
        detect_splice_candidates(
            values,
            fs,
            intervals,
            speech_intervals=speech,
            clipping_event_intervals=clipping_event_intervals,
            parameters=parameters,
        )
        if "splice" in enabled
        else []
    )
    candidate_ledger = pd.DataFrame(dropout + duplicate + splice)
    if len(candidate_ledger):
        candidate_ledger.insert(0, "logical_recording_id", str(logical_recording_id))
        candidate_ledger.insert(1, "qtemp_measurement_version", MEASUREMENT_VERSION)
        candidate_ledger = candidate_ledger.sort_values(
            ["channel_index", "start_sec", "event_type", "candidate_id"],
            na_position="last",
        ).reset_index(drop=True)
    disposition_ledger = arbitrate_candidates(candidate_ledger, parameters=parameters)
    event_ledger = _merge_recording_events(disposition_ledger, parameters=parameters)
    if len(event_ledger):
        event_ledger.insert(0, "logical_recording_id", str(logical_recording_id))
        event_ledger.insert(1, "qtemp_measurement_version", MEASUREMENT_VERSION)
        event_ledger["disposition"] = "accepted"

    features = reconstruct_recording_features(event_ledger, eligible_duration_sec)
    recording = {**base, "qtemp_status": "measured"}
    recording.update(features)
    for feature in ANALYSIS_FEATURES:
        recording[f"{feature}_status"] = "measured_positive" if features[feature] > 0 else "measured_zero"

    for event_type, prefix in (
        ("dropout", "dropout"),
        ("frozen_audio", "frozen_audio"),
        ("splice", "splice"),
    ):
        typed_candidates = (
            disposition_ledger.loc[disposition_ledger["event_type"].eq(event_type)]
            if len(disposition_ledger)
            else disposition_ledger
        )
        typed_events = (
            event_ledger.loc[event_ledger["event_type"].eq(event_type)]
            if len(event_ledger)
            else event_ledger
        )
        recording[f"qtemp_{prefix}_candidate_count"] = int(len(typed_candidates))
        for disposition in DISPOSITIONS:
            recording[f"qtemp_{prefix}_{disposition}_candidate_count"] = int(
                typed_candidates["disposition"].eq(disposition).sum()
            ) if len(typed_candidates) else 0
        recording[f"qtemp_{prefix}_accepted_event_count"] = int(len(typed_events))
        low, high = poisson_rate_interval(len(typed_events), eligible_duration_sec)
        recording[f"qtemp_{prefix}_rate_ci95_low_per_min"] = low
        recording[f"qtemp_{prefix}_rate_ci95_high_per_min"] = high

    return QTEMPExtraction(
        recording=recording,
        candidate_ledger=candidate_ledger,
        disposition_ledger=disposition_ledger,
        event_ledger=event_ledger,
        exposure_ledger=exposure_ledger,
    )


# ---------------------------------------------------------------------------
# Synthetic perturbation helpers used by governed tests and notebook.
# ---------------------------------------------------------------------------

def inject_dropout(
    waveform: np.ndarray,
    sample_rate_hz: int,
    start_sec: float,
    duration_ms: float,
    *,
    mode: Literal["zero", "constant", "attenuated"] = "zero",
    constant_value: float = 0.0,
    attenuation_db: float = -60.0,
    channel: int | None = None,
) -> np.ndarray:
    values = _channels_float(waveform).copy()
    start = int(round(start_sec * sample_rate_hz))
    end = min(values.shape[0], start + int(round(duration_ms * sample_rate_hz / 1000.0)))
    if start < 0 or start >= end:
        raise ValueError("invalid dropout injection interval")
    channels = range(values.shape[1]) if channel is None else [int(channel)]
    for channel_index in channels:
        if mode == "zero":
            values[start:end, channel_index] = 0.0
        elif mode == "constant":
            values[start:end, channel_index] = float(constant_value)
        elif mode == "attenuated":
            values[start:end, channel_index] *= 10.0 ** (attenuation_db / 20.0)
        else:
            raise ValueError(f"unsupported dropout mode: {mode}")
    return values[:, 0] if np.asarray(waveform).ndim == 1 else values


def inject_zero_dropout(
    waveform: np.ndarray,
    sample_rate_hz: int,
    start_sec: float,
    duration_ms: float,
    value: float = 0.0,
) -> np.ndarray:
    return inject_dropout(
        waveform,
        sample_rate_hz,
        start_sec,
        duration_ms,
        mode="constant" if value != 0.0 else "zero",
        constant_value=value,
    )


def inject_duplicate(
    waveform: np.ndarray,
    sample_rate_hz: int,
    source_start_sec: float,
    target_start_sec: float,
    duration_ms: float,
    *,
    perturbation_sd: float = 0.0,
    random_seed: int = 20260731,
    channel: int | None = None,
) -> np.ndarray:
    values = _channels_float(waveform).copy()
    source_start = int(round(source_start_sec * sample_rate_hz))
    target_start = int(round(target_start_sec * sample_rate_hz))
    length = int(round(duration_ms * sample_rate_hz / 1000.0))
    if min(source_start, target_start, length) < 0 or source_start + length > values.shape[0] or target_start + length > values.shape[0]:
        raise ValueError("duplicate injection exceeds waveform")
    channels = range(values.shape[1]) if channel is None else [int(channel)]
    rng = np.random.default_rng(random_seed)
    for channel_index in channels:
        copy = values[source_start : source_start + length, channel_index].copy()
        if perturbation_sd > 0:
            copy += rng.normal(0.0, perturbation_sd, size=copy.size)
        values[target_start : target_start + length, channel_index] = copy
    return values[:, 0] if np.asarray(waveform).ndim == 1 else values


def inject_consecutive_duplicate(
    waveform: np.ndarray,
    sample_rate_hz: int,
    source_start_sec: float,
    duration_ms: float,
    *,
    perturbation_sd: float = 0.0,
    random_seed: int = 20260731,
    channel: int | None = None,
) -> np.ndarray:
    target_start_sec = source_start_sec + duration_ms / 1000.0
    return inject_duplicate(
        waveform,
        sample_rate_hz,
        source_start_sec,
        target_start_sec,
        duration_ms,
        perturbation_sd=perturbation_sd,
        random_seed=random_seed,
        channel=channel,
    )


def inject_splice_delete(
    waveform: np.ndarray,
    sample_rate_hz: int,
    start_sec: float,
    duration_ms: float,
) -> tuple[np.ndarray, float]:
    raw = np.asarray(waveform)
    start = int(round(start_sec * sample_rate_hz))
    length = int(round(duration_ms * sample_rate_hz / 1000.0))
    if start < 0 or start + length >= raw.shape[0]:
        raise ValueError("splice injection exceeds waveform")
    return np.concatenate([raw[:start], raw[start + length :]], axis=0), start / sample_rate_hz


def inject_splice_insert(
    waveform: np.ndarray,
    sample_rate_hz: int,
    target_sec: float,
    inserted: np.ndarray,
) -> tuple[np.ndarray, float]:
    raw = np.asarray(waveform)
    target = int(round(target_sec * sample_rate_hz))
    if target < 0 or target > raw.shape[0]:
        raise ValueError("insert target exceeds waveform")
    insert_values = np.asarray(inserted)
    if raw.ndim != insert_values.ndim:
        raise ValueError("inserted waveform dimensionality mismatch")
    return np.concatenate([raw[:target], insert_values, raw[target:]], axis=0), target / sample_rate_hz



def inject_splice_replace(
    waveform: np.ndarray,
    sample_rate_hz: int,
    target_sec: float,
    replacement: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Replace equal-length support and return both abrupt join boundaries."""

    raw = np.asarray(waveform)
    replacement_values = np.asarray(replacement)
    if raw.ndim != replacement_values.ndim:
        raise ValueError("replacement waveform dimensionality mismatch")
    target = int(round(target_sec * sample_rate_hz))
    length = int(replacement_values.shape[0])
    if length <= 0 or target < 0 or target + length > raw.shape[0]:
        raise ValueError("replacement interval exceeds waveform")
    output = raw.copy()
    output[target:target + length] = replacement_values
    return output, (
        target / sample_rate_hz,
        (target + length) / sample_rate_hz,
    )


def match_events_to_truth_points(
    event_ledger: pd.DataFrame,
    *,
    event_type: str,
    truth_times_sec: Sequence[float],
    tolerance_ms: float = 10.0,
) -> dict:
    """Match accepted point events to one or more truth boundaries.

    Returns boundary-level recall and localization errors without assuming that
    an inserted/replaced segment has only one relevant join.
    """

    truth = np.asarray(list(truth_times_sec), dtype=np.float64)
    truth = truth[np.isfinite(truth)]
    if truth.size == 0:
        return {
            "truth_boundary_count": 0,
            "matched_boundary_count": 0,
            "boundary_recall": np.nan,
            "median_abs_error_ms": np.nan,
            "maximum_abs_error_ms": np.nan,
        }
    if (
        event_ledger.empty
        or "event_type" not in event_ledger.columns
        or "start_sec" not in event_ledger.columns
    ):
        return {
            "truth_boundary_count": int(truth.size),
            "matched_boundary_count": 0,
            "boundary_recall": 0.0,
            "median_abs_error_ms": np.nan,
            "maximum_abs_error_ms": np.nan,
        }

    observed = pd.to_numeric(
        event_ledger.loc[
            event_ledger["event_type"].eq(event_type), "start_sec"
        ],
        errors="coerce",
    ).dropna().to_numpy(dtype=np.float64)
    tolerance_sec = tolerance_ms / 1000.0
    available = list(range(len(observed)))
    errors: list[float] = []

    for truth_time in truth:
        if not available:
            break
        distances = np.abs(observed[available] - truth_time)
        local = int(np.argmin(distances))
        if float(distances[local]) <= tolerance_sec:
            errors.append(float(distances[local] * 1000.0))
            available.pop(local)

    return {
        "truth_boundary_count": int(truth.size),
        "matched_boundary_count": int(len(errors)),
        "boundary_recall": float(len(errors) / truth.size),
        "median_abs_error_ms": (
            float(np.median(errors)) if errors else np.nan
        ),
        "maximum_abs_error_ms": (
            float(np.max(errors)) if errors else np.nan
        ),
    }


def apply_gain_step(
    waveform: np.ndarray,
    sample_rate_hz: int,
    start_sec: float,
    gain_db: float,
) -> np.ndarray:
    output = np.asarray(waveform, dtype=np.float64).copy()
    start = int(round(start_sec * sample_rate_hz))
    output[start:] *= 10.0 ** (gain_db / 20.0)
    return output


def inject_impulse(
    waveform: np.ndarray,
    sample_rate_hz: int,
    time_sec: float,
    amplitude: float,
) -> np.ndarray:
    output = np.asarray(waveform, dtype=np.float64).copy()
    index = int(round(time_sec * sample_rate_hz))
    output[index] += amplitude
    return output


def hard_clip(waveform: np.ndarray, threshold: float) -> np.ndarray:
    return np.clip(np.asarray(waveform, dtype=np.float64), -abs(threshold), abs(threshold))


def match_events_to_truth(
    event_ledger: pd.DataFrame,
    *,
    event_type: str,
    truth_start_sec: float,
    truth_end_sec: float | None = None,
    tolerance_ms: float = 10.0,
) -> dict:
    """Return nearest-event localization and overlap evidence for validation tables."""

    if truth_end_sec is None:
        truth_end_sec = truth_start_sec
    if event_ledger.empty or "event_type" not in event_ledger:
        return {
            "detected": False,
            "matched_event_id": pd.NA,
            "start_error_ms": np.nan,
            "end_error_ms": np.nan,
            "intersection_over_union": 0.0,
        }
    local = event_ledger.loc[event_ledger["event_type"].eq(event_type)].copy()
    if local.empty:
        return {
            "detected": False,
            "matched_event_id": pd.NA,
            "start_error_ms": np.nan,
            "end_error_ms": np.nan,
            "intersection_over_union": 0.0,
        }
    local["distance"] = np.abs(local["start_sec"].astype(float) - truth_start_sec)
    row = local.sort_values("distance").iloc[0]
    start_error_ms = 1000.0 * (float(row["start_sec"]) - truth_start_sec)
    end_error_ms = 1000.0 * (float(row["end_sec"]) - truth_end_sec)
    intersection = max(0.0, min(float(row["end_sec"]), truth_end_sec) - max(float(row["start_sec"]), truth_start_sec))
    union = max(float(row["end_sec"]), truth_end_sec) - min(float(row["start_sec"]), truth_start_sec)
    iou = intersection / union if union > 0 else float(abs(start_error_ms) <= tolerance_ms)
    detected = bool(abs(start_error_ms) <= tolerance_ms or iou > 0)
    return {
        "detected": detected,
        "matched_event_id": row.get("event_id", pd.NA),
        "start_error_ms": start_error_ms,
        "end_error_ms": end_error_ms,
        "intersection_over_union": float(iou),
    }
