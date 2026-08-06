"""QCHAN v3.0.1: cohort-relative channel/device spectral descriptors.

The module implements a four-feature profile.  It does not identify a device,
estimate a microphone transfer function, or construct a scalar quality score.
All reference-dependent values are meaningful only within their frozen
reference vintage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import signal, stats


MEASUREMENT_VERSION = "qchan-v3.0.1"

ANALYSIS_FEATURES = (
    "qchan_ltas_distance_db",
    "qchan_rolloff95_deficit_hz",
    "qchan_highband_ratio_deficit",
    "qchan_tilt_steepening_db_per_oct",
)
PRIMARY_FEATURES = (
    "qchan_ltas_distance_db",
    "qchan_rolloff95_deficit_hz",
)
SECONDARY_FEATURES = (
    "qchan_highband_ratio_deficit",
    "qchan_tilt_steepening_db_per_oct",
)

FEATURE_DEFINITIONS = {
    "qchan_ltas_distance_db": {
        "display_name": "Reference-relative LTAS distance",
        "subdomain": "spectral coloration",
        "role": "primary nonordinal",
        "unit": "dB RMS",
        "estimand": (
            "RMS difference between target and frozen-reference "
            "gain-normalized one-third-octave log-LTAS"
        ),
        "orientation": (
            "Higher means greater spectral deviation; not intrinsically worse."
        ),
        "claim_boundary": (
            "Cohort-relative spectral deviation; not device identification "
            "or a pure transfer-function estimate."
        ),
        "minimum_support": (
            "Target: at least 3 s guarded strict speech; reference: at least "
            "5 other subjects and 8 recordings in the same task stratum."
        ),
        "known_confounds": (
            "ALS phenotype, phonetic composition, sex, age, articulation, "
            "additive noise, source bandwidth, and reference composition."
        ),
        "evidence_class": "study-specific reference-relative estimator",
    },
    "qchan_rolloff95_deficit_hz": {
        "display_name": "Reference-relative rolloff95 deficit",
        "subdomain": "bandwidth attenuation",
        "role": "primary",
        "unit": "Hz",
        "estimand": "max(0, reference rolloff95 - target rolloff95)",
        "orientation": (
            "Higher means less upper spectral extent than the reference."
        ),
        "claim_boundary": (
            "One-sided bandwidth-loss proxy; not a codec or device label; "
            "upward deviations are retained only in the audit precursor."
        ),
        "minimum_support": (
            "Same target/reference support as LTAS distance; native source "
            "Nyquist and bandwidth status must be retained."
        ),
        "known_confounds": (
            "Fricative content, dysarthria, additive noise, task, sex, age, "
            "and source sample rate."
        ),
        "evidence_class": "study-specific reference-relative estimator",
    },
    "qchan_highband_ratio_deficit": {
        "display_name": "Reference-relative high-band deficit",
        "subdomain": "bandwidth attenuation",
        "role": "secondary",
        "unit": "proportion",
        "estimand": (
            "max(0, reference minus target) for 3-7.5-kHz / "
            "0.1-7.5-kHz integrated power"
        ),
        "orientation": "Higher means less relative high-band energy.",
        "claim_boundary": (
            "Secondary attenuation proxy; content-dependent and expected to "
            "overlap rolloff95."
        ),
        "minimum_support": "Same target/reference support as LTAS distance.",
        "known_confounds": (
            "Frication, articulation, additive high-frequency noise, source "
            "sample rate, and task."
        ),
        "evidence_class": "study-specific reference-relative estimator",
    },
    "qchan_tilt_steepening_db_per_oct": {
        "display_name": "Reference-relative spectral-tilt steepening",
        "subdomain": "spectral coloration",
        "role": "secondary phenotype-sensitive",
        "unit": "dB/octave",
        "estimand": (
            "max(0, reference Theil-Sen log-LTAS slope - target slope), "
            "100-4000 Hz"
        ),
        "orientation": (
            "Higher means a steeper downward spectral tilt than the reference."
        ),
        "claim_boundary": (
            "Secondary phenotype-sensitive proxy; not independent evidence "
            "when redundant with bandwidth features."
        ),
        "minimum_support": "Same target/reference support as LTAS distance.",
        "known_confounds": (
            "Glottal source, vocal effort, dysarthria, sex, age, phonetic mix, "
            "and additive noise."
        ),
        "evidence_class": "study-specific reference-relative estimator",
    },
}


@dataclass(frozen=True)
class QChanParameters:
    analysis_sample_rate_hz: int = 16_000
    frame_ms: float = 40.0
    hop_ms: float = 10.0
    speech_boundary_guard_ms: float = 200.0
    n_fft: int = 2048
    analysis_low_hz: float = 100.0
    analysis_high_hz: float = 7500.0
    highband_low_hz: float = 3000.0
    highband_high_hz: float = 7500.0
    tilt_low_hz: float = 100.0
    tilt_high_hz: float = 4000.0
    rolloff_fraction: float = 0.95
    octave_fraction: int = 3
    relative_psd_floor_db: float = -80.0
    minimum_guarded_speech_sec: float = 3.0
    minimum_frames: int = 100
    moderate_support_sec: float = 5.0
    high_support_sec: float = 10.0
    minimum_reference_subjects: int = 5
    minimum_reference_recordings: int = 8
    reference_requires_full_analysis_band: bool = True
    task_matching_required: bool = True
    random_seed: int = 20260730

    def __post_init__(self) -> None:
        if self.analysis_sample_rate_hz < 2 * self.analysis_high_hz:
            raise ValueError("Analysis rate does not support the requested upper band.")
        if not 0 < self.frame_ms or not 0 < self.hop_ms:
            raise ValueError("Frame and hop durations must be positive.")
        if self.hop_ms > self.frame_ms:
            raise ValueError("Hop duration cannot exceed frame duration.")
        if not 0 < self.rolloff_fraction < 1:
            raise ValueError("Rolloff fraction must lie strictly between zero and one.")
        if self.minimum_reference_subjects < 2:
            raise ValueError("At least two reference subjects are required.")
        if self.minimum_reference_recordings < self.minimum_reference_subjects:
            raise ValueError(
                "Reference recording requirement cannot be smaller than "
                "the reference subject requirement."
            )

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMETERS = QChanParameters()


@dataclass(frozen=True)
class TimeInterval:
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class RecordingSpectrum:
    logical_recording_id: str
    frequencies_hz: np.ndarray
    normalized_psd_per_hz: np.ndarray
    status: str
    support_tier: str
    guarded_speech_support_sec: float
    valid_frame_count: int
    guarded_segment_count: int
    zero_frame_count: int
    source_sample_rate_hz: float
    source_nyquist_hz: float
    source_bandwidth_limited: bool
    spectrum_sha256: str


@dataclass(frozen=True)
class ReferenceSpectrum:
    reference_key: str
    task_stratum: str
    excluded_subject_id: str
    frequencies_hz: np.ndarray
    normalized_psd_per_hz: np.ndarray
    status: str
    member_recording_ids: tuple[str, ...]
    member_subject_ids: tuple[str, ...]
    recording_count: int
    subject_count: int
    reference_sha256: str
    reference_vintage_sha256: str


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    normalized = np.asarray(array, dtype="<f8")
    return sha256(normalized.tobytes(order="C")).hexdigest()


def _renormalize_psd(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    parameters: QChanParameters,
) -> np.ndarray:
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    mask = (
        (frequencies_hz >= parameters.analysis_low_hz)
        & (frequencies_hz <= parameters.analysis_high_hz)
        & np.isfinite(psd)
        & (psd >= 0)
    )
    if mask.sum() < 2:
        raise ValueError("Insufficient finite PSD support in the analysis band.")
    total = float(np.trapezoid(psd[mask], frequencies_hz[mask]))
    if not np.isfinite(total) or total <= np.finfo(np.float64).tiny:
        raise ValueError("Analysis-band spectral power is zero or non-finite.")
    normalized = np.maximum(psd, 0.0) / total
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def _guard_and_merge_intervals(
    intervals: Sequence[TimeInterval],
    duration_sec: float,
    parameters: QChanParameters,
) -> list[TimeInterval]:
    guard = parameters.speech_boundary_guard_ms / 1000.0
    guarded: list[tuple[float, float]] = []
    for interval in intervals:
        start = max(0.0, float(interval.start_sec) + guard)
        end = min(float(duration_sec), float(interval.end_sec) - guard)
        if end > start:
            guarded.append((start, end))
    guarded.sort()
    merged: list[list[float]] = []
    for start, end in guarded:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [TimeInterval(start, end) for start, end in merged]


def _support_tier(support_sec: float, parameters: QChanParameters) -> str:
    if support_sec >= parameters.high_support_sec:
        return "high"
    if support_sec >= parameters.moderate_support_sec:
        return "moderate"
    if support_sec >= parameters.minimum_guarded_speech_sec:
        return "minimum"
    return "unavailable"


def extract_recording_spectrum(
    waveform: np.ndarray,
    sample_rate_hz: int,
    *,
    strict_speech: Sequence[TimeInterval],
    logical_recording_id: str,
    source_sample_rate_hz: int | None = None,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> RecordingSpectrum:
    """Estimate a gain-normalized LTAS without crossing segment boundaries.

    The caller supplies the DC-removed analysis waveform.  No pre-emphasis,
    equalization, peak normalization, or amplitude compression is performed.
    """

    y = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if sample_rate_hz != parameters.analysis_sample_rate_hz:
        raise ValueError(
            "QCHAN expects the frozen analysis-rate waveform; source rate is "
            "retained separately for the bandwidth audit."
        )
    if not np.isfinite(y).all():
        raise ValueError("Waveform contains NaN or infinite values.")
    source_rate = int(
        sample_rate_hz
        if source_sample_rate_hz is None
        else source_sample_rate_hz
    )
    if source_rate <= 0:
        raise ValueError("Source sample rate must be positive.")
    source_nyquist = source_rate / 2.0
    bandwidth_limited = source_nyquist < parameters.analysis_high_hz
    frequencies = np.fft.rfftfreq(
        parameters.n_fft, d=1.0 / sample_rate_hz
    ).astype(np.float64)
    empty = np.full_like(frequencies, np.nan, dtype=np.float64)

    guarded = _guard_and_merge_intervals(
        strict_speech, len(y) / sample_rate_hz, parameters
    )
    support_sec = float(sum(x.end_sec - x.start_sec for x in guarded))
    frame_n = int(round(parameters.frame_ms * sample_rate_hz / 1000.0))
    hop_n = int(round(parameters.hop_ms * sample_rate_hz / 1000.0))
    if frame_n < 2 or parameters.n_fft < frame_n:
        raise ValueError("Invalid frame or FFT geometry.")
    window = signal.windows.hann(frame_n, sym=False).astype(np.float64)
    window_energy = float(np.sum(window * window))
    accumulator = np.zeros(len(frequencies), dtype=np.float64)
    valid_frames = 0
    zero_frames = 0

    for interval in guarded:
        start = int(math.ceil(interval.start_sec * sample_rate_hz))
        end = int(math.floor(interval.end_sec * sample_rate_hz))
        if end - start < frame_n:
            continue
        for frame_start in range(start, end - frame_n + 1, hop_n):
            frame = y[frame_start : frame_start + frame_n].copy()
            frame -= float(np.mean(frame))
            rms = float(np.sqrt(np.mean(frame * frame)))
            if rms <= 1e-12:
                zero_frames += 1
                continue
            transformed = np.fft.rfft(frame * window, n=parameters.n_fft)
            periodogram = (
                np.abs(transformed) ** 2
                / (sample_rate_hz * window_energy)
            )
            if parameters.n_fft % 2 == 0:
                periodogram[1:-1] *= 2.0
            else:
                periodogram[1:] *= 2.0
            accumulator += periodogram
            valid_frames += 1

    if support_sec < parameters.minimum_guarded_speech_sec:
        status = "insufficient_strict_speech_support"
        normalized = empty
    elif valid_frames == 0:
        status = "digital_zero_speech"
        normalized = empty
    elif valid_frames < parameters.minimum_frames:
        status = "insufficient_valid_frame_support"
        normalized = empty
    else:
        normalized = _renormalize_psd(
            frequencies, accumulator / valid_frames, parameters
        )
        status = "measured"

    spectrum_hash = (
        _array_hash(normalized)
        if np.isfinite(normalized).all()
        else ""
    )
    return RecordingSpectrum(
        logical_recording_id=str(logical_recording_id),
        frequencies_hz=frequencies,
        normalized_psd_per_hz=normalized,
        status=status,
        support_tier=_support_tier(support_sec, parameters),
        guarded_speech_support_sec=support_sec,
        valid_frame_count=valid_frames,
        guarded_segment_count=len(guarded),
        zero_frame_count=zero_frames,
        source_sample_rate_hz=float(source_rate),
        source_nyquist_hz=float(source_nyquist),
        source_bandwidth_limited=bool(bandwidth_limited),
        spectrum_sha256=spectrum_hash,
    )


def third_octave_centers_hz(
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> np.ndarray:
    centers = []
    center = float(parameters.analysis_low_hz)
    ratio = 2.0 ** (1.0 / parameters.octave_fraction)
    while center <= parameters.analysis_high_hz * (1 + 1e-12):
        centers.append(center)
        center *= ratio
    return np.asarray(centers, dtype=np.float64)


def smoothed_log_ltas_db(
    frequencies_hz: np.ndarray,
    normalized_psd_per_hz: np.ndarray,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean PSD density in fixed one-third-octave bands, in dB.

    Density rather than integrated band power is used so unequal band widths
    do not create an artificial +3 dB/octave slope for a flat PSD.
    """

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    psd = np.asarray(normalized_psd_per_hz, dtype=np.float64)
    if frequencies.shape != psd.shape or not np.isfinite(psd).all():
        raise ValueError("A finite PSD on the expected frequency grid is required.")
    centers = third_octave_centers_hz(parameters)
    half_ratio = 2.0 ** (1.0 / (2.0 * parameters.octave_fraction))
    positive = psd[
        (frequencies >= parameters.analysis_low_hz)
        & (frequencies <= parameters.analysis_high_hz)
        & (psd > 0)
    ]
    if positive.size == 0:
        raise ValueError("The analysis-band PSD is zero.")
    floor = float(np.max(positive)) * 10.0 ** (
        parameters.relative_psd_floor_db / 10.0
    )
    levels = []
    for center in centers:
        low = max(parameters.analysis_low_hz, center / half_ratio)
        high = min(parameters.analysis_high_hz, center * half_ratio)
        mask = (frequencies >= low) & (frequencies <= high)
        if mask.sum() < 2:
            levels.append(np.nan)
            continue
        width = float(frequencies[mask][-1] - frequencies[mask][0])
        mean_density = (
            float(np.trapezoid(psd[mask], frequencies[mask])) / width
            if width > 0 else np.nan
        )
        levels.append(10.0 * np.log10(max(mean_density, floor)))
    levels_array = np.asarray(levels, dtype=np.float64)
    if not np.isfinite(levels_array).all():
        raise ValueError("One-third-octave LTAS contains non-finite bands.")
    return centers, levels_array


def integrated_band_power(
    frequencies_hz: np.ndarray,
    normalized_psd_per_hz: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    psd = np.asarray(normalized_psd_per_hz, dtype=np.float64)
    mask = (
        (frequencies >= low_hz)
        & (frequencies <= high_hz)
        & np.isfinite(psd)
        & (psd >= 0)
    )
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def spectral_rolloff_hz(
    frequencies_hz: np.ndarray,
    normalized_psd_per_hz: np.ndarray,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> float:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    psd = np.asarray(normalized_psd_per_hz, dtype=np.float64)
    mask = (
        (frequencies >= parameters.analysis_low_hz)
        & (frequencies <= parameters.analysis_high_hz)
        & np.isfinite(psd)
        & (psd >= 0)
    )
    f = frequencies[mask]
    p = psd[mask]
    if f.size < 2:
        return np.nan
    increments = 0.5 * (p[1:] + p[:-1]) * np.diff(f)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    if cumulative[-1] <= np.finfo(np.float64).tiny:
        return np.nan
    target = parameters.rolloff_fraction * cumulative[-1]
    return float(np.interp(target, cumulative, f))


def highband_ratio(
    frequencies_hz: np.ndarray,
    normalized_psd_per_hz: np.ndarray,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> float:
    total = integrated_band_power(
        frequencies_hz,
        normalized_psd_per_hz,
        parameters.analysis_low_hz,
        parameters.analysis_high_hz,
    )
    high = integrated_band_power(
        frequencies_hz,
        normalized_psd_per_hz,
        parameters.highband_low_hz,
        parameters.highband_high_hz,
    )
    return (
        float(high / total)
        if np.isfinite(total) and total > 0 and np.isfinite(high)
        else np.nan
    )


def spectral_tilt_db_per_octave(
    frequencies_hz: np.ndarray,
    normalized_psd_per_hz: np.ndarray,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> float:
    centers, levels = smoothed_log_ltas_db(
        frequencies_hz, normalized_psd_per_hz, parameters
    )
    mask = (
        (centers >= parameters.tilt_low_hz)
        & (centers <= parameters.tilt_high_hz)
        & np.isfinite(levels)
    )
    if mask.sum() < 6:
        return np.nan
    return float(
        stats.theilslopes(
            levels[mask], np.log2(centers[mask])
        ).slope
    )


def spectral_descriptors(
    frequencies_hz: np.ndarray,
    normalized_psd_per_hz: np.ndarray,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> dict[str, float]:
    return {
        "rolloff95_hz": spectral_rolloff_hz(
            frequencies_hz, normalized_psd_per_hz, parameters
        ),
        "highband_ratio": highband_ratio(
            frequencies_hz, normalized_psd_per_hz, parameters
        ),
        "tilt_db_per_oct": spectral_tilt_db_per_octave(
            frequencies_hz, normalized_psd_per_hz, parameters
        ),
    }


def reference_vintage_sha256(
    spectra: Mapping[str, RecordingSpectrum],
    metadata: pd.DataFrame,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> str:
    required = {
        "logical_recording_id", "subject_id", "task_stratum"
    }
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata lacks required columns: {sorted(required)}")
    if metadata.loc[:, sorted(required)].isna().any().any():
        raise ValueError("Reference metadata contains missing identity values.")
    normalized_metadata = metadata.loc[:, sorted(required)].copy()
    for column in required:
        normalized_metadata[column] = (
            normalized_metadata[column].astype("string").str.strip()
        )
    if normalized_metadata.loc[:, sorted(required)].eq("").any().any():
        raise ValueError("Reference metadata contains blank identity values.")

    rows = []
    for row in normalized_metadata.sort_values(
        ["task_stratum", "subject_id", "logical_recording_id"]
    ).itertuples(index=False):
        recording_id = str(row.logical_recording_id)
        spectrum = spectra.get(recording_id)
        if spectrum is None or spectrum.status != "measured":
            continue
        if (
            parameters.reference_requires_full_analysis_band
            and spectrum.source_nyquist_hz < parameters.analysis_high_hz
        ):
            continue
        rows.append({
            "logical_recording_id": recording_id,
            "subject_id": str(row.subject_id),
            "task_stratum": str(row.task_stratum),
            "spectrum_sha256": spectrum.spectrum_sha256,
        })
    return _stable_hash({
        "measurement_version": MEASUREMENT_VERSION,
        "parameters": parameters.to_dict(),
        "eligible_members": rows,
    })


def build_subject_balanced_loso_references(
    spectra: Mapping[str, RecordingSpectrum],
    metadata: pd.DataFrame,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> dict[str, ReferenceSpectrum]:
    """Build task-matched references, first within subject, then across subjects."""

    required = {
        "logical_recording_id", "subject_id", "task_stratum"
    }
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata lacks required columns: {sorted(required)}")
    meta = metadata.loc[:, sorted(required)].copy()
    if meta.isna().any().any():
        raise ValueError("Reference metadata contains missing identity values.")
    for column in required:
        meta[column] = meta[column].astype("string").str.strip()
    if meta.eq("").any().any():
        raise ValueError("Reference metadata contains blank identity values.")
    if meta["logical_recording_id"].duplicated().any():
        raise ValueError("Reference metadata contains duplicate recording IDs.")
    vintage = reference_vintage_sha256(spectra, meta, parameters)
    references: dict[str, ReferenceSpectrum] = {}
    cache: dict[tuple[str, str], ReferenceSpectrum] = {}

    for row in meta.itertuples(index=False):
        recording_id = str(row.logical_recording_id)
        task = str(row.task_stratum)
        subject = str(row.subject_id)
        cache_key = (task, subject)
        if cache_key not in cache:
            candidates = meta.loc[
                meta["task_stratum"].eq(task)
                & ~meta["subject_id"].eq(subject)
            ].copy()
            eligible_rows = []
            for candidate in candidates.itertuples(index=False):
                candidate_id = str(candidate.logical_recording_id)
                spectrum = spectra.get(candidate_id)
                if spectrum is None or spectrum.status != "measured":
                    continue
                if (
                    parameters.reference_requires_full_analysis_band
                    and spectrum.source_nyquist_hz
                    < parameters.analysis_high_hz
                ):
                    continue
                eligible_rows.append({
                    "logical_recording_id": candidate_id,
                    "subject_id": str(candidate.subject_id),
                    "spectrum": spectrum,
                })
            member_ids = tuple(
                sorted(item["logical_recording_id"] for item in eligible_rows)
            )
            member_subjects = tuple(
                sorted({item["subject_id"] for item in eligible_rows})
            )
            reference_key = _stable_hash({
                "task_stratum": task,
                "excluded_subject_id": subject,
                "reference_vintage_sha256": vintage,
            })[:20]
            if (
                len(member_ids) < parameters.minimum_reference_recordings
                or len(member_subjects)
                < parameters.minimum_reference_subjects
            ):
                frequencies = next(
                    (
                        spectrum.frequencies_hz
                        for spectrum in spectra.values()
                    ),
                    np.array([], dtype=np.float64),
                )
                reference = ReferenceSpectrum(
                    reference_key=reference_key,
                    task_stratum=task,
                    excluded_subject_id=subject,
                    frequencies_hz=np.asarray(frequencies, dtype=np.float64),
                    normalized_psd_per_hz=np.full(
                        len(frequencies), np.nan, dtype=np.float64
                    ),
                    status="reference_unavailable",
                    member_recording_ids=member_ids,
                    member_subject_ids=member_subjects,
                    recording_count=len(member_ids),
                    subject_count=len(member_subjects),
                    reference_sha256="",
                    reference_vintage_sha256=vintage,
                )
            else:
                subject_spectra = []
                frequencies = None
                for reference_subject in member_subjects:
                    local = [
                        item["spectrum"]
                        for item in eligible_rows
                        if item["subject_id"] == reference_subject
                    ]
                    if frequencies is None:
                        frequencies = local[0].frequencies_hz
                    if any(
                        not np.array_equal(
                            frequencies, spectrum.frequencies_hz
                        )
                        for spectrum in local
                    ):
                        raise ValueError(
                            "Reference spectra do not share a frequency grid."
                        )
                    subject_psd = np.median(
                        np.vstack([
                            spectrum.normalized_psd_per_hz
                            for spectrum in local
                        ]),
                        axis=0,
                    )
                    subject_spectra.append(
                        _renormalize_psd(
                            frequencies, subject_psd, parameters
                        )
                    )
                reference_psd = _renormalize_psd(
                    frequencies,
                    np.median(np.vstack(subject_spectra), axis=0),
                    parameters,
                )
                reference_hash = _stable_hash({
                    "reference_key": reference_key,
                    "member_recording_ids": member_ids,
                    "member_subject_ids": member_subjects,
                    "psd_sha256": _array_hash(reference_psd),
                    "parameters": parameters.to_dict(),
                })
                reference = ReferenceSpectrum(
                    reference_key=reference_key,
                    task_stratum=task,
                    excluded_subject_id=subject,
                    frequencies_hz=np.asarray(frequencies, dtype=np.float64),
                    normalized_psd_per_hz=reference_psd,
                    status="measured",
                    member_recording_ids=member_ids,
                    member_subject_ids=member_subjects,
                    recording_count=len(member_ids),
                    subject_count=len(member_subjects),
                    reference_sha256=reference_hash,
                    reference_vintage_sha256=vintage,
                )
            cache[cache_key] = reference
        references[recording_id] = cache[cache_key]
    return references


def compute_reference_relative_features(
    observation: RecordingSpectrum,
    reference: ReferenceSpectrum,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> dict[str, object]:
    base: dict[str, object] = {
        "logical_recording_id": observation.logical_recording_id,
        "qchan_measurement_version": MEASUREMENT_VERSION,
        "qchan_reference_key": reference.reference_key,
        "qchan_reference_sha256": reference.reference_sha256,
        "qchan_reference_vintage_sha256": (
            reference.reference_vintage_sha256
        ),
        "qchan_reference_recording_count": reference.recording_count,
        "qchan_reference_subject_count": reference.subject_count,
        "qchan_guarded_speech_support_sec": (
            observation.guarded_speech_support_sec
        ),
        "qchan_valid_frame_count": observation.valid_frame_count,
        "qchan_guarded_segment_count": observation.guarded_segment_count,
        "qchan_zero_frame_count": observation.zero_frame_count,
        "qchan_source_sample_rate_hz": observation.source_sample_rate_hz,
        "qchan_source_nyquist_hz": observation.source_nyquist_hz,
        "qchan_source_bandwidth_limited": (
            observation.source_bandwidth_limited
        ),
        "qchan_support_tier": observation.support_tier,
    }
    unavailable_status = (
        observation.status
        if observation.status != "measured"
        else reference.status
    )
    if observation.status != "measured" or reference.status != "measured":
        for feature in ANALYSIS_FEATURES:
            base[feature] = np.nan
            base[f"{feature}_status"] = unavailable_status
            base[f"{feature}_support_tier"] = "unavailable"
        base["qchan_primary_available_count"] = 0
        base["qchan_primary_analysis_eligible"] = False
        base["qchan_family_status"] = unavailable_status
        return base
    if not np.array_equal(
        observation.frequencies_hz, reference.frequencies_hz
    ):
        raise ValueError("Observation and reference frequency grids differ.")

    obs_centers, obs_log = smoothed_log_ltas_db(
        observation.frequencies_hz,
        observation.normalized_psd_per_hz,
        parameters,
    )
    ref_centers, ref_log = smoothed_log_ltas_db(
        reference.frequencies_hz,
        reference.normalized_psd_per_hz,
        parameters,
    )
    if not np.array_equal(obs_centers, ref_centers):
        raise ValueError("Observation and reference LTAS bands differ.")
    obs_desc = spectral_descriptors(
        observation.frequencies_hz,
        observation.normalized_psd_per_hz,
        parameters,
    )
    ref_desc = spectral_descriptors(
        reference.frequencies_hz,
        reference.normalized_psd_per_hz,
        parameters,
    )
    signed_differences = {
        "qchan_rolloff95_signed_difference_hz": (
            ref_desc["rolloff95_hz"] - obs_desc["rolloff95_hz"]
        ),
        "qchan_highband_ratio_signed_difference": (
            ref_desc["highband_ratio"] - obs_desc["highband_ratio"]
        ),
        "qchan_tilt_signed_difference_db_per_oct": (
            ref_desc["tilt_db_per_oct"] - obs_desc["tilt_db_per_oct"]
        ),
    }
    ltas_distance = float(np.sqrt(np.mean((obs_log - ref_log) ** 2)))
    descriptor_values = [ltas_distance, *signed_differences.values()]
    if not np.isfinite(descriptor_values).all():
        for feature in ANALYSIS_FEATURES:
            base[feature] = np.nan
            base[f"{feature}_status"] = "descriptor_nonfinite"
            base[f"{feature}_support_tier"] = "unavailable"
        base.update(signed_differences)
        base["qchan_primary_available_count"] = 0
        base["qchan_primary_analysis_eligible"] = False
        base["qchan_family_status"] = "descriptor_nonfinite"
        return base

    values = {
        "qchan_ltas_distance_db": ltas_distance,
        "qchan_rolloff95_deficit_hz": float(max(
            0.0, signed_differences["qchan_rolloff95_signed_difference_hz"]
        )),
        "qchan_highband_ratio_deficit": float(max(
            0.0, signed_differences["qchan_highband_ratio_signed_difference"]
        )),
        "qchan_tilt_steepening_db_per_oct": float(max(
            0.0, signed_differences["qchan_tilt_signed_difference_db_per_oct"]
        )),
    }
    base.update(values)
    base.update(signed_differences)
    base.update({
        "qchan_observed_rolloff95_hz": obs_desc["rolloff95_hz"],
        "qchan_reference_rolloff95_hz": ref_desc["rolloff95_hz"],
        "qchan_observed_highband_ratio": obs_desc["highband_ratio"],
        "qchan_reference_highband_ratio": ref_desc["highband_ratio"],
        "qchan_observed_tilt_db_per_oct": obs_desc["tilt_db_per_oct"],
        "qchan_reference_tilt_db_per_oct": ref_desc["tilt_db_per_oct"],
    })
    for feature in ANALYSIS_FEATURES:
        base[f"{feature}_status"] = "measured"
        base[f"{feature}_support_tier"] = observation.support_tier
    base["qchan_primary_available_count"] = len(PRIMARY_FEATURES)
    base["qchan_primary_analysis_eligible"] = True
    base["qchan_family_status"] = "measured"
    return base


def feature_registry_frame() -> pd.DataFrame:
    rows = []
    for name in ANALYSIS_FEATURES:
        rows.append({"name": name, **FEATURE_DEFINITIONS[name]})
    return pd.DataFrame(rows)


def apply_gain_db(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    return np.asarray(waveform, dtype=np.float64) * 10.0 ** (
        float(gain_db) / 20.0
    )


def lowpass_filter(
    waveform: np.ndarray,
    sample_rate_hz: int,
    cutoff_hz: float,
    order: int = 8,
) -> np.ndarray:
    if cutoff_hz >= 0.98 * sample_rate_hz / 2:
        return np.asarray(waveform, dtype=np.float64).copy()
    sos = signal.butter(
        order,
        cutoff_hz / (sample_rate_hz / 2),
        btype="low",
        output="sos",
    )
    return signal.sosfiltfilt(
        sos, np.asarray(waveform, dtype=np.float64)
    )


def smooth_high_shelf(
    waveform: np.ndarray,
    sample_rate_hz: int,
    gain_db: float,
    transition_hz: float = 2500.0,
) -> np.ndarray:
    y = np.asarray(waveform, dtype=np.float64)
    frequencies = np.fft.rfftfreq(len(y), 1.0 / sample_rate_hz)
    log_position = np.clip(
        (
            np.log2(np.maximum(frequencies, 1.0) / transition_hz)
            + 1.0
        )
        / 2.0,
        0.0,
        1.0,
    )
    blend = log_position * log_position * (3.0 - 2.0 * log_position)
    gain = 10.0 ** (float(gain_db) * blend / 20.0)
    return np.fft.irfft(np.fft.rfft(y) * gain, n=len(y))


def broad_notch_filter(
    waveform: np.ndarray,
    sample_rate_hz: int,
    center_hz: float = 1500.0,
    width_octaves: float = 0.75,
    depth_db: float = -18.0,
) -> np.ndarray:
    y = np.asarray(waveform, dtype=np.float64)
    frequencies = np.fft.rfftfreq(len(y), 1.0 / sample_rate_hz)
    distance = np.abs(
        np.log2(np.maximum(frequencies, 1.0) / center_hz)
    )
    weight = np.clip(
        1.0 - distance / (width_octaves / 2.0), 0.0, 1.0
    )
    weight = weight * weight * (3.0 - 2.0 * weight)
    gain = 10.0 ** (float(depth_db) * weight / 20.0)
    return np.fft.irfft(np.fft.rfft(y) * gain, n=len(y))


def synthetic_speech_like(
    duration_sec: float = 12.0,
    sample_rate_hz: int = 16_000,
    seed: int = 20260730,
) -> np.ndarray:
    """Deterministic broadband speech-like validation signal.

    This is a signal-processing fixture, not a physiological speech model.
    """

    rng = np.random.default_rng(seed)
    n = int(round(duration_sec * sample_rate_hz))
    time = np.arange(n, dtype=np.float64) / sample_rate_hz
    excitation = rng.normal(size=n)
    sos = signal.butter(
        4,
        [80.0 / (sample_rate_hz / 2), 7600.0 / (sample_rate_hz / 2)],
        btype="band",
        output="sos",
    )
    carrier = signal.sosfilt(sos, excitation)
    envelope = 0.25 + 0.75 * (
        0.5 + 0.5 * np.sin(2 * np.pi * 2.4 * time)
    ) ** 2
    high = signal.sosfilt(
        signal.butter(
            3,
            [3000.0 / (sample_rate_hz / 2),
             7600.0 / (sample_rate_hz / 2)],
            btype="band",
            output="sos",
        ),
        rng.normal(size=n),
    )
    frication = np.zeros(n, dtype=np.float64)
    for center in np.arange(0.6, duration_sec, 0.85):
        mask = np.abs(time - center) < 0.055
        frication[mask] = high[mask]
    y = carrier * envelope + 0.22 * frication
    peak = float(np.max(np.abs(y)))
    return 0.30 * y / max(peak, np.finfo(np.float64).tiny)


def full_span_interval(waveform: np.ndarray, sample_rate_hz: int) -> list[TimeInterval]:
    guard_sec = DEFAULT_PARAMETERS.speech_boundary_guard_ms / 1000.0
    return [
        TimeInterval(
            -guard_sec,
            len(np.asarray(waveform)) / sample_rate_hz + guard_sec,
        )
    ]
