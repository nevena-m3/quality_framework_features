"""QADD v4.2 reviewed measurement functions for extrinsic additive acoustic interference.

The functions in this module are the authoritative implementation used by the
QADD validation notebook.  They deliberately return a *vector* of estimands;
there is no scalar family score.

All level measurements are made on the repository's mono, DC-removed, 16-kHz
analysis view.  Frame levels are AC-RMS values relative to decoded full scale.
They are therefore analysis-view dBFS measurements, not native-stream dBFS,
sound-pressure level, or a calibrated physical SNR.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy import signal

MEASUREMENT_VERSION = "qadd-v4.2.0-candidate"


@dataclass(frozen=True, order=True)
class TimeInterval:
    """A half-open time interval in seconds."""

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


@dataclass(frozen=True)
class QADDParameters:
    """Frozen signal-processing and minimum-support parameters for QADD v4.2.

    Support classes describe usable signal quantity and interval diversity
    only. They do not assert empirical robustness to deletion of a pause.
    """

    measurement_version: str = MEASUREMENT_VERSION
    analysis_sample_rate_hz: int = 16_000
    frame_ms: float = 30.0
    hop_ms: float = 10.0
    dbfs_floor_db: float = -120.0
    pause_guard_ms: float = 200.0
    minimum_residual_pause_ms: float = 100.0
    speech_guard_ms: float = 50.0

    # Conservative operating ceiling selected below the notebook-calibrated
    # maximum (5%). At 2%, the P90 absolute biases are approximately 0.23 dB
    # for pause median/contrast and 0.40 dB for IQR across 0.5-6 s supports.
    maximum_floor_censored_fraction: float = 0.02

    level_min_pause_sec: float = 0.30
    level_min_frames: int = 20
    level_moderate_pause_sec: float = 1.50
    level_high_support_pause_sec: float = 3.00
    level_moderate_intervals: int = 2
    level_high_support_intervals: int = 3

    dispersion_min_pause_sec: float = 0.50
    dispersion_min_frames: int = 40
    dispersion_moderate_pause_sec: float = 1.50
    dispersion_high_support_pause_sec: float = 3.00
    dispersion_moderate_intervals: int = 2
    dispersion_high_support_intervals: int = 3

    contrast_min_speech_sec: float = 1.00
    contrast_min_speech_frames: int = 50
    contrast_moderate_speech_sec: float = 2.00
    contrast_high_support_speech_sec: float = 5.00

    flatness_window_ms: float = 250.0
    flatness_hop_ms: float = 125.0
    flatness_low_hz: float = 80.0
    # The upper 500 Hz below the 8-kHz analysis Nyquist is excluded because a
    # prespecified MP3/Opus round-trip control showed a large codec-dependent
    # flatness shift at 7.5 kHz and negligible shift at 7.0 kHz.  This keeps a
    # channel-codec edge effect out of the additive-interference descriptor.
    flatness_high_hz: float = 7000.0
    flatness_min_windows: int = 3
    flatness_moderate_windows: int = 6
    flatness_high_support_windows: int = 12

    hum_window_ms: float = 500.0
    hum_hop_ms: float = 250.0
    hum_max_hz: float = 1000.0
    hum_harmonic_count: int = 4
    # A 500-ms window has 2-Hz DFT-bin spacing at 16 kHz.  A +/-2-Hz tone
    # region therefore captures the nominal bin and one-bin frequency drift
    # without overlapping the sidebands that begin 6 Hz from the harmonic.
    hum_tone_half_width_hz: float = 2.0
    hum_sideband_inner_hz: float = 6.0
    hum_sideband_outer_hz: float = 14.0
    hum_supported_harmonic_db: float = 6.0
    hum_min_supported_harmonics: int = 3
    hum_min_windows: int = 2
    hum_moderate_windows: int = 4
    hum_high_support_windows: int = 8

    random_seed: int = 20260729

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMETERS = QADDParameters()


FEATURE_DEFINITIONS = (
    {
        "feature": "qadd_pause_ac_level_dbfs_median",
        "display_name": "Guarded-pause AC level (median)",
        "role": "primary",
        "unit": "analysis-view dBFS",
        "direction": "higher means more recorded non-floor pause energy",
        "estimator": "median 30-ms AC-RMS frame level; 10-ms hop",
        "construct": "recorded pause energy consistent with extrinsic additive interference",
        "establishment": "RMS/dBFS are established; the guarded-pause application is study-specific",
        "claim_limit": "not dB SPL, native-stream dBFS, environmental exposure, or physical SNR",
    },
    {
        "feature": "qadd_pause_level_iqr_db",
        "display_name": "Guarded-pause level IQR",
        "role": "secondary",
        "unit": "dB",
        "direction": "contextual; higher means greater within-recording variation",
        "estimator": "Q75 minus Q25 of eligible pause-frame AC levels",
        "construct": "temporal heterogeneity of recorded pause energy",
        "establishment": "IQR is established robust dispersion; this application is study-specific",
        "claim_limit": "not an ordinal severity measure and sensitive to noise nonstationarity",
    },
    {
        "feature": "qadd_speech_pause_level_contrast_db",
        "display_name": "Speech-pause level contrast",
        "role": "mixed_secondary",
        "unit": "dB",
        "direction": "lower means less within-recording level separation",
        "estimator": "median strict-speech AC level minus median guarded-pause AC level",
        "construct": "within-recording speech-to-pause level separation",
        "establishment": "level contrast is established; interpretation as a QC marker is study-specific",
        "claim_limit": "not physical SNR; also depends on speech intensity, distance, gain, and physiology",
    },
    {
        "feature": "qadd_pause_spectral_flatness",
        "display_name": "Guarded-pause spectral flatness",
        "role": "secondary_descriptor",
        "unit": "ratio",
        "direction": "non-ordinal; high is broadband-like and low is tonal/structured",
        "estimator": "median power-spectrum flatness from 80-7000 Hz in 250-ms windows",
        "construct": "spectral type of recorded pause interference",
        "establishment": "spectral flatness is established; the band/aggregation are prespecified here",
        "claim_limit": "neither endpoint is universally worse; channel bandwidth is a confounder",
    },
    {
        "feature": "qadd_mains_hum_comb_score_db",
        "display_name": "50/60-Hz mains-hum comb score",
        "role": "targeted_descriptor",
        "unit": "dB",
        "direction": "higher means stronger 50/60-Hz harmonic structure",
        "estimator": "max 50/60-Hz median local contrast across harmonics 1-4",
        "construct": "mains-frequency harmonic interference in recorded pauses",
        "establishment": "harmonic prominence is established; this exact robust comb is study-specific",
        "claim_limit": "not a perceptual threshold; requires off-grid and colored-noise false-positive checks",
    },
)


PRIMARY_FEATURES = ("qadd_pause_ac_level_dbfs_median",)
SECONDARY_FEATURES = (
    "qadd_pause_level_iqr_db",
    "qadd_speech_pause_level_contrast_db",
    "qadd_pause_spectral_flatness",
)
TARGETED_FEATURES = ("qadd_mains_hum_comb_score_db",)
ANALYSIS_FEATURES = PRIMARY_FEATURES + SECONDARY_FEATURES + TARGETED_FEATURES


@dataclass
class QADDExtraction:
    """Recording-level output plus reconstructable audit ledgers."""

    recording: dict
    frame_ledger: pd.DataFrame
    interval_ledger: pd.DataFrame
    spectral_ledger: pd.DataFrame


def feature_registry_frame() -> pd.DataFrame:
    """Return the versioned QADD feature contract."""

    frame = pd.DataFrame(FEATURE_DEFINITIONS)
    if frame["feature"].duplicated().any():
        raise ValueError("QADD feature definitions contain duplicate names")
    contract = {
        "qadd_pause_ac_level_dbfs_median": {
            "signal_region": "guarded internal strict-nonspeech",
            "support_field": "qadd_pause_level_support_tier",
            "status_field": "qadd_pause_ac_level_dbfs_median_status",
            "mathematical_range": "[-120, 0] dBFS for unclipped normalized PCM analysis view",
            "positive_control": "unit-slope response to injected pause-noise level",
            "discriminant_control": "independent of noise spectral color at matched AC-RMS",
            "known_confounding": "device gain, distance, room noise, channel processing, residual biological sound",
        },
        "qadd_pause_level_iqr_db": {
            "signal_region": "guarded internal strict-nonspeech",
            "support_field": "qadd_pause_dispersion_support_tier",
            "status_field": "qadd_pause_level_iqr_db_status",
            "mathematical_range": "[0, +infinity) dB",
            "positive_control": "increases under amplitude-modulated or changing pause noise",
            "discriminant_control": "invariant to global gain",
            "known_confounding": "number/distribution of pauses, transient residuals, automatic gain control",
        },
        "qadd_speech_pause_level_contrast_db": {
            "signal_region": "strict speech and guarded internal strict-nonspeech",
            "support_field": "qadd_speech_pause_contrast_support_tier",
            "status_field": "qadd_speech_pause_level_contrast_db_status",
            "mathematical_range": "unbounded difference in dB",
            "positive_control": "decreases as injected pause-noise level rises",
            "discriminant_control": "invariant to global gain",
            "known_confounding": "speech intensity, bulbar impairment, microphone distance, gain control",
        },
        "qadd_pause_spectral_flatness": {
            "signal_region": "non-floor guarded internal strict-nonspeech windows",
            "support_field": "qadd_flatness_support_tier",
            "status_field": "qadd_pause_spectral_flatness_status",
            "mathematical_range": "(0, 1]",
            "positive_control": "higher for broadband noise than tonal interference",
            "discriminant_control": "invariant to global gain; low for off-grid tones",
            "known_confounding": "channel bandwidth, codec filtering, spectral coloration",
        },
        "qadd_mains_hum_comb_score_db": {
            "signal_region": "non-floor guarded internal strict-nonspeech windows",
            "support_field": "qadd_hum_support_tier",
            "status_field": "qadd_mains_hum_comb_score_db_status",
            "mathematical_range": "unbounded local power contrast in dB",
            "positive_control": "increases for 50/60-Hz harmonic combs",
            "discriminant_control": "colored-noise and 53-Hz off-grid nulls",
            "known_confounding": "frequency drift, narrowband machinery, spectral leakage, support count",
        },
    }
    contract_frame = pd.DataFrame.from_dict(contract, orient="index")
    frame = frame.join(contract_frame, on="feature", validate="one_to_one")
    frame.insert(0, "measurement_version", MEASUREMENT_VERSION)
    frame["signal_view"] = "mono, DC-removed, resampled 16-kHz analysis view"
    frame["missing_value_behavior"] = (
        "analysis value is NaN unless feature status begins with ok_; raw estimate retained"
    )
    frame["analysis_eligible"] = True
    frame["composite_use_prohibited"] = True
    return frame


def _as_interval(item) -> TimeInterval:
    if isinstance(item, TimeInterval):
        return item
    if hasattr(item, "start_sec") and hasattr(item, "end_sec"):
        return TimeInterval(float(item.start_sec), float(item.end_sec))
    if isinstance(item, Sequence) and len(item) == 2:
        return TimeInterval(float(item[0]), float(item[1]))
    raise TypeError(f"Cannot interpret {type(item).__name__} as a time interval")


def normalize_intervals(
    intervals: Iterable[TimeInterval], duration_sec: float
) -> list[TimeInterval]:
    """Clip, sort, and merge overlapping/touching intervals."""

    duration_sec = max(0.0, float(duration_sec))
    clipped = []
    for raw in intervals:
        item = _as_interval(raw)
        candidate = TimeInterval(
            max(0.0, float(item.start_sec)),
            min(duration_sec, float(item.end_sec)),
        )
        if candidate.duration_sec > 0:
            clipped.append(candidate)
    clipped.sort(key=lambda item: (item.start_sec, item.end_sec))
    merged: list[TimeInterval] = []
    for item in clipped:
        if merged and item.start_sec <= merged[-1].end_sec + 1e-12:
            merged[-1] = TimeInterval(
                merged[-1].start_sec, max(merged[-1].end_sec, item.end_sec)
            )
        else:
            merged.append(item)
    return merged


def intersect_intervals(
    left: Iterable[TimeInterval], right: Iterable[TimeInterval]
) -> list[TimeInterval]:
    """Return pairwise intersections of two normalized interval sets."""

    a = sorted((_as_interval(item) for item in left), key=lambda item: item.start_sec)
    b = sorted((_as_interval(item) for item in right), key=lambda item: item.start_sec)
    result: list[TimeInterval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i].start_sec, b[j].start_sec)
        end = min(a[i].end_sec, b[j].end_sec)
        if end > start:
            result.append(TimeInterval(start, end))
        if a[i].end_sec <= b[j].end_sec:
            i += 1
        else:
            j += 1
    return result


def interval_union_duration(intervals: Iterable[TimeInterval]) -> float:
    """Duration of an interval union without counting overlap twice."""

    items = [_as_interval(item) for item in intervals]
    if not items:
        return 0.0
    duration = max(item.end_sec for item in items)
    return float(sum(item.duration_sec for item in normalize_intervals(items, duration)))


def internal_pauses(
    primary_speech: Iterable[TimeInterval], duration_sec: float
) -> list[TimeInterval]:
    """Internal gaps between consecutive primary-speech intervals."""

    speech = normalize_intervals(primary_speech, duration_sec)
    return [
        TimeInterval(left.end_sec, right.start_sec)
        for left, right in pairwise(speech)
        if right.start_sec > left.end_sec
    ]


def guarded_internal_pauses(
    primary_speech: Iterable[TimeInterval],
    duration_sec: float,
    *,
    strict_nonspeech: Iterable[TimeInterval] | None = None,
    guard_ms: float | None = None,
    minimum_residual_ms: float | None = None,
    parameters: QADDParameters = DEFAULT_PARAMETERS,
) -> list[TimeInterval]:
    """Derive guarded internal pauses and optionally intersect a frozen strict view."""

    guard_sec = (
        parameters.pause_guard_ms if guard_ms is None else float(guard_ms)
    ) / 1000.0
    minimum_sec = (
        parameters.minimum_residual_pause_ms
        if minimum_residual_ms is None
        else float(minimum_residual_ms)
    ) / 1000.0
    pauses = []
    for gap in internal_pauses(primary_speech, duration_sec):
        candidate = TimeInterval(gap.start_sec + guard_sec, gap.end_sec - guard_sec)
        if candidate.duration_sec >= minimum_sec:
            pauses.append(candidate)
    if strict_nonspeech is not None:
        strict = normalize_intervals(strict_nonspeech, duration_sec)
        pauses = intersect_intervals(pauses, strict)
        pauses = [item for item in pauses if item.duration_sec >= minimum_sec]
    return pauses


def erode_intervals(
    intervals: Iterable[TimeInterval],
    duration_sec: float,
    *,
    guard_ms: float,
    minimum_ms: float,
) -> list[TimeInterval]:
    """Erode both boundaries of every interval without concatenating intervals."""

    guard_sec = float(guard_ms) / 1000.0
    minimum_sec = float(minimum_ms) / 1000.0
    result = []
    for item in normalize_intervals(intervals, duration_sec):
        candidate = TimeInterval(item.start_sec + guard_sec, item.end_sec - guard_sec)
        if candidate.duration_sec >= minimum_sec:
            result.append(candidate)
    return result


def ac_rms_measurement(
    samples: np.ndarray, *, floor_db: float = -120.0
) -> tuple[float, float, bool, bool]:
    """Return stored dBFS, linear AC-RMS, floor flag, and exact-zero flag."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("AC-RMS expects a one-dimensional waveform")
    if values.size == 0:
        return np.nan, np.nan, False, False
    if not np.isfinite(values).all():
        raise ValueError("AC-RMS input contains NaN or infinity")
    centered = values - float(np.mean(values, dtype=np.float64))
    rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
    floor_amplitude = float(10.0 ** (float(floor_db) / 20.0))
    at_floor = bool(rms <= floor_amplitude * (1.0 + 1e-12))
    stored = float(20.0 * np.log10(max(rms, floor_amplitude)))
    return max(stored, float(floor_db)), rms, at_floor, bool(np.all(values == 0.0))


def _frame_table(
    waveform: np.ndarray,
    sample_rate: int,
    intervals: list[TimeInterval],
    *,
    region: str,
    logical_recording_id: str,
    parameters: QADDParameters,
) -> pd.DataFrame:
    frame_n = max(1, round(parameters.frame_ms * sample_rate / 1000.0))
    hop_n = max(1, round(parameters.hop_ms * sample_rate / 1000.0))
    rows = []
    for interval_index, interval in enumerate(intervals):
        start_sample = max(0, round(interval.start_sec * sample_rate))
        end_sample = min(len(waveform), round(interval.end_sec * sample_rate))
        for frame_index, start in enumerate(
            range(start_sample, end_sample - frame_n + 1, hop_n)
        ):
            level, rms, at_floor, exact_zero = ac_rms_measurement(
                waveform[start : start + frame_n], floor_db=parameters.dbfs_floor_db
            )
            rows.append(
                {
                    "logical_recording_id": str(logical_recording_id),
                    "region": region,
                    "interval_index": interval_index,
                    "frame_index_in_interval": frame_index,
                    "frame_start_sec": start / sample_rate,
                    "frame_end_sec": (start + frame_n) / sample_rate,
                    "rms_dbfs": level,
                    "rms_linear": rms,
                    "at_computational_floor": at_floor,
                    "exact_zero_frame": exact_zero,
                }
            )
    return pd.DataFrame(rows)


def _effective_frame_support_sec(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    intervals = [
        TimeInterval(float(row.frame_start_sec), float(row.frame_end_sec))
        for row in frame.itertuples(index=False)
    ]
    return interval_union_duration(intervals)


def power_spectrum(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the Hann-windowed one-sided power spectral density used by QADD."""

    values = np.asarray(samples, dtype=np.float64)
    values = values - float(np.mean(values))
    window = signal.windows.hann(values.size, sym=False)
    scale = float(sample_rate * np.sum(window * window))
    spectrum = np.fft.rfft(values * window)
    psd = np.abs(spectrum) ** 2 / max(scale, np.finfo(float).tiny)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sample_rate)
    return frequencies, psd


def spectral_flatness_from_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    *,
    low_hz: float,
    high_hz: float,
) -> float:
    """Power-spectrum flatness (geometric mean divided by arithmetic mean)."""

    band = (
        np.isfinite(frequencies)
        & np.isfinite(psd)
        & (frequencies >= float(low_hz))
        & (frequencies <= float(high_hz))
    )
    power = np.asarray(psd[band], dtype=np.float64)
    if power.size < 3 or np.all(power <= 0):
        return np.nan
    tiny = max(np.finfo(float).tiny, float(np.max(power)) * 1e-15)
    return float(np.exp(np.mean(np.log(power + tiny))) / np.mean(power + tiny))


def hum_comb_score_from_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    fundamental_hz: float,
    *,
    parameters: QADDParameters = DEFAULT_PARAMETERS,
) -> tuple[float, int, int]:
    """Return a robust harmonic-comb score and harmonic support counts.

    Each of the first four harmonics is compared with frequency-local sidebands.
    The window score is their median. Requiring consistent low-order harmonic
    evidence prevents an off-grid comb from passing through accidental alignment
    with a few high-order 50/60-Hz harmonics. The recording-level maximum over
    50 and 60 Hz still requires count-matched colored-noise validation.
    """

    contrasts = []
    centers = float(fundamental_hz) * np.arange(
        1, parameters.hum_harmonic_count + 1, dtype=float
    )
    for center in centers[centers <= parameters.hum_max_hz]:
        tone = np.abs(frequencies - center) <= parameters.hum_tone_half_width_hz
        sidebands = (
            (
                (frequencies >= center - parameters.hum_sideband_outer_hz)
                & (frequencies <= center - parameters.hum_sideband_inner_hz)
            )
            | (
                (frequencies >= center + parameters.hum_sideband_inner_hz)
                & (frequencies <= center + parameters.hum_sideband_outer_hz)
            )
        )
        if not tone.any() or not sidebands.any():
            continue
        tone_power = float(np.mean(psd[tone]))
        reference_power = float(np.median(psd[sidebands]))
        tiny = max(np.finfo(float).tiny, reference_power * 1e-12)
        contrasts.append(10.0 * np.log10((tone_power + tiny) / (reference_power + tiny)))
    if not contrasts:
        return np.nan, 0, 0
    contrasts_array = np.asarray(contrasts, dtype=np.float64)
    supported = int(
        np.sum(contrasts_array >= parameters.hum_supported_harmonic_db)
    )
    return (
        float(np.median(contrasts_array)),
        supported,
        int(contrasts_array.size),
    )


def _spectral_window_table(
    waveform: np.ndarray,
    sample_rate: int,
    pauses: list[TimeInterval],
    *,
    logical_recording_id: str,
    parameters: QADDParameters,
) -> pd.DataFrame:
    rows = []
    configurations = (
        ("flatness", parameters.flatness_window_ms, parameters.flatness_hop_ms),
        ("hum", parameters.hum_window_ms, parameters.hum_hop_ms),
    )
    for kind, window_ms, hop_ms in configurations:
        window_n = max(1, round(window_ms * sample_rate / 1000.0))
        hop_n = max(1, round(hop_ms * sample_rate / 1000.0))
        for interval_index, interval in enumerate(pauses):
            start_sample = max(0, round(interval.start_sec * sample_rate))
            end_sample = min(len(waveform), round(interval.end_sec * sample_rate))
            for window_index, start in enumerate(
                range(start_sample, end_sample - window_n + 1, hop_n)
            ):
                samples = waveform[start : start + window_n]
                level, _, at_floor, exact_zero = ac_rms_measurement(
                    samples, floor_db=parameters.dbfs_floor_db
                )
                row = {
                    "logical_recording_id": str(logical_recording_id),
                    "window_kind": kind,
                    "interval_index": interval_index,
                    "window_index_in_interval": window_index,
                    "window_start_sec": start / sample_rate,
                    "window_end_sec": (start + window_n) / sample_rate,
                    "window_ac_level_dbfs": level,
                    "at_computational_floor": at_floor,
                    "exact_zero_window": exact_zero,
                    "valid_acoustic_window": not at_floor,
                    "spectral_flatness": np.nan,
                    "hum_score_50_db": np.nan,
                    "hum_score_60_db": np.nan,
                    "hum_score_max_db": np.nan,
                    "hum_winner_hz": np.nan,
                    "hum_supported_harmonic_count": np.nan,
                    "hum_evaluated_harmonic_count": np.nan,
                    "hum_supported_harmonic_count_50": np.nan,
                    "hum_evaluated_harmonic_count_50": np.nan,
                    "hum_supported_harmonic_count_60": np.nan,
                    "hum_evaluated_harmonic_count_60": np.nan,
                }
                if not at_floor:
                    frequencies, psd = power_spectrum(samples, sample_rate)
                    if kind == "flatness":
                        row["spectral_flatness"] = spectral_flatness_from_psd(
                            frequencies,
                            psd,
                            low_hz=parameters.flatness_low_hz,
                            high_hz=min(
                                parameters.flatness_high_hz, sample_rate / 2.0
                            ),
                        )
                    else:
                        score_50, supported_50, evaluated_50 = hum_comb_score_from_psd(
                            frequencies, psd, 50.0, parameters=parameters
                        )
                        score_60, supported_60, evaluated_60 = hum_comb_score_from_psd(
                            frequencies, psd, 60.0, parameters=parameters
                        )
                        if np.isfinite(score_50) or np.isfinite(score_60):
                            if np.nan_to_num(score_50, nan=-np.inf) >= np.nan_to_num(
                                score_60, nan=-np.inf
                            ):
                                winner = 50.0
                                supported = supported_50
                                evaluated = evaluated_50
                            else:
                                winner = 60.0
                                supported = supported_60
                                evaluated = evaluated_60
                            row.update(
                                {
                                    "hum_score_50_db": score_50,
                                    "hum_score_60_db": score_60,
                                    "hum_score_max_db": max(score_50, score_60),
                                    "hum_winner_hz": winner,
                                    "hum_supported_harmonic_count": supported,
                                    "hum_evaluated_harmonic_count": evaluated,
                                    "hum_supported_harmonic_count_50": supported_50,
                                    "hum_evaluated_harmonic_count_50": evaluated_50,
                                    "hum_supported_harmonic_count_60": supported_60,
                                    "hum_evaluated_harmonic_count_60": evaluated_60,
                                }
                            )
                rows.append(row)
    return pd.DataFrame(rows)


def _interval_ledger(
    pauses: list[TimeInterval],
    pause_frames: pd.DataFrame,
    spectral: pd.DataFrame,
    *,
    logical_recording_id: str,
) -> pd.DataFrame:
    rows = []
    for interval_index, interval in enumerate(pauses):
        local = pause_frames.loc[
            pause_frames["interval_index"].eq(interval_index)
        ] if not pause_frames.empty else pd.DataFrame()
        nonfloor = local.loc[
            ~local["at_computational_floor"].astype(bool)
        ] if not local.empty else pd.DataFrame()
        local_spectral = spectral.loc[
            spectral["interval_index"].eq(interval_index)
        ] if not spectral.empty else pd.DataFrame()
        rows.append(
            {
                "logical_recording_id": str(logical_recording_id),
                "interval_index": interval_index,
                "interval_start_sec": interval.start_sec,
                "interval_end_sec": interval.end_sec,
                "interval_duration_sec": interval.duration_sec,
                "pause_frame_count_total": len(local),
                "pause_frame_count_nonfloor": len(nonfloor),
                "pause_effective_nonfloor_support_sec": _effective_frame_support_sec(
                    nonfloor
                ),
                "pause_at_floor_frame_fraction": (
                    float(local["at_computational_floor"].astype(bool).mean())
                    if len(local)
                    else np.nan
                ),
                "pause_exact_zero_frame_fraction": (
                    float(local["exact_zero_frame"].astype(bool).mean())
                    if len(local)
                    else np.nan
                ),
                "pause_ac_level_dbfs_median_raw": (
                    float(nonfloor["rms_dbfs"].median()) if len(nonfloor) else np.nan
                ),
                "flatness_valid_window_count": int(
                    (
                        local_spectral["window_kind"].eq("flatness")
                        & local_spectral["valid_acoustic_window"].astype(bool)
                    ).sum()
                )
                if len(local_spectral)
                else 0,
                "hum_valid_window_count": int(
                    (
                        local_spectral["window_kind"].eq("hum")
                        & local_spectral["valid_acoustic_window"].astype(bool)
                    ).sum()
                )
                if len(local_spectral)
                else 0,
            }
        )
    return pd.DataFrame(rows)


def _level_support_tier(
    support_sec: float,
    frame_count: int,
    interval_count: int,
    *,
    dispersion: bool,
    parameters: QADDParameters,
) -> str:
    if dispersion:
        minimum_sec = parameters.dispersion_min_pause_sec
        minimum_frames = parameters.dispersion_min_frames
        moderate_sec = parameters.dispersion_moderate_pause_sec
        high_support_sec = parameters.dispersion_high_support_pause_sec
        moderate_intervals = parameters.dispersion_moderate_intervals
        high_support_intervals = parameters.dispersion_high_support_intervals
    else:
        minimum_sec = parameters.level_min_pause_sec
        minimum_frames = parameters.level_min_frames
        moderate_sec = parameters.level_moderate_pause_sec
        high_support_sec = parameters.level_high_support_pause_sec
        moderate_intervals = parameters.level_moderate_intervals
        high_support_intervals = parameters.level_high_support_intervals
    if support_sec < minimum_sec or frame_count < minimum_frames or interval_count < 1:
        return "unavailable"
    if support_sec >= high_support_sec and interval_count >= high_support_intervals:
        return "high"
    if support_sec >= moderate_sec and interval_count >= moderate_intervals:
        return "moderate"
    return "minimum"


def _count_support_tier(
    count: int, minimum: int, moderate: int, high_support: int
) -> str:
    if count < minimum:
        return "unavailable"
    if count >= high_support:
        return "high"
    if count >= moderate:
        return "moderate"
    return "minimum"


def _level_feature_status(
    tier: str, floor_fraction: float, *, parameters: QADDParameters
) -> str:
    if tier == "unavailable":
        return "insufficient_support"
    if not np.isfinite(floor_fraction):
        return "insufficient_support"
    if floor_fraction > parameters.maximum_floor_censored_fraction:
        return "floor_censored"
    return f"ok_{tier}"


def _spectral_feature_status(tier: str) -> str:
    """Status for descriptors computed only from explicitly non-floor windows.

    Unlike the level estimands, spectral estimands do not substitute the
    computational floor into their calculation.  Their missingness is therefore
    governed by valid non-floor window support, while floor-window fractions are
    retained as audit variables rather than used as an unrelated censoring gate.
    """

    if tier == "unavailable":
        return "insufficient_support"
    return f"ok_{tier}"


def _is_publishable_status(status: str) -> bool:
    return str(status).startswith("ok_")


def _finite(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.loc[np.isfinite(numeric)]


def _raw_feature_estimates(
    frame_ledger: pd.DataFrame, spectral_ledger: pd.DataFrame
) -> dict:
    pause_all = frame_ledger.loc[frame_ledger["region"].eq("pause")]
    speech_all = frame_ledger.loc[frame_ledger["region"].eq("speech")]
    pause = pause_all.loc[~pause_all["at_computational_floor"].astype(bool)]
    speech = speech_all.loc[~speech_all["at_computational_floor"].astype(bool)]
    pause_values = _finite(pause.get("rms_dbfs", pd.Series(dtype=float)))
    speech_values = _finite(speech.get("rms_dbfs", pd.Series(dtype=float)))

    flatness = spectral_ledger.loc[
        spectral_ledger["window_kind"].eq("flatness")
        & spectral_ledger["valid_acoustic_window"].astype(bool)
    ]
    flatness_values = _finite(flatness.get("spectral_flatness", pd.Series(dtype=float)))

    hum = spectral_ledger.loc[
        spectral_ledger["window_kind"].eq("hum")
        & spectral_ledger["valid_acoustic_window"].astype(bool)
    ]
    hum_50 = _finite(hum.get("hum_score_50_db", pd.Series(dtype=float)))
    hum_60 = _finite(hum.get("hum_score_60_db", pd.Series(dtype=float)))
    recording_50 = float(hum_50.median()) if len(hum_50) else np.nan
    recording_60 = float(hum_60.median()) if len(hum_60) else np.nan
    hum_score = (
        float(np.nanmax([recording_50, recording_60]))
        if np.isfinite(recording_50) or np.isfinite(recording_60)
        else np.nan
    )
    hum_winner = (
        50.0
        if np.nan_to_num(recording_50, nan=-np.inf)
        >= np.nan_to_num(recording_60, nan=-np.inf)
        else 60.0
    )
    winner_column = "hum_score_50_db" if hum_winner == 50.0 else "hum_score_60_db"
    winner_support_column = (
        "hum_supported_harmonic_count_50"
        if hum_winner == 50.0
        else "hum_supported_harmonic_count_60"
    )
    winner_windows = hum.loc[np.isfinite(pd.to_numeric(hum[winner_column], errors="coerce"))]

    return {
        "qadd_pause_ac_level_dbfs_median_raw_estimate": (
            float(pause_values.median()) if len(pause_values) else np.nan
        ),
        "qadd_pause_level_iqr_db_raw_estimate": (
            float(pause_values.quantile(0.75) - pause_values.quantile(0.25))
            if len(pause_values)
            else np.nan
        ),
        "qadd_speech_pause_level_contrast_db_raw_estimate": (
            float(speech_values.median() - pause_values.median())
            if len(speech_values) and len(pause_values)
            else np.nan
        ),
        "qadd_pause_spectral_flatness_raw_estimate": (
            float(flatness_values.median()) if len(flatness_values) else np.nan
        ),
        "qadd_mains_hum_comb_score_db_raw_estimate": hum_score,
        "qadd_mains_hum_score_50_db_raw": recording_50,
        "qadd_mains_hum_score_60_db_raw": recording_60,
        "qadd_mains_hum_winner_hz": hum_winner if np.isfinite(hum_score) else np.nan,
        "qadd_mains_hum_supported_harmonic_count_median": (
            float(
                pd.to_numeric(
                    winner_windows[winner_support_column], errors="coerce"
                ).median()
            )
            if len(winner_windows)
            else np.nan
        ),
        "qadd_pause_ac_level_dbfs_median_floor_inclusive": (
            float(_finite(pause_all["rms_dbfs"]).median()) if len(pause_all) else np.nan
        ),
        "qadd_pause_level_iqr_db_floor_inclusive": (
            float(
                _finite(pause_all["rms_dbfs"]).quantile(0.75)
                - _finite(pause_all["rms_dbfs"]).quantile(0.25)
            )
            if len(pause_all)
            else np.nan
        ),
    }


def reconstruct_raw_features(
    frame_ledger: pd.DataFrame, spectral_ledger: pd.DataFrame
) -> dict:
    """Reconstruct all raw estimands from the saved ledgers."""

    required_frame = {"region", "rms_dbfs", "at_computational_floor"}
    required_spectral = {
        "window_kind",
        "valid_acoustic_window",
        "spectral_flatness",
        "hum_score_50_db",
        "hum_score_60_db",
    }
    if not required_frame.issubset(frame_ledger):
        raise ValueError(f"Frame ledger is missing {sorted(required_frame - set(frame_ledger))}")
    if not required_spectral.issubset(spectral_ledger):
        raise ValueError(
            f"Spectral ledger is missing {sorted(required_spectral - set(spectral_ledger))}"
        )
    return _raw_feature_estimates(frame_ledger, spectral_ledger)


def cluster_delete_one_diagnostics(
    frame_ledger: pd.DataFrame,
    spectral_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute every raw QADD estimand after deleting one whole pause.

    Rows are pause-interval perturbations, not independent observations.
    Cohort summaries must therefore first reduce them to one value per
    recording and feature. Large changes are retained as diagnostics because
    they can represent genuine nonstationary or rare interference.
    """

    required_frame = {
        "logical_recording_id",
        "region",
        "interval_index",
        "rms_dbfs",
        "at_computational_floor",
    }
    required_spectral = {
        "logical_recording_id",
        "interval_index",
        "window_kind",
        "valid_acoustic_window",
        "spectral_flatness",
        "hum_score_50_db",
        "hum_score_60_db",
    }
    if not required_frame.issubset(frame_ledger):
        raise ValueError(
            f"Frame ledger is missing {sorted(required_frame - set(frame_ledger))}"
        )
    if not required_spectral.issubset(spectral_ledger):
        raise ValueError(
            "Spectral ledger is missing "
            f"{sorted(required_spectral - set(spectral_ledger))}"
        )

    pause_frames = frame_ledger.loc[frame_ledger["region"].eq("pause")]
    interval_ids = sorted(
        pd.to_numeric(pause_frames["interval_index"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    output_columns = [
        "logical_recording_id",
        "omitted_interval",
        "feature",
        "full_raw_estimate",
        "deleted_raw_estimate",
        "absolute_change",
        "remaining_pause_interval_count",
    ]
    if len(interval_ids) < 2:
        return pd.DataFrame(columns=output_columns)

    identifiers = pd.concat(
        [
            frame_ledger["logical_recording_id"].dropna().astype(str),
            spectral_ledger["logical_recording_id"].dropna().astype(str),
        ],
        ignore_index=True,
    ).unique()
    logical_recording_id = str(identifiers[0]) if len(identifiers) else ""
    if len(identifiers) > 1:
        raise ValueError("Cluster-deletion ledgers contain more than one recording")

    full = reconstruct_raw_features(frame_ledger, spectral_ledger)
    rows: list[dict] = []
    for omitted in interval_ids:
        keep_frame = ~(
            frame_ledger["region"].eq("pause")
            & pd.to_numeric(frame_ledger["interval_index"], errors="coerce").eq(omitted)
        )
        keep_spectral = ~pd.to_numeric(
            spectral_ledger["interval_index"], errors="coerce"
        ).eq(omitted)
        deleted = reconstruct_raw_features(
            frame_ledger.loc[keep_frame],
            spectral_ledger.loc[keep_spectral],
        )
        for feature in ANALYSIS_FEATURES:
            raw_name = f"{feature}_raw_estimate"
            full_value = float(full.get(raw_name, np.nan))
            deleted_value = float(deleted.get(raw_name, np.nan))
            rows.append(
                {
                    "logical_recording_id": logical_recording_id,
                    "omitted_interval": int(omitted),
                    "feature": feature,
                    "full_raw_estimate": full_value,
                    "deleted_raw_estimate": deleted_value,
                    "absolute_change": (
                        abs(deleted_value - full_value)
                        if np.isfinite(full_value) and np.isfinite(deleted_value)
                        else np.nan
                    ),
                    "remaining_pause_interval_count": len(interval_ids) - 1,
                }
            )
    return pd.DataFrame(rows, columns=output_columns)


def summarize_cluster_deletion(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return one clustered sensitivity summary per recording and feature."""

    required = {"logical_recording_id", "feature", "absolute_change"}
    if not required.issubset(diagnostics):
        raise ValueError(
            f"Diagnostics are missing {sorted(required - set(diagnostics))}"
        )
    output_columns = [
        "logical_recording_id",
        "feature",
        "finite_deletion_count",
        "delete_one_median_absolute_change",
        "delete_one_p90_absolute_change",
        "delete_one_max_absolute_change",
    ]
    if diagnostics.empty:
        return pd.DataFrame(columns=output_columns)

    rows = []
    for (recording_id, feature), group in diagnostics.groupby(
        ["logical_recording_id", "feature"], sort=False
    ):
        values = _finite(group["absolute_change"])
        rows.append(
            {
                "logical_recording_id": recording_id,
                "feature": feature,
                "finite_deletion_count": int(len(values)),
                "delete_one_median_absolute_change": (
                    float(values.median()) if len(values) else np.nan
                ),
                "delete_one_p90_absolute_change": (
                    float(values.quantile(0.90)) if len(values) else np.nan
                ),
                "delete_one_max_absolute_change": (
                    float(values.max()) if len(values) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def extract_qadd(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    primary_speech: Iterable[TimeInterval],
    strict_speech: Iterable[TimeInterval] | None = None,
    strict_internal_nonspeech: Iterable[TimeInterval] | None = None,
    logical_recording_id: str = "",
    speech_intervals_are_guarded: bool = False,
    pause_intervals_are_guarded: bool = False,
    parameters: QADDParameters = DEFAULT_PARAMETERS,
) -> QADDExtraction:
    """Extract the complete QADD v4.2 vector and audit ledgers for one recording."""

    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("QADD expects one mono analysis waveform")
    if values.size == 0:
        raise ValueError("QADD waveform is empty")
    if not np.isfinite(values).all():
        raise ValueError("QADD waveform contains NaN or infinity")
    if int(sample_rate) != parameters.analysis_sample_rate_hz:
        raise ValueError(
            "QADD v4.2 requires the frozen 16-kHz analysis view; "
            f"received {sample_rate} Hz"
        )

    duration_sec = len(values) / float(sample_rate)
    primary = normalize_intervals(primary_speech, duration_sec)
    speech_source = primary if strict_speech is None else normalize_intervals(
        strict_speech, duration_sec
    )
    if speech_intervals_are_guarded:
        speech = [
            item for item in speech_source
            if item.duration_sec >= parameters.frame_ms / 1000.0
        ]
        speech_source_name = "provided_guarded_strict_speech"
        speech_guard_applied_ms = 0.0
    else:
        speech = erode_intervals(
            speech_source,
            duration_sec,
            guard_ms=parameters.speech_guard_ms,
            minimum_ms=parameters.frame_ms,
        )
        speech_source_name = (
            "derived_guarded_primary_speech"
            if strict_speech is None
            else "provided_speech_with_qadd_guard"
        )
        speech_guard_applied_ms = float(parameters.speech_guard_ms)
    if pause_intervals_are_guarded:
        if strict_internal_nonspeech is None:
            raise ValueError(
                "pause_intervals_are_guarded=True requires strict_internal_nonspeech"
            )
        pauses = normalize_intervals(strict_internal_nonspeech, duration_sec)
        pauses = [
            item
            for item in pauses
            if item.duration_sec >= parameters.minimum_residual_pause_ms / 1000.0
        ]
        pause_source = "provided_guarded_strict_nonspeech"
    else:
        pauses = guarded_internal_pauses(
            primary,
            duration_sec,
            strict_nonspeech=strict_internal_nonspeech,
            parameters=parameters,
        )
        pause_source = (
            "derived_guard_intersected_frozen_strict_nonspeech"
            if strict_internal_nonspeech is not None
            else "derived_guarded_internal_pauses"
        )

    pause_frames = _frame_table(
        values,
        sample_rate,
        pauses,
        region="pause",
        logical_recording_id=logical_recording_id,
        parameters=parameters,
    )
    speech_frames = _frame_table(
        values,
        sample_rate,
        speech,
        region="speech",
        logical_recording_id=logical_recording_id,
        parameters=parameters,
    )
    frame_ledger = pd.concat([pause_frames, speech_frames], ignore_index=True)
    if frame_ledger.empty:
        frame_ledger = pd.DataFrame(
            columns=[
                "logical_recording_id",
                "region",
                "interval_index",
                "frame_index_in_interval",
                "frame_start_sec",
                "frame_end_sec",
                "rms_dbfs",
                "rms_linear",
                "at_computational_floor",
                "exact_zero_frame",
            ]
        )

    spectral_ledger = _spectral_window_table(
        values,
        sample_rate,
        pauses,
        logical_recording_id=logical_recording_id,
        parameters=parameters,
    )
    if spectral_ledger.empty:
        spectral_ledger = pd.DataFrame(
            columns=[
                "logical_recording_id",
                "window_kind",
                "interval_index",
                "window_index_in_interval",
                "window_start_sec",
                "window_end_sec",
                "window_ac_level_dbfs",
                "at_computational_floor",
                "exact_zero_window",
                "valid_acoustic_window",
                "spectral_flatness",
                "hum_score_50_db",
                "hum_score_60_db",
                "hum_score_max_db",
                "hum_winner_hz",
                "hum_supported_harmonic_count",
                "hum_evaluated_harmonic_count",
                "hum_supported_harmonic_count_50",
                "hum_evaluated_harmonic_count_50",
                "hum_supported_harmonic_count_60",
                "hum_evaluated_harmonic_count_60",
            ]
        )
    interval_ledger = _interval_ledger(
        pauses,
        pause_frames,
        spectral_ledger,
        logical_recording_id=logical_recording_id,
    )

    raw = _raw_feature_estimates(frame_ledger, spectral_ledger)
    pause_nonfloor = pause_frames.loc[
        ~pause_frames["at_computational_floor"].astype(bool)
    ] if len(pause_frames) else pause_frames
    speech_nonfloor = speech_frames.loc[
        ~speech_frames["at_computational_floor"].astype(bool)
    ] if len(speech_frames) else speech_frames
    pause_support_sec = _effective_frame_support_sec(pause_nonfloor)
    speech_support_sec = _effective_frame_support_sec(speech_nonfloor)
    pause_interval_count = (
        int(pause_nonfloor["interval_index"].nunique()) if len(pause_nonfloor) else 0
    )
    pause_floor_fraction = (
        float(pause_frames["at_computational_floor"].astype(bool).mean())
        if len(pause_frames)
        else np.nan
    )
    speech_floor_fraction = (
        float(speech_frames["at_computational_floor"].astype(bool).mean())
        if len(speech_frames)
        else np.nan
    )
    exact_zero_fraction = (
        float(pause_frames["exact_zero_frame"].astype(bool).mean())
        if len(pause_frames)
        else np.nan
    )

    level_tier = _level_support_tier(
        pause_support_sec,
        len(pause_nonfloor),
        pause_interval_count,
        dispersion=False,
        parameters=parameters,
    )
    dispersion_tier = _level_support_tier(
        pause_support_sec,
        len(pause_nonfloor),
        pause_interval_count,
        dispersion=True,
        parameters=parameters,
    )
    if (
        level_tier == "unavailable"
        or speech_support_sec < parameters.contrast_min_speech_sec
        or len(speech_nonfloor) < parameters.contrast_min_speech_frames
    ):
        contrast_tier = "unavailable"
    elif (
        level_tier == "high"
        and speech_support_sec >= parameters.contrast_high_support_speech_sec
    ):
        contrast_tier = "high"
    elif (
        level_tier in {"moderate", "high"}
        and speech_support_sec >= parameters.contrast_moderate_speech_sec
    ):
        contrast_tier = "moderate"
    else:
        contrast_tier = "minimum"

    valid_flatness = spectral_ledger.loc[
        spectral_ledger["window_kind"].eq("flatness")
        & spectral_ledger["valid_acoustic_window"].astype(bool)
        & np.isfinite(pd.to_numeric(spectral_ledger["spectral_flatness"], errors="coerce"))
    ]
    valid_hum = spectral_ledger.loc[
        spectral_ledger["window_kind"].eq("hum")
        & spectral_ledger["valid_acoustic_window"].astype(bool)
        & np.isfinite(pd.to_numeric(spectral_ledger["hum_score_max_db"], errors="coerce"))
    ]
    flatness_tier = _count_support_tier(
        len(valid_flatness),
        parameters.flatness_min_windows,
        parameters.flatness_moderate_windows,
        parameters.flatness_high_support_windows,
    )
    hum_tier = _count_support_tier(
        len(valid_hum),
        parameters.hum_min_windows,
        parameters.hum_moderate_windows,
        parameters.hum_high_support_windows,
    )
    flatness_windows = spectral_ledger.loc[
        spectral_ledger["window_kind"].eq("flatness")
    ]
    hum_windows = spectral_ledger.loc[spectral_ledger["window_kind"].eq("hum")]
    flatness_floor_fraction = (
        float(flatness_windows["at_computational_floor"].astype(bool).mean())
        if len(flatness_windows)
        else np.nan
    )
    hum_floor_fraction = (
        float(hum_windows["at_computational_floor"].astype(bool).mean())
        if len(hum_windows)
        else np.nan
    )

    statuses = {
        "qadd_pause_ac_level_dbfs_median_status": _level_feature_status(
            level_tier, pause_floor_fraction, parameters=parameters
        ),
        "qadd_pause_level_iqr_db_status": _level_feature_status(
            dispersion_tier, pause_floor_fraction, parameters=parameters
        ),
        "qadd_speech_pause_level_contrast_db_status": _level_feature_status(
            contrast_tier, pause_floor_fraction, parameters=parameters
        ),
        "qadd_pause_spectral_flatness_status": _spectral_feature_status(flatness_tier),
        "qadd_mains_hum_comb_score_db_status": _spectral_feature_status(hum_tier),
    }

    recording = {
        "logical_recording_id": str(logical_recording_id),
        "qadd_measurement_version": parameters.measurement_version,
        "qadd_signal_view": "mono_dc_removed_resampled_16khz_analysis_view",
        "qadd_level_definition": "30ms_ac_rms_dbfs_10ms_hop",
        "qadd_pause_source": pause_source,
        "qadd_speech_source": speech_source_name,
        "qadd_speech_guard_applied_ms": speech_guard_applied_ms,
        "qadd_pause_guard_ms": parameters.pause_guard_ms,
        "qadd_minimum_residual_pause_ms": parameters.minimum_residual_pause_ms,
        "qadd_speech_guard_ms": parameters.speech_guard_ms,
        "qadd_pause_interval_count_total": len(pauses),
        "qadd_pause_interval_count_nonfloor": pause_interval_count,
        "qadd_pause_support_sec_total": interval_union_duration(pauses),
        "qadd_pause_effective_nonfloor_support_sec": pause_support_sec,
        "qadd_speech_effective_nonfloor_support_sec": speech_support_sec,
        "qadd_pause_frame_count_total": len(pause_frames),
        "qadd_pause_frame_count_nonfloor": len(pause_nonfloor),
        "qadd_speech_frame_count_total": len(speech_frames),
        "qadd_speech_frame_count_nonfloor": len(speech_nonfloor),
        "qadd_pause_at_floor_frame_fraction": pause_floor_fraction,
        "qadd_pause_exact_zero_frame_fraction": exact_zero_fraction,
        "qadd_speech_at_floor_frame_fraction": speech_floor_fraction,
        "qadd_flatness_valid_window_count": len(valid_flatness),
        "qadd_hum_valid_window_count": len(valid_hum),
        "qadd_flatness_window_count_total": len(flatness_windows),
        "qadd_hum_window_count_total": len(hum_windows),
        "qadd_flatness_at_floor_window_fraction": flatness_floor_fraction,
        "qadd_hum_at_floor_window_fraction": hum_floor_fraction,
        "qadd_pause_level_support_tier": level_tier,
        "qadd_pause_dispersion_support_tier": dispersion_tier,
        "qadd_speech_pause_contrast_support_tier": contrast_tier,
        "qadd_flatness_support_tier": flatness_tier,
        "qadd_hum_support_tier": hum_tier,
        **raw,
        **statuses,
    }
    recording["qadd_mains_hum_null_p95_db"] = np.nan
    recording["qadd_mains_hum_excess_over_null_p95_db"] = np.nan
    recording["qadd_mains_hum_joint_evidence_above_null"] = pd.NA
    recording["qadd_mains_hum_null_reference_window_count"] = pd.NA
    recording["qadd_mains_hum_null_calibration_status"] = "not_applied"
    for feature in ANALYSIS_FEATURES:
        raw_name = f"{feature}_raw_estimate"
        status_name = f"{feature}_status"
        recording[feature] = (
            recording[raw_name]
            if _is_publishable_status(recording[status_name])
            else np.nan
        )

    primary_status = statuses["qadd_pause_ac_level_dbfs_median_status"]
    recording["qadd_primary_analysis_eligible"] = _is_publishable_status(primary_status)
    recording["qadd_family_status"] = (
        "primary_available"
        if recording["qadd_primary_analysis_eligible"]
        else "floor_censored"
        if primary_status == "floor_censored"
        else "unavailable"
    )
    return QADDExtraction(recording, frame_ledger, interval_ledger, spectral_ledger)


def apply_hum_null_calibration(
    recording: dict,
    null_p95_db: float,
    *,
    minimum_supported_harmonics: int = DEFAULT_PARAMETERS.hum_min_supported_harmonics,
) -> dict:
    """Attach a prespecified, count-matched hum-null reference to one result.

    The raw comb score remains the analysis feature.  The null P95 and excess
    are interpretive audit companions and must be produced from an independent,
    prespecified simulation recipe matched to the recording's valid hum-window
    count.
    """

    calibrated = dict(recording)
    threshold = float(null_p95_db)
    score = float(
        calibrated.get("qadd_mains_hum_comb_score_db_raw_estimate", np.nan)
    )
    if not np.isfinite(threshold):
        calibrated["qadd_mains_hum_null_p95_db"] = np.nan
        calibrated["qadd_mains_hum_excess_over_null_p95_db"] = np.nan
        calibrated["qadd_mains_hum_null_calibration_status"] = "invalid_null_reference"
        return calibrated
    calibrated["qadd_mains_hum_null_p95_db"] = threshold
    calibrated["qadd_mains_hum_excess_over_null_p95_db"] = (
        score - threshold if np.isfinite(score) else np.nan
    )
    supported = float(
        calibrated.get("qadd_mains_hum_supported_harmonic_count_median", np.nan)
    )
    calibrated["qadd_mains_hum_joint_evidence_above_null"] = (
        bool(
            score > threshold
            and supported >= int(minimum_supported_harmonics)
        )
        if np.isfinite(score) and np.isfinite(supported)
        else pd.NA
    )
    calibrated["qadd_mains_hum_null_calibration_status"] = (
        "applied" if np.isfinite(score) else "feature_unavailable"
    )
    return calibrated


def compare_reconstruction(
    extraction: QADDExtraction, *, absolute_tolerance: float = 1e-10
) -> pd.DataFrame:
    """Compare stored raw estimates with independent ledger reconstruction."""

    reconstructed = reconstruct_raw_features(
        extraction.frame_ledger, extraction.spectral_ledger
    )
    rows = []
    for feature, rebuilt in reconstructed.items():
        stored = extraction.recording.get(feature, np.nan)
        both_missing = not np.isfinite(stored) and not np.isfinite(rebuilt)
        difference = (
            abs(float(stored) - float(rebuilt))
            if np.isfinite(stored) and np.isfinite(rebuilt)
            else np.nan
        )
        rows.append(
            {
                "raw_estimate": feature,
                "stored": stored,
                "reconstructed": rebuilt,
                "absolute_difference": difference,
                "pass": bool(both_missing or difference <= absolute_tolerance),
            }
        )
    return pd.DataFrame(rows)
