from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Interval:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def normalize_intervals(intervals: Iterable[Interval], duration_sec: float) -> list[Interval]:
    clipped = [
        Interval(max(0.0, float(item.start_sec)), min(duration_sec, float(item.end_sec)))
        for item in intervals
        if float(item.end_sec) > float(item.start_sec)
    ]
    clipped = [item for item in clipped if item.duration_sec > 0]
    clipped.sort(key=lambda item: (item.start_sec, item.end_sec))
    merged: list[Interval] = []
    for item in clipped:
        if merged and item.start_sec <= merged[-1].end_sec:
            merged[-1] = Interval(merged[-1].start_sec, max(merged[-1].end_sec, item.end_sec))
        else:
            merged.append(item)
    return merged


def bridge_and_filter(
    intervals: Iterable[Interval],
    *,
    duration_sec: float,
    bridge_gap_sec: float,
    min_speech_sec: float,
) -> list[Interval]:
    normalized = normalize_intervals(intervals, duration_sec)
    bridged: list[Interval] = []
    for item in normalized:
        if bridged and item.start_sec - bridged[-1].end_sec <= bridge_gap_sec:
            bridged[-1] = Interval(bridged[-1].start_sec, item.end_sec)
        else:
            bridged.append(item)
    return [item for item in bridged if item.duration_sec >= min_speech_sec]


def erode_intervals(intervals: Iterable[Interval], edge_sec: float) -> list[Interval]:
    return [
        Interval(item.start_sec + edge_sec, item.end_sec - edge_sec)
        for item in intervals
        if item.duration_sec > 2 * edge_sec
    ]


def complement_intervals(intervals: Iterable[Interval], duration_sec: float) -> list[Interval]:
    speech = normalize_intervals(intervals, duration_sec)
    result: list[Interval] = []
    cursor = 0.0
    for item in speech:
        if item.start_sec > cursor:
            result.append(Interval(cursor, item.start_sec))
        cursor = max(cursor, item.end_sec)
    if cursor < duration_sec:
        result.append(Interval(cursor, duration_sec))
    return result


def internal_nonspeech(intervals: Iterable[Interval], duration_sec: float) -> list[Interval]:
    speech = normalize_intervals(intervals, duration_sec)
    if len(speech) < 2:
        return []
    return [
        Interval(left.end_sec, right.start_sec)
        for left, right in zip(speech[:-1], speech[1:])
        if right.start_sec > left.end_sec
    ]


def build_segmentation_views(
    raw_speech: Iterable[Interval],
    *,
    duration_sec: float,
    bridge_gap_ms: float = 100,
    min_speech_ms: float = 250,
    strict_speech_edge_ms: float = 50,
    strict_nonspeech_edge_ms: float = 200,
) -> dict[str, list[Interval]]:
    """Create genuinely distinct raw, primary, strict-speech, and strict-nonspeech views."""
    raw = normalize_intervals(raw_speech, duration_sec)
    primary = bridge_and_filter(
        raw,
        duration_sec=duration_sec,
        bridge_gap_sec=bridge_gap_ms / 1000,
        min_speech_sec=min_speech_ms / 1000,
    )
    strict_speech = erode_intervals(primary, strict_speech_edge_ms / 1000)
    strict_internal_nonspeech = erode_intervals(
        internal_nonspeech(primary, duration_sec), strict_nonspeech_edge_ms / 1000
    )
    return {
        "raw_speech": raw,
        "primary_speech": primary,
        "strict_speech": strict_speech,
        "strict_internal_nonspeech": strict_internal_nonspeech,
    }


def silero_speech_intervals(
    waveform_16k: np.ndarray,
    *,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 0,
    onnx: bool = True,
    model=None,
) -> list[Interval]:
    """Run installed, version-pinned Silero and return sample-derived intervals.

    The analysis default is deliberately unpadded. Silero's ``speech_pad_ms`` is a
    convenience expansion of every detected region, not evidence about the acoustic
    onset or offset. Padding is therefore unsuitable for a boundary-sensitive
    measurement pipeline unless it is requested explicitly.
    """
    stamps = silero_speech_timestamps(
        waveform_16k,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        onnx=onnx,
        model=model,
    )
    return [Interval(float(item["start"]) / 16000, float(item["end"]) / 16000) for item in stamps]


def load_silero_model(*, onnx: bool = True):
    """Load one version-pinned Silero model for reuse across recordings/profiles."""
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=onnx)


def silero_speech_timestamps(
    waveform_16k: np.ndarray,
    *,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 0,
    onnx: bool = True,
    model=None,
) -> list[dict[str, int]]:
    """Return sample-index timestamps without display-frame quantization."""
    import torch
    from silero_vad import get_speech_timestamps

    if model is None:
        model = load_silero_model(onnx=onnx)
    waveform = torch.from_numpy(np.asarray(waveform_16k, dtype=np.float32))
    stamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )
    return [{"start": int(item["start"]), "end": int(item["end"])} for item in stamps]


BOUNDARY_AUDIT_COLUMNS = [
    "segment_index",
    "start_sec",
    "end_sec",
    "duration_sec",
    "display_start_sec",
    "display_end_sec",
    "display_onset_delta_ms",
    "display_offset_delta_ms",
    "onset_pre_rms_dbfs",
    "onset_inside_rms_dbfs",
    "onset_contrast_db",
    "offset_inside_rms_dbfs",
    "offset_post_rms_dbfs",
    "offset_contrast_db",
    "ambiguous_onset",
    "ambiguous_offset",
    "boundary_review_flag",
]


def _window_rms_dbfs(
    signal: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> float:
    start = max(0, int(round(start_sec * sample_rate)))
    stop = min(len(signal), int(round(end_sec * sample_rate)))
    if stop <= start:
        return float("nan")
    rms = float(np.sqrt(np.mean(np.square(signal[start:stop], dtype=np.float64))))
    return float(20.0 * np.log10(max(rms, 1e-12)))


def boundary_alignment_diagnostics(
    waveform: np.ndarray,
    sample_rate: int,
    speech_intervals: Iterable[Interval],
    *,
    displayed_segments: pd.DataFrame | None = None,
    window_ms: float = 120,
    guard_ms: float = 20,
    minimum_contrast_db: float = 3.0,
) -> pd.DataFrame:
    """Audit exact Silero boundaries without moving them using an energy heuristic.

    Local RMS contrast is a review signal only. Low-intensity/breathy ALS speech can
    have weak energy contrast, so this function never snaps, trims, or expands a
    boundary automatically. ``display_*`` fields quantify the separate 30-ms
    compatibility-frame representation used by the original-style plot.
    """
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    intervals = list(speech_intervals)
    display_speech = pd.DataFrame()
    if displayed_segments is not None and not displayed_segments.empty:
        display_speech = displayed_segments.loc[
            displayed_segments["segment_type"].astype(str).eq("speech")
        ].reset_index(drop=True)
    window_sec = float(window_ms) / 1000.0
    guard_sec = float(guard_ms) / 1000.0
    if window_sec <= guard_sec:
        raise ValueError("boundary audit window_ms must be greater than guard_ms")

    rows = []
    for index, interval in enumerate(intervals):
        start = float(interval.start_sec)
        end = float(interval.end_sec)
        onset_pre = _window_rms_dbfs(
            signal, sample_rate, start - window_sec, start - guard_sec
        )
        onset_inside = _window_rms_dbfs(
            signal, sample_rate, start + guard_sec, min(end, start + window_sec)
        )
        offset_inside = _window_rms_dbfs(
            signal, sample_rate, max(start, end - window_sec), end - guard_sec
        )
        offset_post = _window_rms_dbfs(
            signal, sample_rate, end + guard_sec, end + window_sec
        )
        onset_contrast = onset_inside - onset_pre
        offset_contrast = offset_inside - offset_post
        ambiguous_onset = bool(
            np.isfinite(onset_contrast) and onset_contrast < minimum_contrast_db
        )
        ambiguous_offset = bool(
            np.isfinite(offset_contrast) and offset_contrast < minimum_contrast_db
        )

        display_start = float("nan")
        display_end = float("nan")
        if index < len(display_speech):
            display_start = float(display_speech.loc[index, "start_sec"])
            display_end = float(display_speech.loc[index, "end_sec"])
        rows.append(
            {
                "segment_index": index,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "display_start_sec": display_start,
                "display_end_sec": display_end,
                "display_onset_delta_ms": (
                    1000.0 * (display_start - start)
                    if np.isfinite(display_start)
                    else np.nan
                ),
                "display_offset_delta_ms": (
                    1000.0 * (display_end - end)
                    if np.isfinite(display_end)
                    else np.nan
                ),
                "onset_pre_rms_dbfs": onset_pre,
                "onset_inside_rms_dbfs": onset_inside,
                "onset_contrast_db": onset_contrast,
                "offset_inside_rms_dbfs": offset_inside,
                "offset_post_rms_dbfs": offset_post,
                "offset_contrast_db": offset_contrast,
                "ambiguous_onset": ambiguous_onset,
                "ambiguous_offset": ambiguous_offset,
                "boundary_review_flag": ambiguous_onset or ambiguous_offset,
            }
        )
    return pd.DataFrame(rows, columns=BOUNDARY_AUDIT_COLUMNS)


LEGACY_SILERO_FRAME_COLUMNS = [
    "frame_idx",
    "mid_sec",
    "rms",
    "rms_db",
    "speech_vad_raw",
    "speech_vad_smooth",
    "speech_mask_strict",
    "nonspeech_mask_strict",
    "threshold",
    "frame_ms",
]

LEGACY_SILERO_SEGMENT_COLUMNS = [
    "segment_type",
    "start_sec",
    "end_sec",
    "duration_sec",
    "run_start_frame",
    "run_end_frame",
    "segment_role",
]

LEGACY_SILERO_SUMMARY_COLUMNS = [
    "method",
    "duration_sec",
    "sample_rate_analysis",
    "threshold",
    "frame_ms",
    "min_speech_duration_ms",
    "min_silence_duration_ms",
    "speech_pad_ms",
    "n_frames",
    "n_segments_total",
    "n_speech_segments",
    "n_internal_nonspeech_segments",
    "speech_fraction",
    "leading_nonspeech_sec",
    "trailing_nonspeech_sec",
    "longest_internal_nonspeech_sec",
    "rms_db_median",
    "rms_db_std",
    "file_name",
    "file_path",
    "ID_norm",
    "Diagnosis",
    "severity_bin",
    "Recording date",
    "qc_status",
    "qc_flags",
    "segments_path",
    "frames_path",
    "plot_path",
]


def _boolean_runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    """Return contiguous boolean runs as ``(value, start, end_exclusive)``."""
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return []
    runs: list[tuple[bool, int, int]] = []
    start = 0
    current = values[0]
    for index in range(1, len(values)):
        if values[index] != current:
            runs.append((bool(current), int(start), int(index)))
            start = index
            current = values[index]
    runs.append((bool(current), int(start), int(len(values))))
    return runs


def legacy_silero_artifacts(
    waveform: np.ndarray,
    sample_rate: int,
    speech_timestamps: Iterable[dict[str, int]],
    *,
    threshold: float = 0.5,
    frame_ms: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the original per-recording Silero frame and segment CSVs exactly.

    These diagnostic tables intentionally retain the original column names, including
    the three identical speech-mask aliases. The scientifically distinct raw, primary,
    strict-speech, and guarded-nonspeech views are saved separately in the aggregate
    interval table and are the views used by downstream Q-metric extraction.
    """
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if sample_rate not in {8000, 16000}:
        raise ValueError(f"Silero diagnostic artifacts require 8 or 16 kHz, got {sample_rate}")
    frame_len = int(sample_rate * frame_ms / 1000.0)
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")

    sample_mask = np.zeros(len(signal), dtype=bool)
    for timestamp in speech_timestamps:
        start = int(max(0, timestamp["start"]))
        stop = int(min(len(signal), timestamp["end"]))
        if stop > start:
            sample_mask[start:stop] = True

    if len(signal) == 0:
        return (
            pd.DataFrame(columns=LEGACY_SILERO_FRAME_COLUMNS),
            pd.DataFrame(columns=LEGACY_SILERO_SEGMENT_COLUMNS),
        )

    n_frames = int(np.ceil(len(signal) / frame_len))
    total_len = n_frames * frame_len
    padded_signal = np.pad(signal, (0, total_len - len(signal)), mode="constant")
    padded_mask = np.pad(sample_mask, (0, total_len - len(sample_mask)), mode="constant")

    frame_mask = []
    midpoints = []
    rms = []
    rms_db = []
    for start in range(0, total_len, frame_len):
        audio_chunk = padded_signal[start : start + frame_len]
        mask_chunk = padded_mask[start : start + frame_len]
        value = float(np.sqrt(np.mean(audio_chunk**2))) if len(audio_chunk) else 0.0
        frame_mask.append(float(np.mean(mask_chunk)) > 0.5)
        midpoints.append((start + frame_len / 2.0) / sample_rate)
        rms.append(value)
        rms_db.append(20.0 * np.log10(max(value, 1e-10)))

    frame_mask_array = np.asarray(frame_mask, dtype=bool)
    frames = pd.DataFrame(
        {
            "frame_idx": np.arange(len(frame_mask_array), dtype=int),
            "mid_sec": np.asarray(midpoints),
            "rms": np.asarray(rms),
            "rms_db": np.asarray(rms_db),
            "speech_vad_raw": frame_mask_array,
            "speech_vad_smooth": frame_mask_array,
            "speech_mask_strict": frame_mask_array,
            "nonspeech_mask_strict": ~frame_mask_array,
            "threshold": threshold,
            "frame_ms": frame_ms,
        },
        columns=LEGACY_SILERO_FRAME_COLUMNS,
    )

    segment_rows = []
    frame_hop_sec = frame_len / sample_rate
    for value, start, stop in _boolean_runs(frame_mask_array):
        start_sec = start * frame_hop_sec
        end_sec = stop * frame_hop_sec
        segment_rows.append(
            {
                "segment_type": "speech" if value else "nonspeech",
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": float(end_sec - start_sec),
                "run_start_frame": int(start),
                "run_end_frame": int(stop),
            }
        )
    segments = pd.DataFrame(segment_rows)
    if segments.empty:
        segments = pd.DataFrame(columns=LEGACY_SILERO_SEGMENT_COLUMNS)
    else:
        roles = []
        for index, row in segments.iterrows():
            if row["segment_type"] == "speech":
                roles.append("speech")
            elif index == segments.index.min():
                roles.append("leading_nonspeech")
            elif index == segments.index.max():
                roles.append("trailing_nonspeech")
            else:
                roles.append("internal_nonspeech")
        segments["segment_role"] = roles
        segments = segments[LEGACY_SILERO_SEGMENT_COLUMNS]
    return frames, segments


def summarize_legacy_silero_artifacts(
    waveform: np.ndarray,
    sample_rate: int,
    frames: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    threshold: float,
    frame_ms: int,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> dict[str, object]:
    """Reproduce the original recording-level Silero summary quantities."""
    signal = np.asarray(waveform, dtype=float).reshape(-1)
    speech = segments["segment_type"].eq("speech") if not segments.empty else pd.Series(dtype=bool)
    internal = (
        segments["segment_role"].eq("internal_nonspeech")
        if not segments.empty
        else pd.Series(dtype=bool)
    )

    def role_duration(role: str, reducer: str = "first") -> float:
        if segments.empty:
            return 0.0
        values = segments.loc[segments["segment_role"].eq(role), "duration_sec"]
        if values.empty:
            return 0.0
        return float(values.max() if reducer == "max" else values.iloc[0])

    return {
        "method": "silero_vad",
        "duration_sec": float(len(signal) / sample_rate) if sample_rate > 0 else np.nan,
        "sample_rate_analysis": int(sample_rate),
        "threshold": float(threshold),
        "frame_ms": int(frame_ms),
        "min_speech_duration_ms": int(min_speech_ms),
        "min_silence_duration_ms": int(min_silence_ms),
        "speech_pad_ms": int(speech_pad_ms),
        "n_frames": int(len(frames)),
        "n_segments_total": int(len(segments)),
        "n_speech_segments": int(speech.sum()),
        "n_internal_nonspeech_segments": int(internal.sum()),
        "speech_fraction": (
            float(frames["speech_vad_smooth"].astype(bool).mean()) if not frames.empty else np.nan
        ),
        "leading_nonspeech_sec": role_duration("leading_nonspeech"),
        "trailing_nonspeech_sec": role_duration("trailing_nonspeech"),
        "longest_internal_nonspeech_sec": role_duration("internal_nonspeech", reducer="max"),
        "rms_db_median": (float(np.median(frames["rms_db"])) if not frames.empty else np.nan),
        "rms_db_std": (float(np.std(frames["rms_db"])) if not frames.empty else np.nan),
    }


def intervals_to_frame(views: dict[str, list[Interval]], file_name: str) -> pd.DataFrame:
    rows = []
    for view, intervals in views.items():
        for index, item in enumerate(intervals):
            rows.append(
                {
                    "file_name": file_name,
                    "view": view,
                    "interval_index": index,
                    "start_sec": item.start_sec,
                    "end_sec": item.end_sec,
                    "duration_sec": item.duration_sec,
                }
            )
    return pd.DataFrame(rows)


def segmentation_frame_diagnostics(
    waveform: np.ndarray,
    sample_rate: int,
    views: dict[str, list[Interval]],
    *,
    frame_ms: float = 30,
    hop_ms: float = 10,
) -> pd.DataFrame:
    """Compute auditable frame RMS and interval-membership traces for plotting/QC."""
    signal = np.asarray(waveform, dtype=float).reshape(-1)
    frame = max(1, int(round(sample_rate * frame_ms / 1000)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000)))
    if signal.size == 0:
        return pd.DataFrame(
            columns=[
                "start_sec",
                "end_sec",
                "mid_sec",
                "rms_dbfs",
                "raw_speech",
                "primary_speech",
                "strict_speech",
                "strict_internal_nonspeech",
            ]
        )
    starts = np.arange(0, max(1, signal.size - frame + 1), hop, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)
    rows = []
    for start in starts:
        stop = min(signal.size, start + frame)
        chunk = signal[start:stop]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        rows.append(
            {
                "start_sec": start / sample_rate,
                "end_sec": stop / sample_rate,
                "mid_sec": (start + stop) / (2 * sample_rate),
                "rms_dbfs": 20 * np.log10(max(rms, 1e-12)),
            }
        )
    output = pd.DataFrame(rows)
    midpoints = output["mid_sec"].to_numpy()
    for view in [
        "raw_speech",
        "primary_speech",
        "strict_speech",
        "strict_internal_nonspeech",
    ]:
        output[view] = frame_membership(midpoints, views.get(view, []))
    return output


def summarize_segmentation(
    waveform: np.ndarray,
    sample_rate: int,
    views: dict[str, list[Interval]],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarize the primary Silero view using the original reading-task QC quantities."""
    signal = np.asarray(waveform, dtype=float).reshape(-1)
    duration = signal.size / sample_rate if sample_rate > 0 else 0.0
    primary = views.get("primary_speech", [])
    internal = internal_nonspeech(primary, duration)
    frames = segmentation_frame_diagnostics(signal, sample_rate, views)
    speech_duration = float(sum(interval.duration_sec for interval in primary))
    summary = {
        "duration_sec": float(duration),
        "speech_duration_sec": speech_duration,
        "speech_fraction": speech_duration / duration if duration > 0 else np.nan,
        "n_speech_segments": int(len(primary)),
        "n_internal_nonspeech_segments": int(len(internal)),
        "longest_internal_nonspeech_sec": float(
            max((interval.duration_sec for interval in internal), default=0.0)
        ),
        "rms_db_median": float(frames["rms_dbfs"].median()) if not frames.empty else np.nan,
        "rms_db_std": float(frames["rms_dbfs"].std(ddof=0)) if not frames.empty else np.nan,
    }
    return summary, frames


def classify_reading_segmentation(summary: dict[str, object]) -> dict[str, str]:
    """Preserve the original ALS-reading triage: hard failures vs soft review flags."""
    flags: list[str] = []
    speech_fraction = pd.to_numeric(
        pd.Series([summary.get("speech_fraction")]), errors="coerce"
    ).iloc[0]
    duration = pd.to_numeric(pd.Series([summary.get("duration_sec")]), errors="coerce").iloc[0]
    longest_pause = pd.to_numeric(
        pd.Series([summary.get("longest_internal_nonspeech_sec")]), errors="coerce"
    ).iloc[0]
    rms_median = pd.to_numeric(pd.Series([summary.get("rms_db_median")]), errors="coerce").iloc[0]
    rms_std = pd.to_numeric(pd.Series([summary.get("rms_db_std")]), errors="coerce").iloc[0]
    n_segments = int(summary.get("n_speech_segments", 0) or 0)

    if n_segments == 0:
        flags.append("no_speech_detected")
    if pd.notna(speech_fraction) and speech_fraction < 0.05:
        flags.append("very_low_speech_fraction")
    if pd.notna(duration) and duration < 1.0:
        flags.append("very_short_file")
    if pd.notna(rms_median) and pd.notna(rms_std) and rms_median < -60 and rms_std < 3:
        flags.append("near_silent_or_noise_only")
    if n_segments > 25:
        flags.append("extreme_fragmentation")
    if pd.notna(longest_pause) and longest_pause > 5.0:
        flags.append("extreme_internal_pause")

    hard = {
        "no_speech_detected",
        "very_low_speech_fraction",
        "very_short_file",
        "near_silent_or_noise_only",
    }
    soft = {"extreme_fragmentation", "extreme_internal_pause"}
    if hard.intersection(flags):
        status = "excluded"
    elif soft.intersection(flags):
        status = "flagged"
    else:
        status = "accepted"
    return {"qc_status": status, "qc_flags": ";".join(flags)}


def plot_segmentation_diagnostic(
    waveform: np.ndarray,
    sample_rate: int,
    frames: pd.DataFrame,
    segments: pd.DataFrame,
    summary: dict[str, object],
    *,
    file_name: str,
    save_path: str | Path,
    show: bool = False,
):
    """Save the original pipeline's four-panel Silero diagnostic figure."""
    import matplotlib.pyplot as plt

    signal = np.asarray(waveform, dtype=float).reshape(-1)
    time = np.arange(signal.size) / sample_rate

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.5, 1.2, 1.2]},
    )
    ax_wave, ax_rms, ax_mask, ax_segments = axes
    fig.suptitle(
        f"{file_name}\n"
        f"duration={summary.get('duration_sec', np.nan):.2f}s | "
        f"speech_fraction={summary.get('speech_fraction', np.nan):.3f} | "
        f"n_speech_segments={summary.get('n_speech_segments', 0)}",
        y=0.98,
        fontsize=11,
    )
    if signal.size > 0:
        ax_wave.plot(time, signal, linewidth=0.5)
    ax_wave.axhline(0, linewidth=0.6, alpha=0.5)
    ax_wave.set_ylabel("Amplitude")

    if not frames.empty:
        ax_rms.plot(
            frames["mid_sec"],
            frames["rms_db"],
            linewidth=0.8,
            label="RMS dB",
        )
        ax_rms.legend(loc="upper right", fontsize=8)
    ax_rms.set_ylabel("RMS dB")

    if not frames.empty:
        ax_mask.step(
            frames["mid_sec"],
            frames["speech_vad_smooth"].astype(int),
            where="mid",
            label="Silero speech mask",
        )
        ax_mask.legend(loc="upper right", fontsize=8)
    ax_mask.set_yticks([])
    ax_mask.set_ylabel("Masks")

    role_colors = {
        "speech": "#66bb6a",
        "leading_nonspeech": "#9e9e9e",
        "internal_nonspeech": "#ffa726",
        "trailing_nonspeech": "#ab47bc",
    }
    used_labels: set[str] = set()
    if not segments.empty:
        for _, segment in segments.iterrows():
            role = str(segment["segment_role"])
            label = role if role not in used_labels else None
            used_labels.add(role)
            ymin, ymax = (0.5, 1.0) if role == "speech" else (0.0, 0.5)
            ax_segments.axvspan(
                segment["start_sec"],
                segment["end_sec"],
                ymin=ymin,
                ymax=ymax,
                alpha=0.35,
                color=role_colors.get(role, "#bdbdbd"),
                label=label,
            )
        for boundary_sec in segments["end_sec"].iloc[:-1]:
            ax_segments.axvline(
                boundary_sec,
                linestyle="--",
                linewidth=0.5,
                alpha=0.4,
                color="black",
            )
    ax_segments.set_ylim(0, 2)
    ax_segments.set_yticks([0.5, 1.5])
    ax_segments.set_yticklabels(["non-speech", "speech"])
    ax_segments.set_xlabel("Time (s)")
    ax_segments.set_ylabel("Segments")
    if used_labels:
        ax_segments.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def plot_boundary_alignment_audit(
    waveform: np.ndarray,
    sample_rate: int,
    exact_speech: Iterable[Interval],
    boundary_audit: pd.DataFrame,
    *,
    file_name: str,
    save_path: str | Path,
    minimum_contrast_db: float = 3.0,
    show: bool = False,
):
    """Plot sample-index boundaries, frame-display deltas, and local edge evidence."""
    import matplotlib.pyplot as plt

    signal = np.asarray(waveform, dtype=float).reshape(-1)
    intervals = list(exact_speech)
    time = np.arange(signal.size) / sample_rate
    frame_table = segmentation_frame_diagnostics(
        signal,
        sample_rate,
        {
            "raw_speech": intervals,
            "primary_speech": intervals,
            "strict_speech": [],
            "strict_internal_nonspeech": [],
        },
    )
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [2.0, 1.4, 1.1, 1.1]},
    )
    ax_wave, ax_rms, ax_delta, ax_contrast = axes
    fig.suptitle(
        f"{file_name}\n"
        "sample-index analysis boundaries; energy contrast is a review flag, not an "
        "automatic boundary correction",
        y=0.99,
        fontsize=11,
    )
    ax_wave.plot(time, signal, linewidth=0.5, color="#2b7bba")
    ax_wave.axhline(0, linewidth=0.5, color="black", alpha=0.4)
    for index, interval in enumerate(intervals):
        ax_wave.axvspan(
            interval.start_sec,
            interval.end_sec,
            color="#66bb6a",
            alpha=0.18,
            label="exact primary speech" if index == 0 else None,
        )
        ax_wave.axvline(interval.start_sec, color="#1b7837", linewidth=0.8)
        ax_wave.axvline(interval.end_sec, color="#b2182b", linewidth=0.8)
    ax_wave.set_ylabel("Amplitude")
    if intervals:
        ax_wave.legend(loc="upper right", fontsize=8)

    if not frame_table.empty:
        ax_rms.plot(
            frame_table["mid_sec"],
            frame_table["rms_dbfs"],
            linewidth=0.8,
            color="#2b7bba",
        )
    for interval in intervals:
        ax_rms.axvline(interval.start_sec, color="#1b7837", linewidth=0.6, alpha=0.8)
        ax_rms.axvline(interval.end_sec, color="#b2182b", linewidth=0.6, alpha=0.8)
    ax_rms.set_ylabel("RMS dBFS")

    if not boundary_audit.empty:
        positions = boundary_audit["segment_index"].to_numpy(dtype=float)
        ax_delta.axhline(0, color="black", linewidth=0.7)
        ax_delta.scatter(
            positions - 0.08,
            boundary_audit["display_onset_delta_ms"],
            s=24,
            label="displayed onset − exact onset",
            color="#1b7837",
        )
        ax_delta.scatter(
            positions + 0.08,
            boundary_audit["display_offset_delta_ms"],
            s=24,
            label="displayed offset − exact offset",
            color="#b2182b",
        )
        ax_delta.legend(loc="upper right", fontsize=8)
    ax_delta.set_ylabel("30-ms display\ndelta (ms)")

    if not boundary_audit.empty:
        positions = boundary_audit["segment_index"].to_numpy(dtype=float)
        ax_contrast.axhline(
            minimum_contrast_db,
            color="#cc8a00",
            linestyle="--",
            linewidth=0.8,
            label=f"review threshold ({minimum_contrast_db:g} dB)",
        )
        ax_contrast.scatter(
            positions - 0.08,
            boundary_audit["onset_contrast_db"],
            s=24,
            label="onset local contrast",
            color="#1b7837",
        )
        ax_contrast.scatter(
            positions + 0.08,
            boundary_audit["offset_contrast_db"],
            s=24,
            label="offset local contrast",
            color="#b2182b",
        )
        ax_contrast.legend(loc="upper right", fontsize=8)
    ax_contrast.set_ylabel("Inside − outside\nRMS (dB)")
    ax_contrast.set_xlabel("Speech segment index")

    for axis in axes:
        axis.grid(alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def plot_segmentation_failure(
    *,
    file_name: str,
    error: str,
    save_path: str | Path,
) -> None:
    """Save an excluded placeholder so every failed recording remains visually auditable."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    ax.text(0.02, 0.80, file_name, fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.02,
        0.55,
        "SEGMENTATION EXCLUDED — processing failure",
        color="#B22222",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(0.02, 0.30, error, fontsize=9, wrap=True, transform=ax.transAxes)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


SEGMENTATION_ADJUDICATION_COLUMNS = [
    "logical_recording_id",
    "file_name",
    "automatic_qc_status",
    "qc_flags",
    "task_completed_as_instructed",
    "automatic_task_exclusion",
    "automatic_exclusion_reason",
    "accepted_outlier",
    "accepted_outlier_max_abs_robust_z",
    "review_required",
    "review_reasons",
    "decision",
    "boundary_source",
    "reviewer",
    "review_date",
    "notes",
]

MANUAL_SEGMENTATION_COLUMNS = [
    "logical_recording_id",
    "file_name",
    "segment_index",
    "start_sec",
    "end_sec",
    "reviewer",
    "review_date",
    "notes",
]

DEFAULT_SEGMENTATION_OUTLIER_FEATURES = [
    "duration_sec",
    "speech_fraction",
    "n_speech_segments",
    "n_internal_nonspeech_segments",
    "leading_nonspeech_sec",
    "trailing_nonspeech_sec",
    "longest_internal_nonspeech_sec",
    "rms_db_median",
    "rms_db_std",
]

DEFAULT_SEGMENTATION_ACCEPTED_GUARDRAILS = {
    "speech_fraction_below": 0.10,
    "n_speech_segments_at_least": 20,
    "longest_internal_nonspeech_sec_at_least": 4.0,
    "boundary_low_contrast_fraction_at_least": 0.50,
    "duration_sec_below": 2.0,
}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"true", "1", "yes", "y"}
        )
    ).fillna(False)


def segmentation_review_selection(
    summary: pd.DataFrame,
    review_config: dict | None = None,
) -> pd.DataFrame:
    """Create a diagnosis/outcome-independent mandatory review queue.

    Every automatically flagged/excluded recording is selected. Accepted recordings
    are selected if a segmentation-only guardrail is crossed or any prespecified
    summary quantity has an absolute median/MAD robust z score at or above the
    configured threshold.
    """
    required = {"logical_recording_id", "file_name", "qc_status", "qc_flags"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Segmentation summary is missing columns: {sorted(missing)}")
    config = review_config or {}
    threshold = float(config.get("robust_z_threshold", 4.5))
    minimum_n = int(config.get("minimum_accepted_reference_n", 20))
    features = list(config.get("accepted_outlier_features", DEFAULT_SEGMENTATION_OUTLIER_FEATURES))
    guardrails = config.get("accepted_guardrails", DEFAULT_SEGMENTATION_ACCEPTED_GUARDRAILS)

    result = summary[["logical_recording_id", "file_name", "qc_status", "qc_flags"]].copy()
    if "Task Completed as Instructed" in summary.columns:
        task_completed = summary["Task Completed as Instructed"]
    elif "task_completed_as_instructed" in summary.columns:
        task_completed = summary["task_completed_as_instructed"]
    else:
        task_completed = pd.Series("", index=summary.index, dtype=object)
    result["task_completed_as_instructed"] = task_completed.fillna("").astype(str).str.strip()
    result["automatic_task_exclusion"] = (
        result["task_completed_as_instructed"].str.upper().eq("NO")
    )
    result["automatic_exclusion_reason"] = np.where(
        result["automatic_task_exclusion"],
        "Frozen metadata gate: Task Completed as Instructed = NO",
        "",
    )
    result["accepted_outlier"] = False
    result["accepted_outlier_max_abs_robust_z"] = np.nan
    reasons: dict[int, list[str]] = {index: [] for index in result.index}
    accepted = summary["qc_status"].astype(str).str.lower().eq("accepted")
    task_excluded = result["automatic_task_exclusion"].astype(bool)

    for index in summary.index[task_excluded]:
        reasons[index].append("metadata_gate:task_completed_as_instructed=NO")

    robust_z_by_feature: dict[str, pd.Series] = {}
    for feature in features:
        if feature not in summary.columns:
            continue
        values = pd.to_numeric(summary[feature], errors="coerce")
        reference = values.loc[accepted & values.notna()]
        if len(reference) < minimum_n:
            continue
        median = float(reference.median())
        mad = float(np.median(np.abs(reference.to_numpy() - median)))
        if not np.isfinite(mad) or mad <= 0:
            continue
        robust_z = 0.6744897501960817 * (values - median) / mad
        robust_z_by_feature[feature] = robust_z
        selected = accepted & ~task_excluded & robust_z.abs().ge(threshold)
        for index in summary.index[selected]:
            reasons[index].append(f"accepted_robust_outlier:{feature}:z={robust_z.loc[index]:.2f}")

    if robust_z_by_feature:
        robust_z_frame = pd.DataFrame(robust_z_by_feature)
        result["accepted_outlier_max_abs_robust_z"] = robust_z_frame.abs().max(axis=1)

    guardrail_specs = [
        (
            "speech_fraction",
            "speech_fraction_below",
            lambda values, cutoff: values.lt(cutoff),
        ),
        (
            "n_speech_segments",
            "n_speech_segments_at_least",
            lambda values, cutoff: values.ge(cutoff),
        ),
        (
            "longest_internal_nonspeech_sec",
            "longest_internal_nonspeech_sec_at_least",
            lambda values, cutoff: values.ge(cutoff),
        ),
        (
            "boundary_low_contrast_fraction",
            "boundary_low_contrast_fraction_at_least",
            lambda values, cutoff: values.ge(cutoff),
        ),
        (
            "duration_sec",
            "duration_sec_below",
            lambda values, cutoff: values.lt(cutoff),
        ),
    ]
    for feature, key, comparator in guardrail_specs:
        if feature not in summary.columns or key not in guardrails:
            continue
        cutoff = float(guardrails[key])
        values = pd.to_numeric(summary[feature], errors="coerce")
        selected = accepted & ~task_excluded & values.notna() & comparator(values, cutoff)
        for index in summary.index[selected]:
            reasons[index].append(f"accepted_guardrail:{feature}:{values.loc[index]:.4g}")

    nonaccepted = ~accepted
    for index in summary.index[nonaccepted]:
        status = str(summary.loc[index, "qc_status"]).lower()
        raw_flags = summary.loc[index, "qc_flags"]
        flags = "" if pd.isna(raw_flags) else str(raw_flags).strip()
        reasons[index].append(f"automatic_status:{status}")
        if flags:
            reasons[index].append(f"automatic_flags:{flags}")

    result["accepted_outlier"] = [
        bool(accepted.loc[index] and not task_excluded.loc[index] and reasons[index])
        for index in result.index
    ]
    result["review_required"] = [
        bool(reasons[index]) and not bool(task_excluded.loc[index])
        for index in result.index
    ]
    result["review_reasons"] = [";".join(reasons[index]) for index in result.index]
    return result


def segmentation_adjudication_template(
    summary: pd.DataFrame,
    review_config: dict | None = None,
) -> pd.DataFrame:
    """Create the complete review sheet and prefill only non-review accepted rows."""
    template = segmentation_review_selection(summary, review_config).rename(
        columns={"qc_status": "automatic_qc_status"}
    )
    system_excluded = _bool_series(template["automatic_task_exclusion"])
    template["decision"] = np.select(
        [system_excluded, template["review_required"]],
        ["EXCLUDE", ""],
        default="KEEP",
    )
    template["boundary_source"] = np.select(
        [system_excluded, template["review_required"]],
        ["NONE", ""],
        default="AUTO",
    )
    template["reviewer"] = ""
    template["review_date"] = ""
    template["notes"] = np.where(
        system_excluded,
        template["automatic_exclusion_reason"],
        "",
    )
    return template[SEGMENTATION_ADJUDICATION_COLUMNS].sort_values(
        ["automatic_qc_status", "file_name"]
    )


def segmentation_pending_reviews(adjudication: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are not yet complete enough to freeze."""
    missing_columns = [
        column for column in SEGMENTATION_ADJUDICATION_COLUMNS if column not in adjudication.columns
    ]
    if missing_columns:
        raise ValueError(f"Segmentation adjudication is missing columns: {missing_columns}")
    work = adjudication[SEGMENTATION_ADJUDICATION_COLUMNS].copy()
    decision = work["decision"].astype(str).str.strip().str.upper()
    source = work["boundary_source"].astype(str).str.strip().str.upper()
    review_required = _bool_series(work["review_required"])
    automatic_task_exclusion = _bool_series(work["automatic_task_exclusion"])
    reviewer_missing = work["reviewer"].astype(str).str.strip().eq("")
    date_missing = work["review_date"].astype(str).str.strip().eq("")
    notes_missing = work["notes"].astype(str).str.strip().eq("")
    source_invalid = (decision.eq("KEEP") & ~source.isin(["AUTO", "MANUAL"])) | (
        decision.eq("EXCLUDE") & source.ne("NONE")
    )
    locked_task_exclusion_invalid = automatic_task_exclusion & ~(
        decision.eq("EXCLUDE") & source.eq("NONE")
    )
    manual_review = ~automatic_task_exclusion & (
        review_required | decision.eq("EXCLUDE") | source.eq("MANUAL")
    )
    pending = (
        ~decision.isin(["KEEP", "EXCLUDE"])
        | source_invalid
        | locked_task_exclusion_invalid
        | (manual_review & reviewer_missing)
        | (manual_review & date_missing)
        | (
            ~automatic_task_exclusion
            & (decision.eq("EXCLUDE") | source.eq("MANUAL"))
            & notes_missing
        )
        | (automatic_task_exclusion & notes_missing)
    )
    return work.loc[pending].copy()


def apply_segmentation_adjudication(
    summary: pd.DataFrame,
    adjudication: pd.DataFrame,
    review_config: dict | None = None,
) -> pd.DataFrame:
    """Validate eligibility and boundary-source decisions for the segmentation freeze."""
    missing_columns = [
        column for column in SEGMENTATION_ADJUDICATION_COLUMNS if column not in adjudication.columns
    ]
    if missing_columns:
        raise ValueError(f"Segmentation adjudication is missing columns: {missing_columns}")
    work = adjudication[SEGMENTATION_ADJUDICATION_COLUMNS].copy()
    work["logical_recording_id"] = work["logical_recording_id"].astype(str)
    work["decision"] = work["decision"].astype(str).str.strip().str.upper()
    work["boundary_source"] = work["boundary_source"].astype(str).str.strip().str.upper()
    work["review_required"] = _bool_series(work["review_required"])
    work["accepted_outlier"] = _bool_series(work["accepted_outlier"])
    work["automatic_task_exclusion"] = _bool_series(
        work["automatic_task_exclusion"]
    )
    duplicated = work["logical_recording_id"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "Duplicate segmentation adjudications: "
            f"{sorted(work.loc[duplicated, 'logical_recording_id'].unique())}"
        )
    expected = set(summary["logical_recording_id"].astype(str))
    observed = set(work["logical_recording_id"])
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"Stale segmentation adjudication; missing={missing}, extra={extra}")

    expected_review = segmentation_review_selection(summary, review_config)
    expected_review["logical_recording_id"] = expected_review["logical_recording_id"].astype(str)
    expected_by_id = expected_review.set_index("logical_recording_id")
    work_by_id = work.set_index("logical_recording_id")
    review_mismatch = (
        work_by_id["review_required"] != expected_by_id.loc[work_by_id.index, "review_required"]
    )
    outlier_mismatch = (
        work_by_id["accepted_outlier"] != expected_by_id.loc[work_by_id.index, "accepted_outlier"]
    )
    status_mismatch = work_by_id["automatic_qc_status"].astype(str) != expected_by_id.loc[
        work_by_id.index, "qc_status"
    ].astype(str)
    flags_mismatch = work_by_id["qc_flags"].fillna("").astype(str) != expected_by_id.loc[
        work_by_id.index, "qc_flags"
    ].fillna("").astype(str)
    reasons_mismatch = work_by_id["review_reasons"].fillna("").astype(str) != expected_by_id.loc[
        work_by_id.index, "review_reasons"
    ].fillna("").astype(str)
    task_value_mismatch = (
        work_by_id["task_completed_as_instructed"].fillna("").astype(str)
        != expected_by_id.loc[
            work_by_id.index, "task_completed_as_instructed"
        ].fillna("").astype(str)
    )
    task_gate_mismatch = (
        work_by_id["automatic_task_exclusion"]
        != _bool_series(
            expected_by_id.loc[work_by_id.index, "automatic_task_exclusion"]
        )
    )
    task_reason_mismatch = (
        work_by_id["automatic_exclusion_reason"].fillna("").astype(str)
        != expected_by_id.loc[
            work_by_id.index, "automatic_exclusion_reason"
        ].fillna("").astype(str)
    )
    if (
        review_mismatch.any()
        or outlier_mismatch.any()
        or status_mismatch.any()
        or flags_mismatch.any()
        or reasons_mismatch.any()
        or task_value_mismatch.any()
        or task_gate_mismatch.any()
        or task_reason_mismatch.any()
    ):
        affected = sorted(
            set(work_by_id.index[review_mismatch])
            | set(work_by_id.index[outlier_mismatch])
            | set(work_by_id.index[status_mismatch])
            | set(work_by_id.index[flags_mismatch])
            | set(work_by_id.index[reasons_mismatch])
            | set(work_by_id.index[task_value_mismatch])
            | set(work_by_id.index[task_gate_mismatch])
            | set(work_by_id.index[task_reason_mismatch])
        )
        raise ValueError(
            "Segmentation review-selection fields were changed or are stale. "
            f"Regenerate the review template. Affected IDs: {affected}"
        )

    invalid = ~work["decision"].isin(["KEEP", "EXCLUDE"])
    if invalid.any():
        rows = work.loc[
            invalid,
            ["file_name", "automatic_qc_status", "qc_flags", "decision"],
        ].to_dict("records")
        raise ValueError(f"Every segmentation row must be KEEP or EXCLUDE. Unresolved rows: {rows}")

    keep_source_invalid = work["decision"].eq("KEEP") & ~work["boundary_source"].isin(
        ["AUTO", "MANUAL"]
    )
    exclude_source_invalid = work["decision"].eq("EXCLUDE") & work["boundary_source"].ne("NONE")
    if keep_source_invalid.any() or exclude_source_invalid.any():
        affected = work.loc[
            keep_source_invalid | exclude_source_invalid,
            ["file_name", "decision", "boundary_source"],
        ].to_dict("records")
        raise ValueError(
            "KEEP requires boundary_source AUTO or MANUAL; EXCLUDE requires NONE. "
            f"Invalid rows: {affected}"
        )

    task_gate_invalid = work["automatic_task_exclusion"] & ~(
        work["decision"].eq("EXCLUDE") & work["boundary_source"].eq("NONE")
    )
    if task_gate_invalid.any():
        files = sorted(work.loc[task_gate_invalid, "file_name"].astype(str))
        raise ValueError(
            "Rows with Task Completed as Instructed = NO are locked to "
            f"EXCLUDE + NONE. Invalid files: {files}"
        )

    requires_reviewer = (
        ~work["automatic_task_exclusion"]
        & (
            work["review_required"]
            | work["decision"].eq("EXCLUDE")
            | work["boundary_source"].eq("MANUAL")
        )
    )
    missing_reviewer = requires_reviewer & work["reviewer"].astype(str).str.strip().eq("")
    if missing_reviewer.any():
        files = sorted(work.loc[missing_reviewer, "file_name"].astype(str))
        raise ValueError(f"Required segmentation reviews need a reviewer: {files}")
    missing_date = requires_reviewer & work["review_date"].astype(str).str.strip().eq("")
    if missing_date.any():
        files = sorted(work.loc[missing_date, "file_name"].astype(str))
        raise ValueError(f"Required segmentation reviews need a review date: {files}")
    needs_reason = (
        ~work["automatic_task_exclusion"]
        & (work["decision"].eq("EXCLUDE") | work["boundary_source"].eq("MANUAL"))
    ) | work["automatic_task_exclusion"]
    missing_reason = needs_reason & work["notes"].astype(str).str.strip().eq("")
    if missing_reason.any():
        files = sorted(work.loc[missing_reason, "file_name"].astype(str))
        raise ValueError(f"Exclusions and manual boundaries require notes: {files}")

    result = summary.merge(
        work,
        on=["logical_recording_id", "file_name"],
        how="left",
        validate="one_to_one",
    )
    result["segmentation_analysis_eligible"] = result["decision"].eq("KEEP")
    result["segmentation_decision_source"] = np.select(
        [
            result["automatic_task_exclusion"],
            result["decision"].eq("EXCLUDE"),
            result["boundary_source"].eq("MANUAL"),
            result["review_required"],
        ],
        [
            "automatic_task_not_performed",
            "manual_exclusion",
            "manual_boundaries",
            "reviewed_auto_boundaries",
        ],
        default="automatic_accepted",
    )
    return result


def validate_manual_segmentation_overrides(
    overrides: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Validate complete, ordered, non-overlapping manual speech intervals."""
    manual_decisions = decisions.loc[
        decisions["decision"].eq("KEEP") & decisions["boundary_source"].eq("MANUAL")
    ].copy()
    expected_ids = set(manual_decisions["logical_recording_id"].astype(str))
    if overrides.empty and not expected_ids:
        return pd.DataFrame(columns=MANUAL_SEGMENTATION_COLUMNS)
    missing_columns = [
        column for column in MANUAL_SEGMENTATION_COLUMNS if column not in overrides.columns
    ]
    if missing_columns:
        raise ValueError(f"Manual segmentation table is missing columns: {missing_columns}")

    work = overrides[MANUAL_SEGMENTATION_COLUMNS].copy()
    work["logical_recording_id"] = work["logical_recording_id"].astype(str)
    observed_ids = set(work["logical_recording_id"])
    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            "Manual interval rows must exist only for KEEP/MANUAL recordings; "
            f"missing={missing_ids}, extra={extra_ids}"
        )

    decision_lookup = manual_decisions.set_index(
        manual_decisions["logical_recording_id"].astype(str)
    )
    work["segment_index"] = pd.to_numeric(work["segment_index"], errors="coerce")
    work["start_sec"] = pd.to_numeric(work["start_sec"], errors="coerce")
    work["end_sec"] = pd.to_numeric(work["end_sec"], errors="coerce")
    invalid_numeric = (
        work[["segment_index", "start_sec", "end_sec"]].isna().any(axis=1)
        | ~np.isfinite(work["start_sec"])
        | ~np.isfinite(work["end_sec"])
    )
    if invalid_numeric.any():
        raise ValueError(
            "Manual segmentation has non-numeric values: "
            f"{work.loc[invalid_numeric, ['file_name', 'segment_index', 'start_sec', 'end_sec']].to_dict('records')}"
        )
    non_integer = work["segment_index"].mod(1).ne(0) | work["segment_index"].lt(0)
    if non_integer.any():
        raise ValueError("Manual segment_index values must be non-negative integers.")
    work["segment_index"] = work["segment_index"].astype(int)

    validated = []
    for logical_id, group in work.groupby("logical_recording_id", sort=False):
        decision = decision_lookup.loc[logical_id]
        group = group.sort_values(["start_sec", "end_sec"]).copy()
        expected_file = str(decision["file_name"])
        if group["file_name"].astype(str).ne(expected_file).any():
            raise ValueError(f"Manual interval filename mismatch for {logical_id}")
        if group["segment_index"].duplicated().any():
            raise ValueError(f"Duplicate manual segment_index for {expected_file}")
        if group["segment_index"].tolist() != list(range(len(group))):
            raise ValueError(
                f"Manual segment_index must be sequential from zero for {expected_file}"
            )
        if group["start_sec"].lt(0).any() or group["end_sec"].le(group["start_sec"]).any():
            raise ValueError(f"Invalid manual interval duration for {expected_file}")
        duration = float(decision["duration_sec"])
        if group["end_sec"].gt(duration + 1e-6).any():
            raise ValueError(
                f"Manual interval exceeds {duration:.6f}s duration for {expected_file}"
            )
        starts = group["start_sec"].to_numpy()
        ends = group["end_sec"].to_numpy()
        if len(group) > 1 and np.any(starts[1:] < ends[:-1] - 1e-9):
            raise ValueError(f"Overlapping manual speech intervals for {expected_file}")
        reviewer = str(decision["reviewer"]).strip()
        if group["reviewer"].astype(str).str.strip().ne(reviewer).any():
            raise ValueError(f"Manual interval reviewer mismatch for {expected_file}")
        validated.append(group)
    return pd.concat(validated, ignore_index=True)[MANUAL_SEGMENTATION_COLUMNS]


def freeze_segmentation_intervals(
    intervals: pd.DataFrame,
    decisions: pd.DataFrame,
    manual_overrides: pd.DataFrame,
    *,
    strict_speech_edge_ms: float = 50,
    strict_nonspeech_edge_ms: float = 200,
) -> pd.DataFrame:
    """Apply manual primary-boundary overrides and attach frozen decision provenance."""
    required = {
        "file_name",
        "logical_recording_id",
        "profile",
        "view",
        "start_sec",
        "end_sec",
        "duration_sec",
    }
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"Segmentation intervals are missing columns: {sorted(missing)}")
    overrides = validate_manual_segmentation_overrides(manual_overrides, decisions)
    frozen = intervals.copy()
    frozen["logical_recording_id"] = frozen["logical_recording_id"].astype(str)
    frozen["segmentation_boundary_source"] = "automatic_silero"

    manual_decisions = decisions.loc[
        decisions["decision"].eq("KEEP") & decisions["boundary_source"].eq("MANUAL")
    ]
    replacement_frames = []
    for decision in manual_decisions.itertuples():
        logical_id = str(decision.logical_recording_id)
        manual_rows = overrides.loc[overrides["logical_recording_id"].eq(logical_id)].sort_values(
            "segment_index"
        )
        manual_speech = [
            Interval(float(row.start_sec), float(row.end_sec)) for row in manual_rows.itertuples()
        ]
        primary = normalize_intervals(manual_speech, float(decision.duration_sec))
        manual_views = {
            "raw_speech": primary,
            "primary_speech": primary,
            "strict_speech": erode_intervals(primary, strict_speech_edge_ms / 1000),
            "strict_internal_nonspeech": erode_intervals(
                internal_nonspeech(primary, float(decision.duration_sec)),
                strict_nonspeech_edge_ms / 1000,
            ),
        }
        frozen = frozen.loc[
            ~(frozen["logical_recording_id"].eq(logical_id) & frozen["profile"].eq("primary"))
        ].copy()
        replacement = intervals_to_frame(manual_views, str(decision.file_name))
        replacement["profile"] = "primary"
        replacement["logical_recording_id"] = logical_id
        replacement["segmentation_boundary_source"] = "manual_override"
        replacement_frames.append(replacement)
    if replacement_frames:
        frozen = pd.concat([frozen, *replacement_frames], ignore_index=True)

    decision_columns = [
        "logical_recording_id",
        "decision",
        "boundary_source",
        "review_required",
        "review_reasons",
        "reviewer",
        "review_date",
        "segmentation_analysis_eligible",
        "segmentation_decision_source",
    ]
    decision_frame = decisions[decision_columns].copy()
    decision_frame["logical_recording_id"] = decision_frame["logical_recording_id"].astype(str)
    frozen = frozen.merge(
        decision_frame,
        on="logical_recording_id",
        how="left",
        validate="many_to_one",
    )
    if frozen["decision"].isna().any():
        raise ValueError("Frozen intervals contain recordings without a decision.")

    eligible_ids = set(
        decisions.loc[decisions["segmentation_analysis_eligible"], "logical_recording_id"].astype(
            str
        )
    )
    primary_speech_ids = set(
        frozen.loc[
            frozen["profile"].eq("primary")
            & frozen["view"].eq("primary_speech")
            & frozen["duration_sec"].gt(0),
            "logical_recording_id",
        ].astype(str)
    )
    missing_support = sorted(eligible_ids - primary_speech_ids)
    if missing_support:
        raise ValueError(
            "KEEP recordings must have primary speech support. Use MANUAL boundaries "
            f"or EXCLUDE: {missing_support}"
        )
    return frozen.sort_values(["file_name", "profile", "view", "start_sec", "end_sec"]).reset_index(
        drop=True
    )


def frame_membership(midpoints_sec: np.ndarray, intervals: Iterable[Interval]) -> np.ndarray:
    midpoints = np.asarray(midpoints_sec, dtype=float)
    mask = np.zeros(midpoints.shape, dtype=bool)
    for item in intervals:
        mask |= (midpoints >= item.start_sec) & (midpoints < item.end_sec)
    return mask
