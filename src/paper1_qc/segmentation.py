from __future__ import annotations

from dataclasses import dataclass
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
    speech_pad_ms: int = 30,
    onnx: bool = True,
) -> list[Interval]:
    """Run the installed, version-pinned Silero package without ``torch.hub`` network access."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=onnx)
    waveform = torch.from_numpy(np.asarray(waveform_16k, dtype=np.float32))
    stamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )
    return [Interval(float(item["start"]), float(item["end"])) for item in stamps]


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


def frame_membership(midpoints_sec: np.ndarray, intervals: Iterable[Interval]) -> np.ndarray:
    midpoints = np.asarray(midpoints_sec, dtype=float)
    mask = np.zeros(midpoints.shape, dtype=bool)
    for item in intervals:
        mask |= (midpoints >= item.start_sec) & (midpoints < item.end_sec)
    return mask

