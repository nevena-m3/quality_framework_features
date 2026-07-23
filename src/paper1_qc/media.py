from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .provenance import sha256_file


MEDIA_EXTENSIONS = {".wav", ".webm", ".mp4", ".m4a", ".ogg", ".flac", ".mov"}


@dataclass
class AudioViews:
    native: np.ndarray  # shape: samples x channels, raw decoded scale
    sample_rate_native: int
    mono_native: np.ndarray  # no level normalization and no DC removal
    analysis_16k: np.ndarray  # mono, DC-removed, resampled for VAD/frame analyses
    probe: dict
    decode_stderr: str


def discover_media(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            files.append(path.resolve())
        elif path.is_dir():
            files.extend(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in MEDIA_EXTENSIONS
            )
    return sorted(set(files))


def probe_media(path: str | Path, ffprobe: str) -> dict:
    path = Path(path)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    result = {
        "file_path": str(path.resolve()),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size if path.exists() else np.nan,
        "probe_ok": completed.returncode == 0,
        "probe_error": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        return result

    payload = json.loads(completed.stdout)
    streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"]
    result["audio_stream_count"] = len(streams)
    if not streams:
        result["probe_ok"] = False
        result["probe_error"] = "No audio stream"
        return result

    stream = streams[0]
    fmt = payload.get("format", {})

    def numeric(value, cast=float):
        try:
            return cast(value)
        except (TypeError, ValueError):
            return np.nan

    result.update(
        {
            "codec_name": stream.get("codec_name"),
            "codec_long_name": stream.get("codec_long_name"),
            "sample_format": stream.get("sample_fmt"),
            "sample_rate_hz": numeric(stream.get("sample_rate"), int),
            "channels": numeric(stream.get("channels"), int),
            "channel_layout": stream.get("channel_layout"),
            "bits_per_raw_sample": numeric(stream.get("bits_per_raw_sample"), int),
            "stream_bit_rate": numeric(stream.get("bit_rate"), int),
            "stream_duration_sec": numeric(stream.get("duration")),
            "container_format": fmt.get("format_name"),
            "container_duration_sec": numeric(fmt.get("duration")),
            "container_bit_rate": numeric(fmt.get("bit_rate"), int),
        }
    )
    return result


def decode_audio_views(
    path: str | Path,
    *,
    ffmpeg: str,
    ffprobe: str,
    analysis_rate: int = 16000,
) -> AudioViews:
    """Decode native channels before creating a separate 16-kHz analysis view.

    No peak normalization, dynamic-range processing, clipping, or level scaling is applied.
    Quality metrics that depend on digital full scale must use ``native``/``mono_native``.
    """
    probe = probe_media(path, ffprobe)
    if not probe.get("probe_ok"):
        raise RuntimeError(f"ffprobe failed for {path}: {probe.get('probe_error')}")
    sample_rate = int(probe["sample_rate_hz"])
    channels = int(probe["channels"])
    if sample_rate <= 0 or channels <= 0:
        raise ValueError(f"Invalid native stream geometry for {path}: {sample_rate=} {channels=}")

    cmd = [
        ffmpeg,
        "-nostdin",
        "-v",
        "warning",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True)
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {stderr}")
    flat = np.frombuffer(completed.stdout, dtype="<f4")
    if flat.size == 0 or flat.size % channels:
        raise RuntimeError(f"Decoded buffer has invalid size for {path}")
    native = flat.reshape(-1, channels).astype(np.float32, copy=False)
    if not np.isfinite(native).all():
        raise RuntimeError(f"Decoded waveform contains NaN/Inf: {path}")
    mono_native = native.mean(axis=1, dtype=np.float64).astype(np.float32)
    mono_dc = mono_native - float(np.mean(mono_native, dtype=np.float64))

    if sample_rate == analysis_rate:
        analysis = mono_dc.astype(np.float32, copy=True)
    else:
        divisor = math.gcd(sample_rate, analysis_rate)
        analysis = resample_poly(
            mono_dc,
            up=analysis_rate // divisor,
            down=sample_rate // divisor,
        ).astype(np.float32)

    return AudioViews(native, sample_rate, mono_native, analysis, probe, stderr)


def build_media_inventory(
    paths: Iterable[str | Path],
    *,
    ffprobe: str,
    ffmpeg: str | None = None,
    compute_hashes: bool = True,
) -> pd.DataFrame:
    rows = []
    for path in discover_media(paths):
        row = probe_media(path, ffprobe)
        if ffmpeg is not None:
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "warning",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-f",
                    "null",
                    "-",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            row["full_decode_ok"] = completed.returncode == 0
            row["full_decode_warning"] = completed.stderr.strip()
        row["sha256"] = sha256_file(path) if compute_hashes else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def reconcile_inventory_with_metadata(
    inventory: pd.DataFrame, metadata_media_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one coverage row per expected or observed filename and duplicate-name audit."""
    if inventory.empty:
        observed = pd.DataFrame(columns=["file_name", "observed_count", "observed_paths"])
    else:
        observed = (
            inventory.groupby("file_name", as_index=False)
            .agg(observed_count=("file_path", "size"), observed_paths=("file_path", lambda x: " | ".join(sorted(x))))
        )
    expected = (
        metadata_media_rows.groupby("Raw Media File name", as_index=False)
        .size()
        .rename(columns={"Raw Media File name": "file_name", "size": "metadata_count"})
    )
    coverage = expected.merge(observed, on="file_name", how="outer", indicator=True)
    coverage["coverage_status"] = coverage["_merge"].map(
        {"left_only": "metadata_missing_on_disk", "right_only": "disk_missing_in_metadata", "both": "matched"}
    )
    coverage = coverage.drop(columns="_merge")
    duplicates = coverage.loc[coverage["observed_count"].fillna(0) > 1].copy()
    return coverage, duplicates


def audit_native_metadata_consistency(
    inventory: pd.DataFrame, metadata_media_rows: pd.DataFrame, *, duration_tolerance_sec: float = 0.5
) -> pd.DataFrame:
    """Compare native FFprobe/decode evidence with row-level platform metadata."""
    if inventory.empty:
        return pd.DataFrame(
            [{"severity": "error", "rule": "empty_media_inventory", "message": "No media files discovered."}]
        )
    joined = inventory.merge(
        metadata_media_rows,
        left_on="file_name",
        right_on="Raw Media File name",
        how="left",
        suffixes=("_native", "_metadata"),
        indicator=True,
    )
    rows = []
    for _, row in joined.iterrows():
        common = {"file_name": row.get("file_name"), "file_path": row.get("file_path")}
        if row["_merge"] == "left_only":
            rows.append(
                {**common, "severity": "error", "rule": "disk_file_missing_metadata", "message": "Observed disk file has no metadata row."}
            )
            continue
        if not bool(row.get("probe_ok", False)):
            rows.append(
                {**common, "severity": "error", "rule": "ffprobe_failed", "message": str(row.get("probe_error"))}
            )
        if "full_decode_ok" in joined.columns and not bool(row.get("full_decode_ok", False)):
            rows.append(
                {**common, "severity": "error", "rule": "full_decode_failed", "message": str(row.get("full_decode_warning"))}
            )
        elif str(row.get("full_decode_warning", "")).strip():
            rows.append(
                {**common, "severity": "review", "rule": "full_decode_warning", "message": str(row.get("full_decode_warning"))}
            )

        native_rate = pd.to_numeric(pd.Series([row.get("sample_rate_hz")]), errors="coerce").iloc[0]
        metadata_rate = pd.to_numeric(pd.Series([row.get("Sampling Rate")]), errors="coerce").iloc[0]
        if pd.notna(native_rate) and pd.notna(metadata_rate) and int(native_rate) != int(metadata_rate):
            rows.append(
                {
                    **common,
                    "severity": "error",
                    "rule": "native_metadata_sample_rate_mismatch",
                    "message": f"native={native_rate}; metadata={metadata_rate}",
                }
            )

        native_duration = pd.to_numeric(
            pd.Series([row.get("stream_duration_sec")]), errors="coerce"
        ).iloc[0]
        if pd.isna(native_duration):
            native_duration = pd.to_numeric(
                pd.Series([row.get("container_duration_sec")]), errors="coerce"
            ).iloc[0]
        metadata_duration = pd.to_numeric(pd.Series([row.get("Duration (s)")]), errors="coerce").iloc[0]
        if (
            pd.notna(native_duration)
            and pd.notna(metadata_duration)
            and abs(float(native_duration) - float(metadata_duration)) > duration_tolerance_sec
        ):
            rows.append(
                {
                    **common,
                    "severity": "review",
                    "rule": "native_metadata_duration_mismatch",
                    "message": f"native={native_duration:.3f}s; metadata={metadata_duration:.3f}s",
                }
            )
    return pd.DataFrame(rows)
