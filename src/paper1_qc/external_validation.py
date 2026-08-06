"""External validation and final-audit utilities for reviewed recording-quality features.

This module does not replace, rescale, or edit any reviewed feature estimator.
It provides a separate audit layer that:

1. inventories the frozen indicator contracts;
2. runs or imports external comparator measurements;
3. evaluates window/support evidence;
4. quantifies convergent and discriminant relationships; and
5. produces an evidence table for a human-signed final verdict.

External tools are comparators, not ground truth. A correlation with a perceptual
quality model cannot establish the physical cause of a no-reference waveform
observable, and disagreement does not by itself invalidate an analytically
verified estimator.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests


AUDIT_VERSION = "external-validation-audit-v0.1.0"
DEFAULT_OUTPUT_RELATIVE = Path("outputs") / "06_external_validation"
LATEST_RELEASE_RELATIVE = Path("MAIN outputs") / "02_FEATURE_LATEST"
REVIEWED_ROOT_RELATIVE = Path("MAIN outputs") / "02_FEATURE_REVIEWED"


@dataclass(frozen=True)
class ComparatorSpec:
    """Declared relationship between a reviewed indicator and an external output."""

    family_code: str
    feature: str
    comparator: str
    comparator_column: str
    relationship_class: str
    expected_direction: str
    direct_equivalence: bool
    required: bool
    rationale: str
    claim_limit: str


@dataclass(frozen=True)
class ToolAvailability:
    tool: str
    available: bool
    mode: str
    version: str | None
    detail: str


@dataclass(frozen=True)
class VerdictRule:
    """Transparent final-audit decision thresholds.

    These rules generate a candidate disposition. They never authorize automatic
    modification of a reviewed extractor.
    """

    minimum_external_n: int = 40
    minimum_absolute_spearman_for_convergence: float = 0.30
    minimum_directional_bootstrap_probability: float = 0.90
    maximum_missing_fraction_for_primary: float = 0.25
    minimum_window_stability: float = 0.80


DEFAULT_COMPARATOR_SPECS: tuple[ComparatorSpec, ...] = (
    ComparatorSpec(
        "QGAIN",
        "qgain_typical_speech_level_dbfs",
        "ffmpeg_astats",
        "ffmpeg_rms_level_db",
        "related_primitive",
        "positive",
        False,
        True,
        "Both summarize decoded digital level, but QGAIN is speech-gated and robustly aggregated.",
        "Agreement supports level measurement only; it does not identify gain control or vocal intensity.",
    ),
    ComparatorSpec(
        "QGAIN",
        "qgain_typical_speech_level_dbfs",
        "pyloudnorm",
        "pyloudnorm_integrated_lufs",
        "complementary_global_level",
        "positive",
        False,
        False,
        "LUFS and guarded speech dBFS are distinct level constructs expected to covary.",
        "LUFS includes frequency weighting and gating and is not interchangeable with dBFS.",
    ),
    ComparatorSpec(
        "QGAIN",
        "qgain_within_segment_iqr_db",
        "ffmpeg_ebur128",
        "ffmpeg_loudness_range_lu",
        "complementary_dynamics",
        "positive",
        False,
        False,
        "Both reflect level variation at different time scales and with different weighting.",
        "LRA is a programme-level loudness statistic, not a within-segment dispersion estimate.",
    ),
    ComparatorSpec(
        "QADD",
        "qadd_pause_ac_level_dbfs_median",
        "dnsmos",
        "dnsmos_bak",
        "perceptual_convergence",
        "negative",
        False,
        False,
        "Higher pause energy should tend to accompany poorer predicted background quality.",
        "DNSMOS BAK is a learned perceptual score and is not a physical noise-floor estimate.",
    ),
    ComparatorSpec(
        "QADD",
        "qadd_pause_spectral_flatness",
        "librosa",
        "librosa_spectral_flatness_median",
        "related_primitive",
        "positive",
        False,
        False,
        "Both calculate spectral flatness, but the signal regions and bands differ.",
        "Whole-recording flatness is not equivalent to guarded-pause 80-7000 Hz flatness.",
    ),
    ComparatorSpec(
        "QREV",
        "qrev_srmr_norm",
        "srmrpy",
        "srmrpy_srmr",
        "closest_algorithmic_comparator",
        "positive",
        False,
        False,
        "Both are SRMR-family modulation metrics.",
        "Implementations, normalization, segmentation, and support rules must be compared explicitly.",
    ),
    ComparatorSpec(
        "QREV",
        "qrev_tail_excess_100ms_db",
        "nisqa",
        "nisqa_coloration",
        "perceptual_convergence",
        "negative",
        False,
        False,
        "Residual smearing may reduce predicted coloration quality.",
        "NISQA coloration is perceptual and not specific to reverberation or post-offset tails.",
    ),
    ComparatorSpec(
        "QCHAN",
        "qchan_rolloff95_deficit_hz",
        "librosa",
        "librosa_rolloff95_hz",
        "related_primitive",
        "negative",
        False,
        True,
        "A larger cohort-relative rolloff deficit should accompany a lower raw rolloff estimate.",
        "The raw rolloff is not cohort-relative and remains sensitive to speech content and physiology.",
    ),
    ComparatorSpec(
        "QCHAN",
        "qchan_highband_ratio_deficit",
        "librosa",
        "librosa_highband_ratio",
        "related_primitive",
        "negative",
        False,
        True,
        "A larger reference-relative deficit should accompany a lower raw high-band ratio.",
        "The comparator does not reproduce the frozen task-matched reference construction.",
    ),
    ComparatorSpec(
        "QDIST",
        "qdist_hard_clipped_sample_fraction",
        "native_peak_audit",
        "native_near_fullscale_fraction",
        "related_sample_measure",
        "positive",
        False,
        True,
        "Both inspect native-sample occupancy near decoded full scale.",
        "Near-full-scale samples are not necessarily accepted hard-clipping plateaus.",
    ),
    ComparatorSpec(
        "QDIST",
        "qdist_hard_clip_event_rate_per_min",
        "ffmpeg_astats",
        "ffmpeg_peak_count",
        "weak_external_correspondence",
        "positive",
        False,
        False,
        "Repeated clipped plateaus may increase peak-count statistics.",
        "FFmpeg peak count is not a hard-clipping event detector.",
    ),
    ComparatorSpec(
        "QTEMP",
        "qtemp_dropout_event_rate_per_min",
        "essentia",
        "essentia_discontinuity_count",
        "algorithmic_candidate_comparator",
        "positive",
        False,
        False,
        "Both attempt to localize abrupt waveform discontinuities.",
        "Essentia discontinuities do not uniquely identify network dropouts or duplicated audio.",
    ),
)


def discover_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook, script, or current directory."""

    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root containing pyproject.toml and src/. "
        "Start JupyterLab from the repository root."
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def load_audit_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = Path(config_path) if config_path else root / "config" / "external_validation.yaml"
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        example = root / "config" / "external_validation.example.yaml"
        raise FileNotFoundError(
            f"Missing local audit configuration: {path}\n"
            f"Copy {example} to {path} and review every path and external-tool setting."
        )
    config = load_yaml(path)
    config["_config_path"] = str(path)
    return config


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_latest_release(project_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(project_root).resolve()
    release = root / LATEST_RELEASE_RELATIVE
    feature_candidates = [
        release / "recording_features.parquet",
        release / "recording_features.csv",
    ]
    registry_candidates = [
        release / "feature_registry.parquet",
        release / "feature_registry.csv",
    ]
    feature_path = next((path for path in feature_candidates if path.is_file()), None)
    registry_path = next((path for path in registry_candidates if path.is_file()), None)
    if feature_path is None or registry_path is None:
        raise FileNotFoundError(
            "The latest reviewed release is missing. Run `paper1-qc --config "
            "config\\project.yaml reviewed-release` before external validation."
        )
    features = _read_table(feature_path)
    registry = _read_table(registry_path)
    if "logical_recording_id" not in features:
        raise ValueError("Latest reviewed feature table lacks logical_recording_id")
    if features["logical_recording_id"].astype(str).duplicated().any():
        raise ValueError("Latest reviewed feature table has duplicate recording identities")
    return features, registry


def comparator_registry_frame(
    specs: Sequence[ComparatorSpec] = DEFAULT_COMPARATOR_SPECS,
) -> pd.DataFrame:
    return pd.DataFrame([asdict(spec) for spec in specs])


def write_audit_registry(
    project_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    features, registry = load_latest_release(root)
    destination = Path(output_dir) if output_dir else root / DEFAULT_OUTPUT_RELATIVE / "00_registry"
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)

    comparator_registry = comparator_registry_frame()
    comparator_registry.to_csv(destination / "external_comparator_registry.csv", index=False)

    audit = registry.copy()
    audit["audit_version"] = AUDIT_VERSION
    audit["external_comparator_count"] = (
        audit["feature"]
        .map(comparator_registry.groupby("feature").size())
        .fillna(0)
        .astype(int)
    )
    audit["external_comparator_required_count"] = (
        audit["feature"]
        .map(
            comparator_registry.loc[comparator_registry["required"]]
            .groupby("feature")
            .size()
        )
        .fillna(0)
        .astype(int)
    )
    audit["final_audit_status"] = "PENDING"
    audit["extractor_change_authorized"] = False
    audit.to_csv(destination / "indicator_audit_registry.csv", index=False)

    pd.DataFrame(
        [
            {
                "audit_version": AUDIT_VERSION,
                "recordings": len(features),
                "indicators": len(audit),
                "comparator_relationships": len(comparator_registry),
                "note": (
                    "Comparator mappings declare hypotheses. They are not validation "
                    "results and do not imply equivalence."
                ),
            }
        ]
    ).to_csv(destination / "registry_summary.csv", index=False)

    return {
        "indicator_registry": destination / "indicator_audit_registry.csv",
        "comparator_registry": destination / "external_comparator_registry.csv",
        "summary": destination / "registry_summary.csv",
    }


def _version_from_command(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first[:300] or None


def _import_version(module_name: str) -> tuple[bool, str | None, str]:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", None)
    return True, str(version) if version is not None else None, "import succeeded"


def tool_availability(config: Mapping[str, Any]) -> pd.DataFrame:
    tools = config.get("tools", {}) if isinstance(config, Mapping) else {}
    rows: list[ToolAvailability] = []

    ffmpeg = str(tools.get("ffmpeg", "ffmpeg"))
    ffmpeg_path = shutil.which(ffmpeg) if ffmpeg.lower() != "disabled" else None
    rows.append(
        ToolAvailability(
            "ffmpeg",
            bool(ffmpeg_path),
            "executable",
            _version_from_command([ffmpeg_path, "-version"]) if ffmpeg_path else None,
            ffmpeg_path or "not found on PATH / disabled",
        )
    )

    for key, module_name in (
        ("librosa", "librosa"),
        ("pyloudnorm", "pyloudnorm"),
        ("essentia", "essentia"),
        ("torchaudio", "torchaudio"),
        ("srmrpy", "srmrpy"),
    ):
        enabled = str(tools.get(key, "auto")).lower() != "disabled"
        available, version, detail = _import_version(module_name) if enabled else (False, None, "disabled")
        rows.append(ToolAvailability(key, available, "python_import", version, detail))

    imported = config.get("imported_scores", {})
    for name in ("nisqa", "dnsmos", "squim"):
        raw_path = imported.get(name)
        if raw_path:
            path = Path(str(raw_path))
            rows.append(
                ToolAvailability(
                    name,
                    path.is_file(),
                    "precomputed_csv",
                    None,
                    str(path),
                )
            )
        else:
            rows.append(
                ToolAvailability(
                    name,
                    False,
                    "precomputed_csv",
                    None,
                    "no local CSV configured",
                )
            )
    return pd.DataFrame([asdict(row) for row in rows])


def load_audio_manifest(project_root: str | Path, config: Mapping[str, Any]) -> pd.DataFrame:
    root = Path(project_root).resolve()
    manifest_value = config.get("audio_manifest")
    if not manifest_value:
        raise ValueError(
            "external_validation.yaml must define `audio_manifest`. The CSV must "
            "contain logical_recording_id and audio_path."
        )
    path = Path(str(manifest_value))
    if not path.is_absolute():
        path = root / path
    manifest = pd.read_csv(path)
    required = {"logical_recording_id", "audio_path"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Audio manifest lacks required columns: {missing}")
    manifest = manifest.copy()
    manifest["logical_recording_id"] = manifest["logical_recording_id"].astype(str)
    if manifest["logical_recording_id"].duplicated().any():
        raise ValueError("Audio manifest contains duplicate logical_recording_id values")
    manifest["audio_path"] = manifest["audio_path"].map(
        lambda value: str((root / str(value)).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    manifest["audio_exists"] = manifest["audio_path"].map(lambda value: Path(value).is_file())
    return manifest


def _parse_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return np.nan


def ffmpeg_astats(audio_path: str | Path, ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    executable = shutil.which(ffmpeg) or ffmpeg
    command = [
        executable,
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        "astats=metadata=0:reset=0",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)
    text = (completed.stderr or "") + "\n" + (completed.stdout or "")
    if completed.returncode != 0:
        raise RuntimeError(text[-2000:])
    return {
        "ffmpeg_peak_level_db": _parse_float(text, r"Peak level dB:\s*([-+0-9.eE]+)"),
        "ffmpeg_rms_level_db": _parse_float(text, r"RMS level dB:\s*([-+0-9.eE]+)"),
        "ffmpeg_rms_peak_db": _parse_float(text, r"RMS peak dB:\s*([-+0-9.eE]+)"),
        "ffmpeg_rms_trough_db": _parse_float(text, r"RMS trough dB:\s*([-+0-9.eE]+)"),
        "ffmpeg_dynamic_range_db": _parse_float(text, r"Dynamic range:\s*([-+0-9.eE]+)"),
        "ffmpeg_noise_floor_db": _parse_float(text, r"Noise floor dB:\s*([-+0-9.eE]+)"),
        "ffmpeg_peak_count": _parse_float(text, r"Peak count:\s*([-+0-9.eE]+)"),
        "ffmpeg_zero_crossings_rate": _parse_float(text, r"Zero crossings rate:\s*([-+0-9.eE]+)"),
    }


def ffmpeg_ebur128(audio_path: str | Path, ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    executable = shutil.which(ffmpeg) or ffmpeg
    command = [
        executable,
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-filter_complex",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)
    text = (completed.stderr or "") + "\n" + (completed.stdout or "")
    if completed.returncode != 0:
        raise RuntimeError(text[-2000:])
    summaries = text.split("Summary:")
    summary = summaries[-1] if len(summaries) > 1 else text
    return {
        "ffmpeg_integrated_lufs": _parse_float(summary, r"\bI:\s*([-+0-9.eE]+)\s*LUFS"),
        "ffmpeg_loudness_range_lu": _parse_float(summary, r"\bLRA:\s*([-+0-9.eE]+)\s*LU"),
        "ffmpeg_true_peak_dbfs": _parse_float(summary, r"\bPeak:\s*([-+0-9.eE]+)\s*dBFS"),
    }


def native_peak_audit(audio_path: str | Path, near_fullscale: float = 0.999) -> dict[str, Any]:
    samples, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float64")
    mono = samples.mean(axis=1)
    finite = np.isfinite(mono)
    if not finite.all():
        mono = mono[finite]
    if not len(mono):
        raise ValueError("No finite decoded samples")
    abs_values = np.abs(mono)
    peak = float(np.max(abs_values))
    threshold = float(near_fullscale)
    return {
        "native_sample_rate_hz": int(sample_rate),
        "native_duration_sec": float(len(mono) / sample_rate),
        "native_peak_abs": peak,
        "native_near_fullscale_fraction": float(np.mean(abs_values >= threshold)),
        "native_exact_peak_fraction": float(np.mean(abs_values == peak)),
        "native_dc_offset": float(np.mean(mono)),
    }


def _librosa_comparators(audio_path: str | Path, highband_hz: float = 4000.0) -> dict[str, Any]:
    import librosa

    waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    if not len(waveform):
        raise ValueError("No decoded samples")
    rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]
    flatness = librosa.feature.spectral_flatness(y=waveform, n_fft=2048, hop_length=512)[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=waveform,
        sr=sample_rate,
        roll_percent=0.95,
        n_fft=2048,
        hop_length=512,
    )[0]
    spectrum = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=512)) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    total = np.sum(spectrum, axis=0)
    high = np.sum(spectrum[frequencies >= highband_hz], axis=0)
    ratio = np.divide(high, total, out=np.full_like(high, np.nan), where=total > 0)
    return {
        "librosa_rms_median": float(np.nanmedian(rms)),
        "librosa_rms_iqr": float(np.nanquantile(rms, 0.75) - np.nanquantile(rms, 0.25)),
        "librosa_spectral_flatness_median": float(np.nanmedian(flatness)),
        "librosa_rolloff95_hz": float(np.nanmedian(rolloff)),
        "librosa_highband_ratio": float(np.nanmedian(ratio)),
    }


def _pyloudnorm_comparator(audio_path: str | Path) -> dict[str, Any]:
    import pyloudnorm as pyln

    samples, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float64")
    mono = samples.mean(axis=1)
    meter = pyln.Meter(sample_rate)
    return {"pyloudnorm_integrated_lufs": float(meter.integrated_loudness(mono))}


def _essentia_comparator(audio_path: str | Path) -> dict[str, Any]:
    import essentia.standard as es

    loader = es.MonoLoader(filename=str(audio_path))
    waveform = loader()
    detector = es.DiscontinuityDetector()
    locations, amplitudes = detector(waveform)
    return {
        "essentia_discontinuity_count": int(len(locations)),
        "essentia_discontinuity_max_amplitude": (
            float(np.max(amplitudes)) if len(amplitudes) else 0.0
        ),
    }


def _load_imported_score(
    project_root: Path,
    name: str,
    value: str | Path,
    id_column: str,
) -> pd.DataFrame:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    frame = pd.read_csv(path)
    if id_column not in frame:
        raise ValueError(f"{name} imported score table lacks ID column {id_column!r}")
    frame = frame.rename(columns={id_column: "logical_recording_id"}).copy()
    frame["logical_recording_id"] = frame["logical_recording_id"].astype(str)
    if frame["logical_recording_id"].duplicated().any():
        raise ValueError(f"{name} imported score table has duplicate recording IDs")
    renamed = {
        column: column if column.startswith(f"{name}_") else f"{name}_{column}"
        for column in frame.columns
        if column != "logical_recording_id"
    }
    return frame.rename(columns=renamed)


def run_external_comparators(
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    recording_ids: Iterable[str] | None = None,
) -> dict[str, Path]:
    """Run available comparators and import configured model scores.

    Failures are logged per recording/tool and never silently converted to zero.
    """

    root = Path(project_root).resolve()
    destination = Path(output_dir) if output_dir else root / DEFAULT_OUTPUT_RELATIVE / "01_comparators"
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)

    manifest = load_audio_manifest(root, config)
    requested = set(map(str, recording_ids)) if recording_ids is not None else None
    if requested is not None:
        manifest = manifest.loc[manifest["logical_recording_id"].isin(requested)].copy()

    run_config = config.get("run", {})
    max_recordings = run_config.get("max_recordings")
    if max_recordings is not None:
        manifest = manifest.head(int(max_recordings)).copy()

    availability = tool_availability(config)
    availability.to_csv(destination / "tool_availability.csv", index=False)
    available = dict(zip(availability["tool"], availability["available"]))
    tools = config.get("tools", {})
    ffmpeg_name = str(tools.get("ffmpeg", "ffmpeg"))
    near_fullscale = float(config.get("native_peak_audit", {}).get("near_fullscale", 0.999))

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        row: dict[str, Any] = {
            "logical_recording_id": str(item.logical_recording_id),
            "audio_path": str(item.audio_path),
            "audio_exists": bool(item.audio_exists),
        }
        if not item.audio_exists:
            errors.append(
                {
                    "logical_recording_id": item.logical_recording_id,
                    "tool": "manifest",
                    "error_type": "FileNotFoundError",
                    "message": item.audio_path,
                }
            )
            rows.append(row)
            continue

        tasks = [
            ("native_peak_audit", lambda: native_peak_audit(item.audio_path, near_fullscale)),
        ]
        if available.get("ffmpeg", False):
            tasks.extend(
                [
                    ("ffmpeg_astats", lambda: ffmpeg_astats(item.audio_path, ffmpeg_name)),
                    ("ffmpeg_ebur128", lambda: ffmpeg_ebur128(item.audio_path, ffmpeg_name)),
                ]
            )
        if available.get("librosa", False):
            tasks.append(("librosa", lambda: _librosa_comparators(item.audio_path)))
        if available.get("pyloudnorm", False):
            tasks.append(("pyloudnorm", lambda: _pyloudnorm_comparator(item.audio_path)))
        if available.get("essentia", False):
            tasks.append(("essentia", lambda: _essentia_comparator(item.audio_path)))

        for tool_name, task in tasks:
            try:
                row.update(task())
                row[f"{tool_name}_status"] = "ok"
            except Exception as exc:
                row[f"{tool_name}_status"] = "error"
                errors.append(
                    {
                        "logical_recording_id": item.logical_recording_id,
                        "tool": tool_name,
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
        rows.append(row)

    measurements = pd.DataFrame(rows)
    imported = config.get("imported_scores", {})
    imported_id_columns = config.get("imported_id_columns", {})
    for name, value in imported.items():
        if not value:
            continue
        frame = _load_imported_score(
            root,
            str(name),
            value,
            str(imported_id_columns.get(name, "logical_recording_id")),
        )
        measurements = measurements.merge(
            frame,
            on="logical_recording_id",
            how="left",
            validate="one_to_one",
        )

    measurements.to_csv(destination / "external_measurements.csv", index=False)
    error_columns = [
        "logical_recording_id",
        "tool",
        "error_type",
        "message",
    ]
    pd.DataFrame(errors, columns=error_columns).to_csv(
        destination / "external_errors.csv", index=False
    )
    manifest.to_csv(destination / "resolved_audio_manifest.csv", index=False)
    return {
        "measurements": destination / "external_measurements.csv",
        "errors": destination / "external_errors.csv",
        "availability": destination / "tool_availability.csv",
        "manifest": destination / "resolved_audio_manifest.csv",
    }


def _candidate_support_columns(frame: pd.DataFrame, feature: str) -> list[str]:
    family_prefix = feature.split("_", 1)[0] + "_"
    terms = (
        "support",
        "frame_count",
        "window_count",
        "interval_count",
        "boundary_count",
        "segment_count",
        "event_count",
        "duration_sec",
        "span_sec",
        "exposure",
        "status",
        "available",
        "censor",
    )
    return [
        column
        for column in frame.columns
        if column.startswith(family_prefix) and any(term in column.lower() for term in terms)
    ]


def support_audit(
    project_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    features, registry = load_latest_release(root)
    destination = Path(output_dir) if output_dir else root / DEFAULT_OUTPUT_RELATIVE / "02_window_support"
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in registry.itertuples(index=False):
        feature = str(spec.feature)
        values = pd.to_numeric(features.get(feature), errors="coerce")
        status_field = getattr(spec, "status_field", f"{feature}_status")
        support_columns = _candidate_support_columns(features, feature)
        row = {
            "feature": feature,
            "family_code": getattr(spec, "family_code", feature.split("_", 1)[0].upper()),
            "role": getattr(spec, "role", ""),
            "n_total": int(len(features)),
            "n_available": int(values.notna().sum()),
            "missing_fraction": float(values.isna().mean()),
            "unique_nonmissing": int(values.dropna().nunique()),
            "status_field_present": bool(status_field in features.columns),
            "support_column_count": int(len(support_columns)),
            "support_columns": "|".join(support_columns),
            "window_support_verdict": "PENDING_EVIDENCE_REVIEW",
        }
        rows.append(row)
    summary = pd.DataFrame(rows)

    reviewed_root = root / REVIEWED_ROOT_RELATIVE
    patterns = (
        "*window*sensitiv*.csv",
        "*parameter*sensitiv*.csv",
        "*support*.csv",
        "*deletion*.csv",
        "*resampl*.csv",
        "*invariance*.csv",
    )
    evidence_paths: set[Path] = set()
    if reviewed_root.is_dir():
        for pattern in patterns:
            evidence_paths.update(reviewed_root.rglob(pattern))
    evidence = pd.DataFrame(
        [
            {
                "file_name": path.name,
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(evidence_paths)
            if path.is_file()
        ]
    )
    summary.to_csv(destination / "feature_support_summary.csv", index=False)
    evidence.to_csv(destination / "existing_sensitivity_evidence_inventory.csv", index=False)
    return {
        "support_summary": destination / "feature_support_summary.csv",
        "evidence_inventory": destination / "existing_sensitivity_evidence_inventory.csv",
    }


def _cluster_bootstrap_spearman(
    frame: pd.DataFrame,
    x: str,
    y: str,
    cluster: str | None,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    local = frame[[x, y] + ([cluster] if cluster and cluster in frame else [])].copy()
    local[x] = pd.to_numeric(local[x], errors="coerce")
    local[y] = pd.to_numeric(local[y], errors="coerce")
    local = local.dropna(subset=[x, y])
    observed = stats.spearmanr(local[x], local[y]).statistic if len(local) >= 3 else np.nan
    if len(local) < 3 or iterations <= 0:
        return float(observed), np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    if cluster and cluster in local:
        groups = local[cluster].astype(str).unique()
        if len(groups) < 2:
            cluster = None
    for _ in range(iterations):
        if cluster and cluster in local:
            sampled = rng.choice(groups, size=len(groups), replace=True)
            pieces = []
            for index, group in enumerate(sampled):
                piece = local.loc[local[cluster].astype(str).eq(str(group))].copy()
                piece["_bootstrap_cluster"] = index
                pieces.append(piece)
            resampled = pd.concat(pieces, ignore_index=True)
        else:
            indices = rng.integers(0, len(local), size=len(local))
            resampled = local.iloc[indices]
        value = stats.spearmanr(resampled[x], resampled[y]).statistic
        if np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return float(observed), np.nan, np.nan, np.nan
    array = np.asarray(estimates)
    return (
        float(observed),
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
        float(np.mean(array > 0)),
    )


def convergent_validity_audit(
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    specs: Sequence[ComparatorSpec] = DEFAULT_COMPARATOR_SPECS,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    destination = Path(output_dir) if output_dir else root / DEFAULT_OUTPUT_RELATIVE / "03_convergence"
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)

    features, _ = load_latest_release(root)
    comparator_path = root / DEFAULT_OUTPUT_RELATIVE / "01_comparators" / "external_measurements.csv"
    if not comparator_path.is_file():
        raise FileNotFoundError(
            f"Missing comparator table: {comparator_path}. Run notebook 01 first."
        )
    comparators = pd.read_csv(comparator_path)
    merged = features.merge(
        comparators,
        on="logical_recording_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_external"),
    )
    bootstrap = config.get("statistics", {})
    iterations = int(bootstrap.get("bootstrap_iterations", 2000))
    seed = int(bootstrap.get("random_seed", 20260806))
    cluster = str(bootstrap.get("cluster_column", "SubjectID"))
    if cluster not in merged.columns:
        cluster = None

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        if spec.feature not in merged or spec.comparator_column not in merged:
            rows.append(
                {
                    **asdict(spec),
                    "n": 0,
                    "spearman_rho": np.nan,
                    "p_value": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "bootstrap_probability_positive": np.nan,
                    "direction_matches": False,
                    "availability_status": "missing_column",
                }
            )
            continue
        local = merged[[spec.feature, spec.comparator_column] + ([cluster] if cluster else [])].copy()
        local[spec.feature] = pd.to_numeric(local[spec.feature], errors="coerce")
        local[spec.comparator_column] = pd.to_numeric(
            local[spec.comparator_column], errors="coerce"
        )
        local = local.dropna(subset=[spec.feature, spec.comparator_column])
        n = len(local)
        if n >= 3:
            test = stats.spearmanr(local[spec.feature], local[spec.comparator_column])
            rho = float(test.statistic)
            p_value = float(test.pvalue)
            rho_b, low, high, probability_positive = _cluster_bootstrap_spearman(
                local,
                spec.feature,
                spec.comparator_column,
                cluster,
                iterations=iterations,
                seed=seed + index,
            )
            if np.isfinite(rho_b):
                rho = rho_b
        else:
            rho = p_value = low = high = probability_positive = np.nan
        expected_sign = 1 if spec.expected_direction == "positive" else -1
        direction_matches = bool(np.isfinite(rho) and np.sign(rho) == expected_sign)
        probability_expected = (
            probability_positive
            if expected_sign > 0
            else (1.0 - probability_positive if np.isfinite(probability_positive) else np.nan)
        )
        rows.append(
            {
                **asdict(spec),
                "n": int(n),
                "spearman_rho": rho,
                "p_value": p_value,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_probability_positive": probability_positive,
                "bootstrap_probability_expected_direction": probability_expected,
                "direction_matches": direction_matches,
                "availability_status": "ok" if n >= 3 else "insufficient_complete_pairs",
            }
        )
    results = pd.DataFrame(rows)
    valid = results["p_value"].notna()
    results["p_fdr"] = np.nan
    if valid.any():
        results.loc[valid, "p_fdr"] = multipletests(
            results.loc[valid, "p_value"].to_numpy(float),
            method="fdr_bh",
        )[1]
    results.to_csv(destination / "convergent_validity_results.csv", index=False)
    merged.to_parquet(destination / "feature_external_merged.parquet", index=False)
    return {
        "results": destination / "convergent_validity_results.csv",
        "merged": destination / "feature_external_merged.parquet",
    }


def _manual_evidence_path(root: Path) -> Path:
    return root / DEFAULT_OUTPUT_RELATIVE / "04_final_verdict" / "manual_evidence_review.csv"


def build_final_verdict_template(
    project_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    rule: VerdictRule = VerdictRule(),
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    destination = Path(output_dir) if output_dir else root / DEFAULT_OUTPUT_RELATIVE / "04_final_verdict"
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)

    _, registry = load_latest_release(root)
    support_path = root / DEFAULT_OUTPUT_RELATIVE / "02_window_support" / "feature_support_summary.csv"
    convergence_path = (
        root / DEFAULT_OUTPUT_RELATIVE / "03_convergence" / "convergent_validity_results.csv"
    )
    support = pd.read_csv(support_path) if support_path.is_file() else pd.DataFrame()
    convergence = pd.read_csv(convergence_path) if convergence_path.is_file() else pd.DataFrame()

    template = registry.copy()
    if not support.empty:
        template = template.merge(
            support[
                [
                    "feature",
                    "n_available",
                    "missing_fraction",
                    "support_column_count",
                    "window_support_verdict",
                ]
            ],
            on="feature",
            how="left",
            validate="one_to_one",
        )
    else:
        template["n_available"] = np.nan
        template["missing_fraction"] = np.nan
        template["support_column_count"] = np.nan
        template["window_support_verdict"] = "NOT_RUN"

    if not convergence.empty:
        grouped = convergence.groupby("feature", dropna=False).agg(
            external_relationship_count=("feature", "size"),
            external_relationships_with_data=("n", lambda values: int(np.sum(np.asarray(values) >= 3))),
            external_required_relationships_passed=(
                "direction_matches",
                lambda values: int(np.sum(pd.Series(values).fillna(False))),
            ),
            maximum_absolute_external_rho=(
                "spearman_rho",
                lambda values: float(np.nanmax(np.abs(values)))
                if np.isfinite(pd.to_numeric(values, errors="coerce")).any()
                else np.nan,
            ),
        )
        template = template.merge(grouped, left_on="feature", right_index=True, how="left")
    else:
        template["external_relationship_count"] = 0
        template["external_relationships_with_data"] = 0
        template["external_required_relationships_passed"] = 0
        template["maximum_absolute_external_rho"] = np.nan

    manual_columns = {
        "implementation_reaudit": "PENDING",
        "synthetic_known_truth": "PENDING",
        "window_parameter_stability": "PENDING",
        "leave_one_unit_out_stability": "PENDING",
        "boundary_or_segmentation_sensitivity": "PENDING",
        "competing_mechanism_discrimination": "PENDING",
        "clinical_entanglement_review": "PENDING",
        "external_comparator_interpretation": "PENDING",
        "human_case_review": "PENDING",
        "proposed_action": "PENDING",
        "final_grade": "PENDING",
        "extractor_change_required": "PENDING",
        "change_rationale": "",
        "authorized_new_measurement_version": "",
        "reviewer": "",
        "review_date": "",
    }
    for column, default in manual_columns.items():
        template[column] = default
    template["audit_version"] = AUDIT_VERSION
    template["automatic_change_authorized"] = False
    template["decision_note"] = (
        "No extractor change is authorized until a feature-level final verdict is signed "
        "and a new measurement version is declared."
    )

    manual_path = destination / "manual_evidence_review.csv"
    if manual_path.is_file():
        previous = pd.read_csv(manual_path)
        preserved = [
            column for column in manual_columns if column in previous.columns
        ]
        template = template.drop(columns=preserved, errors="ignore").merge(
            previous[["feature", *preserved]],
            on="feature",
            how="left",
            validate="one_to_one",
        )
        for column, default in manual_columns.items():
            if column not in template:
                template[column] = default
            template[column] = template[column].fillna(default)

    template.to_csv(manual_path, index=False)
    pd.DataFrame([asdict(rule)]).to_csv(destination / "verdict_rules.csv", index=False)
    return {
        "manual_review": manual_path,
        "rules": destination / "verdict_rules.csv",
    }


def finalize_verdicts(
    project_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    destination = Path(output_dir) if output_dir else root / DEFAULT_OUTPUT_RELATIVE / "04_final_verdict"
    if not destination.is_absolute():
        destination = root / destination
    manual_path = destination / "manual_evidence_review.csv"
    if not manual_path.is_file():
        raise FileNotFoundError(manual_path)
    frame = pd.read_csv(manual_path)

    required_review_fields = [
        "implementation_reaudit",
        "synthetic_known_truth",
        "window_parameter_stability",
        "leave_one_unit_out_stability",
        "boundary_or_segmentation_sensitivity",
        "competing_mechanism_discrimination",
        "clinical_entanglement_review",
        "external_comparator_interpretation",
        "human_case_review",
        "proposed_action",
        "final_grade",
        "extractor_change_required",
        "reviewer",
        "review_date",
    ]
    missing = sorted(set(required_review_fields).difference(frame.columns))
    if missing:
        raise ValueError(f"Manual evidence table lacks columns: {missing}")

    pending = frame[required_review_fields].astype(str).apply(
        lambda column: column.str.strip().str.upper().isin({"", "PENDING", "NAN"})
    )
    frame["final_audit_complete"] = ~pending.any(axis=1)
    frame["extractor_change_authorized"] = (
        frame["final_audit_complete"]
        & frame["extractor_change_required"].astype(str).str.upper().eq("YES")
        & frame["authorized_new_measurement_version"].astype(str).str.strip().ne("")
        & frame["change_rationale"].astype(str).str.strip().ne("")
    )
    frame["release_status"] = np.where(
        frame["final_audit_complete"],
        np.where(
            frame["extractor_change_authorized"],
            "REVISE_UNDER_NEW_MEASUREMENT_VERSION",
            "FINAL_AUDIT_SIGNED_NO_EXTRACTOR_CHANGE",
        ),
        "PENDING_FINAL_AUDIT",
    )
    final_path = destination / "final_feature_audit_verdicts.csv"
    frame.to_csv(final_path, index=False)

    summary = (
        frame.groupby(["family_code", "release_status", "final_grade"], dropna=False)
        .size()
        .rename("feature_count")
        .reset_index()
    )
    summary.to_csv(destination / "final_family_audit_summary.csv", index=False)
    return {
        "verdicts": final_path,
        "summary": destination / "final_family_audit_summary.csv",
    }


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_audit_manifest(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    audit_root = root / DEFAULT_OUTPUT_RELATIVE
    rows = []
    for path in sorted(audit_root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    destination = audit_root / "audit_artifact_manifest.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination
