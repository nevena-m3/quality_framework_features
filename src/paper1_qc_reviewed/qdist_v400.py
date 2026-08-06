"""QDIST v4.0.0 reviewed preflight wrapper.

This module does not replace or loosen the frozen QDIST v3.1.1 detector. It
adapts that detector to the common reviewed G1-G10 and A-J evidence contract.
The eventual v4.0.0 release is allowed only if cohort numerical equivalence to
v3.1.1 is demonstrated exactly.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats

import paper1_qc.qdist as legacy

MEASUREMENT_VERSION = "qdist-v4.0.0-candidate"
LEGACY_MEASUREMENT_VERSION = "qdist-v3.1.1"
REVIEWED_ORCHESTRATION_VERSION = "qdist-v4.0.0-preflight-orchestration-v2"
PREFLIGHT_HOTFIX_REVISION = "legacy-api-compat-r1"
PREFLIGHT_PANEL_STEMS = (
    "A_construct_response",
    "B_discriminant_specificity",
    "C_transformation_contract",
)
ANALYSIS_FEATURES = (
    "qdist_hard_clipped_frame_fraction",
    "qdist_hard_clip_event_rate_per_min",
    "qdist_hard_clipped_sample_fraction",
)
PRIMARY_FEATURES = ANALYSIS_FEATURES[:2]
SECONDARY_FEATURES = ANALYSIS_FEATURES[2:]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def save_table(frame: pd.DataFrame, directory: Path, stem: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.csv"
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def save_figure_bundle(
    fig,
    directory: Path,
    stem: str,
    *,
    source: pd.DataFrame,
    caption: str,
    provenance: dict[str, Any],
) -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": directory / f"{stem}.png",
        "svg": directory / f"{stem}.svg",
        "pdf": directory / f"{stem}.pdf",
        "source_csv": directory / f"{stem}.source.csv",
        "caption": directory / f"{stem}.caption.md",
        "provenance": directory / f"{stem}.provenance.json",
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    source.to_csv(paths["source_csv"], index=False)
    paths["caption"].write_text(caption.strip() + "\n", encoding="utf-8")
    write_json(paths["provenance"], provenance)
    return {key: str(value) for key, value in paths.items()}


def parameter_payload() -> dict[str, Any]:
    parameters = legacy.DEFAULT_PARAMETERS
    if hasattr(parameters, "to_dict"):
        return parameters.to_dict()
    if is_dataclass(parameters):
        return asdict(parameters)
    return {
        name: value
        for name, value in vars(parameters).items()
        if not name.startswith("_")
    }


def _result_summary(result) -> dict[str, Any]:
    """Return the frozen detector's recording summary across supported APIs."""
    if hasattr(result, "summary"):
        summary = getattr(result, "summary")
    elif hasattr(result, "recording"):
        summary = getattr(result, "recording")
    elif isinstance(result, dict):
        summary = result
    else:
        raise TypeError(
            "QDIST extraction result exposes neither .summary nor .recording."
        )
    if not isinstance(summary, dict):
        try:
            summary = dict(summary)
        except Exception as exc:
            raise TypeError("QDIST recording summary is not mapping-like.") from exc
    return summary


def _local_quantize_pcm(
    waveform: np.ndarray,
    bit_depth: int,
) -> tuple[np.ndarray, float]:
    """Quantize a validation fixture to the normalized signed-PCM lattice."""
    bits = int(bit_depth)
    if bits < 2 or bits > 32:
        raise ValueError("bit_depth must be between 2 and 32.")
    samples = np.asarray(waveform, dtype=np.float64)
    scale = float(2 ** (bits - 1))
    minimum_code = -(2 ** (bits - 1))
    maximum_code = 2 ** (bits - 1) - 1
    codes = np.rint(samples * scale)
    codes = np.clip(codes, minimum_code, maximum_code)
    return codes / scale, 1.0 / scale


def quantize_fixture(
    waveform: np.ndarray,
    bit_depth: int,
) -> tuple[np.ndarray, float]:
    """Adapt either frozen quantizer signature without changing the detector."""
    function = getattr(legacy, "quantize_pcm", None)
    if not callable(function):
        return _local_quantize_pcm(waveform, bit_depth)
    output = function(np.asarray(waveform, dtype=np.float64), int(bit_depth))
    if isinstance(output, tuple):
        if len(output) != 2:
            raise RuntimeError(
                "paper1_qc.qdist.quantize_pcm returned an unsupported tuple."
            )
        pcm, step = output
        step = float(step)
    else:
        pcm = output
        step = 1.0 / float(2 ** (int(bit_depth) - 1))
    pcm = np.asarray(pcm, dtype=np.float64)
    if pcm.shape != np.asarray(waveform).shape:
        raise RuntimeError("QDIST fixture quantizer changed waveform geometry.")
    if not np.isfinite(pcm).all() or not np.isfinite(step) or step <= 0:
        raise RuntimeError("QDIST fixture quantizer returned an invalid lattice.")
    return pcm, step


def synthetic_speech_like(
    *,
    sample_rate_hz: int = 48_000,
    duration_sec: float = 4.0,
    seed: int = 991,
) -> np.ndarray:
    """Deterministic speech-like validation fixture owned by the reviewed layer.

    This is test material, not a production estimator dependency. Silent guards
    make frame-aligned translation tests non-wrapping, while low-level seeded
    noise prevents accidental exact floating-point plateaus before PCM
    quantization.
    """
    fs = int(sample_rate_hz)
    duration = float(duration_sec)
    if fs <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("sample_rate_hz and duration_sec must be positive.")
    n_samples = max(1, int(round(fs * duration)))
    t = np.arange(n_samples, dtype=np.float64) / fs
    rng = np.random.default_rng(int(seed))

    f0 = 118.0 + 18.0 * np.sin(2.0 * np.pi * 0.63 * t)
    f0 += 5.0 * np.sin(2.0 * np.pi * 1.37 * t + 0.4)
    phase = 2.0 * np.pi * np.cumsum(f0) / fs
    voiced = np.zeros(n_samples, dtype=np.float64)
    phases = rng.uniform(-np.pi, np.pi, size=10)
    for harmonic in range(1, 11):
        voiced += (
            np.sin(harmonic * phase + phases[harmonic - 1])
            / harmonic ** 1.15
        )

    syllabic = 0.22 + 0.78 * (
        0.5 + 0.5 * np.sin(2.0 * np.pi * 2.7 * t + 0.3)
    ) ** 1.4
    phrase = 0.75 + 0.25 * np.sin(2.0 * np.pi * 0.31 * t - 0.2)
    noise = rng.normal(0.0, 1.0, size=n_samples)
    if fs > 6_000:
        sos = signal.butter(4, 2_000, btype="highpass", fs=fs, output="sos")
        noise = signal.sosfilt(sos, noise)
    waveform = syllabic * phrase * (voiced + 0.08 * noise)
    waveform += 3e-4 * rng.normal(size=n_samples)

    guard = min(int(round(0.25 * fs)), max(0, n_samples // 5))
    if guard:
        waveform[:guard] = 0.0
        waveform[-guard:] = 0.0
        fade = min(int(round(0.025 * fs)), guard)
        if fade > 1 and n_samples > 2 * guard:
            ramp = np.sin(np.linspace(0.0, np.pi / 2.0, fade)) ** 2
            waveform[guard:guard + fade] *= ramp
            waveform[-guard - fade:-guard] *= ramp[::-1]

    peak = float(np.max(np.abs(waveform)))
    if peak <= 0 or not np.isfinite(peak):
        raise RuntimeError("Synthetic QDIST fixture is degenerate.")
    return (0.98 / peak) * waveform


def hard_clip(
    waveform: np.ndarray,
    limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact symmetric hard clipping and return its exact truth mask."""
    samples = np.asarray(waveform, dtype=np.float64)
    threshold = float(limit)
    if not np.isfinite(threshold) or threshold <= 0 or threshold > 1:
        raise ValueError("Hard-clipping limit must lie in (0, 1].")
    truth = np.abs(samples) > threshold
    return np.clip(samples, -threshold, threshold), truth


def soft_clip_tanh(waveform: np.ndarray, drive: float) -> np.ndarray:
    """Smooth tanh saturation used only as a discriminant/scope fixture."""
    drive = float(drive)
    if not np.isfinite(drive) or drive <= 0:
        raise ValueError("drive must be positive and finite.")
    samples = np.asarray(waveform, dtype=np.float64)
    return np.tanh(drive * samples) / np.tanh(drive)


def detector_frame_length_ms() -> float:
    parameters = legacy.DEFAULT_PARAMETERS
    for field in ("frame_length_ms", "frame_ms"):
        value = getattr(parameters, field, None)
        if value is not None and np.isfinite(float(value)) and float(value) > 0:
            return float(value)
    raise RuntimeError(
        "Frozen QDIST parameters expose neither frame_length_ms nor frame_ms."
    )


def detector_frame_length_samples(fs: int) -> int:
    return max(1, int(round(int(fs) * detector_frame_length_ms() / 1000.0)))


def detector_entrypoint_name() -> str:
    if callable(getattr(legacy, "extract_qdist", None)):
        return "extract_qdist"
    if callable(getattr(legacy, "analyze_hard_clipping", None)):
        return "analyze_hard_clipping"
    return ""


def legacy_contract() -> dict[str, Any]:
    # Only production detector symbols are part of the frozen API contract.
    # Synthetic signal generators are reviewed validation fixtures and are not
    # required exports of paper1_qc.qdist.
    required = [
        "ANALYSIS_FEATURES",
        "DEFAULT_PARAMETERS",
    ]
    missing = [name for name in required if not hasattr(legacy, name)]
    entrypoint = detector_entrypoint_name()
    if not entrypoint:
        missing.append("extract_qdist_or_analyze_hard_clipping")
    features = tuple(getattr(legacy, "ANALYSIS_FEATURES", ()))
    version = str(getattr(legacy, "MEASUREMENT_VERSION", ""))
    helper_resolution = {
        "synthetic_speech_like": "reviewed_local",
        "hard_clip": "reviewed_local",
        "soft_clip_tanh": "reviewed_local",
        "quantize_pcm": (
            "legacy_adapter"
            if callable(getattr(legacy, "quantize_pcm", None))
            else "reviewed_local"
        ),
        "detector_entrypoint": entrypoint or "missing",
        "result_summary": "summary_or_recording_runtime_adapter",
    }
    return {
        "missing_symbols": missing,
        "legacy_features": features,
        "legacy_measurement_version": version,
        "features_exact": features == ANALYSIS_FEATURES,
        "version_exact": version == LEGACY_MEASUREMENT_VERSION,
        "parameter_payload": parameter_payload(),
        "fixture_helpers_owned_by_reviewed_preflight": True,
        "resolved_helper_api": helper_resolution,
        "detector_frame_length_ms": detector_frame_length_ms(),
        "detector_entrypoint": entrypoint,
    }


def analyze_pcm(
    waveform: np.ndarray,
    fs: int,
    recording_id: str,
    bit_depth: int = 16,
):
    pcm, step = quantize_fixture(np.asarray(waveform, dtype=float), bit_depth)
    entrypoint = detector_entrypoint_name()
    if entrypoint == "extract_qdist":
        provenance_class = getattr(legacy, "NativeSignalProvenance", None)
        if provenance_class is None:
            raise RuntimeError(
                "extract_qdist is present but NativeSignalProvenance is missing."
            )
        provenance = provenance_class(
            native_view_verified=True,
            known_preprocessing_applied=False,
            codec_name=f"pcm_s{bit_depth}le",
            sample_format=f"s{bit_depth}",
            bits_per_raw_sample=int(bit_depth),
        )
        result = legacy.extract_qdist(
            pcm,
            int(fs),
            logical_recording_id=recording_id,
            provenance=provenance,
            parameters=legacy.DEFAULT_PARAMETERS,
        )
    elif entrypoint == "analyze_hard_clipping":
        result = legacy.analyze_hard_clipping(
            pcm,
            int(fs),
            logical_recording_id=recording_id,
            quantization_step=step,
            source_bit_depth=bit_depth,
            source_subtype=f"pcm_s{bit_depth}le",
            parameters=legacy.DEFAULT_PARAMETERS,
        )
    else:
        raise RuntimeError(
            "Frozen QDIST exposes neither extract_qdist nor analyze_hard_clipping."
        )
    return result, pcm, step


def summary_features(result) -> dict[str, float]:
    summary = _result_summary(result)
    output: dict[str, Any] = {}
    for feature in ANALYSIS_FEATURES:
        output[feature] = float(summary[feature])
    event_count = summary.get(
        "qdist_event_count",
        summary.get("qdist_hard_clip_event_count", np.nan),
    )
    plateau_count = summary.get(
        "qdist_plateau_count",
        summary.get("qdist_accepted_plateau_count", np.nan),
    )
    output["qdist_event_count"] = float(event_count)
    output["qdist_plateau_count"] = float(plateau_count)
    output["qdist_status"] = str(summary.get("qdist_status", ""))
    return output


def is_positive(result) -> bool:
    summary = _result_summary(result)
    value = summary.get(
        "qdist_event_count",
        summary.get("qdist_hard_clip_event_count", 0),
    )
    value = float(value or 0)
    return bool(np.isfinite(value) and value > 0)


def plateau_mask(result, n_samples: int) -> np.ndarray:
    mask = np.zeros(int(n_samples), dtype=bool)
    ledger = result.accepted_plateau_ledger
    if ledger is None or len(ledger) == 0:
        return mask
    for row in ledger.itertuples(index=False):
        start = int(getattr(row, "start_sample_task"))
        end = int(getattr(row, "end_sample_task_exclusive"))
        mask[max(0, start):min(n_samples, end)] = True
    return mask


def reconstruct_mono(result, n_samples: int, fs: int) -> dict[str, float]:
    plateau = result.accepted_plateau_ledger
    episodes = result.episode_ledger
    frame_length = detector_frame_length_samples(fs)
    complete_frames = int(n_samples // frame_length)
    affected = np.zeros(complete_frames, dtype=bool)
    accepted_samples = 0
    if plateau is not None and len(plateau):
        for row in plateau.itertuples(index=False):
            start = int(getattr(row, "start_sample_task"))
            end = int(getattr(row, "end_sample_task_exclusive"))
            accepted_samples += max(0, end - start)
            if complete_frames:
                first = max(0, start // frame_length)
                last = min(
                    complete_frames - 1,
                    max(start, end - 1) // frame_length,
                )
                if last >= first:
                    affected[first:last + 1] = True
    event_count = 0 if episodes is None else int(len(episodes))
    duration_min = (n_samples / fs) / 60.0
    return {
        "qdist_hard_clipped_frame_fraction": (
            float(affected.mean()) if complete_frames else np.nan
        ),
        "qdist_hard_clip_event_rate_per_min": (
            event_count / duration_min if duration_min > 0 else np.nan
        ),
        "qdist_hard_clipped_sample_fraction": (
            accepted_samples / n_samples if n_samples > 0 else np.nan
        ),
    }


def _speech(fs: int, duration: float, seed: int) -> np.ndarray:
    return synthetic_speech_like(
        sample_rate_hz=int(fs),
        duration_sec=float(duration),
        seed=int(seed),
    )

def run_preflight(output_root: Path, *, run_codecs: bool = True) -> dict[str, Any]:
    output_root = Path(output_root)
    tables = output_root / "tables"
    figures = output_root / "figures"
    validation = output_root / "validation"
    audit = output_root / "audit"
    manifests = output_root / "manifests"
    for directory in [tables, figures, validation, audit, manifests]:
        directory.mkdir(parents=True, exist_ok=True)

    contract = legacy_contract()
    if contract["missing_symbols"]:
        raise RuntimeError(f"Legacy QDIST API is incomplete: {contract['missing_symbols']}")
    if not contract["features_exact"] or not contract["version_exact"]:
        raise RuntimeError(f"Legacy QDIST identity mismatch: {contract}")

    fs = 48_000
    base = _speech(fs, 6.0, 4101)
    clean_result, clean_pcm, step = analyze_pcm(base, fs, "CLEAN")
    clipped_float, truth = hard_clip(base, 0.65)
    clipped_result, clipped_pcm, _ = analyze_pcm(clipped_float, fs, "CLIPPED")
    inverted_result, _, _ = analyze_pcm(-clipped_float, fs, "INVERTED")
    shift = detector_frame_length_samples(fs)
    shifted = np.concatenate([np.zeros(shift), clipped_float[:-shift]])
    shifted_result, _, _ = analyze_pcm(shifted, fs, "SHIFTED")

    reconstructed = reconstruct_mono(clipped_result, len(clipped_pcm), fs)
    reconstruction_rows = []
    for feature in ANALYSIS_FEATURES:
        observed = float(_result_summary(clipped_result)[feature])
        rebuilt = float(reconstructed[feature])
        reconstruction_rows.append({
            "feature": feature,
            "observed": observed,
            "reconstructed": rebuilt,
            "absolute_error": abs(observed - rebuilt),
            "passed": bool(np.isclose(observed, rebuilt, rtol=0, atol=1e-12)),
        })
    reconstruction = pd.DataFrame(reconstruction_rows)

    dose_rows = []
    for seed in [4201, 4202, 4203]:
        source = _speech(fs, 6.0, seed)
        for threshold in [0.95, 0.85, 0.75, 0.65, 0.55]:
            clipped, truth_mask = hard_clip(source, threshold)
            result, pcm, _ = analyze_pcm(clipped, fs, f"DOSE_{seed}_{threshold}")
            predicted = plateau_mask(result, len(pcm))
            truth_mask = np.asarray(truth_mask, dtype=bool)[:len(predicted)]
            tp = int(np.count_nonzero(predicted & truth_mask))
            fp = int(np.count_nonzero(predicted & ~truth_mask))
            fn = int(np.count_nonzero(~predicted & truth_mask))
            precision = tp / (tp + fp) if tp + fp else np.nan
            recall = tp / (tp + fn) if tp + fn else np.nan
            f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan
            dose_rows.append({
                "seed": seed,
                "threshold": threshold,
                "true_clipped_sample_fraction": float(truth_mask.mean()),
                "sample_precision": precision,
                "sample_recall": recall,
                "sample_f1": f1,
                **summary_features(result),
            })
    dose = pd.DataFrame(dose_rows)
    dose_summary = dose.groupby("threshold", as_index=False).median(numeric_only=True)

    t = np.arange(fs * 4) / fs
    rng = np.random.default_rng(4301)
    controls = {
        "clean_speech": _speech(fs, 4.0, 4302),
        "sine_440Hz": 0.995 * np.sin(2 * np.pi * 440 * t),
        "triangle_440Hz": 0.95 * signal.sawtooth(2 * np.pi * 440 * t, width=0.5),
        "sawtooth_440Hz": 0.95 * signal.sawtooth(2 * np.pi * 440 * t),
        "broadband_noise": np.clip(rng.normal(0, 0.2, len(t)), -0.95, 0.95),
        "single_impulse": np.zeros_like(t),
        "click_train": np.zeros_like(t),
        "dc_offset_speech": 0.75 * _speech(fs, 4.0, 4303) + 0.12,
    }
    controls["single_impulse"][len(t)//2] = 0.99
    controls["click_train"][::fs//2] = 0.99
    control_rows = []
    for name, waveform in controls.items():
        result, _, _ = analyze_pcm(waveform, fs, f"CONTROL_{name}")
        control_rows.append({"control": name, **summary_features(result), "positive": is_positive(result)})
    control_table = pd.DataFrame(control_rows)

    soft_rows = []
    soft_source = _speech(fs, 4.0, 4401)
    for drive in [1.0, 2.0, 4.0, 8.0]:
        waveform = soft_clip_tanh(soft_source, drive)
        result, _, _ = analyze_pcm(waveform, fs, f"SOFT_{drive}")
        soft_rows.append({"drive": drive, **summary_features(result), "positive": is_positive(result)})
    soft_table = pd.DataFrame(soft_rows)

    quant_rows = []
    quant_source = _speech(fs, 4.0, 4501)
    quant_clipped, _ = hard_clip(quant_source, 0.60)
    for bits in [8, 10, 12, 16, 24]:
        for condition, waveform in [("clean", quant_source), ("hard_clipped", quant_clipped)]:
            result, _, _ = analyze_pcm(waveform, fs, f"Q{bits}_{condition}", bit_depth=bits)
            quant_rows.append({
                "bit_depth": bits,
                "condition": condition,
                **summary_features(result),
                "positive": is_positive(result),
            })
    quant_table = pd.DataFrame(quant_rows)

    transform_rows = [
        {"condition": "baseline", **summary_features(clipped_result)},
        {"condition": "polarity_inversion", **summary_features(inverted_result)},
        {"condition": "common_time_shift", **summary_features(shifted_result)},
    ]
    for gain in [0.25, 0.50, 0.75, 1.00]:
        result, _, _ = analyze_pcm(clipped_float * gain, fs, f"ATTEN_{gain}")
        transform_rows.append({"condition": f"post_clip_gain_{gain}", **summary_features(result)})
    with TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        wav = temporary / "clipped_pcm16.wav"
        sf.write(wav, clipped_pcm, fs, subtype="PCM_16")
        roundtrip, roundtrip_fs = sf.read(wav, always_2d=False, dtype="float64")
        result, _, _ = analyze_pcm(roundtrip, int(roundtrip_fs), "PCM16_ROUNDTRIP")
        transform_rows.append({"condition": "lossless_pcm16_roundtrip", **summary_features(result)})
        resampled = signal.resample_poly(clipped_pcm, 1, 3)
        result, _, _ = analyze_pcm(resampled, 16_000, "RESAMPLED_16K")
        transform_rows.append({"condition": "resampled_16k_characterization", **summary_features(result)})

        if run_codecs and shutil.which("ffmpeg"):
            for label, codec_args, suffix in [
                ("opus_64k", ["-c:a", "libopus", "-b:a", "64k"], ".webm"),
                ("aac_96k", ["-c:a", "aac", "-b:a", "96k"], ".m4a"),
            ]:
                encoded = temporary / f"encoded{suffix}"
                decoded = temporary / f"decoded_{label}.wav"
                completed = subprocess.run(
                    ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(wav), *codec_args, str(encoded)],
                    capture_output=True, text=True,
                )
                if completed.returncode != 0:
                    transform_rows.append({"condition": label, "qdist_status": "codec_unavailable"})
                    continue
                completed = subprocess.run(
                    ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(encoded), "-c:a", "pcm_s16le", str(decoded)],
                    capture_output=True, text=True,
                )
                if completed.returncode != 0:
                    transform_rows.append({"condition": label, "qdist_status": "decode_failed"})
                    continue
                array, rate = sf.read(decoded, always_2d=False, dtype="float64")
                result, _, _ = analyze_pcm(array, int(rate), label)
                transform_rows.append({"condition": label, **summary_features(result)})
    transformations = pd.DataFrame(transform_rows)

    short_source = _speech(fs, 1.0, 4601)
    short_clipped, _ = hard_clip(short_source, 0.60)
    short_result, _, _ = analyze_pcm(short_clipped, fs, "SHORT_SUPPORT")
    long_source = _speech(fs, 4.0, 4602)
    long_clipped, _ = hard_clip(long_source, 0.60)
    long_result, _, _ = analyze_pcm(long_clipped, fs, "LONG_SUPPORT")
    support = pd.DataFrame([
        {"condition": "short_1s", **summary_features(short_result)},
        {"condition": "long_4s", **summary_features(long_result)},
    ])

    dose_order = dose_summary.sort_values("true_clipped_sample_fraction")
    sample_rho = stats.spearmanr(
        dose_order["true_clipped_sample_fraction"],
        dose_order["qdist_hard_clipped_sample_fraction"],
    ).statistic
    frame_rho = stats.spearmanr(
        dose_order["true_clipped_sample_fraction"],
        dose_order["qdist_hard_clipped_frame_fraction"],
    ).statistic
    moderate = dose.loc[dose["true_clipped_sample_fraction"].ge(0.001)]
    baseline = transformations.loc[transformations["condition"].eq("baseline")].iloc[0]
    polarity = transformations.loc[transformations["condition"].eq("polarity_inversion")].iloc[0]
    shifted_row = transformations.loc[transformations["condition"].eq("common_time_shift")].iloc[0]
    lossless = transformations.loc[transformations["condition"].eq("lossless_pcm16_roundtrip")].iloc[0]

    checks = pd.DataFrame([
        {"gate":"G1","check":"frozen production detector API complete","passed":not contract["missing_symbols"],"observed":str(contract["missing_symbols"]),"required":"none missing"},
        {"gate":"G1","check":"legacy frozen measurement identity exact","passed":contract["version_exact"],"observed":contract["legacy_measurement_version"],"required":LEGACY_MEASUREMENT_VERSION},
        {"gate":"G1","check":"exact three-feature registry","passed":contract["features_exact"],"observed":str(contract["legacy_features"]),"required":str(ANALYSIS_FEATURES)},
        {"gate":"G2","check":"clean speech valid zero","passed":not is_positive(clean_result),"observed":summary_features(clean_result),"required":"zero accepted episodes"},
        {"gate":"G2","check":"hard clipping detected","passed":is_positive(clipped_result),"observed":summary_features(clipped_result),"required":"positive"},
        {"gate":"G2","check":"all three views reconstruct from ledgers","passed":bool(reconstruction["passed"].all()),"observed":reconstruction.to_dict("records"),"required":"absolute error <=1e-12"},
        {"gate":"G3","check":"polarity equivariance","passed":all(np.isclose(float(baseline[f]), float(polarity[f]), rtol=0, atol=1e-12) for f in ANALYSIS_FEATURES),"observed":polarity.to_dict(),"required":"all features exact"},
        {"gate":"G3","check":"common time-shift invariance","passed":all(np.isclose(float(baseline[f]), float(shifted_row[f]), rtol=0, atol=1e-12) for f in ANALYSIS_FEATURES),"observed":shifted_row.to_dict(),"required":"all features exact"},
        {"gate":"G3","check":"lossless PCM16 roundtrip","passed":all(np.isclose(float(baseline[f]), float(lossless[f]), rtol=0, atol=1e-12) for f in ANALYSIS_FEATURES),"observed":lossless.to_dict(),"required":"all features exact"},
        {"gate":"G3","check":"resampling and lossy codecs characterized","passed":set(["resampled_16k_characterization"]).issubset(set(transformations["condition"])),"observed":sorted(transformations["condition"].astype(str).tolist()),"required":"native plus characterization rows"},
        {"gate":"G4","check":"sample burden ordered with true clipped burden","passed":bool(np.isfinite(sample_rho) and sample_rho >= 0.90),"observed":sample_rho,"required":">=0.90"},
        {"gate":"G4","check":"frame burden ordered with true clipped burden","passed":bool(np.isfinite(frame_rho) and frame_rho >= 0.90),"observed":frame_rho,"required":">=0.90"},
        {"gate":"G4","check":"synthetic sample precision","passed":bool(moderate["sample_precision"].median() >= 0.99),"observed":float(moderate["sample_precision"].median()),"required":">=0.99"},
        {"gate":"G4","check":"synthetic sample recall","passed":bool(moderate["sample_recall"].median() >= 0.70),"observed":float(moderate["sample_recall"].median()),"required":">=0.70"},
        {"gate":"G5","check":"natural extrema and impulses remain negative","passed":bool(~control_table["positive"].astype(bool).any()),"observed":control_table[["control","positive","qdist_status"]].to_dict("records"),"required":"no positives"},
        {"gate":"G5","check":"clean coarse PCM quantization does not create clipping evidence","passed":bool(~quant_table.loc[quant_table["bit_depth"].le(12) & quant_table["condition"].eq("clean"),"positive"].astype(bool).any()),"observed":quant_table.loc[quant_table["bit_depth"].le(12) & quant_table["condition"].eq("clean")].to_dict("records"),"required":"no clean-signal false positives at 8, 10, or 12 bit; true hard-clipped controls characterized separately"},
        {"gate":"G5","check":"moderate smooth saturation not promoted to hard-clipping feature","passed":bool(~soft_table.loc[soft_table["drive"].le(4.0),"positive"].astype(bool).any()),"observed":soft_table.to_dict("records"),"required":"drive <=4 nonpositive"},
        {"gate":"G6","check":"minimum exposure contract","passed":bool(not is_positive(short_result) and is_positive(long_result)),"observed":support.to_dict("records"),"required":"1s unavailable/nonpositive; 4s measurable positive"},
    ])

    save_table(dose, tables, "qdist_v400_synthetic_dose_long")
    save_table(dose_summary, tables, "qdist_v400_synthetic_dose_summary")
    save_table(reconstruction, tables, "qdist_v400_reconstruction_preflight")
    save_table(control_table, tables, "qdist_v400_discriminant_controls")
    save_table(soft_table, tables, "qdist_v400_soft_scope_characterization")
    save_table(quant_table, tables, "qdist_v400_quantization_characterization")
    save_table(transformations, tables, "qdist_v400_transformation_controls")
    save_table(support, tables, "qdist_v400_support_preflight")
    save_table(checks, validation, "qdist_v400_preflight_checks")

    provenance = {
        "measurement_version": MEASUREMENT_VERSION,
        "legacy_measurement_version": LEGACY_MEASUREMENT_VERSION,
        "reviewed_orchestration_version": REVIEWED_ORCHESTRATION_VERSION,
        "preflight_hotfix_revision": PREFLIGHT_HOTFIX_REVISION,
        "parameter_payload": parameter_payload(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].plot(dose_order["true_clipped_sample_fraction"], dose_order["qdist_hard_clipped_sample_fraction"], marker="o")
    axes[0].set(xlabel="True clipped sample fraction", ylabel="Detected clipped sample fraction", title="Sample-burden response")
    axes[1].plot(dose_order["true_clipped_sample_fraction"], dose_order["qdist_hard_clipped_frame_fraction"], marker="o")
    axes[1].set(xlabel="True clipped sample fraction", ylabel="Clipped frame fraction", title="Frame-prevalence response")
    axes[2].plot(dose_order["true_clipped_sample_fraction"], dose_order["qdist_hard_clip_event_rate_per_min"], marker="o")
    axes[2].set(xlabel="True clipped sample fraction", ylabel="Episodes/min", title="Episode-rate response")
    axes[3].plot(dose_order["true_clipped_sample_fraction"], dose_order["sample_precision"], marker="o", label="Precision")
    axes[3].plot(dose_order["true_clipped_sample_fraction"], dose_order["sample_recall"], marker="o", label="Recall")
    axes[3].set(xlabel="True clipped sample fraction", ylabel="Score", ylim=(-0.02,1.02), title="Exact truth recovery")
    axes[3].legend()
    fig.tight_layout()
    bundle_a = save_figure_bundle(
        fig, figures, "A_construct_response", source=dose,
        caption="Controlled hard-clipping dose response and exact synthetic truth recovery for the three QDIST views. Increasing known clipped-sample burden must produce ordered sample and frame burden, while precision and recall are evaluated against the exact injected mask.",
        provenance={**provenance, "panel":"A"},
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].bar(control_table["control"], control_table["qdist_event_count"])
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set(ylabel="Accepted episodes", title="Natural-extrema and impulse controls")
    axes[1].plot(soft_table["drive"], soft_table["qdist_event_count"], marker="o")
    axes[1].set(xlabel="Tanh drive", ylabel="Accepted episodes", title="Smooth-saturation scope")
    for condition, local in quant_table.groupby("condition"):
        axes[2].plot(local["bit_depth"], local["qdist_event_count"], marker="o", label=condition)
    axes[2].set(xlabel="Declared PCM bit depth", ylabel="Accepted episodes", title="Quantization behavior")
    axes[2].legend()
    axes[3].bar(["Clean", "Hard clipped"], [is_positive(clean_result), is_positive(clipped_result)])
    axes[3].set(ylabel="Detector positive", title="Specificity versus construct response", ylim=(0,1.2))
    fig.tight_layout()
    source_b = pd.concat([
        control_table.assign(source_table="controls"),
        soft_table.assign(source_table="soft_scope"),
        quant_table.assign(source_table="quantization"),
    ], ignore_index=True, sort=False)
    bundle_b = save_figure_bundle(
        fig, figures, "B_discriminant_specificity", source=source_b,
        caption="Discriminant controls for natural extrema, impulses, smooth saturation, and quantization. The reviewed hard-clipping detector must remain specific to plateau-like saturation evidence, fail closed under coarse quantization, and not be interpreted as a general soft-clipping or compression detector.",
        provenance={**provenance, "panel":"B"},
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot = transformations.copy()
    axes[0].bar(plot["condition"], pd.to_numeric(plot["qdist_hard_clipped_frame_fraction"], errors="coerce"))
    axes[0].tick_params(axis="x", rotation=70)
    axes[0].set(ylabel="Frame fraction", title="Frame-prevalence behavior")
    axes[1].bar(plot["condition"], pd.to_numeric(plot["qdist_hard_clip_event_rate_per_min"], errors="coerce"))
    axes[1].tick_params(axis="x", rotation=70)
    axes[1].set(ylabel="Episodes/min", title="Episode-rate behavior")
    axes[2].bar(plot["condition"], pd.to_numeric(plot["qdist_hard_clipped_sample_fraction"], errors="coerce"))
    axes[2].tick_params(axis="x", rotation=70)
    axes[2].set(ylabel="Sample fraction", title="Sample-burden behavior")
    fig.tight_layout()
    bundle_c = save_figure_bundle(
        fig, figures, "C_transformation_contract", source=transformations,
        caption="Transformation contract for polarity inversion, common time shifting, post-clipping attenuation, lossless PCM roundtrip, resampling, and lossy codec characterization. Native-waveform inspection is authoritative; resampling and lossy encoding are characterized because they can erase plateau morphology rather than being treated as invariances.",
        provenance={**provenance, "panel":"C"},
    )
    plt.close(fig)

    figure_index = pd.DataFrame([
        {"panel":"A","stem":"A_construct_response", **bundle_a},
        {"panel":"B","stem":"B_discriminant_specificity", **bundle_b},
        {"panel":"C","stem":"C_transformation_contract", **bundle_c},
    ])
    save_table(figure_index, tables, "qdist_v400_preflight_figure_index")

    blocking_pass = bool(checks["passed"].astype(bool).all())
    manifest = {
        "measurement_version": MEASUREMENT_VERSION,
        "legacy_measurement_version": LEGACY_MEASUREMENT_VERSION,
        "reviewed_orchestration_version": REVIEWED_ORCHESTRATION_VERSION,
        "preflight_hotfix_revision": PREFLIGHT_HOTFIX_REVISION,
        "candidate_only": True,
        "preflight_blocking_checks_pass": blocking_pass,
        "cohort_extraction_completed": False,
        "freeze_allowed": False,
        "publish_and_freeze": False,
        "scientific_review_decision": "PENDING",
        "analysis_features": list(ANALYSIS_FEATURES),
        "family_scalar_constructed": False,
        "standalone_gate_allowed": False,
        "complete_nonlinear_distortion_claim_allowed": False,
        "panels_complete": ["A","B","C"],
        "panel_i_status": "APPLICABLE_pending_event_verification",
        "legacy_contract": contract,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifests / "qdist_v400_preflight_candidate_manifest.json", manifest)
    return {
        "checks": checks,
        "figure_index": figure_index,
        "manifest": manifest,
        "output_root": str(output_root),
    }
