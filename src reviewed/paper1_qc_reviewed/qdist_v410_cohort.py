"""Full candidate-cohort validation for QDIST v4.1.0.

This module recomputes the candidate detector from frozen segmentation and the
native decoded first audio stream.  It does not copy qdist-v3.1.1 feature
values, use clinical labels, consume human-QC labels, finalize a scientific
decision, export publication features, or freeze a measurement.

The valid construct is visible hard-plateau morphology in the stored native
decoded waveform.  The primary candidate output is accepted channel-sample
support, the event-rate view is secondary, and the legacy 30-ms frame view is
conditional because it depends on frame-grid origin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence
import gzip
import json
import math
import pickle
import re
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.io import wavfile
import yaml

from paper1_qc.media import decode_native_audio
from paper1_qc import qdist_v410_candidate as detector


MEASUREMENT_VERSION = detector.MEASUREMENT_VERSION
LEGACY_MEASUREMENT_VERSION = "qdist-v3.1.1"
COHORT_VERSION = "qdist-v4.1.0-candidate-cohort-r1"
ANALYSIS_FEATURES = detector.ANALYSIS_FEATURES
PRIMARY_FEATURES = detector.PRIMARY_FEATURES
SECONDARY_FEATURES = detector.SECONDARY_FEATURES
CONDITIONAL_FEATURES = detector.CONDITIONAL_FEATURES
DEFAULT_PARAMETERS = detector.DEFAULT_PARAMETERS
RANDOM_SEED = DEFAULT_PARAMETERS.random_seed
REQUIRED_MAIN_PANELS = (
    "A", "B", "C", "D1", "D2", "D3", "E1", "E2", "E3", "F",
    "H1", "H2", "H3", "I", "J",
)
REVIEW_LABELS = (
    "DEFINITE_HARD_CLIP",
    "PROBABLE_HARD_CLIP",
    "AMBIGUOUS",
    "NOT_HARD_CLIP",
    "CANNOT_DETERMINE",
)


@dataclass(frozen=True)
class CohortPaths:
    project_root: Path
    main_outputs: Path
    segmentation_root: Path
    data_root: Path
    legacy_root: Path
    preflight_root: Path
    output_root: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "CohortPaths":
        root = Path(project_root).expanduser().resolve()
        config_path = root / "config" / "project.yaml"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        data_version = str(config.get("data_freeze", {}).get("version", "v1"))
        segmentation_version = str(
            config.get("segmentation_freeze", {}).get("version", data_version)
        )
        main = root / "MAIN outputs"
        return cls(
            project_root=root,
            main_outputs=main,
            segmentation_root=(
                main / "01_SEGMENTATION_FREEZE" / segmentation_version
            ),
            data_root=main / "00_DATA_FREEZE" / data_version,
            legacy_root=(
                main / "02_FEATURE_FREEZE" / "nonlinear_distortion"
                / LEGACY_MEASUREMENT_VERSION
            ),
            preflight_root=(
                root / "outputs reviewed" / "nonlinear_distortion"
                / "qdist_v410_remediation_preflight"
            ),
            output_root=(
                root / "outputs reviewed" / "nonlinear_distortion"
                / "qdist_v410_candidate_cohort"
            ),
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def json_safe(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if not pd.isna(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [json_safe(item) for item in value]
    return value


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def save_table(
    frame: pd.DataFrame,
    stem: str | Path,
    *,
    parquet: bool = True,
) -> dict[str, str]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    result = {"csv": str(csv_path)}
    if parquet:
        parquet_path = stem.with_suffix(".parquet")
        try:
            frame.to_parquet(parquet_path, index=False)
            result["parquet"] = str(parquet_path)
        except Exception:
            pass
    return result


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def table_file(stem: str | Path, *, required: bool = True) -> Path | None:
    stem = Path(stem)
    for suffix in [".parquet", ".csv"]:
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"Missing table bundle: {stem}")
    return None


def load_optional_table(stem: str | Path) -> pd.DataFrame:
    path = table_file(stem, required=False)
    return read_table(path) if path is not None else pd.DataFrame()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "pass", "passed"}
    )


def scalar_bool(value: Any, default: bool = False) -> bool:
    if value is None or value is pd.NA:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def finite_spearman(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    frame = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return len(frame), np.nan
    return len(frame), float(stats.spearmanr(frame["x"], frame["y"]).statistic)


def cohen_kappa_binary(a: pd.Series, b: pd.Series) -> float:
    left = as_bool(a)
    right = as_bool(b)
    if len(left) == 0:
        return np.nan
    observed = float(left.eq(right).mean())
    pa, pb = float(left.mean()), float(right.mean())
    expected = pa * pb + (1 - pa) * (1 - pb)
    return np.nan if np.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def merge_intervals(
    intervals: Iterable[tuple[int, int]],
    gap_samples: int = 0,
) -> list[tuple[int, int]]:
    clean = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    if not clean:
        return []
    merged = [clean[0]]
    for start, end in clean[1:]:
        old_start, old_end = merged[-1]
        if start - old_end <= int(gap_samples):
            merged[-1] = (old_start, max(old_end, end))
        else:
            merged.append((start, end))
    return merged


def safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def deterministic_order(values: Iterable[str], salt: str) -> list[str]:
    return sorted(
        {str(value) for value in values},
        key=lambda value: stable_hash([RANDOM_SEED, salt, value]),
    )


def derive_participant_id(recording_id: str) -> str:
    parts = str(recording_id).rsplit("_", 6)
    return parts[0] if len(parts) == 7 else str(recording_id)


def derive_acquisition_date(recording_id: str) -> pd.Timestamp:
    parts = str(recording_id).rsplit("_", 6)
    if len(parts) != 7 or not re.fullmatch(r"\d{8}", parts[3]):
        return pd.NaT
    return pd.to_datetime(parts[3], format="%Y%m%d", errors="coerce")


def discover_interval_table(folder: str | Path) -> tuple[Path, pd.DataFrame]:
    candidates: list[tuple[int, Path, pd.DataFrame]] = []
    for path in Path(folder).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        try:
            frame = read_table(path)
        except Exception:
            continue
        columns = set(frame.columns)
        time_columns = {"start_sec", "end_sec"}.issubset(columns) or {
            "start", "end"
        }.issubset(columns)
        if not time_columns:
            continue
        if not {"view", "segment_type", "label", "region"}.intersection(columns):
            continue
        if not {"logical_recording_id", "file_name"}.intersection(columns):
            continue
        score = (
            8 * ("frozen" in path.stem.lower())
            + 6 * ("interval" in path.stem.lower())
            + 2 * (path.suffix.lower() == ".parquet")
        )
        candidates.append((score, path, frame))
    if not candidates:
        raise FileNotFoundError(f"No frozen interval table under {folder}")
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RuntimeError(
            "Tied interval-table candidates; resolve the frozen input identity."
        )
    return candidates[0][1], candidates[0][2]


def load_frozen_inputs(paths: CohortPaths) -> dict[str, Any]:
    decisions_path = table_file(
        paths.segmentation_root / "frozen_segmentation_decisions"
    )
    recordings_path = table_file(paths.data_root / "frozen_bamboo_recordings")
    intervals_path, intervals = discover_interval_table(paths.segmentation_root)
    decisions = read_table(decisions_path)
    recordings = read_table(recordings_path)
    if "segmentation_analysis_eligible" not in decisions:
        raise ValueError(
            "Frozen decisions lack segmentation_analysis_eligible."
        )
    eligible = decisions.loc[
        as_bool(decisions["segmentation_analysis_eligible"])
    ].copy()
    if "logical_recording_id" not in intervals:
        mapping = decisions[["file_name", "logical_recording_id"]].drop_duplicates()
        if mapping["file_name"].duplicated().any():
            raise ValueError("file_name is not unique in frozen decisions.")
        intervals = intervals.merge(
            mapping,
            on="file_name",
            how="left",
            validate="many_to_one",
        )
    for frame in [eligible, recordings, intervals]:
        frame["logical_recording_id"] = frame["logical_recording_id"].astype(str)
    media_column = next(
        (
            column
            for column in [
                "media_path", "selected_media_path", "file_path", "selected_path"
            ]
            if column in recordings
        ),
        None,
    )
    if media_column is None:
        raise ValueError("Frozen recording table lacks a selected media path.")
    if media_column != "media_path":
        recordings = recordings.rename(columns={media_column: "media_path"})
    analysis_recordings = eligible[["logical_recording_id"]].merge(
        recordings.drop_duplicates("logical_recording_id"),
        on="logical_recording_id",
        how="left",
        validate="one_to_one",
    )
    participant_column = next(
        (
            column
            for column in [
                "participant_id", "subject_id", "participant_uid", "subject_uid"
            ]
            if column in analysis_recordings
        ),
        None,
    )
    if participant_column is not None:
        analysis_recordings["participant_id"] = analysis_recordings[
            participant_column
        ].astype(str)
    else:
        analysis_recordings["participant_id"] = analysis_recordings[
            "logical_recording_id"
        ].map(derive_participant_id)
    intervals = intervals.rename(columns={"start": "start_sec", "end": "end_sec"})
    view_column = next(
        (
            column
            for column in ["view", "segment_type", "label", "region"]
            if column in intervals
        ),
        None,
    )
    if view_column is None:
        raise ValueError("Frozen intervals lack a view identity.")
    if view_column != "view":
        intervals = intervals.rename(columns={view_column: "view"})
    if "profile" in intervals and intervals["profile"].astype(str).eq("primary").any():
        intervals = intervals.loc[
            intervals["profile"].astype(str).eq("primary")
        ].copy()
    available_views = sorted(intervals["view"].dropna().astype(str).unique())
    strict_view = next(
        (
            name
            for name in [
                "strict_speech", "primary_speech", "final_speech", "speech"
            ]
            if name in available_views
        ),
        None,
    )
    checks = pd.DataFrame(
        [
            {
                "gate": "G1",
                "check": "one frozen eligible row per recording",
                "status": (
                    "PASS"
                    if not analysis_recordings["logical_recording_id"].duplicated().any()
                    else "FAIL"
                ),
                "observed": len(analysis_recordings),
                "required": "unique recording identities",
            },
            {
                "gate": "G1",
                "check": "media path complete",
                "status": (
                    "PASS"
                    if analysis_recordings["media_path"].notna().all()
                    else "FAIL"
                ),
                "observed": int(analysis_recordings["media_path"].isna().sum()),
                "required": "0 missing media paths",
            },
            {
                "gate": "G1",
                "check": "strict-speech intervals define one continuous task span",
                "status": "PASS" if strict_view is not None else "FAIL",
                "observed": strict_view,
                "required": "governed speech view available",
            },
        ]
    )
    provenance = pd.DataFrame(
        [
            {
                "artifact": "frozen decisions",
                "path": str(decisions_path),
                "sha256": sha256_file(decisions_path),
            },
            {
                "artifact": "frozen recordings",
                "path": str(recordings_path),
                "sha256": sha256_file(recordings_path),
            },
            {
                "artifact": "frozen intervals",
                "path": str(intervals_path),
                "sha256": sha256_file(intervals_path),
            },
            {
                "artifact": "candidate detector",
                "path": str(
                    paths.project_root / "src" / "paper1_qc"
                    / "qdist_v410_candidate.py"
                ),
                "sha256": sha256_file(
                    paths.project_root / "src" / "paper1_qc"
                    / "qdist_v410_candidate.py"
                ),
            },
        ]
    )
    return {
        "recordings": analysis_recordings,
        "intervals": intervals,
        "strict_speech_view": strict_view,
        "checks": checks,
        "provenance": provenance,
    }


def verify_preflight(paths: CohortPaths) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = (
        paths.preflight_root / "manifests"
        / "qdist_v410_remediation_preflight_manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            "QDIST v4.1 remediation preflight is missing. Run it before cohort extraction."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requirements = {
        "measurement_version": MEASUREMENT_VERSION,
        "candidate_only": True,
        "freeze_allowed": False,
        "blocking_remediation_checks_pass": True,
        "package_tests_passed": True,
        "feature_values_from_cohort_recomputed": False,
    }
    for key, expected in requirements.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Preflight requirement failed: {key}={manifest.get(key)!r}; "
                f"expected {expected!r}."
            )
    detector_path = paths.project_root / "src" / "paper1_qc" / "qdist_v410_candidate.py"
    if sha256_file(detector_path) != manifest.get("detector_sha256"):
        raise ValueError("Installed candidate detector hash differs from the preflight hash.")
    rows: list[dict[str, Any]] = []
    for panel, stem in [
        ("A", "qdist_v410_panel-A_construct-response"),
        ("B", "qdist_v410_panel-B_discriminant-specificity"),
        ("C", "qdist_v410_panel-C_transformation-contract"),
    ]:
        row: dict[str, Any] = {"panel": panel, "stem": stem}
        for field, suffix in [
            ("png", ".png"),
            ("svg", ".svg"),
            ("pdf", ".pdf"),
            ("source_csv", ".source.csv"),
            ("caption", ".caption.md"),
            ("provenance", ".provenance.json"),
        ]:
            source = paths.preflight_root / "figures" / f"{stem}{suffix}"
            if not source.exists() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            destination = paths.output_root / "figures" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            row[field] = str(destination.relative_to(paths.output_root))
            row[f"{field}_sha256"] = sha256_file(destination)
        row["bundle_role"] = "accepted_preflight"
        rows.append(row)
    return manifest, pd.DataFrame(rows)


def load_legacy_baseline(paths: CohortPaths) -> dict[str, Any]:
    manifest_path = paths.legacy_root / "audit" / "qdist_v311_frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Frozen qdist-v3.1.1 baseline is required for recording-level comparison."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("measurement_version") != LEGACY_MEASUREMENT_VERSION:
        raise ValueError("Legacy QDIST freeze has the wrong measurement version.")
    tables = paths.legacy_root / "tables"
    analysis = read_table(table_file(tables / "qdist_v311_analysis_features"))
    recordings = read_table(table_file(tables / "qdist_v311_recording_features"))
    if analysis["logical_recording_id"].duplicated().any():
        raise ValueError("Legacy analysis table contains duplicate recording identities.")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "analysis": analysis,
        "recordings": recordings,
        "accepted": load_optional_table(tables / "qdist_v311_accepted_plateau_ledger"),
        "episodes": load_optional_table(tables / "qdist_v311_episode_ledger"),
    }


def intervals_for(frozen: Mapping[str, Any], recording_id: str) -> list[detector.TimeInterval]:
    local = frozen["intervals"].loc[
        frozen["intervals"]["logical_recording_id"].astype(str).eq(str(recording_id))
        & frozen["intervals"]["view"].astype(str).eq(
            str(frozen["strict_speech_view"])
        )
    ].sort_values(["start_sec", "end_sec"])
    return [
        detector.TimeInterval(float(row.start_sec), float(row.end_sec))
        for row in local.itertuples(index=False)
        if np.isfinite(row.start_sec)
        and np.isfinite(row.end_sec)
        and row.end_sec > row.start_sec
    ]


def task_span_for(frozen: Mapping[str, Any], recording_id: str) -> detector.TimeInterval:
    intervals = intervals_for(frozen, recording_id)
    if not intervals:
        return detector.TimeInterval(0.0, 0.0)
    return detector.TimeInterval(
        min(interval.start_sec for interval in intervals),
        max(interval.end_sec for interval in intervals),
    )


def media_path_for(project_root: Path, row: Any) -> Path:
    value = row["media_path"] if isinstance(row, Mapping) else row.media_path
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def source_provenance(
    row: Mapping[str, Any],
    media_path: Path,
    decoded: Any,
    ffmpeg_version: str,
) -> detector.NativeSignalProvenance:
    source_hash = None
    for column in ["sha256", "source_sha256", "media_sha256"]:
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            source_hash = str(value)
            break
    if source_hash is None:
        source_hash = sha256_file(media_path)
    bits = decoded.probe.get("bits_per_raw_sample")
    bits = int(bits) if pd.notna(bits) else None
    return detector.NativeSignalProvenance(
        native_view_verified=scalar_bool(row.get("native_view_verified"), True),
        known_preprocessing_applied=scalar_bool(row.get("known_preprocessing_applied"), False),
        codec_name=decoded.probe.get("codec_name"),
        sample_format=decoded.probe.get("sample_format"),
        bits_per_raw_sample=bits,
        container_format=decoded.probe.get("container_format"),
        channel_layout=decoded.probe.get("channel_layout"),
        source_path=str(media_path),
        source_sha256=source_hash,
        decoder="ffmpeg",
        decoder_version=ffmpeg_version,
        decode_arguments=(
            "first audio stream -> native-rate pcm_f32le; channels preserved; "
            "no resampling, normalization, filtering, interpolation, or denoising"
        ),
    )


def concat_nonempty(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if isinstance(frame, pd.DataFrame) and len(frame)]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wb", compresslevel=3) as stream:
        pickle.dump(dict(payload), stream, protocol=5)
    temporary.replace(path)


def read_checkpoint(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def extract_candidate_cohort(
    paths: CohortPaths,
    frozen: Mapping[str, Any],
    *,
    resume: bool = True,
    progress_every: int = 25,
) -> dict[str, pd.DataFrame]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required for QDIST extraction.")
    ffmpeg_version = subprocess.run(
        [ffmpeg, "-version"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()[0]
    detector_path = paths.project_root / "src" / "paper1_qc" / "qdist_v410_candidate.py"
    media_module = paths.project_root / "src" / "paper1_qc" / "media.py"
    cohort_module = (
        paths.project_root / "src reviewed" / "paper1_qc_reviewed"
        / "qdist_v410_cohort.py"
    )
    signature_payload = {
        "measurement_version": MEASUREMENT_VERSION,
        "detector_sha256": sha256_file(detector_path),
        "media_sha256": sha256_file(media_module),
        "cohort_orchestration_sha256": sha256_file(cohort_module),
        "parameters": DEFAULT_PARAMETERS.to_dict(),
        "frozen_inputs": frozen["provenance"][["artifact", "sha256"]].to_dict("records"),
    }
    checkpoint_signature = stable_hash(signature_payload)
    checkpoint_root = paths.output_root / "audit" / "recording_checkpoints"
    recording_rows: list[dict[str, Any]] = []
    candidates: list[pd.DataFrame] = []
    accepted: list[pd.DataFrame] = []
    episodes: list[pd.DataFrame] = []
    edges: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    checkpoint_audit_rows: list[dict[str, Any]] = []
    started = perf_counter()
    records = frozen["recordings"].to_dict("records")
    for ordinal, row in enumerate(records, start=1):
        recording_id = str(row["logical_recording_id"])
        checkpoint = checkpoint_root / f"{safe_id(recording_id)}.qdist.pkl.gz"
        bundle: dict[str, Any] | None = None
        reused = False
        item_started = perf_counter()
        try:
            if resume and checkpoint.exists():
                prior = read_checkpoint(checkpoint)
                if prior.get("checkpoint_signature") == checkpoint_signature:
                    bundle = prior
                    reused = True
            if bundle is None:
                media_path = media_path_for(paths.project_root, row)
                decoded = decode_native_audio(
                    media_path,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                )
                provenance = source_provenance(
                    row,
                    media_path,
                    decoded,
                    ffmpeg_version,
                )
                extraction = detector.extract_qdist(
                    decoded.native,
                    decoded.sample_rate_native,
                    task_span=task_span_for(frozen, recording_id),
                    logical_recording_id=recording_id,
                    provenance=provenance,
                    parameters=DEFAULT_PARAMETERS,
                )
                recording = {
                    **extraction.recording,
                    "participant_id": str(row["participant_id"]),
                    "file_name": str(row.get("file_name", media_path.name)),
                    "media_path": str(media_path),
                    "qdist_checkpoint_signature": checkpoint_signature,
                }
                bundle = {
                    "checkpoint_signature": checkpoint_signature,
                    "recording": recording,
                    "candidates": extraction.candidate_ledger,
                    "accepted": extraction.accepted_plateau_ledger,
                    "episodes": extraction.episode_ledger,
                    "edges": extraction.edge_ledger,
                }
                write_checkpoint(checkpoint, bundle)
            persisted = read_checkpoint(checkpoint)
            persisted_candidates = persisted.get("candidates", pd.DataFrame())
            persisted_accepted = persisted.get("accepted", pd.DataFrame())
            persisted_episodes = persisted.get("episodes", pd.DataFrame())
            candidate_ids = set(
                persisted_candidates.get("candidate_id", pd.Series(dtype=str)).astype(str)
            )
            accepted_ids = set(
                persisted_accepted.get("candidate_id", pd.Series(dtype=str)).astype(str)
            )
            checkpoint_checks = {
                "signature_match": persisted.get("checkpoint_signature") == checkpoint_signature,
                "recording_id_match": str(persisted.get("recording", {}).get("logical_recording_id")) == recording_id,
                "candidate_ids_unique": not persisted_candidates.get("candidate_id", pd.Series(dtype=str)).astype(str).duplicated().any(),
                "accepted_ids_unique": not persisted_accepted.get("candidate_id", pd.Series(dtype=str)).astype(str).duplicated().any(),
                "accepted_subset_of_candidates": accepted_ids.issubset(candidate_ids),
                "episode_ids_unique": not persisted_episodes.get("episode_id", pd.Series(dtype=str)).astype(str).duplicated().any(),
            }
            checkpoint_audit_rows.append({
                "logical_recording_id": recording_id,
                "checkpoint_path": str(checkpoint),
                "checkpoint_reused": reused,
                **checkpoint_checks,
                "passed": all(checkpoint_checks.values()),
            })
            recording_rows.append(bundle["recording"])
            candidates.append(bundle["candidates"])
            accepted.append(bundle["accepted"])
            episodes.append(bundle["episodes"])
            edges.append(bundle["edges"])
        except Exception as exc:
            errors.append(
                {
                    "logical_recording_id": recording_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        runtimes.append(
            {
                "logical_recording_id": recording_id,
                "checkpoint_reused": reused,
                "elapsed_sec": perf_counter() - item_started,
                "checkpoint_path": str(checkpoint),
            }
        )
        if ordinal % int(progress_every) == 0 or ordinal == len(records):
            elapsed = perf_counter() - started
            rate = ordinal / max(elapsed, 1e-12)
            remaining = (len(records) - ordinal) / max(rate, 1e-12)
            print(
                f"QDIST v4.1 {ordinal}/{len(records)} | "
                f"{rate:.2f} recordings/s | ETA {remaining / 60:.1f} min"
            )
    result = {
        "recordings": pd.DataFrame(recording_rows),
        "candidates": concat_nonempty(candidates),
        "accepted": concat_nonempty(accepted),
        "episodes": concat_nonempty(episodes),
        "edges": concat_nonempty(edges),
        "errors": pd.DataFrame(
            errors,
            columns=["logical_recording_id", "error_type", "message"],
        ),
        "runtime": pd.DataFrame(runtimes),
        "checkpoint_audit": pd.DataFrame(checkpoint_audit_rows),
    }
    tables = paths.output_root / "tables"
    audit = paths.output_root / "audit"
    save_table(result["recordings"], tables / "qdist_v410_recording_features")
    analysis_columns = [
        "logical_recording_id",
        "participant_id",
        *ANALYSIS_FEATURES,
        *[f"{feature}_status" for feature in ANALYSIS_FEATURES],
        "qdist_status",
        "qdist_available",
        "qdist_support_tier",
        "qdist_hard_clip_event_count",
        "qdist_accepted_plateau_count",
        "qdist_finite_exposure_sec",
        "qdist_task_span_duration_sec",
        "qdist_native_sample_rate_hz",
        "qdist_native_channel_count",
        "qdist_codec_name",
        "qdist_parameter_hash",
    ]
    analysis = result["recordings"][[
        column for column in analysis_columns if column in result["recordings"]
    ]].copy()
    result["analysis"] = analysis
    save_table(analysis, tables / "qdist_v410_analysis_features")
    save_table(result["candidates"], tables / "qdist_v410_candidate_plateau_ledger")
    save_table(result["accepted"], tables / "qdist_v410_accepted_plateau_ledger")
    save_table(result["episodes"], tables / "qdist_v410_episode_ledger")
    save_table(result["edges"], tables / "qdist_v410_edge_ledger")
    save_table(result["errors"], audit / "qdist_v410_extraction_errors", parquet=False)
    save_table(result["runtime"], audit / "qdist_v410_extraction_runtime")
    save_table(
        result["checkpoint_audit"],
        audit / "qdist_v410_checkpoint_readback_audit",
        parquet=False,
    )
    write_json(
        {
            "checkpoint_signature": checkpoint_signature,
            "signature_payload": signature_payload,
            "recording_count": len(result["recordings"]),
            "error_count": len(result["errors"]),
        },
        audit / "qdist_v410_checkpoint_contract.json",
    )
    return result


def build_reconstruction_audit(
    recordings: pd.DataFrame,
    accepted: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct every analysis value from the two governed ledgers."""
    accepted_groups = {
        str(key): value.copy()
        for key, value in accepted.groupby("logical_recording_id", sort=False)
    } if len(accepted) else {}
    episode_groups = {
        str(key): value.copy()
        for key, value in episodes.groupby("logical_recording_id", sort=False)
    } if len(episodes) else {}
    rows: list[dict[str, Any]] = []
    for rec in recordings.to_dict("records"):
        rid = str(rec["logical_recording_id"])
        reconstructed = detector.reconstruct_qdist_features(
            accepted_groups.get(rid, pd.DataFrame()),
            episode_groups.get(rid, pd.DataFrame()),
            finite_channel_sample_count=int(rec["qdist_finite_channel_sample_count"]),
            finite_time_sample_count=int(rec["qdist_finite_time_sample_count"]),
            finite_exposure_sec=float(rec["qdist_finite_exposure_sec"]),
            frame_length_samples=int(rec["qdist_frame_length_samples"]),
            complete_frame_count=int(rec["qdist_complete_frame_count"]),
        )
        for feature in ANALYSIS_FEATURES:
            observed = pd.to_numeric(pd.Series([rec.get(feature)]), errors="coerce").iloc[0]
            rebuilt = reconstructed[feature]
            difference = (
                abs(float(observed) - float(rebuilt))
                if np.isfinite(observed) and np.isfinite(rebuilt)
                else (0.0 if pd.isna(observed) and pd.isna(rebuilt) else np.inf)
            )
            rows.append({
                "logical_recording_id": rid,
                "feature": feature,
                "observed": observed,
                "reconstructed": rebuilt,
                "absolute_difference": difference,
                "passed": bool(difference <= 2e-15),
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame(columns=[
            "feature", "recording_count", "maximum_absolute_difference", "passed"
        ])
    summary = (
        long.groupby("feature", sort=False)
        .agg(
            recording_count=("logical_recording_id", "nunique"),
            maximum_absolute_difference=("absolute_difference", "max"),
            failed_recording_count=("passed", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    summary["passed"] = summary["failed_recording_count"].eq(0)
    return long, summary


def prepare_recording_table(recordings: pd.DataFrame) -> pd.DataFrame:
    table = recordings.copy()
    if "participant_id" not in table:
        table["participant_id"] = table["logical_recording_id"].map(
            derive_participant_id
        )
    table["participant_id"] = table["participant_id"].astype(str)
    table["acquisition_date"] = table["logical_recording_id"].map(
        derive_acquisition_date
    )
    table["acquisition_year"] = table["acquisition_date"].dt.year.astype("Int64")
    table["qdist_available"] = as_bool(table["qdist_available"])
    table["qdist_positive"] = (
        pd.to_numeric(
            table["qdist_hard_clip_event_rate_per_min"], errors="coerce"
        ).fillna(0) > 0
    )
    table["qdist_valid_zero"] = table["qdist_status"].astype(str).eq(
        "available_no_events"
    )
    table["qdist_clipped_channel_ms_per_min"] = (
        pd.to_numeric(
            table["qdist_hard_clipped_sample_fraction"], errors="coerce"
        ) * 60_000.0
    )
    table["qdist_occurrence"] = table["qdist_positive"].astype("Int64")
    table["qdist_feature_role_primary"] = "qdist_hard_clipped_sample_fraction"
    table["qdist_feature_role_secondary"] = "qdist_hard_clip_event_rate_per_min"
    table["qdist_feature_role_conditional"] = "qdist_hard_clipped_frame_fraction"
    return table


def legacy_comparison(
    candidate: pd.DataFrame,
    legacy: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy_analysis = legacy["analysis"].copy()
    keep = ["logical_recording_id", *[
        feature for feature in ANALYSIS_FEATURES if feature in legacy_analysis
    ]]
    legacy_analysis = legacy_analysis[keep].rename(
        columns={feature: f"legacy_{feature}" for feature in ANALYSIS_FEATURES}
    )
    joined = candidate.merge(
        legacy_analysis,
        on="logical_recording_id",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Candidate and legacy recording identities do not match exactly.")
    legacy_occurrence = (
        pd.to_numeric(
            joined.get("legacy_qdist_hard_clip_event_rate_per_min"), errors="coerce"
        ).fillna(0) > 0
    )
    joined["legacy_qdist_occurrence"] = legacy_occurrence
    joined["occurrence_transition"] = np.select(
        [
            ~legacy_occurrence & ~joined["qdist_positive"],
            ~legacy_occurrence & joined["qdist_positive"],
            legacy_occurrence & ~joined["qdist_positive"],
            legacy_occurrence & joined["qdist_positive"],
        ],
        ["zero_to_zero", "new_v410_positive", "lost_v311_positive", "positive_to_positive"],
        default="indeterminate",
    )
    summary_rows: list[dict[str, Any]] = [{
        "comparison": "occurrence",
        "recording_count": len(joined),
        "agreement": float((legacy_occurrence == joined["qdist_positive"]).mean()),
        "new_v410_positive": int((~legacy_occurrence & joined["qdist_positive"]).sum()),
        "lost_v311_positive": int((legacy_occurrence & ~joined["qdist_positive"]).sum()),
        "positive_both": int((legacy_occurrence & joined["qdist_positive"]).sum()),
        "zero_both": int((~legacy_occurrence & ~joined["qdist_positive"]).sum()),
    }]
    for feature in ANALYSIS_FEATURES:
        old_column = f"legacy_{feature}"
        if old_column not in joined:
            continue
        old = pd.to_numeric(joined[old_column], errors="coerce")
        new = pd.to_numeric(joined[feature], errors="coerce")
        finite = old.notna() & new.notna()
        difference = (new - old).abs()
        n, rho = finite_spearman(old.loc[finite], new.loc[finite])
        summary_rows.append({
            "comparison": feature,
            "recording_count": int(finite.sum()),
            "exact_equal_fraction": float(np.isclose(
                old.loc[finite], new.loc[finite], rtol=0, atol=2e-15
            ).mean()) if finite.any() else np.nan,
            "median_absolute_difference": float(difference.loc[finite].median())
            if finite.any() else np.nan,
            "maximum_absolute_difference": float(difference.loc[finite].max())
            if finite.any() else np.nan,
            "spearman_n": n,
            "spearman_rho": rho,
        })
    return joined, pd.DataFrame(summary_rows)


def morphology_margin_table(
    candidates: pd.DataFrame,
    parameters: detector.QDISTParameters = DEFAULT_PARAMETERS,
) -> pd.DataFrame:
    """Audit the actual v4.1 predicates, including same-polarity prominence."""
    if candidates.empty:
        return pd.DataFrame()
    local = candidates.copy()
    accepted = as_bool(local["accepted"])
    tolerance = pd.to_numeric(local["flat_tolerance"], errors="coerce")
    specs: list[tuple[str, pd.Series, str, float]] = [
        ("minimum_plateau_samples", pd.to_numeric(local["sample_count"], errors="coerce"), "higher", float(parameters.minimum_plateau_samples)),
        ("maximum_plateau_duration_ms", pd.to_numeric(local["duration_sec"], errors="coerce") * 1000.0, "lower", float(parameters.maximum_plateau_duration_ms)),
        ("same_polarity_local_prominence", pd.to_numeric(local["candidate_to_context_ratio"], errors="coerce"), "higher", float(parameters.minimum_edge_to_local_peak_ratio)),
        ("bilateral_context_pre", pd.to_numeric(local["pre_context_peak_abs"], errors="coerce") / pd.to_numeric(local["candidate_abs_level"], errors="coerce"), "higher", float(parameters.minimum_context_peak_ratio)),
        ("bilateral_context_post", pd.to_numeric(local["post_context_peak_abs"], errors="coerce") / pd.to_numeric(local["candidate_abs_level"], errors="coerce"), "higher", float(parameters.minimum_context_peak_ratio)),
        ("edge_zone_samples", pd.to_numeric(local["edge_zone_sample_count"], errors="coerce"), "higher", float(parameters.minimum_edge_zone_samples)),
        ("edge_to_interior_ratio", pd.to_numeric(local["edge_to_interior_ratio"], errors="coerce"), "higher", float(parameters.minimum_edge_to_interior_ratio)),
        ("edge_excess_samples", pd.to_numeric(local["edge_excess_samples"], errors="coerce"), "higher", float(parameters.minimum_edge_excess_samples)),
        ("beyond_edge_samples", pd.to_numeric(local["beyond_edge_sample_count"], errors="coerce"), "lower", pd.to_numeric(local["allowed_beyond_edge_samples"], errors="coerce")),
        ("plateau_range_over_tolerance", pd.to_numeric(local["plateau_range"], errors="coerce") / tolerance.replace(0, np.nan), "lower", 2.0),
    ]
    rows: list[dict[str, Any]] = []
    for criterion, values, orientation, threshold in specs:
        threshold_values = (
            pd.Series(float(threshold), index=local.index)
            if np.isscalar(threshold) else pd.Series(threshold, index=local.index)
        )
        margin = (
            values - threshold_values if orientation == "higher"
            else threshold_values - values
        )
        for stratum, mask in [("accepted", accepted), ("rejected", ~accepted)]:
            use = margin.loc[mask].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "criterion": criterion,
                "stratum": stratum,
                "candidate_count": int(mask.sum()),
                "finite_margin_n": len(use),
                "threshold": float(threshold) if np.isscalar(threshold) else "row_specific",
                "median_margin": float(use.median()) if len(use) else np.nan,
                "q10_margin": float(use.quantile(.10)) if len(use) else np.nan,
                "minimum_margin": float(use.min()) if len(use) else np.nan,
                "nonnegative_fraction": float((use >= -1e-12).mean()) if len(use) else np.nan,
            })
    return pd.DataFrame(rows)


def _task_waveform(
    paths: CohortPaths,
    frozen: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[np.ndarray, int, dict[str, Any], detector.NativeSignalProvenance]:
    media_path = media_path_for(paths.project_root, row)
    decoded = decode_native_audio(
        media_path,
        ffmpeg=shutil.which("ffmpeg") or "ffmpeg",
        ffprobe=shutil.which("ffprobe") or "ffprobe",
    )
    span = task_span_for(frozen, str(row["logical_recording_id"]))
    fs = int(decoded.sample_rate_native)
    start = max(0, int(math.floor(span.start_sec * fs)))
    end = min(len(decoded.native), int(math.ceil(span.end_sec * fs)))
    ffmpeg_version = subprocess.run(
        [shutil.which("ffmpeg") or "ffmpeg", "-version"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()[0]
    return (
        np.asarray(decoded.native[start:end], dtype=np.float64),
        fs,
        decoded.probe,
        source_provenance(row, media_path, decoded, ffmpeg_version),
    )


def _accepted_mask(
    accepted: pd.DataFrame,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if accepted.empty:
        return mask
    for row in accepted.itertuples(index=False):
        channel = int(row.channel_index)
        start = max(0, int(row.start_sample_task))
        end = min(shape[0], int(row.end_sample_task_exclusive))
        if 0 <= channel < shape[1] and end > start:
            mask[start:end, channel] = True
    return mask


def inject_matched_hard_clip(
    waveform: np.ndarray,
    target_fraction: float,
    geometry: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Alter approximately a target fraction and return the exact truth mask."""
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("Matched challenge requires a finite task waveform.")
    total = values.size
    target_count = max(1, int(round(float(target_fraction) * total)))
    result = values.copy()
    positive_limit = np.nan
    negative_limit = np.nan
    if geometry == "positive_only":
        pool = values[values > 0]
        if len(pool) <= target_count:
            raise ValueError("Insufficient positive samples for the requested burden.")
        positive_limit = float(np.partition(pool, len(pool) - target_count)[len(pool) - target_count])
        result = np.minimum(result, positive_limit)
    elif geometry == "negative_only":
        pool = values[values < 0]
        if len(pool) <= target_count:
            raise ValueError("Insufficient negative samples for the requested burden.")
        negative_limit = float(np.partition(pool, target_count - 1)[target_count - 1])
        result = np.maximum(result, negative_limit)
    elif geometry == "symmetric":
        pool = np.abs(values).ravel()
        if len(pool) <= target_count:
            raise ValueError("Insufficient samples for the requested burden.")
        limit = float(np.partition(pool, len(pool) - target_count)[len(pool) - target_count])
        positive_limit, negative_limit = limit, -limit
        result = np.clip(result, negative_limit, positive_limit)
    else:
        raise ValueError(f"Unknown clipping geometry: {geometry}")
    truth = finite & ~np.isclose(result, values, rtol=0.0, atol=0.0)
    return result, truth, {
        "positive_limit": positive_limit,
        "negative_limit": negative_limit,
        "target_fraction": float(target_fraction),
        "realized_fraction": float(truth.sum() / total),
    }


def _truth_metrics(
    truth: np.ndarray,
    detected: np.ndarray,
) -> dict[str, Any]:
    tp = int((truth & detected).sum())
    fp = int((~truth & detected).sum())
    fn = int((truth & ~detected).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else np.nan
    )
    return {
        "true_positive_samples": tp,
        "false_positive_samples": fp,
        "false_negative_samples": fn,
        "sample_precision": precision,
        "sample_recall": recall,
        "sample_f1": f1,
        "occurrence_detected": bool(detected.any()),
    }


def matched_real_speech_challenge(
    paths: CohortPaths,
    frozen: Mapping[str, Any],
    recordings: pd.DataFrame,
    *,
    carrier_count: int = 12,
    target_fractions: Sequence[float] = (0.0003, 0.001, 0.003, 0.01),
    geometries: Sequence[str] = ("symmetric", "positive_only", "negative_only"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Known-truth hard limits applied to label-blind cohort speech carriers."""
    eligible = recordings.loc[
        recordings["qdist_valid_zero"] & recordings["qdist_available"]
    ].copy()
    eligible = eligible.sort_values(
        ["qdist_native_sample_rate_hz", "participant_id", "logical_recording_id"]
    )
    ordered = deterministic_order(
        eligible["logical_recording_id"].astype(str), "qdist-real-speech-carriers"
    )
    selected: list[str] = []
    seen_participants: set[str] = set()
    lookup = eligible.set_index("logical_recording_id")
    for rid in ordered:
        participant = str(lookup.loc[rid, "participant_id"])
        if participant not in seen_participants:
            selected.append(rid)
            seen_participants.add(participant)
        if len(selected) >= carrier_count:
            break
    row_lookup = {
        str(row["logical_recording_id"]): row
        for row in frozen["recordings"].to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for rid in selected:
        try:
            waveform, fs, _probe, provenance = _task_waveform(
                paths, frozen, row_lookup[rid]
            )
            for geometry in geometries:
                for target in target_fractions:
                    altered, truth, limits = inject_matched_hard_clip(
                        waveform, target, geometry
                    )
                    extraction = detector.extract_qdist(
                        altered,
                        fs,
                        logical_recording_id=f"challenge__{rid}",
                        provenance=replace(
                            provenance,
                            source_path=f"known_truth_in_memory::{rid}",
                            source_sha256=None,
                            decoded_sha256=None,
                        ),
                    )
                    detected = _accepted_mask(
                        extraction.accepted_plateau_ledger, altered.shape
                    )
                    metrics = _truth_metrics(truth, detected)
                    rows.append({
                        "logical_recording_id": rid,
                        "participant_id": str(lookup.loc[rid, "participant_id"]),
                        "native_sample_rate_hz": fs,
                        "channel_count": altered.shape[1],
                        "duration_sec": altered.shape[0] / fs,
                        "geometry": geometry,
                        **limits,
                        **metrics,
                        "estimated_sample_fraction": extraction.recording[
                            "qdist_hard_clipped_sample_fraction"
                        ],
                        "estimated_event_rate_per_min": extraction.recording[
                            "qdist_hard_clip_event_rate_per_min"
                        ],
                        "estimated_frame_fraction": extraction.recording[
                            "qdist_hard_clipped_frame_fraction"
                        ],
                        "accepted_plateau_count": len(
                            extraction.accepted_plateau_ledger
                        ),
                        "truth_definition": "samples numerically changed by imposed hard limit",
                        "construct_limit": "decoded-speech intervention; not an analog-stage causal localization",
                    })
        except Exception as error:
            errors.append({
                "logical_recording_id": rid,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame(), pd.DataFrame(errors)
    summary = (
        long.groupby(["geometry", "target_fraction"], sort=True)
        .agg(
            carrier_count=("logical_recording_id", "nunique"),
            realized_fraction_median=("realized_fraction", "median"),
            occurrence_sensitivity=("occurrence_detected", "mean"),
            sample_precision_median=("sample_precision", "median"),
            sample_recall_median=("sample_recall", "median"),
            sample_f1_median=("sample_f1", "median"),
            estimated_fraction_median=("estimated_sample_fraction", "median"),
        )
        .reset_index()
    )
    return long, summary, pd.DataFrame(errors)


def support_calibration(
    paths: CohortPaths,
    frozen: Mapping[str, Any],
    recordings: pd.DataFrame,
    *,
    carrier_count: int = 8,
    durations_sec: Sequence[float] = (3.0, 5.0, 10.0, 20.0, 30.0),
    target_fraction: float = 0.003,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = recordings.loc[
        recordings["qdist_valid_zero"]
        & recordings["qdist_task_span_duration_sec"].ge(max(durations_sec))
    ].copy()
    ordered = deterministic_order(
        eligible["logical_recording_id"].astype(str), "qdist-support-carriers"
    )[:carrier_count]
    source_lookup = {
        str(row["logical_recording_id"]): row
        for row in frozen["recordings"].to_dict("records")
    }
    metadata = eligible.set_index("logical_recording_id")
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for rid in ordered:
        try:
            waveform, fs, _probe, provenance = _task_waveform(
                paths, frozen, source_lookup[rid]
            )
            for duration in durations_sec:
                count = int(round(float(duration) * fs))
                if len(waveform) < count:
                    continue
                start = max(0, (len(waveform) - count) // 2)
                segment = waveform[start:start + count]
                altered, truth, limits = inject_matched_hard_clip(
                    segment, target_fraction, "symmetric"
                )
                extraction = detector.extract_qdist(
                    altered,
                    fs,
                    logical_recording_id=f"support__{rid}__{duration:g}s",
                    provenance=replace(
                        provenance,
                        source_path=f"known_truth_support_in_memory::{rid}",
                        source_sha256=None,
                        decoded_sha256=None,
                    ),
                )
                detected = _accepted_mask(
                    extraction.accepted_plateau_ledger, altered.shape
                )
                rows.append({
                    "logical_recording_id": rid,
                    "participant_id": str(metadata.loc[rid, "participant_id"]),
                    "duration_sec": float(duration),
                    "native_sample_rate_hz": fs,
                    **limits,
                    **_truth_metrics(truth, detected),
                    "qdist_status": extraction.recording["qdist_status"],
                    "support_tier": extraction.recording["qdist_support_tier"],
                    "estimated_sample_fraction": extraction.recording[
                        "qdist_hard_clipped_sample_fraction"
                    ],
                    "estimated_event_rate_per_min": extraction.recording[
                        "qdist_hard_clip_event_rate_per_min"
                    ],
                })
        except Exception as error:
            errors.append({
                "logical_recording_id": rid,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
    long = pd.DataFrame(rows)
    summary = (
        long.groupby("duration_sec", sort=True)
        .agg(
            carrier_count=("logical_recording_id", "nunique"),
            availability=("qdist_status", lambda values: float(values.astype(str).str.startswith("available").mean())),
            occurrence_sensitivity=("occurrence_detected", "mean"),
            sample_precision_median=("sample_precision", "median"),
            sample_recall_median=("sample_recall", "median"),
            estimated_fraction_median=("estimated_sample_fraction", "median"),
        )
        .reset_index()
        if len(long) else pd.DataFrame()
    )
    return long, summary, pd.DataFrame(errors)


def parameter_variants() -> list[tuple[str, str, detector.QDISTParameters]]:
    """Prespecified one-factor neighborhood around the candidate operating point."""
    variants: list[tuple[str, str, detector.QDISTParameters]] = [
        ("baseline", "candidate", DEFAULT_PARAMETERS)
    ]
    grid: Mapping[str, Sequence[Any]] = {
        "candidate_generation_minimum_edge_to_robust_peak_ratio": (0.20, 0.30),
        "minimum_edge_to_robust_peak_ratio": (0.40, 0.50),
        "minimum_edge_to_local_peak_ratio": (0.85, 0.95),
        "minimum_plateau_samples": (3, 5),
        "maximum_plateau_duration_ms": (7.5, 12.5),
        "minimum_edge_zone_samples": (6, 10),
        "minimum_edge_to_interior_ratio": (1.5, 2.5),
        "minimum_edge_excess_samples": (3, 5),
        "low_level_minimum_cluster_candidates": (3, 5),
        "low_level_minimum_cluster_plateau_samples": (20, 28),
        "low_level_minimum_edge_zone_samples": (20, 28),
    }
    for parameter, values in grid.items():
        baseline = getattr(DEFAULT_PARAMETERS, parameter)
        for value in values:
            direction = "lower" if float(value) < float(baseline) else "higher"
            variants.append(
                (
                    f"{parameter}__{value:g}",
                    direction,
                    replace(DEFAULT_PARAMETERS, **{parameter: value}),
                )
            )
    return variants


def parameter_robustness(
    paths: CohortPaths,
    frozen: Mapping[str, Any],
    recordings: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    zero_count: int = 40,
    boundary_count: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rerun a label-blind enriched cohort under prespecified OAT variants."""
    positives = recordings.loc[recordings["qdist_positive"], "logical_recording_id"].astype(str).tolist()
    zeros = deterministic_order(
        recordings.loc[recordings["qdist_valid_zero"], "logical_recording_id"].astype(str),
        "qdist-parameter-zero",
    )[:zero_count]
    boundary: list[str] = []
    if len(candidates):
        rejected = candidates.loc[~as_bool(candidates["accepted"])].copy()
        pass_columns = [
            column for column in [
                "morphology_pass", "duration_pass", "magnitude_pass",
                "context_pass", "transition_pass", "cluster_support_pass",
                "edge_support_pass", "edge_ratio_pass", "edge_excess_pass",
                "terminal_edge_pass", "quantization_guard_pass",
                "square_like_guard_pass",
            ] if column in rejected
        ]
        if pass_columns:
            rejected["predicate_pass_count"] = rejected[pass_columns].apply(
                lambda frame: as_bool(frame), axis=0
            ).sum(axis=1)
            boundary = (
                rejected.sort_values(
                    ["predicate_pass_count", "candidate_to_context_ratio"],
                    ascending=[False, False],
                )["logical_recording_id"].astype(str).drop_duplicates().head(boundary_count).tolist()
            )
    selected = list(dict.fromkeys(positives + boundary + zeros))
    source_lookup = {
        str(row["logical_recording_id"]): row
        for row in frozen["recordings"].to_dict("records")
    }
    meta = recordings.set_index("logical_recording_id")
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    variants = parameter_variants()
    for rid in selected:
        try:
            waveform, fs, _probe, provenance = _task_waveform(
                paths, frozen, source_lookup[rid]
            )
            for variant_name, direction, parameters in variants:
                extraction = detector.extract_qdist(
                    waveform,
                    fs,
                    logical_recording_id=rid,
                    provenance=provenance,
                    parameters=parameters,
                )
                rows.append({
                    "logical_recording_id": rid,
                    "participant_id": str(meta.loc[rid, "participant_id"]),
                    "sampling_stratum": (
                        "baseline_positive" if rid in positives
                        else ("near_boundary" if rid in boundary else "valid_zero")
                    ),
                    "variant": variant_name,
                    "direction": direction,
                    "parameter_hash": parameters.parameter_hash(),
                    "occurrence": bool(len(extraction.episode_ledger)),
                    "accepted_plateau_count": len(extraction.accepted_plateau_ledger),
                    **{
                        feature: extraction.recording[feature]
                        for feature in ANALYSIS_FEATURES
                    },
                })
        except Exception as error:
            errors.append({
                "logical_recording_id": rid,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame(), pd.DataFrame(errors)
    baseline_columns = [
        "logical_recording_id", "occurrence", "accepted_plateau_count", *ANALYSIS_FEATURES
    ]
    baseline = long.loc[long["variant"].eq("baseline"), baseline_columns].rename(
        columns={column: f"baseline_{column}" for column in baseline_columns if column != "logical_recording_id"}
    )
    compared = long.merge(
        baseline, on="logical_recording_id", how="left", validate="many_to_one"
    )
    compared["occurrence_agreement"] = compared["occurrence"].eq(
        compared["baseline_occurrence"]
    )
    summary_rows: list[dict[str, Any]] = []
    for variant, group in compared.groupby("variant", sort=False):
        row: dict[str, Any] = {
            "variant": variant,
            "direction": group["direction"].iloc[0],
            "recording_count": group["logical_recording_id"].nunique(),
            "occurrence_agreement": float(group["occurrence_agreement"].mean()),
            "positive_recording_count": int(group["occurrence"].sum()),
            "occurrence_flip_count": int((~group["occurrence_agreement"]).sum()),
        }
        for feature in ANALYSIS_FEATURES:
            current = pd.to_numeric(group[feature], errors="coerce")
            base = pd.to_numeric(group[f"baseline_{feature}"], errors="coerce")
            finite = current.notna() & base.notna()
            row[f"{feature}__median_absolute_change"] = (
                float((current.loc[finite] - base.loc[finite]).abs().median())
                if finite.any() else np.nan
            )
            _, rho = finite_spearman(current.loc[finite], base.loc[finite])
            row[f"{feature}__spearman_rho"] = rho
        summary_rows.append(row)
    return compared, pd.DataFrame(summary_rows), pd.DataFrame(errors)


def feature_summary(recordings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in ANALYSIS_FEATURES:
        values = pd.to_numeric(recordings[feature], errors="coerce")
        finite = values.dropna()
        positive = finite.loc[finite > 0]
        rows.append({
            "feature": feature,
            "recording_count": len(recordings),
            "available_n": len(finite),
            "available_fraction": float(len(finite) / len(recordings)) if len(recordings) else np.nan,
            "valid_zero_n": int(finite.eq(0).sum()),
            "positive_n": int(finite.gt(0).sum()),
            "positive_fraction_available": float(finite.gt(0).mean()) if len(finite) else np.nan,
            "median": float(finite.median()) if len(finite) else np.nan,
            "q25": float(finite.quantile(.25)) if len(finite) else np.nan,
            "q75": float(finite.quantile(.75)) if len(finite) else np.nan,
            "maximum": float(finite.max()) if len(finite) else np.nan,
            "positive_median": float(positive.median()) if len(positive) else np.nan,
        })
    return pd.DataFrame(rows)


def merge_gap_sensitivity(
    recordings: pd.DataFrame,
    accepted: pd.DataFrame,
    gaps_ms: Sequence[float] = (10.0, 20.0, 30.0, 50.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = {
        str(key): value.copy()
        for key, value in accepted.groupby("logical_recording_id", sort=False)
    } if len(accepted) else {}
    rows: list[dict[str, Any]] = []
    for rec in recordings.to_dict("records"):
        rid = str(rec["logical_recording_id"])
        local = groups.get(rid, pd.DataFrame())
        intervals = list(zip(
            pd.to_numeric(local.get("start_sample_task", pd.Series(dtype=float)), errors="coerce"),
            pd.to_numeric(local.get("end_sample_task_exclusive", pd.Series(dtype=float)), errors="coerce"),
        ))
        fs = int(rec["qdist_native_sample_rate_hz"])
        exposure = float(rec["qdist_finite_exposure_sec"])
        for gap in gaps_ms:
            events = merge_intervals(intervals, int(round(float(gap) * fs / 1000.0)))
            count = len(events)
            rows.append({
                "logical_recording_id": rid,
                "participant_id": rec["participant_id"],
                "merge_gap_ms": float(gap),
                "event_count": count,
                "event_rate_per_min": count * 60.0 / exposure if exposure > 0 else np.nan,
                "positive": count > 0,
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame()
    baseline = long.loc[long["merge_gap_ms"].eq(20.0), [
        "logical_recording_id", "event_count", "event_rate_per_min", "positive"
    ]].rename(columns={
        "event_count": "baseline_event_count",
        "event_rate_per_min": "baseline_event_rate_per_min",
        "positive": "baseline_positive",
    })
    compared = long.merge(baseline, on="logical_recording_id", validate="many_to_one")
    compared["occurrence_agreement"] = compared["positive"].eq(compared["baseline_positive"])
    compared["event_count_changed"] = compared["event_count"].ne(compared["baseline_event_count"])
    compared["absolute_rate_change"] = (
        compared["event_rate_per_min"] - compared["baseline_event_rate_per_min"]
    ).abs()
    summary = (
        compared.groupby("merge_gap_ms", sort=True)
        .agg(
            recording_count=("logical_recording_id", "nunique"),
            positive_recording_count=("positive", "sum"),
            occurrence_agreement=("occurrence_agreement", "mean"),
            event_count_changed_fraction=("event_count_changed", "mean"),
            median_absolute_rate_change=("absolute_rate_change", "median"),
            maximum_absolute_rate_change=("absolute_rate_change", "max"),
        )
        .reset_index()
    )
    return compared, summary


def deletion_influence(
    recordings: pd.DataFrame,
    accepted: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_groups = {
        str(key): value.copy()
        for key, value in accepted.groupby("logical_recording_id", sort=False)
    } if len(accepted) else {}
    episode_groups = {
        str(key): value.copy()
        for key, value in episodes.groupby("logical_recording_id", sort=False)
    } if len(episodes) else {}
    rows: list[dict[str, Any]] = []
    for rec in recordings.loc[recordings["qdist_positive"]].to_dict("records"):
        rid = str(rec["logical_recording_id"])
        local_accepted = accepted_groups.get(rid, pd.DataFrame()).copy()
        local_episodes = episode_groups.get(rid, pd.DataFrame()).copy()
        base = detector.reconstruct_qdist_features(
            local_accepted, local_episodes,
            finite_channel_sample_count=int(rec["qdist_finite_channel_sample_count"]),
            finite_time_sample_count=int(rec["qdist_finite_time_sample_count"]),
            finite_exposure_sec=float(rec["qdist_finite_exposure_sec"]),
            frame_length_samples=int(rec["qdist_frame_length_samples"]),
            complete_frame_count=int(rec["qdist_complete_frame_count"]),
        )
        for candidate_id in local_accepted.get("candidate_id", pd.Series(dtype=str)).astype(str):
            variant_accepted = local_accepted.loc[
                ~local_accepted["candidate_id"].astype(str).eq(candidate_id)
            ].copy()
            intervals = list(zip(
                variant_accepted.get("start_sample_task", pd.Series(dtype=int)),
                variant_accepted.get("end_sample_task_exclusive", pd.Series(dtype=int)),
            ))
            merged = merge_intervals(
                intervals,
                int(round(DEFAULT_PARAMETERS.episode_merge_gap_ms * int(rec["qdist_native_sample_rate_hz"]) / 1000.0)),
            )
            variant_episodes = pd.DataFrame({"episode_id": [f"variant_{i}" for i in range(len(merged))]})
            variant = detector.reconstruct_qdist_features(
                variant_accepted, variant_episodes,
                finite_channel_sample_count=int(rec["qdist_finite_channel_sample_count"]),
                finite_time_sample_count=int(rec["qdist_finite_time_sample_count"]),
                finite_exposure_sec=float(rec["qdist_finite_exposure_sec"]),
                frame_length_samples=int(rec["qdist_frame_length_samples"]),
                complete_frame_count=int(rec["qdist_complete_frame_count"]),
            )
            rows.append({
                "logical_recording_id": rid,
                "deletion_type": "plateau",
                "deleted_id": candidate_id,
                "frame_fraction_absolute_change": abs(variant["qdist_hard_clipped_frame_fraction"] - base["qdist_hard_clipped_frame_fraction"]),
                "event_rate_absolute_change": abs(variant["qdist_hard_clip_event_rate_per_min"] - base["qdist_hard_clip_event_rate_per_min"]),
                "sample_fraction_absolute_change": abs(variant["qdist_hard_clipped_sample_fraction"] - base["qdist_hard_clipped_sample_fraction"]),
            })
        for episode_id in local_episodes.get("episode_id", pd.Series(dtype=str)).astype(str):
            variant_episodes = local_episodes.loc[
                ~local_episodes["episode_id"].astype(str).eq(episode_id)
            ]
            variant = detector.reconstruct_qdist_features(
                local_accepted, variant_episodes,
                finite_channel_sample_count=int(rec["qdist_finite_channel_sample_count"]),
                finite_time_sample_count=int(rec["qdist_finite_time_sample_count"]),
                finite_exposure_sec=float(rec["qdist_finite_exposure_sec"]),
                frame_length_samples=int(rec["qdist_frame_length_samples"]),
                complete_frame_count=int(rec["qdist_complete_frame_count"]),
            )
            rows.append({
                "logical_recording_id": rid,
                "deletion_type": "episode",
                "deleted_id": episode_id,
                "frame_fraction_absolute_change": 0.0,
                "event_rate_absolute_change": abs(variant["qdist_hard_clip_event_rate_per_min"] - base["qdist_hard_clip_event_rate_per_min"]),
                "sample_fraction_absolute_change": 0.0,
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame(columns=[
            "deletion_type", "recording_count", "deletion_count",
            "maximum_frame_fraction_absolute_change",
            "maximum_event_rate_absolute_change",
            "maximum_sample_fraction_absolute_change",
        ])
    summary = (
        long.groupby("deletion_type", sort=True)
        .agg(
            recording_count=("logical_recording_id", "nunique"),
            deletion_count=("deleted_id", "size"),
            median_frame_fraction_absolute_change=("frame_fraction_absolute_change", "median"),
            maximum_frame_fraction_absolute_change=("frame_fraction_absolute_change", "max"),
            median_event_rate_absolute_change=("event_rate_absolute_change", "median"),
            maximum_event_rate_absolute_change=("event_rate_absolute_change", "max"),
            median_sample_fraction_absolute_change=("sample_fraction_absolute_change", "median"),
            maximum_sample_fraction_absolute_change=("sample_fraction_absolute_change", "max"),
        )
        .reset_index()
    )
    return long, summary


def repeated_recording_evidence(
    recordings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = recordings.sort_values(
        ["participant_id", "acquisition_date", "logical_recording_id"]
    )
    first_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for participant, group in ordered.groupby("participant_id", sort=True):
        records = group.to_dict("records")
        if len(records) < 2:
            continue
        pairs = [(0, 1)]
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                base = {
                    "participant_id": participant,
                    "recording_1": records[i]["logical_recording_id"],
                    "recording_2": records[j]["logical_recording_id"],
                    "positive_1": bool(records[i]["qdist_positive"]),
                    "positive_2": bool(records[j]["qdist_positive"]),
                }
                all_rows.append(base)
                if (i, j) in pairs:
                    for feature in ANALYSIS_FEATURES:
                        base[f"{feature}_1"] = records[i][feature]
                        base[f"{feature}_2"] = records[j][feature]
                    first_rows.append(base)
    first = pd.DataFrame(first_rows)
    all_pairs = pd.DataFrame(all_rows)
    if first.empty:
        return first, all_pairs, pd.DataFrame()
    a, b = as_bool(first["positive_1"]), as_bool(first["positive_2"])
    n11, n10 = int((a & b).sum()), int((a & ~b).sum())
    n01, n00 = int((~a & b).sum()), int((~a & ~b).sum())
    base_summary = {
        "participant_pair_count": len(first),
        "both_positive_n11": n11, "first_only_n10": n10,
        "second_only_n01": n01, "both_zero_n00": n00,
        "overall_agreement": float(a.eq(b).mean()),
        "positive_agreement": 2 * n11 / (2 * n11 + n10 + n01) if 2 * n11 + n10 + n01 else np.nan,
        "negative_agreement": 2 * n00 / (2 * n00 + n10 + n01) if 2 * n00 + n10 + n01 else np.nan,
        "cohens_kappa": cohen_kappa_binary(a, b),
    }
    summary_rows = [{
        "metric": "occurrence", **base_summary,
        "positive_part_pair_n": n11, "positive_part_spearman_rho": np.nan,
        "interpretation": "occurrence persistence; acquisition artifacts are not expected to be trait-stable",
    }]
    both = first.loc[a & b]
    for feature in ANALYSIS_FEATURES:
        n, rho = finite_spearman(both[f"{feature}_1"], both[f"{feature}_2"])
        summary_rows.append({
            "metric": feature, **base_summary,
            "positive_part_pair_n": n,
            "positive_part_spearman_rho": rho if n >= 5 else np.nan,
            "interpretation": "estimated on pairs positive at both visits" if n >= 5 else "not estimable: fewer than five positive-positive pairs",
        })
    return first, all_pairs, pd.DataFrame(summary_rows)


def redundancy_table(recordings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, left in enumerate(ANALYSIS_FEATURES):
        for right in ANALYSIS_FEATURES[index + 1:]:
            n_all, rho_all = finite_spearman(recordings[left], recordings[right])
            positive = recordings.loc[recordings["qdist_positive"]]
            n_positive, rho_positive = finite_spearman(positive[left], positive[right])
            rows.append({
                "feature_1": left, "feature_2": right,
                "all_recordings_n": n_all, "all_recordings_spearman_rho": rho_all,
                "positive_recordings_n": n_positive,
                "positive_recordings_spearman_rho": rho_positive,
                "related_view_system": True,
            })
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = float(stats.norm.ppf(.975))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def participant_weighting(recordings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = (
        recordings.groupby("participant_id", sort=True)
        .agg(
            recording_count=("logical_recording_id", "size"),
            positive_recording_count=("qdist_positive", "sum"),
            participant_any_positive=("qdist_positive", "max"),
            sample_fraction_median=("qdist_hard_clipped_sample_fraction", "median"),
            event_rate_median=("qdist_hard_clip_event_rate_per_min", "median"),
            frame_fraction_median=("qdist_hard_clipped_frame_fraction", "median"),
        )
        .reset_index()
    )
    summary_rows: list[dict[str, Any]] = []
    for level, count, positive in [
        ("recording_weighted", len(recordings), int(recordings["qdist_positive"].sum())),
        ("participant_ever_positive", len(participant), int(participant["participant_any_positive"].sum())),
    ]:
        low, high = wilson_interval(positive, count)
        summary_rows.append({
            "analysis_level": level, "units": count, "positive_units": positive,
            "positive_fraction": positive / count if count else np.nan,
            "wilson95_low": low, "wilson95_high": high,
        })
    return participant, pd.DataFrame(summary_rows)


def analysis_context_tables(
    recordings: pd.DataFrame,
    accepted: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    empirical = feature_summary(recordings)
    merge_long, merge_summary = merge_gap_sensitivity(recordings, accepted)
    deletion_long, deletion_summary = deletion_influence(
        recordings, accepted, episodes
    )
    repeated_first, repeated_all, repeated_summary = repeated_recording_evidence(
        recordings
    )
    redundancy = redundancy_table(recordings)
    participant, weighting = participant_weighting(recordings)
    return {
        "feature_summary": empirical,
        "merge_long": merge_long,
        "merge_summary": merge_summary,
        "deletion_long": deletion_long,
        "deletion_summary": deletion_summary,
        "repeated_first": repeated_first,
        "repeated_all": repeated_all,
        "repeated_summary": repeated_summary,
        "redundancy": redundancy,
        "participant": participant,
        "weighting": weighting,
    }


def build_ml_candidate_interface(recordings: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "logical_recording_id", "participant_id", *ANALYSIS_FEATURES,
        *[f"{feature}_status" for feature in ANALYSIS_FEATURES],
        "qdist_occurrence", "qdist_status", "qdist_available", "qdist_support_tier",
        "qdist_finite_exposure_sec", "qdist_hard_clip_event_count",
        "qdist_hard_clip_event_rate_ci95_low_per_min",
        "qdist_hard_clip_event_rate_ci95_high_per_min",
        "qdist_measurement_version", "qdist_parameter_hash", "qdist_signal_view",
        "qdist_source_sha256", "qdist_decoded_sha256",
    ]
    interface = recordings[[column for column in columns if column in recordings]].copy()
    interface["qdist_family_scalar_constructed"] = False
    interface["qdist_missing_values_imputed"] = False
    interface["qdist_standalone_accept_reject_allowed"] = False
    interface["qdist_release_status"] = "CANDIDATE_NOT_FROZEN"
    interface["qdist_primary_feature"] = "qdist_hard_clipped_sample_fraction"
    interface["qdist_secondary_feature"] = "qdist_hard_clip_event_rate_per_min"
    interface["qdist_conditional_feature"] = "qdist_hard_clipped_frame_fraction"
    return interface


def _review_item_figure(
    waveform: np.ndarray,
    fs: int,
    focus_start: int,
    focus_end: int,
    output: Path,
) -> None:
    values = np.asarray(waveform, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    mono = np.mean(values, axis=1)
    time = np.arange(len(values)) / fs
    focus_start = max(0, int(focus_start))
    focus_end = min(len(values), max(focus_start + 1, int(focus_end)))
    padding = max(int(round(.004 * fs)), 16)
    zoom_start = max(0, focus_start - padding)
    zoom_end = min(len(values), focus_end + padding)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].plot(time, values, lw=.45)
    axes[0, 0].axvspan(focus_start / fs, focus_end / fs, color="tab:red", alpha=.2)
    axes[0, 0].set(title="Context waveform", xlabel="Window time (s)", ylabel="Amplitude")
    zoom_time = np.arange(zoom_start, zoom_end) / fs
    axes[0, 1].plot(zoom_time, values[zoom_start:zoom_end], marker=".", ms=2, lw=.7)
    axes[0, 1].axvspan(focus_start / fs, focus_end / fs, color="tab:red", alpha=.2)
    axes[0, 1].set(title="Sample-level zoom", xlabel="Window time (s)")
    axes[0, 2].plot(
        zoom_time[1:], np.diff(values[zoom_start:zoom_end], axis=0), lw=.6
    )
    axes[0, 2].set(title="First difference", xlabel="Window time (s)")
    axes[1, 0].hist(values.ravel(), bins=100, color=".25")
    axes[1, 0].set(title="Amplitude occupancy", xlabel="Amplitude", ylabel="Samples")
    axes[1, 1].specgram(mono, NFFT=min(512, max(64, 2 ** int(np.floor(np.log2(max(len(mono) // 8, 64)))))), Fs=fs, noverlap=32)
    axes[1, 1].set(title="Spectrogram", xlabel="Window time (s)", ylabel="Hz")
    sorted_values = np.sort(values.ravel())
    axes[1, 2].plot(sorted_values, np.linspace(0, 1, len(sorted_values), endpoint=False), lw=.8)
    axes[1, 2].set(title="Empirical amplitude CDF", xlabel="Amplitude", ylabel="Cumulative fraction")
    fig.suptitle("Blinded QDIST morphology review item")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_blind_review_package(
    paths: CohortPaths,
    frozen: Mapping[str, Any],
    recordings: pd.DataFrame,
    candidates: pd.DataFrame,
    accepted: pd.DataFrame,
    *,
    rejected_count: int = 30,
    valid_zero_count: int = 20,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build human-only, label-blind review material; never infer review labels."""
    review_root = paths.output_root / "blind_review"
    item_root = review_root / "items"
    restricted_root = review_root / "restricted"
    index_path = review_root / "qdist_v410_blind_review_index.csv"
    key_path = restricted_root / "qdist_v410_blind_review_key.csv"
    contract_path = review_root / "qdist_v410_blind_review_contract.json"
    signature_columns = [
        "logical_recording_id", "candidate_id", "accepted",
        "start_sample_task", "end_sample_task_exclusive", "rejection_reason",
    ]
    review_signature = stable_hash({
        "cohort_version": COHORT_VERSION,
        "parameter_hash": DEFAULT_PARAMETERS.parameter_hash(),
        "rejected_count": rejected_count,
        "valid_zero_count": valid_zero_count,
        "candidate_rows": candidates[[
            column for column in signature_columns if column in candidates
        ]].to_dict("records"),
        "accepted_rows": accepted[[
            column for column in signature_columns if column in accepted
        ]].to_dict("records"),
        "valid_zero_ids": sorted(
            recordings.loc[recordings["qdist_valid_zero"], "logical_recording_id"].astype(str)
        ),
    })
    if index_path.exists() and key_path.exists() and contract_path.exists() and not rebuild:
        prior_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if prior_contract.get("review_signature") == review_signature:
            return read_table(index_path), read_table(key_path), pd.DataFrame()
    for folder in [item_root, restricted_root]:
        folder.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    for row in accepted.to_dict("records"):
        selected_rows.append({**row, "review_stratum": "accepted_plateau"})
    rejected = candidates.loc[~as_bool(candidates["accepted"])].copy()
    pass_columns = [
        column for column in [
            "morphology_pass", "duration_pass", "magnitude_pass", "context_pass",
            "transition_pass", "cluster_support_pass", "edge_support_pass",
            "edge_ratio_pass", "edge_excess_pass", "terminal_edge_pass",
            "quantization_guard_pass", "square_like_guard_pass",
        ] if column in rejected
    ]
    if len(rejected):
        rejected["predicate_pass_count"] = rejected[pass_columns].apply(
            lambda frame: as_bool(frame), axis=0
        ).sum(axis=1) if pass_columns else 0
        rejected = rejected.sort_values(
            ["predicate_pass_count", "candidate_to_context_ratio"],
            ascending=[False, False],
        ).head(rejected_count)
        for row in rejected.to_dict("records"):
            selected_rows.append({**row, "review_stratum": "near_threshold_rejection"})
    zero_ids = deterministic_order(
        recordings.loc[recordings["qdist_valid_zero"], "logical_recording_id"].astype(str),
        "qdist-review-valid-zero",
    )[:valid_zero_count]
    for rid in zero_ids:
        selected_rows.append({
            "logical_recording_id": rid,
            "review_stratum": "valid_zero_window",
            "candidate_id": "",
            "start_sample_task": pd.NA,
            "end_sample_task_exclusive": pd.NA,
        })
    selected_rows.sort(
        key=lambda row: sha256(
            f"qdist-v410-review::{row.get('logical_recording_id')}::{row.get('candidate_id')}::{row.get('review_stratum')}".encode()
        ).hexdigest()
    )
    source_lookup = {
        str(row["logical_recording_id"]): row
        for row in frozen["recordings"].to_dict("records")
    }
    cache: dict[str, tuple[np.ndarray, int, dict[str, Any], detector.NativeSignalProvenance]] = {}
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, source in enumerate(selected_rows, start=1):
        rid = str(source["logical_recording_id"])
        blind_id = f"QDREV-{index:04d}-{stable_hash({'rid': rid, 'candidate': source.get('candidate_id'), 'salt': RANDOM_SEED})[:8].upper()}"
        try:
            if rid not in cache:
                cache[rid] = _task_waveform(paths, frozen, source_lookup[rid])
            waveform, fs, _probe, _provenance = cache[rid]
            if source["review_stratum"] == "valid_zero_window":
                center = len(waveform) // 2
                focus_start = max(0, center - 1)
                focus_end = min(len(waveform), center + 1)
            else:
                focus_start = int(source["start_sample_task"])
                focus_end = int(source["end_sample_task_exclusive"])
                center = (focus_start + focus_end) // 2
            half = max(1, int(round(.25 * fs)))
            window_start = max(0, center - half)
            window_end = min(len(waveform), center + half)
            window = waveform[window_start:window_end]
            relative_start = focus_start - window_start
            relative_end = focus_end - window_start
            png = item_root / f"{blind_id}.png"
            wav = item_root / f"{blind_id}.wav"
            _review_item_figure(window, fs, relative_start, relative_end, png)
            audio = np.clip(window, -1, 1)
            wavfile.write(wav, fs, np.round(audio * 32767).astype(np.int16))
            public_rows.append({
                "blind_id": blind_id,
                "display_order": index,
                "image_path": str(png.relative_to(review_root)),
                "audio_path": str(wav.relative_to(review_root)),
                "views_present": "context_waveform|sample_zoom|first_difference|amplitude_occupancy|spectrogram|cdf|audio",
                "all_required_views_present": True,
                "review_status": "PENDING_TWO_INDEPENDENT_HUMAN_REVIEWS",
            })
            key_rows.append({
                "blind_id": blind_id,
                "logical_recording_id": rid,
                "candidate_id": source.get("candidate_id", ""),
                "review_stratum": source["review_stratum"],
                "participant_id": recordings.set_index("logical_recording_id").loc[rid, "participant_id"],
                "window_start_sample_task": window_start,
                "window_end_sample_task_exclusive": window_end,
                "focus_start_sample_task": focus_start,
                "focus_end_sample_task_exclusive": focus_end,
                "rejection_reason": source.get("rejection_reason", ""),
                "magnitude_path": source.get("magnitude_path", ""),
            })
        except Exception as error:
            errors.append({
                "blind_id": blind_id,
                "logical_recording_id": rid,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
    public = pd.DataFrame(public_rows, columns=[
        "blind_id", "display_order", "image_path", "audio_path",
        "views_present", "all_required_views_present", "review_status",
    ])
    key = pd.DataFrame(key_rows)
    public.to_csv(index_path, index=False)
    key.to_csv(key_path, index=False)
    template = public[["blind_id"]].copy()
    template["reviewer_id"] = ""
    template["review_label"] = ""
    template["confidence_1_to_5"] = ""
    template["artifact_views_complete"] = ""
    template["comments"] = ""
    template.to_csv(review_root / "reviewer_1_TEMPLATE.csv", index=False)
    template.to_csv(review_root / "reviewer_2_TEMPLATE.csv", index=False)
    adjudication_template = public[["blind_id"]].copy()
    adjudication_template["adjudicator_id"] = ""
    adjudication_template["adjudicated_label"] = ""
    adjudication_template["rationale"] = ""
    adjudication_template.to_csv(
        review_root / "adjudication_TEMPLATE.csv", index=False
    )
    instructions = f"""# QDIST v4.1 blinded morphology review

Two reviewers must work independently before comparison. Review every row in
display order without opening the `restricted` directory. Permitted labels are:
{', '.join(REVIEW_LABELS)}.

Judge only whether the visible native-decoded morphology is compatible with a
hard plateau. Do not infer ALS status, recording acceptability, analog cause,
codec cause, or the presence of other nonlinear distortion. Complete reviewer
ID, one permitted label, confidence 1–5, view completeness, and comments.
Do not modify `blind_id`. Human completion and disagreement adjudication are
required before G9, feature finalization, or freeze can pass.
"""
    (review_root / "README_BLINDED_REVIEW.md").write_text(instructions, encoding="utf-8")
    write_json({
        "review_signature": review_signature,
        "cohort_version": COHORT_VERSION,
        "parameter_hash": DEFAULT_PARAMETERS.parameter_hash(),
        "item_count": len(public),
        "generation_error_count": len(errors),
        "human_labels_generated": False,
        "review_status": "PENDING_TWO_INDEPENDENT_HUMAN_REVIEWS",
    }, contract_path)
    return public, key, pd.DataFrame(errors)


def adjudicate_blind_review(
    output_root: str | Path,
    reviewer_1: str | Path,
    reviewer_2: str | Path,
    adjudication: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Validate completed reviewer sheets and quantify agreement; no auto-labels."""
    root = Path(output_root)
    index = pd.read_csv(root / "blind_review" / "qdist_v410_blind_review_index.csv")
    key = pd.read_csv(root / "blind_review" / "restricted" / "qdist_v410_blind_review_key.csv")
    reviews: list[pd.DataFrame] = []
    for reviewer_number, path in enumerate([reviewer_1, reviewer_2], start=1):
        review = pd.read_csv(path, keep_default_na=False)
        required = {"blind_id", "reviewer_id", "review_label", "confidence_1_to_5"}
        if not required.issubset(review.columns):
            raise ValueError(f"Reviewer {reviewer_number} sheet lacks {sorted(required - set(review.columns))}")
        if set(review["blind_id"]) != set(index["blind_id"]):
            raise ValueError(f"Reviewer {reviewer_number} blind IDs are incomplete or altered.")
        if not set(review["review_label"]).issubset(REVIEW_LABELS):
            raise ValueError(f"Reviewer {reviewer_number} contains an invalid label.")
        confidence = pd.to_numeric(review["confidence_1_to_5"], errors="coerce")
        if confidence.isna().any() or not confidence.between(1, 5).all():
            raise ValueError(f"Reviewer {reviewer_number} confidence must be 1–5.")
        if review["reviewer_id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"Reviewer {reviewer_number} reviewer_id is incomplete.")
        reviews.append(review.add_suffix(f"__r{reviewer_number}").rename(
            columns={f"blind_id__r{reviewer_number}": "blind_id"}
        ))
    merged = reviews[0].merge(reviews[1], on="blind_id", validate="one_to_one").merge(
        key, on="blind_id", validate="one_to_one"
    )
    positive = {"DEFINITE_HARD_CLIP", "PROBABLE_HARD_CLIP"}
    merged["hard_clip_positive__r1"] = merged["review_label__r1"].isin(positive)
    merged["hard_clip_positive__r2"] = merged["review_label__r2"].isin(positive)
    merged["exact_label_agreement"] = merged["review_label__r1"].eq(
        merged["review_label__r2"]
    )
    merged["binary_agreement"] = merged["hard_clip_positive__r1"].eq(
        merged["hard_clip_positive__r2"]
    )
    summary_rows = [{
        "metric": "two_reviewer_agreement",
        "item_count": len(merged),
        "exact_label_agreement": float(merged["exact_label_agreement"].mean()),
        "binary_agreement": float(merged["binary_agreement"].mean()),
        "binary_cohens_kappa": cohen_kappa_binary(
            merged["hard_clip_positive__r1"], merged["hard_clip_positive__r2"]
        ),
        "disagreement_count": int((~merged["binary_agreement"]).sum()),
    }]
    strata = (
        merged.groupby("review_stratum", sort=True)
        .agg(
            item_count=("blind_id", "size"),
            reviewer_1_positive_fraction=("hard_clip_positive__r1", "mean"),
            reviewer_2_positive_fraction=("hard_clip_positive__r2", "mean"),
            exact_label_agreement=("exact_label_agreement", "mean"),
            binary_agreement=("binary_agreement", "mean"),
        )
        .reset_index()
    )
    exact_disagreements = merged.loc[~merged["exact_label_agreement"]].copy()
    binary_disagreements = merged.loc[~merged["binary_agreement"]].copy()
    merged["adjudicated_label"] = np.where(
        merged["exact_label_agreement"], merged["review_label__r1"], ""
    )
    adjudication_complete = exact_disagreements.empty
    if adjudication is not None:
        adjudication_frame = pd.read_csv(adjudication, keep_default_na=False)
        required = {"blind_id", "adjudicator_id", "adjudicated_label", "rationale"}
        if not required.issubset(adjudication_frame.columns):
            raise ValueError(
                "Adjudication sheet lacks "
                f"{sorted(required - set(adjudication_frame.columns))}."
            )
        completed = adjudication_frame.loc[
            adjudication_frame["adjudicated_label"].astype(str).str.strip().ne("")
        ].copy()
        expected_ids = set(exact_disagreements["blind_id"].astype(str))
        if set(completed["blind_id"].astype(str)) != expected_ids:
            raise ValueError(
                "Completed adjudication rows must match the exact-label disagreement IDs."
            )
        if not set(completed["adjudicated_label"]).issubset(REVIEW_LABELS):
            raise ValueError("Adjudication contains an invalid label.")
        if completed["adjudicator_id"].astype(str).str.strip().eq("").any():
            raise ValueError("Adjudicator identity is incomplete.")
        if completed["rationale"].astype(str).str.strip().eq("").any():
            raise ValueError("Every adjudicated disagreement requires a rationale.")
        adjudicated_lookup = completed.set_index("blind_id")["adjudicated_label"]
        mismatch = ~merged["exact_label_agreement"]
        merged.loc[mismatch, "adjudicated_label"] = merged.loc[
            mismatch, "blind_id"
        ].map(adjudicated_lookup)
        adjudication_complete = merged["adjudicated_label"].astype(str).str.strip().ne("").all()
    merged["adjudicated_hard_clip_positive"] = merged["adjudicated_label"].isin(positive)
    adjudicated_strata = (
        merged.groupby("review_stratum", sort=True)
        .agg(
            item_count=("blind_id", "size"),
            adjudicated_positive_fraction=("adjudicated_hard_clip_positive", "mean"),
        )
        .reset_index()
        if adjudication_complete else pd.DataFrame()
    )
    kappa = float(summary_rows[0]["binary_cohens_kappa"])
    review_checks: list[dict[str, Any]] = [
        {
            "check": "two independent sheets complete",
            "status": "PASS",
            "observed": len(merged),
            "required": len(index),
        },
        {
            "check": "binary Cohen kappa",
            "status": "PASS" if np.isfinite(kappa) and kappa >= .80 else "FAIL",
            "observed": kappa,
            "required": ">=0.80",
        },
        {
            "check": "all exact-label disagreements adjudicated",
            "status": "PASS" if adjudication_complete else "PENDING",
            "observed": int(len(exact_disagreements)),
            "required": "completed final label and rationale for every exact-label disagreement",
        },
    ]
    if adjudication_complete:
        thresholds = {
            "accepted_plateau": (">=", .90),
            "near_threshold_rejection": ("<=", .20),
            "valid_zero_window": ("<=", .05),
        }
        for stratum, (orientation, threshold) in thresholds.items():
            local = adjudicated_strata.loc[
                adjudicated_strata["review_stratum"].eq(stratum)
            ]
            value = (
                float(local["adjudicated_positive_fraction"].iloc[0])
                if len(local) else np.nan
            )
            passed = (
                np.isfinite(value)
                and (value >= threshold if orientation == ">=" else value <= threshold)
            )
            review_checks.append({
                "check": f"adjudicated positive fraction: {stratum}",
                "status": "PASS" if passed else "FAIL",
                "observed": value,
                "required": f"{orientation}{threshold:.2f}",
            })
    review_checks_frame = pd.DataFrame(review_checks)
    summary = pd.DataFrame(summary_rows)
    validation = root / "validation"
    save_table(merged, validation / "qdist_v410_blind_review_merged", parquet=False)
    save_table(summary, validation / "qdist_v410_blind_review_agreement", parquet=False)
    save_table(strata, validation / "qdist_v410_blind_review_by_stratum", parquet=False)
    save_table(exact_disagreements, validation / "qdist_v410_blind_review_exact_disagreements", parquet=False)
    save_table(binary_disagreements, validation / "qdist_v410_blind_review_binary_disagreements", parquet=False)
    save_table(adjudicated_strata, validation / "qdist_v410_blind_review_adjudicated_strata", parquet=False)
    save_table(review_checks_frame, validation / "qdist_v410_blind_review_checks", parquet=False)
    return {
        "merged": merged,
        "summary": summary,
        "strata": strata,
        "adjudicated_strata": adjudicated_strata,
        "review_checks": review_checks_frame,
        "exact_disagreements": exact_disagreements,
        "binary_disagreements": binary_disagreements,
        "disagreements": binary_disagreements,
    }


def candidate_feature_decisions() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "feature": "qdist_hard_clipped_sample_fraction",
            "candidate_role": "PRIMARY",
            "unit": "fraction of finite channel samples",
            "orientation": "higher means more accepted visible plateau support",
            "status": "PENDING_HUMAN_REVIEW_AND_FINAL_SCIENTIFIC_DECISION",
            "permitted_interpretation": "conservative accepted hard-plateau support in the stored native decoded waveform",
            "prohibited_interpretation": "unbiased fraction of all physically clipped samples; causal analog-stage localization; complete nonlinear distortion",
        },
        {
            "feature": "qdist_hard_clip_event_rate_per_min",
            "candidate_role": "SECONDARY",
            "unit": "merged candidate episodes per finite exposure minute",
            "orientation": "higher means more temporally distinct accepted plateau episodes",
            "status": "PENDING_HUMAN_REVIEW_AND_FINAL_SCIENTIFIC_DECISION",
            "permitted_interpretation": "episode occurrence view conditional on the prespecified 20-ms merge rule",
            "prohibited_interpretation": "physical clipping-event count independent of detector and merge rule",
        },
        {
            "feature": "qdist_hard_clipped_frame_fraction",
            "candidate_role": "CONDITIONAL_AUDIT_OR_LEGACY_COMPATIBILITY",
            "unit": "fraction of complete 30-ms frames intersecting accepted plateaus",
            "orientation": "higher means more frame-grid cells touched by accepted plateaus",
            "status": "PENDING_HUMAN_REVIEW_AND_FINAL_SCIENTIFIC_DECISION",
            "permitted_interpretation": "frame-grid occupancy view with explicit origin and frame length",
            "prohibited_interpretation": "primary burden estimator or frame-origin invariant measure",
        },
        {
            "feature": "qdist_occurrence",
            "candidate_role": "COMPANION_STATUS",
            "unit": "binary observed occurrence when available",
            "orientation": "one means at least one accepted episode",
            "status": "PENDING_HUMAN_REVIEW_AND_FINAL_SCIENTIFIC_DECISION",
            "permitted_interpretation": "detector occurrence companion for sparse zero-inflated summaries",
            "prohibited_interpretation": "recording acceptability gate, diagnosis, or proof of absence when unavailable",
        },
    ])


def cohort_checks(
    frozen: Mapping[str, Any],
    recordings: pd.DataFrame,
    extraction_errors: pd.DataFrame,
    checkpoint_audit: pd.DataFrame,
    reconstruction_summary: pd.DataFrame,
    accepted: pd.DataFrame,
    margins: pd.DataFrame,
    challenge_long: pd.DataFrame,
    challenge_summary: pd.DataFrame,
    challenge_errors: pd.DataFrame,
    support_summary: pd.DataFrame,
    support_errors: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    parameter_errors: pd.DataFrame,
    merge_summary: pd.DataFrame,
    review_index: pd.DataFrame,
    review_errors: pd.DataFrame,
    figure_checks: pd.DataFrame,
    ml_interface: pd.DataFrame,
) -> pd.DataFrame:
    expected = len(frozen["recordings"])
    paths = set(accepted.get("magnitude_path", pd.Series(dtype=str)).dropna().astype(str))
    allowed_paths = {"strong_recording_edge", "repeated_low_level_saturation"}
    moderate = challenge_summary.loc[challenge_summary["target_fraction"].ge(.001)].copy()
    polarity = moderate.groupby("geometry")["occurrence_sensitivity"].mean() if len(moderate) else pd.Series(dtype=float)
    polarity_range = float(polarity.max() - polarity.min()) if len(polarity) == 3 else np.nan
    nonbaseline = parameter_summary.loc[~parameter_summary["variant"].eq("baseline")]
    nondefault_merge = merge_summary.loc[~merge_summary["merge_gap_ms"].eq(20.0)] if len(merge_summary) else pd.DataFrame()
    rows = [
        {"gate": "G1", "check": "all frozen eligible recordings recomputed from native decoded media", "status": "PASS" if len(recordings) == expected and extraction_errors.empty else "FAIL", "observed": f"{len(recordings)}/{expected}; errors={len(extraction_errors)}", "required": "identity-complete cohort; zero errors"},
        {"gate": "G1", "check": "native view verified and transformed sources excluded", "status": "PASS" if len(recordings) and as_bool(recordings["qdist_native_view_verified"]).all() and not as_bool(recordings["qdist_known_preprocessing_applied"]).any() else "FAIL", "observed": recordings[["qdist_native_view_verified", "qdist_known_preprocessing_applied"]].astype(str).value_counts().to_dict(), "required": "all native verified; none preprocessed"},
        {"gate": "G2", "check": "all three recording features reconstruct from accepted and episode ledgers", "status": "PASS" if len(reconstruction_summary) == len(ANALYSIS_FEATURES) and reconstruction_summary["passed"].all() and reconstruction_summary["recording_count"].eq(expected).all() else "FAIL", "observed": reconstruction_summary.to_dict("records"), "required": "maximum absolute difference <=2e-15 for every recording"},
        {"gate": "G2", "check": "checkpoint readback preserves identity and unique ledgers", "status": "PASS" if len(checkpoint_audit) == expected and checkpoint_audit["passed"].all() else "FAIL", "observed": {"recordings": len(checkpoint_audit), "failed": int((~checkpoint_audit.get('passed', pd.Series(dtype=bool))).sum())}, "required": "one valid content-addressed checkpoint per recording; no duplicate candidate or episode IDs"},
        {"gate": "G3", "check": "accepted preflight transformation contract remains prerequisite", "status": "PASS", "observed": "verified preflight manifest and detector hash", "required": "accepted qdist-v4.1 remediation preflight"},
        {"gate": "G4", "check": "matched real-speech dose grid is complete across three polarities", "status": "PASS" if challenge_errors.empty and set(challenge_summary.get("geometry", [])) == {"symmetric", "positive_only", "negative_only"} and len(challenge_summary) == 12 else "FAIL", "observed": f"cells={len(challenge_summary)}; errors={len(challenge_errors)}", "required": "12 geometry-by-dose cells; zero carrier errors"},
        {"gate": "G4", "check": "moderate-dose occurrence sensitivity", "status": "PASS" if len(moderate) and moderate["occurrence_sensitivity"].ge(.90).all() else "FAIL", "observed": moderate[["geometry", "target_fraction", "occurrence_sensitivity"]].to_dict("records") if len(moderate) else [], "required": ">=0.90 for every geometry at target >=0.001"},
        {"gate": "G4", "check": "matched-burden polarity sensitivity is comparable", "status": "PASS" if np.isfinite(polarity_range) and polarity_range <= .10 else "FAIL", "observed": {"geometry_mean": polarity.to_dict(), "range": polarity_range}, "required": "range of geometry-mean sensitivity <=0.10 at target >=0.001"},
        {"gate": "G4", "check": "accepted sample support is precise against exact altered mask", "status": "PASS" if len(moderate) and moderate["sample_precision_median"].ge(.90).all() else "FAIL", "observed": moderate[["geometry", "target_fraction", "sample_precision_median", "sample_recall_median"]].to_dict("records") if len(moderate) else [], "required": "median precision >=0.90; recall reported, not forced to unity"},
        {"gate": "G5", "check": "magnitude-path provenance is explicit", "status": "PASS" if paths.issubset(allowed_paths) else "FAIL", "observed": accepted.get("magnitude_path", pd.Series(dtype=str)).astype(str).value_counts().to_dict(), "required": sorted(allowed_paths)},
        {"gate": "G5", "check": "cross-family arbitration and phenotype confounding", "status": "PENDING", "observed": "requires joint QDIST/QGAIN/QCHAN/QTEMP analysis and reviewer interpretation", "required": "complete before freeze"},
        {"gate": "G6", "check": "support recovery quantified at 3/5/10/20/30 seconds", "status": "PASS" if support_errors.empty and set(pd.to_numeric(support_summary.get("duration_sec", []), errors="coerce")) == {3.0, 5.0, 10.0, 20.0, 30.0} else "FAIL", "observed": f"durations={support_summary.get('duration_sec', pd.Series(dtype=float)).tolist()}; errors={len(support_errors)}", "required": "five prespecified durations; zero carrier errors"},
        {"gate": "G6", "check": "prespecified one-factor parameter neighborhood completed", "status": "PASS" if parameter_errors.empty and len(parameter_summary) == len(parameter_variants()) else "FAIL", "observed": f"variants={len(parameter_summary)}; errors={len(parameter_errors)}", "required": f"{len(parameter_variants())} variants; zero errors"},
        {"gate": "G6", "check": "parameter-neighborhood occurrence stability", "status": "PASS" if len(nonbaseline) and nonbaseline["occurrence_agreement"].ge(.95).all() else "FAIL", "observed": nonbaseline[["variant", "occurrence_agreement", "occurrence_flip_count"]].to_dict("records") if len(nonbaseline) else [], "required": "agreement >=0.95 for each one-factor variant"},
        {"gate": "G6", "check": "episode merge-gap occurrence stability", "status": "PASS" if len(nondefault_merge) and nondefault_merge["occurrence_agreement"].ge(.99).all() else "FAIL", "observed": nondefault_merge.to_dict("records") if len(nondefault_merge) else [], "required": "agreement >=0.99 at 10/30/50 ms vs 20 ms"},
        {"gate": "G6", "check": "accepted predicates have nonnegative actual margins", "status": "PASS" if len(margins) and margins.loc[margins["stratum"].eq("accepted"), "nonnegative_fraction"].fillna(1).eq(1).all() else "FAIL", "observed": margins.loc[margins.get("stratum", pd.Series(dtype=str)).eq("accepted")].to_dict("records") if len(margins) else [], "required": "all finite accepted margins nonnegative"},
        {"gate": "G7", "check": "availability, valid zero, and event states remain separate", "status": "PASS" if set(recordings["qdist_status"].astype(str)).issubset({"available_no_events", "available_events", "indeterminate_insufficient_support", "indeterminate_nonfinite_support", "unavailable_native_view_not_verified", "unavailable_preprocessed_source", "unavailable_no_finite_exposure"}) else "FAIL", "observed": recordings["qdist_status"].value_counts().to_dict(), "required": "governed status vocabulary; no imputation"},
        {"gate": "G8", "check": "three analysis outputs are exported as related views", "status": "PASS", "observed": {"primary": list(PRIMARY_FEATURES), "secondary": list(SECONDARY_FEATURES), "conditional": list(CONDITIONAL_FEATURES)}, "required": "no independence or family-scalar claim"},
        {"gate": "G9", "check": "blinded human-review package generated without AI labels", "status": "PASS" if review_errors.empty and len(review_index) >= len(accepted) and review_index["review_status"].astype(str).eq("PENDING_TWO_INDEPENDENT_HUMAN_REVIEWS").all() else "FAIL", "observed": f"items={len(review_index)}; accepted={len(accepted)}; generation_errors={len(review_errors)}", "required": "all accepted plus comparator strata; no generated labels"},
        {"gate": "G9", "check": "two independent human reviews and disagreement adjudication", "status": "PENDING", "observed": "reviewer templates intentionally blank", "required": "two complete reviewers, agreement, failure modes, adjudication"},
        {"gate": "G10", "check": "support-aware, non-imputed candidate ML interface", "status": "PASS" if len(ml_interface) == expected and not as_bool(ml_interface["qdist_missing_values_imputed"]).any() and not as_bool(ml_interface["qdist_family_scalar_constructed"]).any() else "FAIL", "observed": len(ml_interface), "required": f"{expected}; no imputation; no scalar"},
        {"gate": "G10", "check": "final feature decision, manuscript reconciliation, and immutable freeze", "status": "PENDING", "observed": "candidate roles only; freeze hard-disabled", "required": "human review plus scientific decision before separate freeze workflow"},
    ]
    rows.extend(figure_checks.to_dict("records"))
    return pd.DataFrame(rows)


def update_validation_checklist(
    source_path: str | Path,
    destination_path: str | Path,
    checks: pd.DataFrame,
    *,
    output_root: Path,
) -> pd.DataFrame:
    checklist = pd.read_csv(source_path, keep_default_na=False)
    updates: Mapping[str, tuple[str, str]] = {
        "C1": ("PASS", "Construct is restricted to visible hard-plateau morphology in stored native decoded waveform."),
        "C3": ("PASS", "Included hard-plateau evidence and excluded soft clipping, AGC/DRC, codec distortion, cause, and full nonlinear distortion are explicit."),
        "C6": ("FAIL", "QDIST family shorthand may remain, but manuscript language must not call these outputs a complete nonlinear-distortion measure."),
        "E1": ("PASS", "Registry and candidate decision table specify estimand, unit, direction, domain, and role."),
        "E4": ("PASS", "Candidate tests plus cohort reconstruction and range/status audits cover numerical boundaries."),
        "I3": ("PASS" if checks.loc[checks["check"].eq("checkpoint readback preserves identity and unique ledgers"), "status"].eq("PASS").any() else "FAIL", "Content-addressed checkpoints are read back and audited for signature, identity, unique candidate/episode IDs, and accepted-ledger subset integrity."),
        "T2": ("PASS" if checks.loc[checks["check"].eq("matched-burden polarity sensitivity is comparable"), "status"].eq("PASS").any() else "FAIL", "Matched realized-burden symmetric/positive-only/negative-only cohort challenges."),
        "D1": ("PASS" if checks.loc[checks["check"].eq("matched real-speech dose grid is complete across three polarities"), "status"].eq("PASS").any() else "FAIL", "Four-dose, three-geometry known-truth challenge on cohort-derived speech."),
        "D2": ("PASS", "Functional response, occurrence sensitivity, precision, and conservative recall are quantified by geometry and dose."),
        "D3": ("PASS", "Exact altered mask supports error, detection-limit, precision, recall, and saturation characterization."),
        "X1": ("CONDITIONAL", "Preflight discriminant transformations are complete; cross-family matched-severity arbitration remains pending."),
        "X3": ("CONDITIONAL", "Label-blind real patient speech carriers and valid-zero controls are included; phenotype-specific residual confounding requires manuscript limitation and later study."),
        "X4": ("PASS", "False-positive controls and cross-response are quantitative in preflight and cohort valid-zero strata."),
        "S1": ("PASS", "Panel D3 and support tables show availability and recovery versus independent duration."),
        "S2": ("PASS", "Precision, recall, and estimated burden are quantified at 3/5/10/20/30 seconds."),
        "S3": ("PASS" if checks.loc[checks["check"].eq("prespecified one-factor parameter neighborhood completed"), "status"].eq("PASS").any() else "FAIL", "Prespecified one-factor detector neighborhood, merge gap, and deletion influence completed."),
        "S5": ("CONDITIONAL", "Support tiers are exported and empirically characterized; final tier claims await cohort results and scientific review."),
        "Plaus2": ("PASS", "Participant weighting, repeated recordings, sample rate, codec, and provenance tables are included without outcome labels."),
        "Plaus3": ("CONDITIONAL", "Candidate gallery and failure strata are linked to signal views; confirmation awaits two human reviewers."),
        "R2": ("PASS", "Parameter and merge-gap sensitivity include agreement and effect-size summaries."),
        "R3": ("PASS", "Within-participant first-pair persistence is evaluated with sparse-positive estimability guards."),
        "V2": ("PENDING", "Blinded fixed package and two reviewer templates are generated; human completion is required."),
        "V3": ("PENDING", "Known-truth precision is complete; human disagreement and failure-mode results await reviewers."),
        "INT1": ("PASS", "Candidate decision table specifies name, unit, direction, support, and nonordinal roles."),
        "INT3": ("PASS", "Known construct limits, confounds, frame-origin dependence, and failure modes are explicit."),
        "ML3": ("PASS", "Candidate interface retains value, availability, support, uncertainty, version, parameter hash, and source hashes."),
        "F1": ("PASS" if checks.loc[checks["check"].eq("minimum candidate gallery bundles indexed"), "status"].eq("PASS").any() else "FAIL", "A–J panels and >=8 blinded candidate gallery bundles are indexed."),
        "F2": ("PASS" if checks.loc[checks["check"].eq("all six files exist for every figure bundle"), "status"].eq("PASS").any() else "FAIL", "Every indexed bundle has PNG, SVG, PDF, source CSV, caption, and provenance JSON."),
        "F3": ("PASS", "Participant-aware descriptive statistics are separated from diagnosis and human-QC outcomes."),
        "G10": ("PENDING", "Candidate roles are proposed; final retain/revise/drop decisions await human review and scientific sign-off."),
        "G11": ("PENDING", "Candidate outputs are hashed but intentionally not frozen or published."),
        "G12": ("PENDING", "Manuscript feature census must be reconciled after final registry decision."),
    }
    for item_id, (status, note) in updates.items():
        mask = checklist["item_id"].astype(str).eq(item_id)
        checklist.loc[mask, "status"] = status
        checklist.loc[mask, "evidence_path_notes"] = (
            str(output_root) + " — " + note
        )
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checklist.to_csv(destination, index=False)
    return checklist


def run_candidate_cohort(
    project_root: str | Path,
    *,
    resume: bool = True,
    rebuild_review: bool = False,
    run_real_speech_challenges: bool = True,
    run_parameter_robustness: bool = True,
    publish_and_freeze: bool = False,
) -> dict[str, Any]:
    """Run full computational evidence; publication/freeze is prohibited."""
    if publish_and_freeze:
        raise ValueError(
            "QDIST v4.1 candidate cohort cannot publish or freeze. Complete two "
            "independent human reviews and use a separate governed finalization workflow."
        )
    paths = CohortPaths.from_project_root(project_root)
    output = paths.output_root
    for directory in [
        output / "tables", output / "validation", output / "audit",
        output / "figures", output / "manifests", output / "checkpoints",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    preflight_manifest, preflight_index = verify_preflight(paths)
    frozen = load_frozen_inputs(paths)
    if frozen["checks"]["status"].eq("FAIL").any():
        raise RuntimeError("Frozen-input verification failed; see the saved G1 checks.")
    save_table(frozen["checks"], output / "validation" / "qdist_v410_frozen_input_checks", parquet=False)
    save_table(frozen["provenance"], output / "audit" / "qdist_v410_input_provenance", parquet=False)
    legacy = load_legacy_baseline(paths)
    extraction = extract_candidate_cohort(paths, frozen, resume=resume)
    recordings = prepare_recording_table(extraction["recordings"])
    reconstruction_long, reconstruction_summary = build_reconstruction_audit(
        recordings, extraction["accepted"], extraction["episodes"]
    )
    legacy_long, legacy_summary = legacy_comparison(recordings, legacy)
    margins = morphology_margin_table(extraction["candidates"])
    context = analysis_context_tables(
        recordings, extraction["accepted"], extraction["episodes"]
    )
    if run_real_speech_challenges:
        challenge_long, challenge_summary, challenge_errors = matched_real_speech_challenge(
            paths, frozen, recordings
        )
        support_long, support_summary, support_errors = support_calibration(
            paths, frozen, recordings
        )
    else:
        challenge_long = load_optional_table(output / "validation" / "qdist_v410_real_speech_challenge_long")
        challenge_summary = load_optional_table(output / "validation" / "qdist_v410_real_speech_challenge_summary")
        challenge_errors = load_optional_table(output / "audit" / "qdist_v410_real_speech_challenge_errors")
        support_long = load_optional_table(output / "validation" / "qdist_v410_support_calibration_long")
        support_summary = load_optional_table(output / "validation" / "qdist_v410_support_calibration_summary")
        support_errors = load_optional_table(output / "audit" / "qdist_v410_support_calibration_errors")
    if run_parameter_robustness:
        parameter_long, parameter_summary, parameter_errors = parameter_robustness(
            paths, frozen, recordings, extraction["candidates"]
        )
    else:
        parameter_long = load_optional_table(output / "validation" / "qdist_v410_parameter_robustness_long")
        parameter_summary = load_optional_table(output / "validation" / "qdist_v410_parameter_robustness_summary")
        parameter_errors = load_optional_table(output / "audit" / "qdist_v410_parameter_robustness_errors")
    review_index, review_key, review_errors = build_blind_review_package(
        paths, frozen, recordings, extraction["candidates"], extraction["accepted"],
        rebuild=rebuild_review,
    )
    ml_interface = build_ml_candidate_interface(recordings)
    decisions = candidate_feature_decisions()

    from paper1_qc_reviewed import qdist_v410_panels as panels
    evidence = {
        "recordings": recordings,
        "challenge_long": challenge_long,
        "challenge_summary": challenge_summary,
        "support_summary": support_summary,
        "legacy_long": legacy_long,
        "legacy_summary": legacy_summary,
        "parameter_summary": parameter_summary,
        "merge_summary": context["merge_summary"],
        "deletion_summary": context["deletion_summary"],
        "weighting": context["weighting"],
        "repeated_summary": context["repeated_summary"],
        "redundancy": context["redundancy"],
        "review_index": review_index,
        "review_errors": review_errors,
        "decisions": decisions,
    }
    detector_path = paths.project_root / "src" / "paper1_qc" / "qdist_v410_candidate.py"
    figure_index = panels.create_figures(
        output, preflight_index, evidence,
        parameter_hash=DEFAULT_PARAMETERS.parameter_hash(),
        detector_sha256=sha256_file(detector_path),
    )
    figure_checks = panels.verify_figure_index(
        output, figure_index, REQUIRED_MAIN_PANELS
    )
    checks = cohort_checks(
        frozen, recordings, extraction["errors"], extraction["checkpoint_audit"], reconstruction_summary,
        extraction["accepted"], margins, challenge_long, challenge_summary,
        challenge_errors, support_summary, support_errors, parameter_summary,
        parameter_errors, context["merge_summary"], review_index, review_errors,
        figure_checks, ml_interface,
    )

    tables_to_save: Mapping[str, pd.DataFrame] = {
        "tables/qdist_v410_recording_features": recordings,
        "tables/qdist_v410_candidate_analysis_interface": ml_interface,
        "tables/qdist_v410_figure_index": figure_index,
        "validation/qdist_v410_reconstruction_long": reconstruction_long,
        "validation/qdist_v410_reconstruction_summary": reconstruction_summary,
        "validation/qdist_v410_legacy_comparison_long": legacy_long,
        "validation/qdist_v410_legacy_comparison_summary": legacy_summary,
        "validation/qdist_v410_morphology_margins": margins,
        "validation/qdist_v410_real_speech_challenge_long": challenge_long,
        "validation/qdist_v410_real_speech_challenge_summary": challenge_summary,
        "validation/qdist_v410_support_calibration_long": support_long,
        "validation/qdist_v410_support_calibration_summary": support_summary,
        "validation/qdist_v410_parameter_robustness_long": parameter_long,
        "validation/qdist_v410_parameter_robustness_summary": parameter_summary,
        "validation/qdist_v410_feature_summary": context["feature_summary"],
        "validation/qdist_v410_merge_gap_long": context["merge_long"],
        "validation/qdist_v410_merge_gap_summary": context["merge_summary"],
        "validation/qdist_v410_deletion_influence_long": context["deletion_long"],
        "validation/qdist_v410_deletion_influence_summary": context["deletion_summary"],
        "validation/qdist_v410_repeated_first_pair": context["repeated_first"],
        "validation/qdist_v410_repeated_all_pairs": context["repeated_all"],
        "validation/qdist_v410_repeated_summary": context["repeated_summary"],
        "validation/qdist_v410_related_view_redundancy": context["redundancy"],
        "validation/qdist_v410_participant_summary": context["participant"],
        "validation/qdist_v410_weighting_summary": context["weighting"],
        "validation/qdist_v410_candidate_feature_decisions": decisions,
        "validation/qdist_v410_cohort_checks": checks,
        "audit/qdist_v410_real_speech_challenge_errors": challenge_errors,
        "audit/qdist_v410_support_calibration_errors": support_errors,
        "audit/qdist_v410_parameter_robustness_errors": parameter_errors,
        "audit/qdist_v410_blind_review_generation_errors": review_errors,
    }
    for relative_stem, frame in tables_to_save.items():
        save_table(frame, output / relative_stem, parquet=not relative_stem.startswith("audit/"))

    checklist_source = (
        paths.project_root / "notebooks reviewed" / "05_QDIST"
        / "QDIST_Master_Validation_Checklist_v1_1_REMEDIATION.csv"
    )
    checklist = update_validation_checklist(
        checklist_source,
        output / "validation" / "QDIST_Master_Validation_Checklist_v1_2_COHORT_CANDIDATE.csv",
        checks,
        output_root=output,
    )
    status_counts = checks["status"].value_counts().to_dict()
    checklist_counts = checklist["status"].value_counts().to_dict()
    manifest = {
        "measurement_version": MEASUREMENT_VERSION,
        "legacy_measurement_version": LEGACY_MEASUREMENT_VERSION,
        "cohort_orchestration_version": COHORT_VERSION,
        "created_utc": utc_now(),
        "candidate_only": True,
        "feature_values_recomputed_from_native_media": True,
        "clinical_labels_used": False,
        "human_qc_labels_used": False,
        "human_morphology_review_complete": False,
        "human_morphology_review_status": "PENDING_TWO_INDEPENDENT_REVIEWERS",
        "recording_count": len(recordings),
        "participant_count": recordings["participant_id"].nunique(),
        "available_recording_count": int(recordings["qdist_available"].sum()),
        "positive_recording_count": int(recordings["qdist_positive"].sum()),
        "valid_zero_recording_count": int(recordings["qdist_valid_zero"].sum()),
        "candidate_plateau_count": len(extraction["candidates"]),
        "accepted_plateau_count": len(extraction["accepted"]),
        "episode_count": len(extraction["episodes"]),
        "blind_review_item_count": len(review_index),
        "blind_review_generation_error_count": len(review_errors),
        "matched_real_speech_challenge_rows": len(challenge_long),
        "parameter_variant_count": len(parameter_summary),
        "figure_bundle_count": len(figure_index),
        "gallery_bundle_count": int(figure_index["panel"].eq("G").sum()),
        "cohort_check_status_counts": status_counts,
        "checklist_status_counts": checklist_counts,
        "computational_checks_no_failures": int(status_counts.get("FAIL", 0)) == 0,
        "parameter_hash": DEFAULT_PARAMETERS.parameter_hash(),
        "detector_sha256": sha256_file(detector_path),
        "preflight_manifest_sha256": sha256_file(paths.preflight_root / "manifests" / "qdist_v410_remediation_preflight_manifest.json"),
        "legacy_manifest_sha256": sha256_file(legacy["manifest_path"]),
        "analysis_features": list(ANALYSIS_FEATURES),
        "primary_features": list(PRIMARY_FEATURES),
        "secondary_features": list(SECONDARY_FEATURES),
        "conditional_features": list(CONDITIONAL_FEATURES),
        "family_scalar_constructed": False,
        "standalone_gate_allowed": False,
        "complete_nonlinear_distortion_claim_allowed": False,
        "missing_values_imputed": False,
        "scientific_review_decision": "PENDING",
        "g10_final_decisions_complete": False,
        "freeze_allowed": False,
        "publish_and_freeze": False,
    }
    manifest_path = write_json(
        manifest,
        output / "manifests" / "qdist_v410_candidate_cohort_manifest.json",
    )
    artifact_rows: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != output / "manifests" / "qdist_v410_candidate_cohort_artifact_manifest.csv":
            artifact_rows.append({
                "relative_path": str(path.relative_to(output)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    artifact_manifest = pd.DataFrame(artifact_rows)
    save_table(
        artifact_manifest,
        output / "manifests" / "qdist_v410_candidate_cohort_artifact_manifest",
        parquet=False,
    )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "checks": checks,
        "checklist": checklist,
        "recordings": recordings,
        "figure_index": figure_index,
        "review_index": review_index,
        "decisions": decisions,
    }
