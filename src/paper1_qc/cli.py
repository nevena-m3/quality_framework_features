from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress display is non-essential

    def tqdm(iterable, **_kwargs):
        return iterable


from .config import load_config, resolve_data_path, resolve_executable, resolve_project_path
from .freeze import (
    apply_metadata_analysis_gates,
    attach_media_freeze_gate,
    diagnosis_adjudication_template,
    issue_disposition_table,
    load_adjudications,
    resolve_diagnoses,
)
from .human_qc import (
    agreement_summary,
    load_human_qc_long,
    load_interval_human_qc,
    make_consensus,
    make_extent_consensus,
    rating_design_coverage,
)
from .media import (
    audit_native_metadata_consistency,
    build_media_inventory,
    decode_audio_views,
    reconcile_inventory_with_metadata,
)
from .metadata import audit_metadata_workbook, exact_session_pairs, reconcile_workbooks
from .metrics import extract_all_metrics, rest_reference_metrics
from .provenance import write_run_manifest
from .registry import metric_registry_frame
from .reviewed_features import (
    build_latest_feature_release,
    load_latest_feature_release,
)
from .segmentation import (
    LEGACY_SILERO_FRAME_COLUMNS,
    LEGACY_SILERO_SEGMENT_COLUMNS,
    LEGACY_SILERO_SUMMARY_COLUMNS,
    MANUAL_SEGMENTATION_COLUMNS,
    Interval,
    apply_segmentation_adjudication,
    boundary_alignment_diagnostics,
    build_segmentation_views,
    classify_reading_segmentation,
    freeze_segmentation_intervals,
    intervals_to_frame,
    legacy_silero_artifacts,
    load_silero_model,
    plot_boundary_alignment_audit,
    plot_segmentation_diagnostic,
    plot_segmentation_failure,
    segmentation_adjudication_template,
    segmentation_pending_reviews,
    silero_speech_intervals,
    silero_speech_timestamps,
    summarize_legacy_silero_artifacts,
    validate_manual_segmentation_overrides,
)
from .statistics import (
    compare_binary_label_systems,
    describe_metrics,
    direction_oriented_family_indices,
    family_alignment_matrix,
    matched_family_specificity,
    one_recording_per_participant,
    pairwise_clustered_spearman,
    participant_level_group_contrasts,
    participant_persistence,
    perceptual_links,
    rater_stratified_family_alignment,
)


def _output_root(cfg: dict) -> Path:
    root = Path(cfg["_output_root"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _data_freeze_root(cfg: dict) -> Path:
    version = str(cfg.get("data_freeze", {}).get("version", "v1")).strip()
    if not version or version in {".", ".."} or "/" in version or "\\" in version:
        raise ValueError(f"Invalid data_freeze.version: {version!r}")
    return Path(cfg["_main_output_root"]) / "00_DATA_FREEZE" / version


def _segmentation_freeze_root(cfg: dict) -> Path:
    version = str(
        cfg.get("segmentation_freeze", {}).get(
            "version",
            cfg.get("data_freeze", {}).get("version", "v1"),
        )
    ).strip()
    if not version or version in {".", ".."} or "/" in version or "\\" in version:
        raise ValueError(f"Invalid segmentation_freeze.version: {version!r}")
    return Path(cfg["_main_output_root"]) / "01_SEGMENTATION_FREEZE" / version


def _read_frozen(cfg: dict, stem: str) -> pd.DataFrame:
    path = _data_freeze_root(cfg) / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen input missing: {path}. Run freeze-template, complete the adjudication "
            "file, and run freeze before segmentation."
        )
    return pd.read_csv(path)


def _write_table(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(path_without_suffix.with_suffix(".parquet"), index=False)
    except (ImportError, ValueError, TypeError):
        pass


def _as_bool(series: pd.Series) -> pd.Series:
    """Normalize booleans reloaded from CSV without treating 'False' as truthy."""
    return series.map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().lower() in {"true", "1", "yes", "y"}
        )
    ).fillna(False)


def _audit_kwargs(cfg: dict) -> dict:
    return {
        "sentinel_values": cfg["clinical_alignment"]["sentinel_values"],
        "control_id_patterns": cfg["cohort"]["control_id_patterns"],
        "media_preference": cfg["cohort"]["media_preference"],
        "max_primary_assessment_gap_days": cfg["clinical_alignment"][
            "primary_max_assessment_gap_days"
        ],
    }


def command_audit(cfg: dict) -> None:
    output = _output_root(cfg) / "00_audit"
    inputs = {
        "bamboo": resolve_data_path(cfg, "bamboo_metadata"),
        "rest": resolve_data_path(cfg, "rest_metadata"),
        "combined": resolve_data_path(cfg, "combined_metadata"),
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Metadata files not found: {missing}")
    audits = {
        name: audit_metadata_workbook(path, **_audit_kwargs(cfg)) for name, path in inputs.items()
    }
    for name, audit in audits.items():
        _write_table(audit.clean_media_rows, output / f"{name}_media_rows_audited")
        _write_table(audit.canonical_recordings, output / f"{name}_canonical_recordings")
        _write_table(audit.issues, output / f"{name}_audit_issues")
        _write_table(audit.summary, output / f"{name}_audit_summary")
        _write_table(audit.column_profile, output / f"{name}_column_profile")

    reconciliation, discrepancies = reconcile_workbooks(
        audits["bamboo"], audits["rest"], audits["combined"]
    )
    _write_table(reconciliation, output / "cross_workbook_summary")
    _write_table(discrepancies, output / "cross_workbook_discrepancies")
    pairs = exact_session_pairs(
        audits["bamboo"].canonical_recordings, audits["rest"].canonical_recordings
    )
    _write_table(pairs, output / "exact_bamboo_rest_session_pairs")
    _write_table(metric_registry_frame(), output / "metric_registry")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=inputs.values(),
    )


def command_inventory(cfg: dict, *, hashes: bool) -> None:
    output = _output_root(cfg) / "00_audit"
    ffprobe = resolve_executable(cfg["software"]["ffprobe"], "ffprobe")
    ffmpeg = resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg")
    roots = {
        "bamboo": resolve_data_path(cfg, "bamboo_audio"),
        "rest": resolve_data_path(cfg, "rest_audio"),
        "combined": resolve_data_path(cfg, "combined_audio"),
    }
    inventories = []
    for role, root in roots.items():
        inventory = build_media_inventory(
            [root], ffprobe=ffprobe, ffmpeg=ffmpeg, compute_hashes=hashes
        )
        inventory["dataset_root_role"] = role
        _write_table(inventory, output / f"{role}_media_inventory")
        inventories.append(inventory)
        metadata_path = output / f"{role}_media_rows_audited.parquet"
        metadata_csv = output / f"{role}_media_rows_audited.csv"
        if metadata_path.exists() or metadata_csv.exists():
            metadata = (
                pd.read_parquet(metadata_path)
                if metadata_path.exists()
                else pd.read_csv(metadata_csv)
            )
            coverage, duplicates = reconcile_inventory_with_metadata(inventory, metadata)
            consistency = audit_native_metadata_consistency(inventory, metadata)
            _write_table(coverage, output / f"{role}_media_coverage")
            _write_table(duplicates, output / f"{role}_duplicate_disk_names")
            _write_table(consistency, output / f"{role}_native_metadata_consistency_issues")
    combined_inventory = (
        pd.concat(inventories, ignore_index=True) if inventories else pd.DataFrame()
    )
    _write_table(combined_inventory, output / "all_media_inventory")
    write_run_manifest(
        output / "inventory_run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=roots.values(),
        extra={"compute_sha256": hashes},
    )


def _freeze_settings(cfg: dict) -> dict:
    settings = cfg.get("data_freeze", {})
    return {
        "control_id_patterns": cfg["cohort"]["control_id_patterns"],
        "confirm_control_patterns": bool(
            settings.get("confirm_configured_control_id_patterns", False)
        ),
        "control_rule_evidence": str(settings.get("control_id_rule_evidence", "")).strip(),
        "confirmed_control_subject_ids": settings.get("confirmed_control_subject_ids", []),
        "confirmed_control_subject_evidence": str(
            settings.get("confirmed_control_subject_evidence", "")
        ).strip(),
        "allowed_diagnoses": cfg["cohort"]["allowed_diagnoses"],
        "adjudication_path": resolve_project_path(
            cfg, settings.get("diagnosis_adjudication", "config/metadata_adjudication.csv")
        ),
    }


def command_freeze_template(cfg: dict) -> None:
    audit_root = _output_root(cfg) / "00_audit"
    frames = [
        _read_existing(audit_root, "bamboo_canonical_recordings"),
        _read_existing(audit_root, "rest_canonical_recordings"),
    ]
    settings = _freeze_settings(cfg)
    if settings["confirm_control_patterns"] and not settings["control_rule_evidence"]:
        raise ValueError(
            "data_freeze.control_id_rule_evidence is required when configured ID "
            "patterns are promoted to confirmed controls."
        )
    if (
        settings["confirmed_control_subject_ids"]
        and not settings["confirmed_control_subject_evidence"]
    ):
        raise ValueError(
            "data_freeze.confirmed_control_subject_evidence is required when "
            "exceptional control IDs are confirmed."
        )
    template = diagnosis_adjudication_template(
        frames,
        control_id_patterns=settings["control_id_patterns"],
        confirm_control_patterns=settings["confirm_control_patterns"],
        confirmed_control_subject_ids=settings["confirmed_control_subject_ids"],
    )
    destination = settings["adjudication_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Adjudication file already exists; not overwritten: {destination}")
    else:
        template.to_csv(destination, index=False)
        print(f"Wrote adjudication template: {destination}")
    print(template.to_string(index=False))


def command_freeze(cfg: dict) -> None:
    """Freeze one versioned, hash-bearing cohort before any signal processing."""
    root = _output_root(cfg)
    audit_root = root / "00_audit"
    freeze_root = _data_freeze_root(cfg)
    manifest_path = freeze_root / "data_freeze_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Freeze version already exists and is immutable: {freeze_root}. "
            "Change data_freeze.version only after documenting why a new freeze is needed."
        )

    settings = _freeze_settings(cfg)
    if settings["confirm_control_patterns"] and not settings["control_rule_evidence"]:
        raise ValueError(
            "data_freeze.control_id_rule_evidence is required when configured ID "
            "patterns are promoted to confirmed controls."
        )
    if (
        settings["confirmed_control_subject_ids"]
        and not settings["confirmed_control_subject_evidence"]
    ):
        raise ValueError(
            "data_freeze.confirmed_control_subject_evidence is required when "
            "exceptional control IDs are confirmed."
        )
    adjudications = load_adjudications(settings["adjudication_path"])
    role_inputs: dict[str, dict[str, pd.DataFrame]] = {}
    for role in ["bamboo", "rest"]:
        role_inputs[role] = {
            "canonical": _read_existing(audit_root, f"{role}_canonical_recordings"),
            "media_rows": _read_existing(audit_root, f"{role}_media_rows_audited"),
            "issues": _read_existing(audit_root, f"{role}_audit_issues"),
            "inventory": _read_existing(audit_root, f"{role}_media_inventory"),
        }

    template = diagnosis_adjudication_template(
        [values["canonical"] for values in role_inputs.values()],
        control_id_patterns=settings["control_id_patterns"],
        confirm_control_patterns=settings["confirm_control_patterns"],
        confirmed_control_subject_ids=settings["confirmed_control_subject_ids"],
    )
    required_ids = set(template["SubjectID"])
    supplied_ids = set(adjudications["SubjectID"])
    missing_ids = sorted(required_ids - supplied_ids)
    if missing_ids:
        required_path = audit_root / "diagnosis_resolution_required.csv"
        template.to_csv(required_path, index=False)
        raise ValueError(
            "Diagnosis freeze is blocked. Complete ALS, CONTROLS, or EXCLUDE decisions "
            f"for {missing_ids} in {settings['adjudication_path']}. A review table was "
            f"written to {required_path}."
        )

    ledgers: dict[str, pd.DataFrame] = {}
    dispositions: list[pd.DataFrame] = []
    for role, inputs in role_inputs.items():
        resolved = resolve_diagnoses(
            inputs["canonical"],
            adjudications=adjudications,
            allowed_diagnoses=settings["allowed_diagnoses"],
            control_id_patterns=settings["control_id_patterns"],
            confirm_control_patterns=settings["confirm_control_patterns"],
            control_rule_evidence=settings["control_rule_evidence"],
            confirmed_control_subject_ids=settings["confirmed_control_subject_ids"],
            confirmed_control_subject_evidence=settings["confirmed_control_subject_evidence"],
        )
        pending = sorted(
            resolved.loc[resolved["diagnosis_resolution"].eq("pending"), "SubjectID"].unique()
        )
        if pending:
            raise ValueError(f"Diagnosis remains pending after adjudication: {pending}")
        with_media = attach_media_freeze_gate(
            resolved,
            inputs["inventory"],
            media_rows=inputs["media_rows"],
            media_preference=cfg["cohort"]["media_preference"],
        )
        disposition = issue_disposition_table(
            inputs["issues"], inputs["media_rows"], with_media
        ).assign(dataset_role=role)
        ledger = apply_metadata_analysis_gates(with_media, disposition)
        ledgers[role] = ledger
        dispositions.append(disposition)

    freeze_root.mkdir(parents=True, exist_ok=False)
    for role, ledger in ledgers.items():
        ledger.to_csv(freeze_root / f"{role}_recording_freeze_ledger.csv", index=False)
        ledger.loc[ledger["freeze_included"]].to_csv(
            freeze_root / f"frozen_{role}_recordings.csv", index=False
        )

    issue_dispositions = pd.concat(dispositions, ignore_index=True)
    issue_dispositions.to_csv(freeze_root / "metadata_issue_dispositions.csv", index=False)
    diagnosis_provenance = (
        pd.concat(
            [
                ledger[
                    [
                        "SubjectID",
                        "diagnosis_reported",
                        "diagnosis_analysis",
                        "diagnosis_resolution",
                        "diagnosis_evidence",
                        "target_cohort_eligible",
                    ]
                ]
                for ledger in ledgers.values()
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values("SubjectID")
    )
    diagnosis_provenance.to_csv(freeze_root / "diagnosis_provenance.csv", index=False)

    pair_inputs = {}
    for role, ledger in ledgers.items():
        included = ledger.loc[ledger["freeze_included"] & ledger["date_analysis_eligible"]].copy()
        included["Recording date"] = pd.to_datetime(included["recording_date_analysis"])
        pair_inputs[role] = included
    pairs = exact_session_pairs(pair_inputs["bamboo"], pair_inputs["rest"])
    pairs.to_csv(freeze_root / "frozen_exact_bamboo_rest_pairs.csv", index=False)

    summary_rows = []
    for role, ledger in ledgers.items():
        included = ledger.loc[ledger["freeze_included"]]
        for diagnosis, group in included.groupby("diagnosis_analysis", dropna=False):
            summary_rows.append(
                {
                    "dataset_role": role,
                    "diagnosis_analysis": diagnosis,
                    "participants": group["SubjectID"].nunique(),
                    "logical_recordings": group["logical_recording_id"].nunique(),
                }
            )
        summary_rows.append(
            {
                "dataset_role": role,
                "diagnosis_analysis": "ALL_INCLUDED",
                "participants": included["SubjectID"].nunique(),
                "logical_recordings": included["logical_recording_id"].nunique(),
            }
        )
        summary_rows.append(
            {
                "dataset_role": role,
                "diagnosis_analysis": "EXCLUDED",
                "participants": ledger.loc[~ledger["freeze_included"], "SubjectID"].nunique(),
                "logical_recordings": ledger.loc[
                    ~ledger["freeze_included"], "logical_recording_id"
                ].nunique(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(freeze_root / "freeze_summary.csv", index=False)

    input_paths = [
        audit_root / f"{role}_{stem}.csv"
        for role in ["bamboo", "rest"]
        for stem in [
            "canonical_recordings",
            "media_rows_audited",
            "audit_issues",
            "media_inventory",
        ]
    ]
    input_paths.append(settings["adjudication_path"])
    write_run_manifest(
        manifest_path,
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=input_paths,
        extra={
            "freeze_version": cfg.get("data_freeze", {}).get("version", "v1"),
            "frozen_bamboo_recordings": int(ledgers["bamboo"]["freeze_included"].sum()),
            "frozen_rest_recordings": int(ledgers["rest"]["freeze_included"].sum()),
            "exact_bamboo_rest_pairs": int(len(pairs)),
        },
    )
    print(f"Frozen dataset created: {freeze_root}")
    print(summary.to_string(index=False))


def _read_existing(root: Path, stem: str) -> pd.DataFrame:
    parquet = root / f"{stem}.parquet"
    csv = root / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Expected prior-stage output missing: {parquet} or {csv}")


def command_segment(cfg: dict) -> None:
    root = _output_root(cfg)
    output = root / "01_segmentation"
    tables = output / "tables"
    legacy_segments = output / "segmentation" / "silero" / "segments"
    legacy_frames = output / "segmentation" / "silero" / "frames"
    legacy_summary = output / "segmentation" / "silero" / "summary"
    boundary_tables = output / "segmentation" / "silero" / "boundary_audit"
    legacy_figures = output / "figures" / "segmentation" / "silero"
    boundary_figures = legacy_figures / "boundary_audit"
    legacy_logs = output / "logs"
    for directory in [
        legacy_segments,
        legacy_frames,
        legacy_summary,
        boundary_tables,
        boundary_figures,
        legacy_logs,
        *(legacy_figures / status for status in ["accepted", "flagged", "excluded"]),
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # A rerun is a clean regeneration of this derived stage. Remove only the exact
    # artifact filename patterns owned by the Silero command so stale cohort files or
    # stale accepted/flagged/excluded copies cannot survive.
    for directory, pattern in [
        (legacy_segments, "*_segments.csv"),
        (legacy_frames, "*_frames.csv"),
        (legacy_figures / "accepted", "*_silero.png"),
        (legacy_figures / "flagged", "*_silero.png"),
        (legacy_figures / "excluded", "*_silero.png"),
        (boundary_tables, "*_boundary_audit.csv"),
        (boundary_figures, "*_boundary_audit.png"),
    ]:
        for path in directory.glob(pattern):
            path.unlink()

    old_artifact_roots = [
        root / "segmentation" / "silero",
        root / "figures" / "segmentation" / "silero",
    ]
    if any(path.exists() for path in old_artifact_roots):
        print(
            "NOTICE: pre-v0.8 segmentation artifacts remain outside outputs/01_segmentation. "
            "They are not read by this run and are not deleted automatically."
        )

    canonical = _read_frozen(cfg, "frozen_bamboo_recordings")
    paths_by_name = (
        canonical.set_index("Raw Media File name")["media_path"]
        .map(lambda value: [value])
        .to_dict()
    )
    ffmpeg = resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg")
    ffprobe = resolve_executable(cfg["software"]["ffprobe"], "ffprobe")
    vad = cfg["vad"]
    # The four-panel appearance and 30-ms per-file CSV layout remain compatible with
    # the original pipeline. Analysis timestamps are deliberately unpadded and retain
    # sample-index precision; padding is a display/convenience expansion, not boundary
    # evidence. Existing pre-v0.8 local configs migrate safely through these defaults.
    speech_pad_ms = int(vad.get("analysis_speech_pad_ms", 0))
    diagnostic_frame_ms = int(vad.get("original_pipeline_frame_ms", 30))
    post_vad_bridge_gap_ms = float(vad.get("post_vad_bridge_gap_ms", 0))
    if speech_pad_ms != 0 or post_vad_bridge_gap_ms != 0:
        raise ValueError(
            "Boundary-sensitive primary analysis requires "
            "vad.analysis_speech_pad_ms=0 and vad.post_vad_bridge_gap_ms=0. "
            "Use sensitivity profiles rather than padding or a second bridge pass."
        )
    if diagnostic_frame_ms != 30:
        raise ValueError(
            "Original-style per-recording artifacts require vad.original_pipeline_frame_ms=30."
        )
    boundary_window_ms = float(vad.get("boundary_audit_window_ms", 120))
    boundary_guard_ms = float(vad.get("boundary_audit_guard_ms", 20))
    boundary_contrast_db = float(vad.get("boundary_audit_minimum_contrast_db", 3))
    sensitivity_profile_defaults = {
        "conservative": {
            "threshold": 0.65,
            "min_speech_ms": 250,
            "min_silence_ms": 100,
            "strict_speech_edge_ms": 100,
            "strict_nonspeech_edge_ms": 300,
        },
        "permissive": {
            "threshold": 0.35,
            "min_speech_ms": 100,
            "min_silence_ms": 200,
            "strict_speech_edge_ms": 25,
            "strict_nonspeech_edge_ms": 100,
        },
    }
    resolved_sensitivity_profiles = {}
    for profile, overrides in vad.get("sensitivity_profiles", {}).items():
        resolved = dict(sensitivity_profile_defaults.get(profile, {}))
        resolved.update(
            {
                key: value
                for key, value in overrides.items()
                if key
                in {
                    "threshold",
                    "min_speech_ms",
                    "min_silence_ms",
                    "strict_speech_edge_ms",
                    "strict_nonspeech_edge_ms",
                }
            }
        )
        resolved_sensitivity_profiles[profile] = resolved
    onnx = cfg["software"]["silero_backend"].lower() == "onnx"
    silero_model = load_silero_model(onnx=onnx)
    (legacy_logs / "silero_segmentation_config.json").write_text(
        json.dumps(
            {
                "target_sr": 16000,
                "silero_threshold": float(vad["threshold"]),
                "min_speech_duration_ms": int(vad["min_speech_ms"]),
                "min_silence_duration_ms": int(vad["min_silence_ms"]),
                "speech_pad_ms": speech_pad_ms,
                "post_vad_bridge_gap_ms": post_vad_bridge_gap_ms,
                "analysis_timestamp_resolution_samples": 1,
                "frame_ms": diagnostic_frame_ms,
                "frame_representation_role": "original-style visualization and CSV compatibility",
                "boundary_audit_window_ms": boundary_window_ms,
                "boundary_audit_guard_ms": boundary_guard_ms,
                "boundary_audit_minimum_contrast_db": boundary_contrast_db,
                "sensitivity_profiles": resolved_sensitivity_profiles,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    errors = []
    summaries = []
    for _, row in tqdm(canonical.iterrows(), total=len(canonical), desc="Segment Bamboo"):
        file_name = row["Raw Media File name"]
        logical_id = row["logical_recording_id"]
        stem = Path(str(file_name)).stem
        segments_path = legacy_segments / f"{stem}_segments.csv"
        frames_path = legacy_frames / f"{stem}_frames.csv"
        boundary_table_path = boundary_tables / f"{stem}_boundary_audit.csv"
        boundary_plot_path = boundary_figures / f"{stem}_boundary_audit.png"
        candidates = paths_by_name.get(file_name, [])
        if len(candidates) != 1:
            error = f"expected 1 path; found {len(candidates)}"
            errors.append({"file_name": file_name, "stage": "resolve_path", "error": error})
            pd.DataFrame(columns=LEGACY_SILERO_SEGMENT_COLUMNS).to_csv(segments_path, index=False)
            pd.DataFrame(columns=LEGACY_SILERO_FRAME_COLUMNS).to_csv(frames_path, index=False)
            pd.DataFrame().to_csv(boundary_table_path, index=False)
            failure = {
                "method": "silero_vad",
                "file_name": file_name,
                "file_path": row.get("media_path", pd.NA),
                "ID_norm": row["SubjectID"],
                "Diagnosis": row["diagnosis_analysis"],
                "severity_bin": row.get("severity_bin", pd.NA),
                "Recording date": row.get("Recording date", pd.NA),
                "logical_recording_id": logical_id,
                "SubjectID": row["SubjectID"],
                "diagnosis_analysis": row["diagnosis_analysis"],
                "Task Completed as Instructed": row.get("Task Completed as Instructed", pd.NA),
                "qc_status": "excluded",
                "qc_flags": "processing_error",
                "processing_error": error,
                "segments_path": str(segments_path),
                "frames_path": str(frames_path),
                "boundary_audit_path": str(boundary_table_path),
                "boundary_plot_path": "",
            }
            failure_path = legacy_figures / "excluded" / f"{stem}_silero.png"
            plot_segmentation_failure(
                file_name=file_name,
                error=error,
                save_path=failure_path,
            )
            failure["plot_path"] = str(failure_path)
            summaries.append(failure)
            continue
        try:
            audio = decode_audio_views(candidates[0], ffmpeg=ffmpeg, ffprobe=ffprobe)
            duration = len(audio.analysis_16k) / 16000
            timestamps = silero_speech_timestamps(
                audio.analysis_16k,
                threshold=vad["threshold"],
                min_speech_ms=vad["min_speech_ms"],
                min_silence_ms=vad["min_silence_ms"],
                speech_pad_ms=speech_pad_ms,
                onnx=onnx,
                model=silero_model,
            )
            raw = [Interval(item["start"] / 16000, item["end"] / 16000) for item in timestamps]
            views = build_segmentation_views(
                raw,
                duration_sec=duration,
                bridge_gap_ms=post_vad_bridge_gap_ms,
                # Silero already applies min_speech_duration_ms. A second filter
                # would silently delete valid short regions twice.
                min_speech_ms=0,
                strict_speech_edge_ms=vad["strict_speech_edge_ms"],
                strict_nonspeech_edge_ms=vad["strict_nonspeech_edge_ms"],
            )
            table = intervals_to_frame(views, file_name)
            table["profile"] = "primary"
            table["logical_recording_id"] = logical_id
            rows.append(table)

            frame_diagnostics, legacy_segment_table = legacy_silero_artifacts(
                audio.analysis_16k,
                16000,
                timestamps,
                threshold=vad["threshold"],
                frame_ms=diagnostic_frame_ms,
            )
            legacy_segment_table.to_csv(segments_path, index=False)
            frame_diagnostics.to_csv(frames_path, index=False)
            boundary_audit = boundary_alignment_diagnostics(
                audio.analysis_16k,
                16000,
                views["primary_speech"],
                displayed_segments=legacy_segment_table,
                window_ms=boundary_window_ms,
                guard_ms=boundary_guard_ms,
                minimum_contrast_db=boundary_contrast_db,
            )
            boundary_audit.to_csv(boundary_table_path, index=False)
            plot_boundary_alignment_audit(
                audio.analysis_16k,
                16000,
                views["primary_speech"],
                boundary_audit,
                file_name=f"{file_name} | boundary alignment audit",
                save_path=boundary_plot_path,
                minimum_contrast_db=boundary_contrast_db,
            )
            summary = summarize_legacy_silero_artifacts(
                audio.analysis_16k,
                16000,
                frame_diagnostics,
                legacy_segment_table,
                threshold=vad["threshold"],
                frame_ms=diagnostic_frame_ms,
                min_speech_ms=vad["min_speech_ms"],
                min_silence_ms=vad["min_silence_ms"],
                speech_pad_ms=speech_pad_ms,
            )
            boundary_edges = 2 * len(boundary_audit)
            low_contrast_edges = int(
                boundary_audit["ambiguous_onset"].sum() + boundary_audit["ambiguous_offset"].sum()
            )
            summary.update(
                {
                    "analysis_boundary_semantics": "unpadded_silero_sample_indices",
                    "analysis_timestamp_resolution_samples": 1,
                    "post_vad_bridge_gap_ms": post_vad_bridge_gap_ms,
                    "boundary_edges": boundary_edges,
                    "boundary_low_contrast_edges": low_contrast_edges,
                    "boundary_low_contrast_fraction": (
                        low_contrast_edges / boundary_edges if boundary_edges > 0 else np.nan
                    ),
                    "boundary_min_contrast_db": (
                        float(
                            pd.concat(
                                [
                                    boundary_audit["onset_contrast_db"],
                                    boundary_audit["offset_contrast_db"],
                                ],
                                ignore_index=True,
                            ).min()
                        )
                        if not boundary_audit.empty
                        else np.nan
                    ),
                }
            )
            summary.update(classify_reading_segmentation(summary))
            plot_path = legacy_figures / summary["qc_status"] / f"{stem}_silero.png"
            summary.update(
                {
                    "file_name": file_name,
                    "file_path": candidates[0],
                    "ID_norm": row["SubjectID"],
                    "Diagnosis": row["diagnosis_analysis"],
                    "severity_bin": row.get("severity_bin", pd.NA),
                    "Recording date": row.get("Recording date", pd.NA),
                    "logical_recording_id": logical_id,
                    "SubjectID": row["SubjectID"],
                    "diagnosis_analysis": row["diagnosis_analysis"],
                    "Task Completed as Instructed": row.get("Task Completed as Instructed", pd.NA),
                    "processing_error": "",
                    "segments_path": str(segments_path),
                    "frames_path": str(frames_path),
                    "plot_path": str(plot_path),
                    "boundary_audit_path": str(boundary_table_path),
                    "boundary_plot_path": str(boundary_plot_path),
                }
            )
            plot_segmentation_diagnostic(
                audio.analysis_16k,
                16000,
                frame_diagnostics,
                legacy_segment_table,
                summary,
                file_name=f"{file_name} | silero_vad",
                save_path=plot_path,
            )
            summaries.append(summary)
            for profile, overrides in resolved_sensitivity_profiles.items():
                sensitivity_timestamps = silero_speech_timestamps(
                    audio.analysis_16k,
                    threshold=overrides.get("threshold", vad["threshold"]),
                    min_speech_ms=overrides.get("min_speech_ms", vad["min_speech_ms"]),
                    min_silence_ms=overrides.get("min_silence_ms", vad["min_silence_ms"]),
                    speech_pad_ms=0,
                    onnx=onnx,
                    model=silero_model,
                )
                sensitivity_raw = [
                    Interval(item["start"] / 16000, item["end"] / 16000)
                    for item in sensitivity_timestamps
                ]
                sensitivity_views = build_segmentation_views(
                    sensitivity_raw,
                    duration_sec=duration,
                    bridge_gap_ms=0,
                    min_speech_ms=0,
                    strict_speech_edge_ms=overrides.get(
                        "strict_speech_edge_ms", vad["strict_speech_edge_ms"]
                    ),
                    strict_nonspeech_edge_ms=overrides.get(
                        "strict_nonspeech_edge_ms", vad["strict_nonspeech_edge_ms"]
                    ),
                )
                sensitivity_table = intervals_to_frame(sensitivity_views, file_name)
                sensitivity_table["profile"] = profile
                sensitivity_table["logical_recording_id"] = logical_id
                rows.append(sensitivity_table)
        except Exception as exc:  # error ledger is part of the stage contract
            error = repr(exc)
            errors.append({"file_name": file_name, "stage": "segmentation", "error": error})
            pd.DataFrame(columns=LEGACY_SILERO_SEGMENT_COLUMNS).to_csv(segments_path, index=False)
            pd.DataFrame(columns=LEGACY_SILERO_FRAME_COLUMNS).to_csv(frames_path, index=False)
            pd.DataFrame().to_csv(boundary_table_path, index=False)
            failure = {
                "method": "silero_vad",
                "file_name": file_name,
                "file_path": candidates[0],
                "ID_norm": row["SubjectID"],
                "Diagnosis": row["diagnosis_analysis"],
                "severity_bin": row.get("severity_bin", pd.NA),
                "Recording date": row.get("Recording date", pd.NA),
                "logical_recording_id": logical_id,
                "SubjectID": row["SubjectID"],
                "diagnosis_analysis": row["diagnosis_analysis"],
                "Task Completed as Instructed": row.get("Task Completed as Instructed", pd.NA),
                "qc_status": "excluded",
                "qc_flags": "processing_error",
                "processing_error": error,
                "segments_path": str(segments_path),
                "frames_path": str(frames_path),
                "boundary_audit_path": str(boundary_table_path),
                "boundary_plot_path": "",
            }
            failure_path = legacy_figures / "excluded" / f"{stem}_silero.png"
            plot_segmentation_failure(
                file_name=file_name,
                error=error,
                save_path=failure_path,
            )
            failure["plot_path"] = str(failure_path)
            summaries.append(failure)
    intervals = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_frame = pd.DataFrame(summaries)
    error_frame = pd.DataFrame(errors, columns=["file_name", "stage", "error"])
    qc_counts = (
        summary_frame["qc_status"]
        .value_counts(dropna=False)
        .rename_axis("qc_status")
        .reset_index(name="logical_recordings")
    )
    qc_counts["percent"] = 100 * qc_counts["logical_recordings"] / max(1, len(summary_frame))
    flag_counts = (
        summary_frame["qc_flags"]
        .fillna("")
        .str.split(";")
        .explode()
        .loc[lambda values: values.ne("")]
        .value_counts()
        .rename_axis("qc_flag")
        .reset_index(name="logical_recordings")
    )
    legacy_summary_frame = summary_frame.reindex(columns=LEGACY_SILERO_SUMMARY_COLUMNS)
    legacy_summary_frame.to_csv(legacy_summary / "silero_all_summary.csv", index=False)
    error_frame.to_csv(legacy_summary / "silero_all_errors.csv", index=False)
    legacy_qc_counts = (
        summary_frame["qc_status"]
        .value_counts(dropna=False)
        .rename_axis("qc_status")
        .reset_index(name="n")
    )
    legacy_qc_counts["percent"] = 100 * legacy_qc_counts["n"] / max(1, len(summary_frame))
    legacy_qc_counts.to_csv(legacy_summary / "silero_qc_status_counts.csv", index=False)
    legacy_top_flags = (
        summary_frame["qc_flags"]
        .fillna("")
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(20)
        .rename_axis("qc_flags")
        .reset_index(name="n")
    )
    legacy_top_flags.to_csv(legacy_summary / "silero_top_qc_flags.csv", index=False)
    for destination in [output, tables]:
        _write_table(intervals, destination / "bamboo_segmentation_intervals")
        _write_table(summary_frame, destination / "bamboo_segmentation_summary")
        _write_table(error_frame, destination / "segmentation_errors")
        _write_table(qc_counts, destination / "segmentation_qc_status_counts")
        _write_table(flag_counts, destination / "segmentation_qc_flag_counts")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=canonical["media_path"].tolist(),
        extra={
            "silero_version_expected": cfg["software"]["silero_version"],
            "analysis_boundary_semantics": "unpadded_silero_sample_indices",
            "analysis_speech_pad_ms": speech_pad_ms,
            "post_vad_bridge_gap_ms": post_vad_bridge_gap_ms,
            "diagnostic_frame_ms": diagnostic_frame_ms,
            "boundary_audit_window_ms": boundary_window_ms,
            "boundary_audit_guard_ms": boundary_guard_ms,
            "boundary_audit_minimum_contrast_db": boundary_contrast_db,
            "resolved_sensitivity_profiles": resolved_sensitivity_profiles,
        },
    )


def _segmentation_adjudication_path(cfg: dict) -> Path:
    value = cfg.get("data_freeze", {}).get(
        "segmentation_adjudication", "config/segmentation_adjudication.csv"
    )
    return resolve_project_path(cfg, value)


def _manual_segmentation_overrides_path(cfg: dict) -> Path:
    value = cfg.get("data_freeze", {}).get(
        "manual_segmentation_overrides",
        "config/manual_segmentation_overrides.csv",
    )
    return resolve_project_path(cfg, value)


def _attach_frozen_task_completion(
    cfg: dict,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the frozen task-validity field to pre-v0.9 segmentation summaries."""
    column = "Task Completed as Instructed"
    if column in summary.columns:
        return summary
    frozen_bamboo = _read_frozen(cfg, "frozen_bamboo_recordings")
    required = {"logical_recording_id", column}
    missing = required - set(frozen_bamboo.columns)
    if missing:
        raise ValueError(
            "Frozen Bamboo data cannot enforce the task-completion gate because "
            f"these columns are missing: {sorted(missing)}. Recreate the metadata "
            "freeze before segmentation adjudication."
        )
    lookup = frozen_bamboo[["logical_recording_id", column]].copy()
    lookup["logical_recording_id"] = lookup["logical_recording_id"].astype(str)
    if lookup["logical_recording_id"].duplicated().any():
        raise ValueError(
            "Frozen Bamboo task-completion lookup is not one row per logical recording."
        )
    work = summary.copy()
    work["logical_recording_id"] = work["logical_recording_id"].astype(str)
    return work.merge(
        lookup,
        on="logical_recording_id",
        how="left",
        validate="one_to_one",
    )


def command_segment_template(cfg: dict) -> None:
    output = _output_root(cfg) / "01_segmentation"
    tables = output / "tables"
    summary = _attach_frozen_task_completion(
        cfg,
        _read_existing(output, "bamboo_segmentation_summary"),
    )
    template = segmentation_adjudication_template(summary, cfg.get("segmentation_review", {}))
    destination = _segmentation_adjudication_path(cfg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = pd.read_csv(destination, dtype=str, keep_default_na=False)
        backup = destination.with_name(f"{destination.stem}.pre_v070.csv")
        if not backup.exists():
            backup.write_bytes(destination.read_bytes())
            print(f"Backed up prior review sheet: {backup}")
        existing_by_id = (
            existing.assign(logical_recording_id=existing["logical_recording_id"].astype(str))
            .drop_duplicates("logical_recording_id", keep="last")
            .set_index("logical_recording_id")
        )
        for index, row in template.iterrows():
            logical_id = str(row["logical_recording_id"])
            if logical_id not in existing_by_id.index:
                continue
            if bool(row["automatic_task_exclusion"]):
                # Frozen-metadata task exclusions are regenerated from source and
                # cannot be overridden by a prior human decision.
                continue
            old = existing_by_id.loc[logical_id]
            decision = str(old.get("decision", "")).strip().upper()
            reviewer = str(old.get("reviewer", "")).strip()
            source = str(old.get("boundary_source", "")).strip().upper()
            if not source:
                source = "AUTO" if decision == "KEEP" else ("NONE" if decision == "EXCLUDE" else "")
            preserve = decision in {"KEEP", "EXCLUDE"}
            if bool(row["review_required"]) and row["automatic_qc_status"] == "accepted":
                preserve = preserve and bool(reviewer)
            if not bool(row["review_required"]):
                preserve = preserve and (
                    bool(reviewer) or source == "MANUAL" or decision == "EXCLUDE"
                )
            if preserve:
                template.loc[index, "decision"] = decision
                template.loc[index, "boundary_source"] = source
                template.loc[index, "reviewer"] = reviewer
                template.loc[index, "review_date"] = str(old.get("review_date", "")).strip()
                template.loc[index, "notes"] = str(old.get("notes", "")).strip()
        print(f"Upgraded/refreshed segmentation review sheet: {destination}")
    else:
        print(f"Wrote segmentation review sheet: {destination}")
    template.to_csv(destination, index=False)

    override_path = _manual_segmentation_overrides_path(cfg)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    if not override_path.exists():
        pd.DataFrame(columns=MANUAL_SEGMENTATION_COLUMNS).to_csv(override_path, index=False)
        print(f"Wrote empty manual-boundary table: {override_path}")

    review_queue = template.loc[_as_bool(template["review_required"])].copy()
    _write_table(review_queue, tables / "segmentation_review_queue")
    selection_summary = (
        template.groupby(
            [
                "automatic_qc_status",
                "automatic_task_exclusion",
                "accepted_outlier",
                "review_required",
            ],
            dropna=False,
        )
        .size()
        .rename("logical_recordings")
        .reset_index()
    )
    _write_table(selection_summary, tables / "segmentation_review_selection_summary")
    required = segmentation_pending_reviews(template)
    print(
        required[
            [
                "file_name",
                "automatic_qc_status",
                "review_reasons",
                "decision",
                "boundary_source",
                "reviewer",
            ]
        ].to_string(index=False)
    )
    print(f"Pending required reviews: {len(required)}")


def _materialize_reviewed_segmentation_output(
    *,
    destination: Path,
    published_destination: Path,
    frozen: pd.DataFrame,
    frozen_intervals: pd.DataFrame,
    manual_artifacts: pd.DataFrame,
    cfg: dict,
    input_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the post-review artifact tree without altering automatic outputs."""
    destination.mkdir(parents=True, exist_ok=False)
    silero_root = destination / "segmentation" / "silero"
    figure_root = destination / "figures" / "segmentation" / "silero"
    table_root = destination / "tables"
    log_root = destination / "logs"
    statuses = ["accepted", "flagged", "excluded"]
    for status in statuses:
        for directory in [
            silero_root / "segments" / status,
            silero_root / "frames" / status,
            silero_root / "boundary_audit" / status,
            figure_root / status,
            figure_root / "boundary_audit" / status,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    (silero_root / "summary").mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    manual_by_id = (
        manual_artifacts.assign(
            logical_recording_id=manual_artifacts["logical_recording_id"].astype(str)
        ).set_index("logical_recording_id")
        if not manual_artifacts.empty
        else pd.DataFrame()
    )
    reviewed_rows: list[dict] = []
    missing_optional: list[dict] = []

    def copy_artifact(
        source_value: object,
        target: Path,
        *,
        file_name: str,
        artifact: str,
        required: bool = True,
    ) -> str:
        source_text = "" if pd.isna(source_value) else str(source_value).strip()
        source = Path(source_text) if source_text else None
        if source is None or not source.is_file():
            if required:
                raise FileNotFoundError(
                    f"Cannot materialize reviewed {artifact} for {file_name}; "
                    f"source artifact is missing: {source_text!r}"
                )
            missing_optional.append(
                {
                    "file_name": file_name,
                    "artifact": artifact,
                    "source_path": source_text,
                }
            )
            return ""
        shutil.copy2(source, target)
        return str(published_destination / target.relative_to(destination))

    for _, row in frozen.sort_values("file_name").iterrows():
        logical_id = str(row["logical_recording_id"])
        file_name = str(row["file_name"])
        stem = Path(file_name).stem
        decision = str(row["decision"]).upper()
        boundary_source = str(row["boundary_source"]).upper()
        review_required = bool(_as_bool(pd.Series([row["review_required"]])).iloc[0])
        if decision == "EXCLUDE":
            final_status = "excluded"
            final_reason = str(row["segmentation_decision_source"])
        elif (
            boundary_source == "MANUAL"
            or str(row["automatic_qc_status"]).lower() != "accepted"
            or review_required
        ):
            final_status = "flagged"
            final_reason = (
                "kept_after_required_review" if review_required else "kept_with_manual_boundaries"
            )
        else:
            final_status = "accepted"
            final_reason = "automatic_silero_accepted"

        artifact_row = row
        if boundary_source == "MANUAL":
            if manual_artifacts.empty or logical_id not in manual_by_id.index:
                raise FileNotFoundError(
                    f"Manual artifact index is missing KEEP/MANUAL recording {file_name}."
                )
            artifact_row = manual_by_id.loc[logical_id]

        reviewed = row.to_dict()
        reviewed.update(
            {
                "final_review_status": final_status,
                "final_review_reason": final_reason,
                "analysis_included": decision == "KEEP",
                "automatic_segments_path": str(row.get("segments_path", "")),
                "automatic_frames_path": str(row.get("frames_path", "")),
                "automatic_plot_path": str(row.get("plot_path", "")),
                "automatic_boundary_audit_path": str(row.get("boundary_audit_path", "")),
                "automatic_boundary_plot_path": str(row.get("boundary_plot_path", "")),
            }
        )
        reviewed["segments_path"] = copy_artifact(
            artifact_row.get("segments_path", ""),
            silero_root / "segments" / final_status / f"{stem}_segments.csv",
            file_name=file_name,
            artifact="segments CSV",
        )
        reviewed["frames_path"] = copy_artifact(
            artifact_row.get("frames_path", ""),
            silero_root / "frames" / final_status / f"{stem}_frames.csv",
            file_name=file_name,
            artifact="frames CSV",
        )
        reviewed["plot_path"] = copy_artifact(
            artifact_row.get("plot_path", ""),
            figure_root / final_status / f"{stem}_silero.png",
            file_name=file_name,
            artifact="four-panel segmentation figure",
        )
        reviewed["boundary_audit_path"] = copy_artifact(
            artifact_row.get("boundary_audit_path", ""),
            silero_root / "boundary_audit" / final_status / f"{stem}_boundary_audit.csv",
            file_name=file_name,
            artifact="boundary audit CSV",
            required=False,
        )
        reviewed["boundary_plot_path"] = copy_artifact(
            artifact_row.get("boundary_plot_path", ""),
            figure_root / "boundary_audit" / final_status / f"{stem}_boundary_audit.png",
            file_name=file_name,
            artifact="boundary audit figure",
            required=False,
        )
        reviewed_rows.append(reviewed)

    reviewed_summary = pd.DataFrame(reviewed_rows)
    if len(reviewed_summary) != len(frozen):
        raise RuntimeError("Reviewed-output materialization changed the recording count.")
    if (
        not reviewed_summary["analysis_included"]
        .eq(reviewed_summary["final_review_status"].isin(["accepted", "flagged"]))
        .all()
    ):
        raise RuntimeError("Reviewed-output status and downstream inclusion are inconsistent.")
    status_summary = (
        reviewed_summary.groupby(
            [
                "final_review_status",
                "analysis_included",
                "automatic_qc_status",
                "boundary_source",
                "segmentation_decision_source",
            ],
            dropna=False,
        )
        .size()
        .rename("logical_recordings")
        .reset_index()
    )
    reviewed_summary.to_csv(
        silero_root / "summary" / "silero_after_review_summary.csv",
        index=False,
    )
    reviewed_summary.to_csv(
        table_root / "reviewed_segmentation_recordings.csv",
        index=False,
    )
    frozen_intervals.to_csv(
        table_root / "reviewed_segmentation_intervals.csv",
        index=False,
    )
    status_summary.to_csv(
        table_root / "reviewed_segmentation_status_counts.csv",
        index=False,
    )
    pd.DataFrame(
        missing_optional,
        columns=["file_name", "artifact", "source_path"],
    ).to_csv(table_root / "optional_artifacts_missing.csv", index=False)
    write_run_manifest(
        log_root / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=input_paths,
        extra={
            "recordings": int(len(reviewed_summary)),
            "analysis_included": int(reviewed_summary["analysis_included"].sum()),
            "accepted": int(reviewed_summary["final_review_status"].eq("accepted").sum()),
            "flagged_but_included": int(
                reviewed_summary["final_review_status"].eq("flagged").sum()
            ),
            "excluded": int(reviewed_summary["final_review_status"].eq("excluded").sum()),
            "status_contract": ("accepted and flagged proceed downstream; excluded do not"),
        },
    )
    return reviewed_summary, status_summary


def command_segment_adjudicate(cfg: dict) -> None:
    output = _output_root(cfg) / "01_segmentation"
    reviewed_output = _output_root(cfg) / "01_segmentation_after_review"
    tables = output / "tables"
    freeze_root = _segmentation_freeze_root(cfg)
    if freeze_root.exists() and any(freeze_root.iterdir()):
        raise FileExistsError(
            f"Segmentation freeze already exists and will not be overwritten: {freeze_root}. "
            "If a scientifically justified revision is required, increment "
            "segmentation_freeze.version in config/project.yaml."
        )
    if reviewed_output.exists():
        raise FileExistsError(
            "Reviewed segmentation output already exists and will not be "
            f"overwritten: {reviewed_output}. Keep it with its matching freeze, "
            "or archive/rename it before creating a scientifically revised freeze."
        )
    summary = _attach_frozen_task_completion(
        cfg,
        _read_existing(output, "bamboo_segmentation_summary"),
    )
    path = _segmentation_adjudication_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Segmentation adjudication not found: {path}. Run segment-template first."
        )
    adjudication = pd.read_csv(path, dtype=str, keep_default_na=False)
    frozen = apply_segmentation_adjudication(
        summary,
        adjudication,
        cfg.get("segmentation_review", {}),
    )
    override_path = _manual_segmentation_overrides_path(cfg)
    if override_path.exists():
        manual_overrides = pd.read_csv(override_path, dtype=str, keep_default_na=False)
    else:
        manual_overrides = pd.DataFrame(columns=MANUAL_SEGMENTATION_COLUMNS)
    validated_overrides = validate_manual_segmentation_overrides(manual_overrides, frozen)
    automatic_intervals = _read_existing(output, "bamboo_segmentation_intervals")
    frozen_intervals = freeze_segmentation_intervals(
        automatic_intervals,
        frozen,
        validated_overrides,
        strict_speech_edge_ms=cfg["vad"]["strict_speech_edge_ms"],
        strict_nonspeech_edge_ms=cfg["vad"]["strict_nonspeech_edge_ms"],
    )

    manual_segments_dir = output / "segmentation" / "manual_review" / "segments"
    manual_frames_dir = output / "segmentation" / "manual_review" / "frames"
    manual_boundary_dir = output / "segmentation" / "manual_review" / "boundary_audit"
    manual_figures_dir = output / "figures" / "manual_review"
    manual_boundary_figures_dir = manual_figures_dir / "boundary_audit"
    for directory, pattern in [
        (manual_segments_dir, "*_manual_segments.csv"),
        (manual_frames_dir, "*_manual_frames.csv"),
        (manual_boundary_dir, "*_manual_boundary_audit.csv"),
        (manual_figures_dir, "*_manual.png"),
        (manual_boundary_figures_dir, "*_manual_boundary_audit.png"),
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        for generated_path in directory.glob(pattern):
            generated_path.unlink()

    ffmpeg = resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg")
    ffprobe = resolve_executable(cfg["software"]["ffprobe"], "ffprobe")
    manual_artifact_rows = []
    manual_decisions = frozen.loc[
        frozen["decision"].eq("KEEP") & frozen["boundary_source"].eq("MANUAL")
    ]
    for decision in manual_decisions.itertuples():
        override_rows = validated_overrides.loc[
            validated_overrides["logical_recording_id"].eq(str(decision.logical_recording_id))
        ].sort_values("segment_index")
        timestamps = [
            {
                "start": int(round(float(row.start_sec) * 16000)),
                "end": int(round(float(row.end_sec) * 16000)),
            }
            for row in override_rows.itertuples()
        ]
        audio = decode_audio_views(str(decision.file_path), ffmpeg=ffmpeg, ffprobe=ffprobe)
        frames, segments = legacy_silero_artifacts(
            audio.analysis_16k,
            16000,
            timestamps,
            threshold=cfg["vad"]["threshold"],
            frame_ms=int(cfg["vad"].get("original_pipeline_frame_ms", 30)),
        )
        stem = Path(str(decision.file_name)).stem
        segments_path = manual_segments_dir / f"{stem}_manual_segments.csv"
        frames_path = manual_frames_dir / f"{stem}_manual_frames.csv"
        plot_path = manual_figures_dir / f"{stem}_manual.png"
        boundary_path = manual_boundary_dir / f"{stem}_manual_boundary_audit.csv"
        boundary_plot_path = manual_boundary_figures_dir / f"{stem}_manual_boundary_audit.png"
        segments.to_csv(segments_path, index=False)
        frames.to_csv(frames_path, index=False)
        manual_summary = summarize_legacy_silero_artifacts(
            audio.analysis_16k,
            16000,
            frames,
            segments,
            threshold=cfg["vad"]["threshold"],
            frame_ms=int(cfg["vad"].get("original_pipeline_frame_ms", 30)),
            min_speech_ms=cfg["vad"]["min_speech_ms"],
            min_silence_ms=cfg["vad"]["min_silence_ms"],
            speech_pad_ms=0,
        )
        plot_segmentation_diagnostic(
            audio.analysis_16k,
            16000,
            frames,
            segments,
            manual_summary,
            file_name=f"{decision.file_name} | manual segmentation",
            save_path=plot_path,
        )
        manual_primary = [
            Interval(float(row.start_sec), float(row.end_sec)) for row in override_rows.itertuples()
        ]
        boundary_audit = boundary_alignment_diagnostics(
            audio.analysis_16k,
            16000,
            manual_primary,
            displayed_segments=segments,
            window_ms=float(cfg["vad"].get("boundary_audit_window_ms", 120)),
            guard_ms=float(cfg["vad"].get("boundary_audit_guard_ms", 20)),
            minimum_contrast_db=float(cfg["vad"].get("boundary_audit_minimum_contrast_db", 3)),
        )
        boundary_audit.to_csv(boundary_path, index=False)
        plot_boundary_alignment_audit(
            audio.analysis_16k,
            16000,
            manual_primary,
            boundary_audit,
            file_name=f"{decision.file_name} | manual boundary audit",
            save_path=boundary_plot_path,
            minimum_contrast_db=float(cfg["vad"].get("boundary_audit_minimum_contrast_db", 3)),
        )
        manual_artifact_rows.append(
            {
                "logical_recording_id": decision.logical_recording_id,
                "file_name": decision.file_name,
                "reviewer": decision.reviewer,
                "review_date": decision.review_date,
                "notes": decision.notes,
                "manual_speech_segments": len(override_rows),
                "segments_path": str(segments_path),
                "frames_path": str(frames_path),
                "plot_path": str(plot_path),
                "boundary_audit_path": str(boundary_path),
                "boundary_plot_path": str(boundary_plot_path),
            }
        )

    _write_table(frozen, tables / "frozen_segmentation_decisions")
    _write_table(frozen_intervals, tables / "frozen_segmentation_intervals")
    _write_table(validated_overrides, tables / "frozen_manual_segmentation_overrides")
    manual_artifact_frame = pd.DataFrame(
        manual_artifact_rows,
        columns=[
            "logical_recording_id",
            "file_name",
            "reviewer",
            "review_date",
            "notes",
            "manual_speech_segments",
            "segments_path",
            "frames_path",
            "plot_path",
            "boundary_audit_path",
            "boundary_plot_path",
        ],
    )
    _write_table(
        manual_artifact_frame,
        tables / "manual_segmentation_artifact_index",
    )
    decision_summary = (
        frozen.groupby(
            [
                "automatic_qc_status",
                "automatic_task_exclusion",
                "review_required",
                "decision",
                "boundary_source",
                "segmentation_decision_source",
            ],
            dropna=False,
        )
        .size()
        .rename("logical_recordings")
        .reset_index()
    )
    _write_table(decision_summary, tables / "segmentation_decision_summary")
    reviewed_input_paths = [
        output / "bamboo_segmentation_summary.csv",
        output / "bamboo_segmentation_intervals.csv",
        path,
        override_path,
    ]
    reviewed_output.parent.mkdir(parents=True, exist_ok=True)
    freeze_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".01_segmentation_after_review.staging-",
        dir=reviewed_output.parent,
    ) as reviewed_staging_directory:
        reviewed_staging = Path(reviewed_staging_directory) / "payload"
        reviewed_summary, reviewed_status_summary = _materialize_reviewed_segmentation_output(
            destination=reviewed_staging,
            published_destination=reviewed_output,
            frozen=frozen,
            frozen_intervals=frozen_intervals,
            manual_artifacts=manual_artifact_frame,
            cfg=cfg,
            input_paths=reviewed_input_paths,
        )
        with tempfile.TemporaryDirectory(
            prefix=f".{freeze_root.name}.staging-",
            dir=freeze_root.parent,
        ) as staging_directory:
            staging = Path(staging_directory)
            frozen.to_csv(staging / "frozen_segmentation_decisions.csv", index=False)
            frozen_intervals.to_csv(staging / "frozen_segmentation_intervals.csv", index=False)
            validated_overrides.to_csv(
                staging / "frozen_manual_segmentation_overrides.csv", index=False
            )
            decision_summary.to_csv(staging / "segmentation_decision_summary.csv", index=False)
            reviewed_status_summary.to_csv(
                staging / "reviewed_segmentation_status_counts.csv",
                index=False,
            )
            write_run_manifest(
                staging / "segmentation_freeze_manifest.json",
                command=sys.argv,
                config_path=cfg["_config_path"],
                input_paths=reviewed_input_paths,
                extra={
                    "segmentation_freeze_version": freeze_root.name,
                    "frozen_recordings": int(len(frozen)),
                    "frozen_interval_rows": int(len(frozen_intervals)),
                    "manual_override_recordings": int(len(manual_decisions)),
                    "reviewed_output": str(reviewed_output),
                    "accepted_recordings": int(
                        reviewed_summary["final_review_status"].eq("accepted").sum()
                    ),
                    "flagged_included_recordings": int(
                        reviewed_summary["final_review_status"].eq("flagged").sum()
                    ),
                    "excluded_recordings": int(
                        reviewed_summary["final_review_status"].eq("excluded").sum()
                    ),
                },
            )
            os.replace(staging, freeze_root)
        os.replace(reviewed_staging, reviewed_output)
    write_run_manifest(
        output / "segmentation_decision_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=[
            output / "bamboo_segmentation_summary.csv",
            output / "bamboo_segmentation_intervals.csv",
            path,
            override_path,
        ],
        extra={
            "frozen_primary_boundary_source_counts": frozen["boundary_source"]
            .value_counts(dropna=False)
            .to_dict(),
            "manual_override_recordings": int(len(manual_decisions)),
        },
    )
    print(decision_summary.to_string(index=False))
    print("\nPost-review status (accepted and flagged proceed; excluded do not):")
    print(reviewed_status_summary.to_string(index=False))
    print(f"\nReviewed segmentation artifacts: {reviewed_output}")


def _views_for_file(
    intervals: pd.DataFrame, file_name: str, *, profile: str = "primary"
) -> dict[str, list[Interval]]:
    subset = intervals.loc[intervals["file_name"] == file_name]
    if "profile" in subset.columns:
        subset = subset.loc[subset["profile"] == profile]
    return {
        view: [Interval(float(row.start_sec), float(row.end_sec)) for row in view_rows.itertuples()]
        for view, view_rows in subset.groupby("view")
    }


def command_extract(cfg: dict, *, profile: str = "primary") -> None:
    root = _output_root(cfg)
    output = root / "02_features"
    canonical = _read_frozen(cfg, "frozen_bamboo_recordings")
    segmentation_decisions = _read_existing(
        _segmentation_freeze_root(cfg), "frozen_segmentation_decisions"
    )
    segmentation_eligible = _as_bool(segmentation_decisions["segmentation_analysis_eligible"])
    eligible_ids = set(
        segmentation_decisions.loc[
            segmentation_eligible,
            "logical_recording_id",
        ].astype(str)
    )
    canonical = canonical.loc[
        canonical["logical_recording_id"].astype(str).isin(eligible_ids)
    ].copy()
    intervals = _read_existing(
        _segmentation_freeze_root(cfg),
        "frozen_segmentation_intervals",
    )
    paths_by_name = (
        canonical.set_index("Raw Media File name")["media_path"]
        .map(lambda value: [value])
        .to_dict()
    )
    ffmpeg = resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg")
    ffprobe = resolve_executable(cfg["software"]["ffprobe"], "ffprobe")
    results = []
    errors = []
    for _, row in tqdm(canonical.iterrows(), total=len(canonical), desc="Extract Q metrics"):
        file_name = row["Raw Media File name"]
        candidates = paths_by_name.get(file_name, [])
        if len(candidates) != 1:
            errors.append(
                {
                    "file_name": file_name,
                    "stage": "resolve_path",
                    "error": f"expected 1 path; found {len(candidates)}",
                }
            )
            continue
        try:
            views = _views_for_file(intervals, file_name, profile=profile)
            required = {
                "raw_speech",
                "primary_speech",
                "strict_speech",
                "strict_internal_nonspeech",
            }
            for missing in required - set(views):
                views[missing] = []
            audio = decode_audio_views(candidates[0], ffmpeg=ffmpeg, ffprobe=ffprobe)
            metrics = extract_all_metrics(audio, views)
            results.append(
                {
                    "file_name": file_name,
                    "logical_recording_id": row["logical_recording_id"],
                    "SubjectID": row["SubjectID"],
                    "Recording date": row["Recording date"],
                    "diagnosis_reported": row["diagnosis_reported"],
                    "diagnosis_analysis": row["diagnosis_analysis"],
                    **metrics,
                }
            )
        except Exception as exc:
            errors.append(
                {"file_name": file_name, "stage": "feature_extraction", "error": repr(exc)}
            )
    feature_frame = pd.DataFrame(results)
    suffix = "" if profile == "primary" else f"_{profile}"
    _write_table(feature_frame, output / f"bamboo_q_metrics{suffix}")
    _write_table(pd.DataFrame(errors), output / f"feature_extraction_errors{suffix}")
    _write_table(metric_registry_frame(), output / "metric_registry")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=canonical["media_path"].tolist(),
        extra={"segmentation_profile": profile},
    )


def command_assemble(cfg: dict) -> None:
    root = _output_root(cfg)
    output = root / "03_dataset_assembly"
    metadata = _read_frozen(cfg, "frozen_bamboo_recordings")
    segmentation = _read_existing(_segmentation_freeze_root(cfg), "frozen_segmentation_decisions")
    if "qc_flags" not in segmentation:
        legacy_flag_columns = [
            column for column in ("qc_flags_y", "qc_flags_x") if column in segmentation
        ]
        if not legacy_flag_columns:
            raise ValueError("Frozen segmentation decisions lack a QC-flags field")
        segmentation["qc_flags"] = segmentation[legacy_flag_columns[0]]
        for column in legacy_flag_columns[1:]:
            segmentation["qc_flags"] = segmentation["qc_flags"].fillna(segmentation[column])
    metadata = metadata.merge(
        segmentation[
            [
                "logical_recording_id",
                "qc_status",
                "qc_flags",
                "accepted_outlier",
                "review_required",
                "review_reasons",
                "decision",
                "boundary_source",
                "reviewer",
                "review_date",
                "segmentation_analysis_eligible",
                "segmentation_decision_source",
            ]
        ],
        on="logical_recording_id",
        how="left",
        validate="one_to_one",
    )
    release = build_latest_feature_release(cfg["_project_root"])
    features, reviewed_registry = load_latest_feature_release(cfg["_project_root"])
    reviewed_names = set(reviewed_registry["feature"].astype(str))
    legacy_names = set(metric_registry_frame()["feature"].astype(str)) - reviewed_names
    leaked = sorted(legacy_names.intersection(features.columns))
    if leaked:
        raise ValueError(f"Legacy feature columns entered the reviewed release: {leaked}")
    if "SubjectID" in features:
        subject_check = metadata[["logical_recording_id", "SubjectID"]].merge(
            features[["logical_recording_id", "SubjectID"]],
            on="logical_recording_id",
            how="inner",
            suffixes=("_metadata", "_features"),
            validate="one_to_one",
        )
        mismatch = (
            subject_check["SubjectID_metadata"]
            .astype(str)
            .ne(subject_check["SubjectID_features"].astype(str))
        )
        if mismatch.any():
            raise ValueError("Reviewed feature SubjectID values do not match frozen metadata")
    features = features.drop(
        columns=["SubjectID", "file_name", "recording_date_analysis", "Recording date"],
        errors="ignore",
    )
    merged = metadata.merge(
        features,
        on="logical_recording_id",
        how="left",
        suffixes=("_metadata", ""),
        validate="one_to_one",
        indicator=True,
    )
    task_completed = merged["Task Completed as Instructed"].astype(str).str.upper()
    merged["hard_exclusion_task_not_completed"] = task_completed.eq("NO")
    merged["review_task_completion_missing"] = merged["Task Completed as Instructed"].isna()
    merged["hard_exclusion_segmentation"] = ~_as_bool(merged["segmentation_analysis_eligible"])
    qgain_status = merged.get("qgain_within_segment_iqr_db_status")
    if qgain_status is None:
        raise ValueError("Reviewed QGAIN status field is missing from the latest release")
    merged["hard_exclusion_no_usable_speech"] = qgain_status.astype(str).ne("measured")
    merged["feature_extraction_missing"] = merged["_merge"].ne("both")
    merged["confirmed_analysis_diagnosis"] = merged["diagnosis_analysis"].isin(
        cfg["cohort"]["allowed_diagnoses"]
    )
    merged["primary_measurement_eligible"] = ~(
        merged["hard_exclusion_task_not_completed"]
        | merged["review_task_completion_missing"]
        | merged["hard_exclusion_segmentation"]
        | merged["hard_exclusion_no_usable_speech"]
        | merged["feature_extraction_missing"]
    )
    merged["diagnosis_contrast_eligible"] = (
        merged["primary_measurement_eligible"] & merged["confirmed_analysis_diagnosis"]
    )
    assessment_delta = pd.to_numeric(merged.get("assessment_recording_delta_days"), errors="coerce")
    merged["clinical_primary_eligible"] = (
        merged["primary_measurement_eligible"]
        & merged["date_analysis_eligible"].fillna(False)
        & assessment_delta.abs().le(cfg["clinical_alignment"]["primary_max_assessment_gap_days"])
        & pd.to_numeric(merged["ALSFRS total score"], errors="coerce").notna()
    )
    merged["clinical_30day_sensitivity_eligible"] = (
        merged["primary_measurement_eligible"]
        & merged["date_analysis_eligible"].fillna(False)
        & assessment_delta.abs().le(
            cfg["clinical_alignment"]["sensitivity_max_assessment_gap_days"]
        )
        & pd.to_numeric(merged["ALSFRS total score"], errors="coerce").notna()
    )
    merged = merged.drop(columns="_merge")
    _write_table(merged, output / "paper1_analysis_dataset")
    eligibility = pd.DataFrame(
        [
            {"criterion": column, "n_true": int(merged[column].fillna(False).sum())}
            for column in [
                "hard_exclusion_task_not_completed",
                "review_task_completion_missing",
                "hard_exclusion_segmentation",
                "hard_exclusion_no_usable_speech",
                "feature_extraction_missing",
                "primary_measurement_eligible",
                "confirmed_analysis_diagnosis",
                "diagnosis_contrast_eligible",
                "clinical_primary_eligible",
                "clinical_30day_sensitivity_eligible",
            ]
        ]
    )
    _write_table(eligibility, output / "eligibility_flow_counts")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=[
            _data_freeze_root(cfg) / "frozen_bamboo_recordings.csv",
            Path(release["output_root"]) / "recording_features.csv",
            Path(release["output_root"]) / "feature_registry.csv",
        ],
        extra={
            "feature_source": "reviewed_latest_release",
            "reviewed_feature_count": int(len(reviewed_registry)),
            "qtemp_validated_primary_feature_count": 0,
        },
    )


def command_reviewed_release(cfg: dict) -> None:
    result = build_latest_feature_release(cfg["_project_root"])
    print(json.dumps(result, indent=2))


def command_rest_reference(cfg: dict) -> None:
    """Extract exact-session Rest contextual metrics without speech VAD."""
    root = _output_root(cfg)
    output = root / "04_analysis" / "rest_reference"
    pairs = _read_frozen(cfg, "frozen_exact_bamboo_rest_pairs")
    rest = _read_frozen(cfg, "frozen_rest_recordings")
    target_ids = set(pairs["logical_recording_id_rest"].astype(str))
    rest = rest.loc[rest["logical_recording_id"].astype(str).isin(target_ids)].copy()
    paths_by_name = (
        rest.set_index("Raw Media File name")["media_path"].map(lambda value: [value]).to_dict()
    )
    ffmpeg = resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg")
    ffprobe = resolve_executable(cfg["software"]["ffprobe"], "ffprobe")
    rows = []
    errors = []
    for _, row in tqdm(rest.iterrows(), total=len(rest), desc="Extract Rest reference"):
        file_name = row["Raw Media File name"]
        candidates = paths_by_name.get(file_name, [])
        if len(candidates) != 1:
            errors.append(
                {
                    "file_name": file_name,
                    "stage": "resolve_path",
                    "error": f"expected 1 path; found {len(candidates)}",
                }
            )
            continue
        try:
            audio = decode_audio_views(candidates[0], ffmpeg=ffmpeg, ffprobe=ffprobe)
            metrics = rest_reference_metrics(audio.analysis_16k, 16000)
            rows.append(
                {
                    "file_name_rest": file_name,
                    "logical_recording_id_rest": row["logical_recording_id"],
                    "SubjectID": row["SubjectID"],
                    "Recording date": row["Recording date"],
                    **metrics,
                }
            )
        except Exception as exc:
            errors.append({"file_name": file_name, "stage": "rest_reference", "error": repr(exc)})
    rest_metrics = pd.DataFrame(rows)
    _write_table(rest_metrics, output / "rest_reference_metrics")
    _write_table(pd.DataFrame(errors), output / "rest_reference_errors")

    bamboo = _read_existing(root / "02_features", "bamboo_q_metrics")
    comparison = pairs.merge(
        rest_metrics,
        on=["logical_recording_id_rest", "SubjectID"],
        how="left",
        validate="one_to_one",
    ).merge(
        bamboo,
        left_on=["logical_recording_id_bamboo", "SubjectID"],
        right_on=["logical_recording_id", "SubjectID"],
        how="left",
        suffixes=("_pair", "_bamboo"),
        validate="one_to_one",
    )
    comparison["bamboo_guarded_pause_minus_rest_level_db"] = pd.to_numeric(
        comparison["qadd_nonspeech_level_dbfs"], errors="coerce"
    ) - pd.to_numeric(comparison["restref_level_dbfs"], errors="coerce")
    _write_table(comparison, output / "exact_session_bamboo_rest_comparison")
    rest_columns = [
        column
        for column in comparison.columns
        if column.startswith("restref_") or column == "bamboo_guarded_pause_minus_rest_level_db"
    ]
    summary_rows = []
    for column in rest_columns:
        numeric = pd.to_numeric(comparison[column], errors="coerce")
        if numeric.notna().any():
            summary_rows.append(
                {
                    "metric": column,
                    "n_exact_session_pairs": int(numeric.notna().sum()),
                    "median": float(numeric.median()),
                    "q1": float(numeric.quantile(0.25)),
                    "q3": float(numeric.quantile(0.75)),
                    "minimum": float(numeric.min()),
                    "maximum": float(numeric.max()),
                }
            )
    _write_table(pd.DataFrame(summary_rows), output / "rest_reference_summary")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=rest["media_path"].tolist(),
    )


def command_encoding_sensitivity(cfg: dict) -> None:
    """Re-extract paired WAV/WEBM encodings; never treat them as independent recordings."""
    root = _output_root(cfg)
    audit_root = root / "00_audit"
    output = root / "04_analysis" / "encoding_sensitivity"
    media_rows = _read_existing(audit_root, "bamboo_media_rows_audited")
    inventory = _read_existing(audit_root, "bamboo_media_inventory")
    frozen_ids = set(
        _read_frozen(cfg, "frozen_bamboo_recordings")["logical_recording_id"].astype(str)
    )
    paired_ids = (
        media_rows.groupby("logical_recording_id")["extension_parsed"]
        .nunique()
        .loc[lambda values: values >= 2]
        .index
    )
    media_rows = media_rows.loc[
        media_rows["logical_recording_id"].astype(str).isin(frozen_ids)
        & media_rows["logical_recording_id"].isin(paired_ids)
        & media_rows["extension_parsed"].isin([".wav", ".webm"])
    ].copy()
    paths_by_name = inventory.groupby("file_name")["file_path"].agg(list).to_dict()
    ffmpeg = resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg")
    ffprobe = resolve_executable(cfg["software"]["ffprobe"], "ffprobe")
    vad = cfg["vad"]
    rows = []
    errors = []
    for _, row in tqdm(media_rows.iterrows(), total=len(media_rows), desc="Encoding sensitivity"):
        file_name = row["Raw Media File name"]
        candidates = paths_by_name.get(file_name, [])
        if len(candidates) != 1:
            errors.append(
                {
                    "file_name": file_name,
                    "stage": "resolve_path",
                    "error": f"expected 1 path; found {len(candidates)}",
                }
            )
            continue
        try:
            audio = decode_audio_views(candidates[0], ffmpeg=ffmpeg, ffprobe=ffprobe)
            duration = len(audio.analysis_16k) / 16000
            raw = silero_speech_intervals(
                audio.analysis_16k,
                threshold=vad["threshold"],
                min_speech_ms=vad["min_speech_ms"],
                min_silence_ms=vad["min_silence_ms"],
                speech_pad_ms=0,
                onnx=cfg["software"]["silero_backend"].lower() == "onnx",
            )
            views = build_segmentation_views(
                raw,
                duration_sec=duration,
                bridge_gap_ms=0,
                min_speech_ms=0,
                strict_speech_edge_ms=vad["strict_speech_edge_ms"],
                strict_nonspeech_edge_ms=vad["strict_nonspeech_edge_ms"],
            )
            rows.append(
                {
                    "logical_recording_id": row["logical_recording_id"],
                    "SubjectID": row["SubjectID"],
                    "encoding": row["extension_parsed"],
                    "file_name": file_name,
                    "duration_analysis_sec": duration,
                    **extract_all_metrics(audio, views),
                }
            )
        except Exception as exc:
            errors.append(
                {"file_name": file_name, "stage": "encoding_sensitivity", "error": repr(exc)}
            )
    metrics = pd.DataFrame(rows)
    _write_table(metrics, output / "paired_encoding_q_metrics_long")
    _write_table(pd.DataFrame(errors), output / "paired_encoding_errors")

    wav = metrics.loc[metrics["encoding"] == ".wav"].copy()
    webm = metrics.loc[metrics["encoding"] == ".webm"].copy()
    paired = wav.merge(
        webm,
        on=["logical_recording_id", "SubjectID"],
        suffixes=("_wav", "_webm"),
        how="inner",
        validate="one_to_one",
    )
    comparison_rows = []
    for feature in metric_registry_frame()["feature"]:
        left_column = f"{feature}_wav"
        right_column = f"{feature}_webm"
        if left_column not in paired or right_column not in paired:
            continue
        left = pd.to_numeric(paired[left_column], errors="coerce")
        right = pd.to_numeric(paired[right_column], errors="coerce")
        complete = left.notna() & right.notna()
        rho = float("nan")
        if (
            complete.sum() >= 10
            and left[complete].nunique() >= 3
            and right[complete].nunique() >= 3
        ):
            rho = float(stats.spearmanr(left[complete], right[complete]).statistic)
        comparison_rows.append(
            {
                "feature": feature,
                "n_complete_pairs": int(complete.sum()),
                "spearman_rho": rho,
                "median_webm_minus_wav": float((right[complete] - left[complete]).median())
                if complete.any()
                else float("nan"),
                "median_absolute_difference": float(
                    (right[complete] - left[complete]).abs().median()
                )
                if complete.any()
                else float("nan"),
            }
        )
    _write_table(paired, output / "paired_encoding_q_metrics_wide")
    _write_table(pd.DataFrame(comparison_rows), output / "paired_encoding_robustness")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=inventory["file_path"].tolist(),
    )


def _has_import_errors(issues: pd.DataFrame) -> bool:
    return (
        not issues.empty
        and issues.get("severity", pd.Series(dtype="string"))
        .astype(str)
        .str.casefold()
        .eq("error")
        .any()
    )


def _rater_workload_and_prevalence(ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        return pd.DataFrame(
            columns=[
                "rater_id",
                "category",
                "n_recordings",
                "positive_recordings",
                "negative_recordings",
                "positive_prevalence",
            ]
        )
    work = ratings.copy()
    work["rating_numeric"] = pd.to_numeric(work["rating"], errors="coerce")
    summary = work.groupby(["rater_id", "category"], as_index=False).agg(
        n_recordings=("file_name", "nunique"),
        positive_recordings=("rating_numeric", lambda values: int((values == 1).sum())),
        negative_recordings=("rating_numeric", lambda values: int((values == 0).sum())),
        positive_prevalence=("rating_numeric", "mean"),
    )
    return summary


def _distributed_design_summary(
    ratings: pd.DataFrame,
    *,
    expected_raters: int,
    expected_rater_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    coverage, _ = rating_design_coverage(ratings, expected_raters=expected_raters)
    observed = sorted(ratings["rater_id"].dropna().astype(str).unique())
    expected = sorted(map(str, expected_rater_names))
    names_match = not expected or observed == expected
    one_per_item = not coverage.empty and coverage["n_raters"].eq(1).all()
    total_match = len(observed) == expected_raters
    valid = bool(one_per_item and total_match and names_match)
    summary = (
        coverage.groupby("category", as_index=False).agg(
            items_total=("file_name", "nunique"),
            minimum_raters_per_item=("n_raters", "min"),
            maximum_raters_per_item=("n_raters", "max"),
        )
        if not coverage.empty
        else pd.DataFrame(
            columns=[
                "category",
                "items_total",
                "minimum_raters_per_item",
                "maximum_raters_per_item",
            ]
        )
    )
    summary["expected_raters_across_study"] = expected_raters
    summary["observed_rater_ids"] = "|".join(observed)
    summary["configured_rater_ids"] = "|".join(expected)
    summary["rater_names_match_configuration"] = names_match
    summary["design_status"] = (
        "valid_distributed_single_independent_rating"
        if valid
        else "blocked_distributed_design_mismatch"
    )
    return coverage, summary, valid


def _broad_metadata_labels(
    features: pd.DataFrame, broad: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direction_rows = [
        {
            "family": family,
            "metadata_column": column,
            "absent_value": broad.get("absent_value", "No"),
            "present_value": broad.get("present_value", "Yes"),
            "normalized_absent": 0,
            "normalized_present": 1,
            "direction": "higher_is_worse",
            "direction_confirmed": bool(broad.get("direction_confirmed", False)),
            "individual_ra_ratings_available": False,
            "reliability_estimable": False,
        }
        for family, column in broad.get("family_columns", {}).items()
    ]
    direction_audit = pd.DataFrame(direction_rows)
    if not broad.get("family_columns") or not broad.get("direction_confirmed", False):
        return pd.DataFrame(), direction_audit

    label_rows = []
    value_map = {
        str(broad.get("absent_value", "No")).strip().upper(): 0,
        str(broad.get("present_value", "Yes")).strip().upper(): 1,
    }
    for family, column in broad["family_columns"].items():
        if column not in features:
            continue
        normalized = features[column].astype("string").str.strip().str.upper().map(value_map)
        for index in features.index[normalized.notna()]:
            label_rows.append(
                {
                    "file_name": features.loc[index, "file_name"],
                    "category": family,
                    "consensus_rating": int(normalized.loc[index]),
                    "consensus_method": (
                        "merged_two_ra_metadata_label_individual_ratings_unavailable"
                    ),
                }
            )
    return pd.DataFrame(label_rows), direction_audit


def _command_human_qc_distributed(
    cfg: dict,
    schema: dict,
    *,
    source: Path,
    output: Path,
    input_paths: list[Path],
) -> None:
    """Analyze distributed main ratings and a separate crossed reliability subset."""
    expected_raters = int(schema.get("expected_raters", 4))
    expected_names = list(schema.get("rater_directory_names", []))
    reliability_name = str(schema.get("reliability_subdirectory", "Reliability"))
    rater_strategy = schema.get("rater_strategy", "parent_directory")
    parser_kwargs = {
        "rater_strategy": rater_strategy,
        "family_map": schema.get("family_map"),
        "context_columns": schema.get("context_columns"),
        "interval_time_base": schema.get("interval_time_base", "absolute"),
        "rater_directory_names": expected_names or None,
    }

    ratings, context, intervals, issues = load_interval_human_qc(
        source,
        exclude_path_parts=[reliability_name],
        **parser_kwargs,
    )
    _write_table(ratings, output / "main_distributed_ratings_long")
    _write_table(ratings, output / "ratings_long")
    _write_table(context, output / "main_context_annotations_not_family_alignment")
    _write_table(context, output / "context_annotations_not_family_alignment")
    _write_table(intervals, output / "main_annotation_intervals_long")
    _write_table(intervals, output / "annotation_intervals_long")
    _write_table(issues, output / "main_import_issues")
    _write_table(issues, output / "import_issues")
    _write_table(
        ratings[
            [
                "file_name",
                "rater_id",
                "category",
                "annotated_duration_sec",
                "annotated_fraction",
                "recording_duration_sec",
            ]
        ],
        output / "main_distributed_extent_labels_secondary",
    )
    main_coverage, main_design, main_valid = _distributed_design_summary(
        ratings,
        expected_raters=expected_raters,
        expected_rater_names=expected_names,
    )
    _write_table(main_coverage, output / "main_distributed_item_coverage")
    _write_table(main_coverage, output / "rating_design_item_coverage")
    _write_table(main_design, output / "main_distributed_design_summary")
    _write_table(main_design, output / "rating_design_summary")
    main_marginals = _rater_workload_and_prevalence(ratings)
    _write_table(main_marginals, output / "main_rater_workload_and_prevalence")

    if _has_import_errors(issues) or not main_valid:
        write_run_manifest(
            output / "run_manifest.json",
            command=sys.argv,
            config_path=cfg["_config_path"],
            input_paths=input_paths,
            extra={
                "status": "blocked_main_distributed_design",
                "reason": (
                    "main_import_errors"
                    if _has_import_errors(issues)
                    else "main_rater_or_item_coverage_mismatch"
                ),
            },
        )
        raise ValueError(
            "Main detailed human-QC import is blocked. Review main_import_issues.csv "
            "and main_distributed_design_summary.csv."
        )

    main_labels = ratings[["file_name", "rater_id", "category", "rating"]].rename(
        columns={"rating": "consensus_rating"}
    )
    main_labels["consensus_method"] = "single_independent_rater_in_distributed_four_ra_design"
    main_labels["n_ratings"] = 1
    _write_table(main_labels, output / "main_distributed_family_labels")

    dataset_root = _output_root(cfg) / "03_dataset_assembly"
    features = _read_existing(dataset_root, "paper1_analysis_dataset")
    features = features.loc[features["primary_measurement_eligible"].fillna(False)].copy()
    _, reviewed_registry = load_latest_feature_release(cfg["_project_root"])
    family_indices, index_audit = direction_oriented_family_indices(
        features, registry_frame=reviewed_registry
    )
    _write_table(family_indices, output / "direction_oriented_q_family_indices_secondary")
    _write_table(index_audit, output / "q_family_index_orientation_audit")

    bootstrap_replicates = cfg["analysis"]["bootstrap_replicates"]
    seed = cfg["project"]["random_seed"]
    minimum_class = cfg["analysis"]["minimum_human_class_recordings"]
    minimum_participants = cfg["analysis"]["minimum_human_class_participants"]
    main_alignment = rater_stratified_family_alignment(
        family_indices,
        main_labels,
        label_system="distributed_four_ra_single_rating_rater_stratified",
        minimum_class_recordings_per_rater=minimum_class,
        minimum_participants=minimum_participants,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    _write_table(
        main_alignment,
        output / "main_distributed_rater_stratified_family_alignment",
    )
    pooled_main_alignment = family_alignment_matrix(
        family_indices,
        main_labels,
        label_system="distributed_four_ra_pooled_single_label_sensitivity",
        minimum_class_recordings=minimum_class,
        minimum_participants=minimum_participants,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    _write_table(
        pooled_main_alignment,
        output / "main_distributed_pooled_family_alignment_sensitivity",
    )
    pooled_specificity = matched_family_specificity(
        family_indices,
        main_labels,
        label_system="distributed_four_ra_pooled_single_label_sensitivity",
        minimum_class_recordings=minimum_class,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    _write_table(
        pooled_specificity,
        output / "main_distributed_pooled_matched_family_specificity_sensitivity",
    )
    if schema.get("perceptual_metric_map"):
        links = perceptual_links(
            features,
            main_labels,
            schema["perceptual_metric_map"],
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            registry_frame=reviewed_registry,
        )
        _write_table(
            links,
            output / "main_distributed_feature_level_links_pooled_secondary",
        )

    reliability_root = source / reliability_name
    reliability_consensus = pd.DataFrame()
    reliability_status = "not_found"
    if reliability_root.exists() and any(reliability_root.rglob("*.csv")):
        input_paths.append(reliability_root)
        rel_ratings, rel_context, rel_intervals, rel_issues = load_interval_human_qc(
            reliability_root,
            **parser_kwargs,
        )
        _write_table(rel_ratings, output / "reliability_ratings_long")
        _write_table(rel_context, output / "reliability_context_annotations_not_family_alignment")
        _write_table(rel_intervals, output / "reliability_annotation_intervals_long")
        _write_table(rel_issues, output / "reliability_import_issues")
        rel_coverage, rel_design = rating_design_coverage(
            rel_ratings, expected_raters=expected_raters
        )
        _write_table(rel_coverage, output / "reliability_item_coverage")
        _write_table(rel_design, output / "reliability_design_summary")
        _write_table(
            _rater_workload_and_prevalence(rel_ratings),
            output / "reliability_rater_workload_and_prevalence",
        )
        observed_rel_names = sorted(rel_ratings["rater_id"].dropna().astype(str).unique())
        configured_rel_names = sorted(map(str, expected_names))
        rel_names_valid = len(observed_rel_names) == expected_raters and (
            not configured_rel_names or observed_rel_names == configured_rel_names
        )
        complete_keys = rel_coverage.loc[
            rel_coverage["complete_expected_raters"], ["file_name", "category"]
        ]
        complete_rel = rel_ratings.merge(
            complete_keys,
            on=["file_name", "category"],
            how="inner",
            validate="many_to_one",
        )
        rel_valid = (
            not _has_import_errors(rel_issues) and rel_names_valid and not complete_rel.empty
        )
        reliability_status = (
            "complete_subset_estimable" if rel_valid else "blocked_import_rater_or_coverage_error"
        )
        _write_table(
            pd.DataFrame(
                [
                    {
                        "status": reliability_status,
                        "expected_raters": expected_raters,
                        "observed_rater_ids": "|".join(observed_rel_names),
                        "configured_rater_ids": "|".join(configured_rel_names),
                        "unique_files_imported": rel_ratings["file_name"].nunique(),
                        "complete_file_family_items": len(complete_keys),
                        "incomplete_file_family_items": int(
                            (~rel_coverage["complete_expected_raters"]).sum()
                        ),
                        "primary_agreement_uses_complete_items_only": True,
                    }
                ]
            ),
            output / "reliability_analysis_status",
        )
        if rel_valid:
            agreement = agreement_summary(
                complete_rel,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
            agreement_sensitivity = agreement_summary(
                rel_ratings,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
            reliability_consensus = make_consensus(
                rel_ratings,
                expected_raters=expected_raters,
                minimum_ratings=expected_raters,
            )
            consensus_three = make_consensus(
                rel_ratings,
                expected_raters=expected_raters,
                minimum_ratings=max(2, expected_raters - 1),
            )
            extent_consensus = make_extent_consensus(
                rel_ratings,
                expected_raters=expected_raters,
                minimum_ratings=expected_raters,
            )
            _write_table(agreement, output / "reliability_interrater_agreement_complete")
            _write_table(
                agreement_sensitivity,
                output / "reliability_interrater_agreement_incomplete_sensitivity",
            )
            _write_table(
                reliability_consensus,
                output / "reliability_four_ra_consensus_primary",
            )
            _write_table(
                consensus_three,
                output / "reliability_four_ra_consensus_three_of_four_sensitivity",
            )
            _write_table(
                extent_consensus,
                output / "reliability_four_ra_extent_consensus_secondary",
            )
            rel_alignment = family_alignment_matrix(
                family_indices,
                reliability_consensus,
                label_system="crossed_reliability_four_ra_consensus",
                minimum_class_recordings=minimum_class,
                minimum_participants=minimum_participants,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
            _write_table(
                rel_alignment,
                output / "reliability_four_ra_consensus_family_alignment",
            )
            rel_specificity = matched_family_specificity(
                family_indices,
                reliability_consensus,
                label_system="crossed_reliability_four_ra_consensus",
                minimum_class_recordings=minimum_class,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
            _write_table(
                rel_specificity,
                output / "reliability_four_ra_consensus_matched_family_specificity",
            )
            shared_main_rel = sorted(
                set(main_labels["category"].dropna())
                & set(reliability_consensus["category"].dropna())
                & {
                    column.removeprefix("qfamily__")
                    for column in family_indices.columns
                    if column.startswith("qfamily__")
                }
            )
            if shared_main_rel:
                main_rel_comparison = compare_binary_label_systems(
                    family_indices,
                    main_labels,
                    reliability_consensus,
                    label_a_name="distributed_main_single_rating",
                    label_b_name="crossed_reliability_four_ra_consensus",
                    shared_families=shared_main_rel,
                    minimum_class_recordings=minimum_class,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed,
                )
                _write_table(
                    main_rel_comparison,
                    output / "main_distributed_vs_reliability_consensus_paired_alignment",
                )
    else:
        _write_table(
            pd.DataFrame(
                [
                    {
                        "status": "not_found",
                        "expected_path": str(reliability_root),
                        "required_layout": (
                            "Reliability/<RA name>/*_segments.csv with the same files "
                            "independently rated by all four RAs"
                        ),
                    }
                ]
            ),
            output / "reliability_analysis_status",
        )

    broad = schema.get("broad_metadata", {})
    broad_labels, direction_audit = _broad_metadata_labels(features, broad)
    _write_table(direction_audit, output / "two_ra_broad_direction_and_scale_audit")
    comparison_targets: list[tuple[str, pd.DataFrame]] = [("main_distributed", main_labels)]
    if not reliability_consensus.empty:
        comparison_targets.append(("reliability_four_ra_consensus", reliability_consensus))
    if not broad_labels.empty:
        _write_table(broad_labels, output / "two_ra_broad_family_labels_normalized")
        broad_alignment = family_alignment_matrix(
            family_indices,
            broad_labels,
            label_system="two_ra_broad_merged_metadata",
            minimum_class_recordings=minimum_class,
            minimum_participants=minimum_participants,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        _write_table(broad_alignment, output / "two_ra_broad_family_alignment_matrix")
        for target_name, target_labels in comparison_targets:
            shared = sorted(
                set(target_labels["category"].dropna())
                & set(broad_labels["category"].dropna())
                & {
                    column.removeprefix("qfamily__")
                    for column in family_indices.columns
                    if column.startswith("qfamily__")
                }
            )
            comparison = compare_binary_label_systems(
                family_indices,
                target_labels,
                broad_labels,
                label_a_name=target_name,
                label_b_name="two_ra_broad_merged_metadata",
                shared_families=shared,
                minimum_class_recordings=minimum_class,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
            _write_table(
                comparison,
                output / f"{target_name}_vs_two_ra_paired_alignment",
            )
    elif broad.get("family_columns"):
        blocked = pd.DataFrame(
            [
                {
                    "status": "blocked",
                    "reason": "broad_metadata_direction_not_confirmed",
                    "required_action": (
                        "Confirm Yes=artifact present and No=artifact absent in the RA "
                        "codebook, then set broad_metadata.direction_confirmed: true."
                    ),
                }
            ]
        )
        for target_name, _ in comparison_targets:
            _write_table(
                blocked,
                output / f"{target_name}_vs_two_ra_paired_alignment",
            )

    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=input_paths,
        extra={
            "status": "completed",
            "rating_design": "distributed_main_with_crossed_reliability_subset",
            "expected_raters": expected_raters,
            "reliability_status": reliability_status,
            "main_estimand": "within_rater_stratified_family_alignment",
            "reliability_estimand": "complete_four_ra_agreement_and_consensus_alignment",
            "family_alignment_not_source_alignment": True,
            "rest_role": "exact_session_contextual_sensitivity_not_primary_goal4_labels",
        },
    )


def command_human_qc(cfg: dict, schema_path: str | None) -> None:
    root = _output_root(cfg)
    output = root / "04_analysis" / "human_qc"
    schema = {}
    if schema_path:
        with Path(schema_path).open("r", encoding="utf-8") as handle:
            schema = yaml.safe_load(handle) or {}
    source = resolve_data_path(cfg, "detailed_human_qc")
    rating_design = schema.get("rating_design", "distributed_single_rating")
    expected_raters = int(schema.get("expected_raters", 4))
    minimum_primary_ratings = int(schema.get("minimum_primary_ratings", expected_raters))
    minimum_sensitivity_ratings = int(
        schema.get("minimum_sensitivity_ratings", max(2, expected_raters - 1))
    )
    input_paths: list[Path] = [source]

    if rating_design in {
        "distributed_single_rating",
        "distributed_with_crossed_reliability",
    }:
        if schema.get("format", "interval_json") != "interval_json":
            raise ValueError(
                "The distributed-with-reliability workflow currently requires "
                "format: interval_json."
            )
        _command_human_qc_distributed(
            cfg,
            schema,
            source=source,
            output=output,
            input_paths=input_paths,
        )
        return

    if schema.get("format", "interval_json") == "interval_json":
        manifest_path = schema.get("rater_manifest")
        if manifest_path:
            manifest_path = Path(manifest_path)
            if not manifest_path.is_absolute():
                manifest_path = Path(cfg["_config_path"]).parent.parent / manifest_path
            input_paths.append(manifest_path)
        ratings, context_ratings, annotation_intervals, issues = load_interval_human_qc(
            source,
            manifest_path=manifest_path,
            rater_strategy=schema.get("rater_strategy", "manifest"),
            family_map=schema.get("family_map"),
            context_columns=schema.get("context_columns"),
            interval_time_base=schema.get("interval_time_base", "absolute"),
        )
        _write_table(annotation_intervals, output / "annotation_intervals_long")
        _write_table(context_ratings, output / "context_annotations_not_family_alignment")
        if rating_design == "distributed_single_rating":
            _write_table(
                ratings[
                    [
                        "file_name",
                        "rater_id",
                        "category",
                        "annotated_duration_sec",
                        "annotated_fraction",
                        "recording_duration_sec",
                    ]
                ],
                output / "distributed_four_ra_extent_labels_secondary",
            )
        else:
            extent_consensus = make_extent_consensus(
                ratings,
                expected_raters=expected_raters,
                minimum_ratings=minimum_primary_ratings,
            )
            _write_table(extent_consensus, output / "four_ra_extent_consensus_secondary")
    else:
        kwargs = {
            key: schema[key]
            for key in ("file_column", "rater_column", "category_columns")
            if key in schema
        }
        ratings, issues = load_human_qc_long(source, **kwargs)

    item_coverage, design_summary = rating_design_coverage(ratings, expected_raters=expected_raters)
    _write_table(item_coverage, output / "rating_design_item_coverage")
    _write_table(design_summary, output / "rating_design_summary")
    _write_table(ratings, output / "ratings_long")
    _write_table(issues, output / "import_issues")

    blocking_import = (
        not issues.empty
        and issues.get("severity", pd.Series(dtype="string")).astype(str).eq("error").any()
    )
    design_estimable = (
        not design_summary.empty
        and design_summary["design_status"].eq("agreement_estimable_on_complete_subset").all()
    )
    if blocking_import or not design_estimable:
        write_run_manifest(
            output / "run_manifest.json",
            command=sys.argv,
            config_path=cfg["_config_path"],
            input_paths=input_paths,
            extra={
                "status": "blocked_before_agreement",
                "expected_raters": expected_raters,
                "reason": (
                    "import_errors" if blocking_import else "no_item_has_all_expected_raters"
                ),
            },
        )
        raise ValueError(
            "Detailed human-QC analysis is blocked. Review import_issues.csv and "
            "rating_design_summary.csv; rater identity and a shared four-RA subset are required."
        )

    complete_keys = item_coverage.loc[
        item_coverage["complete_expected_raters"], ["file_name", "category"]
    ]
    complete_ratings = ratings.merge(
        complete_keys,
        on=["file_name", "category"],
        how="inner",
        validate="many_to_one",
    )
    agreement = agreement_summary(
        complete_ratings,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
    )
    agreement_incomplete_sensitivity = agreement_summary(
        ratings,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
    )
    consensus = make_consensus(
        ratings,
        expected_raters=expected_raters,
        minimum_ratings=minimum_primary_ratings,
    )
    consensus_sensitivity = make_consensus(
        ratings,
        expected_raters=expected_raters,
        minimum_ratings=minimum_sensitivity_ratings,
    )
    rater_marginals = (
        ratings.groupby(["category", "rater_id", "rating"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    rater_marginals["proportion_within_category_rater"] = rater_marginals[
        "n"
    ] / rater_marginals.groupby(["category", "rater_id"])["n"].transform("sum")
    consensus_distribution = (
        consensus.groupby(["category", "consensus_rating"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    consensus_distribution["proportion_within_category"] = consensus_distribution[
        "n"
    ] / consensus_distribution.groupby("category")["n"].transform("sum")
    _write_table(rater_marginals, output / "rater_marginal_distributions")
    _write_table(agreement, output / "interrater_agreement")
    _write_table(
        agreement_incomplete_sensitivity,
        output / "interrater_agreement_incomplete_design_sensitivity",
    )
    _write_table(consensus, output / "four_ra_consensus_primary")
    _write_table(consensus_sensitivity, output / "four_ra_consensus_three_of_four_sensitivity")
    _write_table(consensus_distribution, output / "consensus_distribution")

    features = _read_existing(root / "03_dataset_assembly", "paper1_analysis_dataset")
    features = features.loc[features["primary_measurement_eligible"].fillna(False)].copy()
    _, reviewed_registry = load_latest_feature_release(cfg["_project_root"])
    family_indices, index_audit = direction_oriented_family_indices(
        features, registry_frame=reviewed_registry
    )
    _write_table(family_indices, output / "direction_oriented_q_family_indices_secondary")
    _write_table(index_audit, output / "q_family_index_orientation_audit")
    four_ra_alignment = family_alignment_matrix(
        family_indices,
        consensus,
        label_system="four_ra_detailed_consensus",
        minimum_class_recordings=cfg["analysis"]["minimum_human_class_recordings"],
        minimum_participants=cfg["analysis"]["minimum_human_class_participants"],
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
    )
    _write_table(four_ra_alignment, output / "four_ra_family_alignment_matrix")
    four_ra_specificity = matched_family_specificity(
        family_indices,
        consensus,
        label_system="four_ra_detailed_consensus",
        minimum_class_recordings=cfg["analysis"]["minimum_human_class_recordings"],
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
    )
    _write_table(four_ra_specificity, output / "four_ra_matched_family_specificity")

    if schema.get("perceptual_metric_map"):
        links = perceptual_links(
            features,
            consensus,
            schema["perceptual_metric_map"],
            bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
            seed=cfg["project"]["random_seed"],
            registry_frame=reviewed_registry,
        )
        _write_table(links, output / "four_ra_feature_level_links_secondary")

    broad = schema.get("broad_metadata", {})
    direction_audit = pd.DataFrame(
        [
            {
                "family": family,
                "metadata_column": column,
                "absent_value": broad.get("absent_value", "No"),
                "present_value": broad.get("present_value", "Yes"),
                "normalized_absent": 0,
                "normalized_present": 1,
                "direction": "higher_is_worse",
                "direction_confirmed": bool(broad.get("direction_confirmed", False)),
            }
            for family, column in broad.get("family_columns", {}).items()
        ]
    )
    _write_table(direction_audit, output / "two_ra_broad_direction_and_scale_audit")
    if broad.get("family_columns") and broad.get("direction_confirmed", False):
        label_rows = []
        value_map = {
            str(broad.get("absent_value", "No")).strip().upper(): 0,
            str(broad.get("present_value", "Yes")).strip().upper(): 1,
        }
        for family, column in broad["family_columns"].items():
            if column not in features:
                continue
            normalized = features[column].astype("string").str.strip().str.upper().map(value_map)
            for index in features.index[normalized.notna()]:
                label_rows.append(
                    {
                        "file_name": features.loc[index, "file_name"],
                        "category": family,
                        "consensus_rating": int(normalized.loc[index]),
                        "consensus_method": (
                            "merged_two_ra_metadata_label_individual_ratings_unavailable"
                        ),
                    }
                )
        broad_labels = pd.DataFrame(label_rows)
        _write_table(broad_labels, output / "two_ra_broad_family_labels_normalized")
        broad_alignment = family_alignment_matrix(
            family_indices,
            broad_labels,
            label_system="two_ra_broad_merged_metadata",
            minimum_class_recordings=cfg["analysis"]["minimum_human_class_recordings"],
            minimum_participants=cfg["analysis"]["minimum_human_class_participants"],
            bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
            seed=cfg["project"]["random_seed"],
        )
        _write_table(broad_alignment, output / "two_ra_broad_family_alignment_matrix")
        broad_specificity = matched_family_specificity(
            family_indices,
            broad_labels,
            label_system="two_ra_broad_merged_metadata",
            minimum_class_recordings=cfg["analysis"]["minimum_human_class_recordings"],
            bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
            seed=cfg["project"]["random_seed"],
        )
        _write_table(
            broad_specificity,
            output / "two_ra_broad_matched_family_specificity",
        )
        shared_families = sorted(
            set(consensus["category"].dropna())
            & set(broad_labels["category"].dropna())
            & {
                column.removeprefix("qfamily__")
                for column in family_indices.columns
                if column.startswith("qfamily__")
            }
        )
        comparison = compare_binary_label_systems(
            family_indices,
            consensus,
            broad_labels,
            label_a_name="four_ra_detailed_consensus",
            label_b_name="two_ra_broad_merged_metadata",
            shared_families=shared_families,
            minimum_class_recordings=cfg["analysis"]["minimum_human_class_recordings"],
            bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
            seed=cfg["project"]["random_seed"],
        )
        _write_table(comparison, output / "four_ra_vs_two_ra_paired_alignment")
    elif broad.get("family_columns"):
        _write_table(
            pd.DataFrame(
                [
                    {
                        "status": "blocked",
                        "reason": "broad_metadata_direction_not_confirmed",
                        "required_action": (
                            "Confirm Yes=artifact present and No=artifact absent in the RA codebook, "
                            "then set broad_metadata.direction_confirmed: true."
                        ),
                    }
                ]
            ),
            output / "four_ra_vs_two_ra_paired_alignment",
        )
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=input_paths,
        extra={
            "expected_raters": expected_raters,
            "minimum_primary_ratings": minimum_primary_ratings,
            "minimum_sensitivity_ratings": minimum_sensitivity_ratings,
            "family_alignment_not_source_alignment": True,
        },
    )


def command_broad_qc(cfg: dict) -> None:
    """Legacy broad-QC analysis; Goal 4 uses the paired workflow in ``human-qc``."""
    root = _output_root(cfg)
    output = root / "04_analysis" / "broad_metadata_qc"
    data = _read_existing(root / "03_dataset_assembly", "paper1_analysis_dataset")
    data = data.loc[data["primary_measurement_eligible"].fillna(False)].copy()
    label_columns = {
        "background_noise": "Background Noise",
        "volume_instability": "Volume is Unstable",
        "poor_audio_quality": "Poor Audio Quality",
        "another_person_speaks": "Another Person Speaks",
    }
    label_rows = []
    for category, column in label_columns.items():
        if column not in data:
            continue
        mapped = data[column].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
        for index in data.index[mapped.notna()]:
            label_rows.append(
                {
                    "file_name": data.loc[index, "file_name"],
                    "category": category,
                    "consensus_rating": int(mapped.loc[index]),
                    "consensus_method": "merged_metadata_label_rater_agreement_unavailable",
                    "n_ratings": pd.NA,
                    "requires_adjudication": False,
                }
            )
    labels = pd.DataFrame(label_rows)
    mapping = {
        "background_noise": [
            "qadd_pause_ac_level_dbfs_median",
            "qadd_pause_level_iqr_db",
            "qadd_speech_pause_level_contrast_db",
            "qadd_pause_spectral_flatness",
            "qadd_mains_hum_comb_score_db",
        ],
        "volume_instability": [
            "qgain_within_segment_iqr_db",
            "qgain_between_segment_mad_db",
            "qgain_abs_drift_db_per_min",
        ],
        "poor_audio_quality": [
            "qadd_speech_pause_level_contrast_db",
            "qgain_within_segment_iqr_db",
            "qrev_tail_excess_100ms_db",
            "qdist_hard_clipped_sample_fraction",
        ],
        "another_person_speaks": [
            "qadd_pause_ac_level_dbfs_median",
            "qadd_speech_pause_level_contrast_db",
        ],
    }
    _, reviewed_registry = load_latest_feature_release(cfg["_project_root"])
    links = perceptual_links(
        data,
        labels,
        mapping,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
        registry_frame=reviewed_registry,
    )
    prevalence = (
        labels.groupby("category")["consensus_rating"]
        .agg(n="size", positive="sum", prevalence="mean")
        .reset_index()
    )
    _write_table(labels, output / "broad_metadata_labels_long")
    _write_table(prevalence, output / "broad_metadata_label_prevalence")
    _write_table(links, output / "broad_metadata_perceptual_links")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=[root / "03_dataset_assembly" / "paper1_analysis_dataset.csv"],
    )


def command_describe(cfg: dict) -> None:
    root = _output_root(cfg)
    output = root / "04_analysis" / "descriptive"
    features = _read_existing(root / "03_dataset_assembly", "paper1_analysis_dataset")
    features = features.loc[features["primary_measurement_eligible"].fillna(False)].copy()
    _, reviewed_registry = load_latest_feature_release(cfg["_project_root"])
    summary = describe_metrics(
        features,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
        registry_frame=reviewed_registry,
    )
    _write_table(summary, output / "metric_descriptive_statistics")
    correlations = pairwise_clustered_spearman(
        features,
        minimum_participants=cfg["analysis"]["minimum_complete_participants"],
        bootstrap_replicates=min(500, cfg["analysis"]["bootstrap_replicates"]),
        seed=cfg["project"]["random_seed"],
        registry_frame=reviewed_registry,
    )
    _write_table(correlations, output / "pairwise_clustered_spearman")
    persistence = participant_persistence(
        features,
        minimum_participants=cfg["analysis"]["minimum_complete_participants"],
        registry_frame=reviewed_registry,
    )
    _write_table(persistence, output / "participant_persistence_not_reliability")
    contrasts = participant_level_group_contrasts(
        features,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
        registry_frame=reviewed_registry,
    )
    _write_table(contrasts, output / "exploratory_participant_level_diagnosis_contrasts")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=[root / "03_dataset_assembly" / "paper1_analysis_dataset.csv"],
    )


def command_sensitivity(cfg: dict) -> None:
    root = _output_root(cfg)
    feature_root = root / "02_features"
    output = root / "04_analysis" / "sensitivity"
    primary = _read_existing(feature_root, "bamboo_q_metrics")
    legacy_registry = metric_registry_frame()
    rows = []
    for profile in cfg["vad"].get("sensitivity_profiles", {}):
        comparison = _read_existing(feature_root, f"bamboo_q_metrics_{profile}")
        joined = primary.merge(
            comparison,
            on="logical_recording_id",
            how="outer",
            suffixes=("_primary", f"_{profile}"),
            indicator=True,
            validate="one_to_one",
        )
        for feature in legacy_registry["feature"]:
            left_column = f"{feature}_primary"
            right_column = f"{feature}_{profile}"
            if left_column not in joined or right_column not in joined:
                continue
            left = pd.to_numeric(joined[left_column], errors="coerce")
            right = pd.to_numeric(joined[right_column], errors="coerce")
            complete = left.notna() & right.notna()
            rho = float("nan")
            if (
                complete.sum() >= 10
                and left[complete].nunique() >= 3
                and right[complete].nunique() >= 3
            ):
                rho = float(stats.spearmanr(left[complete], right[complete]).statistic)
            rows.append(
                {
                    "profile": profile,
                    "measurement_layer": "legacy_profile_diagnostic_not_reviewed_release",
                    "feature": feature,
                    "n_primary": int(left.notna().sum()),
                    "n_sensitivity": int(right.notna().sum()),
                    "n_complete_pair": int(complete.sum()),
                    "spearman_rho": rho,
                    "median_signed_difference_sensitivity_minus_primary": float(
                        (right[complete] - left[complete]).median()
                    )
                    if complete.any()
                    else float("nan"),
                    "median_absolute_difference": float(
                        (right[complete] - left[complete]).abs().median()
                    )
                    if complete.any()
                    else float("nan"),
                    "primary_only_recordings": int((joined["_merge"] == "left_only").sum()),
                    "sensitivity_only_recordings": int((joined["_merge"] == "right_only").sum()),
                }
            )
    _write_table(pd.DataFrame(rows), output / "segmentation_profile_robustness")
    assembled = _read_existing(root / "03_dataset_assembly", "paper1_analysis_dataset")
    assembled = assembled.loc[assembled["primary_measurement_eligible"].fillna(False)].copy()
    _, reviewed_registry = load_latest_feature_release(cfg["_project_root"])
    one_per_participant = one_recording_per_participant(
        assembled, seed=cfg["analysis"]["one_recording_per_participant_seed"]
    )
    _write_table(
        one_per_participant[["SubjectID", "logical_recording_id", "file_name"]],
        output / "one_recording_per_participant_selection",
    )
    full_summary = describe_metrics(
        assembled,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
        registry_frame=reviewed_registry,
    ).add_suffix("_full")
    one_summary = describe_metrics(
        one_per_participant,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
        registry_frame=reviewed_registry,
    ).add_suffix("_one_per_participant")
    comparison = full_summary.merge(
        one_summary,
        left_on="feature_full",
        right_on="feature_one_per_participant",
        how="outer",
        validate="one_to_one",
    )
    _write_table(comparison, output / "one_recording_per_participant_robustness")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=[
            feature_root / "bamboo_q_metrics.csv",
            root / "03_dataset_assembly" / "paper1_analysis_dataset.csv",
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper1-qc")
    parser.add_argument("--config", default="config/project.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="Audit and reconcile all metadata workbooks")
    inventory = subparsers.add_parser("inventory", help="Probe and reconcile all media files")
    inventory.add_argument(
        "--no-hashes", action="store_true", help="Skip SHA-256 only for a fast dry run"
    )
    subparsers.add_parser(
        "freeze-template",
        help="Create the local diagnosis-adjudication template without overwriting it",
    )
    subparsers.add_parser(
        "freeze",
        help="Create an immutable, versioned cohort freeze after audit and inventory",
    )
    subparsers.add_parser(
        "segment", help="Run version-pinned Silero and create distinct segmentation views"
    )
    subparsers.add_parser(
        "segment-template",
        help="Create/refresh the mandatory segmentation review and manual-boundary tables",
    )
    subparsers.add_parser(
        "segment-adjudicate",
        help="Validate review decisions/manual boundaries and freeze final intervals",
    )
    extract = subparsers.add_parser(
        "extract", help="Extract registered Q metrics from native and VAD audio views"
    )
    extract.add_argument(
        "--profile", default="primary", help="primary, conservative, or permissive"
    )
    subparsers.add_parser(
        "reviewed-release",
        help="Build the canonical latest release from approved reviewed feature tables",
    )
    subparsers.add_parser(
        "assemble", help="Merge audited metadata and metrics with explicit eligibility gates"
    )
    subparsers.add_parser(
        "rest-reference", help="Extract exact-session Rest context without speech VAD"
    )
    subparsers.add_parser(
        "encoding-sensitivity", help="Re-extract paired WAV/WEBM technical replicates"
    )
    subparsers.add_parser("describe", help="Participant-clustered descriptive inference")
    subparsers.add_parser(
        "sensitivity", help="Compare conservative/permissive segmentation profiles"
    )
    subparsers.add_parser(
        "broad-qc",
        help="Legacy analysis of merged broad metadata QC; Goal 4 is run by human-qc",
    )
    human = subparsers.add_parser(
        "human-qc",
        help=(
            "Audit distributed main and crossed reliability annotations, then run "
            "family-level detailed/2RA alignment"
        ),
    )
    human.add_argument("--schema", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    commands = {
        "audit": lambda: command_audit(cfg),
        "inventory": lambda: command_inventory(cfg, hashes=not args.no_hashes),
        "freeze-template": lambda: command_freeze_template(cfg),
        "freeze": lambda: command_freeze(cfg),
        "segment": lambda: command_segment(cfg),
        "segment-template": lambda: command_segment_template(cfg),
        "segment-adjudicate": lambda: command_segment_adjudicate(cfg),
        "extract": lambda: command_extract(cfg, profile=args.profile),
        "reviewed-release": lambda: command_reviewed_release(cfg),
        "assemble": lambda: command_assemble(cfg),
        "rest-reference": lambda: command_rest_reference(cfg),
        "encoding-sensitivity": lambda: command_encoding_sensitivity(cfg),
        "describe": lambda: command_describe(cfg),
        "sensitivity": lambda: command_sensitivity(cfg),
        "broad-qc": lambda: command_broad_qc(cfg),
        "human-qc": lambda: command_human_qc(cfg, args.schema),
    }
    try:
        commands[args.command]()
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "command": args.command, "error": repr(exc)}, indent=2),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
