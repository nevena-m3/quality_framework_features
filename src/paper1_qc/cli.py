from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress display is non-essential
    def tqdm(iterable, **_kwargs):
        return iterable

from .config import load_config, resolve_data_path, resolve_executable
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
from .segmentation import Interval, build_segmentation_views, intervals_to_frame, silero_speech_intervals
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
)


def _output_root(cfg: dict) -> Path:
    root = Path(cfg["_output_root"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_table(frame: pd.DataFrame, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(path_without_suffix.with_suffix(".parquet"), index=False)
    except (ImportError, ValueError, TypeError):
        pass


def _audit_kwargs(cfg: dict) -> dict:
    return {
        "sentinel_values": cfg["clinical_alignment"]["sentinel_values"],
        "control_id_patterns": cfg["cohort"]["control_id_patterns"],
        "media_preference": cfg["cohort"]["media_preference"],
        "max_primary_assessment_gap_days": cfg["clinical_alignment"]["primary_max_assessment_gap_days"],
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
    audits = {name: audit_metadata_workbook(path, **_audit_kwargs(cfg)) for name, path in inputs.items()}
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
    pairs = exact_session_pairs(audits["bamboo"].canonical_recordings, audits["rest"].canonical_recordings)
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
            metadata = pd.read_parquet(metadata_path) if metadata_path.exists() else pd.read_csv(metadata_csv)
            coverage, duplicates = reconcile_inventory_with_metadata(inventory, metadata)
            consistency = audit_native_metadata_consistency(inventory, metadata)
            _write_table(coverage, output / f"{role}_media_coverage")
            _write_table(duplicates, output / f"{role}_duplicate_disk_names")
            _write_table(consistency, output / f"{role}_native_metadata_consistency_issues")
    combined_inventory = pd.concat(inventories, ignore_index=True) if inventories else pd.DataFrame()
    _write_table(combined_inventory, output / "all_media_inventory")
    write_run_manifest(
        output / "inventory_run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=roots.values(),
        extra={"compute_sha256": hashes},
    )


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
    audit_root = root / "00_audit"
    output = root / "01_segmentation"
    canonical = _read_existing(audit_root, "bamboo_canonical_recordings")
    inventory = _read_existing(audit_root, "bamboo_media_inventory")
    paths_by_name = inventory.groupby("file_name")["file_path"].agg(list).to_dict()
    ffmpeg = resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg")
    ffprobe = resolve_executable(cfg["software"]["ffprobe"], "ffprobe")
    vad = cfg["vad"]
    rows = []
    errors = []
    for _, row in tqdm(canonical.iterrows(), total=len(canonical), desc="Segment Bamboo"):
        file_name = row["Raw Media File name"]
        candidates = paths_by_name.get(file_name, [])
        if len(candidates) != 1:
            errors.append({"file_name": file_name, "stage": "resolve_path", "error": f"expected 1 path; found {len(candidates)}"})
            continue
        try:
            audio = decode_audio_views(candidates[0], ffmpeg=ffmpeg, ffprobe=ffprobe)
            duration = len(audio.analysis_16k) / 16000
            raw = silero_speech_intervals(
                audio.analysis_16k,
                threshold=vad["threshold"],
                min_speech_ms=vad["min_speech_ms"],
                min_silence_ms=vad["min_silence_ms"],
                speech_pad_ms=vad["speech_pad_ms"],
                onnx=cfg["software"]["silero_backend"].lower() == "onnx",
            )
            views = build_segmentation_views(
                raw,
                duration_sec=duration,
                bridge_gap_ms=vad["bridge_gap_ms"],
                min_speech_ms=vad["min_speech_ms"],
                strict_speech_edge_ms=vad["strict_speech_edge_ms"],
                strict_nonspeech_edge_ms=vad["strict_nonspeech_edge_ms"],
            )
            table = intervals_to_frame(views, file_name)
            table["profile"] = "primary"
            table["logical_recording_id"] = row["logical_recording_id"]
            rows.append(table)
            for profile, overrides in vad.get("sensitivity_profiles", {}).items():
                sensitivity_views = build_segmentation_views(
                    raw,
                    duration_sec=duration,
                    bridge_gap_ms=overrides.get("bridge_gap_ms", vad["bridge_gap_ms"]),
                    min_speech_ms=vad["min_speech_ms"],
                    strict_speech_edge_ms=overrides.get(
                        "strict_speech_edge_ms", vad["strict_speech_edge_ms"]
                    ),
                    strict_nonspeech_edge_ms=overrides.get(
                        "strict_nonspeech_edge_ms", vad["strict_nonspeech_edge_ms"]
                    ),
                )
                sensitivity_table = intervals_to_frame(sensitivity_views, file_name)
                sensitivity_table["profile"] = profile
                sensitivity_table["logical_recording_id"] = row["logical_recording_id"]
                rows.append(sensitivity_table)
        except Exception as exc:  # error ledger is part of the stage contract
            errors.append({"file_name": file_name, "stage": "segmentation", "error": repr(exc)})
    intervals = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _write_table(intervals, output / "bamboo_segmentation_intervals")
    _write_table(pd.DataFrame(errors), output / "segmentation_errors")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=inventory["file_path"].tolist(),
        extra={"silero_version_expected": cfg["software"]["silero_version"]},
    )


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
    audit_root = root / "00_audit"
    output = root / "02_features"
    canonical = _read_existing(audit_root, "bamboo_canonical_recordings")
    inventory = _read_existing(audit_root, "bamboo_media_inventory")
    intervals = _read_existing(root / "01_segmentation", "bamboo_segmentation_intervals")
    paths_by_name = inventory.groupby("file_name")["file_path"].agg(list).to_dict()
    ffmpeg = resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg")
    ffprobe = resolve_executable(cfg["software"]["ffprobe"], "ffprobe")
    results = []
    errors = []
    for _, row in tqdm(canonical.iterrows(), total=len(canonical), desc="Extract Q metrics"):
        file_name = row["Raw Media File name"]
        candidates = paths_by_name.get(file_name, [])
        if len(candidates) != 1:
            errors.append({"file_name": file_name, "stage": "resolve_path", "error": f"expected 1 path; found {len(candidates)}"})
            continue
        try:
            views = _views_for_file(intervals, file_name, profile=profile)
            required = {"raw_speech", "primary_speech", "strict_speech", "strict_internal_nonspeech"}
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
                    **metrics,
                }
            )
        except Exception as exc:
            errors.append({"file_name": file_name, "stage": "feature_extraction", "error": repr(exc)})
    feature_frame = pd.DataFrame(results)
    suffix = "" if profile == "primary" else f"_{profile}"
    _write_table(feature_frame, output / f"bamboo_q_metrics{suffix}")
    _write_table(pd.DataFrame(errors), output / f"feature_extraction_errors{suffix}")
    _write_table(metric_registry_frame(), output / "metric_registry")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=inventory["file_path"].tolist(),
        extra={"segmentation_profile": profile},
    )


def command_assemble(cfg: dict) -> None:
    root = _output_root(cfg)
    output = root / "03_dataset_assembly"
    metadata = _read_existing(root / "00_audit", "bamboo_canonical_recordings")
    features = _read_existing(root / "02_features", "bamboo_q_metrics")
    merged = metadata.merge(
        features,
        on=["logical_recording_id", "SubjectID"],
        how="left",
        suffixes=("_metadata", ""),
        validate="one_to_one",
        indicator=True,
    )
    task_completed = merged["Task Completed as Instructed"].astype(str).str.upper()
    merged["hard_exclusion_task_not_completed"] = task_completed.eq("NO")
    merged["review_task_completion_missing"] = merged["Task Completed as Instructed"].isna()
    merged["hard_exclusion_no_usable_speech"] = merged["qgain_status"].eq("insufficient_support")
    merged["feature_extraction_missing"] = merged["_merge"].ne("both")
    merged["confirmed_analysis_diagnosis"] = merged["diagnosis_reported"].isin(
        cfg["cohort"]["allowed_diagnoses"]
    )
    merged["primary_measurement_eligible"] = ~(
        merged["hard_exclusion_task_not_completed"]
        | merged["review_task_completion_missing"]
        | merged["hard_exclusion_no_usable_speech"]
        | merged["feature_extraction_missing"]
    )
    merged["diagnosis_contrast_eligible"] = (
        merged["primary_measurement_eligible"] & merged["confirmed_analysis_diagnosis"]
    )
    assessment_delta = pd.to_numeric(merged.get("assessment_recording_delta_days"), errors="coerce")
    merged["clinical_primary_eligible"] = (
        merged["primary_measurement_eligible"]
        & assessment_delta.abs().le(cfg["clinical_alignment"]["primary_max_assessment_gap_days"])
        & pd.to_numeric(merged["ALSFRS total score"], errors="coerce").notna()
    )
    merged["clinical_30day_sensitivity_eligible"] = (
        merged["primary_measurement_eligible"]
        & assessment_delta.abs().le(cfg["clinical_alignment"]["sensitivity_max_assessment_gap_days"])
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
            root / "00_audit" / "bamboo_canonical_recordings.csv",
            root / "02_features" / "bamboo_q_metrics.csv",
        ],
    )


def command_rest_reference(cfg: dict) -> None:
    """Extract exact-session Rest contextual metrics without speech VAD."""
    root = _output_root(cfg)
    audit_root = root / "00_audit"
    output = root / "04_analysis" / "rest_reference"
    pairs = _read_existing(audit_root, "exact_bamboo_rest_session_pairs")
    rest = _read_existing(audit_root, "rest_canonical_recordings")
    inventory = _read_existing(audit_root, "rest_media_inventory")
    target_ids = set(pairs["logical_recording_id_rest"].astype(str))
    rest = rest.loc[rest["logical_recording_id"].astype(str).isin(target_ids)].copy()
    paths_by_name = inventory.groupby("file_name")["file_path"].agg(list).to_dict()
    ffmpeg = resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg")
    ffprobe = resolve_executable(cfg["software"]["ffprobe"], "ffprobe")
    rows = []
    errors = []
    for _, row in tqdm(rest.iterrows(), total=len(rest), desc="Extract Rest reference"):
        file_name = row["Raw Media File name"]
        candidates = paths_by_name.get(file_name, [])
        if len(candidates) != 1:
            errors.append(
                {"file_name": file_name, "stage": "resolve_path", "error": f"expected 1 path; found {len(candidates)}"}
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
    comparison = (
        pairs.merge(
            rest_metrics,
            on=["logical_recording_id_rest", "SubjectID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            bamboo,
            left_on=["logical_recording_id_bamboo", "SubjectID"],
            right_on=["logical_recording_id", "SubjectID"],
            how="left",
            suffixes=("_pair", "_bamboo"),
            validate="one_to_one",
        )
    )
    comparison["bamboo_guarded_pause_minus_rest_level_db"] = (
        pd.to_numeric(comparison["qadd_nonspeech_level_dbfs"], errors="coerce")
        - pd.to_numeric(comparison["restref_level_dbfs"], errors="coerce")
    )
    _write_table(comparison, output / "exact_session_bamboo_rest_comparison")
    write_run_manifest(
        output / "run_manifest.json",
        command=sys.argv,
        config_path=cfg["_config_path"],
        input_paths=inventory["file_path"].tolist(),
    )


def command_encoding_sensitivity(cfg: dict) -> None:
    """Re-extract paired WAV/WEBM encodings; never treat them as independent recordings."""
    root = _output_root(cfg)
    audit_root = root / "00_audit"
    output = root / "04_analysis" / "encoding_sensitivity"
    media_rows = _read_existing(audit_root, "bamboo_media_rows_audited")
    inventory = _read_existing(audit_root, "bamboo_media_inventory")
    paired_ids = (
        media_rows.groupby("logical_recording_id")["extension_parsed"]
        .nunique()
        .loc[lambda values: values >= 2]
        .index
    )
    media_rows = media_rows.loc[
        media_rows["logical_recording_id"].isin(paired_ids)
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
                {"file_name": file_name, "stage": "resolve_path", "error": f"expected 1 path; found {len(candidates)}"}
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
                speech_pad_ms=vad["speech_pad_ms"],
                onnx=cfg["software"]["silero_backend"].lower() == "onnx",
            )
            views = build_segmentation_views(
                raw,
                duration_sec=duration,
                bridge_gap_ms=vad["bridge_gap_ms"],
                min_speech_ms=vad["min_speech_ms"],
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
            errors.append({"file_name": file_name, "stage": "encoding_sensitivity", "error": repr(exc)})
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
        if complete.sum() >= 10 and left[complete].nunique() >= 3 and right[complete].nunique() >= 3:
            rho = float(stats.spearmanr(left[complete], right[complete]).statistic)
        comparison_rows.append(
            {
                "feature": feature,
                "n_complete_pairs": int(complete.sum()),
                "spearman_rho": rho,
                "median_webm_minus_wav": float((right[complete] - left[complete]).median())
                if complete.any()
                else float("nan"),
                "median_absolute_difference": float((right[complete] - left[complete]).abs().median())
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


def command_human_qc(cfg: dict, schema_path: str | None) -> None:
    root = _output_root(cfg)
    output = root / "04_analysis" / "human_qc"
    schema = {}
    if schema_path:
        with Path(schema_path).open("r", encoding="utf-8") as handle:
            schema = yaml.safe_load(handle) or {}
    source = resolve_data_path(cfg, "detailed_human_qc")
    expected_raters = int(schema.get("expected_raters", 4))
    minimum_primary_ratings = int(schema.get("minimum_primary_ratings", expected_raters))
    minimum_sensitivity_ratings = int(
        schema.get("minimum_sensitivity_ratings", max(2, expected_raters - 1))
    )
    input_paths: list[Path] = [source]

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

    item_coverage, design_summary = rating_design_coverage(
        ratings, expected_raters=expected_raters
    )
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
        and design_summary["design_status"].eq(
            "agreement_estimable_on_complete_subset"
        ).all()
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
    rater_marginals["proportion_within_category_rater"] = rater_marginals["n"] / rater_marginals.groupby(
        ["category", "rater_id"]
    )["n"].transform("sum")
    consensus_distribution = (
        consensus.groupby(["category", "consensus_rating"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    consensus_distribution["proportion_within_category"] = consensus_distribution["n"] / consensus_distribution.groupby(
        "category"
    )["n"].transform("sum")
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
    family_indices, index_audit = direction_oriented_family_indices(features)
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
            normalized = (
                features[column].astype("string").str.strip().str.upper().map(value_map)
            )
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
            "qadd_nonspeech_level_dbfs",
            "qadd_snr_proxy_db",
            "qadd_nonspeech_variability_db",
            "qadd_hum_prominence_db",
            "qadd_transient_rate_per_min",
        ],
        "volume_instability": [
            "qgain_level_iqr_db",
            "qgain_segment_sd_db",
            "qgain_abs_drift_db_per_min",
        ],
        "poor_audio_quality": [
            "qadd_snr_proxy_db",
            "qgain_level_iqr_db",
            "qrev_tail_excess_db",
            "qdist_hard_clip_sample_fraction",
            "qtemp_zero_dropout_rate_per_min",
        ],
        "another_person_speaks": [
            "qadd_nonspeech_level_dbfs",
            "qadd_snr_proxy_db",
            "qadd_transient_rate_per_min",
        ],
    }
    links = perceptual_links(
        data,
        labels,
        mapping,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
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
    summary = describe_metrics(
        features,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
    )
    _write_table(summary, output / "metric_descriptive_statistics")
    correlations = pairwise_clustered_spearman(
        features,
        minimum_participants=cfg["analysis"]["minimum_complete_participants"],
        bootstrap_replicates=min(500, cfg["analysis"]["bootstrap_replicates"]),
        seed=cfg["project"]["random_seed"],
    )
    _write_table(correlations, output / "pairwise_clustered_spearman")
    persistence = participant_persistence(
        features,
        minimum_participants=cfg["analysis"]["minimum_complete_participants"],
    )
    _write_table(persistence, output / "participant_persistence_not_reliability")
    contrasts = participant_level_group_contrasts(
        features,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
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
    registry = metric_registry_frame()
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
        for feature in registry["feature"]:
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
                    "feature": feature,
                    "n_primary": int(left.notna().sum()),
                    "n_sensitivity": int(right.notna().sum()),
                    "n_complete_pair": int(complete.sum()),
                    "spearman_rho": rho,
                    "median_signed_difference_sensitivity_minus_primary": float((right[complete] - left[complete]).median())
                    if complete.any()
                    else float("nan"),
                    "median_absolute_difference": float((right[complete] - left[complete]).abs().median())
                    if complete.any()
                    else float("nan"),
                    "primary_only_recordings": int((joined["_merge"] == "left_only").sum()),
                    "sensitivity_only_recordings": int((joined["_merge"] == "right_only").sum()),
                }
            )
    _write_table(pd.DataFrame(rows), output / "segmentation_profile_robustness")
    assembled = _read_existing(root / "03_dataset_assembly", "paper1_analysis_dataset")
    assembled = assembled.loc[assembled["primary_measurement_eligible"].fillna(False)].copy()
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
    ).add_suffix("_full")
    one_summary = describe_metrics(
        one_per_participant,
        bootstrap_replicates=cfg["analysis"]["bootstrap_replicates"],
        seed=cfg["project"]["random_seed"],
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
    inventory.add_argument("--no-hashes", action="store_true", help="Skip SHA-256 only for a fast dry run")
    subparsers.add_parser("segment", help="Run version-pinned Silero and create distinct segmentation views")
    extract = subparsers.add_parser("extract", help="Extract registered Q metrics from native and VAD audio views")
    extract.add_argument("--profile", default="primary", help="primary, conservative, or permissive")
    subparsers.add_parser("assemble", help="Merge audited metadata and metrics with explicit eligibility gates")
    subparsers.add_parser("rest-reference", help="Extract exact-session Rest context without speech VAD")
    subparsers.add_parser("encoding-sensitivity", help="Re-extract paired WAV/WEBM technical replicates")
    subparsers.add_parser("describe", help="Participant-clustered descriptive inference")
    subparsers.add_parser("sensitivity", help="Compare conservative/permissive segmentation profiles")
    subparsers.add_parser(
        "broad-qc",
        help="Legacy analysis of merged broad metadata QC; Goal 4 is run by human-qc",
    )
    human = subparsers.add_parser(
        "human-qc",
        help="Audit four-RA interval annotations and run family-level 4RA/2RA alignment",
    )
    human.add_argument("--schema", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    commands = {
        "audit": lambda: command_audit(cfg),
        "inventory": lambda: command_inventory(cfg, hashes=not args.no_hashes),
        "segment": lambda: command_segment(cfg),
        "extract": lambda: command_extract(cfg, profile=args.profile),
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
        print(json.dumps({"status": "failed", "command": args.command, "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
