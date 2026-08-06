"""Automated, reviewer-free finalization and measurement freeze for QDIST v4.1.

The workflow keeps the scientific claim narrow: accepted hard-plateau morphology
in the stored native decoded waveform.  It never creates human or AI morphology
labels.  Exact altered-sample masks on cohort-derived speech are converted to
reference sample burden, 30-ms frame occupancy, and 20-ms-merged reference
episodes.  Retained feature roles are then decided by prespecified numerical
gates.  A failed event-level gate demotes event rate instead of blocking the
validated primary sample-burden measurement.

Two states are deliberately separated:

* measurement freeze: immutable implementation, features, roles, evidence;
* publication integration: manuscript wording/census and empirical overlap with
  the remaining families.

The first can be completed here.  The second remains explicitly pending and is
never represented as complete.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

from paper1_qc import qdist_v410_candidate as detector
from paper1_qc_reviewed import qdist_v410_cohort as cohort


FINALIZATION_VERSION = "qdist-v4.1.0-freeze-readiness-r1"
MEASUREMENT_VERSION = "qdist-v4.1.0"
OUTPUT_DIRECTORY = "freeze_readiness_v1"
FREEZE_RELATIVE = Path(
    "MAIN outputs reviewed/06_family_freezes/nonlinear_distortion/qdist-v4.1.0"
)
REQUIRED_VERIFICATION_VERSION = "qdist-v4.1.0-computational-verification-r1"
REQUIRED_SCIENTIFIC_DECISION = (
    "ACCEPT_HARD_CLIPPING_MORPHOLOGY_MEASUREMENT_WITH_EXPLICIT_LIMITATIONS"
)
EVENT_MERGE_GAP_MS = 20.0
FRAME_LENGTH_MS = 30.0

# Prespecified before the enhanced grid is run.  These govern whether event rate
# remains secondary or is automatically demoted to conditional/audit-only.
THRESHOLDS = {
    "sample_micro_precision_min": 0.95,
    "sample_micro_recall_min": 0.85,
    "moderate_occurrence_sensitivity_min": 0.90,
    "event_micro_precision_min": 0.80,
    "event_micro_recall_min": 0.80,
    "event_median_matched_iou_min": 0.50,
    "event_rate_spearman_min": 0.80,
    "moderate_cell_event_f1_median_min": 0.70,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
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


def save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)
    return target


def save_table_bundle(frame: pd.DataFrame, stem: str | Path) -> dict[str, str]:
    stem = Path(stem)
    result = {"csv": str(save_csv(frame, stem.with_suffix(".csv")))}
    parquet = stem.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet, index=False)
        result["parquet"] = str(parquet)
    except Exception:
        pass
    return result


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y", "pass", "passed"}
    )


def portable_path(value: str | Path) -> Path:
    raw = str(value)
    return Path(*PureWindowsPath(raw).parts) if "\\" in raw else Path(raw)


def true_runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.any():
        return []
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def merged_mask_episodes(
    mask: np.ndarray,
    *,
    gap_samples: int,
) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim == 1:
        values = values[:, None]
    intervals: list[tuple[int, int]] = []
    for channel in range(values.shape[1]):
        intervals.extend(true_runs(values[:, channel]))
    return cohort.merge_intervals(intervals, gap_samples=gap_samples)


def episode_intervals(episode_ledger: pd.DataFrame) -> list[tuple[int, int]]:
    if episode_ledger.empty:
        return []
    return [
        (int(row.start_sample_task), int(row.end_sample_task_exclusive))
        for row in episode_ledger.itertuples(index=False)
    ]


def interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if intersection and union else 0.0


def match_events(
    reference: Sequence[tuple[int, int]],
    predicted: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    reference = list(reference)
    predicted = list(predicted)
    if not reference and not predicted:
        return {
            "reference_event_count": 0,
            "predicted_event_count": 0,
            "matched_event_count": 0,
            "event_precision": 1.0,
            "event_recall": 1.0,
            "event_f1": 1.0,
            "matched_iou_median": 1.0,
            "matched_iou_min": 1.0,
        }
    if not reference:
        return {
            "reference_event_count": 0,
            "predicted_event_count": len(predicted),
            "matched_event_count": 0,
            "event_precision": 0.0,
            "event_recall": 1.0,
            "event_f1": 0.0,
            "matched_iou_median": math.nan,
            "matched_iou_min": math.nan,
        }
    if not predicted:
        return {
            "reference_event_count": len(reference),
            "predicted_event_count": 0,
            "matched_event_count": 0,
            "event_precision": 1.0,
            "event_recall": 0.0,
            "event_f1": 0.0,
            "matched_iou_median": math.nan,
            "matched_iou_min": math.nan,
        }
    scores = np.array(
        [[interval_iou(ref, pred) for pred in predicted] for ref in reference],
        dtype=float,
    )
    ref_indices, pred_indices = linear_sum_assignment(1.0 - scores)
    matched_ious = [
        float(scores[ref_index, pred_index])
        for ref_index, pred_index in zip(ref_indices, pred_indices)
        if scores[ref_index, pred_index] > 0.0
    ]
    matched = len(matched_ious)
    precision = matched / len(predicted)
    recall = matched / len(reference)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_event_count": len(reference),
        "predicted_event_count": len(predicted),
        "matched_event_count": matched,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "matched_iou_median": float(np.median(matched_ious)) if matched_ious else math.nan,
        "matched_iou_min": float(np.min(matched_ious)) if matched_ious else math.nan,
    }


def reference_frame_fraction(
    truth_mask: np.ndarray,
    *,
    frame_length_samples: int,
    complete_frame_count: int,
) -> tuple[int, float]:
    values = np.asarray(truth_mask, dtype=bool)
    if values.ndim == 2:
        values = values.any(axis=1)
    usable = min(len(values), int(frame_length_samples) * int(complete_frame_count))
    complete = usable // int(frame_length_samples)
    if complete <= 0:
        return 0, math.nan
    frames = values[: complete * int(frame_length_samples)].reshape(
        complete, int(frame_length_samples)
    )
    affected = int(frames.any(axis=1).sum())
    return affected, affected / complete


def verify_source_and_verification(cohort_root: Path) -> dict[str, Any]:
    candidate_manifest_path = (
        cohort_root / "manifests/qdist_v410_candidate_cohort_manifest.json"
    )
    verification_root = cohort_root / "computational_verification_v1"
    verification_manifest_path = (
        verification_root / "manifests/qdist_v410_computational_verification_manifest.json"
    )
    source_audit_path = (
        verification_root / "validation/qdist_v410_source_artifact_verification.csv"
    )
    for path in (candidate_manifest_path, verification_manifest_path, source_audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    verification_manifest = json.loads(
        verification_manifest_path.read_text(encoding="utf-8")
    )
    source_audit = read_csv(source_audit_path)
    recorded_source_exact = bool(
        len(source_audit) == 923
        and as_bool(source_audit["exists"]).all()
        and as_bool(source_audit["size_matches"]).all()
        and as_bool(source_audit["sha256_matches"]).all()
        and not as_bool(source_audit["permitted_review_archive_exclusion"]).any()
    )
    candidate_artifact_manifest_path = (
        cohort_root / "manifests/qdist_v410_candidate_cohort_artifact_manifest.csv"
    )
    candidate_artifacts = read_csv(candidate_artifact_manifest_path)
    direct_failures: list[str] = []
    for row in candidate_artifacts.to_dict("records"):
        path = cohort_root / portable_path(row["relative_path"])
        if not path.is_file():
            direct_failures.append(f"missing:{row['relative_path']}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            direct_failures.append(f"size:{row['relative_path']}")
            continue
        if sha256_file(path) != str(row["sha256"]):
            direct_failures.append(f"sha256:{row['relative_path']}")
    source_exact = bool(
        recorded_source_exact
        and len(candidate_artifacts) == 923
        and not direct_failures
    )
    required = {
        "verification_version": REQUIRED_VERIFICATION_VERSION,
        "source_artifacts_exactly_verified": True,
        "source_review_archive_exclusion_count": 0,
        "human_review_performed": False,
        "human_review_required": False,
        "human_or_ai_morphology_labels_generated": False,
        "scientific_decision": REQUIRED_SCIENTIFIC_DECISION,
        "feature_values_changed": False,
        "detector_or_thresholds_changed": False,
    }
    mismatches = {
        key: {"observed": verification_manifest.get(key), "required": expected}
        for key, expected in required.items()
        if verification_manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Computational verification manifest mismatch: {mismatches}")
    if not source_exact:
        raise RuntimeError(
            "All 923 source artifacts must verify directly with no exclusions. "
            + "; ".join(direct_failures[:20])
        )
    return {
        "candidate_manifest": candidate_manifest,
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "verification_manifest": verification_manifest,
        "verification_manifest_sha256": sha256_file(verification_manifest_path),
        "source_artifact_count": len(source_audit),
        "source_artifacts_exact": source_exact,
    }


def enhanced_exact_reference_grid(
    project_root: str | Path,
    cohort_root: str | Path,
) -> pd.DataFrame:
    project_root = Path(project_root).expanduser().resolve()
    cohort_root = Path(cohort_root).expanduser().resolve()
    source_challenge = read_csv(
        cohort_root / "validation/qdist_v410_real_speech_challenge_long.csv"
    )
    if len(source_challenge) != 144:
        raise RuntimeError(
            f"Expected 144 known-truth challenge rows; found {len(source_challenge)}."
        )
    paths = cohort.CohortPaths.from_project_root(project_root)
    frozen = cohort.load_frozen_inputs(paths)
    frozen_rows = {
        str(row["logical_recording_id"]): row
        for row in frozen["recordings"].to_dict("records")
    }
    carrier_cache: dict[str, tuple[np.ndarray, int, Any]] = {}
    rows: list[dict[str, Any]] = []
    ordered = source_challenge.sort_values(
        ["logical_recording_id", "geometry", "target_fraction"]
    )
    for source_row in ordered.to_dict("records"):
        recording_id = str(source_row["logical_recording_id"])
        if recording_id not in carrier_cache:
            waveform, fs, _probe, provenance = cohort._task_waveform(
                paths, frozen, frozen_rows[recording_id]
            )
            carrier_cache[recording_id] = (waveform, fs, provenance)
        waveform, fs, provenance = carrier_cache[recording_id]
        target = float(source_row["target_fraction"])
        geometry = str(source_row["geometry"])
        altered, truth, limits = cohort.inject_matched_hard_clip(
            waveform, target, geometry
        )
        extraction = detector.extract_qdist(
            altered,
            fs,
            logical_recording_id=(
                f"freeze_truth__{recording_id}__{geometry}__{target:.4f}"
            ),
            provenance=replace(
                provenance,
                source_path=f"known_truth_in_memory::{recording_id}",
                source_sha256=None,
                decoded_sha256=None,
            ),
        )
        detected = cohort._accepted_mask(
            extraction.accepted_plateau_ledger, altered.shape
        )
        sample = cohort._truth_metrics(truth, detected)
        gap_samples = int(round(EVENT_MERGE_GAP_MS * fs / 1000.0))
        reference_events = merged_mask_episodes(truth, gap_samples=gap_samples)
        predicted_events = episode_intervals(extraction.episode_ledger)
        event = match_events(reference_events, predicted_events)
        duration_sec = float(extraction.recording["qdist_finite_exposure_sec"])
        reference_rate = (
            event["reference_event_count"] * 60.0 / duration_sec
            if duration_sec > 0
            else math.nan
        )
        predicted_rate = float(
            extraction.recording["qdist_hard_clip_event_rate_per_min"]
        )
        frame_count, true_frame_fraction = reference_frame_fraction(
            truth,
            frame_length_samples=int(
                extraction.recording["qdist_frame_length_samples"]
            ),
            complete_frame_count=int(
                extraction.recording["qdist_complete_frame_count"]
            ),
        )
        predicted_frame_fraction = float(
            extraction.recording["qdist_hard_clipped_frame_fraction"]
        )
        rows.append(
            {
                "logical_recording_id": recording_id,
                "participant_id": str(source_row["participant_id"]),
                "native_sample_rate_hz": fs,
                "channel_count": altered.shape[1],
                "duration_sec": duration_sec,
                "geometry": geometry,
                "target_fraction": target,
                "realized_fraction": limits["realized_fraction"],
                **sample,
                **event,
                "reference_event_rate_per_min": reference_rate,
                "predicted_event_rate_per_min": predicted_rate,
                "event_count_error": (
                    event["predicted_event_count"] - event["reference_event_count"]
                ),
                "event_rate_error_per_min": predicted_rate - reference_rate,
                "reference_affected_frame_count": frame_count,
                "reference_frame_fraction": true_frame_fraction,
                "predicted_frame_fraction": predicted_frame_fraction,
                "frame_fraction_error": predicted_frame_fraction
                - true_frame_fraction,
                "episode_merge_gap_ms": EVENT_MERGE_GAP_MS,
                "frame_length_ms": FRAME_LENGTH_MS,
                "truth_definition": (
                    "altered native channel-samples; reference episodes are exact-mask "
                    "runs merged across channels with the governed 20-ms rule"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 144 or result["logical_recording_id"].nunique() != 12:
        raise RuntimeError("Enhanced exact-reference grid is incomplete.")
    return result


def summarize_enhanced_grid(
    grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [
        "target_fraction",
        "true_positive_samples",
        "false_positive_samples",
        "false_negative_samples",
        "sample_precision",
        "sample_recall",
        "event_precision",
        "event_recall",
        "event_f1",
        "matched_iou_median",
        "reference_event_count",
        "predicted_event_count",
        "reference_event_rate_per_min",
        "predicted_event_rate_per_min",
        "event_count_error",
        "event_rate_error_per_min",
        "reference_frame_fraction",
        "predicted_frame_fraction",
        "frame_fraction_error",
    ]
    work = grid.copy()
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (geometry, dose), group in work.groupby(
        ["geometry", "target_fraction"], sort=True
    ):
        sample_tp = int(group["true_positive_samples"].sum())
        sample_fp = int(group["false_positive_samples"].sum())
        sample_fn = int(group["false_negative_samples"].sum())
        matched = int(group["matched_event_count"].sum())
        predicted = int(group["predicted_event_count"].sum())
        reference = int(group["reference_event_count"].sum())
        event_precision = matched / predicted if predicted else 1.0
        event_recall = matched / reference if reference else 1.0
        rows.append(
            {
                "geometry": geometry,
                "target_fraction": float(dose),
                "carrier_count": group["logical_recording_id"].nunique(),
                "occurrence_sensitivity": float(as_bool(group["occurrence_detected"]).mean()),
                "sample_micro_precision": sample_tp / (sample_tp + sample_fp),
                "sample_micro_recall": sample_tp / (sample_tp + sample_fn),
                "event_micro_precision": event_precision,
                "event_micro_recall": event_recall,
                "event_f1_median": float(group["event_f1"].median()),
                "matched_iou_median": float(group["matched_iou_median"].median()),
                "event_count_error_median": float(group["event_count_error"].median()),
                "event_count_absolute_error_median": float(
                    group["event_count_error"].abs().median()
                ),
                "event_rate_error_per_min_median": float(
                    group["event_rate_error_per_min"].median()
                ),
                "frame_fraction_absolute_error_median": float(
                    group["frame_fraction_error"].abs().median()
                ),
            }
        )
    by_cell = pd.DataFrame(rows)
    sample_tp = int(work["true_positive_samples"].sum())
    sample_fp = int(work["false_positive_samples"].sum())
    sample_fn = int(work["false_negative_samples"].sum())
    matched = int(work["matched_event_count"].sum())
    predicted = int(work["predicted_event_count"].sum())
    reference = int(work["reference_event_count"].sum())
    finite_rate = work[
        ["reference_event_rate_per_min", "predicted_event_rate_per_min"]
    ].dropna()
    rate_spearman = (
        float(stats.spearmanr(finite_rate.iloc[:, 0], finite_rate.iloc[:, 1]).statistic)
        if len(finite_rate) >= 3
        and finite_rate.iloc[:, 0].nunique() > 1
        and finite_rate.iloc[:, 1].nunique() > 1
        else math.nan
    )
    summary = pd.DataFrame(
        [
            {
                "challenge_rows": len(work),
                "carrier_count": work["logical_recording_id"].nunique(),
                "geometry_count": work["geometry"].nunique(),
                "dose_count": work["target_fraction"].nunique(),
                "sample_micro_precision": sample_tp / (sample_tp + sample_fp),
                "sample_micro_recall": sample_tp / (sample_tp + sample_fn),
                "reference_event_count": reference,
                "predicted_event_count": predicted,
                "matched_event_count": matched,
                "event_micro_precision": matched / predicted if predicted else 1.0,
                "event_micro_recall": matched / reference if reference else 1.0,
                "event_f1_median": float(work["event_f1"].median()),
                "matched_iou_median": float(work["matched_iou_median"].median()),
                "event_rate_spearman": rate_spearman,
                "event_count_absolute_error_median": float(
                    work["event_count_error"].abs().median()
                ),
                "frame_fraction_absolute_error_median": float(
                    work["frame_fraction_error"].abs().median()
                ),
                "frame_fraction_absolute_error_max": float(
                    work["frame_fraction_error"].abs().max()
                ),
            }
        ]
    )
    return by_cell, summary


def decide_roles(
    by_cell: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    global_row = summary.iloc[0]
    moderate = by_cell.loc[by_cell["target_fraction"].ge(0.001)]
    sample_pass = bool(
        float(global_row["sample_micro_precision"])
        >= THRESHOLDS["sample_micro_precision_min"]
        and float(global_row["sample_micro_recall"])
        >= THRESHOLDS["sample_micro_recall_min"]
    )
    occurrence_pass = bool(
        len(moderate)
        and moderate["occurrence_sensitivity"].ge(
            THRESHOLDS["moderate_occurrence_sensitivity_min"]
        ).all()
    )
    event_tests = {
        "event micro precision": float(global_row["event_micro_precision"])
        >= THRESHOLDS["event_micro_precision_min"],
        "event micro recall": float(global_row["event_micro_recall"])
        >= THRESHOLDS["event_micro_recall_min"],
        "matched-event temporal IoU": float(global_row["matched_iou_median"])
        >= THRESHOLDS["event_median_matched_iou_min"],
        "event-rate monotonic agreement": float(global_row["event_rate_spearman"])
        >= THRESHOLDS["event_rate_spearman_min"],
        "moderate-cell event F1": bool(
            len(moderate)
            and moderate["event_f1_median"].ge(
                THRESHOLDS["moderate_cell_event_f1_median_min"]
            ).all()
        ),
    }
    event_pass = all(event_tests.values())
    checks = pd.DataFrame(
        [
            {
                "gate": "G0",
                "check": "completed cohort source is exact",
                "status": "PASS",
                "observed": "923/923 exact; exclusions=0",
                "required": "all source artifacts exact; no exclusions",
            },
            {
                "gate": "G9-SAMPLE",
                "check": "exact-mask primary sample burden",
                "status": "PASS" if sample_pass else "FAIL",
                "observed": (
                    f"micro precision={global_row['sample_micro_precision']:.6f}; "
                    f"micro recall={global_row['sample_micro_recall']:.6f}"
                ),
                "required": (
                    f"precision>={THRESHOLDS['sample_micro_precision_min']:.2f}; "
                    f"recall>={THRESHOLDS['sample_micro_recall_min']:.2f}"
                ),
            },
            {
                "gate": "G9-STATUS",
                "check": "moderate-dose occurrence sensitivity",
                "status": "PASS" if occurrence_pass else "FAIL",
                "observed": f"minimum cell={moderate['occurrence_sensitivity'].min():.6f}",
                "required": f">={THRESHOLDS['moderate_occurrence_sensitivity_min']:.2f}",
            },
            {
                "gate": "G9-EVENT",
                "check": "exact-reference event-rate qualification",
                "status": "PASS" if event_pass else "CONDITIONAL_DEMOTION",
                "observed": json.dumps(
                    {
                        "event_micro_precision": float(global_row["event_micro_precision"]),
                        "event_micro_recall": float(global_row["event_micro_recall"]),
                        "matched_iou_median": float(global_row["matched_iou_median"]),
                        "event_rate_spearman": float(global_row["event_rate_spearman"]),
                        "component_tests": event_tests,
                    },
                    sort_keys=True,
                ),
                "required": "all prespecified event qualification components",
            },
            {
                "gate": "G9-FRAME",
                "check": "exact-reference 30-ms frame occupancy quantified",
                "status": "PASS_WITH_CONDITIONAL_ROLE",
                "observed": (
                    f"median absolute error={global_row['frame_fraction_absolute_error_median']:.8g}; "
                    f"maximum={global_row['frame_fraction_absolute_error_max']:.8g}"
                ),
                "required": "error reported; role remains audit-only because grid-origin dependence is established",
            },
            {
                "gate": "G9-HUMAN",
                "check": "human or AI morphology labels",
                "status": "N/A",
                "observed": "none generated or required",
                "required": "exact altered-mask criterion; real detections remain operational",
            },
            {
                "gate": "G10",
                "check": "automatic feature-role decision complete",
                "status": "PASS" if sample_pass and occurrence_pass else "FAIL",
                "observed": (
                    "event rate retained secondary"
                    if event_pass
                    else "event rate automatically demoted to conditional/audit"
                ),
                "required": "primary and companion status pass; event feature is retained or demoted deterministically",
            },
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "feature": "qdist_hard_clipped_sample_fraction",
                "display_name": "Accepted hard-plateau channel-sample fraction",
                "final_role": "PRIMARY",
                "decision": "RETAIN",
                "model_default": True,
                "analysis_use": "main QDIST burden view",
                "unit": "fraction of finite native channel-samples",
                "rationale": "Exact altered-mask sample precision and recall pass; invariant to frame origin and episode merge gap.",
                "permitted_interpretation": "conservative accepted hard-plateau support in the stored native decoded waveform",
                "prohibited_interpretation": "unbiased fraction of all physically clipped samples; causal stage; complete nonlinear distortion",
            },
            {
                "feature": "qdist_hard_clip_event_rate_per_min",
                "display_name": "Accepted hard-plateau episode rate",
                "final_role": "SECONDARY" if event_pass else "CONDITIONAL_AUDIT",
                "decision": "RETAIN" if event_pass else "RETAIN_CONDITIONALLY",
                "model_default": bool(event_pass),
                "analysis_use": (
                    "secondary temporal-occurrence view"
                    if event_pass
                    else "audit/sensitivity only; exclude from default models"
                ),
                "unit": "20-ms-merged accepted episodes per finite exposure minute",
                "rationale": (
                    "Exact-mask reference episodes pass prespecified event-level qualification."
                    if event_pass
                    else "One or more exact-reference event-level gates failed; deterministic demotion prevents overclaiming."
                ),
                "permitted_interpretation": "detector-defined episode rate conditional on the governed 20-ms merge rule",
                "prohibited_interpretation": "physical device-state event count or disease-independent temporal rate",
            },
            {
                "feature": "qdist_hard_clipped_frame_fraction",
                "display_name": "Accepted hard-plateau 30-ms frame occupancy",
                "final_role": "CONDITIONAL_AUDIT",
                "decision": "RETAIN_CONDITIONALLY",
                "model_default": False,
                "analysis_use": "audit/legacy compatibility only; exclude from default models",
                "unit": "fraction of complete 30-ms frames intersecting accepted plateaus",
                "rationale": "Exact frame reference is reported, but grid-origin dependence and redundancy prevent primary use.",
                "permitted_interpretation": "occupancy on the explicitly declared 30-ms grid",
                "prohibited_interpretation": "frame-origin-invariant burden or independent biomarker",
            },
            {
                "feature": "qdist_occurrence",
                "display_name": "QDIST operational occurrence",
                "final_role": "COMPANION_STATUS",
                "decision": "RETAIN_AS_STATUS",
                "model_default": False,
                "analysis_use": "status and zero-inflation summaries; not an independent feature",
                "unit": "binary occurrence when QDIST is available",
                "rationale": "Moderate-dose occurrence sensitivity passes and structural zeros remain explicit.",
                "permitted_interpretation": "at least one detector-defined QDIST episode was observed",
                "prohibited_interpretation": "recording acceptability, diagnosis, or proof that no distortion exists",
            },
        ]
    )
    measurement_freeze_allowed = bool(sample_pass and occurrence_pass)
    return checks, decisions, measurement_freeze_allowed


def cross_family_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "other_family": "QGAIN",
                "overlap": "excessive level or gain may precede hard-plateau morphology",
                "arbitration_rule": "co-report; neither family suppresses the other; do not infer causal direction",
                "qdist_guard": "QDIST requires accepted plateau morphology, not level alone",
                "integration_status": "contract_defined_empirical_joint_audit_pending",
            },
            {
                "other_family": "QCHAN",
                "overlap": "device, codec, or channel processing may erase or create plateau-like morphology",
                "arbitration_rule": "co-report with native codec/channel provenance; no device-stage attribution",
                "qdist_guard": "lossy re-encoding sensitivity remains a stated limitation",
                "integration_status": "contract_defined_empirical_joint_audit_pending",
            },
            {
                "other_family": "QTEMP",
                "overlap": "dropouts, splices, or glitches may include abrupt extrema",
                "arbitration_rule": "QTEMP uses QDIST intervals as a clipping guard; overlap is tagged compound, not double-attributed causally",
                "qdist_guard": "QDIST morphology remains measurable but is not interpreted as a temporal discontinuity",
                "integration_status": "contract_defined_qtemp_freeze_pending",
            },
            {
                "other_family": "QADD/QREV",
                "overlap": "noise or reverberation can alter edge prominence and detectability",
                "arbitration_rule": "co-report and test interactions later; do not residualize or suppress QDIST during extraction",
                "qdist_guard": "primary interpretation remains waveform morphology only",
                "integration_status": "contract_defined_empirical_joint_audit_pending",
            },
        ]
    )


def clean_checklist(
    source: pd.DataFrame,
    *,
    event_pass: bool,
) -> pd.DataFrame:
    result = source.copy()
    result["status"] = result["status"].astype(str)
    updates: dict[str, tuple[str, str, str]] = {
        "C4": (
            "PASS",
            "freeze_readiness_v1/tables/qdist_v410_cross_family_arbitration_contract.csv",
            "Non-exclusive observable-family arbitration rules are frozen; empirical joint-family analysis remains a publication-integration task.",
        ),
        "C5": (
            "CONDITIONAL",
            "freeze_readiness_v1/reports/QDIST_v410_MEASUREMENT_FREEZE_DECISION.md",
            "Disease/content independence is not claimed; phenotype and phonetic interactions remain mandatory downstream analyses.",
        ),
        "C6": (
            "CONDITIONAL",
            "freeze_readiness_v1/reports/QDIST_v410_CANONICAL_MANUSCRIPT_WORDING.md",
            "Frozen construct label is native-waveform hard-clipping morphology; manuscript replacement remains pending.",
        ),
        "X2": (
            "CONDITIONAL",
            "freeze_readiness_v1/tables/qdist_v410_cross_family_arbitration_contract.csv",
            "Arbitration behavior is specified; empirical overlap audit waits for the final QTEMP registry.",
        ),
        "V2": (
            "N/A",
            "freeze_readiness_v1/validation/qdist_v410_freeze_readiness_checks.csv",
            "No manual reviewers are available or required for the exact altered-mask criterion. No human or AI labels were generated.",
        ),
        "V3": (
            "PASS",
            "freeze_readiness_v1/validation/qdist_v410_enhanced_exact_reference_grid.csv",
            "Exact sample, episode, and frame references are generated deterministically on label-blind cohort-derived speech.",
        ),
        "G10": (
            "PASS",
            "freeze_readiness_v1/tables/qdist_v410_final_feature_decisions.csv",
            (
                "All roles are final; event rate passed exact-reference qualification."
                if event_pass
                else "All roles are final; event rate was automatically demoted after exact-reference qualification."
            ),
        ),
        "G11": (
            "PENDING",
            "freeze_readiness_v1/manifests/qdist_v410_freeze_readiness_manifest.json",
            "Ready for the separate atomic measurement-seal step; no independent human review is required.",
        ),
        "G12": (
            "PENDING",
            "freeze_readiness_v1/reports/QDIST_v410_CANONICAL_MANUSCRIPT_WORDING.md",
            "Publication manuscript wording and the global feature census remain downstream integration tasks.",
        ),
    }
    for item_id, (status, evidence, note) in updates.items():
        mask = result["item_id"].astype(str).eq(item_id)
        if mask.any():
            result.loc[mask, "status"] = status
            result.loc[mask, "evidence_path_notes"] = evidence
            result.loc[mask, "reviewer_note"] = note
    stale = result["reviewer_note"].astype(str).str.contains(
        "independent review|required.*reviewer|two independent", case=False, regex=True
    )
    if stale.any():
        raise RuntimeError(
            "Checklist still contains stale mandatory-review wording: "
            + ", ".join(result.loc[stale, "item_id"].astype(str))
        )
    return result


def build_registry(decisions: pd.DataFrame) -> pd.DataFrame:
    definitions = {item.name: item for item in detector.FEATURE_DEFINITIONS}
    rows: list[dict[str, Any]] = []
    for decision in decisions.to_dict("records"):
        feature = decision["feature"]
        definition = definitions.get(feature)
        rows.append(
            {
                **decision,
                "family": "QDIST",
                "family_construct": "native-waveform hard-clipping morphology",
                "measurement_version": MEASUREMENT_VERSION,
                "signal_view": "first decoded native-rate stream; native channels preserved",
                "estimand": getattr(definition, "estimand", decision["permitted_interpretation"]),
                "minimum_support": getattr(definition, "minimum_support", "QDIST available"),
                "known_confounds": getattr(definition, "known_confounds", "speech content and acquisition chain"),
                "source_ledger": getattr(definition, "source_ledger", "derived recording state"),
                "publication_status": "ready_for_measurement_freeze",
                "family_scalar_constructed": False,
                "standalone_accept_reject_allowed": False,
                "complete_nonlinear_distortion_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def build_analysis_tables(
    cohort_root: Path,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(cohort_root / "tables/qdist_v410_recording_features.csv")
    roles = decisions.set_index("feature")["final_role"].to_dict()
    base = [
        "logical_recording_id",
        "participant_id",
        "qdist_hard_clipped_sample_fraction",
        "qdist_hard_clip_event_rate_per_min",
        "qdist_hard_clipped_frame_fraction",
        "qdist_occurrence",
        "qdist_hard_clipped_sample_fraction_status",
        "qdist_hard_clip_event_rate_per_min_status",
        "qdist_hard_clipped_frame_fraction_status",
        "qdist_status",
        "qdist_available",
        "qdist_support_tier",
        "qdist_hard_clip_event_count",
        "qdist_accepted_plateau_count",
        "qdist_finite_exposure_sec",
        "qdist_hard_clip_event_rate_ci95_low_per_min",
        "qdist_hard_clip_event_rate_ci95_high_per_min",
        "qdist_native_sample_rate_hz",
        "qdist_native_channel_count",
        "qdist_codec_name",
        "qdist_parameter_hash",
        "qdist_source_sha256",
        "qdist_decoded_sha256",
    ]
    analysis = source[base].copy()
    analysis["qdist_measurement_version"] = MEASUREMENT_VERSION
    analysis["qdist_primary_feature"] = "qdist_hard_clipped_sample_fraction"
    analysis["qdist_event_feature_role"] = roles[
        "qdist_hard_clip_event_rate_per_min"
    ]
    analysis["qdist_frame_feature_role"] = roles[
        "qdist_hard_clipped_frame_fraction"
    ]
    analysis["qdist_occurrence_role"] = roles["qdist_occurrence"]
    analysis["qdist_family_scalar_constructed"] = False
    analysis["qdist_missing_values_imputed"] = False
    model_features = ["qdist_hard_clipped_sample_fraction"]
    if bool(
        decisions.loc[
            decisions["feature"].eq("qdist_hard_clip_event_rate_per_min"),
            "model_default",
        ].iloc[0]
    ):
        model_features.append("qdist_hard_clip_event_rate_per_min")
    model_columns = [
        "logical_recording_id",
        "participant_id",
        *model_features,
        "qdist_occurrence",
        "qdist_available",
        "qdist_status",
        "qdist_support_tier",
        "qdist_finite_exposure_sec",
        "qdist_parameter_hash",
        "qdist_source_sha256",
        "qdist_decoded_sha256",
    ]
    model = analysis[model_columns].copy()
    model["qdist_model_default_features"] = "|".join(model_features)
    model["qdist_occurrence_is_status_not_feature"] = True
    model["qdist_standalone_accept_reject_allowed"] = False
    model["qdist_complete_nonlinear_distortion_claim_allowed"] = False
    return analysis, model


def save_bundle(
    fig: plt.Figure,
    output: Path,
    stem: str,
    *,
    source: pd.DataFrame,
    caption: str,
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    png = output / f"{stem}.png"
    svg = output / f"{stem}.svg"
    pdf = output / f"{stem}.pdf"
    source_csv = output / f"{stem}.source.csv"
    caption_path = output / f"{stem}.caption.md"
    provenance_path = output / f"{stem}.provenance.json"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    save_csv(source, source_csv)
    caption_path.write_text(caption.strip() + "\n", encoding="utf-8")
    write_json(provenance, provenance_path)
    return {
        "png": png.as_posix(),
        "svg": svg.as_posix(),
        "pdf": pdf.as_posix(),
        "source_csv": source_csv.as_posix(),
        "caption": caption_path.as_posix(),
        "provenance": provenance_path.as_posix(),
    }


def build_final_figures(
    output: Path,
    by_cell: pd.DataFrame,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    figures = output / "figures"
    provenance = {
        "measurement_version": MEASUREMENT_VERSION,
        "finalization_version": FINALIZATION_VERSION,
        "human_review_performed": False,
        "criterion_reference": "exact altered-sample mask on cohort-derived speech",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(by_cell))
    labels = [
        f"{geometry.replace('_only', '')}\n{dose:g}"
        for geometry, dose in zip(by_cell["geometry"], by_cell["target_fraction"])
    ]
    axes[0].plot(x, by_cell["event_micro_precision"], "o", label="precision")
    axes[0].plot(x, by_cell["event_micro_recall"], "s", label="recall")
    axes[0].set_xticks(x, labels, rotation=55, ha="right")
    axes[0].set_ylim(0, 1.03)
    axes[0].set(title="Exact-reference event matching", ylabel="Event metric")
    axes[0].legend()
    axes[1].scatter(
        by_cell["target_fraction"],
        by_cell["event_f1_median"],
        c=pd.Categorical(by_cell["geometry"]).codes,
    )
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.03)
    axes[1].set(
        title="Episode qualification by burden",
        xlabel="Target altered fraction",
        ylabel="Median event F1",
    )
    axes[2].plot(
        x,
        by_cell["frame_fraction_absolute_error_median"],
        "o-",
        color="#8c564b",
    )
    axes[2].set_xticks(x, labels, rotation=55, ha="right")
    axes[2].set(
        title="Conditional 30-ms frame reference",
        ylabel="Median absolute fraction error",
    )
    fig.suptitle("K. Exact episode and frame reference on cohort-derived speech")
    fig.tight_layout()
    paths_k = save_bundle(
        fig,
        figures,
        "qdist_v410_panel-K_exact-event-frame-reference",
        source=pd.concat(
            [
                by_cell.assign(source_section="geometry_dose"),
                summary.assign(source_section="global"),
            ],
            ignore_index=True,
            sort=False,
        ),
        caption=(
            "Reviewer-free exact-mask qualification of QDIST episode rate and the "
            "conditional 30-ms frame view. Reference episodes are altered-mask runs "
            "merged with the same governed 20-ms rule used by the detector."
        ),
        provenance={**provenance, "panel": "K"},
    )
    fig, axis = plt.subplots(figsize=(13, 5.2))
    axis.axis("off")
    visible = decisions[
        ["display_name", "final_role", "decision", "analysis_use"]
    ].copy()
    table = axis.table(
        cellText=visible.values,
        colLabels=["Output", "Final role", "Decision", "Permitted analysis use"],
        cellLoc="left",
        loc="center",
        colWidths=[0.27, 0.17, 0.18, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.0)
    axis.set_title(
        "J. Final QDIST v4.1 measurement roles (publication integration remains separate)",
        pad=16,
    )
    fig.tight_layout()
    paths_j = save_bundle(
        fig,
        figures,
        "qdist_v410_panel-J_final-feature-decisions-v2",
        source=decisions,
        caption=(
            "Final reviewer-free QDIST feature decisions after exact sample, episode, "
            "and frame reference validation. Occurrence is a companion status, not an "
            "independent feature."
        ),
        provenance={**provenance, "panel": "J"},
    )
    return pd.DataFrame(
        [
            {"panel": "J", "stem": "qdist_v410_panel-J_final-feature-decisions-v2", **paths_j},
            {"panel": "K", "stem": "qdist_v410_panel-K_exact-event-frame-reference", **paths_k},
        ]
    )


def canonical_manuscript_wording(decisions: pd.DataFrame) -> str:
    event = decisions.loc[
        decisions["feature"].eq("qdist_hard_clip_event_rate_per_min")
    ].iloc[0]
    return f"""# QDIST v4.1 canonical manuscript wording

## Construct label

Use **native-waveform hard-clipping morphology**. Do not use QDIST as a synonym
for complete nonlinear distortion.

## Methods wording

QDIST quantified detector-accepted hard-plateau morphology in the first decoded
native-rate audio stream while preserving native channels. The primary measure
was the fraction of finite native channel-samples covered by accepted plateaus.
The detector-defined 20-ms-merged episode rate was {str(event['decision']).lower()}
as a {str(event['final_role']).lower().replace('_', ' ')} view. The 30-ms frame
occupancy was retained only for audit/legacy compatibility because it depends on
frame-grid origin. A binary occurrence variable was retained as companion status
and was not counted as an independent feature.

## Validation wording

On 144 exact-mask interventions applied to 12 label-blind cohort-derived speech
recordings across three clipping geometries and four burdens, sample-, episode-,
and frame-reference behavior was evaluated computationally. No manual reviewer
or AI morphology labels were generated. Real-cohort positives are therefore
reported as operational detections, not human-confirmed physical clipping.

## Prohibited wording

Do not claim total harmonic distortion, complete nonlinear distortion, soft
clipping, compression, limiting, AGC/DRC, codec distortion, causal device-stage
localization, disease independence, or standalone recording acceptability.

## Feature census

- One primary QDIST analysis feature: `qdist_hard_clipped_sample_fraction`.
- Event-rate role: `{event['final_role']}` / `{event['decision']}`.
- One conditional audit view: `qdist_hard_clipped_frame_fraction`.
- One companion status: `qdist_occurrence`, not counted as an independent feature.
"""


def decision_report(
    summary: pd.DataFrame,
    checks: pd.DataFrame,
    decisions: pd.DataFrame,
) -> str:
    row = summary.iloc[0]
    event = decisions.loc[
        decisions["feature"].eq("qdist_hard_clip_event_rate_per_min")
    ].iloc[0]
    check_lines = ["| Gate | Check | Status |", "|---|---|---|"]
    for item in checks[["gate", "check", "status"]].to_dict("records"):
        check_lines.append(
            f"| {item['gate']} | {item['check']} | {item['status']} |"
        )
    check_table = "\n".join(check_lines)
    return f"""# QDIST v4.1 automated measurement-freeze decision

## Decision

**ACCEPT FOR IMMUTABLE MEASUREMENT FREEZE WITH EXPLICIT SCOPE LIMITS.**

The frozen construct is accepted hard-plateau morphology in the stored native
decoded waveform. This is not a complete nonlinear-distortion measure and does
not identify the causal acquisition stage.

## Exact-reference evidence

- Challenge rows: {int(row['challenge_rows'])}
- Carriers: {int(row['carrier_count'])}
- Sample micro precision: {row['sample_micro_precision']:.6f}
- Sample micro recall: {row['sample_micro_recall']:.6f}
- Reference episodes: {int(row['reference_event_count'])}
- Predicted episodes: {int(row['predicted_event_count'])}
- Event micro precision: {row['event_micro_precision']:.6f}
- Event micro recall: {row['event_micro_recall']:.6f}
- Median matched-event IoU: {row['matched_iou_median']:.6f}
- Event-rate Spearman agreement: {row['event_rate_spearman']:.6f}
- Event-rate decision: {event['decision']} ({event['final_role']})

## Reviewer policy

Manual review was not performed and is not required for this sample-defined
criterion. No reviewer forms or AI morphology labels were generated. Real-cohort
detections remain operational detections.

## Freeze boundary

The measurement implementation, parameters, input view, feature roles, tables,
figures, and computational evidence may be frozen. Manuscript reconciliation,
global feature census, phenotype/content interaction analysis, and empirical
cross-family overlap analysis remain explicit publication-integration tasks.

## Automatic gate status

{check_table}
"""


def feature_passports(decisions: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for row in decisions.to_dict("records"):
        text = f"""# {row['feature']}

- Family: QDIST — native-waveform hard-clipping morphology
- Measurement version: {MEASUREMENT_VERSION}
- Final role: {row['final_role']}
- Decision: {row['decision']}
- Default model input: {bool(row['model_default'])}
- Unit: {row['unit']}
- Analysis use: {row['analysis_use']}
- Rationale: {row['rationale']}
- Permitted interpretation: {row['permitted_interpretation']}
- Prohibited interpretation: {row['prohibited_interpretation']}
- Human/AI morphology labels: none
- Real-cohort truth status: unlabeled operational detections
- Missingness: never imputed to zero; availability and support must accompany use
"""
        (output / f"{row['feature']}.md").write_text(text, encoding="utf-8")


def build_artifact_manifest(root: Path, name: str) -> Path:
    manifest = root / "manifests" / name
    excluded = {manifest.relative_to(root).as_posix()}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    save_csv(pd.DataFrame(rows), manifest)
    return manifest


def verify_artifact_manifest(root: Path, name: str) -> None:
    manifest = root / "manifests" / name
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = read_csv(manifest)
    failures: list[str] = []
    for row in rows.to_dict("records"):
        path = root / portable_path(row["relative_path"])
        if not path.is_file():
            failures.append(f"missing:{row['relative_path']}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"size:{row['relative_path']}")
        if sha256_file(path) != str(row["sha256"]):
            failures.append(f"sha256:{row['relative_path']}")
    if failures:
        raise RuntimeError(
            "Artifact-manifest verification failed: " + "; ".join(failures[:20])
        )


def load_existing_finalization(output: Path) -> dict[str, Any] | None:
    manifest_path = output / "manifests/qdist_v410_freeze_readiness_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("finalization_version") != FINALIZATION_VERSION
        or manifest.get("measurement_freeze_allowed") is not True
        or manifest.get("freeze_status") != "ready_for_atomic_measurement_freeze"
    ):
        return None
    verify_artifact_manifest(
        output, "qdist_v410_freeze_readiness_artifact_manifest.csv"
    )
    return {
        "output_root": output,
        "manifest": manifest,
        "checks": read_csv(
            output / "validation/qdist_v410_freeze_readiness_checks.csv"
        ),
        "decisions": read_csv(
            output / "tables/qdist_v410_final_feature_decisions.csv"
        ),
    }


def finalize_measurement(
    project_root: str | Path,
    *,
    reuse_complete: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    paths = cohort.CohortPaths.from_project_root(project_root)
    cohort_root = paths.output_root
    output = cohort_root / OUTPUT_DIRECTORY
    if output.exists():
        existing = load_existing_finalization(output) if reuse_complete else None
        if existing is not None:
            return existing
        raise FileExistsError(
            f"Incomplete or incompatible finalization directory exists: {output}"
        )
    source = verify_source_and_verification(cohort_root)
    temporary = output.with_name(f".{OUTPUT_DIRECTORY}.staging.{sha256(utc_now().encode()).hexdigest()[:12]}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        grid = enhanced_exact_reference_grid(project_root, cohort_root)
        by_cell, summary = summarize_enhanced_grid(grid)
        checks, decisions, freeze_allowed = decide_roles(by_cell, summary)
        if not freeze_allowed or checks["status"].eq("FAIL").any():
            save_csv(checks, temporary / "validation/qdist_v410_freeze_readiness_checks.csv")
            raise RuntimeError(
                "Primary QDIST freeze gates failed; no freeze is authorized."
            )
        event_pass = bool(
            decisions.loc[
                decisions["feature"].eq("qdist_hard_clip_event_rate_per_min"),
                "decision",
            ].iloc[0]
            == "RETAIN"
        )
        verification_checklist = read_csv(
            cohort_root
            / "computational_verification_v1/validation/QDIST_Master_Validation_Checklist_v1_3_COMPUTATIONAL_VERIFICATION.csv"
        )
        checklist = clean_checklist(verification_checklist, event_pass=event_pass)
        arbitration = cross_family_contract()
        registry = build_registry(decisions)
        analysis, model = build_analysis_tables(cohort_root, decisions)
        save_table_bundle(
            grid,
            temporary / "validation/qdist_v410_enhanced_exact_reference_grid",
        )
        save_table_bundle(
            by_cell,
            temporary / "validation/qdist_v410_enhanced_exact_reference_by_cell",
        )
        save_csv(
            summary,
            temporary / "validation/qdist_v410_enhanced_exact_reference_summary.csv",
        )
        save_csv(
            checks,
            temporary / "validation/qdist_v410_freeze_readiness_checks.csv",
        )
        save_csv(
            checklist,
            temporary / "validation/QDIST_Master_Validation_Checklist_v1_4_MEASUREMENT_FREEZE_READY.csv",
        )
        save_csv(
            decisions,
            temporary / "tables/qdist_v410_final_feature_decisions.csv",
        )
        save_table_bundle(
            registry,
            temporary / "tables/qdist_v410_feature_registry",
        )
        save_table_bundle(
            analysis,
            temporary / "tables/qdist_v410_analysis_features",
        )
        save_table_bundle(
            model,
            temporary / "tables/qdist_v410_model_interface",
        )
        save_csv(
            arbitration,
            temporary / "tables/qdist_v410_cross_family_arbitration_contract.csv",
        )
        figure_index = build_final_figures(temporary, by_cell, summary, decisions)
        save_csv(
            figure_index,
            temporary / "tables/qdist_v410_finalization_figure_index.csv",
        )
        reports = temporary / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "QDIST_v410_MEASUREMENT_FREEZE_DECISION.md").write_text(
            decision_report(summary, checks, decisions), encoding="utf-8"
        )
        (reports / "QDIST_v410_CANONICAL_MANUSCRIPT_WORDING.md").write_text(
            canonical_manuscript_wording(decisions), encoding="utf-8"
        )
        feature_passports(decisions, temporary / "feature_passports")
        manifest = {
            "measurement_version": MEASUREMENT_VERSION,
            "finalization_version": FINALIZATION_VERSION,
            "created_utc": utc_now(),
            "freeze_status": "ready_for_atomic_measurement_freeze",
            "measurement_freeze_allowed": True,
            "publication_integration_complete": False,
            "standalone_publication_freeze_allowed": False,
            "manuscript_reconciliation_complete": False,
            "cross_family_arbitration_contract_defined": True,
            "cross_family_empirical_audit_complete": False,
            "qtemp_final_registry_available": False,
            "source_artifacts_exactly_verified": source["source_artifacts_exact"],
            "source_artifact_count": source["source_artifact_count"],
            "source_candidate_manifest_sha256": source["candidate_manifest_sha256"],
            "source_verification_manifest_sha256": source[
                "verification_manifest_sha256"
            ],
            "detector_or_thresholds_changed": False,
            "cohort_feature_values_changed": False,
            "human_review_performed": False,
            "human_review_required": False,
            "human_or_ai_morphology_labels_generated": False,
            "real_cohort_ground_truth_status": "unlabeled_operational_detections",
            "criterion_reference": "exact altered-sample mask on cohort-derived speech",
            "challenge_rows": int(summary.iloc[0]["challenge_rows"]),
            "sample_micro_precision": float(
                summary.iloc[0]["sample_micro_precision"]
            ),
            "sample_micro_recall": float(summary.iloc[0]["sample_micro_recall"]),
            "event_micro_precision": float(
                summary.iloc[0]["event_micro_precision"]
            ),
            "event_micro_recall": float(summary.iloc[0]["event_micro_recall"]),
            "event_rate_spearman": float(
                summary.iloc[0]["event_rate_spearman"]
            ),
            "event_rate_final_decision": str(
                decisions.loc[
                    decisions["feature"].eq(
                        "qdist_hard_clip_event_rate_per_min"
                    ),
                    "decision",
                ].iloc[0]
            ),
            "feature_decisions_complete": True,
            "primary_feature": "qdist_hard_clipped_sample_fraction",
            "conditional_frame_view": "qdist_hard_clipped_frame_fraction",
            "companion_status": "qdist_occurrence",
            "family_scalar_constructed": False,
            "standalone_accept_reject_allowed": False,
            "complete_nonlinear_distortion_claim_allowed": False,
            "thresholds": THRESHOLDS,
        }
        write_json(
            manifest,
            temporary
            / "manifests/qdist_v410_freeze_readiness_manifest.json",
        )
        build_artifact_manifest(
            temporary, "qdist_v410_freeze_readiness_artifact_manifest.csv"
        )
        temporary.replace(output)
        return {
            "output_root": output,
            "manifest": manifest,
            "checks": checks,
            "decisions": decisions,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def copy_selected_candidate_artifacts(cohort_root: Path, destination: Path) -> None:
    for folder in ("tables", "validation", "manifests"):
        source = cohort_root / folder
        if source.exists():
            shutil.copytree(source, destination / "source_candidate" / folder)
    audit_destination = destination / "source_candidate" / "audit"
    audit_destination.mkdir(parents=True, exist_ok=True)
    for path in (cohort_root / "audit").glob("*"):
        if path.is_file():
            shutil.copy2(path, audit_destination / path.name)
    shutil.copytree(
        cohort_root / "computational_verification_v1",
        destination / "computational_verification_v1",
    )


def provenance_files(project_root: Path) -> list[Path]:
    candidates = [
        project_root / "src/paper1_qc/qdist_v410_candidate.py",
        project_root
        / "src reviewed/paper1_qc_reviewed/qdist_v410_cohort.py",
        project_root
        / "src reviewed/paper1_qc_reviewed/qdist_v410_computational_verification.py",
        project_root
        / "src reviewed/paper1_qc_reviewed/qdist_v410_freeze_readiness.py",
        project_root / "tests reviewed/test_qdist_v410_candidate.py",
        project_root
        / "tests reviewed/test_qdist_v410_computational_verification.py",
        project_root
        / "tests reviewed/test_qdist_v410_freeze_readiness.py",
        project_root
        / "notebooks reviewed/05_QDIST/QDIST_V4_1_0_MEASUREMENT_FREEZE_CONTRACT.md",
        project_root
        / "notebooks reviewed/05_QDIST/QDIST_v410_AUTOMATED_FREEZE_PROTOCOL.md",
    ]
    return [path for path in candidates if path.is_file()]


def canonical_destinations(project_root: Path) -> dict[str, Path]:
    base = project_root / "MAIN outputs reviewed"
    return {
        "registry_csv": base / "00_feature_registry/qdist_v410_feature_registry.csv",
        "registry_parquet": base
        / "00_feature_registry/qdist_v410_feature_registry.parquet",
        "analysis_csv": base / "01_analysis_features/qdist_v410_analysis_features.csv",
        "analysis_parquet": base
        / "01_analysis_features/qdist_v410_analysis_features.parquet",
        "model_csv": base / "04_model_ready_features/qdist_v410_model_interface.csv",
        "model_parquet": base
        / "04_model_ready_features/qdist_v410_model_interface.parquet",
        "passports": base
        / "05_feature_passports/nonlinear_distortion/qdist-v4.1.0",
    }


def seal_measurement(
    project_root: str | Path,
    *,
    executed_notebook: str | Path,
) -> Path:
    project_root = Path(project_root).expanduser().resolve()
    executed_notebook = Path(executed_notebook).expanduser().resolve()
    if not executed_notebook.is_file():
        raise FileNotFoundError(executed_notebook)
    notebook = json.loads(executed_notebook.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
    ]
    if not code_cells or any(cell.get("execution_count") is None for cell in code_cells):
        raise RuntimeError("The finalization notebook is not fully executed.")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError(f"The finalization notebook contains errors: {errors}")
    streams = "\n".join(
        "".join(output.get("text", []))
        if isinstance(output.get("text"), list)
        else str(output.get("text", ""))
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    if "QDIST v4.1.0 MEASUREMENT FREEZE READY" not in streams:
        raise RuntimeError("Finalization completion marker is absent.")
    paths = cohort.CohortPaths.from_project_root(project_root)
    cohort_root = paths.output_root
    finalization = cohort_root / OUTPUT_DIRECTORY
    manifest_path = (
        finalization / "manifests/qdist_v410_freeze_readiness_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("measurement_freeze_allowed") is not True
        or manifest.get("freeze_status") != "ready_for_atomic_measurement_freeze"
        or manifest.get("source_artifact_count") != 923
    ):
        raise RuntimeError("Finalization manifest does not authorize measurement freeze.")
    verify_artifact_manifest(
        finalization, "qdist_v410_freeze_readiness_artifact_manifest.csv"
    )
    target = project_root / FREEZE_RELATIVE
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite immutable freeze: {target}")
    destinations = canonical_destinations(project_root)
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing partial canonical publication; destinations exist: "
            + "; ".join(existing)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".qdist-v4.1.0.staging.{sha256(utc_now().encode()).hexdigest()[:12]}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        copy_selected_candidate_artifacts(cohort_root, staging)
        shutil.copytree(finalization, staging / "finalization")
        shutil.copytree(finalization / "feature_passports", staging / "feature_passports")
        shutil.copytree(finalization / "figures", staging / "figures")
        (staging / "tables").mkdir(parents=True, exist_ok=True)
        for path in (finalization / "tables").glob("qdist_v410_*.*"):
            if path.is_file():
                shutil.copy2(path, staging / "tables" / path.name)
        (staging / "validation").mkdir(parents=True, exist_ok=True)
        for path in (finalization / "validation").glob("*"):
            if path.is_file():
                shutil.copy2(path, staging / "validation" / path.name)
        provenance = staging / "provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            executed_notebook,
            provenance / "05_nonlinear_distortion_QDIST_v4_1_0_EXECUTED_FINAL.ipynb",
        )
        for path in provenance_files(project_root):
            shutil.copy2(path, provenance / path.name)
        registry_path = staging / "tables/qdist_v410_feature_registry.csv"
        registry = pd.read_csv(registry_path)
        registry["publication_status"] = "frozen"
        save_csv(registry, registry_path)
        try:
            registry.to_parquet(
                staging / "tables/qdist_v410_feature_registry.parquet", index=False
            )
        except Exception:
            pass
        checklist_path = staging / "validation/QDIST_Master_Validation_Checklist_v1_4_MEASUREMENT_FREEZE_READY.csv"
        checklist = read_csv(checklist_path)
        g11 = checklist["item_id"].astype(str).eq("G11")
        checklist.loc[g11, "status"] = "PASS"
        checklist.loc[g11, "evidence_path_notes"] = (
            "manifests/qdist_v410_freeze_manifest.json; "
            "manifests/qdist_v410_freeze_inventory.csv"
        )
        checklist.loc[g11, "reviewer_note"] = (
            "Executed notebook, governed code/tests, final tables, figures, source evidence, and hashes are sealed immutably."
        )
        save_csv(checklist, checklist_path)
        excluded = {
            "manifests/qdist_v410_freeze_manifest.json",
            "manifests/qdist_v410_freeze_inventory.csv",
            "FROZEN_QDIST_V4_1_0.txt",
        }
        inventory_rows = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = path.relative_to(staging).as_posix()
            if relative not in excluded:
                inventory_rows.append(
                    {
                        "relative_path": relative,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        inventory_path = staging / "manifests/qdist_v410_freeze_inventory.csv"
        save_csv(pd.DataFrame(inventory_rows), inventory_path)
        freeze_manifest = {
            **manifest,
            "freeze_status": "frozen",
            "measurement_freeze_allowed": False,
            "frozen_utc": utc_now(),
            "executed_notebook_relative_path": (
                "provenance/05_nonlinear_distortion_QDIST_v4_1_0_EXECUTED_FINAL.ipynb"
            ),
            "executed_notebook_sha256": sha256_file(executed_notebook),
            "freeze_inventory_sha256": sha256_file(inventory_path),
            "artifact_count_excluding_seal_files": len(inventory_rows),
            "publication_integration_complete": False,
            "publication_integration_tasks": [
                "reconcile manuscript wording and global feature census",
                "run empirical joint-family overlap audit after QTEMP freeze",
                "test phenotype and speech-content interactions in downstream analyses",
            ],
        }
        write_json(
            freeze_manifest,
            staging / "manifests/qdist_v410_freeze_manifest.json",
        )
        (staging / "FROZEN_QDIST_V4_1_0.txt").write_text(
            "QDIST v4.1.0 measurement is frozen. Never overwrite.\n"
            "Publication integration remains explicitly pending.\n",
            encoding="utf-8",
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    sources = {
        "registry_csv": target / "tables/qdist_v410_feature_registry.csv",
        "registry_parquet": target / "tables/qdist_v410_feature_registry.parquet",
        "analysis_csv": target / "tables/qdist_v410_analysis_features.csv",
        "analysis_parquet": target / "tables/qdist_v410_analysis_features.parquet",
        "model_csv": target / "tables/qdist_v410_model_interface.csv",
        "model_parquet": target / "tables/qdist_v410_model_interface.parquet",
        "passports": target / "feature_passports",
    }
    for key, destination in destinations.items():
        source = sources[key]
        if not source.exists():
            if key.endswith("parquet"):
                continue
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--project-root", required=True)
    finalize_parser.add_argument("--no-reuse", action="store_true")
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--project-root", required=True)
    seal_parser.add_argument("--executed-notebook", required=True)
    args = parser.parse_args(argv)
    if args.command == "finalize":
        result = finalize_measurement(
            args.project_root, reuse_complete=not args.no_reuse
        )
        print(json.dumps(json_safe(result["manifest"]), indent=2, sort_keys=True))
        return 0
    target = seal_measurement(
        args.project_root, executed_notebook=args.executed_notebook
    )
    print(f"QDIST v4.1.0 MEASUREMENT FROZEN: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
