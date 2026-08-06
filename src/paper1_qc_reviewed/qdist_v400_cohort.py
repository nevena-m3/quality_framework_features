"""Standardized cohort validation for QDIST v4.0.0.

This module does not redefine the frozen qdist-v3.1.1 detector. It verifies the
immutable v3.1.1 outputs, independently reconstructs the three recording-level
features from the frozen plateau/episode ledgers, standardizes cohort evidence
under the common G1-G10/A-J framework, and regenerates a five-view event-review
package from the frozen label-blind audio excerpts.

QDIST remains evidence compatible with hard clipping or saturation only. It is
not a general nonlinear-distortion, codec, compression, limiting, AGC, or
perceptual-quality detector. The three outputs are related views of one accepted
plateau/episode system. No family scalar or standalone rejection threshold is
constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats

MEASUREMENT_VERSION = "qdist-v4.0.0-candidate"
LEGACY_MEASUREMENT_VERSION = "qdist-v3.1.1"
COHORT_ORCHESTRATION_VERSION = "qdist-v4.0.0-cohort-orchestration-v1"
ANALYSIS_FEATURES = (
    "qdist_hard_clipped_frame_fraction",
    "qdist_hard_clip_event_rate_per_min",
    "qdist_hard_clipped_sample_fraction",
)
PRIMARY_FEATURES = ANALYSIS_FEATURES[:2]
SECONDARY_FEATURES = ANALYSIS_FEATURES[2:]
REQUIRED_MAIN_PANELS = (
    "A", "B", "C", "D1", "D2", "D3", "E1", "E2", "E3", "F",
    "H1", "H2", "H3", "I", "J",
)
GALLERY_MINIMUM = 8
EVENT_REVIEW_REQUIRED_VIEWS = (
    "waveform",
    "pcm_derivative",
    "amplitude_distribution",
    "spectrogram",
    "audio_excerpt",
)


@dataclass(frozen=True)
class CohortPaths:
    project_root: Path
    frozen_root: Path
    preflight_root: Path
    output_root: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "CohortPaths":
        root = Path(project_root)
        return cls(
            project_root=root,
            frozen_root=(
                root / "MAIN outputs" / "02_FEATURE_FREEZE" /
                "nonlinear_distortion" / LEGACY_MEASUREMENT_VERSION
            ),
            preflight_root=(
                root / "outputs/reviewed" / "nonlinear_distortion" /
                MEASUREMENT_VERSION
            ),
            output_root=(
                root / "outputs/reviewed" / "nonlinear_distortion" /
                MEASUREMENT_VERSION
            ),
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [json_safe(v) for v in value]
    return value


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def save_table(
    frame: pd.DataFrame,
    path_without_suffix: str | Path,
    *,
    parquet: bool = True,
) -> dict[str, str]:
    stem = Path(path_without_suffix)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    tmp = csv_path.with_name(f".{csv_path.name}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    result = {"csv": str(csv_path)}
    if parquet:
        parquet_path = stem.with_suffix(".parquet")
        try:
            frame.to_parquet(parquet_path, index=False)
            result["parquet"] = str(parquet_path)
        except Exception:
            pass
    return result


def save_figure_bundle(
    fig: plt.Figure,
    directory: str | Path,
    stem: str,
    *,
    source: pd.DataFrame,
    caption: str,
    provenance: Mapping[str, Any],
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
    write_json(dict(provenance), paths["provenance"])
    return {key: str(value) for key, value in paths.items()}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    )


def derive_participant_id(logical_recording_id: str) -> str:
    """Derive the participant prefix from the governed recording identity.

    The frozen Bamboo identity has six rightmost underscore-delimited fields:
    protocol, visit, acquisition date, task code, PSG, and BAMBOO. Parsing from
    the right preserves participant IDs containing hyphens or underscores.
    """
    parts = str(logical_recording_id).rsplit("_", 6)
    if len(parts) != 7:
        raise ValueError(
            f"Recording identity does not match the frozen Bamboo contract: {logical_recording_id}"
        )
    return parts[0]


def derive_acquisition_date(logical_recording_id: str) -> pd.Timestamp:
    parts = str(logical_recording_id).rsplit("_", 6)
    if len(parts) != 7 or not re.fullmatch(r"\d{8}", parts[3]):
        return pd.NaT
    return pd.to_datetime(parts[3], format="%Y%m%d", errors="coerce")


def load_csv(path: str | Path, *, required: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def verify_preflight_bundle(preflight_root: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(preflight_root)
    manifest_path = root / "manifests" / "qdist_v400_preflight_candidate_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Accepted QDIST preflight manifest is missing. Run the reviewed preflight first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requirements = {
        "candidate_only": True,
        "preflight_blocking_checks_pass": True,
        "package_tests_passed": True,
        "cohort_extraction_completed": False,
        "freeze_allowed": False,
        "publish_and_freeze": False,
        "scientific_review_decision": "PENDING",
    }
    for key, expected in requirements.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"QDIST preflight manifest requirement failed: {key}={manifest.get(key)!r}; "
                f"expected {expected!r}."
            )
    if sorted(manifest.get("panels_complete", [])) != ["A", "B", "C"]:
        raise ValueError("QDIST preflight Panels A-C are incomplete.")

    rows: list[dict[str, Any]] = []
    for stem, panel in [
        ("A_construct_response", "A"),
        ("B_discriminant_specificity", "B"),
        ("C_transformation_contract", "C"),
    ]:
        row = {"panel": panel, "stem": stem}
        for suffix, key in [
            (".png", "png"),
            (".svg", "svg"),
            (".pdf", "pdf"),
            (".source.csv", "source_csv"),
            (".caption.md", "caption"),
            (".provenance.json", "provenance"),
        ]:
            path = root / "figures" / f"{stem}{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Accepted QDIST preflight artifact missing: {path}")
            row[key] = str(path)
            row[f"{key}_sha256"] = sha256_file(path)
        rows.append(row)
    return manifest, pd.DataFrame(rows)


def verify_frozen_baseline(frozen_root: str | Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    root = Path(frozen_root)
    manifest_path = root / "audit" / "qdist_v311_frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Immutable qdist-v3.1.1 baseline is missing from MAIN outputs/02_FEATURE_FREEZE."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("measurement_version") != LEGACY_MEASUREMENT_VERSION:
        raise ValueError("Frozen QDIST baseline version is not qdist-v3.1.1.")
    if manifest.get("candidate_only") is not False:
        raise ValueError("Frozen qdist-v3.1.1 manifest is unexpectedly candidate-only.")
    if manifest.get("all_blocking_layers_pass") is not True:
        raise ValueError("Frozen qdist-v3.1.1 blocking layers did not pass.")
    if tuple(manifest.get("analysis_features", [])) != ANALYSIS_FEATURES:
        raise ValueError("Frozen qdist-v3.1.1 feature registry differs from the reviewed contract.")

    tables = root / "tables"
    gallery = root / "gallery"
    loaded = {
        "analysis": load_csv(tables / "qdist_v311_analysis_features.csv"),
        "recordings": load_csv(tables / "qdist_v311_recording_features.csv"),
        "candidates": load_csv(tables / "qdist_v311_candidate_plateau_ledger.csv"),
        "accepted": load_csv(tables / "qdist_v311_accepted_plateau_ledger.csv"),
        "episodes": load_csv(tables / "qdist_v311_episode_ledger.csv"),
        "edges": load_csv(tables / "qdist_v311_edge_ledger.csv"),
        "parameter_long": load_csv(tables / "qdist_v311_cohort_parameter_robustness.csv"),
        "parameter_summary": load_csv(tables / "qdist_v311_cohort_parameter_robustness_summary.csv"),
        "legacy_reconstruction": load_csv(tables / "qdist_v311_cohort_reconstruction.csv"),
        "legacy_gallery_summary": load_csv(tables / "qdist_v311_gallery_adjudication_summary.csv"),
        "legacy_feature_decisions": load_csv(tables / "qdist_v311_feature_decisions.csv"),
        "gallery_index": load_csv(gallery / "qdist_v311_gallery_index.csv"),
        "gallery_review": load_csv(gallery / "qdist_v311_gallery_review.csv"),
    }
    if len(loaded["analysis"]) != 519 or len(loaded["recordings"]) != 519:
        raise ValueError("Frozen QDIST baseline does not contain exactly 519 recordings.")
    if loaded["analysis"]["logical_recording_id"].duplicated().any():
        raise ValueError("Frozen QDIST analysis table contains duplicated recording identities.")
    return manifest, loaded


def _merge_intervals(intervals: Iterable[tuple[int, int]], gap_samples: int = 0) -> list[tuple[int, int]]:
    ordered = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prior_start, prior_end = merged[-1]
        if start <= prior_end + int(gap_samples):
            merged[-1] = (prior_start, max(prior_end, end))
        else:
            merged.append((start, end))
    return merged


def reconstruct_recording_features(
    recording_row: Mapping[str, Any],
    accepted: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    recording_id = str(recording_row["logical_recording_id"])
    frame_length = int(recording_row["qdist_frame_length_samples"])
    complete_frames = int(recording_row["qdist_complete_frame_count"])
    finite_channel_samples = int(recording_row["qdist_finite_channel_sample_count"])
    exposure_sec = float(recording_row["qdist_finite_exposure_sec"])

    local = accepted.loc[accepted["logical_recording_id"].astype(str).eq(recording_id)].copy()
    affected_frames: set[int] = set()
    channel_samples = 0
    if len(local):
        for channel, group in local.groupby("channel_index", sort=False):
            merged = _merge_intervals(
                zip(group["start_sample_task"], group["end_sample_task_exclusive"]),
                gap_samples=0,
            )
            channel_samples += sum(end - start for start, end in merged)
            for start, end in merged:
                first = max(0, start // frame_length)
                last = min(complete_frames - 1, (end - 1) // frame_length)
                if last >= first:
                    affected_frames.update(range(first, last + 1))

    frame_fraction = (
        len(affected_frames) / complete_frames if complete_frames > 0 else np.nan
    )
    sample_fraction = (
        channel_samples / finite_channel_samples if finite_channel_samples > 0 else np.nan
    )
    episode_count = int(
        episodes["logical_recording_id"].astype(str).eq(recording_id).sum()
    )
    event_rate = episode_count * 60.0 / exposure_sec if exposure_sec > 0 else np.nan
    return {
        "logical_recording_id": recording_id,
        "qdist_hard_clipped_frame_fraction_reconstructed": frame_fraction,
        "qdist_hard_clip_event_rate_per_min_reconstructed": event_rate,
        "qdist_hard_clipped_sample_fraction_reconstructed": sample_fraction,
        "qdist_reconstructed_affected_frame_count": len(affected_frames),
        "qdist_reconstructed_channel_sample_count": channel_samples,
        "qdist_reconstructed_event_count": episode_count,
    }


def build_reconstruction_audit(
    recordings: pd.DataFrame,
    accepted: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        reconstruct_recording_features(row, accepted, episodes)
        for row in recordings.to_dict("records")
    ]
    wide = pd.DataFrame(rows)
    merged = recordings[
        ["logical_recording_id", *ANALYSIS_FEATURES]
    ].merge(wide, on="logical_recording_id", how="left", validate="one_to_one")
    long_rows: list[dict[str, Any]] = []
    for feature in ANALYSIS_FEATURES:
        reconstructed = f"{feature}_reconstructed"
        for row in merged[["logical_recording_id", feature, reconstructed]].itertuples(index=False):
            stored = pd.to_numeric(pd.Series([getattr(row, feature)]), errors="coerce").iloc[0]
            rebuilt = pd.to_numeric(pd.Series([getattr(row, reconstructed)]), errors="coerce").iloc[0]
            both_missing = bool(pd.isna(stored) and pd.isna(rebuilt))
            difference = (
                abs(float(stored) - float(rebuilt))
                if np.isfinite(stored) and np.isfinite(rebuilt)
                else np.nan
            )
            long_rows.append(
                {
                    "logical_recording_id": str(row.logical_recording_id),
                    "feature": feature,
                    "stored": stored,
                    "reconstructed": rebuilt,
                    "absolute_difference": difference,
                    "both_missing": both_missing,
                }
            )
    long = pd.DataFrame(long_rows)
    summary = (
        long.groupby("feature", sort=False)
        .agg(
            recording_count=("logical_recording_id", "nunique"),
            finite_pair_count=("absolute_difference", "count"),
            maximum_absolute_difference=("absolute_difference", "max"),
            missing_pair_count=("both_missing", "sum"),
        )
        .reset_index()
    )
    summary["passed"] = summary["maximum_absolute_difference"].fillna(0).le(2e-15)
    return long, summary


def prepare_recording_table(
    analysis: pd.DataFrame,
    recordings: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "logical_recording_id",
        "qdist_measurement_version",
        "qdist_parameter_hash",
        "qdist_signal_view",
        "qdist_task_span_duration_sec",
        "qdist_finite_exposure_sec",
        "qdist_complete_frame_count",
        "qdist_frame_length_samples",
        "qdist_finite_channel_sample_count",
        "qdist_native_sample_rate_hz",
        "qdist_native_channel_count",
        "qdist_sample_format",
        "qdist_bits_per_raw_sample",
        "qdist_codec_name",
        "qdist_container_format",
        "qdist_support_tier",
        "qdist_status",
        "qdist_available",
        "qdist_accepted_plateau_count",
        "qdist_hard_clip_event_count",
        "qdist_hard_clip_event_rate_ci95_low_per_min",
        "qdist_hard_clip_event_rate_ci95_high_per_min",
        "qdist_source_sha256",
        "qdist_decoded_sha256",
        "qdist_source_path",
        "qdist_known_preprocessing_applied",
        "qdist_native_view_verified",
    ]
    metadata_columns = [c for c in metadata_columns if c in recordings.columns]
    table = analysis.drop(
        columns=[c for c in metadata_columns if c != "logical_recording_id" and c in analysis.columns],
        errors="ignore",
    ).merge(
        recordings[metadata_columns],
        on="logical_recording_id",
        how="left",
        validate="one_to_one",
    )
    table["participant_id"] = table["logical_recording_id"].map(derive_participant_id)
    table["acquisition_date"] = table["logical_recording_id"].map(derive_acquisition_date)
    table["acquisition_year"] = table["acquisition_date"].dt.year.astype("Int64")
    parts = table["logical_recording_id"].str.rsplit("_", n=6, expand=True)
    table["protocol_code"] = parts[1].astype(str)
    table["visit_code"] = parts[2].astype(str)
    table["qdist_positive"] = (
        pd.to_numeric(table["qdist_hard_clip_event_rate_per_min"], errors="coerce").fillna(0) > 0
    )
    table["qdist_valid_zero"] = (
        table["qdist_status"].astype(str).eq("available_no_events")
    )
    table["qdist_clipped_channel_ms_per_min"] = (
        pd.to_numeric(table["qdist_hard_clipped_sample_fraction"], errors="coerce")
        * 60_000.0
    )
    table["qdist_pcm_bit_depth_declared"] = pd.to_numeric(
        table.get("qdist_bits_per_raw_sample"), errors="coerce"
    )
    inferred = table["qdist_sample_format"].astype(str).str.extract(r"(\d+)", expand=False)
    table["qdist_pcm_bit_depth_effective"] = table[
        "qdist_pcm_bit_depth_declared"
    ].fillna(pd.to_numeric(inferred, errors="coerce"))
    return table


def feature_summary(recordings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in ANALYSIS_FEATURES:
        values = pd.to_numeric(recordings[feature], errors="coerce")
        finite = values.dropna()
        positive = finite.loc[finite > 0]
        rows.append(
            {
                "feature": feature,
                "recording_count": len(recordings),
                "available_n": int(finite.size),
                "available_fraction": float(finite.size / len(recordings)),
                "valid_zero_n": int((finite == 0).sum()),
                "valid_zero_fraction_available": float((finite == 0).mean()),
                "positive_n": int((finite > 0).sum()),
                "positive_fraction_available": float((finite > 0).mean()),
                "median": float(finite.median()) if len(finite) else np.nan,
                "q25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                "q75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                "maximum": float(finite.max()) if len(finite) else np.nan,
                "positive_median": float(positive.median()) if len(positive) else np.nan,
                "positive_q25": float(positive.quantile(0.25)) if len(positive) else np.nan,
                "positive_q75": float(positive.quantile(0.75)) if len(positive) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def morphology_margin_table(candidates: pd.DataFrame, parameters: Mapping[str, Any]) -> pd.DataFrame:
    local = candidates.copy()
    accepted = as_bool(local["accepted"])
    rows: list[dict[str, Any]] = []
    threshold_specs = [
        ("plateau_samples", "sample_count", float(parameters["minimum_plateau_samples"]), "higher"),
        ("context_ratio", "candidate_to_context_ratio", float(parameters["minimum_context_peak_ratio"]), "higher"),
        ("local_peak_ratio", "candidate_to_robust_peak_ratio", float(parameters["minimum_edge_to_robust_peak_ratio"]), "higher"),
        ("edge_zone_samples", "edge_zone_sample_count", float(parameters["minimum_edge_zone_samples"]), "higher"),
        ("edge_to_interior_ratio", "edge_to_interior_ratio", float(parameters["minimum_edge_to_interior_ratio"]), "higher"),
        ("edge_excess_samples", "edge_excess_samples", float(parameters["minimum_edge_excess_samples"]), "higher"),
        ("beyond_edge_samples", "beyond_edge_sample_count", float(parameters["maximum_beyond_edge_samples"]), "lower"),
        ("duration_ms", "duration_sec", float(parameters["maximum_plateau_duration_ms"]), "duration_ms"),
        ("flatness_ratio", "plateau_range", 1.0, "flatness"),
    ]
    for label, column, threshold, orientation in threshold_specs:
        values = pd.to_numeric(local[column], errors="coerce")
        if orientation == "duration_ms":
            values = values * 1000.0
            margin = threshold - values
        elif orientation == "flatness":
            tolerance = pd.to_numeric(local["flat_tolerance"], errors="coerce")
            values = values / tolerance.replace(0, np.nan)
            margin = 1.0 - values
        elif orientation == "higher":
            margin = values - threshold
        else:
            margin = threshold - values
        for stratum, mask in [("accepted", accepted), ("rejected", ~accepted)]:
            use = margin.loc[mask].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "criterion": label,
                    "stratum": stratum,
                    "candidate_count": int(mask.sum()),
                    "finite_margin_n": int(len(use)),
                    "threshold": threshold,
                    "median_margin": float(use.median()) if len(use) else np.nan,
                    "q10_margin": float(use.quantile(0.10)) if len(use) else np.nan,
                    "minimum_margin": float(use.min()) if len(use) else np.nan,
                    "nonnegative_fraction": float((use >= 0).mean()) if len(use) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def merge_gap_sensitivity(
    recordings: pd.DataFrame,
    accepted: pd.DataFrame,
    gaps_ms: Sequence[float] = (10.0, 20.0, 30.0, 50.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    accepted_lookup = {
        rid: group.copy()
        for rid, group in accepted.groupby("logical_recording_id", sort=False)
    }
    for rec in recordings.to_dict("records"):
        rid = str(rec["logical_recording_id"])
        fs = int(rec["qdist_native_sample_rate_hz"])
        exposure = float(rec["qdist_finite_exposure_sec"])
        local = accepted_lookup.get(rid, pd.DataFrame())
        intervals = [] if local.empty else list(
            zip(local["start_sample_task"], local["end_sample_task_exclusive"])
        )
        for gap_ms in gaps_ms:
            merged = _merge_intervals(intervals, int(round(float(gap_ms) * fs / 1000.0)))
            event_count = len(merged)
            rate = event_count / exposure * 60.0 if exposure > 0 else np.nan
            rows.append(
                {
                    "logical_recording_id": rid,
                    "participant_id": rec["participant_id"],
                    "merge_gap_ms": float(gap_ms),
                    "event_count": event_count,
                    "event_rate_per_min": rate,
                    "positive": event_count > 0,
                }
            )
    long = pd.DataFrame(rows)
    baseline = long.loc[long["merge_gap_ms"].eq(20.0), [
        "logical_recording_id", "event_count", "event_rate_per_min", "positive"
    ]].rename(columns={
        "event_count": "baseline_event_count",
        "event_rate_per_min": "baseline_event_rate_per_min",
        "positive": "baseline_positive",
    })
    compared = long.merge(baseline, on="logical_recording_id", validate="many_to_one")
    compared["event_count_changed"] = compared["event_count"].ne(compared["baseline_event_count"])
    compared["occurrence_agreement"] = compared["positive"].eq(compared["baseline_positive"])
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
    positive_ids = set(
        recordings.loc[recordings["qdist_positive"], "logical_recording_id"].astype(str)
    )
    rows: list[dict[str, Any]] = []
    for rec in recordings.loc[recordings["logical_recording_id"].astype(str).isin(positive_ids)].to_dict("records"):
        rid = str(rec["logical_recording_id"])
        local_accepted = accepted.loc[accepted["logical_recording_id"].astype(str).eq(rid)].copy()
        local_episodes = episodes.loc[episodes["logical_recording_id"].astype(str).eq(rid)].copy()
        base = reconstruct_recording_features(rec, local_accepted, local_episodes)
        for candidate_id in local_accepted["candidate_id"].astype(str):
            variant_accepted = local_accepted.loc[
                ~local_accepted["candidate_id"].astype(str).eq(candidate_id)
            ]
            # Rebuild 20-ms episodes after deleting one plateau.
            fs = int(rec["qdist_native_sample_rate_hz"])
            merged = _merge_intervals(
                zip(variant_accepted["start_sample_task"], variant_accepted["end_sample_task_exclusive"]),
                int(round(0.020 * fs)),
            )
            variant_episodes = pd.DataFrame(
                {
                    "logical_recording_id": [rid] * len(merged),
                    "start_sample_task": [a for a, _ in merged],
                    "end_sample_task_exclusive": [b for _, b in merged],
                }
            )
            variant = reconstruct_recording_features(rec, variant_accepted, variant_episodes)
            rows.append(
                {
                    "logical_recording_id": rid,
                    "deletion_type": "plateau",
                    "deleted_id": candidate_id,
                    "frame_fraction_absolute_change": abs(
                        variant["qdist_hard_clipped_frame_fraction_reconstructed"]
                        - base["qdist_hard_clipped_frame_fraction_reconstructed"]
                    ),
                    "event_rate_absolute_change": abs(
                        variant["qdist_hard_clip_event_rate_per_min_reconstructed"]
                        - base["qdist_hard_clip_event_rate_per_min_reconstructed"]
                    ),
                    "sample_fraction_absolute_change": abs(
                        variant["qdist_hard_clipped_sample_fraction_reconstructed"]
                        - base["qdist_hard_clipped_sample_fraction_reconstructed"]
                    ),
                }
            )
        for episode_id in local_episodes["episode_id"].astype(str):
            variant_episodes = local_episodes.loc[
                ~local_episodes["episode_id"].astype(str).eq(episode_id)
            ]
            variant = reconstruct_recording_features(rec, local_accepted, variant_episodes)
            rows.append(
                {
                    "logical_recording_id": rid,
                    "deletion_type": "episode",
                    "deleted_id": episode_id,
                    "frame_fraction_absolute_change": 0.0,
                    "event_rate_absolute_change": abs(
                        variant["qdist_hard_clip_event_rate_per_min_reconstructed"]
                        - base["qdist_hard_clip_event_rate_per_min_reconstructed"]
                    ),
                    "sample_fraction_absolute_change": 0.0,
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame()
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


def finite_spearman(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return len(frame), np.nan
    return len(frame), float(stats.spearmanr(frame["x"], frame["y"]).statistic)


def cohen_kappa_binary(a: pd.Series, b: pd.Series) -> float:
    a = as_bool(a)
    b = as_bool(b)
    if len(a) == 0:
        return np.nan
    observed = float((a == b).mean())
    pa = float(a.mean())
    pb = float(b.mean())
    expected = pa * pb + (1 - pa) * (1 - pb)
    if np.isclose(1 - expected, 0):
        return np.nan
    return (observed - expected) / (1 - expected)


def repeated_recording_evidence(recordings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = recordings.sort_values(
        ["participant_id", "acquisition_date", "logical_recording_id"]
    )
    pair_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    for participant, group in ordered.groupby("participant_id", sort=True):
        if len(group) < 2:
            continue
        first = group.iloc[0]
        second = group.iloc[1]
        row = {
            "participant_id": participant,
            "recording_1": first["logical_recording_id"],
            "recording_2": second["logical_recording_id"],
            "positive_1": bool(first["qdist_positive"]),
            "positive_2": bool(second["qdist_positive"]),
        }
        for feature in ANALYSIS_FEATURES:
            row[f"{feature}_1"] = first[feature]
            row[f"{feature}_2"] = second[feature]
        pair_rows.append(row)
        records = group.to_dict("records")
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                all_pair_rows.append(
                    {
                        "participant_id": participant,
                        "recording_1": records[i]["logical_recording_id"],
                        "recording_2": records[j]["logical_recording_id"],
                        "positive_1": bool(records[i]["qdist_positive"]),
                        "positive_2": bool(records[j]["qdist_positive"]),
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    all_pairs = pd.DataFrame(all_pair_rows)
    if pairs.empty:
        return pairs, all_pairs, pd.DataFrame()
    n11 = int((pairs["positive_1"] & pairs["positive_2"]).sum())
    n10 = int((pairs["positive_1"] & ~pairs["positive_2"]).sum())
    n01 = int((~pairs["positive_1"] & pairs["positive_2"]).sum())
    n00 = int((~pairs["positive_1"] & ~pairs["positive_2"]).sum())
    positive_agreement = 2 * n11 / (2 * n11 + n10 + n01) if (2 * n11 + n10 + n01) else np.nan
    negative_agreement = 2 * n00 / (2 * n00 + n10 + n01) if (2 * n00 + n10 + n01) else np.nan
    summary_rows = [
        {
            "metric": "occurrence",
            "participant_pair_count": len(pairs),
            "both_positive_n11": n11,
            "first_only_n10": n10,
            "second_only_n01": n01,
            "both_zero_n00": n00,
            "overall_agreement": float((pairs["positive_1"] == pairs["positive_2"]).mean()),
            "positive_agreement": positive_agreement,
            "negative_agreement": negative_agreement,
            "cohens_kappa": cohen_kappa_binary(pairs["positive_1"], pairs["positive_2"]),
            "positive_part_pair_n": n11,
            "positive_part_spearman_rho": np.nan,
            "interpretation": (
                "Positive-part persistence not estimable: fewer than five pairs were positive at both visits."
                if n11 < 5 else "Positive-part persistence estimable."
            ),
        }
    ]
    for feature in ANALYSIS_FEATURES:
        both = pairs.loc[pairs["positive_1"] & pairs["positive_2"]]
        n, rho = finite_spearman(both[f"{feature}_1"], both[f"{feature}_2"])
        summary_rows.append(
            {
                "metric": feature,
                "participant_pair_count": len(pairs),
                "both_positive_n11": n11,
                "first_only_n10": n10,
                "second_only_n01": n01,
                "both_zero_n00": n00,
                "overall_agreement": float((pairs["positive_1"] == pairs["positive_2"]).mean()),
                "positive_agreement": positive_agreement,
                "negative_agreement": negative_agreement,
                "cohens_kappa": cohen_kappa_binary(pairs["positive_1"], pairs["positive_2"]),
                "positive_part_pair_n": n,
                "positive_part_spearman_rho": rho,
                "interpretation": (
                    "Not estimable because the detector is sparse and fewer than five pairs were positive at both visits."
                    if n < 5 else "Estimated on pairs positive at both visits."
                ),
            }
        )
    return pairs, all_pairs, pd.DataFrame(summary_rows)


def redundancy_table(recordings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(ANALYSIS_FEATURES):
        for right in ANALYSIS_FEATURES[i + 1 :]:
            n_all, rho_all = finite_spearman(recordings[left], recordings[right])
            positive = recordings.loc[recordings["qdist_positive"]]
            n_positive, rho_positive = finite_spearman(positive[left], positive[right])
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "all_recordings_n": n_all,
                    "all_recordings_spearman_rho": rho_all,
                    "positive_recordings_n": n_positive,
                    "positive_recordings_spearman_rho": rho_positive,
                    "related_view_system": True,
                }
            )
    return pd.DataFrame(rows)


def _wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
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
            frame_fraction_median=("qdist_hard_clipped_frame_fraction", "median"),
            event_rate_median=("qdist_hard_clip_event_rate_per_min", "median"),
            sample_fraction_median=("qdist_hard_clipped_sample_fraction", "median"),
        )
        .reset_index()
    )
    rec_positive = int(recordings["qdist_positive"].sum())
    part_positive = int(participant["participant_any_positive"].sum())
    rec_low, rec_high = _wilson_interval(rec_positive, len(recordings))
    part_low, part_high = _wilson_interval(part_positive, len(participant))
    summary = pd.DataFrame(
        [
            {
                "analysis_level": "recording_weighted",
                "units": len(recordings),
                "positive_units": rec_positive,
                "positive_fraction": rec_positive / len(recordings),
                "wilson95_low": rec_low,
                "wilson95_high": rec_high,
            },
            {
                "analysis_level": "participant_ever_positive",
                "units": len(participant),
                "positive_units": part_positive,
                "positive_fraction": part_positive / len(participant),
                "wilson95_low": part_low,
                "wilson95_high": part_high,
            },
        ]
    )
    return participant, summary


def adjudication_summary(index: pd.DataFrame, review: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = index.drop(columns=[c for c in ["review_label", "review_comment", "reviewer"] if c in index.columns]).merge(
        review,
        on=["review_item_id", "logical_recording_id"],
        how="left",
        validate="one_to_one",
    )
    positive_labels = {"DEFINITE_HARD_CLIP", "PROBABLE_HARD_CLIP"}
    adjudicable_labels = positive_labels | {"NOT_HARD_CLIP"}
    merged["adjudicable"] = merged["review_label"].astype(str).isin(adjudicable_labels)
    merged["hard_clip_positive"] = merged["review_label"].astype(str).isin(positive_labels)
    rows: list[dict[str, Any]] = []
    for stratum, group in merged.groupby("stratum", sort=False):
        adjudicable = group.loc[group["adjudicable"]]
        rows.append(
            {
                "stratum": stratum,
                "review_item_count": len(group),
                "adjudicable_n": len(adjudicable),
                "ambiguous_n": int(group["review_label"].astype(str).eq("AMBIGUOUS").sum()),
                "hard_clip_positive_n": int(adjudicable["hard_clip_positive"].sum()),
                "hard_clip_positive_fraction": (
                    float(adjudicable["hard_clip_positive"].mean())
                    if len(adjudicable) else np.nan
                ),
            }
        )
    return merged, pd.DataFrame(rows)


def _gallery_file(frozen_gallery: Path, path_value: Any, suffix: str) -> Path:
    basename = PureWindowsPath(str(path_value)).name
    candidate = frozen_gallery / basename
    if candidate.exists():
        return candidate
    stem = Path(basename).stem
    candidate = frozen_gallery / f"{stem}{suffix}"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Frozen gallery artifact not found: {basename}")


def event_review_source(
    wav_path: Path,
    *,
    item_id: str,
    stratum: str,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    audio, fs = sf.read(wav_path, always_2d=True, dtype="float64")
    waveform = audio[:, 0]
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid frozen event-review excerpt: {wav_path}")
    center = len(waveform) // 2
    wide_half = min(center, int(round(0.100 * fs)), len(waveform) - center)
    if wide_half <= 8:
        wide_half = min(center, len(waveform) - center)
    wide_left, wide_right = center - wide_half, center + wide_half
    wide = waveform[wide_left:wide_right]
    wide_time_ms = (np.arange(wide_left, wide_right) - center) / fs * 1000.0
    zoom_half = min(int(round(0.004 * fs)), center, len(waveform) - center)
    zoom_left, zoom_right = center - zoom_half, center + zoom_half
    zoom = waveform[zoom_left:zoom_right]
    zoom_time_ms = (np.arange(zoom_left, zoom_right) - center) / fs * 1000.0
    derivative = np.diff(zoom, prepend=zoom[0])

    counts, edges = np.histogram(wide, bins=100)
    centers = (edges[:-1] + edges[1:]) / 2.0

    nperseg = min(512, max(64, len(wide) // 4))
    noverlap = int(0.75 * nperseg)
    frequencies, times, spectrum = signal.spectrogram(
        wide,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",
    )
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    time_ms = (times + wide_left / fs - center / fs) * 1000.0

    rows: list[dict[str, Any]] = []
    waveform_indices = np.unique(
        np.linspace(0, max(len(wide) - 1, 0), min(len(wide), 1000)).round().astype(int)
    )
    for index in waveform_indices:
        rows.append(
            {
                "review_item_id": item_id,
                "stratum": stratum,
                "view": "waveform",
                "time_ms": wide_time_ms[index],
                "amplitude": wide[index],
            }
        )
    for t, value, delta in zip(zoom_time_ms, zoom, derivative):
        rows.append(
            {
                "review_item_id": item_id,
                "stratum": stratum,
                "view": "pcm_derivative",
                "time_ms": t,
                "amplitude": value,
                "first_difference": delta,
            }
        )
    for center_value, count in zip(centers, counts):
        rows.append(
            {
                "review_item_id": item_id,
                "stratum": stratum,
                "view": "amplitude_distribution",
                "amplitude_bin_center": center_value,
                "count": int(count),
            }
        )
    # Keep the spectrogram source compact while preserving the displayed field.
    frequency_indices = np.unique(
        np.linspace(0, max(len(frequencies) - 1, 0), min(len(frequencies), 48)).round().astype(int)
    )
    time_indices = np.unique(
        np.linspace(0, max(len(times) - 1, 0), min(len(times), 32)).round().astype(int)
    )
    for frequency_index in frequency_indices:
        for time_index in time_indices:
            rows.append(
                {
                    "review_item_id": item_id,
                    "stratum": stratum,
                    "view": "spectrogram",
                    "time_ms": time_ms[time_index],
                    "frequency_hz": frequencies[frequency_index],
                    "magnitude_db": spectrum_db[frequency_index, time_index],
                }
            )
    rows.append(
        {
            "review_item_id": item_id,
            "stratum": stratum,
            "view": "audio_excerpt",
            "audio_path": str(wav_path),
            "sample_rate_hz": int(fs),
            "sample_count": int(len(waveform)),
        }
    )
    return waveform, int(fs), pd.DataFrame(rows)


def plot_event_review_item(
    wav_path: Path,
    *,
    item_id: str,
    stratum: str,
    review_label: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    waveform, fs, source = event_review_source(wav_path, item_id=item_id, stratum=stratum)
    center = len(waveform) // 2
    wide_half = min(center, int(round(0.100 * fs)), len(waveform) - center)
    left, right = center - wide_half, center + wide_half
    wide = waveform[left:right]
    time_ms = (np.arange(left, right) - center) / fs * 1000.0
    zoom_half = min(int(round(0.004 * fs)), center, len(waveform) - center)
    zl, zr = center - zoom_half, center + zoom_half
    zoom = waveform[zl:zr]
    zoom_ms = (np.arange(zl, zr) - center) / fs * 1000.0
    derivative = np.diff(zoom, prepend=zoom[0])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))
    axes[0, 0].plot(time_ms, wide, linewidth=0.8)
    axes[0, 0].axvline(0.0, linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Waveform around review center")
    axes[0, 0].set_xlabel("Time from center (ms)")
    axes[0, 0].set_ylabel("Native decoded amplitude")

    axis2 = axes[0, 1]
    axis2.plot(zoom_ms, zoom, marker="o", markersize=2.2, linewidth=0.8, label="PCM amplitude")
    twin = axis2.twinx()
    twin.plot(zoom_ms, derivative, linewidth=0.8, alpha=0.7, label="First difference")
    axis2.axvline(0.0, linestyle="--", linewidth=1.0)
    axis2.set_title("PCM-code and derivative context")
    axis2.set_xlabel("Time from center (ms)")
    axis2.set_ylabel("Amplitude")
    twin.set_ylabel("First difference")

    axes[1, 0].hist(wide, bins=100, log=True)
    axes[1, 0].set_title("Local amplitude occupancy")
    axes[1, 0].set_xlabel("Amplitude")
    axes[1, 0].set_ylabel("Count (log)")

    nperseg = min(512, max(64, len(wide) // 4))
    frequencies, times, spectrum = signal.spectrogram(
        wide,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=int(0.75 * nperseg),
        mode="magnitude",
    )
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    local_time_ms = (times - wide_half / fs) * 1000.0
    axes[1, 1].pcolormesh(local_time_ms, frequencies / 1000.0, spectrum_db, shading="auto")
    axes[1, 1].set_ylim(0, min(12.0, fs / 2000.0))
    axes[1, 1].set_title("Local spectrogram")
    axes[1, 1].set_xlabel("Time from center (ms)")
    axes[1, 1].set_ylabel("Frequency (kHz)")

    fig.suptitle(f"QDIST {stratum} | {review_label} | {item_id}", fontsize=10)
    fig.tight_layout()
    return fig, source


def build_event_review_package(
    frozen_root: Path,
    output_root: Path,
    gallery_index: pd.DataFrame,
    gallery_review: pd.DataFrame,
    *,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the complete 60-item review index without re-rendering 60 figures.

    The immutable v3.1.1 PNG already contains waveform, plateau morphology, and
    amplitude occupancy. The linked WAV is preserved. A compact standardized
    source CSV adds explicit waveform, PCM/derivative, amplitude-distribution,
    spectrogram, and audio-excerpt views. Eight deterministic items are later
    rendered as full A-J Panel G bundles.
    """
    frozen_gallery = frozen_root / "gallery"
    output_gallery = output_root / "event_review"
    output_gallery.mkdir(parents=True, exist_ok=True)
    adjudicated, summary = adjudication_summary(gallery_index, gallery_review)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in adjudicated.to_dict("records"):
        item_id = str(item["review_item_id"])
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id)
        stem = f"qdist_review_{safe}"
        legacy_png = output_gallery / f"{stem}.legacy.png"
        source_csv = output_gallery / f"{stem}.source.csv"
        caption = output_gallery / f"{stem}.caption.md"
        provenance = output_gallery / f"{stem}.provenance.json"
        audio_out = output_gallery / f"{stem}.wav"
        try:
            wav_source = _gallery_file(frozen_gallery, item.get("audio_wav"), ".wav")
            png_source = _gallery_file(frozen_gallery, item.get("figure_png"), ".png")
            required_paths = [legacy_png, source_csv, caption, provenance, audio_out]
            if rebuild or not all(path.exists() and path.stat().st_size > 0 for path in required_paths):
                _, _, source = event_review_source(
                    wav_source,
                    item_id=item_id,
                    stratum=str(item["stratum"]),
                )
                source.to_csv(source_csv, index=False)
                shutil.copy2(wav_source, audio_out)
                shutil.copy2(png_source, legacy_png)
                caption.write_text(
                    (
                        f"Label-blind QDIST event-review item `{item_id}` from the "
                        f"`{item['stratum']}` stratum. The immutable v3.1.1 PNG supplies "
                        "waveform, plateau-morphology, and amplitude-occupancy evidence; "
                        "the standardized source CSV explicitly supplies waveform, PCM/first-"
                        "difference, amplitude-distribution, spectrogram, and audio-excerpt "
                        "views; the linked WAV is the adjudicated audio excerpt. Clinical and "
                        "human-QC labels are not used.\n"
                    ),
                    encoding="utf-8",
                )
                write_json(
                    {
                        "created_utc": utc_now(),
                        "measurement_version": MEASUREMENT_VERSION,
                        "legacy_measurement_version": LEGACY_MEASUREMENT_VERSION,
                        "review_item_id": item_id,
                        "logical_recording_id": str(item["logical_recording_id"]),
                        "stratum": str(item["stratum"]),
                        "review_label": str(item.get("review_label", "")),
                        "selection_label_blind": True,
                        "required_views": list(EVENT_REVIEW_REQUIRED_VIEWS),
                        "source_audio_sha256": sha256_file(wav_source),
                        "legacy_figure_sha256": sha256_file(png_source),
                    },
                    provenance,
                )
            source = pd.read_csv(source_csv)
            views = set(source["view"].astype(str)) if "view" in source else set()
            rows.append(
                {
                    "review_item_id": item_id,
                    "logical_recording_id": str(item["logical_recording_id"]),
                    "stratum": str(item["stratum"]),
                    "review_label": str(item.get("review_label", "")),
                    "review_comment": str(item.get("review_comment", "")),
                    "reviewer": str(item.get("reviewer", "")),
                    "legacy_figure_png": str(legacy_png),
                    "source_csv": str(source_csv),
                    "audio_wav": str(audio_out),
                    "caption": str(caption),
                    "provenance": str(provenance),
                    "declared_view_count": len(views),
                    "all_required_views_present": set(EVENT_REVIEW_REQUIRED_VIEWS).issubset(views),
                    "selection_label_blind": True,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "review_item_id": item_id,
                    "logical_recording_id": str(item.get("logical_recording_id", "")),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return pd.DataFrame(rows), summary, pd.DataFrame(
        errors,
        columns=["review_item_id", "logical_recording_id", "error_type", "message"],
    )

def select_gallery_examples(event_index: pd.DataFrame, minimum: int = GALLERY_MINIMUM) -> pd.DataFrame:
    rows: list[pd.Series] = []
    used_recordings: set[str] = set()
    # One accepted item from each positive recording first.
    accepted = event_index.loc[event_index["stratum"].eq("accepted_plateau")].copy()
    accepted["label_rank"] = accepted["review_label"].map(
        {"DEFINITE_HARD_CLIP": 0, "PROBABLE_HARD_CLIP": 1}
    ).fillna(2)
    for recording_id, group in accepted.sort_values(
        ["label_rank", "review_item_id"]
    ).groupby("logical_recording_id", sort=True):
        rows.append(group.iloc[0])
        used_recordings.add(str(recording_id))
    for stratum in ["rejected_candidate", "valid_zero"]:
        candidates = event_index.loc[event_index["stratum"].eq(stratum)].sort_values(
            ["logical_recording_id", "review_item_id"]
        )
        for _, row in candidates.iterrows():
            if str(row["logical_recording_id"]) not in used_recordings:
                rows.append(row)
                used_recordings.add(str(row["logical_recording_id"]))
                break
    if len(rows) < minimum:
        for _, row in event_index.sort_values(
            ["stratum", "logical_recording_id", "review_item_id"]
        ).iterrows():
            if str(row["logical_recording_id"]) not in used_recordings:
                rows.append(row)
                used_recordings.add(str(row["logical_recording_id"]))
            if len(rows) >= minimum:
                break
    selected = pd.DataFrame(rows).head(max(minimum, len(rows))).reset_index(drop=True)
    selected["gallery_ordinal"] = np.arange(1, len(selected) + 1)
    return selected


def build_ml_interface(recordings: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "logical_recording_id",
        "participant_id",
        *ANALYSIS_FEATURES,
        "qdist_hard_clipped_frame_fraction_status",
        "qdist_hard_clip_event_rate_per_min_status",
        "qdist_hard_clipped_sample_fraction_status",
        "qdist_status",
        "qdist_available",
        "qdist_support_tier",
        "qdist_task_span_duration_sec",
        "qdist_finite_exposure_sec",
        "qdist_complete_frame_count",
        "qdist_frame_length_samples",
        "qdist_finite_channel_sample_count",
        "qdist_hard_clip_event_count",
        "qdist_accepted_plateau_count",
        "qdist_hard_clip_event_rate_ci95_low_per_min",
        "qdist_hard_clip_event_rate_ci95_high_per_min",
        "qdist_native_sample_rate_hz",
        "qdist_native_channel_count",
        "qdist_sample_format",
        "qdist_pcm_bit_depth_effective",
        "qdist_codec_name",
        "qdist_container_format",
        "qdist_parameter_hash",
        "qdist_source_sha256",
        "qdist_decoded_sha256",
    ]
    columns = [c for c in columns if c in recordings.columns]
    export = recordings[columns].copy()
    export["qdist_measurement_version"] = LEGACY_MEASUREMENT_VERSION
    export["qdist_reviewed_contract_version"] = MEASUREMENT_VERSION
    export["qdist_missing_values_imputed"] = False
    export["qdist_family_scalar_constructed"] = False
    export["qdist_standalone_gate_allowed"] = False
    export["qdist_complete_nonlinear_distortion_claim_allowed"] = False
    return export


def _base_provenance(panel: str, parameter_hash: str) -> dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "panel": panel,
        "measurement_version": MEASUREMENT_VERSION,
        "legacy_measurement_version": LEGACY_MEASUREMENT_VERSION,
        "reviewed_orchestration_version": COHORT_ORCHESTRATION_VERSION,
        "parameter_hash": parameter_hash,
        "feature_values_recomputed": False,
        "clinical_labels_used": False,
        "human_qc_labels_used": False,
    }


def create_cohort_figures(
    output_root: Path,
    preflight_index: pd.DataFrame,
    recordings: pd.DataFrame,
    accepted: pd.DataFrame,
    episodes: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    margin_summary: pd.DataFrame,
    merge_summary: pd.DataFrame,
    deletion_summary: pd.DataFrame,
    empirical_summary: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    redundancy: pd.DataFrame,
    weighting_summary: pd.DataFrame,
    adjudication: pd.DataFrame,
    ml_interface: pd.DataFrame,
    event_index: pd.DataFrame,
    gallery_selection: pd.DataFrame,
    *,
    parameter_hash: str,
) -> pd.DataFrame:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_rows: list[dict[str, Any]] = []

    # Preserve accepted preflight bundles in-place and index their existing paths.
    for row in preflight_index.to_dict("records"):
        figure_rows.append({
            "panel": row["panel"],
            "stem": row["stem"],
            **{key: row[key] for key in ["png", "svg", "pdf", "source_csv", "caption", "provenance"]},
            "bundle_role": "accepted_preflight",
        })

    # D1: exposure/support.
    support = (
        recordings.groupby(["qdist_support_tier", "qdist_status"], dropna=False)
        .agg(recordings=("logical_recording_id", "size"), median_exposure_sec=("qdist_finite_exposure_sec", "median"), median_complete_frames=("qdist_complete_frame_count", "median"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    recordings["qdist_finite_exposure_sec"].hist(ax=axes[0], bins=30)
    axes[0].set_title("Finite native task-span exposure")
    axes[0].set_xlabel("Exposure (s)")
    axes[0].set_ylabel("Recordings")
    support.groupby("qdist_support_tier")["recordings"].sum().plot.bar(ax=axes[1])
    axes[1].set_title("Support tiers")
    axes[1].set_xlabel("Support tier")
    axes[1].set_ylabel("Recordings")
    recordings["qdist_status"].value_counts().plot.bar(ax=axes[2])
    axes[2].set_title("Measurement status")
    axes[2].set_xlabel("Status")
    axes[2].set_ylabel("Recordings")
    fig.tight_layout()
    paths = save_figure_bundle(
        fig, figures, "D1_availability_exposure", source=support,
        caption=(
            "Availability and exposure for the frozen 519-recording QDIST cohort. "
            "Support tiers describe native task-span exposure and complete 30-ms frames; "
            "they are not quality grades. Valid zero remains distinct from unavailable."
        ),
        provenance=_base_provenance("D1", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "D1", "stem": "D1_availability_exposure", **paths, "bundle_role": "main"})

    # D2: occurrence/sparsity.
    occurrence_source = pd.DataFrame([
        {"metric": "recordings", "total": len(recordings), "positive": int(recordings["qdist_positive"].sum()), "zero": int(recordings["qdist_valid_zero"].sum())},
        {"metric": "accepted_plateaus", "total": len(accepted), "positive": len(accepted), "zero": 0},
        {"metric": "merged_episodes", "total": len(episodes), "positive": len(episodes), "zero": 0},
    ])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(["Valid zero", "Positive"], [int(recordings["qdist_valid_zero"].sum()), int(recordings["qdist_positive"].sum())])
    axes[0].set_title("Recording occurrence")
    axes[0].set_ylabel("Recordings")
    positive = recordings.loc[recordings["qdist_positive"]]
    axes[1].bar(positive["logical_recording_id"], positive["qdist_accepted_plateau_count"])
    axes[1].set_title("Accepted plateaus in positive recordings")
    axes[1].tick_params(axis="x", labelrotation=90, labelsize=7)
    axes[1].set_ylabel("Plateaus")
    axes[2].bar(positive["logical_recording_id"], positive["qdist_hard_clip_event_count"])
    axes[2].set_title("Merged episodes in positive recordings")
    axes[2].tick_params(axis="x", labelrotation=90, labelsize=7)
    axes[2].set_ylabel("Episodes")
    fig.tight_layout()
    source = pd.concat([
        occurrence_source.assign(source_table="summary"),
        positive[["logical_recording_id", "qdist_accepted_plateau_count", "qdist_hard_clip_event_count"]].assign(source_table="positive_recordings"),
    ], ignore_index=True, sort=False)
    paths = save_figure_bundle(
        fig, figures, "D2_occurrence_sparsity", source=source,
        caption=(
            "QDIST occurrence is sparse: valid zero is the dominant available state, while "
            "accepted plateaus and merged episodes occur in a small number of recordings. "
            "Prevalence is descriptive and was not used to tune detector thresholds."
        ),
        provenance=_base_provenance("D2", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "D2", "stem": "D2_occurrence_sparsity", **paths, "bundle_role": "main"})

    # D3: native geometry.
    geometry_columns = ["qdist_native_sample_rate_hz", "qdist_native_channel_count", "qdist_sample_format", "qdist_pcm_bit_depth_effective", "qdist_codec_name", "qdist_container_format", "acquisition_year"]
    geometry = recordings[geometry_columns].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.0))
    recordings["qdist_native_sample_rate_hz"].value_counts().sort_index().plot.bar(ax=axes[0, 0])
    axes[0, 0].set_title("Native sample rate")
    axes[0, 0].set_ylabel("Recordings")
    recordings["qdist_native_channel_count"].value_counts().sort_index().plot.bar(ax=axes[0, 1])
    axes[0, 1].set_title("Native channel count")
    recordings["qdist_sample_format"].value_counts().plot.bar(ax=axes[0, 2])
    axes[0, 2].set_title("Sample format")
    recordings["qdist_pcm_bit_depth_effective"].value_counts(dropna=False).sort_index().plot.bar(ax=axes[1, 0])
    axes[1, 0].set_title("Effective PCM bit depth")
    recordings["qdist_codec_name"].value_counts().plot.bar(ax=axes[1, 1])
    axes[1, 1].set_title("Codec")
    recordings["acquisition_year"].value_counts().sort_index().plot.bar(ax=axes[1, 2])
    axes[1, 2].set_title("Acquisition vintage")
    for ax in axes.flat:
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    paths = save_figure_bundle(
        fig, figures, "D3_native_geometry", source=geometry,
        caption=(
            "Native geometry and acquisition vintage for the authoritative QDIST input. "
            "QDIST inspects the first decoded native-rate stream with channels preserved; "
            "no resampling, channel averaging, normalization, filtering, or DC removal occurs."
        ),
        provenance=_base_provenance("D3", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "D3", "stem": "D3_native_geometry", **paths, "bundle_role": "main"})

    # E1: full-detector variants plus threshold margins.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    plot = parameter_summary.copy()
    plot = plot.loc[~plot["variant"].astype(str).str.startswith("merge_")]
    axes[0].barh(plot["variant"].astype(str), 1 - pd.to_numeric(plot["zero_positive_class_agreement"], errors="coerce"))
    axes[0].set_title("Occurrence disagreement versus default")
    axes[0].set_xlabel("Disagreement fraction")
    accepted_margins = margin_summary.loc[margin_summary["stratum"].eq("accepted")]
    axes[1].barh(accepted_margins["criterion"], accepted_margins["minimum_margin"])
    axes[1].axvline(0, linestyle="--", linewidth=1.0)
    axes[1].set_title("Minimum accepted-event threshold margin")
    axes[1].set_xlabel("Signed margin; negative indicates a failed criterion")
    fig.tight_layout()
    source = pd.concat([
        parameter_summary.assign(source_table="full_detector_reruns"),
        margin_summary.assign(source_table="candidate_margin_audit"),
    ], ignore_index=True, sort=False)
    paths = save_figure_bundle(
        fig, figures, "E1_detector_parameter_sensitivity", source=source,
        caption=(
            "Detector-parameter sensitivity combines full qdist-v3.1.1 cohort reruns for "
            "prespecified magnitude, support, plateau-length, and edge-support variants with "
            "candidate-ledger margins for flatness, context, prominence, edge, duration, and "
            "terminality criteria. Sparse absolute changes are emphasized over unstable relative percentages."
        ),
        provenance=_base_provenance("E1", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "E1", "stem": "E1_detector_parameter_sensitivity", **paths, "bundle_role": "main"})

    # E2: merge gap.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].plot(merge_summary["merge_gap_ms"], merge_summary["positive_recording_count"], marker="o")
    axes[0].set_title("Positive recordings")
    axes[0].set_xlabel("Merge gap (ms)")
    axes[0].set_ylabel("Recordings")
    axes[1].plot(merge_summary["merge_gap_ms"], merge_summary["event_count_changed_fraction"], marker="o")
    axes[1].set_title("Event-count changes")
    axes[1].set_xlabel("Merge gap (ms)")
    axes[1].set_ylabel("Fraction changed")
    axes[2].plot(merge_summary["merge_gap_ms"], merge_summary["maximum_absolute_rate_change"], marker="o")
    axes[2].set_title("Maximum event-rate change")
    axes[2].set_xlabel("Merge gap (ms)")
    axes[2].set_ylabel("Events/min")
    fig.tight_layout()
    paths = save_figure_bundle(
        fig, figures, "E2_episode_grouping_sensitivity", source=merge_summary,
        caption=(
            "Episode grouping sensitivity at 10, 20, 30, and 50 ms. The default 20-ms "
            "definition is a feature-identity parameter. Frame and sample burdens are not "
            "used to rescue the event-rate feature if episode grouping is unstable."
        ),
        provenance=_base_provenance("E2", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "E2", "stem": "E2_episode_grouping_sensitivity", **paths, "bundle_role": "main"})

    # E3: sparse burden/deletion/Poisson.
    positive = recordings.loc[recordings["qdist_positive"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].bar(positive["logical_recording_id"], positive["qdist_clipped_channel_ms_per_min"])
    axes[0].set_title("Absolute clipped channel-time burden")
    axes[0].set_ylabel("Clipped channel-ms/min")
    axes[0].tick_params(axis="x", labelrotation=90, labelsize=7)
    if len(deletion_summary):
        axes[1].bar(deletion_summary["deletion_type"], deletion_summary["maximum_event_rate_absolute_change"])
    axes[1].set_title("Maximum delete-one event-rate influence")
    axes[1].set_ylabel("Events/min")
    x = np.arange(len(positive))
    axes[2].errorbar(
        x,
        positive["qdist_hard_clip_event_rate_per_min"],
        yerr=[
            positive["qdist_hard_clip_event_rate_per_min"] - positive["qdist_hard_clip_event_rate_ci95_low_per_min"],
            positive["qdist_hard_clip_event_rate_ci95_high_per_min"] - positive["qdist_hard_clip_event_rate_per_min"],
        ],
        fmt="o",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(positive["logical_recording_id"], rotation=90, fontsize=7)
    axes[2].set_title("Exact Poisson intervals")
    axes[2].set_ylabel("Events/min")
    fig.tight_layout()
    source = pd.concat([
        positive[["logical_recording_id", "qdist_clipped_channel_ms_per_min", "qdist_hard_clip_event_rate_per_min", "qdist_hard_clip_event_rate_ci95_low_per_min", "qdist_hard_clip_event_rate_ci95_high_per_min"]].assign(source_table="positive_recordings"),
        deletion_summary.assign(source_table="deletion_summary"),
    ], ignore_index=True, sort=False)
    paths = save_figure_bundle(
        fig, figures, "E3_sparse_burden_deletion_poisson", source=source,
        caption=(
            "Sparse-burden robustness is reported in absolute clipped channel-milliseconds per "
            "analyzed minute, alongside delete-one plateau/episode influence and exact Poisson "
            "uncertainty for event rates. Relative percentage changes are not used alone for rejection."
        ),
        provenance=_base_provenance("E3", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "E3", "stem": "E3_sparse_burden_deletion_poisson", **paths, "bundle_role": "main"})

    # F: empirical distributions.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, feature in zip(axes, ANALYSIS_FEATURES):
        values = pd.to_numeric(recordings[feature], errors="coerce")
        ax.hist(values, bins=30)
        ax.set_title(feature.replace("qdist_", "").replace("_", " "))
        ax.set_ylabel("Recordings")
        ax.set_xlabel("Feature value")
    fig.tight_layout()
    paths = save_figure_bundle(
        fig, figures, "F_empirical_distributions", source=empirical_summary,
        caption=(
            "Empirical QDIST distributions. Exact zeros are valid available observations, not "
            "missing values. Positive-part summaries are reported separately because the family is sparse."
        ),
        provenance=_base_provenance("F", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "F", "stem": "F_empirical_distributions", **paths, "bundle_role": "main"})

    # G: eight deterministic full bundles from the standardized event-review package.
    for row in gallery_selection.to_dict("records"):
        item_id = str(row["review_item_id"])
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id)
        fig, source = plot_event_review_item(
            Path(row["audio_wav"]),
            item_id=item_id,
            stratum=str(row["stratum"]),
            review_label=str(row["review_label"]),
        )
        stem = f"G_{int(row['gallery_ordinal']):02d}_{safe}"
        paths = save_figure_bundle(
            fig, figures, stem, source=source,
            caption=(
                f"Deterministic label-blind QDIST signal example `{item_id}` from the "
                f"`{row['stratum']}` stratum. The bundle contains waveform, PCM/derivative, "
                "amplitude-distribution, spectrogram, and linked audio-excerpt evidence."
            ),
            provenance={
                **_base_provenance("G", parameter_hash),
                "review_item_id": item_id,
                "logical_recording_id": str(row["logical_recording_id"]),
                "stratum": str(row["stratum"]),
                "review_label": str(row["review_label"]),
                "selection_label_blind": True,
                "audio_wav": str(row["audio_wav"]),
            },
        )
        plt.close(fig)
        figure_rows.append({"panel": "G", "stem": stem, **paths, "audio_wav": str(row["audio_wav"]), "bundle_role": "gallery"})

    # H1: repeated-recording occurrence/persistence.
    occurrence = repeated_summary.loc[repeated_summary["metric"].eq("occurrence")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].bar(
        ["Both zero", "First only", "Second only", "Both positive"],
        [occurrence["both_zero_n00"], occurrence["first_only_n10"], occurrence["second_only_n01"], occurrence["both_positive_n11"]],
    )
    axes[0].set_title("First-two-recording occurrence pairs")
    axes[0].set_ylabel("Participants")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[1].bar(
        ["Overall", "Positive", "Negative"],
        [occurrence["overall_agreement"], occurrence["positive_agreement"], occurrence["negative_agreement"]],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Occurrence agreement")
    axes[1].set_ylabel("Agreement")
    fig.tight_layout()
    paths = save_figure_bundle(
        fig, figures, "H1_repeated_recording_persistence", source=repeated_summary,
        caption=(
            "Repeated-recording evidence separates occurrence agreement from positive-part "
            "magnitude persistence. Because only one first-two visit pair was positive at both "
            "visits, positive-part Spearman/ICC estimates are explicitly not treated as estimable."
        ),
        provenance=_base_provenance("H1", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "H1", "stem": "H1_repeated_recording_persistence", **paths, "bundle_role": "main"})

    # H2: related-view redundancy.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    labels = [f"{a.split('qdist_')[-1]}\nvs\n{b.split('qdist_')[-1]}" for a, b in zip(redundancy["feature_1"], redundancy["feature_2"])]
    axes[0].bar(labels, redundancy["all_recordings_spearman_rho"])
    axes[0].set_ylim(-1, 1)
    axes[0].set_title("All-recording Spearman correlation")
    axes[0].tick_params(axis="x", labelrotation=25, labelsize=8)
    axes[1].bar(labels, redundancy["positive_recordings_spearman_rho"])
    axes[1].set_ylim(-1, 1)
    axes[1].set_title("Positive-recording Spearman correlation")
    axes[1].tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.tight_layout()
    paths = save_figure_bundle(
        fig, figures, "H2_related_view_redundancy", source=redundancy,
        caption=(
            "Redundancy among frame prevalence, episode rate, and channel-sample burden. "
            "These are related views reconstructed from one accepted plateau/episode system, "
            "not independent distortion mechanisms and not components of a family scalar."
        ),
        provenance=_base_provenance("H2", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "H2", "stem": "H2_related_view_redundancy", **paths, "bundle_role": "main"})

    # H3: weighting/clustering.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].bar(weighting_summary["analysis_level"], weighting_summary["positive_fraction"])
    axes[0].errorbar(
        np.arange(len(weighting_summary)),
        weighting_summary["positive_fraction"],
        yerr=[
            weighting_summary["positive_fraction"] - weighting_summary["wilson95_low"],
            weighting_summary["wilson95_high"] - weighting_summary["positive_fraction"],
        ],
        fmt="none",
    )
    axes[0].set_title("Recording versus participant weighting")
    axes[0].set_ylabel("Positive fraction")
    axes[0].tick_params(axis="x", labelrotation=25)
    participant_counts = recordings.groupby("participant_id")["qdist_positive"].sum().value_counts().sort_index()
    participant_counts.plot.bar(ax=axes[1])
    axes[1].set_title("Positive recordings per participant")
    axes[1].set_xlabel("Positive recording count")
    axes[1].set_ylabel("Participants")
    fig.tight_layout()
    source = pd.concat([
        weighting_summary.assign(source_table="weighting_summary"),
        participant_counts.rename("participants").rename_axis("positive_recording_count").reset_index().assign(source_table="participant_clustering"),
    ], ignore_index=True, sort=False)
    paths = save_figure_bundle(
        fig, figures, "H3_participant_weighting_clustering", source=source,
        caption=(
            "Recording-weighted occurrence is compared with participant-ever-positive occurrence, "
            "with Wilson intervals and participant clustering. Participant identity is retained for "
            "grouped analysis but is not an ML predictor."
        ),
        provenance=_base_provenance("H3", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "H3", "stem": "H3_participant_weighting_clustering", **paths, "bundle_role": "main"})

    # I: event verification.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].bar(adjudication["stratum"], adjudication["review_item_count"])
    axes[0].set_title("Complete review strata")
    axes[0].set_ylabel("Review items")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[1].bar(adjudication["stratum"], adjudication["hard_clip_positive_fraction"])
    axes[1].axhline(0.90, linestyle="--", linewidth=1.0)
    axes[1].axhline(0.20, linestyle=":", linewidth=1.0)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Adjudicable hard-clip-positive fraction")
    axes[1].set_ylabel("Fraction")
    axes[1].tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    paths = save_figure_bundle(
        fig, figures, "I_event_verification", source=adjudication,
        caption=(
            "Label-blind event verification covers every accepted plateau, near-threshold rejected "
            "candidates, and deterministic valid-zero controls. Ambiguous rejected candidates are "
            "excluded from the rejected-candidate precision denominator rather than forced negative."
        ),
        provenance=_base_provenance("I", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "I", "stem": "I_event_verification", **paths, "bundle_role": "main"})

    # J: ML handoff.
    status_counts = ml_interface["qdist_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="recordings")
    contract_rows = pd.DataFrame([
        {"contract_item": "feature_values_retained", "value": True},
        {"contract_item": "status_retained", "value": True},
        {"contract_item": "exposure_retained", "value": True},
        {"contract_item": "event_count_retained", "value": True},
        {"contract_item": "poisson_interval_retained", "value": True},
        {"contract_item": "version_and_hashes_retained", "value": True},
        {"contract_item": "missing_values_imputed", "value": False},
        {"contract_item": "family_scalar_constructed", "value": False},
        {"contract_item": "standalone_gate_allowed", "value": False},
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].bar(status_counts["status"].astype(str), status_counts["recordings"])
    axes[0].set_title("ML export status counts")
    axes[0].set_ylabel("Recordings")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[1].axis("off")
    text = "\n".join(
        f"{row.contract_item}: {row.value}" for row in contract_rows.itertuples(index=False)
    )
    axes[1].text(0.02, 0.98, text, va="top", family="monospace")
    axes[1].set_title("Support-aware handoff contract")
    fig.tight_layout()
    source = pd.concat([
        status_counts.assign(source_table="status_counts"),
        contract_rows.assign(source_table="contract"),
    ], ignore_index=True, sort=False)
    paths = save_figure_bundle(
        fig, figures, "J_ml_handoff", source=source,
        caption=(
            "Support-aware, non-imputed ML handoff. Feature values remain accompanied by status, "
            "exposure, event count, exact Poisson interval, measurement version, parameter hash, "
            "and source/decode provenance. No scalar or standalone reject threshold is exported."
        ),
        provenance=_base_provenance("J", parameter_hash),
    )
    plt.close(fig)
    figure_rows.append({"panel": "J", "stem": "J_ml_handoff", **paths, "bundle_role": "main"})

    return pd.DataFrame(figure_rows)


def verify_figure_index(index: pd.DataFrame, minimum_gallery: int = GALLERY_MINIMUM) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    present_main = set(index.loc[index["panel"].ne("G"), "panel"].astype(str))
    checks.append({
        "gate": "G10",
        "check": "all applicable main panels A-J present",
        "passed": set(REQUIRED_MAIN_PANELS).issubset(present_main),
        "observed": sorted(present_main),
        "required": list(REQUIRED_MAIN_PANELS),
    })
    gallery_count = int(index["panel"].astype(str).eq("G").sum())
    checks.append({
        "gate": "G10",
        "check": "at least eight deterministic signal examples",
        "passed": gallery_count >= minimum_gallery,
        "observed": gallery_count,
        "required": f">={minimum_gallery}",
    })
    missing: list[str] = []
    for row in index.to_dict("records"):
        for field in ["png", "svg", "pdf", "source_csv", "caption", "provenance"]:
            path = Path(str(row.get(field, "")))
            if not path.exists() or path.stat().st_size == 0:
                missing.append(f"{row.get('stem')}::{field}::{path}")
    checks.append({
        "gate": "G10",
        "check": "all indexed figure artifacts exist and are nonempty",
        "passed": not missing,
        "observed": missing,
        "required": "none missing",
    })
    return pd.DataFrame(checks)


def cohort_checks(
    recordings: pd.DataFrame,
    reconstruction_summary: pd.DataFrame,
    accepted: pd.DataFrame,
    episodes: pd.DataFrame,
    margin_summary: pd.DataFrame,
    merge_summary: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    event_index: pd.DataFrame,
    event_errors: pd.DataFrame,
    adjudication: pd.DataFrame,
    figure_checks: pd.DataFrame,
    ml_interface: pd.DataFrame,
) -> pd.DataFrame:
    accepted_row = adjudication.loc[adjudication["stratum"].eq("accepted_plateau")].iloc[0]
    rejected_row = adjudication.loc[adjudication["stratum"].eq("rejected_candidate")].iloc[0]
    zero_row = adjudication.loc[adjudication["stratum"].eq("valid_zero")].iloc[0]
    merge_nondefault = merge_summary.loc[~merge_summary["merge_gap_ms"].eq(20.0)]
    rows = [
        {"gate": "G1", "check": "519 recordings and 224 participants retained", "passed": len(recordings) == 519 and recordings["participant_id"].nunique() == 224, "observed": f"{len(recordings)} recordings; {recordings['participant_id'].nunique()} participants", "required": "519 recordings; 224 participants"},
        {"gate": "G1", "check": "native signal view verified and no preprocessing", "passed": as_bool(recordings["qdist_native_view_verified"]).all() and not as_bool(recordings["qdist_known_preprocessing_applied"]).any(), "observed": f"native={as_bool(recordings['qdist_native_view_verified']).sum()}; preprocessed={as_bool(recordings['qdist_known_preprocessing_applied']).sum()}", "required": "519 native; 0 preprocessed"},
        {"gate": "G2", "check": "all three features reconstruct exactly for all recordings", "passed": bool(reconstruction_summary["passed"].all()) and reconstruction_summary["recording_count"].eq(519).all(), "observed": reconstruction_summary.to_dict("records"), "required": "max absolute difference <=2e-15 (double-precision CSV roundtrip tolerance)"},
        {"gate": "G4", "check": "accepted cohort plateaus retain explicit magnitude-path provenance", "passed": accepted["magnitude_path"].notna().all() and set(accepted["magnitude_path"].astype(str)).issubset({"strong_recording_edge", "low_level_repeated_edge"}), "observed": {"path_counts": accepted["magnitude_path"].astype(str).value_counts().to_dict(), "qualification": "Only strong_recording_edge occurred among accepted cohort plateaus; the low-level repeated-edge pathway remains construct-tested but empirically unrepresented."}, "required": "every accepted plateau has one declared allowed magnitude path; empirical presence of both paths is not required"},
        {"gate": "G5", "check": "square-like ambiguity guard passed for accepted plateaus", "passed": as_bool(accepted["square_like_guard_pass"]).all(), "observed": int(as_bool(accepted["square_like_guard_pass"]).sum()), "required": len(accepted)},
        {"gate": "G6", "check": "accepted morphology margins remain nonnegative", "passed": bool(margin_summary.loc[margin_summary["stratum"].eq("accepted"), "minimum_margin"].fillna(0).ge(-1e-12).all()), "observed": margin_summary.loc[margin_summary["stratum"].eq("accepted")].to_dict("records"), "required": "all accepted margins >=0"},
        {"gate": "G6", "check": "10/20/30/50-ms merge-gap occurrence agreement characterized", "passed": set(merge_summary["merge_gap_ms"].astype(float)) == {10.0, 20.0, 30.0, 50.0}, "observed": merge_summary.to_dict("records"), "required": [10, 20, 30, 50]},
        {"gate": "G6", "check": "merge-gap occurrence remains stable", "passed": bool(merge_nondefault["occurrence_agreement"].ge(0.99).all()), "observed": merge_nondefault[["merge_gap_ms", "occurrence_agreement", "event_count_changed_fraction"]].to_dict("records"), "required": "occurrence agreement >=0.99"},
        {"gate": "G7", "check": "availability and valid zero distinguished", "passed": recordings["qdist_available"].astype(bool).all() and set(recordings["qdist_status"].astype(str)) == {"available_no_events", "available_events"}, "observed": recordings["qdist_status"].value_counts().to_dict(), "required": "available zeros and available positives; no hidden missingness"},
        {"gate": "G7", "check": "positive prevalence remains descriptive and label-free", "passed": True, "observed": int(recordings["qdist_positive"].sum()), "required": "no prevalence tuning"},
        {"gate": "G8", "check": "three outputs labeled related views", "passed": True, "observed": list(ANALYSIS_FEATURES), "required": "no independence or scalar claim"},
        {"gate": "G8", "check": "positive-part repeated magnitude not overstated", "passed": repeated_summary.loc[repeated_summary["metric"].ne("occurrence"), "positive_part_pair_n"].lt(5).all() and repeated_summary.loc[repeated_summary["metric"].ne("occurrence"), "positive_part_spearman_rho"].isna().all(), "observed": repeated_summary.to_dict("records"), "required": "not estimable when positive pair n<5"},
        {"gate": "G9", "check": "complete event-review item count", "passed": len(event_index) == 60, "observed": len(event_index), "required": 60},
        {"gate": "G9", "check": "every event-review item declares five linked views", "passed": bool(event_index["all_required_views_present"].all()), "observed": int(event_index["all_required_views_present"].sum()), "required": len(event_index)},
        {"gate": "G9", "check": "event-review generation errors", "passed": event_errors.empty, "observed": len(event_errors), "required": 0},
        {"gate": "G9", "check": "accepted-event positive fraction", "passed": float(accepted_row["hard_clip_positive_fraction"]) >= 0.90, "observed": float(accepted_row["hard_clip_positive_fraction"]), "required": ">=0.90"},
        {"gate": "G9", "check": "rejected-candidate positive fraction", "passed": float(rejected_row["hard_clip_positive_fraction"]) <= 0.20, "observed": float(rejected_row["hard_clip_positive_fraction"]), "required": "<=0.20"},
        {"gate": "G9", "check": "valid-zero positive fraction", "passed": float(zero_row["hard_clip_positive_fraction"]) == 0.0, "observed": float(zero_row["hard_clip_positive_fraction"]), "required": 0.0},
        {"gate": "G9", "check": "event review is label-blind to clinical and human-QC outcomes", "passed": bool(event_index["selection_label_blind"].all()), "observed": True, "required": True},
        {"gate": "G10", "check": "support-aware non-imputed ML interface complete", "passed": len(ml_interface) == 519 and not ml_interface["qdist_missing_values_imputed"].any() and not ml_interface["qdist_family_scalar_constructed"].any(), "observed": len(ml_interface), "required": 519},
    ]
    rows.extend(figure_checks.to_dict("records"))
    return pd.DataFrame(rows)


def provisional_feature_decisions(merge_summary: pd.DataFrame) -> pd.DataFrame:
    nondefault = merge_summary.loc[~merge_summary["merge_gap_ms"].eq(20.0)]
    stable = bool(nondefault["occurrence_agreement"].ge(0.99).all())
    return pd.DataFrame([
        {
            "feature": "qdist_hard_clipped_frame_fraction",
            "provisional_role": "RETAIN_PRIMARY",
            "basis": "Exact reconstruction; conservative construct response; sparse cohort prevalence; independent of episode merge gap.",
            "finalization_status": "PENDING_POST_COHORT_REVIEW",
        },
        {
            "feature": "qdist_hard_clip_event_rate_per_min",
            "provisional_role": "RETAIN_PRIMARY_EVENT" if stable else "REVISE_OR_AUDIT_ONLY",
            "basis": "Exact episode-ledger reconstruction; merge-gap sensitivity evaluated separately; exact Poisson uncertainty retained.",
            "finalization_status": "PENDING_POST_COHORT_REVIEW",
        },
        {
            "feature": "qdist_hard_clipped_sample_fraction",
            "provisional_role": "RETAIN_SECONDARY",
            "basis": "Exact channel-sample burden; sparse absolute clipped ms/min emphasized; related to frame/event views.",
            "finalization_status": "PENDING_POST_COHORT_REVIEW",
        },
    ])


def run_cohort_review(
    project_root: str | Path,
    *,
    build_event_review: bool = True,
    rebuild_event_review: bool = False,
    scientific_review_decision: str = "PENDING",
    publish_and_freeze: bool = False,
) -> dict[str, Any]:
    if publish_and_freeze:
        raise ValueError("Cohort evidence notebook cannot publish or freeze QDIST.")
    if scientific_review_decision != "PENDING":
        raise ValueError("Cohort evidence notebook must retain SCIENTIFIC_REVIEW_DECISION='PENDING'.")

    paths = CohortPaths.from_project_root(project_root)
    output = paths.output_root
    for directory in [
        output / "tables", output / "validation", output / "audit", output / "figures",
        output / "manifests", output / "event_review",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    preflight_manifest, preflight_index = verify_preflight_bundle(paths.preflight_root)
    frozen_manifest, data = verify_frozen_baseline(paths.frozen_root)
    parameters = frozen_manifest["parameters"]
    parameter_hash = str(data["recordings"]["qdist_parameter_hash"].iloc[0])

    recording_table = prepare_recording_table(data["analysis"], data["recordings"])
    reconstruction_long, reconstruction_summary = build_reconstruction_audit(
        data["recordings"], data["accepted"], data["episodes"]
    )
    empirical = feature_summary(recording_table)
    margins = morphology_margin_table(data["candidates"], parameters)
    merge_long, merge_summary = merge_gap_sensitivity(recording_table, data["accepted"])
    deletion_long, deletion_summary = deletion_influence(recording_table, data["accepted"], data["episodes"])
    repeated_pairs, all_pairs, repeated_summary = repeated_recording_evidence(recording_table)
    redundancy = redundancy_table(recording_table)
    participant_table, weighting_summary = participant_weighting(recording_table)
    ml_interface = build_ml_interface(recording_table)

    if build_event_review:
        event_index, adjudication, event_errors = build_event_review_package(
            paths.frozen_root,
            output,
            data["gallery_index"],
            data["gallery_review"],
            rebuild=rebuild_event_review,
        )
    else:
        event_index = load_csv(output / "tables" / "qdist_v400_event_review_index.csv")
        adjudication = load_csv(output / "validation" / "qdist_v400_event_adjudication_summary.csv")
        event_errors = load_csv(output / "audit" / "qdist_v400_event_review_errors.csv", required=False)

    gallery_selection = select_gallery_examples(event_index)

    figure_index = create_cohort_figures(
        output,
        preflight_index,
        recording_table,
        data["accepted"],
        data["episodes"],
        data["parameter_summary"],
        margins,
        merge_summary,
        deletion_summary,
        empirical,
        repeated_summary,
        redundancy,
        weighting_summary,
        adjudication,
        ml_interface,
        event_index,
        gallery_selection,
        parameter_hash=parameter_hash,
    )
    figure_checks = verify_figure_index(figure_index)
    checks = cohort_checks(
        recording_table,
        reconstruction_summary,
        data["accepted"],
        data["episodes"],
        margins,
        merge_summary,
        repeated_summary,
        event_index,
        event_errors,
        adjudication,
        figure_checks,
        ml_interface,
    )
    decisions = provisional_feature_decisions(merge_summary)

    # Persist all evidence.
    save_table(recording_table, output / "tables" / "qdist_v400_recording_features")
    save_table(reconstruction_long, output / "validation" / "qdist_v400_reconstruction_long")
    save_table(reconstruction_summary, output / "validation" / "qdist_v400_reconstruction_summary")
    save_table(empirical, output / "validation" / "qdist_v400_empirical_feature_summary")
    save_table(margins, output / "validation" / "qdist_v400_morphology_margin_summary")
    save_table(data["parameter_long"], output / "validation" / "qdist_v400_parameter_sensitivity_long")
    save_table(data["parameter_summary"], output / "validation" / "qdist_v400_parameter_sensitivity_summary")
    save_table(merge_long, output / "validation" / "qdist_v400_merge_gap_sensitivity_long")
    save_table(merge_summary, output / "validation" / "qdist_v400_merge_gap_sensitivity_summary")
    save_table(deletion_long, output / "validation" / "qdist_v400_deletion_influence_long")
    save_table(deletion_summary, output / "validation" / "qdist_v400_deletion_influence_summary")
    save_table(repeated_pairs, output / "validation" / "qdist_v400_repeated_recording_first_pair")
    save_table(all_pairs, output / "validation" / "qdist_v400_repeated_recording_all_pairs")
    save_table(repeated_summary, output / "validation" / "qdist_v400_repeated_recording_summary")
    save_table(redundancy, output / "validation" / "qdist_v400_related_view_redundancy")
    save_table(participant_table, output / "validation" / "qdist_v400_participant_summary")
    save_table(weighting_summary, output / "validation" / "qdist_v400_weighting_summary")
    save_table(event_index, output / "tables" / "qdist_v400_event_review_index")
    save_table(adjudication, output / "validation" / "qdist_v400_event_adjudication_summary")
    save_table(event_errors, output / "audit" / "qdist_v400_event_review_errors", parquet=False)
    save_table(gallery_selection, output / "tables" / "qdist_v400_gallery_selection", parquet=False)
    save_table(ml_interface, output / "tables" / "qdist_v400_ml_interface")
    save_table(figure_index, output / "tables" / "qdist_v400_figure_index", parquet=False)
    save_table(checks, output / "validation" / "qdist_v400_cohort_checks", parquet=False)
    save_table(decisions, output / "validation" / "qdist_v400_g10_feature_decisions_provisional", parquet=False)

    blocking_pass = bool(as_bool(checks["passed"]).all())
    manifest = {
        "measurement_version": MEASUREMENT_VERSION,
        "legacy_measurement_version": LEGACY_MEASUREMENT_VERSION,
        "reviewed_orchestration_version": COHORT_ORCHESTRATION_VERSION,
        "created_utc": utc_now(),
        "candidate_only": True,
        "accepted_preflight": True,
        "preflight_blocking_checks_pass": bool(preflight_manifest["preflight_blocking_checks_pass"]),
        "package_tests_passed": bool(preflight_manifest["package_tests_passed"]),
        "cohort_extraction_completed": True,
        "cohort_standardization_completed": True,
        "cohort_evidence_complete": blocking_pass,
        "recording_count": len(recording_table),
        "participant_count": recording_table["participant_id"].nunique(),
        "available_recording_count": int(recording_table["qdist_available"].astype(bool).sum()),
        "positive_recording_count": int(recording_table["qdist_positive"].sum()),
        "valid_zero_recording_count": int(recording_table["qdist_valid_zero"].sum()),
        "accepted_plateau_count": len(data["accepted"]),
        "episode_count": len(data["episodes"]),
        "candidate_plateau_count": len(data["candidates"]),
        "event_review_item_count": len(event_index),
        "event_review_error_count": len(event_errors),
        "event_review_all_five_views": bool(event_index["all_required_views_present"].all()),
        "gallery_bundle_count": int(figure_index["panel"].eq("G").sum()),
        "main_figure_bundle_count": int(figure_index["panel"].ne("G").sum()),
        "figure_bundle_count": len(figure_index),
        "required_panels_complete": bool(as_bool(figure_checks["passed"]).all()),
        "panel_i_status": "APPLICABLE_complete_event_verification",
        "numerical_equivalence_to_qdist_v311": bool(reconstruction_summary["passed"].all()),
        "feature_values_recomputed": False,
        "parameter_hash": parameter_hash,
        "analysis_features": list(ANALYSIS_FEATURES),
        "family_scalar_constructed": False,
        "standalone_gate_allowed": False,
        "complete_nonlinear_distortion_claim_allowed": False,
        "missing_values_imputed": False,
        "scientific_review_decision": "PENDING",
        "freeze_allowed": False,
        "publish_and_freeze": False,
        "g10_final_decisions_complete": False,
        "frozen_manifest_sha256": sha256_file(paths.frozen_root / "audit" / "qdist_v311_frozen_manifest.json"),
    }
    write_json(manifest, output / "manifests" / "qdist_v400_cohort_candidate_manifest.json")
    return {
        "manifest": manifest,
        "checks": checks,
        "recording_table": recording_table,
        "figure_index": figure_index,
        "event_index": event_index,
        "adjudication": adjudication,
        "decisions": decisions,
    }
