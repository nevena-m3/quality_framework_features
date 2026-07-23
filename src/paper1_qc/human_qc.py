from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FILE_ALIASES = ["file_name", "filename", "raw_media_file_name", "raw_media_file", "recording"]
RATER_ALIASES = ["rater_id", "rater", "annotator", "reviewer", "ra"]


# These are perceptual artifact families, not putative source labels. The mapping is
# intentionally frozen before objective Q values are inspected.
DEFAULT_INTERVAL_FAMILY_MAP = {
    "Environmental noise": "additive_interference",
    "Volume unstable": "gain_dynamics",
    "Reverberation/echo": "reverberation_tail",
    "Platform effects": "channel_device",
    "Clipping": "nonlinear_distortion",
    "Temporal discontinuities": "temporal_discontinuity",
}

# These annotations remain useful for audit and sensitivity work, but they do not have a
# one-to-one mechanistic Q-family target and are excluded from matched-family validation.
DEFAULT_CONTEXT_COLUMNS = {
    "Any non-task related content": "task_content_contamination",
    "Competing speech": "competing_speech_context",
}


def _snake(value: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())).strip("_")


def _basename(value: object) -> str:
    return re.split(r"[\\/]", str(value).strip())[-1]


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    lookup = {_snake(column): column for column in columns}
    for alias in aliases:
        if _snake(alias) in lookup:
            return lookup[_snake(alias)]
    return None


def _read_rating_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported human-QC file: {path}")


def _merge_intervals(
    intervals: list[tuple[float, float]], *, tolerance_sec: float = 1e-6
) -> list[tuple[float, float]]:
    """Return the union of intervals so overlapping sublabels are not double-counted."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + tolerance_sec:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _read_rater_manifest(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Human-QC rater manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype="string")
    required = {"relative_path", "rater_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Rater manifest is missing required columns: {sorted(missing)}")
    manifest["relative_path"] = (
        manifest["relative_path"].str.replace("\\", "/", regex=False).str.strip().str.lstrip("./")
    )
    manifest["rater_id"] = manifest["rater_id"].str.strip()
    if manifest["relative_path"].duplicated().any():
        duplicates = manifest.loc[manifest["relative_path"].duplicated(False), "relative_path"].tolist()
        raise ValueError(f"Rater manifest has duplicate relative_path values: {duplicates[:10]}")
    if manifest["rater_id"].isna().any() or manifest["rater_id"].eq("").any():
        raise ValueError("Every rater manifest row must contain a non-empty rater_id")
    return dict(zip(manifest["relative_path"], manifest["rater_id"]))


def _resolve_interval_rater(
    source: Path,
    root: Path,
    manifest: dict[str, str],
    *,
    strategy: str,
    rater_directory_names: list[str] | None = None,
) -> tuple[str, str | None]:
    relative = source.relative_to(root).as_posix() if root.is_dir() else source.name
    if relative in manifest:
        return manifest[relative], None
    if rater_directory_names and root.is_dir():
        canonical = {name.casefold(): name for name in rater_directory_names}
        matches = [
            canonical[part.casefold()]
            for part in source.relative_to(root).parts[:-1]
            if part.casefold() in canonical
        ]
        if len(set(matches)) == 1:
            # These directory names were explicitly declared in the frozen schema, so
            # this is deterministic rater identification rather than heuristic inference.
            return matches[0], None
        if len(set(matches)) > 1:
            return f"UNRESOLVED::{relative}", "multiple_rater_directories_in_path"
    if strategy == "parent_directory" and root.is_dir() and source.parent != root:
        return source.parent.name.strip(), "rater_inferred_from_parent_directory"
    # A segment filename is a recording identifier in the supplied exports. Treating its
    # stem as a rater would create one pseudo-rater per recording and invalid agreement.
    return f"UNRESOLVED::{relative}", "rater_identity_unresolved"


def load_interval_human_qc(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    rater_strategy: str = "manifest",
    family_map: dict[str, str] | None = None,
    context_columns: dict[str, str] | None = None,
    interval_time_base: str = "absolute",
    boundary_tolerance_sec: float = 0.05,
    exclude_path_parts: list[str] | None = None,
    rater_directory_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse the supplied segment-annotation JSON exports.

    Returns four tables:
    ``family_ratings`` (one binary family-presence row per item/rater/family),
    ``context_ratings`` (non-family content/source annotations),
    ``annotation_intervals`` (auditable event-level intervals), and ``issues``.

    The parser never infers a rater from the recording filename. Rater identity must come
    from an explicit manifest or, when configured, from the immediate parent directory.
    """
    root = Path(path)
    excluded = {part.casefold() for part in (exclude_path_parts or [])}
    files = [root] if root.is_file() else sorted(
        candidate
        for candidate in root.rglob("*.csv")
        if not any(part.casefold() in excluded for part in candidate.relative_to(root).parts)
    )
    if not files:
        raise FileNotFoundError(f"No segment-annotation CSV files found under {root}")
    manifest = _read_rater_manifest(manifest_path)
    family_map = family_map or DEFAULT_INTERVAL_FAMILY_MAP
    context_columns = context_columns or DEFAULT_CONTEXT_COLUMNS
    if interval_time_base not in {"absolute", "segment_relative"}:
        raise ValueError("interval_time_base must be 'absolute' or 'segment_relative'")

    interval_rows: list[dict] = []
    issue_rows: list[dict] = []
    recording_rows: list[dict] = []
    for source in files:
        relative = source.relative_to(root).as_posix() if root.is_dir() else source.name
        rater_id, rater_issue = _resolve_interval_rater(
            source,
            root,
            manifest,
            strategy=rater_strategy,
            rater_directory_names=rater_directory_names,
        )
        if rater_issue:
            issue_rows.append(
                {
                    "source_file": relative,
                    "severity": (
                        "error"
                        if rater_issue
                        in {
                            "rater_identity_unresolved",
                            "multiple_rater_directories_in_path",
                        }
                        else "review"
                    ),
                    "issue": rater_issue,
                    "rater_id": rater_id,
                }
            )
        frame = pd.read_csv(source)
        file_column = _find_column(frame.columns, FILE_ALIASES)
        required = {
            "onset_seconds_absolute",
            "offset_seconds_absolute",
            "duration_seconds",
        }
        missing = required - set(frame.columns)
        if file_column is None or missing:
            issue_rows.append(
                {
                    "source_file": relative,
                    "severity": "error",
                    "issue": "missing_required_interval_columns",
                    "detail": ",".join(sorted(missing | ({"file_name"} if file_column is None else set()))),
                }
            )
            continue
        available_annotation_columns = [
            column for column in [*family_map, *context_columns] if column in frame.columns
        ]
        missing_family_columns = [column for column in family_map if column not in frame.columns]
        for column in missing_family_columns:
            issue_rows.append(
                {
                    "source_file": relative,
                    "severity": "error",
                    "issue": "missing_family_annotation_column",
                    "category_column": column,
                }
            )
        for file_value, file_frame in frame.groupby(file_column, dropna=False, sort=False):
            file_name = _basename(file_value)
            starts = pd.to_numeric(file_frame["onset_seconds_absolute"], errors="coerce")
            ends = pd.to_numeric(file_frame["offset_seconds_absolute"], errors="coerce")
            duration = float(ends.max()) if ends.notna().any() else np.nan
            recording_rows.append(
                {
                    "file_name": file_name,
                    "rater_id": rater_id,
                    "recording_duration_sec": duration,
                    "source_file": relative,
                }
            )
            for row_index, row in file_frame.iterrows():
                segment_start = pd.to_numeric(
                    pd.Series([row["onset_seconds_absolute"]]), errors="coerce"
                ).iloc[0]
                segment_end = pd.to_numeric(
                    pd.Series([row["offset_seconds_absolute"]]), errors="coerce"
                ).iloc[0]
                if not np.isfinite(segment_start) or not np.isfinite(segment_end):
                    issue_rows.append(
                        {
                            "source_file": relative,
                            "severity": "error",
                            "issue": "non_numeric_segment_boundary",
                            "row_index": int(row_index),
                        }
                    )
                    continue
                for category_column in available_annotation_columns:
                    raw = row[category_column]
                    if pd.isna(raw) or str(raw).strip() == "":
                        issue_rows.append(
                            {
                                "source_file": relative,
                                "severity": "error",
                                "issue": "missing_annotation_json",
                                "row_index": int(row_index),
                                "category_column": category_column,
                            }
                        )
                        continue
                    try:
                        payload = json.loads(str(raw))
                    except json.JSONDecodeError as exc:
                        issue_rows.append(
                            {
                                "source_file": relative,
                                "severity": "error",
                                "issue": "invalid_annotation_json",
                                "row_index": int(row_index),
                                "category_column": category_column,
                                "detail": str(exc),
                            }
                        )
                        continue
                    if not isinstance(payload, dict):
                        issue_rows.append(
                            {
                                "source_file": relative,
                                "severity": "error",
                                "issue": "annotation_json_not_object",
                                "row_index": int(row_index),
                                "category_column": category_column,
                            }
                        )
                        continue
                    family = family_map.get(category_column)
                    context_group = context_columns.get(category_column)
                    for subcategory, raw_intervals in payload.items():
                        if not isinstance(raw_intervals, list):
                            issue_rows.append(
                                {
                                    "source_file": relative,
                                    "severity": "error",
                                    "issue": "annotation_intervals_not_list",
                                    "row_index": int(row_index),
                                    "category_column": category_column,
                                    "subcategory": subcategory,
                                }
                            )
                            continue
                        for event_number, pair in enumerate(raw_intervals, start=1):
                            valid_pair = isinstance(pair, list) and len(pair) == 2
                            try:
                                event_start, event_end = (float(pair[0]), float(pair[1]))
                            except (TypeError, ValueError, IndexError):
                                valid_pair = False
                            if not valid_pair or not np.isfinite(event_start) or not np.isfinite(event_end):
                                issue_rows.append(
                                    {
                                        "source_file": relative,
                                        "severity": "error",
                                        "issue": "invalid_annotation_interval",
                                        "row_index": int(row_index),
                                        "category_column": category_column,
                                        "subcategory": subcategory,
                                        "event_number": event_number,
                                    }
                                )
                                continue
                            if interval_time_base == "segment_relative":
                                event_start += float(segment_start)
                                event_end += float(segment_start)
                            if event_end <= event_start:
                                issue_rows.append(
                                    {
                                        "source_file": relative,
                                        "severity": "error",
                                        "issue": "nonpositive_annotation_interval",
                                        "row_index": int(row_index),
                                        "category_column": category_column,
                                        "subcategory": subcategory,
                                        "event_number": event_number,
                                    }
                                )
                                continue
                            if (
                                event_start < segment_start - boundary_tolerance_sec
                                or event_end > segment_end + boundary_tolerance_sec
                            ):
                                issue_rows.append(
                                    {
                                        "source_file": relative,
                                        "severity": "error",
                                        "issue": "annotation_outside_segment_boundary",
                                        "row_index": int(row_index),
                                        "category_column": category_column,
                                        "subcategory": subcategory,
                                        "event_number": event_number,
                                        "event_start_sec": event_start,
                                        "event_end_sec": event_end,
                                        "segment_start_sec": float(segment_start),
                                        "segment_end_sec": float(segment_end),
                                    }
                                )
                                continue
                            interval_rows.append(
                                {
                                    "file_name": file_name,
                                    "rater_id": rater_id,
                                    "family": family,
                                    "context_group": context_group,
                                    "category_column": category_column,
                                    "subcategory": str(subcategory),
                                    "start_sec": event_start,
                                    "end_sec": event_end,
                                    "duration_sec": event_end - event_start,
                                    "segment_start_sec": float(segment_start),
                                    "segment_end_sec": float(segment_end),
                                    "source_file": relative,
                                }
                            )

    recordings = pd.DataFrame(recording_rows)
    intervals = pd.DataFrame(interval_rows)
    duplicates = recordings.duplicated(["file_name", "rater_id"], keep=False)
    for row in recordings.loc[duplicates].itertuples():
        issue_rows.append(
            {
                "source_file": row.source_file,
                "severity": "error",
                "issue": "duplicate_recording_for_same_rater",
                "file_name": row.file_name,
                "rater_id": row.rater_id,
            }
        )

    def summarize(*, grouping_column: str, expected_values: list[str]) -> pd.DataFrame:
        rows = []
        for recording in recordings.itertuples():
            for value in expected_values:
                if intervals.empty:
                    subset = intervals
                else:
                    subset = intervals.loc[
                        (intervals["file_name"] == recording.file_name)
                        & (intervals["rater_id"] == recording.rater_id)
                        & (intervals[grouping_column] == value)
                    ]
                raw_intervals = list(zip(subset.get("start_sec", []), subset.get("end_sec", [])))
                union = _merge_intervals(raw_intervals)
                annotated_duration = float(sum(end - start for start, end in union))
                denominator = float(recording.recording_duration_sec)
                rows.append(
                    {
                        "file_name": recording.file_name,
                        "rater_id": recording.rater_id,
                        "category": value,
                        "rating": int(annotated_duration > 0),
                        "annotated_duration_sec": annotated_duration,
                        "annotated_fraction": (
                            annotated_duration / denominator
                            if np.isfinite(denominator) and denominator > 0
                            else np.nan
                        ),
                        "event_count_raw": len(raw_intervals),
                        "event_count_union": len(union),
                        "recording_duration_sec": denominator,
                        "source_file": recording.source_file,
                    }
                )
        return pd.DataFrame(rows)

    family_ratings = summarize(
        grouping_column="family", expected_values=sorted(set(family_map.values()))
    )
    context_ratings = summarize(
        grouping_column="context_group", expected_values=sorted(set(context_columns.values()))
    )
    return family_ratings, context_ratings, intervals, pd.DataFrame(issue_rows)


def rating_design_coverage(
    ratings: pd.DataFrame, *, expected_raters: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit whether the detailed rating design supports a four-RA agreement analysis."""
    if ratings.empty:
        item_family = pd.DataFrame(
            columns=[
                "file_name",
                "category",
                "n_raters",
                "rater_ids",
                "expected_raters",
                "complete_expected_raters",
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "category": pd.NA,
                    "items_total": 0,
                    "items_with_2plus_raters": 0,
                    "items_complete_expected_raters": 0,
                    "minimum_raters_per_item": 0,
                    "maximum_raters_per_item": 0,
                    "expected_raters": expected_raters,
                    "raters_total_study": 0,
                    "design_status": "blocked_no_valid_rating_rows",
                }
            ]
        )
        return item_family, summary
    item_family = (
        ratings.groupby(["file_name", "category"], as_index=False)
        .agg(
            n_raters=("rater_id", "nunique"),
            rater_ids=("rater_id", lambda values: "|".join(sorted(map(str, set(values))))),
        )
    )
    item_family["expected_raters"] = expected_raters
    item_family["complete_expected_raters"] = item_family["n_raters"].eq(expected_raters)
    summary = (
        item_family.groupby("category", as_index=False)
        .agg(
            items_total=("file_name", "nunique"),
            items_with_2plus_raters=("n_raters", lambda values: int((values >= 2).sum())),
            items_complete_expected_raters=("complete_expected_raters", "sum"),
            minimum_raters_per_item=("n_raters", "min"),
            maximum_raters_per_item=("n_raters", "max"),
        )
    )
    summary["expected_raters"] = expected_raters
    total_raters = int(ratings["rater_id"].nunique())
    summary["raters_total_study"] = total_raters
    summary["design_status"] = np.select(
        [
            summary["items_complete_expected_raters"].le(0),
            pd.Series(total_raters != expected_raters, index=summary.index),
        ],
        [
            "blocked_no_item_has_all_expected_raters",
            "blocked_total_rater_ids_not_expected",
        ],
        default="agreement_estimable_on_complete_subset",
    )
    return item_family, summary


def load_human_qc_long(
    path: str | Path,
    *,
    file_column: str | None = None,
    rater_column: str | None = None,
    category_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a folder/file of independent ratings into item-rater-category long format.

    Accepted inputs are either an already-long table with ``category`` and ``rating``
    columns, or one/more wide CSV/XLSX files. When a wide file has no rater column,
    its filename stem is retained as the rater identifier and must be reviewed.
    """
    root = Path(path)
    files = [root] if root.is_file() else sorted(
        candidate for candidate in root.rglob("*") if candidate.suffix.lower() in {".csv", ".xlsx", ".xls"}
    )
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX human-QC files found under {root}")

    long_frames = []
    issue_rows = []
    for source in files:
        frame = _read_rating_file(source)
        normalized_columns = {_snake(column): column for column in frame.columns}
        if {"category", "rating"}.issubset(normalized_columns):
            current_file = file_column or _find_column(frame.columns, FILE_ALIASES)
            current_rater = rater_column or _find_column(frame.columns, RATER_ALIASES)
            if current_file is None:
                issue_rows.append({"source": source.name, "severity": "error", "issue": "missing_file_column"})
                continue
            work = pd.DataFrame(
                {
                    "file_name": frame[current_file].map(_basename),
                    "rater_id": frame[current_rater] if current_rater else source.stem,
                    "category": frame[normalized_columns["category"]].map(_snake),
                    "rating": frame[normalized_columns["rating"]],
                    "source_file": source.name,
                }
            )
            if current_rater is None:
                issue_rows.append({"source": source.name, "severity": "review", "issue": "rater_inferred_from_filename"})
            long_frames.append(work)
            continue

        current_file = file_column or _find_column(frame.columns, FILE_ALIASES)
        current_rater = rater_column or _find_column(frame.columns, RATER_ALIASES)
        if current_file is None:
            issue_rows.append({"source": source.name, "severity": "error", "issue": "missing_file_column"})
            continue
        mappings = category_columns or {
            _snake(column): column
            for column in frame.columns
            if column not in {current_file, current_rater}
        }
        id_vars = [current_file] + ([current_rater] if current_rater else [])
        selected = frame[id_vars + [column for column in mappings.values() if column in frame.columns]].copy()
        melted = selected.melt(id_vars=id_vars, var_name="source_category", value_name="rating")
        inverse = {column: _snake(category) for category, column in mappings.items()}
        work = pd.DataFrame(
            {
                "file_name": melted[current_file].map(_basename),
                "rater_id": melted[current_rater] if current_rater else source.stem,
                "category": melted["source_category"].map(inverse),
                "rating": melted["rating"],
                "source_file": source.name,
            }
        )
        if current_rater is None:
            issue_rows.append({"source": source.name, "severity": "review", "issue": "rater_inferred_from_filename"})
        long_frames.append(work)

    if not long_frames:
        raise ValueError("No human-QC ratings could be standardized")
    ratings = pd.concat(long_frames, ignore_index=True)
    ratings["rater_id"] = ratings["rater_id"].astype(str).str.strip()
    ratings["category"] = ratings["category"].map(_snake)
    ratings = ratings.loc[ratings["rating"].notna()].reset_index(drop=True)

    duplicates = ratings.duplicated(["file_name", "rater_id", "category"], keep=False)
    for _, row in ratings.loc[duplicates].iterrows():
        issue_rows.append(
            {
                "source": row["source_file"],
                "severity": "error",
                "issue": "duplicate_item_rater_category",
                "file_name": row["file_name"],
                "rater_id": row["rater_id"],
                "category": row["category"],
            }
        )
    return ratings, pd.DataFrame(issue_rows)


def _categorical_matrix(category_frame: pd.DataFrame) -> pd.DataFrame:
    return category_frame.pivot(index="file_name", columns="rater_id", values="rating")


def gwet_ac1(matrix: pd.DataFrame) -> float:
    """Multi-rater Gwet AC1 for nominal ratings with incomplete rows allowed."""
    values = sorted(pd.unique(matrix.to_numpy().ravel()[pd.notna(matrix.to_numpy().ravel())]), key=str)
    if len(values) < 2:
        return np.nan
    category_index = {value: index for index, value in enumerate(values)}
    pair_agreements = []
    counts_total = np.zeros(len(values), dtype=float)
    ratings_total = 0
    for _, row in matrix.iterrows():
        observed = row.dropna().tolist()
        if len(observed) < 2:
            continue
        counts = np.zeros(len(values), dtype=float)
        for value in observed:
            counts[category_index[value]] += 1
        r = len(observed)
        pair_agreements.append(float(np.sum(counts * (counts - 1)) / (r * (r - 1))))
        counts_total += counts
        ratings_total += r
    if not pair_agreements or ratings_total == 0:
        return np.nan
    p = counts_total / ratings_total
    chance = float(np.sum(p * (1 - p)) / (len(values) - 1))
    observed_agreement = float(np.mean(pair_agreements))
    return (observed_agreement - chance) / (1 - chance) if chance < 1 else np.nan


def observed_pair_agreement(matrix: pd.DataFrame) -> float:
    agreements = []
    for _, row in matrix.iterrows():
        observed = row.dropna().to_numpy()
        if len(observed) < 2:
            continue
        for left in range(len(observed) - 1):
            for right in range(left + 1, len(observed)):
                agreements.append(observed[left] == observed[right])
    return float(np.mean(agreements)) if agreements else np.nan


def _fleiss_kappa_complete(matrix: pd.DataFrame) -> float:
    complete = matrix.dropna(axis=0, how="any")
    if len(complete) < 2 or complete.shape[1] < 2:
        return np.nan
    categories = sorted(pd.unique(complete.to_numpy().ravel()), key=str)
    counts = np.column_stack([(complete == category).sum(axis=1).to_numpy() for category in categories])
    n_raters = complete.shape[1]
    agreement_by_item = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    observed = float(np.mean(agreement_by_item))
    proportions = counts.sum(axis=0) / counts.sum()
    chance = float(np.sum(proportions**2))
    return (observed - chance) / (1 - chance) if chance < 1 else np.nan


def icc_two_way_random_absolute(matrix: pd.DataFrame) -> tuple[float, float]:
    """Return ICC(2,1) and ICC(2,k) for a complete target-by-rater numeric matrix."""
    complete = matrix.dropna(axis=0, how="any").astype(float)
    n, k = complete.shape
    if n < 3 or k < 2:
        return np.nan, np.nan
    values = complete.to_numpy()
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_columns = n * np.sum((column_means - grand) ** 2)
    ss_error = np.sum((values - row_means[:, None] - column_means[None, :] + grand) ** 2)
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    icc_single = (ms_rows - ms_error) / (
        ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    )
    icc_average = (ms_rows - ms_error) / (ms_rows + (ms_columns - ms_error) / n)
    return float(icc_single), float(icc_average)


def _bootstrap_matrix_metric(
    matrix: pd.DataFrame,
    statistic,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    if replicates <= 0 or len(matrix) < 3:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        positions = rng.integers(0, len(matrix), len(matrix))
        sample = matrix.iloc[positions].reset_index(drop=True)
        value = statistic(sample)
        if np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return np.nan, np.nan, 0
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high), len(estimates)


def agreement_summary(
    ratings: pd.DataFrame, *, bootstrap_replicates: int = 0, seed: int = 20260713
) -> pd.DataFrame:
    rows = []
    for category, frame in ratings.groupby("category", sort=True):
        matrix = _categorical_matrix(frame)
        complete = matrix.dropna(axis=0, how="any")
        values = pd.unique(frame["rating"].dropna())
        kappa = _fleiss_kappa_complete(matrix)
        numeric = pd.to_numeric(frame["rating"], errors="coerce").notna().all()
        numeric_for_icc = numeric and len(values) >= 3
        icc_single, icc_average = (
            icc_two_way_random_absolute(matrix) if numeric_for_icc else (np.nan, np.nan)
        )
        ac1_low, ac1_high, ac1_boot = _bootstrap_matrix_metric(
            matrix, gwet_ac1, replicates=bootstrap_replicates, seed=seed
        )
        kappa_low, kappa_high, kappa_boot = _bootstrap_matrix_metric(
            matrix, _fleiss_kappa_complete, replicates=bootstrap_replicates, seed=seed + 1
        )
        if numeric_for_icc:
            icc1_low, icc1_high, icc_boot = _bootstrap_matrix_metric(
                matrix,
                lambda sample: icc_two_way_random_absolute(sample)[0],
                replicates=bootstrap_replicates,
                seed=seed + 2,
            )
            icck_low, icck_high, _ = _bootstrap_matrix_metric(
                matrix,
                lambda sample: icc_two_way_random_absolute(sample)[1],
                replicates=bootstrap_replicates,
                seed=seed + 3,
            )
        else:
            icc1_low = icc1_high = icck_low = icck_high = np.nan
            icc_boot = 0
        rows.append(
            {
                "category": category,
                "items_total": matrix.shape[0],
                "raters_total": matrix.shape[1],
                "items_complete_all_raters": len(complete),
                "rating_levels": len(values),
                "observed_pair_agreement": observed_pair_agreement(matrix),
                "gwet_ac1_nominal": gwet_ac1(matrix),
                "gwet_ac1_ci_low": ac1_low,
                "gwet_ac1_ci_high": ac1_high,
                "fleiss_kappa": kappa,
                "fleiss_kappa_ci_low": kappa_low,
                "fleiss_kappa_ci_high": kappa_high,
                "icc_2_1_absolute": icc_single,
                "icc_2_1_ci_low": icc1_low,
                "icc_2_1_ci_high": icc1_high,
                "icc_2_k_absolute": icc_average,
                "icc_2_k_ci_low": icck_low,
                "icc_2_k_ci_high": icck_high,
                "bootstrap_successful_ac1": ac1_boot,
                "bootstrap_successful_kappa": kappa_boot,
                "bootstrap_successful_icc": icc_boot,
            }
        )
    return pd.DataFrame(rows)


def make_consensus(
    ratings: pd.DataFrame,
    *,
    expected_raters: int | None = None,
    minimum_ratings: int | None = None,
) -> pd.DataFrame:
    """Create an explicit consensus after applying a pre-specified coverage gate.

    Tied modes remain missing and are never broken arbitrarily. For the primary four-RA
    analysis, use ``expected_raters=4`` and ``minimum_ratings=4``. A three-of-four
    sensitivity consensus must be saved separately rather than silently mixed with the
    primary complete design.
    """
    rows = []
    for (file_name, category), frame in ratings.groupby(["file_name", "category"], sort=True):
        observed = frame["rating"].dropna()
        modes = observed.mode(dropna=True)
        tied = len(modes) != 1
        numeric = pd.to_numeric(observed, errors="coerce")
        required = minimum_ratings if minimum_ratings is not None else 1
        if expected_raters is not None and frame["rater_id"].nunique() > expected_raters:
            consensus = np.nan
            method = "too_many_raters_requires_design_review"
            tied = False
        elif len(observed) < required:
            consensus = np.nan
            method = "insufficient_rater_coverage"
            tied = False
        elif len(observed) == 0:
            consensus = np.nan
            method = "missing"
        elif numeric.notna().all() and observed.nunique() > 2:
            consensus = float(numeric.median())
            method = "median_ordinal"
            tied = False
        elif tied:
            consensus = np.nan
            method = "tie_requires_adjudication"
        else:
            consensus = modes.iloc[0]
            method = "majority_mode"
        rows.append(
            {
                "file_name": file_name,
                "category": category,
                "consensus_rating": consensus,
                "consensus_method": method,
                "n_ratings": len(observed),
                "n_unique_ratings": observed.nunique(),
                "requires_adjudication": tied,
                "expected_raters": expected_raters,
                "minimum_ratings": required,
            }
        )
    return pd.DataFrame(rows)


def make_extent_consensus(
    ratings: pd.DataFrame,
    *,
    expected_raters: int = 4,
    minimum_ratings: int = 4,
) -> pd.DataFrame:
    """Median consensus for annotated duration/fraction, kept secondary to presence."""
    rows = []
    for (file_name, category), frame in ratings.groupby(["file_name", "category"], sort=True):
        fraction = pd.to_numeric(frame["annotated_fraction"], errors="coerce").dropna()
        duration = pd.to_numeric(frame["annotated_duration_sec"], errors="coerce").dropna()
        enough = len(fraction) >= minimum_ratings
        rows.append(
            {
                "file_name": file_name,
                "category": category,
                "consensus_annotated_fraction": float(fraction.median()) if enough else np.nan,
                "consensus_annotated_duration_sec": float(duration.median()) if enough else np.nan,
                "n_ratings": int(len(fraction)),
                "expected_raters": expected_raters,
                "minimum_ratings": minimum_ratings,
                "extent_consensus_method": (
                    "median_complete_raters" if enough else "insufficient_rater_coverage"
                ),
            }
        )
    return pd.DataFrame(rows)
