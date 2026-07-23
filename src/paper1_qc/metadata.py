from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FILENAME_RE = re.compile(
    r"^(?P<subject>[^_]+)_(?P<protocol>[^_]+)_(?P<iteration>[^_]+)_"
    r"(?P<date>\d{8})_(?P<task_code>[^_]+)_(?P<task>.+)$"
)

CLINICAL_RANGES: dict[str, tuple[float, float]] = {
    "MoCA total score": (0, 30),
    "ECAS total score": (0, 136),
    "Beck Depression Inventory total score": (0, 63),
    "ALSFRS total score": (0, 48),
    "ALSFRS bulbar subscore": (0, 12),
    "PLSFRS bulbar subscore": (0, 12),
    "PLSFRS total score": (0, 48),
    "EAT10 total score": (0, 40),
    "Sentence intelligibility percent": (0, 100),
    "Speaking rate": (0, 400),
}

DATE_COLUMNS = [
    "Recording date",
    "Assessment date",
    "Date of Birth",
    "Date of First Symptom",
    "Date of Diagnosis",
]

STATIC_COLUMNS = [
    "Date of Birth",
    "Sex",
    "Race/Ethnicity",
    "Is English the First Language?",
]

BINARY_QC_COLUMNS = [
    "Task Completed as Instructed",
    "Needs Parsing",
    "Another Person in Frame",
    "Another Person Speaks",
    "Background Noise",
    "Volume is Unstable",
    "Poor Audio Quality",
    "Frozen Video",
    "Video is Unstable",
    "Non-optimal head position",
    "Poor Light",
    "Blurry Image",
    "Upper face obstruction - INANIMATE obstr.",
    "Lower face obstruction - INANIMATE obstr.",
    "Doesn't exist",
]

REQUIRED_COLUMNS = [
    "Raw Media File name",
    "SubjectID",
    "Recording date",
    "Task Name",
    "Diagnosis",
]

REQUIRED_VALUE_COLUMNS = [
    "Raw Media File name",
    "SubjectID",
    "Recording date",
    "Task Name",
]


@dataclass
class MetadataAudit:
    clean_media_rows: pd.DataFrame
    canonical_recordings: pd.DataFrame
    issues: pd.DataFrame
    summary: pd.DataFrame
    column_profile: pd.DataFrame


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    return text if text else None


def normalize_diagnosis(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    upper = text.upper()
    aliases = {
        "ALS": "ALS",
        "CONTROL": "CONTROLS",
        "CONTROLS": "CONTROLS",
        "HEALTHY CONTROL": "CONTROLS",
        "HEALTHY CONTROLS": "CONTROLS",
    }
    return aliases.get(upper, upper)


def parse_media_name(name: object) -> dict[str, object]:
    text = normalize_text(name)
    if text is None:
        return {"parse_ok": False}
    path = Path(text)
    match = FILENAME_RE.match(path.stem)
    if match is None:
        return {
            "parse_ok": False,
            "extension_parsed": path.suffix.lower(),
            "logical_recording_id": path.stem,
        }
    result: dict[str, object] = match.groupdict()
    result["parse_ok"] = True
    result["extension_parsed"] = path.suffix.lower()
    result["date_parsed"] = pd.to_datetime(result.pop("date"), format="%Y%m%d", errors="coerce")
    result["logical_recording_id"] = path.stem
    return result


def _issue(
    rows: list[dict],
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    rule: str,
    severity: str,
    column: str | None,
    message: str,
) -> None:
    mask_series = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    for idx in df.index[mask_series]:
        row = df.loc[idx]
        rows.append(
            {
                "source_row": int(idx) + 2,
                "severity": severity,
                "rule": rule,
                "column": column,
                "subject_id": row.get("SubjectID"),
                "file_name": row.get("Raw Media File name"),
                "value": row.get(column) if column in df.columns else None,
                "message": message,
            }
        )


def _canonicalize_media_rows(df: pd.DataFrame, media_preference: Iterable[str]) -> pd.DataFrame:
    priority = {ext.lower(): rank for rank, ext in enumerate(media_preference)}
    work = df.copy()
    work["_media_priority"] = work["extension_parsed"].map(priority).fillna(len(priority) + 1)
    canonical = (
        work.sort_values(["logical_recording_id", "_media_priority", "Raw Media File name"])
        .drop_duplicates("logical_recording_id", keep="first")
        .drop(columns="_media_priority")
        .reset_index(drop=True)
    )
    return canonical


def audit_metadata_workbook(
    path: str | Path,
    *,
    sentinel_values: Iterable[float] = (999, 9999, -999),
    control_id_patterns: Iterable[str] = (r"^CNEC\d+$", r"^C\d+$", r"^TCV\d+$"),
    media_preference: Iterable[str] = (".wav", ".webm", ".mp4"),
    max_primary_assessment_gap_days: int = 14,
) -> MetadataAudit:
    """Audit one workbook and return lossless media rows plus a canonical recording table.

    Raw columns are retained. Cleaned dates/numbers and provenance fields are added; suspected
    sentinels are converted only in known clinical fields.
    """
    path = Path(path)
    raw = pd.read_excel(path, sheet_name=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = raw.copy()
    issues: list[dict] = []
    df["source_workbook"] = path.name
    df["source_row"] = np.arange(len(df)) + 2

    for column in REQUIRED_VALUE_COLUMNS:
        _issue(
            issues,
            df,
            df[column].isna() | df[column].astype(str).str.strip().eq(""),
            rule="required_value_missing",
            severity="error",
            column=column,
            message="Required metadata value is missing.",
        )

    parsed = pd.DataFrame([parse_media_name(value) for value in df["Raw Media File name"]])
    parsed.index = df.index
    for column in parsed.columns:
        df[column] = parsed[column]

    _issue(
        issues,
        df,
        ~df["parse_ok"].fillna(False),
        rule="filename_unparseable",
        severity="error",
        column="Raw Media File name",
        message="Filename did not match the documented six-part recording pattern.",
    )

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[f"{column}__raw"] = df[column]
            df[column] = pd.to_datetime(df[column], errors="coerce")
            _issue(
                issues,
                df,
                df[f"{column}__raw"].notna() & df[column].isna(),
                rule="date_unparseable",
                severity="error",
                column=column,
                message="Non-missing date could not be parsed.",
            )

    for column, (low, high) in CLINICAL_RANGES.items():
        if column not in df.columns:
            continue
        raw_numeric = pd.to_numeric(df[column], errors="coerce")
        sentinel_mask = raw_numeric.isin(list(sentinel_values))
        _issue(
            issues,
            df,
            sentinel_mask,
            rule="clinical_sentinel",
            severity="error",
            column=column,
            message="Configured sentinel found; converted to missing before analysis.",
        )
        clean = raw_numeric.mask(sentinel_mask)
        range_mask = clean.notna() & ~clean.between(low, high, inclusive="both")
        _issue(
            issues,
            df,
            range_mask,
            rule="clinical_out_of_range",
            severity="error",
            column=column,
            message=f"Value outside prespecified plausible range [{low}, {high}].",
        )
        df[f"{column}__raw"] = df[column]
        df[column] = clean.mask(range_mask)

    for column in BINARY_QC_COLUMNS:
        if column not in df.columns:
            continue
        normalized = df[column].map(lambda value: normalize_text(value).upper() if normalize_text(value) else None)
        invalid = normalized.notna() & ~normalized.isin(["YES", "NO"])
        _issue(
            issues,
            df,
            invalid,
            rule="invalid_binary_qc_value",
            severity="error",
            column=column,
            message="Binary QC field contains a value other than Yes/No/missing.",
        )

    technical_rules = {
        "Frame Rate": (lambda x: (x <= 0) | (x > 240), "Video frame rate must be >0 and <=240 fps."),
        "Sampling Rate": (lambda x: (x < 8000) | (x > 192000), "Audio sampling rate outside 8-192 kHz."),
        "Frame Width": (lambda x: x <= 0, "Frame width must be positive."),
        "Frame Height": (lambda x: x <= 0, "Frame height must be positive."),
        "Duration (s)": (lambda x: x <= 0, "Media duration must be positive."),
    }
    for column, (invalid_rule, message) in technical_rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = numeric.notna() & invalid_rule(numeric)
        _issue(
            issues,
            df,
            invalid,
            rule="technical_value_out_of_range",
            severity="error",
            column=column,
            message=message,
        )

    df["diagnosis_reported"] = df["Diagnosis"].map(normalize_diagnosis)
    control_regexes = [re.compile(pattern, flags=re.IGNORECASE) for pattern in control_id_patterns]
    inferred_control = df["SubjectID"].map(
        lambda value: any(regex.match(str(value)) for regex in control_regexes) if pd.notna(value) else False
    )
    df["diagnosis_inferred_from_id"] = np.where(
        df["diagnosis_reported"].isna() & inferred_control, "CONTROLS", pd.NA
    )
    df["diagnosis_analysis"] = df["diagnosis_reported"]
    df["diagnosis_requires_review"] = df["diagnosis_reported"].isna()

    _issue(
        issues,
        df,
        df["diagnosis_reported"].isna() & inferred_control,
        rule="diagnosis_control_id_candidate",
        severity="review",
        column="Diagnosis",
        message="Identifier matches a stated control pattern, but diagnosis remains unconfirmed.",
    )
    _issue(
        issues,
        df,
        df["diagnosis_reported"].isna() & ~inferred_control,
        rule="diagnosis_missing_unresolved",
        severity="error",
        column="Diagnosis",
        message="Diagnosis is missing and no documented identifier rule applies.",
    )

    parsed_subject = df["subject"] if "subject" in df.columns else pd.Series(pd.NA, index=df.index)
    parsed_date = df["date_parsed"] if "date_parsed" in df.columns else pd.Series(pd.NaT, index=df.index)
    subject_mismatch = (
        parsed_subject.notna()
        & df["SubjectID"].notna()
        & (parsed_subject.astype(str).str.upper() != df["SubjectID"].astype(str).str.upper())
    )
    _issue(
        issues,
        df,
        subject_mismatch,
        rule="filename_subject_mismatch",
        severity="error",
        column="SubjectID",
        message="Filename subject and metadata SubjectID disagree.",
    )

    date_mismatch = (
        parsed_date.notna()
        & df["Recording date"].notna()
        & (parsed_date.dt.normalize() != df["Recording date"].dt.normalize())
    )
    _issue(
        issues,
        df,
        date_mismatch,
        rule="filename_recording_date_mismatch",
        severity="error",
        column="Recording date",
        message="Filename date and metadata recording date disagree.",
    )

    extension_mismatch = (
        df["Extension"].notna()
        & (df["Extension"].astype(str).str.lower() != df["extension_parsed"].astype(str).str.lower())
    )
    _issue(
        issues,
        df,
        extension_mismatch,
        rule="extension_mismatch",
        severity="error",
        column="Extension",
        message="Extension column and filename extension disagree.",
    )

    def comparable_number(value: object) -> str | None:
        try:
            number = float(value)
            return str(int(number)) if number.is_integer() else str(number)
        except (TypeError, ValueError):
            return normalize_text(value)

    for parsed_column, metadata_column in [("protocol", "Protocol ID"), ("iteration", "Iteration")]:
        if parsed_column in df.columns and metadata_column in df.columns:
            mismatch = df.apply(
                lambda row: (
                    comparable_number(row[parsed_column]) is not None
                    and comparable_number(row[metadata_column]) is not None
                    and comparable_number(row[parsed_column]) != comparable_number(row[metadata_column])
                ),
                axis=1,
            )
            _issue(
                issues,
                df,
                mismatch,
                rule="filename_protocol_iteration_mismatch",
                severity="error",
                column=metadata_column,
                message=f"Filename {parsed_column} and metadata {metadata_column} disagree.",
            )

    if "task" in df.columns:
        parsed_task = df["task"].astype(str).str.upper()
        metadata_task = df["Task Name"].astype(str).str.upper()
        task_matches = (
            (parsed_task.str.contains("BAMBOO") & metadata_task.str.contains("BAMBOO"))
            | (parsed_task.str.contains("REST") & metadata_task.eq("REST"))
        )
        _issue(
            issues,
            df,
            df["task"].notna() & ~task_matches,
            rule="filename_task_mismatch",
            severity="error",
            column="Task Name",
            message="Filename task label and metadata task disagree.",
        )

    if "Date of Birth" in df.columns:
        chronology = df["Date of Birth"].notna() & df["Recording date"].notna()
        _issue(
            issues,
            df,
            chronology & (df["Date of Birth"] >= df["Recording date"]),
            rule="birth_not_before_recording",
            severity="error",
            column="Date of Birth",
            message="Date of birth must precede recording date.",
        )
        age_years = (df["Recording date"] - df["Date of Birth"]).dt.days / 365.2425
        df["age_at_recording_years"] = age_years
        _issue(
            issues,
            df,
            chronology & (age_years < 18),
            rule="age_under_18",
            severity="review",
            column="Date of Birth",
            message="Age at recording is below 18 years; verify cohort eligibility and dates.",
        )
        for later_date in ["Date of First Symptom", "Date of Diagnosis", "Assessment date"]:
            if later_date in df.columns:
                _issue(
                    issues,
                    df,
                    df["Date of Birth"].notna()
                    & df[later_date].notna()
                    & (df["Date of Birth"] >= df[later_date]),
                    rule="birth_date_chronology_failure",
                    severity="error",
                    column=later_date,
                    message=f"{later_date} must occur after date of birth.",
                )

    if "Assessment date" in df.columns:
        delta_days = (df["Assessment date"] - df["Recording date"]).dt.days
        df["assessment_recording_delta_days"] = delta_days
        df["assessment_within_primary_window"] = delta_days.abs() <= max_primary_assessment_gap_days
        _issue(
            issues,
            df,
            delta_days.notna() & (delta_days.abs() > max_primary_assessment_gap_days),
            rule="assessment_outside_primary_window",
            severity="review",
            column="Assessment date",
            message=f"Clinical assessment is more than {max_primary_assessment_gap_days} days from recording.",
        )

    for date_a, date_b, rule in [
        ("Date of First Symptom", "Date of Diagnosis", "diagnosis_before_first_symptom"),
        ("Date of First Symptom", "Recording date", "recording_before_first_symptom"),
        ("Date of Diagnosis", "Recording date", "recording_before_diagnosis"),
    ]:
        if date_a in df.columns and date_b in df.columns:
            mask = df[date_a].notna() & df[date_b].notna() & (df[date_a] > df[date_b])
            _issue(
                issues,
                df,
                mask,
                rule=rule,
                severity="review",
                column=date_a,
                message=f"Chronology check failed: {date_a} occurs after {date_b}.",
            )

    if {"ALSFRS total score", "ALSFRS bulbar subscore"}.issubset(df.columns):
        impossible = (
            df["ALSFRS total score"].notna()
            & df["ALSFRS bulbar subscore"].notna()
            & (df["ALSFRS bulbar subscore"] > df["ALSFRS total score"])
        )
        _issue(
            issues,
            df,
            impossible,
            rule="clinical_subscore_exceeds_total",
            severity="error",
            column="ALSFRS bulbar subscore",
            message="ALSFRS bulbar subscore cannot exceed ALSFRS total score.",
        )

    duplicate_names = df["Raw Media File name"].duplicated(keep=False)
    _issue(
        issues,
        df,
        duplicate_names,
        rule="duplicate_media_filename",
        severity="error",
        column="Raw Media File name",
        message="Filename occurs more than once in the workbook.",
    )

    # Static values must not drift within participant after ignoring missing values.
    for column in STATIC_COLUMNS + ["diagnosis_reported"]:
        if column not in df.columns:
            continue
        counts = df.groupby("SubjectID", dropna=False)[column].nunique(dropna=True)
        bad_subjects = counts[counts > 1].index
        _issue(
            issues,
            df,
            df["SubjectID"].isin(bad_subjects),
            rule="within_subject_static_conflict",
            severity="error",
            column=column,
            message=f"Participant has conflicting non-missing {column} values.",
        )

    # WAV/WEBM rows describing one logical recording should agree on key metadata.
    key_columns = [
        "SubjectID",
        "Recording date",
        "Task Name",
        "diagnosis_reported",
        "ALSFRS total score",
        "ALSFRS bulbar subscore",
    ]
    for column in [name for name in key_columns if name in df.columns]:
        counts = df.groupby("logical_recording_id", dropna=False)[column].nunique(dropna=True)
        bad_keys = counts[counts > 1].index
        _issue(
            issues,
            df,
            df["logical_recording_id"].isin(bad_keys),
            rule="paired_media_metadata_conflict",
            severity="error",
            column=column,
            message="Rows for alternate media encodings of the same recording disagree.",
        )

    if "Duration (s)" in df.columns:
        duration_span = df.groupby("logical_recording_id", dropna=False)["Duration (s)"].agg(
            lambda values: pd.to_numeric(values, errors="coerce").max()
            - pd.to_numeric(values, errors="coerce").min()
        )
        bad_keys = duration_span[duration_span > 0.5].index
        _issue(
            issues,
            df,
            df["logical_recording_id"].isin(bad_keys),
            rule="paired_media_duration_disagreement",
            severity="review",
            column="Duration (s)",
            message="Alternate encodings differ in reported duration by more than 0.5 seconds.",
        )

    canonical = _canonicalize_media_rows(df, media_preference)
    issue_frame = pd.DataFrame(issues)
    if issue_frame.empty:
        issue_frame = pd.DataFrame(
            columns=["source_row", "severity", "rule", "column", "subject_id", "file_name", "value", "message"]
        )

    summary_rows = [
        ("source_workbook", path.name),
        ("media_rows", len(df)),
        ("logical_recordings", canonical["logical_recording_id"].nunique()),
        ("participants", canonical["SubjectID"].nunique()),
        ("reported_als_participants", canonical.loc[canonical["diagnosis_reported"] == "ALS", "SubjectID"].nunique()),
        ("reported_control_participants", canonical.loc[canonical["diagnosis_reported"] == "CONTROLS", "SubjectID"].nunique()),
        ("unresolved_diagnosis_participants", canonical.loc[canonical["diagnosis_reported"].isna(), "SubjectID"].nunique()),
    ]
    for severity, count in issue_frame["severity"].value_counts(dropna=False).items():
        summary_rows.append((f"issue_rows_{severity}", int(count)))
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    profile_rows = []
    for order, column in enumerate(raw.columns):
        series = raw[column]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_fraction = float(numeric.notna().sum() / max(1, series.notna().sum()))
        profile_rows.append(
            {
                "column_order": order,
                "column": column,
                "source_dtype": str(series.dtype),
                "nonmissing_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "unique_nonmissing": int(series.nunique(dropna=True)),
                "numeric_fraction_of_nonmissing": numeric_fraction,
                "numeric_min": float(numeric.min()) if numeric.notna().any() else np.nan,
                "numeric_max": float(numeric.max()) if numeric.notna().any() else np.nan,
            }
        )
    column_profile = pd.DataFrame(profile_rows)
    return MetadataAudit(df, canonical, issue_frame, summary, column_profile)


def reconcile_workbooks(
    bamboo: MetadataAudit,
    rest: MetadataAudit,
    combined: MetadataAudit | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile standalone task exports; never append the combined export as new data."""
    b = bamboo.canonical_recordings.assign(workbook_role="bamboo_source")
    r = rest.canonical_recordings.assign(workbook_role="rest_source")
    source = pd.concat([b, r], ignore_index=True)
    rows = [
        {"metric": "bamboo_participants", "value": b["SubjectID"].nunique()},
        {"metric": "rest_participants", "value": r["SubjectID"].nunique()},
        {"metric": "participant_intersection", "value": len(set(b["SubjectID"]) & set(r["SubjectID"]))},
    ]
    discrepancies: list[dict] = []
    if combined is not None:
        combined_names = set(combined.clean_media_rows["Raw Media File name"].astype(str))
        standalone_media_names = set(bamboo.clean_media_rows["Raw Media File name"].astype(str)) | set(
            rest.clean_media_rows["Raw Media File name"].astype(str)
        )
        for name in sorted(combined_names - standalone_media_names):
            discrepancies.append({"type": "combined_only_media", "file_name": name})
        for name in sorted(standalone_media_names - combined_names):
            discrepancies.append({"type": "standalone_only_media", "file_name": name})
        rows.extend(
            [
                {"metric": "combined_media_not_in_standalone", "value": len(combined_names - standalone_media_names)},
                {"metric": "standalone_media_not_in_combined", "value": len(standalone_media_names - combined_names)},
            ]
        )
    return pd.DataFrame(rows), pd.DataFrame(discrepancies)


def exact_session_pairs(bamboo: pd.DataFrame, rest: pd.DataFrame) -> pd.DataFrame:
    """Match Rest and Bamboo only on exact participant/date/iteration; no nearest-date matching."""
    keys = ["SubjectID", "Recording date", "protocol", "iteration"]
    b_cols = keys + ["logical_recording_id", "Raw Media File name"]
    r_cols = keys + ["logical_recording_id", "Raw Media File name"]
    return bamboo[b_cols].merge(
        rest[r_cols], on=keys, how="inner", suffixes=("_bamboo", "_rest"), validate="many_to_many"
    )
