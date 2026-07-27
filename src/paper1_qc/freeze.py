from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ADJUDICATION_COLUMNS = [
    "SubjectID",
    "diagnosis_analysis",
    "evidence_source",
    "reviewer",
    "review_date",
    "notes",
]

ALLOWED_ADJUDICATIONS = {"ALS", "CONTROLS", "EXCLUDE"}


def _normalized_id(value: object) -> str:
    return str(value).strip().upper()


def _confirmed_pattern_mask(subject_ids: pd.Series, patterns: Iterable[str]) -> pd.Series:
    regexes = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    return subject_ids.map(
        lambda value: any(regex.fullmatch(str(value).strip()) for regex in regexes)
        if pd.notna(value)
        else False
    )


def diagnosis_adjudication_template(
    frames: Iterable[pd.DataFrame],
    *,
    control_id_patterns: Iterable[str],
    confirm_control_patterns: bool,
    confirmed_control_subject_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Return one editable row for every diagnosis that still needs a human decision."""
    combined = pd.concat(
        [
            frame[["SubjectID", "diagnosis_reported"]].copy()
            for frame in frames
            if not frame.empty
        ],
        ignore_index=True,
    )
    combined["SubjectID"] = combined["SubjectID"].map(_normalized_id)
    participant = (
        combined.groupby("SubjectID", as_index=False)["diagnosis_reported"]
        .agg(lambda values: next((v for v in values if pd.notna(v)), pd.NA))
    )
    unresolved = participant["diagnosis_reported"].isna()
    if confirm_control_patterns:
        unresolved &= ~_confirmed_pattern_mask(participant["SubjectID"], control_id_patterns)
    confirmed_ids = {_normalized_id(value) for value in confirmed_control_subject_ids}
    unresolved &= ~participant["SubjectID"].isin(confirmed_ids)
    template = participant.loc[unresolved, ["SubjectID"]].sort_values("SubjectID").reset_index(drop=True)
    for column in ADJUDICATION_COLUMNS[1:]:
        template[column] = ""
    template["recommended_review"] = template["SubjectID"].map(
        lambda value: (
            "VERIFY CONTROL SUFFIX"
            if re.fullmatch(r"(?:CNEC\d+|C\d+|TCV\d+)-\d+", value, flags=re.IGNORECASE)
            else "VERIFY WITH DATA MANAGER OR EXCLUDE"
        )
    )
    return template


def load_adjudications(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=ADJUDICATION_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in ADJUDICATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing adjudication columns: {missing}")
    frame = frame[ADJUDICATION_COLUMNS].copy()
    frame["SubjectID"] = frame["SubjectID"].map(_normalized_id)
    frame["diagnosis_analysis"] = frame["diagnosis_analysis"].str.strip().str.upper()
    frame = frame.loc[frame["SubjectID"].ne("")].copy()
    duplicated = frame["SubjectID"].duplicated(keep=False)
    if duplicated.any():
        ids = sorted(frame.loc[duplicated, "SubjectID"].unique())
        raise ValueError(f"Duplicate diagnosis adjudications: {ids}")
    invalid = ~frame["diagnosis_analysis"].isin(ALLOWED_ADJUDICATIONS)
    if invalid.any():
        bad = frame.loc[invalid, ["SubjectID", "diagnosis_analysis"]].to_dict("records")
        raise ValueError(
            "Every adjudication must be ALS, CONTROLS, or EXCLUDE. "
            f"Invalid or blank rows: {bad}"
        )
    missing_evidence = frame["evidence_source"].str.strip().eq("")
    if missing_evidence.any():
        ids = sorted(frame.loc[missing_evidence, "SubjectID"].unique())
        raise ValueError(f"Adjudications require evidence_source: {ids}")
    return frame


def resolve_diagnoses(
    frame: pd.DataFrame,
    *,
    adjudications: pd.DataFrame,
    allowed_diagnoses: Iterable[str],
    control_id_patterns: Iterable[str],
    confirm_control_patterns: bool,
    control_rule_evidence: str,
    confirmed_control_subject_ids: Iterable[str] = (),
    confirmed_control_subject_evidence: str = "",
) -> pd.DataFrame:
    """Create one final diagnosis plus explicit provenance; never overwrite source fields."""
    result = frame.copy()
    result["SubjectID"] = result["SubjectID"].map(_normalized_id)
    result["diagnosis_analysis"] = result["diagnosis_reported"]
    reported = result["diagnosis_reported"].notna()
    result["diagnosis_resolution"] = np.where(reported, "reported_metadata", "pending")
    result["diagnosis_evidence"] = np.where(reported, "Diagnosis column in source workbook", "")

    if confirm_control_patterns:
        exact_control = result["diagnosis_reported"].isna() & _confirmed_pattern_mask(
            result["SubjectID"], control_id_patterns
        )
        result.loc[exact_control, "diagnosis_analysis"] = "CONTROLS"
        result.loc[exact_control, "diagnosis_resolution"] = "confirmed_control_id_rule"
        result.loc[exact_control, "diagnosis_evidence"] = control_rule_evidence

    confirmed_ids = {_normalized_id(value) for value in confirmed_control_subject_ids}
    exceptional_control = (
        result["diagnosis_reported"].isna() & result["SubjectID"].isin(confirmed_ids)
    )
    result.loc[exceptional_control, "diagnosis_analysis"] = "CONTROLS"
    result.loc[exceptional_control, "diagnosis_resolution"] = (
        "investigator_confirmed_control_subject"
    )
    result.loc[exceptional_control, "diagnosis_evidence"] = (
        confirmed_control_subject_evidence
    )

    if not adjudications.empty:
        manual = adjudications.set_index("SubjectID")
        reported_ids = set(result.loc[reported, "SubjectID"])
        attempted_overrides = sorted(reported_ids & set(manual.index))
        if attempted_overrides:
            raise ValueError(
                "Manual adjudication cannot override a non-missing reported diagnosis: "
                f"{attempted_overrides}"
            )
        mapped = result["SubjectID"].map(manual["diagnosis_analysis"])
        evidence = result["SubjectID"].map(manual["evidence_source"])
        has_manual = mapped.notna()
        excluded = mapped.eq("EXCLUDE")
        result.loc[has_manual & ~excluded, "diagnosis_analysis"] = mapped[has_manual & ~excluded]
        result.loc[excluded, "diagnosis_analysis"] = pd.NA
        result.loc[has_manual & ~excluded, "diagnosis_resolution"] = "manual_confirmed"
        result.loc[excluded, "diagnosis_resolution"] = "manual_excluded"
        result.loc[has_manual, "diagnosis_evidence"] = evidence[has_manual]

    allowed = {str(value).upper() for value in allowed_diagnoses}
    nonallowed_reported = reported & ~result["diagnosis_analysis"].isin(allowed)
    result.loc[nonallowed_reported, "diagnosis_analysis"] = pd.NA
    result.loc[nonallowed_reported, "diagnosis_resolution"] = "reported_non_target_excluded"
    result.loc[nonallowed_reported, "diagnosis_evidence"] = (
        "Reported diagnosis is outside the prespecified ALS/control cohort"
    )
    result["diagnosis_contrast_eligible"] = result["diagnosis_analysis"].isin(allowed)
    result["target_cohort_eligible"] = ~result["diagnosis_resolution"].isin(
        ["manual_excluded", "reported_non_target_excluded", "pending"]
    )
    return result


def attach_media_freeze_gate(
    canonical: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    media_rows: pd.DataFrame | None = None,
    media_preference: Iterable[str] = (".wav", ".webm", ".mp4"),
) -> pd.DataFrame:
    """Select one viable encoding per logical recording, preferring WAV when available."""
    result = canonical.copy()
    if inventory.empty:
        result["media_path_count"] = 0
        result["media_probe_ok"] = False
        result["media_full_decode_ok"] = False
        result["media_freeze_eligible"] = False
        result["media_freeze_reason"] = "empty_inventory"
        result["media_selection_reason"] = "no_inventory"
        return result

    grouped = (
        inventory.groupby("file_name", as_index=False)
        .agg(
            media_path_count=("file_path", "size"),
            media_probe_ok=("probe_ok", lambda values: bool(pd.Series(values).fillna(False).all())),
            media_full_decode_ok=(
                "full_decode_ok",
                lambda values: bool(pd.Series(values).fillna(False).all()),
            ),
            media_path=("file_path", lambda values: " | ".join(sorted(map(str, values)))),
            media_sha256=("sha256", lambda values: " | ".join(sorted(set(map(str, values))))),
        )
    )

    if media_rows is not None and not media_rows.empty:
        candidates = media_rows[
            ["logical_recording_id", "Raw Media File name", "extension_parsed"]
        ].drop_duplicates()
        candidates = candidates.merge(
            grouped,
            left_on="Raw Media File name",
            right_on="file_name",
            how="left",
            validate="many_to_one",
        ).drop(columns="file_name")
        candidates["media_path_count"] = candidates["media_path_count"].fillna(0).astype(int)
        candidates["media_probe_ok"] = candidates["media_probe_ok"].fillna(False).astype(bool)
        candidates["media_full_decode_ok"] = (
            candidates["media_full_decode_ok"].fillna(False).astype(bool)
        )
        candidates["media_freeze_eligible"] = (
            candidates["media_path_count"].eq(1)
            & candidates["media_probe_ok"]
            & candidates["media_full_decode_ok"]
        )
        preference = {
            str(extension).lower(): rank for rank, extension in enumerate(media_preference)
        }
        candidates["media_preference_rank"] = (
            candidates["extension_parsed"]
            .astype(str)
            .str.lower()
            .map(preference)
            .fillna(len(preference) + 1)
        )
        candidates = candidates.sort_values(
            [
                "logical_recording_id",
                "media_freeze_eligible",
                "media_preference_rank",
                "Raw Media File name",
            ],
            ascending=[True, False, True, True],
        )
        selected = candidates.drop_duplicates("logical_recording_id", keep="first").rename(
            columns={
                "Raw Media File name": "selected_media_file_name",
                "extension_parsed": "selected_media_extension",
            }
        )
        result = result.merge(
            selected,
            on="logical_recording_id",
            how="left",
            validate="one_to_one",
        )
        has_selected = result["selected_media_file_name"].notna()
        result.loc[has_selected, "Raw Media File name"] = result.loc[
            has_selected, "selected_media_file_name"
        ]
        result.loc[has_selected, "extension_parsed"] = result.loc[
            has_selected, "selected_media_extension"
        ]
        result["media_selection_reason"] = np.where(
            result["media_freeze_eligible"].fillna(False),
            "highest_priority_unique_decodable_encoding",
            "no_unique_decodable_encoding",
        )
    else:
        result = result.merge(
            grouped,
            left_on="Raw Media File name",
            right_on="file_name",
            how="left",
            validate="many_to_one",
        ).drop(columns="file_name")
        result["media_path_count"] = result["media_path_count"].fillna(0).astype(int)
        result["media_probe_ok"] = result["media_probe_ok"].fillna(False).astype(bool)
        result["media_full_decode_ok"] = result["media_full_decode_ok"].fillna(False).astype(bool)
        result["media_freeze_eligible"] = (
            result["media_path_count"].eq(1)
            & result["media_probe_ok"]
            & result["media_full_decode_ok"]
        )
        result["media_selection_reason"] = "canonical_metadata_preference"

    result["media_freeze_reason"] = np.select(
        [
            result["media_path_count"].fillna(0).eq(0),
            result["media_path_count"].fillna(0).gt(1),
            ~result["media_probe_ok"].fillna(False),
            ~result["media_full_decode_ok"].fillna(False),
        ],
        [
            "canonical_file_missing_on_disk",
            "canonical_filename_resolves_to_multiple_paths",
            "ffprobe_failed",
            "full_decode_failed",
        ],
        default="included",
    )
    return result


def issue_disposition_table(
    issues: pd.DataFrame, audited_rows: pd.DataFrame, resolved: pd.DataFrame
) -> pd.DataFrame:
    """Attach a deterministic scientific disposition to every metadata finding."""
    if issues.empty:
        return issues.assign(disposition_status=pd.Series(dtype=str), disposition=pd.Series(dtype=str))
    file_to_logical = audited_rows.set_index("Raw Media File name")["logical_recording_id"]
    output = issues.copy()
    output["logical_recording_id"] = output["file_name"].map(file_to_logical)
    resolution_by_subject = (
        resolved[["SubjectID", "diagnosis_resolution"]]
        .drop_duplicates("SubjectID")
        .set_index("SubjectID")["diagnosis_resolution"]
    )
    output["diagnosis_resolution"] = output["subject_id"].map(resolution_by_subject)

    dispositions = {
        "clinical_sentinel": (
            "resolved",
            "Sentinel converted to missing; affected clinical field is not analyzed.",
        ),
        "clinical_out_of_range": (
            "resolved",
            "Implausible clinical value converted to missing; affected field is not analyzed.",
        ),
        "diagnosis_control_id_candidate": (
            "resolved",
            "Resolved by the frozen diagnosis provenance rule or manual adjudication.",
        ),
        "diagnosis_missing_unresolved": (
            "resolved",
            "Resolved by manual diagnosis adjudication or explicit cohort exclusion.",
        ),
        "technical_value_out_of_range": (
            "resolved",
            "Invalid platform technical value is not used; native FFprobe/decode evidence governs audio processing.",
        ),
        "birth_not_before_recording": (
            "resolved",
            "Invalid birth date and derived age are set missing; recording remains eligible for non-age analyses.",
        ),
        "birth_date_chronology_failure": (
            "resolved",
            "Affected date-dependent clinical field is excluded.",
        ),
        "age_under_18": (
            "resolved",
            "Age is not used because its source dates fail chronology checks.",
        ),
        "filename_recording_date_mismatch": (
            "resolved",
            "Recording is excluded from date-dependent and exact-session analyses; source dates are retained for audit.",
        ),
        "assessment_outside_primary_window": (
            "resolved",
            "Excluded from primary clinical alignment; sensitivity eligibility is computed separately.",
        ),
        "recording_before_diagnosis": (
            "resolved_flagged",
            "Retained for measurement analysis and flagged for a pre-diagnosis sensitivity analysis.",
        ),
        "paired_media_duration_disagreement": (
            "resolved_flagged",
            "Preferred decodable encoding is retained and the recording is flagged for paired-encoding sensitivity.",
        ),
    }
    output["disposition_status"] = output["rule"].map(
        lambda rule: dispositions.get(rule, ("exclude_recording", ""))[0]
    )
    output["disposition"] = output["rule"].map(
        lambda rule: dispositions.get(
            rule,
            (
                "exclude_recording",
                "No prespecified automatic correction exists; recording is excluded from the frozen cohort.",
            ),
        )[1]
    )
    return output


def apply_metadata_analysis_gates(
    resolved: pd.DataFrame, dispositions: pd.DataFrame
) -> pd.DataFrame:
    result = resolved.copy()
    grouped_rules = (
        dispositions.groupby("logical_recording_id")["rule"].agg(set).to_dict()
        if not dispositions.empty
        else {}
    )
    grouped_status = (
        dispositions.groupby("logical_recording_id")["disposition_status"].agg(set).to_dict()
        if not dispositions.empty
        else {}
    )
    rules = result["logical_recording_id"].map(lambda key: grouped_rules.get(key, set()))
    statuses = result["logical_recording_id"].map(lambda key: grouped_status.get(key, set()))
    result["date_analysis_eligible"] = ~rules.map(
        lambda values: "filename_recording_date_mismatch" in values
    )
    result["age_analysis_eligible"] = ~rules.map(
        lambda values: bool(
            {"birth_not_before_recording", "birth_date_chronology_failure", "age_under_18"}
            & values
        )
    )
    invalid_age = ~result["age_analysis_eligible"]
    for column in ["Date of Birth", "age_at_recording_years"]:
        if column in result.columns:
            result.loc[invalid_age, column] = pd.NaT if column == "Date of Birth" else np.nan
    result["recording_date_analysis"] = result["Recording date"]
    result.loc[~result["date_analysis_eligible"], "recording_date_analysis"] = pd.NaT
    result["metadata_freeze_eligible"] = ~statuses.map(
        lambda values: "exclude_recording" in values
    )
    result["freeze_included"] = (
        result["target_cohort_eligible"]
        & result["media_freeze_eligible"]
        & result["metadata_freeze_eligible"]
    )
    result["freeze_exclusion_reason"] = np.select(
        [
            ~result["target_cohort_eligible"],
            ~result["media_freeze_eligible"],
            ~result["metadata_freeze_eligible"],
        ],
        [
            "diagnosis_not_resolved_into_target_cohort",
            result["media_freeze_reason"],
            "unresolved_blocking_metadata_error",
        ],
        default="included",
    )
    return result
