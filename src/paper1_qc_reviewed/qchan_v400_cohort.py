"""Cohort orchestration and empirical validation for QCHAN v4.0.0.

This module does not redefine the four QCHAN estimands. It enforces the frozen
cohort/strict-speech contract, serializes recording and reference spectra, and
provides support, reference-robustness, empirical, reliability, model-interface,
and inventory utilities for the reviewed cohort notebook.

QCHAN remains a feature-first, reference-relative profile. It does not identify
a device, microphone, browser, platform, or codec. Missing support is not zero,
one-sided zeros retain signed precursors, and no family scalar or standalone
rejection threshold is constructed.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Mapping, Sequence
import json
import math
import re

import numpy as np
import pandas as pd
from scipy import stats

from paper1_qc_reviewed.qchan_v400 import (
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    MEASUREMENT_VERSION,
    QChanParameters,
    RecordingSpectrum,
    ReferenceSpectrum,
    TimeInterval,
    compute_reference_relative_features,
    extract_recording_spectrum,
    smoothed_log_ltas_db,
    spectral_descriptors,
)

COHORT_ORCHESTRATION_VERSION = "qchan-v4.0.0-cohort-orchestration-v3"
CANONICAL_PROFILE = "primary"
CANONICAL_STRICT_VIEW = "strict_speech"
ONE_SIDED_FEATURES = (
    "qchan_rolloff95_deficit_hz",
    "qchan_highband_ratio_deficit",
    "qchan_tilt_steepening_db_per_oct",
)
SIGNED_PRECURSORS = {
    "qchan_rolloff95_deficit_hz": "qchan_rolloff95_signed_difference_hz",
    "qchan_highband_ratio_deficit": "qchan_highband_ratio_signed_difference",
    "qchan_tilt_steepening_db_per_oct": "qchan_tilt_signed_difference_db_per_oct",
}


REQUIRED_GALLERY_LINKED_VIEWS = (
    "waveform",
    "spectrogram",
    "target_ltas",
    "reference_ltas",
    "ltas_difference",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def array_hash(array: np.ndarray) -> str:
    normalized = np.asarray(array, dtype="<f8")
    return sha256(normalized.tobytes(order="C")).hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"1", "true", "yes", "y"}
    )


def json_safe(value):
    if value is pd.NA:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    return value


def write_json(payload: Mapping, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2),
        encoding="utf-8",
    )


def remove_global_dc(waveform: np.ndarray) -> tuple[np.ndarray, float, float]:
    y = np.asarray(waveform, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("Canonical analysis waveform must be finite, nonempty, and one-dimensional.")
    before = float(np.mean(y))
    corrected = y - before
    after = float(np.mean(corrected))
    return corrected, before, after


def resolve_media_path(
    raw_path: object,
    *,
    media_root_override: Path | None = None,
    media_path_map: Mapping[str, str] | None = None,
) -> Path:
    raw = str(raw_path)
    mapping = dict(media_path_map or {})
    if raw in mapping:
        candidate = Path(mapping[raw]).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Mapped media path does not exist: {candidate}")

    direct = Path(raw).expanduser()
    if direct.exists():
        return direct

    if media_root_override is not None:
        parts = list(PureWindowsPath(raw).parts)
        marker = next(
            (
                index
                for index, part in enumerate(parts)
                if part.lower() == "bamboo_passage_only"
            ),
            None,
        )
        relative_parts = (
            parts[marker + 1 :]
            if marker is not None
            else [PureWindowsPath(raw).name]
        )
        candidate = Path(media_root_override).expanduser().joinpath(*relative_parts)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Resolved override path does not exist: {candidate}")

    raise FileNotFoundError(
        f"Frozen media path does not resolve: {raw}. "
        "Set MEDIA_ROOT_OVERRIDE or MEDIA_PATH_MAP."
    )


def canonical_interval_subset(
    intervals: pd.DataFrame,
    *,
    view: str = CANONICAL_STRICT_VIEW,
    profile: str = CANONICAL_PROFILE,
    eligible_only: bool = True,
) -> pd.DataFrame:
    required = {
        "logical_recording_id",
        "view",
        "profile",
        "interval_index",
        "start_sec",
        "end_sec",
    }
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"Frozen intervals are missing columns: {sorted(missing)}")

    local = intervals.loc[
        intervals["view"].astype(str).eq(str(view))
        & intervals["profile"].astype(str).eq(str(profile))
    ].copy()
    if eligible_only and "segmentation_analysis_eligible" in local:
        local = local.loc[as_bool(local["segmentation_analysis_eligible"])]
    if "decision" in local:
        local = local.loc[
            local["decision"].astype(str).str.strip().str.upper().eq("KEEP")
        ]

    local["logical_recording_id"] = local["logical_recording_id"].astype(str)
    local["interval_index"] = pd.to_numeric(
        local["interval_index"], errors="raise"
    ).astype(int)
    local["start_sec"] = pd.to_numeric(local["start_sec"], errors="raise")
    local["end_sec"] = pd.to_numeric(local["end_sec"], errors="raise")
    local["frozen_interval_id"] = (
        local["logical_recording_id"]
        + ":"
        + str(view)
        + ":"
        + str(profile)
        + ":"
        + local["interval_index"].astype(str).str.zfill(5)
    )
    local = local.sort_values(
        ["logical_recording_id", "start_sec", "end_sec", "interval_index"]
    ).reset_index(drop=True)

    identity = ["logical_recording_id", "view", "profile", "interval_index"]
    if local.duplicated(identity).any():
        raise ValueError("Canonical strict-speech interval identities are duplicated.")
    if local["frozen_interval_id"].duplicated().any():
        raise ValueError("Deterministic frozen interval IDs are duplicated.")
    if (local["end_sec"] <= local["start_sec"]).any():
        raise ValueError("Canonical strict-speech intervals contain nonpositive durations.")

    overlaps = []
    for recording_id, group in local.groupby("logical_recording_id", sort=False):
        ordered = group.sort_values(["start_sec", "end_sec"])
        starts = ordered["start_sec"].to_numpy(float)
        ends = ordered["end_sec"].to_numpy(float)
        for index in range(1, len(ordered)):
            if starts[index] < ends[index - 1] - 1e-9:
                overlaps.append(
                    {
                        "logical_recording_id": recording_id,
                        "previous_end_sec": ends[index - 1],
                        "next_start_sec": starts[index],
                    }
                )
    if overlaps:
        raise ValueError("Canonical strict-speech intervals overlap.")
    return local


def canonical_interval_contract(
    decisions: pd.DataFrame,
    intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "segmentation_analysis_eligible" not in decisions:
        raise ValueError("Frozen decisions lack segmentation_analysis_eligible.")
    eligible_ids = set(
        decisions.loc[
            as_bool(decisions["segmentation_analysis_eligible"]),
            "logical_recording_id",
        ].astype(str)
    )
    if not eligible_ids:
        raise ValueError("No segmentation-analysis-eligible recordings were found.")

    strict = canonical_interval_subset(intervals)
    strict_ids = set(strict["logical_recording_id"].astype(str))
    missing = sorted(eligible_ids - strict_ids)
    extra = sorted(strict_ids - eligible_ids)
    contract = pd.DataFrame(
        [
            {
                "contract": "canonical_view_profile",
                "observed": f"{CANONICAL_STRICT_VIEW}/{CANONICAL_PROFILE}",
                "required": f"{CANONICAL_STRICT_VIEW}/{CANONICAL_PROFILE}",
                "contract_pass": True,
            },
            {
                "contract": "eligible_recordings_have_strict_intervals",
                "observed": len(strict_ids & eligible_ids),
                "required": len(eligible_ids),
                "contract_pass": not missing,
            },
            {
                "contract": "no_ineligible_recordings_in_strict_table",
                "observed": len(extra),
                "required": 0,
                "contract_pass": not extra,
            },
            {
                "contract": "strict_interval_ids_unique",
                "observed": int(strict["frozen_interval_id"].nunique()),
                "required": len(strict),
                "contract_pass": not strict["frozen_interval_id"].duplicated().any(),
            },
        ]
    )
    return strict, contract


def intervals_for_recording(
    strict_table: pd.DataFrame,
    logical_recording_id: str,
) -> tuple[list[TimeInterval], pd.DataFrame]:
    local = strict_table.loc[
        strict_table["logical_recording_id"].astype(str).eq(
            str(logical_recording_id)
        )
    ].sort_values(["start_sec", "end_sec", "interval_index"])
    intervals = [
        TimeInterval(float(row.start_sec), float(row.end_sec))
        for row in local.itertuples(index=False)
    ]
    return intervals, local.copy()


def resolve_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
    *,
    required: bool = True,
) -> str | None:
    present = [column for column in candidates if column in frame.columns]
    if not present:
        if required:
            raise ValueError(f"Could not resolve {label}; tried {list(candidates)}")
        return None
    if len(present) == 1:
        return present[0]
    # Prefer the candidate with the most nonmissing values, then candidate order.
    ranked = sorted(
        present,
        key=lambda column: (
            -int(frame[column].notna().sum()),
            list(candidates).index(column),
        ),
    )
    return ranked[0]


def subject_column_for(frame: pd.DataFrame) -> str:
    return str(
        resolve_column(
            frame,
            [
                "SubjectID",
                "subject_id",
                "participant_id",
                "ParticipantID",
                "subject_uid",
                "participant_uid",
            ],
            "participant identity",
        )
    )


def date_column_for(frame: pd.DataFrame) -> str:
    return str(
        resolve_column(
            frame,
            [
                "recording_date_analysis",
                "Recording date",
                "recording_date",
                "session_date",
                "date",
            ],
            "recording date",
        )
    )


def media_hash_column(frame: pd.DataFrame) -> str | None:
    return resolve_column(
        frame,
        [
            "media_sha256",
            "selected_media_sha256",
            "file_sha256",
            "sha256",
        ],
        "media SHA-256",
        required=False,
    )


def task_stratum_series(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    candidates = [
        "task_stratum",
        "task_name",
        "task",
        "Task",
        "recording_task",
        "stimulus",
        "passage",
    ]
    present = [column for column in candidates if column in frame.columns]
    for column in present:
        values = frame[column].astype("string").str.strip()
        if values.notna().any() and values.fillna("").ne("").any():
            return values.fillna("UNSPECIFIED_TASK"), column
    # This frozen cohort contains one declared task: the Bamboo passage.
    return pd.Series(
        ["BAMBOO_PASSAGE"] * len(frame),
        index=frame.index,
        dtype="string",
    ), "constant:BAMBOO_PASSAGE"


def safe_recording_filename(logical_recording_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(logical_recording_id))
    suffix = sha256(str(logical_recording_id).encode("utf-8")).hexdigest()[:10]
    return f"{cleaned}_{suffix}"


def spectrum_checkpoint_paths(root: Path, logical_recording_id: str) -> dict[str, Path]:
    stem = safe_recording_filename(logical_recording_id)
    root = Path(root)
    return {
        "spectrum": root / "spectra" / f"{stem}.npz",
        "metadata": root / "records" / f"{stem}.json",
        "media": root / "media_audit" / f"{stem}.json",
        "error": root / "errors" / f"{stem}.json",
    }


def spectrum_checkpoint_complete(paths: Mapping[str, Path]) -> bool:
    return all(Path(paths[key]).exists() for key in ["spectrum", "metadata", "media"])


def save_recording_spectrum(spectrum: RecordingSpectrum, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        logical_recording_id=np.asarray([spectrum.logical_recording_id]),
        frequencies_hz=np.asarray(spectrum.frequencies_hz, dtype=np.float64),
        normalized_psd_per_hz=np.asarray(
            spectrum.normalized_psd_per_hz, dtype=np.float64
        ),
        status=np.asarray([spectrum.status]),
        support_tier=np.asarray([spectrum.support_tier]),
        guarded_speech_support_sec=np.asarray(
            [spectrum.guarded_speech_support_sec], dtype=np.float64
        ),
        valid_frame_count=np.asarray([spectrum.valid_frame_count], dtype=np.int64),
        guarded_segment_count=np.asarray(
            [spectrum.guarded_segment_count], dtype=np.int64
        ),
        zero_frame_count=np.asarray([spectrum.zero_frame_count], dtype=np.int64),
        source_sample_rate_hz=np.asarray(
            [spectrum.source_sample_rate_hz], dtype=np.float64
        ),
        source_nyquist_hz=np.asarray(
            [spectrum.source_nyquist_hz], dtype=np.float64
        ),
        source_bandwidth_limited=np.asarray(
            [spectrum.source_bandwidth_limited], dtype=np.bool_
        ),
        spectrum_sha256=np.asarray([spectrum.spectrum_sha256]),
    )


def load_recording_spectrum(path: Path) -> RecordingSpectrum:
    with np.load(Path(path), allow_pickle=False) as data:
        return RecordingSpectrum(
            logical_recording_id=str(data["logical_recording_id"][0]),
            frequencies_hz=np.asarray(data["frequencies_hz"], dtype=np.float64),
            normalized_psd_per_hz=np.asarray(
                data["normalized_psd_per_hz"], dtype=np.float64
            ),
            status=str(data["status"][0]),
            support_tier=str(data["support_tier"][0]),
            guarded_speech_support_sec=float(
                data["guarded_speech_support_sec"][0]
            ),
            valid_frame_count=int(data["valid_frame_count"][0]),
            guarded_segment_count=int(data["guarded_segment_count"][0]),
            zero_frame_count=int(data["zero_frame_count"][0]),
            source_sample_rate_hz=float(data["source_sample_rate_hz"][0]),
            source_nyquist_hz=float(data["source_nyquist_hz"][0]),
            source_bandwidth_limited=bool(
                data["source_bandwidth_limited"][0]
            ),
            spectrum_sha256=str(data["spectrum_sha256"][0]),
        )


def save_reference_spectrum(reference: ReferenceSpectrum, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        reference_key=np.asarray([reference.reference_key]),
        task_stratum=np.asarray([reference.task_stratum]),
        excluded_subject_id=np.asarray([reference.excluded_subject_id]),
        frequencies_hz=np.asarray(reference.frequencies_hz, dtype=np.float64),
        normalized_psd_per_hz=np.asarray(
            reference.normalized_psd_per_hz, dtype=np.float64
        ),
        status=np.asarray([reference.status]),
        member_recording_ids=np.asarray(reference.member_recording_ids),
        member_subject_ids=np.asarray(reference.member_subject_ids),
        recording_count=np.asarray([reference.recording_count], dtype=np.int64),
        subject_count=np.asarray([reference.subject_count], dtype=np.int64),
        reference_sha256=np.asarray([reference.reference_sha256]),
        reference_vintage_sha256=np.asarray(
            [reference.reference_vintage_sha256]
        ),
    )


def load_reference_spectrum(path: Path) -> ReferenceSpectrum:
    with np.load(Path(path), allow_pickle=False) as data:
        return ReferenceSpectrum(
            reference_key=str(data["reference_key"][0]),
            task_stratum=str(data["task_stratum"][0]),
            excluded_subject_id=str(data["excluded_subject_id"][0]),
            frequencies_hz=np.asarray(data["frequencies_hz"], dtype=np.float64),
            normalized_psd_per_hz=np.asarray(
                data["normalized_psd_per_hz"], dtype=np.float64
            ),
            status=str(data["status"][0]),
            member_recording_ids=tuple(str(x) for x in data["member_recording_ids"]),
            member_subject_ids=tuple(str(x) for x in data["member_subject_ids"]),
            recording_count=int(data["recording_count"][0]),
            subject_count=int(data["subject_count"][0]),
            reference_sha256=str(data["reference_sha256"][0]),
            reference_vintage_sha256=str(
                data["reference_vintage_sha256"][0]
            ),
        )


def reference_inventory_frame(
    references: Mapping[str, ReferenceSpectrum],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = {"logical_recording_id", "subject_id", "task_stratum"}
    if not required.issubset(metadata.columns):
        raise ValueError("Reference metadata is incomplete.")
    lookup = metadata.set_index("logical_recording_id")
    rows = []
    for recording_id, reference in sorted(references.items()):
        target = lookup.loc[str(recording_id)]
        rows.append(
            {
                "logical_recording_id": str(recording_id),
                "target_subject_id": str(target["subject_id"]),
                "task_stratum": str(target["task_stratum"]),
                "reference_status": reference.status,
                "reference_key": reference.reference_key,
                "reference_sha256": reference.reference_sha256,
                "reference_vintage_sha256": reference.reference_vintage_sha256,
                "reference_recording_count": reference.recording_count,
                "reference_subject_count": reference.subject_count,
                "target_subject_excluded": (
                    str(target["subject_id"]) not in reference.member_subject_ids
                ),
                "member_recording_ids_json": json.dumps(
                    list(reference.member_recording_ids),
                    separators=(",", ":"),
                ),
                "member_subject_ids_json": json.dumps(
                    list(reference.member_subject_ids),
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(rows)


def unique_reference_frame(
    references: Mapping[str, ReferenceSpectrum],
) -> pd.DataFrame:
    seen: dict[str, ReferenceSpectrum] = {}
    for reference in references.values():
        seen.setdefault(reference.reference_key, reference)
    rows = []
    for key, reference in sorted(seen.items()):
        rows.append(
            {
                "reference_key": key,
                "task_stratum": reference.task_stratum,
                "excluded_subject_id": reference.excluded_subject_id,
                "reference_status": reference.status,
                "reference_recording_count": reference.recording_count,
                "reference_subject_count": reference.subject_count,
                "reference_sha256": reference.reference_sha256,
                "reference_vintage_sha256": reference.reference_vintage_sha256,
                "member_recording_ids_json": json.dumps(
                    list(reference.member_recording_ids),
                    separators=(",", ":"),
                ),
                "member_subject_ids_json": json.dumps(
                    list(reference.member_subject_ids),
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(rows)


def renormalize_psd(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    parameters: QChanParameters,
) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    values = np.asarray(psd, dtype=np.float64)
    mask = (
        (frequencies >= parameters.analysis_low_hz)
        & (frequencies <= parameters.analysis_high_hz)
        & np.isfinite(values)
        & (values >= 0)
    )
    if mask.sum() < 2:
        raise ValueError("Insufficient PSD support in analysis band.")
    total = float(np.trapezoid(values[mask], frequencies[mask]))
    if not np.isfinite(total) or total <= np.finfo(np.float64).tiny:
        raise ValueError("PSD has zero analysis-band power.")
    normalized = np.maximum(values, 0.0) / total
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def eligible_reference_members(
    spectra: Mapping[str, RecordingSpectrum],
    metadata: pd.DataFrame,
    *,
    task_stratum: str,
    excluded_subject_id: str,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> pd.DataFrame:
    required = {"logical_recording_id", "subject_id", "task_stratum"}
    if not required.issubset(metadata.columns):
        raise ValueError("Reference metadata is incomplete.")
    local = metadata.loc[
        metadata["task_stratum"].astype(str).eq(str(task_stratum))
        & ~metadata["subject_id"].astype(str).eq(str(excluded_subject_id))
    ].copy()
    rows = []
    for row in local.itertuples(index=False):
        recording_id = str(row.logical_recording_id)
        spectrum = spectra.get(recording_id)
        if spectrum is None or spectrum.status != "measured":
            continue
        if (
            parameters.reference_requires_full_analysis_band
            and spectrum.source_nyquist_hz < parameters.analysis_high_hz
        ):
            continue
        rows.append(
            {
                "logical_recording_id": recording_id,
                "subject_id": str(row.subject_id),
                "task_stratum": str(row.task_stratum),
            }
        )
    return pd.DataFrame(rows)


def construct_reference_from_members(
    spectra: Mapping[str, RecordingSpectrum],
    members: pd.DataFrame,
    *,
    task_stratum: str,
    excluded_subject_id: str,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
    mode: str = "subject_balanced",
    reference_label: str = "audit",
) -> ReferenceSpectrum:
    if mode not in {"subject_balanced", "recording_weighted"}:
        raise ValueError(f"Unknown reference mode: {mode}")
    member_ids = tuple(sorted(members["logical_recording_id"].astype(str)))
    member_subjects = tuple(sorted(set(members["subject_id"].astype(str))))
    key = stable_hash(
        {
            "label": reference_label,
            "task_stratum": str(task_stratum),
            "excluded_subject_id": str(excluded_subject_id),
            "mode": mode,
            "member_ids": member_ids,
            "parameters": parameters.to_dict(),
        }
    )[:20]
    vintage = stable_hash(
        {
            "measurement_version": MEASUREMENT_VERSION,
            "task_stratum": str(task_stratum),
            "mode": mode,
            "eligible_members": member_ids,
            "parameters": parameters.to_dict(),
        }
    )
    if (
        len(member_ids) < parameters.minimum_reference_recordings
        or len(member_subjects) < parameters.minimum_reference_subjects
    ):
        frequency = next(
            (
                spectrum.frequencies_hz
                for spectrum in spectra.values()
                if len(spectrum.frequencies_hz)
            ),
            np.array([], dtype=float),
        )
        return ReferenceSpectrum(
            reference_key=key,
            task_stratum=str(task_stratum),
            excluded_subject_id=str(excluded_subject_id),
            frequencies_hz=np.asarray(frequency, dtype=float),
            normalized_psd_per_hz=np.full(len(frequency), np.nan),
            status="reference_unavailable",
            member_recording_ids=member_ids,
            member_subject_ids=member_subjects,
            recording_count=len(member_ids),
            subject_count=len(member_subjects),
            reference_sha256="",
            reference_vintage_sha256=vintage,
        )

    frequency = None
    if mode == "subject_balanced":
        components = []
        for subject_id in member_subjects:
            ids = members.loc[
                members["subject_id"].astype(str).eq(subject_id),
                "logical_recording_id",
            ].astype(str)
            local = [spectra[item] for item in ids]
            if frequency is None:
                frequency = local[0].frequencies_hz
            if any(
                not np.array_equal(frequency, spectrum.frequencies_hz)
                for spectrum in local
            ):
                raise ValueError("Reference spectra do not share a frequency grid.")
            subject_psd = np.median(
                np.vstack([spectrum.normalized_psd_per_hz for spectrum in local]),
                axis=0,
            )
            components.append(renormalize_psd(frequency, subject_psd, parameters))
    else:
        local = [spectra[item] for item in member_ids]
        frequency = local[0].frequencies_hz
        if any(
            not np.array_equal(frequency, spectrum.frequencies_hz)
            for spectrum in local
        ):
            raise ValueError("Reference spectra do not share a frequency grid.")
        components = [
            np.asarray(spectrum.normalized_psd_per_hz, dtype=float)
            for spectrum in local
        ]

    reference_psd = renormalize_psd(
        frequency,
        np.median(np.vstack(components), axis=0),
        parameters,
    )
    reference_hash = stable_hash(
        {
            "reference_key": key,
            "mode": mode,
            "member_recording_ids": member_ids,
            "member_subject_ids": member_subjects,
            "psd_sha256": array_hash(reference_psd),
            "parameters": parameters.to_dict(),
        }
    )
    return ReferenceSpectrum(
        reference_key=key,
        task_stratum=str(task_stratum),
        excluded_subject_id=str(excluded_subject_id),
        frequencies_hz=np.asarray(frequency, dtype=float),
        normalized_psd_per_hz=reference_psd,
        status="measured",
        member_recording_ids=member_ids,
        member_subject_ids=member_subjects,
        recording_count=len(member_ids),
        subject_count=len(member_subjects),
        reference_sha256=reference_hash,
        reference_vintage_sha256=vintage,
    )



def gallery_linked_view_source(
    waveform_source: pd.DataFrame,
    spectrogram_source: pd.DataFrame,
    ltas_source: pd.DataFrame,
    feature_source: pd.DataFrame,
    *,
    unavailable_reason: str = "reference_relative_ltas_unavailable",
) -> pd.DataFrame:
    """Return canonical long-form source data for a QCHAN Panel G example.

    The publication figure shows five linked views: waveform, spectrogram,
    target LTAS, frozen reference LTAS, and target-minus-reference LTAS.
    Each view is represented explicitly in the saved source CSV through the
    ``view`` column. When reference-relative LTAS is unavailable, placeholder
    rows retain all three LTAS view identities with ``view_available=False``;
    missing evidence is therefore explicit rather than silently omitted.

    ``row_type`` is retained for backward compatibility with earlier cohort
    artifacts. ``linked_value`` provides the plotted ordinate for each LTAS
    view without requiring downstream code to infer which legacy value column
    is active.
    """

    def _with_view(frame: pd.DataFrame, view: str) -> pd.DataFrame:
        local = frame.copy()
        local["view"] = str(view)
        local["view_available"] = bool(len(local))
        local["view_unavailable_reason"] = ""
        if "linked_value" not in local.columns:
            local["linked_value"] = np.nan
        return local

    waveform = _with_view(waveform_source, "waveform")
    spectrogram = _with_view(spectrogram_source, "spectrogram")

    ltas_frames: list[pd.DataFrame] = []
    if len(ltas_source):
        mapping = {
            "target_ltas": "observation_ltas_db",
            "reference_ltas": "reference_ltas_db",
            "ltas_difference": "difference_db",
        }
        for view, value_column in mapping.items():
            local = _with_view(ltas_source, view)
            local["linked_value"] = pd.to_numeric(
                local[value_column], errors="coerce"
            )
            for other in mapping.values():
                if other != value_column:
                    local[other] = np.nan
            ltas_frames.append(local)
    else:
        template_columns = list(
            dict.fromkeys(
                [
                    *waveform_source.columns.tolist(),
                    *spectrogram_source.columns.tolist(),
                    *feature_source.columns.tolist(),
                    "row_type",
                    "logical_recording_id",
                    "selection_reason",
                    "time_sec",
                    "frequency_hz",
                    "amplitude",
                    "power_db",
                    "observation_ltas_db",
                    "reference_ltas_db",
                    "difference_db",
                    "view",
                    "view_available",
                    "view_unavailable_reason",
                    "linked_value",
                ]
            )
        )
        common = {}
        for frame in (waveform_source, spectrogram_source, feature_source):
            if len(frame):
                for column in [
                    "logical_recording_id",
                    "selection_reason",
                    "selection_source",
                    "selection_order",
                ]:
                    if column in frame.columns and column not in common:
                        common[column] = frame.iloc[0][column]
        for view in ("target_ltas", "reference_ltas", "ltas_difference"):
            row = {column: np.nan for column in template_columns}
            row.update(common)
            row.update(
                {
                    "row_type": "ltas_unavailable",
                    "view": view,
                    "view_available": False,
                    "view_unavailable_reason": str(unavailable_reason),
                    "linked_value": np.nan,
                }
            )
            ltas_frames.append(pd.DataFrame([row], columns=template_columns))

    features = _with_view(feature_source, "feature_metadata")
    source = pd.concat(
        [waveform, spectrogram, *ltas_frames, features],
        ignore_index=True,
        sort=False,
    )

    observed_views = set(source["view"].astype(str))
    missing_views = set(REQUIRED_GALLERY_LINKED_VIEWS) - observed_views
    if missing_views:
        raise RuntimeError(
            "QCHAN gallery source is missing required linked views: "
            f"{sorted(missing_views)}"
        )
    return source


def stable_hash_order(value: object, *, seed: int) -> str:
    return sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest()


def deterministic_stratified_sample(
    frame: pd.DataFrame,
    *,
    maximum_rows: int,
    stratum_columns: Sequence[str],
    id_column: str = "logical_recording_id",
    seed: int = DEFAULT_PARAMETERS.random_seed,
) -> pd.DataFrame:
    local = frame.copy()
    if local.empty:
        return local
    local["_stable_order"] = local[id_column].astype(str).map(
        lambda value: stable_hash_order(value, seed=seed)
    )
    if len(local) <= int(maximum_rows):
        return local.sort_values("_stable_order").drop(columns="_stable_order")
    groups = [
        group.sort_values("_stable_order").reset_index(drop=True)
        for _, group in local.groupby(
            list(stratum_columns), dropna=False, sort=True
        )
    ]
    selected = []
    row_index = 0
    while len(selected) < int(maximum_rows):
        added = False
        for group in groups:
            if row_index < len(group):
                selected.append(group.iloc[row_index])
                added = True
                if len(selected) >= int(maximum_rows):
                    break
        if not added:
            break
        row_index += 1
    return pd.DataFrame(selected).drop(columns="_stable_order", errors="ignore")


def deterministic_gallery_selection(
    recording_table: pd.DataFrame,
    *,
    maximum_rows: int = 10,
    minimum_rows: int = 8,
    id_column: str = "logical_recording_id",
    seed: int = DEFAULT_PARAMETERS.random_seed,
) -> pd.DataFrame:
    """Select unique, label-blind QCHAN gallery examples deterministically.

    Prespecified signal-derived strata are considered in a fixed order. When
    the most extreme recording for a stratum has already been selected for an
    earlier stratum, the next-ranked unused recording is chosen rather than
    silently dropping that stratum. If fewer than ``minimum_rows`` strata are
    available, unused measured recordings are added through deterministic
    support/bandwidth/LTAS-stratified sampling. No clinical, diagnostic, or
    human-QC field is consulted.

    The function never duplicates a recording merely to satisfy the requested
    minimum. Therefore, a genuinely undersized input remains visibly
    incomplete and the downstream gallery gate can fail honestly.
    """
    maximum_rows = int(maximum_rows)
    minimum_rows = int(minimum_rows)
    if maximum_rows < 1:
        raise ValueError("maximum_rows must be at least one")
    if minimum_rows < 0:
        raise ValueError("minimum_rows must be nonnegative")
    if minimum_rows > maximum_rows:
        raise ValueError("minimum_rows cannot exceed maximum_rows")
    if id_column not in recording_table.columns:
        raise KeyError(f"Missing gallery identity column: {id_column}")

    local = recording_table.copy()
    if local.empty:
        return pd.DataFrame(
            columns=[
                id_column,
                "selection_reason",
                "selection_source",
                "selection_order",
            ]
        )

    local[id_column] = local[id_column].astype(str)
    local = local.drop_duplicates(id_column, keep="first").copy()
    selected: list[dict[str, object]] = []
    used_ids: set[str] = set()

    def append_row(recording_id: str, reason: str, source: str) -> None:
        if len(selected) >= maximum_rows or recording_id in used_ids:
            return
        used_ids.add(recording_id)
        selected.append(
            {
                id_column: recording_id,
                "selection_reason": reason,
                "selection_source": source,
                "selection_order": len(selected) + 1,
            }
        )

    def choose_ranked(
        reason: str,
        sort_column: str,
        *,
        ascending: bool,
        mask: pd.Series | None = None,
    ) -> None:
        if len(selected) >= maximum_rows or sort_column not in local.columns:
            return
        candidates = local.copy()
        if mask is not None:
            aligned = pd.Series(mask, index=local.index).fillna(False).astype(bool)
            candidates = candidates.loc[aligned]
        candidates["_sort_value"] = pd.to_numeric(
            candidates[sort_column], errors="coerce"
        )
        candidates = candidates.loc[
            candidates["_sort_value"].map(np.isfinite)
            & ~candidates[id_column].isin(used_ids)
        ]
        candidates = candidates.sort_values(
            ["_sort_value", id_column],
            ascending=[ascending, True],
            kind="mergesort",
        )
        if len(candidates):
            append_row(
                str(candidates.iloc[0][id_column]),
                reason,
                "prespecified_signal_stratum",
            )

    choose_ranked(
        "low_ltas_distance",
        "qchan_ltas_distance_db",
        ascending=True,
    )
    choose_ranked(
        "high_ltas_distance",
        "qchan_ltas_distance_db",
        ascending=False,
    )
    choose_ranked(
        "high_rolloff_deficit",
        "qchan_rolloff95_deficit_hz",
        ascending=False,
    )
    choose_ranked(
        "high_highband_deficit",
        "qchan_highband_ratio_deficit",
        ascending=False,
    )
    choose_ranked(
        "high_tilt_steepening",
        "qchan_tilt_steepening_db_per_oct",
        ascending=False,
    )

    if {
        "qchan_rolloff95_deficit_hz",
        "qchan_highband_ratio_deficit",
    }.issubset(local.columns):
        rolloff = pd.to_numeric(
            local["qchan_rolloff95_deficit_hz"], errors="coerce"
        )
        highband = pd.to_numeric(
            local["qchan_highband_ratio_deficit"], errors="coerce"
        )
        coloration_mask = (
            rolloff.fillna(np.inf).le(1e-12)
            & highband.fillna(np.inf).le(1e-12)
        )
        choose_ranked(
            "coloration_without_one_sided_bandwidth_deficit",
            "qchan_ltas_distance_db",
            ascending=False,
            mask=coloration_mask,
        )

    if "qchan_source_bandwidth_limited" in local.columns:
        bandwidth_mask = as_bool(
            local["qchan_source_bandwidth_limited"]
        )
        choose_ranked(
            "native_bandwidth_limited",
            "qchan_ltas_distance_db",
            ascending=False,
            mask=bandwidth_mask,
        )

    if "qchan_family_status" in local.columns and len(selected) < maximum_rows:
        unavailable = local.loc[
            ~local["qchan_family_status"].fillna("").astype(str).eq("measured")
            & ~local[id_column].isin(used_ids)
        ].sort_values(id_column, kind="mergesort")
        if len(unavailable):
            append_row(
                str(unavailable.iloc[0][id_column]),
                "conditional_features_unavailable",
                "prespecified_signal_stratum",
            )

    if len(selected) < maximum_rows and "qchan_ltas_distance_db" in local.columns:
        if "qchan_family_status" in local.columns:
            measured_mask = local["qchan_family_status"].fillna("").astype(str).eq(
                "measured"
            )
        else:
            measured_mask = pd.Series(True, index=local.index)
        measured = local.loc[
            measured_mask & ~local[id_column].isin(used_ids)
        ].copy()
        measured["_ltas"] = pd.to_numeric(
            measured["qchan_ltas_distance_db"], errors="coerce"
        )
        measured = measured.loc[measured["_ltas"].map(np.isfinite)]
        if len(measured):
            cohort_median = float(
                pd.to_numeric(
                    local.loc[measured_mask, "qchan_ltas_distance_db"],
                    errors="coerce",
                ).median()
            )
            measured["_median_distance"] = (
                measured["_ltas"] - cohort_median
            ).abs()
            measured = measured.sort_values(
                ["_median_distance", id_column],
                ascending=[True, True],
                kind="mergesort",
            )
            append_row(
                str(measured.iloc[0][id_column]),
                "median_measured_profile",
                "prespecified_signal_stratum",
            )

    choose_ranked(
        "strong_upward_signed_rolloff_difference",
        "qchan_rolloff95_signed_difference_hz",
        ascending=True,
    )

    rows_needed = min(maximum_rows, minimum_rows) - len(selected)
    if rows_needed > 0:
        remaining = local.loc[~local[id_column].isin(used_ids)].copy()
        if "qchan_family_status" in remaining.columns:
            measured_remaining = remaining.loc[
                remaining["qchan_family_status"]
                .fillna("")
                .astype(str)
                .eq("measured")
            ].copy()
            if len(measured_remaining):
                remaining = measured_remaining

        if len(remaining):
            if "qchan_support_tier" not in remaining.columns:
                remaining["qchan_support_tier"] = "unspecified"
            if "qchan_source_bandwidth_limited" in remaining.columns:
                remaining["_bandwidth_stratum"] = as_bool(
                    remaining["qchan_source_bandwidth_limited"]
                ).astype(str)
            else:
                remaining["_bandwidth_stratum"] = "unspecified"

            if "qchan_ltas_distance_db" in remaining.columns:
                ltas = pd.to_numeric(
                    remaining["qchan_ltas_distance_db"], errors="coerce"
                )
                finite_count = int(ltas.map(np.isfinite).sum())
                bin_count = min(4, max(1, finite_count))
                if finite_count and ltas.loc[ltas.map(np.isfinite)].nunique() > 1:
                    remaining["_ltas_stratum"] = pd.qcut(
                        ltas.rank(method="first"),
                        q=bin_count,
                        duplicates="drop",
                    ).astype(str)
                else:
                    remaining["_ltas_stratum"] = "unspecified"
            else:
                remaining["_ltas_stratum"] = "unspecified"

            fill = deterministic_stratified_sample(
                remaining,
                maximum_rows=min(rows_needed, len(remaining)),
                stratum_columns=[
                    "qchan_support_tier",
                    "_bandwidth_stratum",
                    "_ltas_stratum",
                ],
                id_column=id_column,
                seed=seed,
            )
            for row in fill.itertuples(index=False):
                append_row(
                    str(getattr(row, id_column)),
                    f"deterministic_diversity_fill_{len(selected) + 1:02d}",
                    "deterministic_support_bandwidth_ltas_fill",
                )

    result = pd.DataFrame(
        selected,
        columns=[
            id_column,
            "selection_reason",
            "selection_source",
            "selection_order",
        ],
    )
    if len(result):
        if result[id_column].duplicated().any():
            raise RuntimeError("QCHAN gallery selection contains duplicate recordings")
        if result["selection_reason"].duplicated().any():
            raise RuntimeError("QCHAN gallery selection reasons are not unique")
    return result


def empirical_feature_summary(
    recording_table: pd.DataFrame,
    features: Sequence[str] = ANALYSIS_FEATURES,
) -> pd.DataFrame:
    rows = []
    total = len(recording_table)
    for feature in features:
        values = pd.to_numeric(recording_table[feature], errors="coerce")
        finite = values.loc[np.isfinite(values)]
        zero_count = int(np.isclose(finite, 0.0, atol=1e-12).sum())
        rows.append(
            {
                "feature": feature,
                "recordings": total,
                "available_n": len(finite),
                "availability_fraction": len(finite) / max(1, total),
                "zero_n": zero_count,
                "zero_fraction_among_available": (
                    zero_count / len(finite) if len(finite) else np.nan
                ),
                "median": float(finite.median()) if len(finite) else np.nan,
                "q25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                "q75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                "p01": float(finite.quantile(0.01)) if len(finite) else np.nan,
                "p99": float(finite.quantile(0.99)) if len(finite) else np.nan,
                "minimum": float(finite.min()) if len(finite) else np.nan,
                "maximum": float(finite.max()) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def support_availability_summary(recording_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(recording_table)
    tiers = ["unavailable", "minimum", "moderate", "high"]
    for tier in tiers:
        local = recording_table.loc[
            recording_table["qchan_support_tier"].fillna("unavailable").astype(str).eq(tier)
        ]
        for feature in ANALYSIS_FEATURES:
            available = pd.to_numeric(local[feature], errors="coerce").notna()
            rows.append(
                {
                    "support_class": tier,
                    "feature": feature,
                    "recording_count": len(local),
                    "available_n": int(available.sum()),
                    "availability_fraction_within_class": (
                        float(available.mean()) if len(local) else np.nan
                    ),
                    "fraction_of_cohort": len(local) / max(1, total),
                }
            )
    return pd.DataFrame(rows)


def status_missingness_summary(recording_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(recording_table)
    for feature in ANALYSIS_FEATURES:
        status_column = f"{feature}_status"
        statuses = (
            recording_table[status_column].fillna("missing_status").astype(str)
            if status_column in recording_table
            else pd.Series(
                np.where(
                    pd.to_numeric(recording_table[feature], errors="coerce").notna(),
                    "measured",
                    "unavailable",
                )
            )
        )
        for status, count in statuses.value_counts(dropna=False).items():
            rows.append(
                {
                    "feature": feature,
                    "status": str(status),
                    "recording_count": int(count),
                    "fraction": int(count) / max(1, total),
                }
            )
    return pd.DataFrame(rows)


def native_bandwidth_summary(recording_table: pd.DataFrame) -> pd.DataFrame:
    local = recording_table.copy()
    local["qchan_source_sample_rate_hz"] = pd.to_numeric(
        local["qchan_source_sample_rate_hz"], errors="coerce"
    )
    local["qchan_source_nyquist_hz"] = pd.to_numeric(
        local["qchan_source_nyquist_hz"], errors="coerce"
    )
    local["qchan_source_bandwidth_limited"] = as_bool(
        local["qchan_source_bandwidth_limited"]
    )
    rows = []
    for keys, group in local.groupby(
        ["qchan_source_sample_rate_hz", "qchan_source_bandwidth_limited"],
        dropna=False,
        sort=True,
    ):
        row = {
            "source_sample_rate_hz": keys[0],
            "source_bandwidth_limited": bool(keys[1]),
            "recording_count": len(group),
            "participant_count": (
                group["SubjectID"].nunique(dropna=True)
                if "SubjectID" in group
                else np.nan
            ),
        }
        for feature in ANALYSIS_FEATURES:
            row[f"{feature}__available_n"] = int(
                pd.to_numeric(group[feature], errors="coerce").notna().sum()
            )
            row[f"{feature}__median"] = float(
                pd.to_numeric(group[feature], errors="coerce").median()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def icc1_balanced_first_two(
    frame: pd.DataFrame,
    *,
    subject_column: str,
    date_column: str,
    feature: str,
) -> dict[str, float | int]:
    local = frame[[subject_column, date_column, feature]].copy()
    local[date_column] = pd.to_datetime(local[date_column], errors="coerce")
    local[feature] = pd.to_numeric(local[feature], errors="coerce")
    local = local.sort_values([subject_column, date_column])
    local["repeat_index"] = local.groupby(subject_column).cumcount()
    pair = local.loc[local["repeat_index"].isin([0, 1])].pivot(
        index=subject_column,
        columns="repeat_index",
        values=feature,
    ).dropna()
    if len(pair) < 3:
        return {"subject_count": len(pair), "icc1": np.nan}
    values = pair.to_numpy(float)
    n, k = values.shape
    subject_means = values.mean(axis=1)
    grand = values.mean()
    ss_between = k * np.sum((subject_means - grand) ** 2)
    ss_within = np.sum((values - subject_means[:, None]) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    denominator = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator else np.nan
    return {"subject_count": int(n), "icc1": float(icc)}


def repeated_recording_persistence(
    frame: pd.DataFrame,
    *,
    subject_column: str,
    date_column: str,
    features: Sequence[str] = ANALYSIS_FEATURES,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        local = frame[
            [subject_column, date_column, "logical_recording_id", feature]
        ].copy()
        local[date_column] = pd.to_datetime(local[date_column], errors="coerce")
        local[feature] = pd.to_numeric(local[feature], errors="coerce")
        local = local.sort_values(
            [subject_column, date_column, "logical_recording_id"]
        )
        local["repeat_index"] = local.groupby(subject_column).cumcount()
        pair = local.loc[local["repeat_index"].isin([0, 1])].pivot(
            index=subject_column,
            columns="repeat_index",
            values=feature,
        )
        pair = pair.rename(columns={0: "first", 1: "second"}).dropna()
        rho = (
            float(stats.spearmanr(pair["first"], pair["second"]).statistic)
            if len(pair) >= 3
            and pair["first"].nunique() > 1
            and pair["second"].nunique() > 1
            else np.nan
        )
        difference = (pair["second"] - pair["first"]).abs()
        icc = icc1_balanced_first_two(
            frame,
            subject_column=subject_column,
            date_column=date_column,
            feature=feature,
        )
        rows.append(
            {
                "feature": feature,
                "paired_subject_count": len(pair),
                "first_second_spearman": rho,
                "icc1_first_two": icc["icc1"],
                "icc_subject_count": icc["subject_count"],
                "median_absolute_difference": (
                    float(difference.median()) if len(difference) else np.nan
                ),
                "p90_absolute_difference": (
                    float(difference.quantile(0.90))
                    if len(difference)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def pairwise_redundancy(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            pair = frame[[left, right]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            rho = (
                float(stats.spearmanr(pair[left], pair[right]).statistic)
                if len(pair) >= 3
                and pair[left].nunique() > 1
                and pair[right].nunique() > 1
                else np.nan
            )
            rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "pairwise_n": len(pair),
                    "spearman_rho": rho,
                }
            )
    return pd.DataFrame(rows)


def participant_balanced_resampling(
    frame: pd.DataFrame,
    *,
    subject_column: str,
    features: Sequence[str] = ANALYSIS_FEATURES,
    iterations: int = 1000,
    seed: int = DEFAULT_PARAMETERS.random_seed,
) -> pd.DataFrame:
    local = frame.copy()
    local[subject_column] = local[subject_column].astype(str)
    groups = {
        subject: group.copy()
        for subject, group in local.groupby(subject_column, sort=True)
    }
    subjects = sorted(groups)
    rng = np.random.default_rng(int(seed))
    rows = []
    for iteration in range(int(iterations)):
        selected = pd.concat(
            [
                group.iloc[[int(rng.integers(0, len(group)))]]
                for group in groups.values()
            ],
            ignore_index=True,
        )
        for feature in features:
            values = pd.to_numeric(selected[feature], errors="coerce").dropna()
            rows.append(
                {
                    "iteration": iteration,
                    "feature": feature,
                    "participant_count": len(subjects),
                    "available_participant_count": len(values),
                    "availability_fraction": len(values) / max(1, len(subjects)),
                    "median": float(values.median()) if len(values) else np.nan,
                    "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                    "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def participant_balanced_summary(resampling: pd.DataFrame) -> pd.DataFrame:
    if resampling.empty:
        return pd.DataFrame()
    return (
        resampling.groupby("feature", as_index=False)
        .agg(
            iterations=("iteration", "nunique"),
            median_of_medians=("median", "median"),
            p025_median=("median", lambda x: x.quantile(0.025)),
            p975_median=("median", lambda x: x.quantile(0.975)),
            median_availability_fraction=(
                "availability_fraction", "median"
            ),
            p025_availability_fraction=(
                "availability_fraction", lambda x: x.quantile(0.025)
            ),
            p975_availability_fraction=(
                "availability_fraction", lambda x: x.quantile(0.975)
            ),
        )
    )


def reference_robustness_grid(
    targets: pd.DataFrame,
    *,
    spectra: Mapping[str, RecordingSpectrum],
    metadata: pd.DataFrame,
    baseline_references: Mapping[str, ReferenceSpectrum],
    bootstrap_iterations: int = 200,
    maximum_delete_subjects: int = 32,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
    seed: int = DEFAULT_PARAMETERS.random_seed,
) -> pd.DataFrame:
    rows = []
    metadata_lookup = metadata.set_index("logical_recording_id")
    for target_row in targets.itertuples(index=False):
        recording_id = str(target_row.logical_recording_id)
        observation = spectra[recording_id]
        target_meta = metadata_lookup.loc[recording_id]
        subject_id = str(target_meta["subject_id"])
        task = str(target_meta["task_stratum"])
        baseline_reference = baseline_references[recording_id]
        baseline = compute_reference_relative_features(
            observation, baseline_reference, parameters
        )
        members = eligible_reference_members(
            spectra,
            metadata,
            task_stratum=task,
            excluded_subject_id=subject_id,
            parameters=parameters,
        )
        member_subjects = sorted(members["subject_id"].astype(str).unique())

        # Recording-weighted alternative.
        recording_weighted = construct_reference_from_members(
            spectra,
            members,
            task_stratum=task,
            excluded_subject_id=subject_id,
            parameters=parameters,
            mode="recording_weighted",
            reference_label="recording_weighted",
        )
        alternative = compute_reference_relative_features(
            observation, recording_weighted, parameters
        )
        for feature in ANALYSIS_FEATURES:
            rows.append(
                {
                    "logical_recording_id": recording_id,
                    "comparison": "recording_weighted",
                    "iteration": 0,
                    "feature": feature,
                    "baseline_value": baseline[feature],
                    "variant_value": alternative[feature],
                    "absolute_delta": abs(alternative[feature] - baseline[feature])
                    if np.isfinite(alternative[feature])
                    and np.isfinite(baseline[feature])
                    else np.nan,
                    "reference_subject_count": recording_weighted.subject_count,
                    "reference_recording_count": recording_weighted.recording_count,
                }
            )

        # Two deterministic 80% membership vintages.
        for vintage_index in (1, 2):
            ordered = sorted(
                member_subjects,
                key=lambda value: stable_hash_order(
                    f"{recording_id}|{vintage_index}|{value}",
                    seed=seed + vintage_index,
                ),
            )
            keep_n = max(
                parameters.minimum_reference_subjects,
                int(math.ceil(0.80 * len(ordered))),
            )
            chosen = set(ordered[:keep_n])
            vintage_members = members.loc[
                members["subject_id"].astype(str).isin(chosen)
            ]
            vintage_reference = construct_reference_from_members(
                spectra,
                vintage_members,
                task_stratum=task,
                excluded_subject_id=subject_id,
                parameters=parameters,
                mode="subject_balanced",
                reference_label=f"vintage80_{vintage_index}",
            )
            values = compute_reference_relative_features(
                observation, vintage_reference, parameters
            )
            for feature in ANALYSIS_FEATURES:
                rows.append(
                    {
                        "logical_recording_id": recording_id,
                        "comparison": f"vintage80_{vintage_index}",
                        "iteration": 0,
                        "feature": feature,
                        "baseline_value": baseline[feature],
                        "variant_value": values[feature],
                        "absolute_delta": abs(values[feature] - baseline[feature])
                        if np.isfinite(values[feature])
                        and np.isfinite(baseline[feature])
                        else np.nan,
                        "reference_subject_count": vintage_reference.subject_count,
                        "reference_recording_count": vintage_reference.recording_count,
                    }
                )

        # Delete-one-reference-subject audit. Deterministically cap very large pools.
        delete_subjects = sorted(
            member_subjects,
            key=lambda value: stable_hash_order(
                f"{recording_id}|delete|{value}", seed=seed
            ),
        )[: int(maximum_delete_subjects)]
        for omitted_index, omitted_subject in enumerate(delete_subjects):
            reduced_members = members.loc[
                ~members["subject_id"].astype(str).eq(omitted_subject)
            ]
            reduced_reference = construct_reference_from_members(
                spectra,
                reduced_members,
                task_stratum=task,
                excluded_subject_id=subject_id,
                parameters=parameters,
                mode="subject_balanced",
                reference_label=f"delete_{omitted_subject}",
            )
            values = compute_reference_relative_features(
                observation, reduced_reference, parameters
            )
            for feature in ANALYSIS_FEATURES:
                rows.append(
                    {
                        "logical_recording_id": recording_id,
                        "comparison": "delete_one_reference_subject",
                        "iteration": omitted_index,
                        "omitted_subject_id": omitted_subject,
                        "feature": feature,
                        "baseline_value": baseline[feature],
                        "variant_value": values[feature],
                        "absolute_delta": abs(values[feature] - baseline[feature])
                        if np.isfinite(values[feature])
                        and np.isfinite(baseline[feature])
                        else np.nan,
                        "reference_subject_count": reduced_reference.subject_count,
                        "reference_recording_count": reduced_reference.recording_count,
                    }
                )

        # Subject bootstrap with subject-level resampling and recording retention.
        stable_seed = int(
            stable_hash_order(recording_id, seed=seed)[:16], 16
        ) % (2**32 - 1)
        rng = np.random.default_rng(stable_seed)
        for iteration in range(int(bootstrap_iterations)):
            sampled_subjects = rng.choice(
                member_subjects,
                size=len(member_subjects),
                replace=True,
            )
            # Duplicate sampled participants by assigning synthetic identities so
            # the across-subject median reflects bootstrap multiplicity.
            bootstrap_rows = []
            for draw_index, sampled_subject in enumerate(sampled_subjects):
                local = members.loc[
                    members["subject_id"].astype(str).eq(str(sampled_subject))
                ].copy()
                local["subject_id"] = (
                    local["subject_id"].astype(str)
                    + f"__bootstrap_draw_{draw_index:04d}"
                )
                bootstrap_rows.append(local)
            boot_members = pd.concat(bootstrap_rows, ignore_index=True)
            boot_reference = construct_reference_from_members(
                spectra,
                boot_members,
                task_stratum=task,
                excluded_subject_id=subject_id,
                parameters=parameters,
                mode="subject_balanced",
                reference_label=f"bootstrap_{iteration}",
            )
            values = compute_reference_relative_features(
                observation, boot_reference, parameters
            )
            for feature in ANALYSIS_FEATURES:
                rows.append(
                    {
                        "logical_recording_id": recording_id,
                        "comparison": "subject_bootstrap",
                        "iteration": iteration,
                        "feature": feature,
                        "baseline_value": baseline[feature],
                        "variant_value": values[feature],
                        "absolute_delta": abs(values[feature] - baseline[feature])
                        if np.isfinite(values[feature])
                        and np.isfinite(baseline[feature])
                        else np.nan,
                        "reference_subject_count": boot_reference.subject_count,
                        "reference_recording_count": boot_reference.recording_count,
                    }
                )
    return pd.DataFrame(rows)


def summarize_reference_robustness(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    rows = []
    for keys, local in grid.groupby(["comparison", "feature"], sort=True):
        delta = pd.to_numeric(local["absolute_delta"], errors="coerce").dropna()
        baseline = pd.to_numeric(local["baseline_value"], errors="coerce")
        variant = pd.to_numeric(local["variant_value"], errors="coerce")
        pair = pd.DataFrame({"baseline": baseline, "variant": variant}).dropna()
        rho = (
            float(stats.spearmanr(pair["baseline"], pair["variant"]).statistic)
            if len(pair) >= 3
            and pair["baseline"].nunique() > 1
            and pair["variant"].nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "comparison": keys[0],
                "feature": keys[1],
                "target_recording_count": local["logical_recording_id"].nunique(),
                "comparison_rows": len(local),
                "paired_finite_n": len(pair),
                "spearman_rho": rho,
                "median_absolute_delta": (
                    float(delta.median()) if len(delta) else np.nan
                ),
                "p95_absolute_delta": (
                    float(delta.quantile(0.95)) if len(delta) else np.nan
                ),
                "maximum_absolute_delta": (
                    float(delta.max()) if len(delta) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def model_interface_frame(recording_table: pd.DataFrame) -> pd.DataFrame:
    identity_columns = [
        column
        for column in [
            "logical_recording_id",
            "SubjectID",
            "recording_date_analysis",
            "qchan_measurement_version",
            "qchan_reference_key",
            "qchan_reference_sha256",
            "qchan_reference_vintage_sha256",
        ]
        if column in recording_table
    ]
    output = recording_table[identity_columns].copy()
    for feature in ANALYSIS_FEATURES:
        output[feature] = pd.to_numeric(
            recording_table[feature], errors="coerce"
        )
        output[f"{feature}__available"] = output[feature].notna()
        status_column = f"{feature}_status"
        output[f"{feature}__status"] = (
            recording_table[status_column].astype(str)
            if status_column in recording_table
            else np.where(
                output[f"{feature}__available"], "measured", "unavailable"
            )
        )
        output[f"{feature}__missing_reason"] = np.where(
            output[f"{feature}__available"],
            "",
            output[f"{feature}__status"],
        )
        tier_column = f"{feature}_support_tier"
        output[f"{feature}__support_tier"] = (
            recording_table[tier_column].astype(str)
            if tier_column in recording_table
            else np.where(
                output[f"{feature}__available"],
                "available",
                "unavailable",
            )
        )

    for column in [
        "qchan_guarded_speech_support_sec",
        "qchan_valid_frame_count",
        "qchan_guarded_segment_count",
        "qchan_zero_frame_count",
        "qchan_source_sample_rate_hz",
        "qchan_source_nyquist_hz",
        "qchan_source_bandwidth_limited",
        "qchan_support_tier",
        "qchan_reference_recording_count",
        "qchan_reference_subject_count",
        "qchan_family_status",
        "qchan_rolloff95_signed_difference_hz",
        "qchan_highband_ratio_signed_difference",
        "qchan_tilt_signed_difference_db_per_oct",
    ]:
        if column in recording_table:
            output[column] = recording_table[column]

    output["qchan_family_scalar_available"] = False
    output["qchan_standalone_reject_allowed"] = False
    output["qchan_device_identity_status"] = "not_estimated"
    output["qchan_decision_threshold_status"] = "not_calibrated"
    return output


def ltas_source_frame(
    observation: RecordingSpectrum,
    reference: ReferenceSpectrum,
    parameters: QChanParameters = DEFAULT_PARAMETERS,
) -> pd.DataFrame:
    if observation.status != "measured" or reference.status != "measured":
        return pd.DataFrame()
    centers, observation_db = smoothed_log_ltas_db(
        observation.frequencies_hz,
        observation.normalized_psd_per_hz,
        parameters,
    )
    reference_centers, reference_db = smoothed_log_ltas_db(
        reference.frequencies_hz,
        reference.normalized_psd_per_hz,
        parameters,
    )
    if not np.array_equal(centers, reference_centers):
        raise ValueError("Observation and reference LTAS centers differ.")
    return pd.DataFrame(
        {
            "center_frequency_hz": centers,
            "observation_ltas_db": observation_db,
            "reference_ltas_db": reference_db,
            "difference_db": observation_db - reference_db,
        }
    )


def hash_inventory(root: Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)
