"""Cohort orchestration helpers for QADD v4.2.0 reviewed candidate.

This module does not redefine the five QADD estimators.  It provides the
versioned cohort-input contract, interval-provenance mapping, hum-null
calibration, empirical summaries, persistence metrics, subject-balanced
resampling, and model-facing exports used by the reviewed cohort notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Iterable, Mapping, Sequence
import json

import numpy as np
import pandas as pd
from scipy import signal, stats

from paper1_qc_reviewed.qadd_v420 import (
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    QADDParameters,
    TimeInterval,
    hum_comb_score_from_psd,
    power_spectrum,
)

COHORT_ORCHESTRATION_VERSION = "qadd-v4.2.0-cohort-orchestration-v1"
CANONICAL_PROFILE = "primary"
CANONICAL_PRIMARY_VIEW = "primary_speech"
CANONICAL_SPEECH_VIEW = "strict_speech"
CANONICAL_PAUSE_VIEW = "strict_internal_nonspeech"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    """Coerce common CSV boolean encodings without treating nonempty strings as True."""

    return series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"1", "true", "yes", "y"}
    )


def json_safe(value):
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
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(payload: Mapping, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(dict(payload)), indent=2), encoding="utf-8")


def resolve_media_path(
    raw_path: object,
    *,
    media_root_override: Path | None = None,
    media_path_map: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a frozen Windows path on the current machine.

    If a direct path does not exist, the path segment after
    ``Bamboo_passage_only`` is appended to ``media_root_override``.
    """

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
    view: str,
    profile: str = CANONICAL_PROFILE,
    eligible_only: bool = True,
) -> pd.DataFrame:
    """Select exactly one frozen segmentation view/profile without fallback."""

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
        local = local.loc[local["decision"].astype(str).str.upper().eq("KEEP")]

    local["logical_recording_id"] = local["logical_recording_id"].astype(str)
    local["interval_index"] = pd.to_numeric(
        local["interval_index"], errors="raise"
    ).astype(int)
    local["start_sec"] = pd.to_numeric(local["start_sec"], errors="raise")
    local["end_sec"] = pd.to_numeric(local["end_sec"], errors="raise")
    local = local.sort_values(
        ["logical_recording_id", "start_sec", "end_sec", "interval_index"]
    ).reset_index(drop=True)

    identity = ["logical_recording_id", "interval_index"]
    if local.duplicated(identity).any():
        duplicate = local.loc[local.duplicated(identity, keep=False), identity].head(10)
        raise ValueError(
            "Canonical interval identities are duplicated:\n"
            + duplicate.to_string(index=False)
        )

    invalid = local["end_sec"] <= local["start_sec"]
    if invalid.any():
        raise ValueError(
            "Canonical interval table contains nonpositive durations:\n"
            + local.loc[invalid, identity + ["start_sec", "end_sec"]]
            .head(10)
            .to_string(index=False)
        )

    overlap_rows = []
    for recording_id, group in local.groupby("logical_recording_id", sort=False):
        ordered = group.sort_values("start_sec")
        starts = ordered["start_sec"].to_numpy(float)
        ends = ordered["end_sec"].to_numpy(float)
        for index in range(1, len(ordered)):
            if starts[index] < ends[index - 1] - 1e-9:
                overlap_rows.append(
                    {
                        "logical_recording_id": recording_id,
                        "previous_end_sec": ends[index - 1],
                        "next_start_sec": starts[index],
                    }
                )
    if overlap_rows:
        raise ValueError(
            "Canonical intervals overlap:\n"
            + pd.DataFrame(overlap_rows).head(10).to_string(index=False)
        )

    return local


def canonical_interval_contract(
    decisions: pd.DataFrame,
    intervals: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build and validate the complete QADD frozen interval contract."""

    if "segmentation_analysis_eligible" not in decisions:
        raise ValueError("Frozen decisions lack segmentation_analysis_eligible")
    eligible_ids = set(
        decisions.loc[
            as_bool(decisions["segmentation_analysis_eligible"]),
            "logical_recording_id",
        ].astype(str)
    )
    if not eligible_ids:
        raise ValueError("No segmentation-analysis-eligible recordings were found")

    tables = {
        CANONICAL_PRIMARY_VIEW: canonical_interval_subset(
            intervals, view=CANONICAL_PRIMARY_VIEW
        ),
        CANONICAL_SPEECH_VIEW: canonical_interval_subset(
            intervals, view=CANONICAL_SPEECH_VIEW
        ),
        CANONICAL_PAUSE_VIEW: canonical_interval_subset(
            intervals, view=CANONICAL_PAUSE_VIEW
        ),
    }

    rows = []
    for view, table in tables.items():
        ids = set(table["logical_recording_id"].astype(str))
        required_for_every_recording = view != CANONICAL_PAUSE_VIEW
        rows.append(
            {
                "view": view,
                "profile": CANONICAL_PROFILE,
                "recording_count": len(ids),
                "interval_count": len(table),
                "eligible_recording_count": len(eligible_ids),
                "missing_eligible_recording_count": len(eligible_ids - ids),
                "extra_recording_count": len(ids - eligible_ids),
                "required_for_every_recording": required_for_every_recording,
                "contract_pass": (
                    ids == eligible_ids
                    if required_for_every_recording
                    else ids.issubset(eligible_ids)
                ),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary["contract_pass"].all():
        raise ValueError(
            "Canonical QADD interval contract failed:\n"
            + summary.to_string(index=False)
        )
    return tables, summary


def intervals_for_recording(
    interval_table: pd.DataFrame,
    logical_recording_id: str,
) -> tuple[list[TimeInterval], pd.DataFrame]:
    local = interval_table.loc[
        interval_table["logical_recording_id"].astype(str).eq(
            str(logical_recording_id)
        )
    ].sort_values(["start_sec", "end_sec", "interval_index"])
    intervals = [
        TimeInterval(float(row.start_sec), float(row.end_sec))
        for row in local.itertuples(index=False)
    ]
    return intervals, local.reset_index(drop=True)


def attach_interval_provenance(
    ledger: pd.DataFrame,
    *,
    region: str,
    canonical_intervals: pd.DataFrame,
    canonical_view: str,
    canonical_profile: str = CANONICAL_PROFILE,
) -> pd.DataFrame:
    """Map extractor-local interval indices to frozen interval identities."""

    output = ledger.copy()
    if output.empty:
        output["frozen_view"] = pd.Series(dtype="object")
        output["frozen_profile"] = pd.Series(dtype="object")
        output["frozen_interval_index"] = pd.Series(dtype="Int64")
        output["frozen_interval_start_sec"] = pd.Series(dtype="float64")
        output["frozen_interval_end_sec"] = pd.Series(dtype="float64")
        return output

    # Build mapping without relying on reset-index naming.
    mapping = canonical_intervals.sort_values(
        ["start_sec", "end_sec", "interval_index"]
    ).reset_index(drop=True)
    mapping = pd.DataFrame(
        {
            "local_interval_index": np.arange(len(mapping), dtype=int),
            "frozen_interval_index": mapping["interval_index"].astype(int).to_numpy(),
            "frozen_interval_start_sec": mapping["start_sec"].astype(float).to_numpy(),
            "frozen_interval_end_sec": mapping["end_sec"].astype(float).to_numpy(),
        }
    )

    output["interval_index"] = pd.to_numeric(
        output["interval_index"], errors="raise"
    ).astype(int)
    output = output.merge(
        mapping,
        left_on="interval_index",
        right_on="local_interval_index",
        how="left",
        validate="many_to_one",
    )
    if output["frozen_interval_index"].isna().any():
        raise ValueError(
            f"Could not map {region} ledger rows to frozen interval identities"
        )
    output["frozen_view"] = canonical_view
    output["frozen_profile"] = canonical_profile
    output["frozen_interval_index"] = output["frozen_interval_index"].astype(int)
    output = output.drop(columns=["local_interval_index"])
    return output


def colored_noise_window(
    sample_count: int,
    color: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a zero-mean unit-RMS colored-noise window."""

    values = rng.normal(size=int(sample_count))
    if color == "white":
        shaped = values
    else:
        spectrum = np.fft.rfft(values)
        frequencies = np.fft.rfftfreq(values.size)
        scale = np.ones_like(frequencies)
        nonzero = frequencies > 0
        exponent = 0.5 if color == "pink" else 1.0
        scale[nonzero] = frequencies[nonzero] ** (-exponent)
        scale[~nonzero] = 0.0
        shaped = np.fft.irfft(spectrum * scale, n=values.size)
    shaped = shaped - float(np.mean(shaped))
    rms = float(np.sqrt(np.mean(shaped * shaped)))
    return shaped / max(rms, np.finfo(float).tiny)


def hum_null_window_pool(
    *,
    parameters: QADDParameters = DEFAULT_PARAMETERS,
    pool_size: int = 4000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate independent colored-noise window score pairs.

    The score is amplitude invariant, so a single unit-RMS pool can be used
    for all valid acoustic hum windows.  White, pink, and brown windows are
    mixed deterministically.
    """

    rng = np.random.default_rng(
        parameters.random_seed + 101 if seed is None else int(seed)
    )
    window_count = int(pool_size)
    window_n = round(
        parameters.hum_window_ms
        * parameters.analysis_sample_rate_hz
        / 1000.0
    )
    colors = ("white", "pink", "brown")
    rows = []
    for index in range(window_count):
        color = colors[index % len(colors)]
        samples = colored_noise_window(window_n, color, rng)
        frequencies, psd = power_spectrum(
            samples, parameters.analysis_sample_rate_hz
        )
        score_50, support_50, evaluated_50 = hum_comb_score_from_psd(
            frequencies, psd, 50.0, parameters=parameters
        )
        score_60, support_60, evaluated_60 = hum_comb_score_from_psd(
            frequencies, psd, 60.0, parameters=parameters
        )
        rows.append(
            {
                "pool_index": index,
                "color": color,
                "hum_score_50_db": score_50,
                "hum_score_60_db": score_60,
                "hum_supported_harmonics_50": support_50,
                "hum_supported_harmonics_60": support_60,
                "hum_evaluated_harmonics_50": evaluated_50,
                "hum_evaluated_harmonics_60": evaluated_60,
            }
        )
    return pd.DataFrame(rows)


def hum_null_calibration_grid(
    pool: pd.DataFrame,
    *,
    support_counts: Sequence[int],
    iterations: int = 3000,
    seed: int = 20260803,
) -> pd.DataFrame:
    """Build a deterministic count-matched recording-level null grid."""

    required = {"hum_score_50_db", "hum_score_60_db"}
    if not required.issubset(pool):
        raise ValueError(f"Hum-null pool is missing {sorted(required-set(pool))}")
    score_50 = pd.to_numeric(pool["hum_score_50_db"], errors="raise").to_numpy(float)
    score_60 = pd.to_numeric(pool["hum_score_60_db"], errors="raise").to_numpy(float)
    if len(score_50) < 100:
        raise ValueError("Hum-null pool is too small")
    rng = np.random.default_rng(int(seed))
    rows = []
    for count in sorted({int(item) for item in support_counts if int(item) >= 2}):
        indices = rng.integers(0, len(pool), size=(int(iterations), count))
        med_50 = np.median(score_50[indices], axis=1)
        med_60 = np.median(score_60[indices], axis=1)
        maximum = np.maximum(med_50, med_60)
        rows.append(
            {
                "window_count": count,
                "simulation_iterations": int(iterations),
                "null_mean_db": float(np.mean(maximum)),
                "null_sd_db": float(np.std(maximum, ddof=1)),
                "null_p90_db": float(np.quantile(maximum, 0.90)),
                "null_p95_db": float(np.quantile(maximum, 0.95)),
                "null_p99_db": float(np.quantile(maximum, 0.99)),
            }
        )
    return pd.DataFrame(rows)


def observed_hum_support_grid(support_counts: Sequence[int]) -> list[int]:
    """Return every observed valid-window count eligible for hum calibration.

    The recording-level null must be matched to the exact number of valid hum
    windows whenever that count occurs in the cohort.  Counts below two are
    intentionally excluded because the hum descriptor is unavailable there.
    """

    values = sorted(
        {
            int(value)
            for value in support_counts
            if pd.notna(value) and int(value) >= 2
        }
    )
    return values



def conservative_nonincreasing_thresholds(values: Sequence[float]) -> np.ndarray:
    """Enforce a non-increasing threshold curve without lowering any value.

    Monte-Carlo quantiles can fluctuate slightly with support count.  The
    reverse cumulative maximum yields the smallest non-increasing sequence
    that is everywhere at least as conservative as the simulated values.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Threshold values must be one-dimensional")
    if array.size == 0:
        return array.copy()
    if not np.isfinite(array).all():
        raise ValueError("Threshold values must be finite")
    return np.maximum.accumulate(array[::-1])[::-1]

def select_hum_null_reference(
    valid_window_count: int,
    calibration_grid: pd.DataFrame,
) -> tuple[int | None, float]:
    """Use the largest calibrated support count not exceeding observed support.

    A smaller support count has a wider recording-level null and is therefore
    conservative when the exact count is unavailable.
    """

    count = int(valid_window_count)
    if count < 2 or calibration_grid.empty:
        return None, np.nan
    eligible = calibration_grid.loc[
        pd.to_numeric(calibration_grid["window_count"], errors="coerce") <= count
    ]
    if eligible.empty:
        selected = calibration_grid.sort_values("window_count").iloc[0]
    else:
        selected = eligible.sort_values("window_count").iloc[-1]
    return int(selected["window_count"]), float(selected["null_p95_db"])


def empirical_feature_summary(
    recording_table: pd.DataFrame,
    features: Sequence[str] = ANALYSIS_FEATURES,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = pd.to_numeric(recording_table[feature], errors="coerce")
        finite = values[np.isfinite(values)]
        status_column = f"{feature}_status"
        rows.append(
            {
                "feature": feature,
                "recording_count": len(recording_table),
                "available_count": int(len(finite)),
                "available_fraction": float(len(finite) / max(1, len(recording_table))),
                "median": float(finite.median()) if len(finite) else np.nan,
                "q25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                "q75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                "iqr": (
                    float(finite.quantile(0.75) - finite.quantile(0.25))
                    if len(finite)
                    else np.nan
                ),
                "p01": float(finite.quantile(0.01)) if len(finite) else np.nan,
                "p05": float(finite.quantile(0.05)) if len(finite) else np.nan,
                "p95": float(finite.quantile(0.95)) if len(finite) else np.nan,
                "p99": float(finite.quantile(0.99)) if len(finite) else np.nan,
                "minimum": float(finite.min()) if len(finite) else np.nan,
                "maximum": float(finite.max()) if len(finite) else np.nan,
                "status_counts_json": (
                    json.dumps(
                        recording_table[status_column]
                        .value_counts(dropna=False)
                        .to_dict(),
                        sort_keys=True,
                    )
                    if status_column in recording_table
                    else "{}"
                ),
            }
        )
    return pd.DataFrame(rows)


def icc1_balanced_first_two(
    frame: pd.DataFrame,
    *,
    subject_column: str,
    date_column: str,
    feature: str,
) -> dict:
    """Method-of-moments ICC(1,1) using the first two finite repeats."""

    local = frame[[subject_column, date_column, feature]].copy()
    local[feature] = pd.to_numeric(local[feature], errors="coerce")
    local[date_column] = pd.to_datetime(local[date_column], errors="coerce")
    local = local.dropna(subset=[subject_column, feature])
    local = local.sort_values([subject_column, date_column])
    pairs = local.groupby(subject_column, sort=False).head(2)
    counts = pairs.groupby(subject_column)[feature].count()
    eligible_subjects = counts[counts >= 2].index
    pairs = pairs.loc[pairs[subject_column].isin(eligible_subjects)].copy()
    if len(eligible_subjects) < 3:
        return {
            "icc1": np.nan,
            "subject_count": int(len(eligible_subjects)),
            "recording_count": int(len(pairs)),
        }
    pairs["repeat_index"] = pairs.groupby(subject_column).cumcount()
    wide = pairs.pivot(
        index=subject_column, columns="repeat_index", values=feature
    )[[0, 1]].dropna()
    matrix = wide.to_numpy(float)
    n, k = matrix.shape
    subject_means = matrix.mean(axis=1)
    grand = matrix.mean()
    ss_between = k * np.sum((subject_means - grand) ** 2)
    ss_within = np.sum((matrix - subject_means[:, None]) ** 2)
    ms_between = ss_between / max(1, n - 1)
    ms_within = ss_within / max(1, n * (k - 1))
    denominator = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator > 0 else np.nan
    return {
        "icc1": float(icc) if np.isfinite(icc) else np.nan,
        "subject_count": int(n),
        "recording_count": int(n * k),
    }


def repeated_recording_persistence(
    recording_table: pd.DataFrame,
    *,
    subject_column: str,
    date_column: str,
    features: Sequence[str] = ANALYSIS_FEATURES,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        local = recording_table[
            [subject_column, date_column, "logical_recording_id", feature]
        ].copy()
        local[feature] = pd.to_numeric(local[feature], errors="coerce")
        local[date_column] = pd.to_datetime(local[date_column], errors="coerce")
        local = local.dropna(subset=[subject_column, feature])
        local = local.sort_values([subject_column, date_column, "logical_recording_id"])
        first_two = local.groupby(subject_column, sort=False).head(2)
        counts = first_two.groupby(subject_column)[feature].count()
        eligible = counts[counts >= 2].index
        first_two = first_two.loc[first_two[subject_column].isin(eligible)].copy()
        first_two["repeat_index"] = first_two.groupby(subject_column).cumcount()
        wide = first_two.pivot(
            index=subject_column, columns="repeat_index", values=feature
        )
        if 0 in wide and 1 in wide:
            pair = wide[[0, 1]].dropna()
        else:
            pair = pd.DataFrame(columns=[0, 1])
        if len(pair) >= 3:
            spearman = stats.spearmanr(pair[0], pair[1]).statistic
            pearson = stats.pearsonr(pair[0], pair[1]).statistic
            differences = (pair[1] - pair[0]).abs()
        else:
            spearman = pearson = np.nan
            differences = pd.Series(dtype=float)
        icc = icc1_balanced_first_two(
            recording_table,
            subject_column=subject_column,
            date_column=date_column,
            feature=feature,
        )
        rows.append(
            {
                "feature": feature,
                "paired_subject_count": int(len(pair)),
                "first_second_spearman": (
                    float(spearman) if np.isfinite(spearman) else np.nan
                ),
                "first_second_pearson": (
                    float(pearson) if np.isfinite(pearson) else np.nan
                ),
                "median_absolute_difference": (
                    float(differences.median()) if len(differences) else np.nan
                ),
                "p90_absolute_difference": (
                    float(differences.quantile(0.90))
                    if len(differences)
                    else np.nan
                ),
                "icc1_first_two": icc["icc1"],
                "icc_subject_count": icc["subject_count"],
            }
        )
    return pd.DataFrame(rows)


def participant_balanced_resampling(
    recording_table: pd.DataFrame,
    *,
    subject_column: str,
    features: Sequence[str] = ANALYSIS_FEATURES,
    iterations: int = 1000,
    seed: int = 20260803,
) -> pd.DataFrame:
    """Sample one available recording per participant and summarize medians."""

    if subject_column not in recording_table:
        raise ValueError(f"Missing participant column: {subject_column}")
    rng = np.random.default_rng(int(seed))
    grouped = {
        str(subject): group.copy()
        for subject, group in recording_table.groupby(subject_column, dropna=True)
    }
    if not grouped:
        return pd.DataFrame()
    rows = []
    for iteration in range(int(iterations)):
        selected = []
        for group in grouped.values():
            index = int(rng.integers(0, len(group)))
            selected.append(group.iloc[index])
        sample = pd.DataFrame(selected)
        for feature in features:
            values = pd.to_numeric(sample[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "iteration": iteration,
                    "feature": feature,
                    "participant_count": len(sample),
                    "available_participant_count": len(finite),
                    "median": float(finite.median()) if len(finite) else np.nan,
                    "q25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                    "q75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def participant_balanced_summary(resampling: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, local in resampling.groupby("feature", sort=False):
        values = pd.to_numeric(local["median"], errors="coerce").dropna()
        rows.append(
            {
                "feature": feature,
                "iterations": len(local),
                "balanced_median_of_medians": (
                    float(values.median()) if len(values) else np.nan
                ),
                "balanced_median_ci025": (
                    float(values.quantile(0.025)) if len(values) else np.nan
                ),
                "balanced_median_ci975": (
                    float(values.quantile(0.975)) if len(values) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def measurement_long_frame(
    recording_table: pd.DataFrame,
    *,
    features: Sequence[str] = ANALYSIS_FEATURES,
) -> pd.DataFrame:
    support_fields = {
        "qadd_pause_ac_level_dbfs_median": "qadd_pause_level_support_tier",
        "qadd_pause_level_iqr_db": "qadd_pause_dispersion_support_tier",
        "qadd_speech_pause_level_contrast_db": "qadd_speech_pause_contrast_support_tier",
        "qadd_pause_spectral_flatness": "qadd_flatness_support_tier",
        "qadd_mains_hum_comb_score_db": "qadd_hum_support_tier",
    }
    rows = []
    identity_fields = [
        field
        for field in [
            "logical_recording_id",
            "SubjectID",
            "diagnosis_analysis",
            "recording_date_analysis",
        ]
        if field in recording_table
    ]
    for row in recording_table.itertuples(index=False):
        payload = row._asdict()
        identity = {field: payload.get(field) for field in identity_fields}
        for feature in features:
            status_field = f"{feature}_status"
            raw_field = f"{feature}_raw_estimate"
            support_field = support_fields[feature]
            value = payload.get(feature, np.nan)
            rows.append(
                {
                    **identity,
                    "family": "QADD",
                    "measurement_version": payload.get(
                        "qadd_measurement_version", "qadd-v4.2.0-candidate"
                    ),
                    "feature": feature,
                    "value": value,
                    "raw_estimate": payload.get(raw_field, np.nan),
                    "available": bool(np.isfinite(value)),
                    "measurement_status": payload.get(status_field),
                    "support_tier": payload.get(support_field),
                    "standalone_gate_allowed": False,
                    "family_scalar_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def model_interface_frame(
    recording_table: pd.DataFrame,
    *,
    features: Sequence[str] = ANALYSIS_FEATURES,
) -> pd.DataFrame:
    """Wide, non-imputed ML handoff with value/status/support companions."""

    base_columns = [
        column
        for column in [
            "logical_recording_id",
            "SubjectID",
            "diagnosis_analysis",
            "recording_date_analysis",
            "qadd_measurement_version",
            "qadd_signal_view",
            "qadd_family_status",
        ]
        if column in recording_table
    ]
    output = recording_table[base_columns].copy()
    support_fields = {
        "qadd_pause_ac_level_dbfs_median": "qadd_pause_level_support_tier",
        "qadd_pause_level_iqr_db": "qadd_pause_dispersion_support_tier",
        "qadd_speech_pause_level_contrast_db": "qadd_speech_pause_contrast_support_tier",
        "qadd_pause_spectral_flatness": "qadd_flatness_support_tier",
        "qadd_mains_hum_comb_score_db": "qadd_hum_support_tier",
    }
    for feature in features:
        output[feature] = pd.to_numeric(recording_table[feature], errors="coerce")
        output[f"{feature}__available"] = output[feature].notna()
        output[f"{feature}__status"] = recording_table[f"{feature}_status"].astype(str)
        output[f"{feature}__support_tier"] = recording_table[
            support_fields[feature]
        ].astype(str)
    companion_fields = [
        "qadd_pause_effective_nonfloor_support_sec",
        "qadd_speech_effective_nonfloor_support_sec",
        "qadd_pause_interval_count_nonfloor",
        "qadd_pause_at_floor_frame_fraction",
        "qadd_flatness_valid_window_count",
        "qadd_hum_valid_window_count",
        "qadd_mains_hum_null_p95_db",
        "qadd_mains_hum_excess_over_null_p95_db",
        "qadd_mains_hum_joint_evidence_above_null",
        "qadd_mains_hum_null_reference_window_count",
        "qadd_mains_hum_null_calibration_status",
    ]
    for field in companion_fields:
        if field in recording_table:
            output[field] = recording_table[field]
    output["qadd_standalone_reject_allowed"] = False
    output["qadd_family_scalar_available"] = False
    output["qadd_decision_threshold_status"] = "not_calibrated"
    return output


def hash_inventory(root: Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)
