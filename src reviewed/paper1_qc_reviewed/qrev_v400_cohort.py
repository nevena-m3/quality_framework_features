"""Cohort orchestration and empirical-validation helpers for QREV v4.0.0.

This module does not redefine the four QREV estimators. It enforces the frozen
cohort/interval contract and provides support-policy, censoring, precision,
reliability, redundancy, model-interface, and immutable-inventory utilities for
the reviewed cohort notebook.

The three boundary measurements remain conditional. Missing, right-censored,
and zero-valued observations are distinct. No family scalar or standalone
recording-rejection rule is constructed here.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Iterable, Mapping, Sequence
import json

import numpy as np
import pandas as pd
from scipy import stats

from paper1_qc_reviewed.qrev_v400 import (
    ANALYSIS_FEATURES,
    CONDITIONAL_BOUNDARY_FEATURES,
    DEFAULT_PARAMETERS,
    QREVParameters,
    SpeechInterval,
)

COHORT_ORCHESTRATION_VERSION = "qrev-v4.0.0-cohort-orchestration-v1"
CANONICAL_PROFILE = "primary"
CANONICAL_PRIMARY_VIEW = "primary_speech"
CANONICAL_STRICT_VIEW = "strict_speech"
SUPPORT_POLICIES = (2, 3, 4)

BOUNDARY_FEATURE_SPECS = {
    "qrev_tail_excess_100ms_db": {
        "flag": "tail_eligible",
        "ledger_value": "tail_excess_100ms_db",
        "recording_raw": "qrev_tail_excess_100ms_db_raw_estimate",
        "count": "qrev_tail_valid_boundary_count",
        "support_sec": "qrev_tail_valid_pause_support_sec",
    },
    "qrev_tail_persistence_median_sec": {
        "flag": "persistence_eligible",
        "ledger_value": "tail_persistence_sec",
        "recording_raw": "qrev_tail_persistence_median_sec_raw_estimate",
        "count": "qrev_persistence_valid_boundary_count",
        "support_sec": "qrev_persistence_valid_pause_support_sec",
    },
    "qrev_downward_decay_rate_db_per_sec": {
        "flag": "decay_eligible",
        "ledger_value": "downward_decay_rate_db_per_sec",
        "recording_raw": "qrev_downward_decay_rate_db_per_sec_raw_estimate",
        "count": "qrev_decay_valid_boundary_count",
        "support_sec": "qrev_decay_valid_pause_support_sec",
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_waveform_from_audio_views(views: object) -> np.ndarray:
    """Return the canonical 16-kHz analysis waveform from ``AudioViews``.

    ``paper1_qc.media.AudioViews`` exposes this signal as ``analysis_16k``.
    Keeping the interface check here prevents cohort, robustness, bandwidth,
    and gallery paths from silently drifting to a noncanonical audio view.
    """
    if not hasattr(views, "analysis_16k"):
        available = sorted(
            name for name in dir(views)
            if not name.startswith("_")
        )
        raise AttributeError(
            "AudioViews does not expose the required canonical field "
            f"'analysis_16k'; available public attributes: {available}"
        )

    waveform = np.asarray(getattr(views, "analysis_16k"), dtype=np.float64)
    if waveform.ndim != 1:
        raise ValueError(
            "AudioViews.analysis_16k must be one-dimensional; "
            f"observed shape={waveform.shape}"
        )
    if waveform.size == 0:
        raise ValueError("AudioViews.analysis_16k is empty")
    if not np.isfinite(waveform).all():
        raise ValueError("AudioViews.analysis_16k contains NaN or Inf")
    return waveform


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
        duplicate = local.loc[local.duplicated(identity, keep=False), identity].head(10)
        raise ValueError(
            "Canonical interval identities are duplicated:\n"
            + duplicate.to_string(index=False)
        )
    if local["frozen_interval_id"].duplicated().any():
        raise ValueError("Deterministic frozen interval IDs are duplicated")

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
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
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

    primary = canonical_interval_subset(intervals, view=CANONICAL_PRIMARY_VIEW)
    strict = canonical_interval_subset(intervals, view=CANONICAL_STRICT_VIEW)
    tables = {
        CANONICAL_PRIMARY_VIEW: primary,
        CANONICAL_STRICT_VIEW: strict,
    }

    summary_rows = []
    for view, table in tables.items():
        ids = set(table["logical_recording_id"].astype(str))
        summary_rows.append(
            {
                "view": view,
                "profile": CANONICAL_PROFILE,
                "recording_count": len(ids),
                "interval_count": len(table),
                "eligible_recording_count": len(eligible_ids),
                "missing_eligible_recording_count": len(eligible_ids - ids),
                "extra_recording_count": len(ids - eligible_ids),
                "contract_pass": not (eligible_ids - ids) and not (ids - eligible_ids),
            }
        )

    pair_keys = ["logical_recording_id", "interval_index"]
    left = primary[
        pair_keys + ["start_sec", "end_sec", "frozen_interval_id"]
    ].rename(
        columns={
            "start_sec": "primary_start_sec",
            "end_sec": "primary_end_sec",
            "frozen_interval_id": "primary_interval_id",
        }
    )
    right = strict[
        pair_keys + ["start_sec", "end_sec", "frozen_interval_id"]
    ].rename(
        columns={
            "start_sec": "strict_start_sec",
            "end_sec": "strict_end_sec",
            "frozen_interval_id": "strict_interval_id",
        }
    )
    pair_audit = left.merge(right, on=pair_keys, how="outer", indicator=True)
    pair_audit["pair_complete"] = pair_audit["_merge"].eq("both")
    pair_audit["strict_inside_primary"] = (
        pair_audit["pair_complete"]
        & pair_audit["strict_start_sec"].ge(pair_audit["primary_start_sec"] - 1e-9)
        & pair_audit["strict_end_sec"].le(pair_audit["primary_end_sec"] + 1e-9)
    )
    pair_audit["start_erosion_sec"] = (
        pair_audit["strict_start_sec"] - pair_audit["primary_start_sec"]
    )
    pair_audit["end_erosion_sec"] = (
        pair_audit["primary_end_sec"] - pair_audit["strict_end_sec"]
    )
    pair_pass = bool(
        len(pair_audit)
        and pair_audit["pair_complete"].all()
        and pair_audit["strict_inside_primary"].all()
    )
    summary_rows.append(
        {
            "view": "primary_vs_strict_pairing",
            "profile": CANONICAL_PROFILE,
            "recording_count": pair_audit["logical_recording_id"].nunique(),
            "interval_count": len(pair_audit),
            "eligible_recording_count": len(eligible_ids),
            "missing_eligible_recording_count": 0,
            "extra_recording_count": 0,
            "contract_pass": pair_pass,
        }
    )

    summary = pd.DataFrame(summary_rows)
    if not summary["contract_pass"].all():
        raise ValueError(
            "QREV canonical interval contract failed:\n"
            + summary.to_string(index=False)
        )
    return tables, summary, pair_audit.drop(columns="_merge")


def intervals_for_recording(
    canonical_intervals: pd.DataFrame,
    recording_id: str,
) -> tuple[list[SpeechInterval], pd.DataFrame]:
    local = canonical_intervals.loc[
        canonical_intervals["logical_recording_id"].astype(str).eq(str(recording_id))
    ].sort_values(["start_sec", "end_sec", "interval_index"]).copy()
    intervals = [
        SpeechInterval(
            start_sec=float(row.start_sec),
            end_sec=float(row.end_sec),
            interval_id=str(row.frozen_interval_id),
            interval_index=int(row.interval_index),
            view=str(row.view),
            profile=str(row.profile),
        )
        for row in local.itertuples(index=False)
    ]
    return intervals, local


def shift_primary_offsets(
    intervals: Sequence[SpeechInterval],
    shift_ms: float,
    *,
    minimum_speech_sec: float = 0.05,
    separation_sec: float = 0.03,
) -> list[SpeechInterval]:
    """Perturb natural speech offsets while preserving starts and identities."""
    delta = float(shift_ms) / 1000.0
    ordered = list(intervals)
    output: list[SpeechInterval] = []
    for index, item in enumerate(ordered):
        if index < len(ordered) - 1:
            ceiling = ordered[index + 1].start_sec - float(separation_sec)
            end = min(item.end_sec + delta, ceiling)
            end = max(item.start_sec + float(minimum_speech_sec), end)
        else:
            # The final speech offset does not define an internal pause boundary.
            # Keep it unchanged so positive perturbations cannot exceed media duration.
            end = item.end_sec
        output.append(
            SpeechInterval(
                start_sec=item.start_sec,
                end_sec=end,
                interval_id=item.interval_id,
                interval_index=item.interval_index,
                view=item.view,
                profile=item.profile,
            )
        )
    return output


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


def policy_values(
    recording_table: pd.DataFrame,
    *,
    minimum_boundary_count: int,
) -> pd.DataFrame:
    output = recording_table[["logical_recording_id"]].copy()
    output["minimum_boundary_count"] = int(minimum_boundary_count)
    for feature, spec in BOUNDARY_FEATURE_SPECS.items():
        count = pd.to_numeric(recording_table[spec["count"]], errors="coerce").fillna(0)
        raw = pd.to_numeric(recording_table[spec["recording_raw"]], errors="coerce")
        available = count.ge(int(minimum_boundary_count)) & raw.notna()
        output[feature] = raw.where(available)
        output[f"{feature}__available"] = available
        output[f"{feature}__boundary_count"] = count.astype(int)
    output["qrev_srmr_norm"] = pd.to_numeric(
        recording_table["qrev_srmr_norm"], errors="coerce"
    )
    output["qrev_srmr_norm__available"] = output["qrev_srmr_norm"].notna()
    return output


def support_policy_availability(
    recording_table: pd.DataFrame,
    policies: Sequence[int] = SUPPORT_POLICIES,
) -> pd.DataFrame:
    rows = []
    total = len(recording_table)
    for policy in policies:
        values = policy_values(recording_table, minimum_boundary_count=int(policy))
        for feature in CONDITIONAL_BOUNDARY_FEATURES:
            available = values[f"{feature}__available"].astype(bool)
            rows.append(
                {
                    "feature": feature,
                    "minimum_boundary_count": int(policy),
                    "recordings": total,
                    "available_n": int(available.sum()),
                    "availability_fraction": float(available.mean()) if total else np.nan,
                }
            )
    srmr_available = pd.to_numeric(
        recording_table["qrev_srmr_norm"], errors="coerce"
    ).notna()
    rows.append(
        {
            "feature": "qrev_srmr_norm",
            "minimum_boundary_count": 0,
            "recordings": total,
            "available_n": int(srmr_available.sum()),
            "availability_fraction": float(srmr_available.mean()) if total else np.nan,
        }
    )
    return pd.DataFrame(rows)


def delete_one_boundary_grid(
    boundary_ledger: pd.DataFrame,
    *,
    policies: Sequence[int] = SUPPORT_POLICIES,
) -> pd.DataFrame:
    rows = []
    if boundary_ledger.empty:
        return pd.DataFrame()
    for recording_id, local in boundary_ledger.groupby(
        "logical_recording_id", sort=True
    ):
        for feature, spec in BOUNDARY_FEATURE_SPECS.items():
            valid = pd.to_numeric(
                local.loc[as_bool(local[spec["flag"]]), spec["ledger_value"]],
                errors="coerce",
            ).dropna().to_numpy(float)
            if len(valid) < 2:
                continue
            full = float(np.median(valid))
            for omitted_index in range(len(valid)):
                reduced_values = np.delete(valid, omitted_index)
                reduced = (
                    float(np.median(reduced_values))
                    if len(reduced_values)
                    else np.nan
                )
                for policy in policies:
                    full_available = len(valid) >= int(policy)
                    reduced_available = len(reduced_values) >= int(policy)
                    rows.append(
                        {
                            "logical_recording_id": str(recording_id),
                            "feature": feature,
                            "minimum_boundary_count": int(policy),
                            "boundary_count": len(valid),
                            "omitted_index": int(omitted_index),
                            "full_available": bool(full_available),
                            "delete_one_available": bool(reduced_available),
                            "availability_retained": bool(
                                not full_available or reduced_available
                            ),
                            "full_estimate": full if full_available else np.nan,
                            "delete_one_estimate": reduced if reduced_available else np.nan,
                            "absolute_delta": (
                                abs(reduced - full)
                                if full_available
                                and reduced_available
                                and np.isfinite(reduced)
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_delete_one(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    rows = []
    for keys, local in grid.groupby(
        ["feature", "minimum_boundary_count"], sort=True
    ):
        finite = pd.to_numeric(local["absolute_delta"], errors="coerce").dropna()
        full_available = local["full_available"].astype(bool)
        retained = local.loc[full_available, "availability_retained"].astype(bool)
        rows.append(
            {
                "feature": keys[0],
                "minimum_boundary_count": int(keys[1]),
                "recordings": local["logical_recording_id"].nunique(),
                "comparisons": len(local),
                "paired_finite_comparisons": len(finite),
                "availability_retention_fraction": (
                    float(retained.mean()) if len(retained) else np.nan
                ),
                "median_absolute_delta": (
                    float(finite.median()) if len(finite) else np.nan
                ),
                "p95_absolute_delta": (
                    float(finite.quantile(0.95)) if len(finite) else np.nan
                ),
                "maximum_absolute_delta": (
                    float(finite.max()) if len(finite) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_median_precision(
    boundary_ledger: pd.DataFrame,
    *,
    iterations: int = 400,
    seed: int = DEFAULT_PARAMETERS.random_seed,
) -> pd.DataFrame:
    rows = []
    if boundary_ledger.empty:
        return pd.DataFrame()
    for recording_id, local in boundary_ledger.groupby(
        "logical_recording_id", sort=True
    ):
        for feature_index, (feature, spec) in enumerate(BOUNDARY_FEATURE_SPECS.items()):
            values = pd.to_numeric(
                local.loc[as_bool(local[spec["flag"]]), spec["ledger_value"]],
                errors="coerce",
            ).dropna().to_numpy(float)
            if len(values) < 2:
                continue
            stable_seed = int(
                stable_hash_order(
                    f"{recording_id}|{feature}|{seed}",
                    seed=seed + feature_index,
                )[:16],
                16,
            ) % (2**32 - 1)
            rng = np.random.default_rng(stable_seed)
            sampled = rng.choice(
                values,
                size=(int(iterations), len(values)),
                replace=True,
            )
            estimates = np.median(sampled, axis=1)
            rows.append(
                {
                    "logical_recording_id": str(recording_id),
                    "feature": feature,
                    "boundary_count": len(values),
                    "bootstrap_iterations": int(iterations),
                    "bootstrap_median": float(np.median(estimates)),
                    "bootstrap_sd": float(np.std(estimates, ddof=1)),
                    "bootstrap_ci95_low": float(np.quantile(estimates, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(estimates, 0.975)),
                    "bootstrap_ci95_width": float(
                        np.quantile(estimates, 0.975)
                        - np.quantile(estimates, 0.025)
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
    local = frame[[subject_column, date_column, feature]].copy()
    local[feature] = pd.to_numeric(local[feature], errors="coerce")
    local[date_column] = pd.to_datetime(local[date_column], errors="coerce")
    local = local.dropna(subset=[subject_column, date_column, feature])
    local = local.sort_values([subject_column, date_column])
    local = local.groupby(subject_column, as_index=False).head(2)
    counts = local.groupby(subject_column).size()
    subjects = counts.loc[counts.eq(2)].index
    local = local.loc[local[subject_column].isin(subjects)]
    if len(subjects) < 2:
        return {"subject_count": len(subjects), "icc1": np.nan}
    pivot = local.assign(
        repeat_index=local.groupby(subject_column).cumcount()
    ).pivot(index=subject_column, columns="repeat_index", values=feature)
    values = pivot.to_numpy(float)
    n, k = values.shape
    grand = values.mean()
    subject_means = values.mean(axis=1)
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
    censored_column: str = "qrev_persistence_recording_median_censored",
) -> pd.DataFrame:
    rows = []
    for feature in features:
        local = frame[
            [subject_column, date_column, "logical_recording_id", feature]
            + ([censored_column] if feature == "qrev_tail_persistence_median_sec" and censored_column in frame else [])
        ].copy()
        local[date_column] = pd.to_datetime(local[date_column], errors="coerce")
        local[feature] = pd.to_numeric(local[feature], errors="coerce")
        local = local.sort_values([subject_column, date_column, "logical_recording_id"])
        local["repeat_index"] = local.groupby(subject_column).cumcount()
        pair = local.loc[local["repeat_index"].isin([0, 1])].pivot(
            index=subject_column,
            columns="repeat_index",
            values=feature,
        )
        pair = pair.rename(columns={0: "first", 1: "second"})
        pair_complete = pair.dropna()
        rho = (
            float(stats.spearmanr(pair_complete["first"], pair_complete["second"]).statistic)
            if len(pair_complete) >= 3
            and pair_complete["first"].nunique() > 1
            and pair_complete["second"].nunique() > 1
            else np.nan
        )
        diff = (pair_complete["second"] - pair_complete["first"]).abs()
        icc = icc1_balanced_first_two(
            frame,
            subject_column=subject_column,
            date_column=date_column,
            feature=feature,
        )
        row = {
            "feature": feature,
            "paired_subject_count": len(pair_complete),
            "first_second_spearman": rho,
            "icc1_first_two": icc["icc1"],
            "icc_subject_count": icc["subject_count"],
            "median_absolute_difference": (
                float(diff.median()) if len(diff) else np.nan
            ),
            "p90_absolute_difference": (
                float(diff.quantile(0.90)) if len(diff) else np.nan
            ),
        }
        if feature == "qrev_tail_persistence_median_sec" and censored_column in frame:
            censor = local.loc[local["repeat_index"].isin([0, 1])].pivot(
                index=subject_column,
                columns="repeat_index",
                values=censored_column,
            )
            censor = censor.rename(columns={0: "first_censored", 1: "second_censored"})
            joined = pair.join(censor, how="left")
            exact = joined.loc[
                joined[["first", "second"]].notna().all(axis=1)
                & ~joined["first_censored"].fillna(False).astype(bool)
                & ~joined["second_censored"].fillna(False).astype(bool)
            ]
            row["paired_uncensored_subject_count"] = len(exact)
            row["uncensored_first_second_spearman"] = (
                float(stats.spearmanr(exact["first"], exact["second"]).statistic)
                if len(exact) >= 3
                and exact["first"].nunique() > 1
                and exact["second"].nunique() > 1
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def participant_balanced_resampling(
    frame: pd.DataFrame,
    *,
    subject_column: str,
    features: Sequence[str] = ANALYSIS_FEATURES,
    iterations: int = 500,
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
            median_availability_fraction=("availability_fraction", "median"),
            p025_availability_fraction=(
                "availability_fraction",
                lambda x: x.quantile(0.025),
            ),
            p975_availability_fraction=(
                "availability_fraction",
                lambda x: x.quantile(0.975),
            ),
        )
    )


def empirical_feature_summary(
    recording_table: pd.DataFrame,
    features: Sequence[str] = ANALYSIS_FEATURES,
) -> pd.DataFrame:
    rows = []
    total = len(recording_table)
    for feature in features:
        values = pd.to_numeric(recording_table[feature], errors="coerce")
        finite = values.loc[np.isfinite(values)]
        rows.append(
            {
                "feature": feature,
                "recordings": total,
                "available_n": len(finite),
                "availability_fraction": len(finite) / max(1, total),
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


def model_interface_frame(recording_table: pd.DataFrame) -> pd.DataFrame:
    identity_columns = [
        column
        for column in [
            "logical_recording_id",
            "SubjectID",
            "recording_date_analysis",
            "qrev_measurement_version",
            "qrev_signal_view",
            "qrev_boundary_source_view",
            "qrev_boundary_source_profile",
            "qrev_srmr_variant",
            "qrev_srmr_upstream_commit",
        ]
        if column in recording_table
    ]
    output = recording_table[identity_columns].copy()

    for feature in ANALYSIS_FEATURES:
        output[feature] = pd.to_numeric(recording_table[feature], errors="coerce")
        output[f"{feature}__available"] = output[feature].notna()
        status_column = f"{feature}_status"
        output[f"{feature}__status"] = (
            recording_table[status_column].astype(str)
            if status_column in recording_table
            else np.where(output[f"{feature}__available"], "measured", "unavailable")
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
            else np.where(output[f"{feature}__available"], "available", "unavailable")
        )

    for column in [
        "qrev_internal_boundary_count",
        "qrev_tail_valid_boundary_count",
        "qrev_tail_valid_pause_support_sec",
        "qrev_persistence_valid_boundary_count",
        "qrev_persistence_valid_pause_support_sec",
        "qrev_persistence_observed_duration_support_sec",
        "qrev_persistence_right_censored_fraction",
        "qrev_persistence_recording_median_censored",
        "qrev_decay_valid_boundary_count",
        "qrev_decay_valid_pause_support_sec",
        "qrev_srmr_primary_task_span_sec",
        "qrev_srmr_strict_speech_support_sec",
        "qrev_srmr_estimated_working_set_mb",
        "qrev_family_status",
    ]:
        if column in recording_table:
            output[column] = recording_table[column]

    output["qrev_family_scalar_available"] = False
    output["qrev_standalone_reject_allowed"] = False
    output["qrev_decision_threshold_status"] = "not_calibrated"
    return output


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
