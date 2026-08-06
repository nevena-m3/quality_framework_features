from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FamilyRelease:
    order: str
    code: str
    family: str
    version: str
    table_name: str
    registry_name: str


FAMILIES = (
    FamilyRelease(
        "01",
        "QGAIN",
        "gain_dynamics",
        "qgain-v4.1.0",
        "qgain_v410_analysis_features.csv",
        "qgain_v410_feature_registry.csv",
    ),
    FamilyRelease(
        "02",
        "QADD",
        "additive_interference",
        "qadd-v4.2.0",
        "qadd_v420_analysis_features.csv",
        "qadd_v420_feature_registry.csv",
    ),
    FamilyRelease(
        "03",
        "QREV",
        "reverberation_tail",
        "qrev-v4.0.0",
        "qrev_v400_analysis_features.csv",
        "qrev_v400_feature_registry.csv",
    ),
    FamilyRelease(
        "04",
        "QCHAN",
        "channel_device",
        "qchan-v4.0.0",
        "qchan_v400_analysis_features.csv",
        "qchan_v400_feature_registry.csv",
    ),
    FamilyRelease(
        "05",
        "QDIST",
        "nonlinear_distortion",
        "qdist-v4.1.0",
        "qdist_v410_analysis_features.csv",
        "qdist_v410_feature_registry.csv",
    ),
)


# This is a normalized view of the accepted family registries. It does not alter an
# estimator or scientific decision; it gives downstream code one stable schema.
FEATURE_SPECS = (
    # QGAIN
    (
        "qgain_typical_speech_level_dbfs",
        "QGAIN",
        "gain_dynamics",
        "contextual",
        "dBFS",
        "contextual",
        False,
    ),
    (
        "qgain_within_segment_iqr_db",
        "QGAIN",
        "gain_dynamics",
        "primary_mixed",
        "dB",
        "higher",
        True,
    ),
    (
        "qgain_between_segment_mad_db",
        "QGAIN",
        "gain_dynamics",
        "secondary_mixed",
        "dB",
        "higher",
        True,
    ),
    (
        "qgain_abs_drift_db_per_min",
        "QGAIN",
        "gain_dynamics",
        "exploratory",
        "dB/min",
        "higher",
        False,
    ),
    # QADD
    (
        "qadd_pause_ac_level_dbfs_median",
        "QADD",
        "additive_interference",
        "primary_contextual",
        "dBFS",
        "higher",
        True,
    ),
    (
        "qadd_pause_level_iqr_db",
        "QADD",
        "additive_interference",
        "secondary_conditional",
        "dB",
        "higher",
        True,
    ),
    (
        "qadd_speech_pause_level_contrast_db",
        "QADD",
        "additive_interference",
        "secondary_mixed",
        "dB",
        "lower",
        True,
    ),
    (
        "qadd_pause_spectral_flatness",
        "QADD",
        "additive_interference",
        "secondary_nonordinal",
        "ratio",
        "contextual",
        True,
    ),
    (
        "qadd_mains_hum_comb_score_db",
        "QADD",
        "additive_interference",
        "targeted_conditional",
        "dB",
        "higher",
        True,
    ),
    # QREV
    (
        "qrev_tail_excess_100ms_db",
        "QREV",
        "reverberation_tail",
        "primary_conditional",
        "dB",
        "higher",
        True,
    ),
    (
        "qrev_tail_persistence_median_sec",
        "QREV",
        "reverberation_tail",
        "secondary_conditional",
        "s",
        "higher",
        True,
    ),
    (
        "qrev_downward_decay_rate_db_per_sec",
        "QREV",
        "reverberation_tail",
        "exploratory_conditional",
        "dB/s",
        "lower",
        False,
    ),
    (
        "qrev_srmr_norm",
        "QREV",
        "reverberation_tail",
        "established_comparator",
        "ratio",
        "lower",
        True,
    ),
    # QCHAN
    (
        "qchan_ltas_distance_db",
        "QCHAN",
        "channel_device",
        "primary_nonordinal",
        "dB RMS",
        "contextual",
        True,
    ),
    (
        "qchan_rolloff95_deficit_hz",
        "QCHAN",
        "channel_device",
        "primary_one_sided",
        "Hz",
        "higher",
        True,
    ),
    (
        "qchan_highband_ratio_deficit",
        "QCHAN",
        "channel_device",
        "secondary_nonindependent",
        "proportion",
        "higher",
        True,
    ),
    (
        "qchan_tilt_steepening_db_per_oct",
        "QCHAN",
        "channel_device",
        "exploratory_phenotype_sensitive",
        "dB/octave",
        "higher",
        False,
    ),
    # QDIST
    (
        "qdist_hard_clipped_sample_fraction",
        "QDIST",
        "nonlinear_distortion",
        "primary",
        "fraction",
        "higher",
        True,
    ),
    (
        "qdist_hard_clip_event_rate_per_min",
        "QDIST",
        "nonlinear_distortion",
        "secondary",
        "events/min",
        "higher",
        True,
    ),
    (
        "qdist_hard_clipped_frame_fraction",
        "QDIST",
        "nonlinear_distortion",
        "conditional_audit",
        "fraction",
        "higher",
        False,
    ),
)


def reviewed_registry_frame() -> pd.DataFrame:
    versions = {family.code: family.version for family in FAMILIES}
    rows = []
    for feature, code, family, role, unit, worse, default_analysis in FEATURE_SPECS:
        rows.append(
            {
                "feature": feature,
                "family_code": code,
                "family": family,
                "measurement_version": versions[code],
                "role": role,
                "unit": unit,
                "worse": worse,
                "status_field": f"{feature}_status",
                "analysis_eligible": True,
                "default_analysis": bool(default_analysis),
                "validated_primary_set": True,
                "standalone_gate_allowed": False,
                "family_scalar_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _identity_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    preferred = tables["QADD"]
    identity_columns = [
        column
        for column in (
            "logical_recording_id",
            "SubjectID",
            "file_name",
            "recording_date_analysis",
            "Recording date",
        )
        if column in preferred
    ]
    identity = preferred[identity_columns].copy()
    if "logical_recording_id" not in identity:
        raise ValueError("QADD reviewed table lacks logical_recording_id")
    if (
        identity["logical_recording_id"].isna().any()
        or identity["logical_recording_id"].duplicated().any()
    ):
        raise ValueError("Reviewed identity table has missing or duplicate recording IDs")
    return identity


def _validate_cohorts(tables: dict[str, pd.DataFrame]) -> None:
    expected: set[str] | None = None
    for code, frame in tables.items():
        if "logical_recording_id" not in frame:
            raise ValueError(f"{code} table lacks logical_recording_id")
        ids = frame["logical_recording_id"].astype(str)
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"{code} has missing or duplicate recording IDs")
        observed = set(ids)
        if expected is None:
            expected = observed
        elif observed != expected:
            raise ValueError(f"{code} reviewed cohort does not match the other families")


def _family_readme(family: FamilyRelease, feature_count: int) -> str:
    return (
        f"# {family.code} latest reviewed features\n\n"
        f"Measurement version: `{family.version}`  \n"
        f"Validated analysis features: {feature_count}  \n"
        f"Cohort table: `features.csv`  \n"
        f"Normalized release registry: `feature_registry.csv`\n\n"
        "This folder is a concise release view. Full validation evidence, ledgers, "
        "figures, manifests, and scientific contracts remain under "
        "`MAIN outputs/02_FEATURE_REVIEWED/`.\n"
    )


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def build_latest_feature_release(project_root: str | Path) -> dict[str, object]:
    project_root = Path(project_root).resolve()
    reviewed = project_root / "MAIN outputs" / "02_FEATURE_REVIEWED"
    source_tables = reviewed / "01_analysis_features"
    source_registries = reviewed / "00_feature_registry"
    target = project_root / "MAIN outputs" / "02_FEATURE_LATEST"
    target.parent.mkdir(parents=True, exist_ok=True)

    registry = reviewed_registry_frame()
    tables: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, tuple[Path, Path]] = {}
    for family in FAMILIES:
        table_path = source_tables / family.table_name
        registry_path = source_registries / family.registry_name
        tables[family.code] = _read_csv(table_path)
        _read_csv(registry_path)
        source_paths[family.code] = (table_path, registry_path)
    _validate_cohorts(tables)

    merged = _identity_frame(tables)
    for family in FAMILIES:
        specs = registry.loc[registry["family_code"].eq(family.code)]
        columns = ["logical_recording_id"]
        for spec in specs.itertuples(index=False):
            if spec.feature not in tables[family.code]:
                raise ValueError(f"{family.code} is missing reviewed feature {spec.feature}")
            columns.append(spec.feature)
            if spec.status_field in tables[family.code]:
                columns.append(spec.status_field)
        family_status = f"{family.code.lower()}_family_status"
        if family_status in tables[family.code]:
            columns.append(family_status)
        if family.code == "QDIST" and "qdist_available" in tables[family.code]:
            columns.append("qdist_available")
        local = tables[family.code][list(dict.fromkeys(columns))].copy()
        merged = merged.merge(local, on="logical_recording_id", how="left", validate="one_to_one")

    staging = Path(tempfile.mkdtemp(prefix="02_FEATURE_LATEST.staging.", dir=str(target.parent)))
    try:
        _write_table(merged, staging / "recording_features")
        _write_table(registry, staging / "feature_registry")
        family_root = staging / "families"
        family_root.mkdir()
        manifest_rows = []
        for family in FAMILIES:
            destination = family_root / f"{family.order}_{family.code}"
            destination.mkdir()
            specs = registry.loc[registry["family_code"].eq(family.code)].copy()
            family_columns = ["logical_recording_id"]
            for spec in specs.itertuples(index=False):
                family_columns.append(spec.feature)
                if spec.status_field in tables[family.code]:
                    family_columns.append(spec.status_field)
            identity = [
                c
                for c in ("SubjectID", "file_name", "recording_date_analysis", "Recording date")
                if c in tables[family.code]
            ]
            compact = tables[family.code][
                list(dict.fromkeys(["logical_recording_id", *identity, *family_columns[1:]]))
            ]
            _write_table(compact, destination / "features")
            specs.to_csv(destination / "feature_registry.csv", index=False)
            (destination / "README.md").write_text(
                _family_readme(family, len(specs)), encoding="utf-8"
            )
            table_path, registry_path = source_paths[family.code]
            manifest_rows.append(
                {
                    "family_order": family.order,
                    "family_code": family.code,
                    "family": family.family,
                    "measurement_version": family.version,
                    "recordings": len(compact),
                    "analysis_features": len(specs),
                    "source_table": table_path.relative_to(project_root).as_posix(),
                    "source_table_sha256": _sha256(table_path),
                    "source_registry": registry_path.relative_to(project_root).as_posix(),
                    "source_registry_sha256": _sha256(registry_path),
                }
            )

        qtemp = family_root / "06_QTEMP"
        qtemp.mkdir()
        pd.DataFrame(
            [
                {
                    "family_code": "QTEMP",
                    "measurement_version": "qtemp-v1.0.0-analytical-final-no-retained",
                    "validated_primary_feature_count": 0,
                    "disposition": "NO_RETAINED_PRIMARY_FEATURES",
                    "exploratory_table": "MAIN outputs/02_FEATURE_TABLES_EXPLORATORY/qtemp_v100_exploratory_features.csv",
                }
            ]
        ).to_csv(qtemp / "feature_disposition.csv", index=False)
        (qtemp / "README.md").write_text(
            "# QTEMP latest disposition\n\n"
            "QTEMP contributes no validated primary feature to this release. Four deterministic "
            "outputs remain explicitly exploratory/monitoring-only in "
            "`MAIN outputs/02_FEATURE_TABLES_EXPLORATORY/`.\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "family_order": "06",
                "family_code": "QTEMP",
                "family": "temporal_discontinuity",
                "measurement_version": "qtemp-v1.0.0-analytical-final-no-retained",
                "recordings": len(merged),
                "analysis_features": 0,
                "source_table": "",
                "source_table_sha256": "",
                "source_registry": "",
                "source_registry_sha256": "",
            }
        )
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(staging / "release_manifest.csv", index=False)
        pd.DataFrame(
            [
                {
                    "historical_path": "outputs/reviewed",
                    "current_path": "MAIN outputs/02_FEATURE_REVIEWED/00_working_candidates",
                    "scope": "mutable review-stage candidates",
                },
                {
                    "historical_path": "MAIN outputs/reviewed",
                    "current_path": "MAIN outputs/02_FEATURE_REVIEWED",
                    "scope": "reviewed artifacts",
                },
            ]
        ).to_csv(staging / "historical_path_map.csv", index=False)
        (staging / "README.md").write_text(
            "# Latest reviewed feature release\n\n"
            f"This release contains {len(merged)} recordings and {len(registry)} validated "
            "analysis features from QGAIN, QADD, QREV, QCHAN, and QDIST. QTEMP has no "
            "retained primary feature.\n\n"
            "Use `recording_features.csv` for dataset assembly and `feature_registry.csv` "
            "for feature roles, units, directions, versions, and status fields. The `families/` "
            "folders are concise per-family views. Full scientific evidence remains in "
            "`../02_FEATURE_REVIEWED/`. `historical_path_map.csv` resolves paths embedded "
            "in older immutable manifests; those source manifests are intentionally not rewritten.\n",
            encoding="utf-8",
        )
        metadata = {
            "release_schema_version": "1.0.0",
            "recording_count": len(merged),
            "feature_count": len(registry),
            "families_with_features": 5,
            "qtemp_validated_primary_feature_count": 0,
        }
        (staging / "release.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        previous = target.with_name(target.name + ".previous")
        if previous.exists():
            shutil.rmtree(previous)
        if target.exists():
            os.replace(target, previous)
        os.replace(staging, target)
        if previous.exists():
            shutil.rmtree(previous)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "output_root": str(target),
        "recording_count": len(merged),
        "feature_count": len(registry),
        "family_count": 6,
    }


def load_latest_feature_release(project_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(project_root).resolve() / "MAIN outputs" / "02_FEATURE_LATEST"
    features = _read_csv(root / "recording_features.csv")
    registry = _read_csv(root / "feature_registry.csv")
    if features["logical_recording_id"].duplicated().any():
        raise ValueError("Latest reviewed feature release has duplicate recording IDs")
    missing = sorted(set(registry["feature"]) - set(features.columns))
    if missing:
        raise ValueError(f"Latest reviewed feature release is missing columns: {missing}")
    return features, registry
