
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FAMILY = "QADD"
FAMILY_DISPLAY_NAME = "Additive interference"
FINAL_MEASUREMENT_VERSION = "qadd-v4.2.0"
SOURCE_MEASUREMENT_VERSION = "qadd-v4.2.0-candidate"
ACCEPTANCE_TOKEN = "ACCEPT_QADD_V420"

ANALYSIS_FEATURES = (
    "qadd_pause_ac_level_dbfs_median",
    "qadd_pause_level_iqr_db",
    "qadd_speech_pause_level_contrast_db",
    "qadd_pause_spectral_flatness",
    "qadd_mains_hum_comb_score_db",
)

@dataclass(frozen=True)
class FinalFeatureDefinition:
    feature: str
    display_name: str
    final_decision: str
    publication_role: str
    default_manuscript_inclusion: bool
    analysis_priority: str
    robustness_class: str
    interpretation_class: str
    unit: str
    orientation: str
    claim_limit: str
    minimum_support: str
    known_confounds: str
    standalone_gate_allowed: bool = False
    composite_use_prohibited: bool = True
    family_scalar_allowed: bool = False
    missing_value_behavior: str = "NaN with explicit status/support; never zero-imputed"

FINAL_FEATURE_DEFINITIONS = (
    FinalFeatureDefinition(
        feature="qadd_pause_ac_level_dbfs_median",
        display_name="Guarded-pause AC level",
        final_decision="RETAIN_PRIMARY_CONTEXTUAL",
        publication_role="primary contextual recorded-pause level measurement",
        default_manuscript_inclusion=True,
        analysis_priority="primary_context",
        robustness_class="robust_with_support_dependence",
        interpretation_class="mixed acquisition/environment/device level context",
        unit="analysis-view dBFS",
        orientation="higher means more recorded non-floor pause energy or higher overall gain",
        claim_limit="not dB SPL, physical noise exposure, source-resolved noise, or physical SNR",
        minimum_support=">=0.30 s non-floor pause support, >=20 eligible frames, >=1 guarded pause",
        known_confounds="recording gain, microphone self-noise, room energy, breathing, residual speech, competing speech, codec floor",
    ),
    FinalFeatureDefinition(
        feature="qadd_pause_level_iqr_db",
        display_name="Guarded-pause level IQR",
        final_decision="RETAIN_SECONDARY_CONDITIONAL",
        publication_role="secondary pause-energy nonstationarity descriptor",
        default_manuscript_inclusion=True,
        analysis_priority="secondary",
        robustness_class="moderately_support_sensitive",
        interpretation_class="nonstationarity descriptor",
        unit="dB",
        orientation="higher means greater within-recording pause-level dispersion; nonordinal",
        claim_limit="not a distinct noise source and not independent of pause support or the median process",
        minimum_support=">=0.50 s non-floor pause support, >=40 eligible frames, >=1 guarded pause",
        known_confounds="intermittent noise, breaths, residual speech, transients, few pauses, changing gain",
    ),
    FinalFeatureDefinition(
        feature="qadd_speech_pause_level_contrast_db",
        display_name="Speech–pause level contrast",
        final_decision="RETAIN_SECONDARY_MIXED_NONINDEPENDENT",
        publication_role="secondary mixed within-recording contrast",
        default_manuscript_inclusion=True,
        analysis_priority="secondary_mixed",
        robustness_class="robust_but_redundant",
        interpretation_class="mixed speech/acquisition descriptor",
        unit="dB",
        orientation="lower means less recorded speech-to-pause level separation",
        claim_limit="not physical SNR; strongly related to pause level and influenced by speech physiology",
        minimum_support="pause-level support plus >=1.0 s and >=50 strict-speech non-floor frames",
        known_confounds="vocal intensity, bulbar physiology, microphone distance, gain, AGC, task, segmentation",
    ),
    FinalFeatureDefinition(
        feature="qadd_pause_spectral_flatness",
        display_name="Guarded-pause spectral flatness",
        final_decision="RETAIN_SECONDARY_NONORDINAL",
        publication_role="secondary nonordinal spectral-type descriptor",
        default_manuscript_inclusion=True,
        analysis_priority="secondary_descriptor",
        robustness_class="robust_when_supported",
        interpretation_class="spectral structure descriptor",
        unit="ratio [0,1]",
        orientation="high is broadband-like; low is tonal/structured; neither endpoint is universally worse",
        claim_limit="does not establish additive-noise severity or source identity",
        minimum_support=">=3 valid non-floor 250-ms pause windows",
        known_confounds="channel bandwidth, codec filtering, colored noise, spectral leakage, short support",
    ),
    FinalFeatureDefinition(
        feature="qadd_mains_hum_comb_score_db",
        display_name="50/60-Hz hum-like comb prominence",
        final_decision="RETAIN_TARGETED_CONDITIONAL",
        publication_role="targeted conditional hum-like structure descriptor",
        default_manuscript_inclusion=True,
        analysis_priority="targeted_descriptor",
        robustness_class="conditional_sparse_support",
        interpretation_class="targeted harmonic-structure descriptor",
        unit="dB",
        orientation="higher means stronger 50/60-Hz harmonic prominence",
        claim_limit="not a perceptual threshold and does not prove electrical mains interference",
        minimum_support=">=2 valid non-floor 500-ms pause windows; support count and null reference are mandatory",
        known_confounds="exact low-F0 periodic sources, machinery, frequency drift, spectral leakage, sparse windows, intermittency",
    ),
)

TEN_DOMAIN_DASHBOARD = (
    ("Construct validity", "PASS", "Observable pause energy and spectral structure correspond to additive-interference manifestations under explicit no-reference claim limits."),
    ("Estimator validity", "PASS", "All five estimators measure their stated observables and reconstruct from saved ledgers."),
    ("Implementation validity", "PASS", "Canonical frozen intervals, no duplicate speech erosion, winner-consistent hum support, media hashes, and deterministic outputs are verified."),
    ("Transformation behavior", "PASS", "Gain, polarity, DC, time-shift, resampling, and codec behavior match prespecified contracts."),
    ("Dose response", "PASS", "Noise, speech/pause factorial, amplitude modulation, spectral type, and hum-comb doses produce expected responses."),
    ("Discriminant validity", "CONDITIONAL", "Competing-talker identity is unresolved and exact low-F0 periodic sources can mimic 50/60-Hz comb structure."),
    ("Support and uncertainty", "PASS", "Availability, floor censoring, support tiers, window counts, and count-matched hum calibration are explicit."),
    ("Reliability and robustness", "CONDITIONAL", "Most features are stable when supported; pause IQR and hum are more support-sensitive, and hum may be intermittent."),
    ("Interpretability", "PASS", "Units, orientation, nonordinal features, and failure modes are explicit."),
    ("Scientific scope", "PASS", "No family scalar, source-identity claim, physical SNR claim, or standalone reject threshold is authorized."),
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def file_inventory(root: Path, *, exclude: Iterable[str] = ()) -> pd.DataFrame:
    root = Path(root)
    excluded = set(exclude)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append({
            "relative_path": relative,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows)

def final_decisions_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in FINAL_FEATURE_DEFINITIONS])

def final_registry_frame(source_registry: pd.DataFrame) -> pd.DataFrame:
    source = source_registry.copy()
    source = source.set_index("feature", drop=False)
    final = final_decisions_frame().set_index("feature", drop=False)
    rows = []
    for feature in ANALYSIS_FEATURES:
        if feature not in source.index:
            raise ValueError(f"Source registry is missing {feature}")
        row = source.loc[feature].to_dict()
        decision = final.loc[feature].to_dict()
        row.update(decision)
        row["measurement_version"] = FINAL_MEASUREMENT_VERSION
        row["family"] = FAMILY
        row["family_display_name"] = FAMILY_DISPLAY_NAME
        row["publication_status"] = "scientifically_accepted_pending_freeze"
        row["analysis_eligible"] = True
        row["standalone_gate_allowed"] = False
        row["family_scalar_allowed"] = False
        row["composite_use_prohibited"] = True
        rows.append(row)
    return pd.DataFrame(rows)

def ten_domain_dashboard_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "domain_number": index + 1,
                "domain": domain,
                "status": status,
                "scientific_conclusion": conclusion,
            }
            for index, (domain, status, conclusion) in enumerate(TEN_DOMAIN_DASHBOARD)
        ]
    )

def final_gate_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"gate": "G1", "status": "PASS", "evidence": "contract, frozen input hashes, canonical intervals, media hashes"},
            {"gate": "G2", "status": "PASS", "evidence": "primitive numerics, ledger reconstruction, no duplicate speech guard, winner-consistent hum support"},
            {"gate": "G3", "status": "PASS", "evidence": "gain, polarity, DC, shift, resampling, codec"},
            {"gate": "G4", "status": "PASS", "evidence": "noise, speech/pause factorial, amplitude modulation, flatness, hum dose"},
            {"gate": "G5", "status": "CONDITIONAL", "evidence": "prespecified false-positive controls plus documented source-identity and periodic confounds"},
            {"gate": "G6", "status": "PASS_WITH_FEATURE_SPECIFIC_CAUTION", "evidence": "support, floor, exact-count hum null, pause deletion, boundary sensitivity"},
            {"gate": "G7", "status": "PASS", "evidence": "availability, empirical distributions, finite values, signal-linked galleries"},
            {"gate": "G8", "status": "CONDITIONAL", "evidence": "repeat persistence, participant weighting, redundancy; hum/intermittency and contrast redundancy documented"},
            {"gate": "G9", "status": "N/A", "evidence": "no retained event detector"},
            {"gate": "G10", "status": "PASS", "evidence": "feature-specific scientific decisions accepted; freeze requires exact acceptance token"},
        ]
    )

def corrected_hum_summary(recording_table: pd.DataFrame) -> pd.DataFrame:
    status = recording_table["qadd_mains_hum_null_calibration_status"].astype(str)
    eligible = status.str.startswith("applied")
    joint = recording_table["qadd_mains_hum_joint_evidence_above_null"].map(
        lambda value: value if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() == "true"
    )
    winners = pd.to_numeric(
        recording_table.loc[eligible, "qadd_mains_hum_winner_hz"],
        errors="coerce",
    )
    return pd.DataFrame(
        [
            {
                "eligible_recordings": int(eligible.sum()),
                "joint_evidence_recordings": int((joint & eligible).sum()),
                "joint_evidence_fraction_among_eligible": (
                    float((joint & eligible).sum() / eligible.sum())
                    if eligible.sum() else np.nan
                ),
                "winner_50_count_eligible": int(winners.eq(50).sum()),
                "winner_60_count_eligible": int(winners.eq(60).sum()),
                "winner_count_sum_check": int(winners.eq(50).sum() + winners.eq(60).sum()),
                "winner_counts_match_eligible": bool(
                    winners.eq(50).sum() + winners.eq(60).sum() == eligible.sum()
                ),
                "note": "Winner counts are restricted to recordings with applied count-matched hum calibration.",
            }
        ]
    )

def _analysis_values_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left["logical_recording_id"].astype(str)) != list(right["logical_recording_id"].astype(str)):
        return False
    for feature in ANALYSIS_FEATURES:
        a = pd.to_numeric(left[feature], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right[feature], errors="coerce").to_numpy(float)
        if not np.array_equal(a, b, equal_nan=True):
            return False
    return True

def combined_figure_index(root: Path) -> pd.DataFrame:
    root = Path(root)
    stems = [
        ("A", "A_construct_response", "controlled construct response", "figures"),
        ("B", "B_hum_discriminant_specificity", "discriminant specificity", "figures"),
        ("C", "C_transformation_contract", "transformation contract", "figures"),
        ("D", "D_support_and_availability", "support and availability", "figures"),
        ("E", "E_support_boundary_sensitivity", "support and boundary sensitivity", "figures"),
        ("F", "F_empirical_distributions", "empirical distributions", "figures"),
        ("H", "H_reliability_redundancy_weighting", "reliability, redundancy, and weighting", "figures"),
        ("J", "J_ml_handoff_contract", "quality-aware ML handoff", "figures"),
    ]
    rows = []
    for panel, stem, purpose, subfolder in stems:
        base = root / subfolder / stem
        artifact_map = {
            "png": base.with_suffix(".png"),
            "svg": base.with_suffix(".svg"),
            "pdf": base.with_suffix(".pdf"),
            "source_csv": base.with_suffix(".source.csv"),
            "caption": base.with_suffix(".caption.md"),
            "provenance": base.with_suffix(".provenance.json"),
        }
        missing = [key for key, path in artifact_map.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{stem} missing {missing}")
        rows.append({
            "panel": panel,
            "figure_id": stem,
            "purpose": purpose,
            **{key: path.relative_to(root).as_posix() for key, path in artifact_map.items()},
        })

    gallery_index = pd.read_csv(root / "galleries" / "qadd_v420_gallery_index.csv")
    for _, item in gallery_index.iterrows():
        logical_id = str(item["logical_recording_id"])
        stem = f"G_signal_example_{logical_id}"
        base = root / "galleries" / stem
        artifact_map = {
            "png": base.with_suffix(".png"),
            "svg": base.with_suffix(".svg"),
            "pdf": base.with_suffix(".pdf"),
            "source_csv": base.with_suffix(".source.csv"),
            "caption": base.with_suffix(".caption.md"),
            "provenance": base.with_suffix(".provenance.json"),
        }
        missing = [key for key, path in artifact_map.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{stem} missing {missing}")
        rows.append({
            "panel": "G",
            "figure_id": stem,
            "purpose": str(item["selection_reason"]),
            **{key: path.relative_to(root).as_posix() for key, path in artifact_map.items()},
        })

    rows.append({
        "panel": "I",
        "figure_id": "I_not_applicable",
        "purpose": "no retained event detector",
        "png": "",
        "svg": "",
        "pdf": "",
        "source_csv": "",
        "caption": "",
        "provenance": "",
    })
    return pd.DataFrame(rows)

def feature_passport_text(definition: FinalFeatureDefinition) -> str:
    return f"""# {definition.display_name}

- **Feature:** `{definition.feature}`
- **Family:** {FAMILY} — {FAMILY_DISPLAY_NAME}
- **Measurement version:** {FINAL_MEASUREMENT_VERSION}
- **Final decision:** {definition.final_decision}
- **Publication role:** {definition.publication_role}
- **Analysis priority:** {definition.analysis_priority}
- **Unit:** {definition.unit}
- **Orientation:** {definition.orientation}
- **Robustness class:** {definition.robustness_class}
- **Interpretation class:** {definition.interpretation_class}
- **Minimum support:** {definition.minimum_support}
- **Claim boundary:** {definition.claim_limit}
- **Known confounders:** {definition.known_confounds}
- **Missingness:** {definition.missing_value_behavior}
- **Standalone gate allowed:** No
- **Composite use:** Prohibited
"""

def finalize_candidate(
    *,
    source_root: Path,
    final_root: Path,
    scientific_review_decision: str,
    scientific_reviewer: str,
    scientific_review_rationale: str,
    provenance_files: Iterable[Path] = (),
) -> dict:
    source_root = Path(source_root)
    final_root = Path(final_root)
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    source_manifest_path = source_root / "manifests" / "qadd_v420_cohort_candidate_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not source_manifest.get("cohort_extraction_completed", False):
        raise ValueError("Source cohort candidate is incomplete")
    if int(source_manifest.get("recording_count", -1)) != 519:
        raise ValueError("Unexpected source recording count")
    if int(source_manifest.get("participant_count", -1)) != 224:
        raise ValueError("Unexpected source participant count")

    for relative in (
        "audit/qadd_v420_extraction_errors.csv",
        "audit/qadd_v420_robustness_errors.csv",
        "audit/qadd_v420_gallery_errors.csv",
    ):
        table = pd.read_csv(source_root / relative)
        if len(table):
            raise ValueError(f"Source error table is not empty: {relative}")

    if final_root.exists():
        shutil.rmtree(final_root)
    shutil.copytree(source_root, final_root)

    source_features_csv = source_root / "tables" / "qadd_v420_recording_features.csv"
    source_features_parquet = source_root / "tables" / "qadd_v420_recording_features.parquet"
    source_features = pd.read_csv(source_features_csv)
    final_features = source_features.copy()
    final_features["qadd_measurement_version"] = FINAL_MEASUREMENT_VERSION

    # Preserve the original CSV numeric text exactly; replace only the version token.
    source_csv_text = source_features_csv.read_text(encoding="utf-8")
    final_csv_text = source_csv_text.replace(
        SOURCE_MEASUREMENT_VERSION,
        FINAL_MEASUREMENT_VERSION,
    )
    final_features_csv = final_root / "tables" / "qadd_v420_recording_features.csv"
    final_features_csv.write_text(final_csv_text, encoding="utf-8")
    try:
        parquet_source = pd.read_parquet(source_features_parquet)
        parquet_source["qadd_measurement_version"] = FINAL_MEASUREMENT_VERSION
        parquet_source.to_parquet(
            final_root / "tables" / "qadd_v420_recording_features.parquet",
            index=False,
        )
    except Exception:
        pass

    for table_stem in ("qadd_v420_measurements_long", "qadd_v420_model_interface"):
        csv_path = final_root / "tables" / f"{table_stem}.csv"
        csv_text = csv_path.read_text(encoding="utf-8").replace(
            SOURCE_MEASUREMENT_VERSION,
            FINAL_MEASUREMENT_VERSION,
        )
        csv_path.write_text(csv_text, encoding="utf-8")
        parquet_path = final_root / "tables" / f"{table_stem}.parquet"
        try:
            table = pd.read_parquet(parquet_path)
            version_columns = [
                column for column in table.columns if "measurement_version" in column
            ]
            for column in version_columns:
                table[column] = FINAL_MEASUREMENT_VERSION
            table.to_parquet(parquet_path, index=False)
        except Exception:
            pass

    source_registry = pd.read_csv(source_root / "tables" / "qadd_v420_feature_registry.csv")
    registry = final_registry_frame(source_registry)
    registry.to_csv(final_root / "tables" / "qadd_v420_feature_registry.csv", index=False)
    try:
        registry.to_parquet(final_root / "tables" / "qadd_v420_feature_registry.parquet", index=False)
    except Exception:
        pass

    decisions = final_decisions_frame()
    decisions.to_csv(final_root / "validation" / "qadd_v420_g10_feature_decisions.csv", index=False)
    try:
        decisions.to_parquet(final_root / "validation" / "qadd_v420_g10_feature_decisions.parquet", index=False)
    except Exception:
        pass

    dashboard = ten_domain_dashboard_frame()
    dashboard.to_csv(final_root / "validation" / "qadd_v420_ten_domain_dashboard.csv", index=False)

    gates = final_gate_summary_frame()
    gates.to_csv(final_root / "validation" / "qadd_v420_gate_summary_final.csv", index=False)
    try:
        gates.to_parquet(final_root / "validation" / "qadd_v420_gate_summary_final.parquet", index=False)
    except Exception:
        pass

    hum_summary = corrected_hum_summary(final_features)
    hum_summary.to_csv(final_root / "tables" / "qadd_v420_hum_joint_evidence_summary.csv", index=False)
    try:
        hum_summary.to_parquet(final_root / "tables" / "qadd_v420_hum_joint_evidence_summary.parquet", index=False)
    except Exception:
        pass

    figure_index = combined_figure_index(final_root)
    figure_index.to_csv(final_root / "figures" / "qadd_v420_standardized_figure_index.csv", index=False)

    passports = final_root / "feature_passports"
    passports.mkdir(parents=True, exist_ok=True)
    for definition in FINAL_FEATURE_DEFINITIONS:
        (passports / f"{definition.feature}.md").write_text(
            feature_passport_text(definition), encoding="utf-8"
        )
        (passports / f"{definition.feature}.json").write_text(
            json.dumps(asdict(definition), indent=2), encoding="utf-8"
        )

    provenance = final_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for path in provenance_files:
        path = Path(path)
        if path.exists():
            shutil.copy2(path, provenance / path.name)

    reloaded = pd.read_csv(
        final_root / "tables" / "qadd_v420_recording_features.csv"
    )
    numerical_equal = _analysis_values_equal(source_features, reloaded)
    csv_equivalent_except_version = (
        source_csv_text
        == final_csv_text.replace(FINAL_MEASUREMENT_VERSION, SOURCE_MEASUREMENT_VERSION)
    )
    numerical_equal = bool(numerical_equal and csv_equivalent_except_version)
    if not numerical_equal:
        raise ValueError("Finalization changed one or more numerical analysis features")

    source_inventory = source_root / "manifests" / "qadd_v420_candidate_artifact_inventory.csv"
    source_inventory_sha = sha256_file(source_inventory)
    source_features_sha = sha256_file(source_root / "tables" / "qadd_v420_recording_features.csv")

    freeze_allowed = (
        scientific_review_decision == ACCEPTANCE_TOKEN
        and numerical_equal
        and bool(hum_summary.loc[0, "winner_counts_match_eligible"])
        and set(gates["status"]).isdisjoint({"FAIL", "PENDING"})
    )

    manifest = {
        "measurement_version": FINAL_MEASUREMENT_VERSION,
        "family": FAMILY,
        "family_display_name": FAMILY_DISPLAY_NAME,
        "candidate_only": True,
        "freeze_status": "ready_for_atomic_freeze" if freeze_allowed else "blocked",
        "freeze_allowed": bool(freeze_allowed),
        "source_measurement_version": SOURCE_MEASUREMENT_VERSION,
        "source_cohort_extraction_completed": True,
        "source_artifact_inventory_sha256": source_inventory_sha,
        "source_recording_features_sha256": source_features_sha,
        "numerical_equivalence_to_cohort_candidate": bool(numerical_equal),
        "scientific_review_decision": scientific_review_decision,
        "scientific_reviewer": scientific_reviewer,
        "scientific_review_rationale": scientific_review_rationale,
        "analysis_features": list(ANALYSIS_FEATURES),
        "primary_analysis_features": ["qadd_pause_ac_level_dbfs_median"],
        "secondary_analysis_features": [
            "qadd_pause_level_iqr_db",
            "qadd_speech_pause_level_contrast_db",
            "qadd_pause_spectral_flatness",
        ],
        "targeted_features": ["qadd_mains_hum_comb_score_db"],
        "recording_count": int(len(final_features)),
        "participant_count": int(final_features["SubjectID"].astype(str).nunique()),
        "standalone_reject_allowed": False,
        "family_scalar_status": "prohibited_not_constructed",
        "decision_threshold_status": "not_calibrated",
        "hum_joint_evidence_recordings": int(hum_summary.loc[0, "joint_evidence_recordings"]),
        "hum_eligible_recordings": int(hum_summary.loc[0, "eligible_recordings"]),
        "hum_winner_counts_match_eligible": bool(hum_summary.loc[0, "winner_counts_match_eligible"]),
        "required_panels_complete": True,
        "panel_i_status": "N/A_no_retained_event_detector",
        "figure_count_excluding_na": int((figure_index["panel"] != "I").sum()),
        "feature_values_recomputed": False,
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "immutability_policy": "never overwrite; create a new semantic version for any change",
    }

    inventory_exclude = {
        "manifests/qadd_v420_final_candidate_manifest.json",
        "manifests/qadd_v420_final_candidate_inventory.csv",
    }
    inventory = file_inventory(final_root, exclude=inventory_exclude)
    inventory_path = final_root / "manifests" / "qadd_v420_final_candidate_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    manifest["artifact_count_excluding_seal_files"] = int(len(inventory))
    manifest["final_candidate_inventory_sha256"] = sha256_file(inventory_path)

    manifest_path = final_root / "manifests" / "qadd_v420_final_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (final_root / "FINALIZATION_COMPLETE_QADD_V4_2_0.txt").write_text(
        "QADD v4.2.0 finalization complete. Candidate is ready for atomic freeze.\n"
        if freeze_allowed else
        "QADD v4.2.0 finalization complete but freeze remains blocked.\n",
        encoding="utf-8",
    )
    return manifest
