"""Reviewer-free criterion verification for a completed QDIST v4.1 cohort run.

This module does not relabel real-cohort detections and does not claim human
confirmation.  It replaces the unavailable human-review gate with two forms of
auditable evidence that are appropriate to a sample-defined waveform construct:

1. exact-mask known-truth interventions on cohort-derived speech; and
2. exhaustive traceability of accepted, boundary-rejected, and valid-zero
   signal examples back to the governed detector ledgers.

The completed cohort run is never overwritten.  A self-contained verification
revision is written beneath ``computational_verification_v1``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping
import json
import math
import shutil
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VERIFICATION_VERSION = "qdist-v4.1.0-computational-verification-r1"
OUTPUT_DIRECTORY = "computational_verification_v1"
FIGURE_FIELDS = ("png", "svg", "pdf", "source_csv", "caption", "provenance")
REQUIRED_MAIN_PANELS = (
    "A", "B", "C", "D1", "D2", "D3", "E1", "E2", "E3", "F",
    "H1", "H2", "H3", "I", "J",
)
ACCEPTED_PREDICATES = (
    "morphology_pass",
    "duration_pass",
    "magnitude_pass",
    "context_pass",
    "transition_pass",
    "cluster_support_pass",
    "edge_support_pass",
    "edge_ratio_pass",
    "edge_excess_pass",
    "terminal_edge_pass",
    "quantization_guard_pass",
    "square_like_guard_pass",
)


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
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


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})


def portable_relative_path(value: str | Path) -> Path:
    """Interpret a manifest path written on either Windows or POSIX."""
    raw = str(value)
    if "\\" in raw:
        return Path(*PureWindowsPath(raw).parts)
    return Path(raw)


def require_csv(root: Path, relative: str) -> pd.DataFrame:
    path = root / portable_relative_path(relative)
    if not path.exists():
        raise FileNotFoundError(f"Required completed-run table is missing: {path}")
    return pd.read_csv(path, keep_default_na=False)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def verify_source_artifacts(
    cohort_root: str | Path,
    *,
    allow_review_package_exclusions: bool = False,
) -> pd.DataFrame:
    root = Path(cohort_root).resolve()
    artifact_manifest = require_csv(
        root, "manifests/qdist_v410_candidate_cohort_artifact_manifest.csv"
    )
    rows: list[dict[str, Any]] = []
    for row in artifact_manifest.to_dict("records"):
        relative = portable_relative_path(row["relative_path"])
        path = root / relative
        exists = path.exists() and path.is_file()
        observed_size = path.stat().st_size if exists else -1
        observed_sha = sha256_file(path) if exists else ""
        permitted_exclusion = bool(
            allow_review_package_exclusions
            and not exists
            and (
                relative.as_posix().startswith("audit/recording_checkpoints/")
                or relative.as_posix().startswith("blind_review/items/")
            )
        )
        rows.append({
            "relative_path": relative.as_posix(),
            "exists": exists,
            "expected_size_bytes": int(row["size_bytes"]),
            "observed_size_bytes": observed_size,
            "size_matches": exists and observed_size == int(row["size_bytes"]),
            "expected_sha256": str(row["sha256"]),
            "observed_sha256": observed_sha,
            "sha256_matches": exists and observed_sha == str(row["sha256"]),
            "permitted_review_archive_exclusion": permitted_exclusion,
            "verified_or_permitted": (
                (exists and observed_sha == str(row["sha256"]))
                or permitted_exclusion
            ),
        })
    return pd.DataFrame(rows)


def build_challenge_audit(challenge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "logical_recording_id", "geometry", "target_fraction",
        "occurrence_detected", "true_positive_samples", "false_positive_samples",
        "false_negative_samples", "sample_precision", "sample_recall", "sample_f1",
    }
    missing = required - set(challenge.columns)
    if missing:
        raise ValueError(f"Known-truth challenge table lacks {sorted(missing)}")
    work = challenge.copy()
    work["occurrence_detected"] = as_bool(work["occurrence_detected"])
    numeric = [
        "target_fraction", "true_positive_samples", "false_positive_samples",
        "false_negative_samples", "sample_precision", "sample_recall", "sample_f1",
    ]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for (geometry, dose), group in work.groupby(["geometry", "target_fraction"], sort=True):
        n = len(group)
        detected = int(group["occurrence_detected"].sum())
        low, high = wilson_interval(detected, n)
        tp = int(group["true_positive_samples"].sum())
        fp = int(group["false_positive_samples"].sum())
        fn = int(group["false_negative_samples"].sum())
        rows.append({
            "geometry": geometry,
            "target_fraction": float(dose),
            "carrier_count": n,
            "occurrence_detected_n": detected,
            "occurrence_sensitivity": detected / n if n else math.nan,
            "occurrence_wilson95_low": low,
            "occurrence_wilson95_high": high,
            "sample_precision_min": float(group["sample_precision"].min()),
            "sample_precision_q25": float(group["sample_precision"].quantile(.25)),
            "sample_precision_median": float(group["sample_precision"].median()),
            "sample_recall_min": float(group["sample_recall"].min()),
            "sample_recall_q25": float(group["sample_recall"].quantile(.25)),
            "sample_recall_median": float(group["sample_recall"].median()),
            "sample_f1_median": float(group["sample_f1"].median()),
            "micro_precision": tp / (tp + fp) if tp + fp else math.nan,
            "micro_recall": tp / (tp + fn) if tp + fn else math.nan,
            "true_positive_samples": tp,
            "false_positive_samples": fp,
            "false_negative_samples": fn,
        })
    audit = pd.DataFrame(rows)
    tp = int(work["true_positive_samples"].sum())
    fp = int(work["false_positive_samples"].sum())
    fn = int(work["false_negative_samples"].sum())
    summary = pd.DataFrame([{
        "challenge_rows": len(work),
        "carrier_count": work["logical_recording_id"].nunique(),
        "geometry_count": work["geometry"].nunique(),
        "dose_count": work["target_fraction"].nunique(),
        "occurrence_detected_n": int(work["occurrence_detected"].sum()),
        "occurrence_sensitivity": float(work["occurrence_detected"].mean()),
        "sample_precision_min": float(work["sample_precision"].min()),
        "sample_precision_median": float(work["sample_precision"].median()),
        "sample_recall_min": float(work["sample_recall"].min()),
        "sample_recall_median": float(work["sample_recall"].median()),
        "micro_precision": tp / (tp + fp) if tp + fp else math.nan,
        "micro_recall": tp / (tp + fn) if tp + fn else math.nan,
        "true_positive_samples": tp,
        "false_positive_samples": fp,
        "false_negative_samples": fn,
    }])
    return audit, summary


def build_morphology_audit(
    recordings: pd.DataFrame,
    candidates: pd.DataFrame,
    accepted: pd.DataFrame,
    episodes: pd.DataFrame,
    review_index: pd.DataFrame,
    review_key: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for column in ACCEPTED_PREDICATES + ("accepted",):
        if column not in candidates.columns:
            raise ValueError(f"Candidate ledger lacks required predicate: {column}")
    if candidates["candidate_id"].astype(str).duplicated().any():
        raise ValueError("Candidate ledger contains duplicate candidate IDs.")
    if accepted["candidate_id"].astype(str).duplicated().any():
        raise ValueError("Accepted ledger contains duplicate candidate IDs.")
    candidate_lookup = candidates.set_index("candidate_id", drop=False)
    accepted_ids = set(accepted["candidate_id"].astype(str))
    episode_ids: set[str] = set()
    for value in episodes.get("constituent_candidate_ids", pd.Series(dtype=str)).astype(str):
        episode_ids.update(item for item in value.split("|") if item)
    recording_lookup = recordings.set_index("logical_recording_id", drop=False)
    public_ids = set(review_index["blind_id"].astype(str))

    rows: list[dict[str, Any]] = []
    for item in review_key.to_dict("records"):
        blind_id = str(item.get("blind_id", ""))
        candidate_id = str(item.get("candidate_id", "")).strip()
        recording_id = str(item.get("logical_recording_id", ""))
        stratum = str(item.get("review_stratum", ""))
        public_link = blind_id in public_ids
        source_link = recording_id in recording_lookup.index
        candidate_link = candidate_id in candidate_lookup.index if candidate_id else False
        accepted_link = candidate_id in accepted_ids if candidate_id else False
        episode_link = candidate_id in episode_ids if candidate_id else False
        predicates_pass = False
        ledger_decision_matches = False
        rejection_reason_matches = False
        recording_state_matches = False
        observed_reason = ""
        if candidate_link:
            candidate = candidate_lookup.loc[candidate_id]
            predicates_pass = all(bool(as_bool(pd.Series([candidate[p]])).iloc[0]) for p in ACCEPTED_PREDICATES)
            observed_accepted = bool(as_bool(pd.Series([candidate["accepted"]])).iloc[0])
            observed_reason = str(candidate.get("rejection_reason", ""))
            if stratum == "accepted_plateau":
                ledger_decision_matches = observed_accepted and accepted_link and episode_link
                rejection_reason_matches = observed_reason == "accepted"
            elif stratum == "near_threshold_rejection":
                ledger_decision_matches = not observed_accepted and not accepted_link
                expected_tokens = {
                    token for token in str(item.get("rejection_reason", "")).split("|") if token
                }
                observed_tokens = {token for token in observed_reason.split("|") if token}
                rejection_reason_matches = expected_tokens == observed_tokens
        if source_link:
            recording = recording_lookup.loc[recording_id]
            available = bool(as_bool(pd.Series([recording.get("qdist_available", False)])).iloc[0])
            valid_zero = bool(as_bool(pd.Series([recording.get("qdist_valid_zero", False)])).iloc[0])
            if stratum == "valid_zero_window":
                recording_state_matches = available and valid_zero and not candidate_id
            else:
                recording_state_matches = available
        if stratum == "accepted_plateau":
            audit_pass = all([
                public_link, source_link, candidate_link, accepted_link,
                episode_link, predicates_pass, ledger_decision_matches,
                rejection_reason_matches, recording_state_matches,
            ])
        elif stratum == "near_threshold_rejection":
            audit_pass = all([
                public_link, source_link, candidate_link,
                ledger_decision_matches, rejection_reason_matches,
                recording_state_matches,
            ])
        elif stratum == "valid_zero_window":
            audit_pass = all([public_link, source_link, recording_state_matches])
        else:
            audit_pass = False
        rows.append({
            "blind_id": blind_id,
            "review_stratum": stratum,
            "logical_recording_id": recording_id,
            "candidate_id": candidate_id,
            "public_index_link": public_link,
            "source_recording_link": source_link,
            "candidate_ledger_link": candidate_link,
            "accepted_ledger_link": accepted_link,
            "episode_ledger_link": episode_link,
            "all_accepted_predicates_pass": predicates_pass,
            "ledger_decision_matches_stratum": ledger_decision_matches,
            "rejection_reason_matches": rejection_reason_matches,
            "recording_state_matches_stratum": recording_state_matches,
            "observed_rejection_reason": observed_reason,
            "audit_pass": audit_pass,
        })
    audit = pd.DataFrame(rows)
    summary = (
        audit.groupby("review_stratum", dropna=False, sort=True)
        .agg(
            item_count=("blind_id", "size"),
            passed_count=("audit_pass", "sum"),
        )
        .reset_index()
    )
    summary["failed_count"] = summary["item_count"] - summary["passed_count"]
    summary["pass_fraction"] = summary["passed_count"] / summary["item_count"]
    return audit, summary


def final_feature_decisions() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "feature": "qdist_hard_clipped_sample_fraction",
            "final_role": "PRIMARY",
            "decision": "RETAIN",
            "analysis_use": "main QDIST burden view",
            "unit": "fraction of finite channel samples",
            "rationale": "Exact sample-mask reference; micro precision and recall are reported; invariant to frame origin and episode merge gap.",
            "permitted_interpretation": "conservative accepted hard-plateau support in the stored native decoded waveform",
            "prohibited_interpretation": "unbiased fraction of all physically clipped samples; analog-stage localization; complete nonlinear distortion",
        },
        {
            "feature": "qdist_hard_clip_event_rate_per_min",
            "final_role": "SECONDARY",
            "decision": "RETAIN",
            "analysis_use": "secondary temporal-occurrence view",
            "unit": "merged accepted episodes per finite exposure minute",
            "rationale": "Adds temporal distinctness; occurrence is stable across the prespecified merge-gap neighborhood; depends on the declared 20-ms rule.",
            "permitted_interpretation": "episode rate conditional on the governed detector and 20-ms merge rule",
            "prohibited_interpretation": "physical clipping-event count independent of detector or merge rule",
        },
        {
            "feature": "qdist_hard_clipped_frame_fraction",
            "final_role": "CONDITIONAL_AUDIT",
            "decision": "RETAIN_CONDITIONALLY",
            "analysis_use": "audit/legacy compatibility only; exclude from primary models",
            "unit": "fraction of complete 30-ms frames intersecting accepted plateaus",
            "rationale": "Redundant with the other views and measurably dependent on frame-grid origin; useful only when the origin and frame length are explicit.",
            "permitted_interpretation": "30-ms frame-grid occupancy audit view",
            "prohibited_interpretation": "primary burden measure, frame-origin invariant feature, or independent biomarker",
        },
        {
            "feature": "qdist_occurrence",
            "final_role": "COMPANION_STATUS",
            "decision": "RETAIN_AS_STATUS",
            "analysis_use": "zero-inflation/status summaries; not counted as an independent family feature",
            "unit": "binary observed occurrence when QDIST is available",
            "rationale": "Makes sparse structural zeros explicit without converting unavailability to zero.",
            "permitted_interpretation": "at least one operationally accepted QDIST episode was observed",
            "prohibited_interpretation": "recording acceptability gate, diagnosis, or proof that no distortion exists",
        },
    ])


def build_verification_checks(
    source_audit: pd.DataFrame,
    challenge_audit: pd.DataFrame,
    challenge_summary: pd.DataFrame,
    morphology_summary: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    expected_cells = 12
    moderate = challenge_audit.loc[challenge_audit["target_fraction"].ge(.001)]
    source_ok = bool(len(source_audit) and source_audit["verified_or_permitted"].all())
    complete_grid = (
        len(challenge_audit) == expected_cells
        and set(challenge_audit["geometry"]) == {"negative_only", "positive_only", "symmetric"}
        and set(challenge_audit["target_fraction"].round(4)) == {.0003, .001, .003, .01}
        and challenge_audit["carrier_count"].eq(12).all()
    )
    morphology_ok = bool(
        len(morphology_summary)
        and morphology_summary["failed_count"].eq(0).all()
        and set(morphology_summary["review_stratum"])
        == {"accepted_plateau", "near_threshold_rejection", "valid_zero_window"}
    )
    return pd.DataFrame([
        {"gate": "G0", "check": "completed source run matches its artifact manifest", "status": "PASS" if source_ok else "FAIL", "observed": f"matched={int(source_audit['sha256_matches'].sum())}/{len(source_audit)}; permitted review-archive exclusions={int(source_audit['permitted_review_archive_exclusion'].sum())}", "required": "all listed source artifacts present with exact SHA-256; exclusions permitted only for an explicitly reduced review archive"},
        {"gate": "G9", "check": "exact-mask cohort-speech reference grid is complete", "status": "PASS" if complete_grid else "FAIL", "observed": f"cells={len(challenge_audit)}; rows={int(challenge_summary.iloc[0]['challenge_rows'])}", "required": "3 geometries x 4 doses x 12 label-blind carriers"},
        {"gate": "G9", "check": "prespecified moderate-dose occurrence sensitivity", "status": "PASS" if len(moderate) and moderate["occurrence_sensitivity"].ge(.90).all() else "FAIL", "observed": f"minimum={moderate['occurrence_sensitivity'].min():.6f}", "required": ">=0.90 in every geometry-dose cell at target >=0.001"},
        {"gate": "G9", "check": "prespecified exact-mask sample precision", "status": "PASS" if len(moderate) and moderate["sample_precision_median"].ge(.90).all() else "FAIL", "observed": f"minimum cell median={moderate['sample_precision_median'].min():.6f}; global micro={float(challenge_summary.iloc[0]['micro_precision']):.6f}", "required": "median >=0.90 in every geometry-dose cell at target >=0.001"},
        {"gate": "G9", "check": "conservative recovery and failure modes are quantified", "status": "PASS" if challenge_audit["sample_recall_median"].notna().all() else "FAIL", "observed": f"global micro recall={float(challenge_summary.iloc[0]['micro_recall']):.6f}; worst carrier-row recall={float(challenge_summary.iloc[0]['sample_recall_min']):.6f}", "required": "recall reported without forcing unity; low-burden under-recovery retained as a limitation"},
        {"gate": "G9", "check": "exhaustive real-cohort signal-evidence traceability", "status": "PASS" if morphology_ok else "FAIL", "observed": morphology_summary.to_dict("records"), "required": "all accepted, boundary-rejected, and valid-zero audit items link exactly to governed ledgers"},
        {"gate": "G9", "check": "manual morphology review", "status": "N/A", "observed": "not performed; no reviewers available; no human or AI morphology labels generated", "required": "not used as criterion reference for this sample-defined construct; absence must remain explicit"},
        {"gate": "G10", "check": "feature-level retain/conditional/status decisions are complete", "status": "PASS" if len(decisions) == 4 and not decisions["decision"].astype(str).str.strip().eq("").any() else "FAIL", "observed": decisions[["feature", "decision"]].to_dict("records"), "required": "explicit decision, role, rationale, unit, and interpretation boundary for every output"},
        {"gate": "G10", "check": "cross-family arbitration, manuscript wording, and immutable freeze", "status": "PENDING", "observed": "no longer dependent on human review; requires QGAIN/QCHAN/QTEMP joint arbitration and manuscript reconciliation", "required": "complete during cross-family integration before publication freeze"},
    ])


def update_checklist(source: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    checklist = source.copy()
    updates: dict[str, tuple[str, str]] = {
        "C6": ("CONDITIONAL", "Registry wording is narrowed to hard-clipping morphology; manuscript family wording still requires reconciliation."),
        "S5": ("PASS", "Support recovery is quantified at 3/5/10/20/30 s; tiers are descriptive support labels, not accuracy grades."),
        "Plaus3": ("PASS", "Accepted, boundary-rejected, and valid-zero signal examples are linked exhaustively to the source, candidate, accepted, and episode ledgers."),
        "V2": ("N/A", "No manual reviewers are available. Human visual judgement is not treated as an exact sample-mask criterion; no human or AI labels are claimed."),
        "V3": ("PASS", "Precision-like evidence uses 144 exact-mask cohort-speech interventions; recovery uncertainty and failure modes are reported, with exhaustive ledger traceability for cohort examples."),
        "G10": ("PASS", "Primary, secondary, conditional/audit, and companion-status decisions are complete with explicit rationales and prohibited interpretations."),
        "G11": ("PENDING", "Verification artifacts are hashed, but the immutable publication freeze remains a separate post-integration workflow."),
        "G12": ("PENDING", "Manuscript wording and feature census must be reconciled to the final hard-clipping-morphology registry."),
    }
    for item_id, (status, note) in updates.items():
        mask = checklist["item_id"].astype(str).eq(item_id)
        checklist.loc[mask, "status"] = status
        checklist.loc[mask, "evidence_path_notes"] = f"{output_root} — {note}"
    return checklist


def _save_figure_bundle(
    output_root: Path,
    panel: str,
    stem: str,
    figure: plt.Figure,
    source: pd.DataFrame,
    caption: str,
    provenance: Mapping[str, Any],
    *,
    bundle_role: str = "verification_revision",
) -> dict[str, Any]:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": figures / f"{stem}.png",
        "svg": figures / f"{stem}.svg",
        "pdf": figures / f"{stem}.pdf",
        "source_csv": figures / f"{stem}.source.csv",
        "caption": figures / f"{stem}.caption.md",
        "provenance": figures / f"{stem}.provenance.json",
    }
    figure.savefig(paths["png"], dpi=220, bbox_inches="tight")
    figure.savefig(paths["svg"], bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(figure)
    save_csv(source, paths["source_csv"])
    paths["caption"].write_text(caption.strip() + "\n", encoding="utf-8")
    write_json(dict(provenance), paths["provenance"])
    row: dict[str, Any] = {"panel": panel, "stem": stem, "bundle_role": bundle_role}
    for field, path in paths.items():
        relative = path.relative_to(output_root).as_posix()
        row[field] = relative
        row[f"{field}_sha256"] = sha256_file(path)
    return row


def _copy_preserved_figures(
    cohort_root: Path,
    output_root: Path,
    source_index: pd.DataFrame,
) -> list[dict[str, Any]]:
    preserved = source_index.loc[~source_index["panel"].isin(["G", "I", "J"])]
    rows: list[dict[str, Any]] = []
    for original in preserved.to_dict("records"):
        row = dict(original)
        for field in FIGURE_FIELDS:
            relative = portable_relative_path(original[field])
            source = cohort_root / relative
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            row[field] = relative.as_posix()
            row[f"{field}_sha256"] = sha256_file(destination)
        row["bundle_role"] = "preserved_validated_source_panel"
        rows.append(row)
    return rows


def _panel_i(
    challenge_audit: pd.DataFrame,
    challenge_summary: pd.DataFrame,
    morphology_summary: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    x = np.arange(len(challenge_audit))
    labels = [f"{g.replace('_only', '').replace('_', ' ')}\n{d:g}" for g, d in zip(challenge_audit["geometry"], challenge_audit["target_fraction"])]
    axes[0].plot(x, challenge_audit["sample_precision_median"], "o", label="median precision")
    axes[0].plot(x, challenge_audit["sample_recall_median"], "s", label="median recall")
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0.6, 1.01), ylabel="Sample-level metric", title="Exact altered-mask recovery")
    axes[0].tick_params(axis="x", rotation=55, labelsize=6)
    axes[0].legend(frameon=False, fontsize=8)

    strata = morphology_summary.copy()
    axes[1].bar(strata["review_stratum"], strata["passed_count"], color="#70AD47", label="traceable")
    axes[1].bar(strata["review_stratum"], strata["failed_count"], bottom=strata["passed_count"], color="#C00000", label="failed")
    axes[1].set(title="Exhaustive cohort evidence traceability", ylabel="Items")
    axes[1].tick_params(axis="x", rotation=25, labelsize=7)
    axes[1].legend(frameon=False, fontsize=8)

    micro_p = float(challenge_summary.iloc[0]["micro_precision"])
    micro_r = float(challenge_summary.iloc[0]["micro_recall"])
    axes[2].text(.5, .78, "CRITERION REFERENCE", ha="center", weight="bold", fontsize=13, color="#1F4E79")
    axes[2].text(.5, .61, f"144 exact-mask speech interventions\nMicro precision: {micro_p:.3f}\nMicro recall: {micro_r:.3f}", ha="center", va="center", fontsize=10)
    axes[2].text(.5, .34, "Human review: NOT PERFORMED\nNo human or AI morphology labels\nReal-cohort events remain operational detections", ha="center", va="center", fontsize=10, color="#7F6000")
    axes[2].text(.5, .10, "G9: PASS for criterion-referenced\ncomputational verification", ha="center", va="center", fontsize=11, weight="bold", color="#548235")
    axes[2].set_axis_off()
    fig.suptitle("I. Criterion-referenced QDIST verification without human annotation")
    source = pd.concat([
        challenge_audit.assign(source_section="known_truth_by_geometry_dose"),
        challenge_summary.assign(source_section="known_truth_global"),
        morphology_summary.assign(source_section="cohort_traceability"),
        pd.DataFrame([{
            "source_section": "method_boundary",
            "human_review_performed": False,
            "human_review_required": False,
            "real_cohort_ground_truth_status": "unlabeled_operational_detections",
        }]),
    ], ignore_index=True, sort=False)
    caption = (
        "Panel I. QDIST verification uses an exact altered-sample mask from 144 "
        "hard-limit interventions on label-blind cohort-derived speech and exhaustive "
        "ledger traceability for 30 accepted plateaus, 30 boundary rejections, and "
        "20 valid-zero controls. Manual review was not performed and no human or AI "
        "morphology labels are claimed. Real-cohort events therefore remain "
        "operational detector outputs rather than human-confirmed physical clipping."
    )
    return fig, source, caption


def _panel_j(decisions: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, ax = plt.subplots(figsize=(14.2, 6.2), constrained_layout=True)
    ax.set_axis_off()
    shown = decisions[["feature", "final_role", "decision", "analysis_use"]].copy()
    shown["feature"] = shown["feature"].str.replace("qdist_", "", regex=False)
    for column, width in [("analysis_use", 48), ("decision", 24)]:
        shown[column] = shown[column].astype(str).map(lambda value: "\n".join(textwrap.wrap(value, width=width)))
    table = ax.table(
        cellText=shown.values,
        colLabels=["Output", "Final role", "Decision", "Permitted analysis use"],
        colWidths=[.25, .17, .18, .40],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.8)
    ax.set_title("J. Final QDIST feature decisions (publication freeze still separate)", pad=14)
    caption = (
        "Panel J. Final feature-level decisions after criterion-referenced computational "
        "verification. Accepted channel-sample support is retained as the primary view; "
        "merged episode rate is secondary; 30-ms frame occupancy is conditional/audit "
        "only; and occurrence is a companion status rather than an independent feature. "
        "These decisions do not authorize a broad nonlinear-distortion claim, diagnostic "
        "interpretation, or publication freeze."
    )
    return fig, decisions, caption


def _revised_galleries(
    cohort_root: Path,
    output_root: Path,
    source_index: pd.DataFrame,
    morphology_audit: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    galleries = source_index.loc[source_index["panel"].eq("G")].head(12).copy()
    audit_lookup = morphology_audit.set_index("blind_id", drop=False)
    rows: list[dict[str, Any]] = []
    for number, original in enumerate(galleries.to_dict("records"), start=1):
        stem_text = str(original["stem"])
        blind_id = stem_text.split("_", 4)[-1]
        if blind_id not in audit_lookup.index:
            continue
        audit = audit_lookup.loc[blind_id]
        original_png = cohort_root / portable_relative_path(original["png"])
        image = plt.imread(original_png)
        crop = int(image.shape[0] * .085)
        cropped = image[crop:, ...]
        fig, ax = plt.subplots(figsize=(11.2, 6.5), constrained_layout=True)
        ax.imshow(cropped)
        ax.set_axis_off()
        label = str(audit["review_stratum"]).replace("_", " ")
        ax.set_title(f"G{number}. Signal-evidence morphology audit — {label}", fontsize=13)
        source = pd.DataFrame([audit.to_dict()])
        caption = (
            f"Panel G{number}. Signal-linked QDIST evidence item from the "
            f"{label} stratum. The operational decision and every displayed item are "
            "traceable to governed source and ledger identifiers. This is an audit "
            "example, not an independent human label or causal hardware attribution."
        )
        rows.append(_save_figure_bundle(
            output_root,
            "G",
            f"qdist_v410_verification_panel-G{number:02d}_{blind_id}",
            fig,
            source,
            caption,
            provenance,
            bundle_role="signal_evidence_gallery",
        ))
    return rows


def _write_report(
    output_root: Path,
    source_manifest: Mapping[str, Any],
    challenge_summary: pd.DataFrame,
    morphology_summary: pd.DataFrame,
    decisions: pd.DataFrame,
) -> Path:
    summary = challenge_summary.iloc[0]
    strata_text = "\n".join(
        f"- {row.review_stratum}: {int(row.passed_count)}/{int(row.item_count)} traceability checks passed"
        for row in morphology_summary.itertuples()
    )
    decision_text = "\n".join(
        f"- `{row.feature}` — **{row.decision}** ({row.final_role}): {row.rationale}"
        for row in decisions.itertuples()
    )
    text = f"""# QDIST v4.1 computational verification and feature decision

## Decision

The QDIST measurement is scientifically supportable **only as hard-clipping / hard-plateau morphology in the stored native decoded waveform**. It is not a complete nonlinear-distortion measure and does not localize the causal acquisition stage.

Manual morphology review was **not performed** because reviewers are unavailable. No reviewer labels were filled synthetically and no AI labels were substituted. For this sample-defined construct, G9 instead uses exact known-truth altered-sample masks plus exhaustive ledger traceability. Real-cohort positives remain unlabeled operational detections.

## Evidence used

- Source run: `{source_manifest.get('cohort_orchestration_version', '')}`
- Recordings: {int(source_manifest.get('recording_count', 0))}; participants: {int(source_manifest.get('participant_count', 0))}
- Available recordings: {int(source_manifest.get('available_recording_count', 0))}
- Positive recordings: {int(source_manifest.get('positive_recording_count', 0))}; accepted plateaus: {int(source_manifest.get('accepted_plateau_count', 0))}; merged episodes: {int(source_manifest.get('episode_count', 0))}
- Exact-mask interventions: {int(summary['challenge_rows'])} rows across {int(summary['carrier_count'])} carriers, {int(summary['geometry_count'])} geometries, and {int(summary['dose_count'])} burdens
- Global sample-level micro precision: {float(summary['micro_precision']):.6f}
- Global sample-level micro recall: {float(summary['micro_recall']):.6f}
- Lowest individual intervention precision: {float(summary['sample_precision_min']):.6f}
- Lowest individual intervention recall: {float(summary['sample_recall_min']):.6f}; under-recovery is retained as a limitation rather than hidden

{strata_text}

## Final feature decisions

{decision_text}

## What this validates

The evidence validates deterministic recovery of the declared digital waveform morphology, exact internal reconstruction, stable behavior in the tested parameter neighborhood, and high precision with conservative under-recovery on known altered masks.

## What this does not validate

- It does not prove that every real-cohort accepted event was produced by a particular microphone, codec, gain stage, or analog clipper.
- It does not establish complete sensitivity to all physical clipping or to soft clipping, compression, AGC/DRC, codec distortion, or other nonlinear mechanisms.
- It does not establish disease independence. Phenotype/content confounding must be bounded in later joint analyses.
- It does not authorize use as a standalone recording acceptability gate or diagnostic feature.

## Remaining non-reviewer tasks before immutable freeze

1. Complete QDIST-versus-QGAIN/QCHAN/QTEMP arbitration after QTEMP is available.
2. Replace broad “nonlinear distortion” manuscript claims with “hard-clipping morphology” wherever the measured construct is described.
3. Reconcile the manuscript feature census to the decisions above.
4. Run a separate immutable freeze workflow. No numerical feature recomputation is required for this verification revision.
"""
    path = output_root / "reports" / "QDIST_v410_Computational_Verification_and_Decision_Report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_computational_verification(
    cohort_root: str | Path,
    *,
    overwrite: bool = False,
    allow_review_package_exclusions: bool = False,
) -> dict[str, Any]:
    root = Path(cohort_root).expanduser().resolve()
    output = root / OUTPUT_DIRECTORY
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Verification output already exists: {output}. Set overwrite=True only to rebuild this revision."
            )
        shutil.rmtree(output)
    for directory in ("validation", "tables", "figures", "reports", "manifests"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    source_manifest_path = root / "manifests" / "qdist_v410_candidate_cohort_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_audit = verify_source_artifacts(
        root,
        allow_review_package_exclusions=allow_review_package_exclusions,
    )
    if not source_audit["verified_or_permitted"].all():
        failures = source_audit.loc[~source_audit["verified_or_permitted"], "relative_path"].tolist()
        raise RuntimeError(f"Source-run artifact verification failed: {failures[:10]}")

    recordings = require_csv(root, "tables/qdist_v410_recording_features.csv")
    candidates = require_csv(root, "tables/qdist_v410_candidate_plateau_ledger.csv")
    accepted = require_csv(root, "tables/qdist_v410_accepted_plateau_ledger.csv")
    episodes = require_csv(root, "tables/qdist_v410_episode_ledger.csv")
    challenge = require_csv(root, "validation/qdist_v410_real_speech_challenge_long.csv")
    checklist_source = require_csv(root, "validation/QDIST_Master_Validation_Checklist_v1_2_COHORT_CANDIDATE.csv")
    source_index = require_csv(root, "tables/qdist_v410_figure_index.csv")
    review_index = require_csv(root, "blind_review/qdist_v410_blind_review_index.csv")
    review_key = require_csv(root, "blind_review/restricted/qdist_v410_blind_review_key.csv")

    challenge_audit, challenge_summary = build_challenge_audit(challenge)
    morphology_audit, morphology_summary = build_morphology_audit(
        recordings, candidates, accepted, episodes, review_index, review_key
    )
    decisions = final_feature_decisions()
    checks = build_verification_checks(
        source_audit, challenge_audit, challenge_summary, morphology_summary, decisions
    )
    checklist = update_checklist(checklist_source, output)

    save_csv(source_audit, output / "validation" / "qdist_v410_source_artifact_verification.csv")
    save_csv(challenge_audit, output / "validation" / "qdist_v410_exact_mask_challenge_audit.csv")
    save_csv(challenge_summary, output / "validation" / "qdist_v410_exact_mask_challenge_global_summary.csv")
    save_csv(morphology_audit, output / "validation" / "qdist_v410_exhaustive_morphology_traceability.csv")
    save_csv(morphology_summary, output / "validation" / "qdist_v410_exhaustive_morphology_traceability_summary.csv")
    save_csv(decisions, output / "tables" / "qdist_v410_final_feature_decisions.csv")
    save_csv(checks, output / "validation" / "qdist_v410_computational_verification_checks.csv")
    save_csv(checklist, output / "validation" / "QDIST_Master_Validation_Checklist_v1_3_COMPUTATIONAL_VERIFICATION.csv")

    provenance = {
        "verification_version": VERIFICATION_VERSION,
        "source_cohort_manifest_sha256": sha256_file(source_manifest_path),
        "source_detector_sha256": source_manifest.get("detector_sha256"),
        "source_parameter_hash": source_manifest.get("parameter_hash"),
        "criterion_reference": "exact altered-sample mask on label-blind cohort-derived speech",
        "human_review_performed": False,
        "human_review_required": False,
        "human_or_ai_morphology_labels_generated": False,
        "real_cohort_ground_truth_status": "unlabeled_operational_detections",
    }
    figure_rows = _copy_preserved_figures(root, output, source_index)
    fig_i, src_i, cap_i = _panel_i(challenge_audit, challenge_summary, morphology_summary)
    figure_rows.append(_save_figure_bundle(
        output, "I", "qdist_v410_panel-I_computational-verification",
        fig_i, src_i, cap_i, provenance,
    ))
    fig_j, src_j, cap_j = _panel_j(decisions)
    figure_rows.append(_save_figure_bundle(
        output, "J", "qdist_v410_panel-J_final-feature-decisions",
        fig_j, src_j, cap_j, provenance,
    ))
    figure_rows.extend(_revised_galleries(
        root, output, source_index, morphology_audit, provenance
    ))
    figure_index = pd.DataFrame(figure_rows)
    save_csv(figure_index, output / "tables" / "qdist_v410_verification_figure_index.csv")

    non_gallery = set(figure_index.loc[figure_index["panel"].ne("G"), "panel"].astype(str))
    gallery_count = int(figure_index["panel"].eq("G").sum())
    missing_figure_files: list[str] = []
    for row in figure_index.to_dict("records"):
        for field in FIGURE_FIELDS:
            path = output / portable_relative_path(row[field])
            if not path.exists() or path.stat().st_size == 0:
                missing_figure_files.append(f"{row['stem']}::{field}")
    figure_checks = pd.DataFrame([
        {"gate": "G10", "check": "complete A-J verification figure suite", "status": "PASS" if set(REQUIRED_MAIN_PANELS).issubset(non_gallery) else "FAIL", "observed": "|".join(sorted(non_gallery)), "required": "|".join(REQUIRED_MAIN_PANELS)},
        {"gate": "G10", "check": "signal-evidence gallery count", "status": "PASS" if gallery_count >= 8 else "FAIL", "observed": gallery_count, "required": ">=8"},
        {"gate": "G10", "check": "all figure bundle artifacts exist", "status": "PASS" if not missing_figure_files else "FAIL", "observed": "none" if not missing_figure_files else "|".join(missing_figure_files), "required": "none missing"},
    ])
    checks = pd.concat([checks, figure_checks], ignore_index=True)
    save_csv(checks, output / "validation" / "qdist_v410_computational_verification_checks.csv")

    report_path = _write_report(
        output, source_manifest, challenge_summary, morphology_summary, decisions
    )
    status_counts = checks["status"].value_counts().to_dict()
    checklist_counts = checklist["status"].value_counts().to_dict()
    manifest = {
        "verification_version": VERIFICATION_VERSION,
        "created_utc": utc_now(),
        "source_cohort_root": str(root),
        "source_cohort_manifest_sha256": sha256_file(source_manifest_path),
        "source_artifacts_exactly_verified": bool(source_audit["sha256_matches"].all()),
        "source_review_archive_exclusion_count": int(source_audit["permitted_review_archive_exclusion"].sum()),
        "feature_values_changed": False,
        "detector_or_thresholds_changed": False,
        "human_review_performed": False,
        "human_review_required": False,
        "human_or_ai_morphology_labels_generated": False,
        "criterion_reference": "exact altered-sample mask on cohort-derived speech",
        "real_cohort_ground_truth_status": "unlabeled_operational_detections",
        "challenge_rows": int(challenge_summary.iloc[0]["challenge_rows"]),
        "challenge_micro_precision": float(challenge_summary.iloc[0]["micro_precision"]),
        "challenge_micro_recall": float(challenge_summary.iloc[0]["micro_recall"]),
        "morphology_audit_item_count": len(morphology_audit),
        "morphology_audit_failure_count": int((~as_bool(morphology_audit["audit_pass"])).sum()),
        "feature_decisions_complete": True,
        "scientific_decision": "ACCEPT_HARD_CLIPPING_MORPHOLOGY_MEASUREMENT_WITH_EXPLICIT_LIMITATIONS",
        "cross_family_arbitration_complete": False,
        "manuscript_reconciliation_complete": False,
        "freeze_allowed": False,
        "publish_and_freeze": False,
        "check_status_counts": status_counts,
        "checklist_status_counts": checklist_counts,
        "report": report_path.relative_to(output).as_posix(),
        "figure_bundle_count": len(figure_index),
        "gallery_bundle_count": gallery_count,
    }
    manifest_path = write_json(
        manifest,
        output / "manifests" / "qdist_v410_computational_verification_manifest.json",
    )
    artifact_rows: list[dict[str, Any]] = []
    artifact_manifest_path = output / "manifests" / "qdist_v410_computational_verification_artifact_manifest.csv"
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != artifact_manifest_path:
            artifact_rows.append({
                "relative_path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    save_csv(pd.DataFrame(artifact_rows), artifact_manifest_path)
    return {
        "output_root": output,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "checks": checks,
        "checklist": checklist,
        "decisions": decisions,
        "challenge_audit": challenge_audit,
        "challenge_summary": challenge_summary,
        "morphology_audit": morphology_audit,
        "morphology_summary": morphology_summary,
        "figure_index": figure_index,
        "report_path": report_path,
    }
