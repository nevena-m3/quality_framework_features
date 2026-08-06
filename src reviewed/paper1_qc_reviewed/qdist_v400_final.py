from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FAMILY = "QDIST"
FAMILY_DISPLAY_NAME = "Native-waveform hard-clipping morphology"
SOURCE_MEASUREMENT_VERSION = "qdist-v4.0.0-candidate"
FINAL_MEASUREMENT_VERSION = "qdist-v4.0.0"
ACCEPTANCE_TOKEN = "ACCEPT_QDIST_V400"
FIGURE_PACKAGE_VERSION = "qdist-v4.0.0-figures-v1.0.0"
FINALIZATION_REVISION = "qdist-v4.0.0-finalization-r1"

ANALYSIS_FEATURES = (
    "qdist_hard_clipped_frame_fraction",
    "qdist_hard_clip_event_rate_per_min",
    "qdist_hard_clipped_sample_fraction",
)
DISPLAY_NAMES = {
    ANALYSIS_FEATURES[0]: "Clipped-frame fraction",
    ANALYSIS_FEATURES[1]: "Hard-clip episode rate",
    ANALYSIS_FEATURES[2]: "Clipped-sample fraction",
}
UNITS = {
    ANALYSIS_FEATURES[0]: "fraction of complete 30-ms frames",
    ANALYSIS_FEATURES[1]: "episodes/min",
    ANALYSIS_FEATURES[2]: "fraction of finite channel-samples",
}


@dataclass(frozen=True)
class FinalFeatureDefinition:
    feature: str
    display_name: str
    final_decision: str
    publication_role: str
    default_manuscript_inclusion: bool
    default_joint_model_inclusion: bool
    analysis_priority: str
    interpretation_class: str
    unit: str
    orientation: str
    minimum_support: str
    claim_limit: str
    known_qualifications: str
    standalone_gate_allowed: bool = False
    family_scalar_allowed: bool = False
    missing_value_behavior: str = (
        "NaN only when unavailable, with explicit status and exposure; "
        "valid zero remains zero and is never imputed"
    )


FINAL_FEATURE_DEFINITIONS = (
    FinalFeatureDefinition(
        feature=ANALYSIS_FEATURES[0],
        display_name="Hard-clipped frame fraction",
        final_decision="RETAIN_PRIMARY_RELATED_VIEW",
        publication_role=(
            "primary recording-level prevalence view: fraction of complete native-waveform "
            "frames intersecting at least one accepted hard-clipping plateau"
        ),
        default_manuscript_inclusion=True,
        default_joint_model_inclusion=True,
        analysis_priority="primary_occurrence_burden",
        interpretation_class=(
            "merge-gap-independent frame-prevalence view of one accepted plateau system"
        ),
        unit="fraction",
        orientation=(
            "higher means a larger proportion of complete task-span frames intersect "
            "accepted plateau evidence"
        ),
        minimum_support=(
            "finite native task span with the frozen complete-frame and exposure contract"
        ),
        claim_limit=(
            "not total nonlinear distortion, THD, soft clipping, compression, AGC, codec "
            "distortion, perceptual distortion, or proof of where clipping occurred"
        ),
        known_qualifications=(
            "only 6/519 recordings were positive; positive-part repeatability is not "
            "estimable; frame fraction is a related view rather than independent evidence"
        ),
    ),
    FinalFeatureDefinition(
        feature=ANALYSIS_FEATURES[1],
        display_name="Hard-clip episode rate",
        final_decision="RETAIN_PRIMARY_EVENT_WITH_UNCERTAINTY",
        publication_role=(
            "primary event view: merged accepted hard-clipping episodes per finite native "
            "task-span minute, accompanied by event count and exact Poisson interval"
        ),
        default_manuscript_inclusion=True,
        default_joint_model_inclusion=True,
        analysis_priority="primary_event",
        interpretation_class=(
            "episode-frequency view of the same accepted plateau system"
        ),
        unit="episodes/min",
        orientation="higher means more merged hard-clipping episodes per analyzed minute",
        minimum_support=(
            "finite native task span; event count, exposure, merge gap, and exact Poisson "
            "interval must accompany the rate"
        ),
        claim_limit=(
            "not independent of frame/sample burden and not interpretable without the "
            "frozen 20-ms episode merge rule"
        ),
        known_qualifications=(
            "positive recordings are sparse; 10-ms grouping changed event count in 2/519 "
            "recordings but occurrence remained invariant; delete-one influence is large "
            "when counts are small"
        ),
    ),
    FinalFeatureDefinition(
        feature=ANALYSIS_FEATURES[2],
        display_name="Hard-clipped sample fraction",
        final_decision="RETAIN_SECONDARY_RELATED_VIEW",
        publication_role=(
            "secondary direct channel-sample burden view; report with clipped channel-ms/min"
        ),
        default_manuscript_inclusion=True,
        default_joint_model_inclusion=False,
        analysis_priority="secondary_direct_burden",
        interpretation_class=(
            "accepted plateau channel-sample support divided by finite channel-sample exposure"
        ),
        unit="fraction",
        orientation=(
            "higher means more accepted plateau channel-sample support per finite exposure"
        ),
        minimum_support="finite native task span and verified native channel geometry",
        claim_limit=(
            "not an independent mechanism and not a severity scalar; simultaneous "
            "unregularized use with both other QDIST views is discouraged"
        ),
        known_qualifications=(
            "values are extremely small and should be translated to clipped channel-ms/min; "
            "positive-only ranking is based on six recordings"
        ),
    ),
)

TEN_DOMAIN_DASHBOARD = (
    ("Construct validity", "PASS_WITH_SCOPE_LIMIT", "Measures native-waveform plateau morphology compatible with hard clipping or saturation; not the full nonlinear-distortion construct."),
    ("Estimator validity", "PASS", "All three outputs reconstruct from accepted-plateau and merged-episode ledgers for 519 recordings."),
    ("Implementation validity", "PASS", "Native-rate PCM, native channels, task span, complete frames, hashes, and immutable v3.1.1 baseline are verified."),
    ("Transformation behavior", "PASS_WITH_QUALIFICATION", "Polarity, aligned translation, attenuation after clipping, and lossless PCM behave as specified; resampling/lossy coding can erase plateaus."),
    ("Dose response", "PASS_WITH_QUALIFICATION", "Controlled hard-clipping burden yields ordered frame/sample response and perfect precision; recall is conservative at smallest doses."),
    ("Discriminant validity", "PASS_WITH_SCOPE_LIMIT", "Clean speech, natural extrema, quantization, impulses/noise and moderate smooth saturation are not promoted; causal clipping stage is not identifiable."),
    ("Support and uncertainty", "PASS_WITH_QUALIFICATION", "All 519 are available; Poisson intervals, merge-gap sensitivity, threshold margins and delete-one influence are retained because only 15 episodes occur."),
    ("Reliability and robustness", "PASS_WITH_MAJOR_QUALIFICATION", "Negative agreement is high, positive agreement is 0.40, and positive-part magnitude reliability is not estimable."),
    ("Event verification", "PASS_WITH_QUALIFICATION", "All accepted, rejected and valid-zero items were reviewed label-blind; adjudication is AI-assisted and 13 rejected items remain ambiguous."),
    ("Scientific scope and handoff", "PASS", "No scalar, standalone gate, imputation, complete-distortion claim or independent-view claim is authorized."),
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_decisions_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in FINAL_FEATURE_DEFINITIONS])


def ten_domain_dashboard_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"domain": domain, "final_status": status, "scientific_basis": basis}
            for domain, status, basis in TEN_DOMAIN_DASHBOARD
        ]
    )


def final_gate_summary_frame() -> pd.DataFrame:
    rows = [
        ("G1", "PASS", "Construct, input, signal-view, frozen-source and provenance contracts are complete."),
        ("G2", "PASS", "All three features reconstruct for 519/519 recordings within <=2e-15 CSV round-trip tolerance."),
        ("G3", "PASS_WITH_QUALIFICATION", "Native PCM transformations behave as specified; resampling and lossy encoding can erase evidence."),
        ("G4", "PASS_WITH_QUALIFICATION", "Controlled hard clipping has ordered response and perfect precision; smallest burdens are conservatively detected."),
        ("G5", "PASS_WITH_SCOPE_LIMIT", "Specificity controls pass within the hard-plateau construct; broad nonlinear-distortion claims remain prohibited."),
        ("G6", "PASS_WITH_QUALIFICATION", "Occurrence is robust to declared variants, but threshold-adjacent events and sparse rate uncertainty must remain visible."),
        ("G7", "PASS_WITH_QUALIFICATION", "All recordings are available and empirical behavior is plausible, but only six recordings are positive."),
        ("G8", "PASS_WITH_MAJOR_QUALIFICATION", "Outputs are related views; positive-part repeatability is not estimable and positive occurrence agreement is limited."),
        ("G9", "PASS_WITH_QUALIFICATION", "The complete 60-item review meets point-estimate gates; labels are AI-assisted and 13 rejected candidates are ambiguous."),
        ("G10", "PASS", "Feature-specific decisions, figures, support-aware handoff, passports and atomic freeze contracts are complete."),
    ]
    return pd.DataFrame(rows, columns=["gate", "final_status", "final_basis"])


def final_checklist_frame(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    statuses = final_gate_summary_frame().set_index("gate")["final_status"].to_dict()
    frame["final_gate_status"] = frame["gate"].map(statuses)
    frame["final_item_status"] = frame["final_gate_status"]
    frame["finalization_revision"] = FINALIZATION_REVISION
    frame["scientific_review_complete"] = True
    return frame


def analysis_values_equal(left: pd.DataFrame, right: pd.DataFrame, tolerance: float = 2e-15) -> bool:
    key = "logical_recording_id"
    left = left[[key, *ANALYSIS_FEATURES]].sort_values(key).reset_index(drop=True)
    right = right[[key, *ANALYSIS_FEATURES]].sort_values(key).reset_index(drop=True)
    if not left[key].equals(right[key]):
        return False
    for feature in ANALYSIS_FEATURES:
        a = pd.to_numeric(left[feature], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right[feature], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return False
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.any() and float(np.max(np.abs(a[mask] - b[mask]))) > tolerance:
            return False
    return True


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    low = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return low, high


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _save_bundle(root: Path, stem: str, figure: plt.Figure, source: pd.DataFrame, caption: str, source_tables: Sequence[str]) -> None:
    figures = Path(root) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    base = figures / stem
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    source.to_csv(base.with_suffix(".source.csv"), index=False)
    base.with_suffix(".caption.md").write_text(caption.strip() + "\n", encoding="utf-8")
    base.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "figure_stem": stem,
                "family": FAMILY,
                "measurement_version": FINAL_MEASUREMENT_VERSION,
                "figure_package_version": FIGURE_PACKAGE_VERSION,
                "source_tables": list(source_tables),
                "feature_values_recomputed": False,
                "audit_summary_recomputed_from_saved_tables": True,
                "finalization_revision": FINALIZATION_REVISION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _positive_with_labels(features: pd.DataFrame) -> pd.DataFrame:
    local = features.loc[features["qdist_positive"].astype(bool)].copy()
    local = local.sort_values("qdist_hard_clip_event_rate_per_min", ascending=False).reset_index(drop=True)
    local["display_id"] = [f"R{i + 1}" for i in range(len(local))]
    return local


def regenerate_audited_figures(final_root: Path) -> list[str]:
    root = Path(final_root)
    features = pd.read_csv(root / "tables" / "qdist_v400_recording_features.csv")
    param = pd.read_csv(root / "validation" / "qdist_v400_parameter_sensitivity_summary.csv")
    morph = pd.read_csv(root / "validation" / "qdist_v400_morphology_margin_summary.csv")
    merge = pd.read_csv(root / "validation" / "qdist_v400_merge_gap_sensitivity_summary.csv")
    deletion = pd.read_csv(root / "validation" / "qdist_v400_deletion_influence_summary.csv")
    repeat = pd.read_csv(root / "validation" / "qdist_v400_repeated_recording_summary.csv")
    redundancy = pd.read_csv(root / "validation" / "qdist_v400_related_view_redundancy.csv")
    weighting = pd.read_csv(root / "validation" / "qdist_v400_weighting_summary.csv")
    participant = pd.read_csv(root / "validation" / "qdist_v400_participant_summary.csv")
    adjud = pd.read_csv(root / "validation" / "qdist_v400_event_adjudication_summary.csv")
    positive = _positive_with_labels(features)
    regenerated: list[str] = []

    # D2
    d2 = pd.concat(
        [
            pd.DataFrame({"section": "occurrence", "category": ["Valid zero", "Positive"], "value": [513, 6]}),
            positive[["display_id", "logical_recording_id", "participant_id", "qdist_accepted_plateau_count", "qdist_hard_clip_event_count", "qdist_hard_clip_event_rate_per_min", "qdist_clipped_channel_ms_per_min"]].assign(section="positive_recordings"),
        ],
        ignore_index=True,
        sort=False,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), constrained_layout=True)
    axes[0].bar(["Valid zero", "Positive"], [513, 6]); axes[0].set_ylabel("Recordings"); axes[0].set_title("Recording occurrence"); axes[0].text(1, 6, "6/519 (1.16%)", ha="center", va="bottom")
    axes[1].bar(positive["display_id"], positive["qdist_accepted_plateau_count"]); axes[1].set_ylabel("Accepted plateaus"); axes[1].set_title("Positive recordings")
    axes[2].bar(positive["display_id"], positive["qdist_hard_clip_event_count"]); axes[2].set_ylabel("Merged episodes"); axes[2].set_title("Frozen 20-ms grouping")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    _save_bundle(root, "D2_occurrence_sparsity", fig, d2, "Panel D2. QDIST was available for all 519 recordings but positive in only six. Short display identifiers map to full identifiers in the source table. Plateau and episode counts are related summaries of the same detector output.", ["tables/qdist_v400_recording_features.csv"])
    regenerated.append("D2_occurrence_sparsity")

    # E1
    labels = {
        "candidate_floor_20": "Candidate floor 0.20", "candidate_floor_30": "Candidate floor 0.30",
        "recording_floor_40": "Recording floor 0.40", "recording_floor_60": "Recording floor 0.60",
        "low_level_support_relaxed": "Low-level support relaxed", "low_level_support_strict": "Low-level support strict",
        "plateau_min_3": "Minimum plateau 3", "plateau_min_5": "Minimum plateau 5",
        "edge_support_6": "Edge support 6", "edge_support_10": "Edge support 10",
        "merge_10ms": "Merge gap 10 ms", "merge_30ms": "Merge gap 30 ms",
    }
    p = param.copy(); p["display_variant"] = p["variant"].map(labels).fillna(p["variant"]); p = p.sort_values("zero_positive_class_agreement")
    m = morph.loc[morph["stratum"].eq("accepted")].copy(); m["display_criterion"] = m["criterion"].str.replace("_", " "); m = m.sort_values("minimum_margin")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), constrained_layout=True)
    y = np.arange(len(p)); axes[0].barh(y, 1 - p["zero_positive_class_agreement"]); axes[0].set_yticks(y, p["display_variant"]); axes[0].set_xlabel("Occurrence disagreement fraction"); axes[0].set_title("Occurrence sensitivity")
    axes[1].barh(y, p["p90_clipped_duration_change_ms_per_min"]); axes[1].set_yticks(y, p["display_variant"]); axes[1].set_xlabel("90th-percentile absolute change (channel-ms/min)"); axes[1].set_title("Positive-burden sensitivity")
    ym = np.arange(len(m)); axes[2].barh(ym, m["minimum_margin"]); axes[2].set_yticks(ym, m["display_criterion"]); axes[2].axvline(0, linewidth=.8); axes[2].set_xlabel("Minimum signed margin"); axes[2].set_title("Accepted-event threshold margins")
    for ax in axes: ax.grid(axis="x", alpha=.25)
    _save_bundle(root, "E1_detector_parameter_sensitivity", fig, pd.concat([p.assign(section="parameter_sensitivity"), m.assign(section="accepted_margins")], ignore_index=True, sort=False), "Panel E1. Detector-parameter sensitivity in the deterministic 46-recording set. Occurrence changed in one recording under the 0.40 recording floor and minimum-five-sample variants. Accepted candidates remain on the allowed side of all frozen criteria, with threshold-adjacent discrete margins retained.", ["validation/qdist_v400_parameter_sensitivity_summary.csv", "validation/qdist_v400_morphology_margin_summary.csv"])
    regenerated.append("E1_detector_parameter_sensitivity")

    # E2
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), constrained_layout=True)
    x = merge["merge_gap_ms"]
    axes[0].plot(x, merge["positive_recording_count"], marker="o"); axes[0].set_ylabel("Positive recordings"); axes[0].set_title("Occurrence")
    axes[1].plot(x, merge["event_count_changed_fraction"], marker="o"); axes[1].set_ylabel("Fraction with changed event count"); axes[1].set_title("Event-count sensitivity")
    axes[2].plot(x, merge["maximum_absolute_rate_change"], marker="o"); axes[2].set_ylabel("Maximum absolute rate change (episodes/min)"); axes[2].set_title("Rate sensitivity")
    for ax in axes: ax.axvline(20, linestyle="--", linewidth=1, label="Frozen 20 ms"); ax.set_xlabel("Episode merge gap (ms)"); ax.grid(alpha=.25)
    axes[0].legend()
    _save_bundle(root, "E2_episode_grouping_sensitivity", fig, merge, "Panel E2. Occurrence is invariant across 10-, 20-, 30- and 50-ms merge gaps. The 10-ms rule splits episodes in two recordings; 20–50 ms are identical in this cohort. The frozen 20-ms rule is retained.", ["validation/qdist_v400_merge_gap_sensitivity_summary.csv"])
    regenerated.append("E2_episode_grouping_sensitivity")

    # E3
    poisson = positive.copy()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), constrained_layout=True)
    axes[0].bar(poisson["display_id"], poisson["qdist_clipped_channel_ms_per_min"]); axes[0].set_ylabel("Clipped channel-ms/min"); axes[0].set_title("Absolute accepted burden")
    axes[1].bar(deletion["deletion_type"], deletion["maximum_event_rate_absolute_change"]); axes[1].set_ylabel("Maximum absolute event-rate change (episodes/min)"); axes[1].set_title("Delete-one influence")
    center = poisson["qdist_hard_clip_event_rate_per_min"].to_numpy(float); low = poisson["qdist_hard_clip_event_rate_ci95_low_per_min"].to_numpy(float); high = poisson["qdist_hard_clip_event_rate_ci95_high_per_min"].to_numpy(float)
    axes[2].errorbar(poisson["display_id"], center, yerr=np.vstack([center-low, high-center]), fmt="o", capsize=4); axes[2].set_ylabel("Episodes/min"); axes[2].set_title("Exact Poisson 95% intervals")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    _save_bundle(root, "E3_sparse_burden_deletion_poisson", fig, pd.concat([poisson.assign(section="positive_recordings"), deletion.assign(section="deletion_influence")], ignore_index=True, sort=False), "Panel E3. Sparse-burden diagnostics for the six positive recordings. Physical clipped channel-time, delete-one leverage and exact Poisson intervals show why count and uncertainty must accompany event rate.", ["tables/qdist_v400_recording_features.csv", "validation/qdist_v400_deletion_influence_summary.csv"])
    regenerated.append("E3_sparse_burden_deletion_poisson")

    # F
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True); rows = []
    for col, feature in enumerate(ANALYSIS_FEATURES):
        values = pd.to_numeric(features[feature], errors="coerce"); pos = np.sort(values.loc[values > 0].to_numpy(float)); zeros = int((values == 0).sum())
        axes[0, col].bar(["Valid zero", "Positive"], [zeros, len(pos)]); axes[0, col].set_ylabel("Recordings"); axes[0, col].set_title(DISPLAY_NAMES[feature]); axes[0, col].text(1, len(pos), f"n={len(pos)}", ha="center", va="bottom")
        ranks = np.arange(len(pos)); axes[1, col].scatter(ranks, pos, s=45); axes[1, col].set_xticks(ranks, [f"P{i+1}" for i in ranks]); axes[1, col].set_ylabel(UNITS[feature]); axes[1, col].set_title("Positive-only measured values"); axes[1, col].grid(axis="y", alpha=.25)
        rows.append({"feature": feature, "section": "count", "valid_zero_n": zeros, "positive_n": len(pos)})
        rows.extend({"feature": feature, "section": "positive_value", "positive_rank": i+1, "value": value} for i, value in enumerate(pos))
    _save_bundle(root, "F_empirical_distributions", fig, pd.DataFrame(rows), "Panel F. QDIST is zero-inflated by design: 513/519 recordings are valid zero and six are positive for each related view. The lower row displays all six positive values instead of allowing the zero mass to conceal their distribution.", ["tables/qdist_v400_recording_features.csv", "validation/qdist_v400_empirical_feature_summary.csv"])
    regenerated.append("F_empirical_distributions")

    # H1
    occ = repeat.loc[repeat["metric"].eq("occurrence")].iloc[0]
    pairs = pd.DataFrame({"category": ["Both zero", "First only", "Second only", "Both positive"], "participants": [int(occ["both_zero_n00"]), int(occ["first_only_n10"]), int(occ["second_only_n01"]), int(occ["both_positive_n11"])]})
    agreement_rows = []
    specs = [
        ("Overall", int(round(float(occ["overall_agreement"]) * int(occ["participant_pair_count"]))), int(occ["participant_pair_count"])),
        ("Positive", 2 * int(occ["both_positive_n11"]), 2 * int(occ["both_positive_n11"]) + int(occ["first_only_n10"]) + int(occ["second_only_n01"])),
        ("Negative", 2 * int(occ["both_zero_n00"]), 2 * int(occ["both_zero_n00"]) + int(occ["first_only_n10"]) + int(occ["second_only_n01"])),
    ]
    for label, success, total in specs:
        low, high = wilson_interval(success, total); agreement_rows.append({"agreement": label, "successes": success, "trials": total, "estimate": success/total, "wilson95_low": low, "wilson95_high": high})
    agreement = pd.DataFrame(agreement_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    axes[0].bar(pairs["category"], pairs["participants"]); axes[0].tick_params(axis="x", rotation=25); axes[0].set_ylabel("Participants"); axes[0].set_title("First-two-recording occurrence pairs")
    xx = np.arange(len(agreement)); est = agreement["estimate"].to_numpy(float); axes[1].errorbar(xx, est, yerr=np.vstack([est-agreement["wilson95_low"], agreement["wilson95_high"]-est]), fmt="o", capsize=5); axes[1].set_xticks(xx, agreement["agreement"]); axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Agreement with Wilson 95% interval"); axes[1].set_title("Occurrence agreement"); axes[1].text(.02, .05, "Positive-part magnitude reliability: not estimable\n(only one pair positive at both visits)", transform=axes[1].transAxes, va="bottom")
    _save_bundle(root, "H1_repeated_recording_persistence", fig, pd.concat([pairs.assign(section="pair_counts"), agreement.assign(section="agreement")], ignore_index=True, sort=False), "Panel H1. Among 158 first-two recording pairs, overall and negative agreement are high because QDIST is sparse, while positive agreement is 0.40. Only one pair is positive twice, so positive-part correlation or ICC is not estimable.", ["validation/qdist_v400_repeated_recording_summary.csv"])
    regenerated.append("H1_repeated_recording_persistence")

    # H2
    red = redundancy.copy(); red["pair"] = ["Frame vs event", "Frame vs sample", "Event vs sample"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    axes[0].bar(red["pair"], red["all_recordings_spearman_rho"]); axes[0].set_ylim(0, 1.03); axes[0].tick_params(axis="x", rotation=20); axes[0].set_title("All recordings (n=519; zero-inflated)")
    axes[1].bar(red["pair"], red["positive_recordings_spearman_rho"]); axes[1].set_ylim(0, 1.03); axes[1].tick_params(axis="x", rotation=20); axes[1].set_title("Positive recordings only (n=6)")
    for ax in axes: ax.set_ylabel("Spearman rho"); ax.grid(axis="y", alpha=.25)
    _save_bundle(root, "H2_related_view_redundancy", fig, red, "Panel H2. The three outputs are related views of one plateau-and-episode system. All-recording correlations are dominated by 513 shared valid zeros; positive-only correlations are descriptive and based on six recordings. No scalar or independence claim is justified.", ["validation/qdist_v400_related_view_redundancy.csv"])
    regenerated.append("H2_related_view_redundancy")

    # H3
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    xx = np.arange(len(weighting)); centers = weighting["positive_fraction"].to_numpy(float); axes[0].errorbar(xx, centers, yerr=np.vstack([centers-weighting["wilson95_low"], weighting["wilson95_high"]-centers]), fmt="o", capsize=5); axes[0].set_xticks(xx, ["Recording weighted", "Participant ever-positive"]); axes[0].set_ylabel("Positive fraction with Wilson 95% interval"); axes[0].set_title("Recording and participant weighting")
    counts = participant["positive_recording_count"].value_counts().sort_index(); axes[1].bar(counts.index.astype(str), counts.values); axes[1].set_xlabel("Positive recordings per participant"); axes[1].set_ylabel("Participants"); axes[1].set_title("Participant clustering")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    _save_bundle(root, "H3_participant_weighting_clustering", fig, pd.concat([weighting.assign(section="weighting"), counts.rename_axis("positive_recording_count").reset_index(name="participant_count").assign(section="participant_counts")], ignore_index=True, sort=False), "Panel H3. Six of 519 recordings are positive (1.16%), while five of 224 participants are ever positive (2.23%). Four participants have one positive recording and one has two; uncertainty reflects the small counts.", ["validation/qdist_v400_weighting_summary.csv", "validation/qdist_v400_participant_summary.csv"])
    regenerated.append("H3_participant_weighting_clustering")

    # I
    rows = []
    for row in adjud.itertuples(index=False):
        successes = int(row.hard_clip_positive_n); total = int(row.adjudicable_n); low, high = clopper_pearson(successes, total)
        rows.append({"stratum": row.stratum, "review_item_count": int(row.review_item_count), "adjudicable_n": total, "ambiguous_n": int(row.ambiguous_n), "hard_clip_positive_n": successes, "fraction": float(row.hard_clip_positive_fraction), "cp95_low": low, "cp95_high": high})
    ai = pd.DataFrame(rows)
    order = ["rejected_candidate", "valid_zero", "accepted_plateau"]; ai["order"] = ai["stratum"].map({x:i for i,x in enumerate(order)}); ai = ai.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), constrained_layout=True)
    axes[0].bar(ai["stratum"], ai["adjudicable_n"], label="Adjudicable"); axes[0].bar(ai["stratum"], ai["ambiguous_n"], bottom=ai["adjudicable_n"], label="Ambiguous"); axes[0].set_ylabel("Review items"); axes[0].set_title("Complete 60-item review"); axes[0].tick_params(axis="x", rotation=20); axes[0].legend()
    xx = np.arange(len(ai)); center = ai["fraction"].to_numpy(float); axes[1].errorbar(xx, center, yerr=np.vstack([center-ai["cp95_low"], ai["cp95_high"]-center]), fmt="o", capsize=5); axes[1].set_xticks(xx, ai["stratum"], rotation=20); axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Hard-clip-positive fraction among adjudicable items"); axes[1].set_title("Blinded morphology adjudication"); axes[1].axhline(.90, linestyle="--", linewidth=1); axes[1].axhline(.20, linestyle=":", linewidth=1); axes[1].text(.02, .06, "AI-assisted review; exact intervals shown\n13/20 rejected candidates were ambiguous", transform=axes[1].transAxes)
    _save_bundle(root, "I_event_verification", fig, ai.drop(columns="order"), "Panel I. Complete label-blind review of 30 accepted plateaus, 20 rejected candidates and 10 valid-zero controls. Point-estimate gates are met. Exact Clopper–Pearson intervals and 13 ambiguous rejected candidates are shown. Review is AI-assisted and is not independent human ground truth.", ["validation/qdist_v400_event_adjudication_summary.csv", "tables/qdist_v400_event_review_index.csv"])
    regenerated.append("I_event_verification")

    # J
    status_counts = pd.DataFrame({"status": ["available_no_events", "available_events"], "recordings": [513, 6]})
    contract = {
        "feature_values_retained": True, "status_retained": True, "exposure_retained": True,
        "event_count_retained": True, "poisson_interval_retained": True, "native_geometry_retained": True,
        "version_and_hashes_retained": True, "missing_values_imputed": False,
        "family_scalar_constructed": False, "standalone_gate_allowed": False,
        "complete_nonlinear_distortion_claim_allowed": False, "related_views_declared": True,
    }
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), constrained_layout=True)
    axes[0].bar(status_counts["status"], status_counts["recordings"]); axes[0].tick_params(axis="x", rotation=20); axes[0].set_ylabel("Recordings"); axes[0].set_title("ML export status counts")
    axes[1].axis("off"); axes[1].set_title("Support-aware handoff contract"); axes[1].text(0, .98, "\n".join(f"{key}: {value}" for key,value in contract.items()), va="top", family="monospace")
    jsource = pd.concat([status_counts.assign(section="status_counts"), pd.DataFrame([contract]).assign(section="contract")], ignore_index=True, sort=False)
    _save_bundle(root, "J_ml_handoff", fig, jsource, "Panel J. The ML handoff retains feature values, status, exposure, event counts, Poisson intervals, native geometry, version and hashes. It does not impute missing values, construct a scalar, allow a standalone gate or authorize a complete nonlinear-distortion claim.", ["tables/qdist_v400_ml_interface.csv", "tables/qdist_v400_recording_features.csv"])
    regenerated.append("J_ml_handoff")
    return regenerated


def regenerate_event_review_pngs(final_root: Path) -> int:
    root = Path(final_root)
    index = pd.read_csv(root / "tables" / "qdist_v400_event_review_index.csv")
    count = 0
    for row in index.itertuples(index=False):
        stem = f"qdist_review_{row.review_item_id}"
        source_path = root / "event_review" / f"{stem}.source.csv"
        data = pd.read_csv(source_path)
        wave = data.loc[data["view"].eq("waveform")].copy()
        pcm = data.loc[data["view"].eq("pcm_derivative")].copy()
        occ = data.loc[data["view"].eq("amplitude_distribution")].copy()
        spec = data.loc[data["view"].eq("spectrogram")].copy()
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
        if len(wave):
            xcol = "time_ms" if "time_ms" in wave else wave.columns[1]; ycol = "amplitude" if "amplitude" in wave else wave.columns[-1]
            axes[0,0].plot(pd.to_numeric(wave[xcol], errors="coerce"), pd.to_numeric(wave[ycol], errors="coerce"), linewidth=.8)
        axes[0,0].set_title("Waveform around review center"); axes[0,0].set_xlabel("Time (ms)"); axes[0,0].set_ylabel("Native decoded amplitude")
        if len(pcm):
            xcol = "time_ms" if "time_ms" in pcm else pcm.columns[1]
            amp = "amplitude" if "amplitude" in pcm else None; diff = "first_difference" if "first_difference" in pcm else None
            if amp: axes[0,1].plot(pd.to_numeric(pcm[xcol], errors="coerce"), pd.to_numeric(pcm[amp], errors="coerce"), linewidth=1)
            if diff:
                ax2 = axes[0,1].twinx(); ax2.plot(pd.to_numeric(pcm[xcol], errors="coerce"), pd.to_numeric(pcm[diff], errors="coerce"), alpha=.45, linewidth=.7); ax2.set_ylabel("First difference")
        axes[0,1].set_title("PCM-code and first-difference context"); axes[0,1].set_xlabel("Time (ms)"); axes[0,1].set_ylabel("Amplitude")
        if len(occ):
            xcol = "amplitude_bin_center" if "amplitude_bin_center" in occ else ("amplitude" if "amplitude" in occ else occ.columns[1]); ycol = "count" if "count" in occ else occ.columns[-1]
            axes[1,0].bar(pd.to_numeric(occ[xcol], errors="coerce"), pd.to_numeric(occ[ycol], errors="coerce"), width=0.01)
            axes[1,0].set_yscale("log")
        axes[1,0].set_title("Local amplitude occupancy"); axes[1,0].set_xlabel("Amplitude"); axes[1,0].set_ylabel("Count (log)")
        if len(spec):
            time_col = "time_ms" if "time_ms" in spec else "time_sec" if "time_sec" in spec else None
            freq_col = "frequency_hz" if "frequency_hz" in spec else None
            value_col = "power_db" if "power_db" in spec else "magnitude_db" if "magnitude_db" in spec else None
            if time_col and freq_col and value_col:
                pivot = spec.pivot_table(index=freq_col, columns=time_col, values=value_col, aggfunc="mean")
                axes[1,1].imshow(pivot.to_numpy(), origin="lower", aspect="auto", extent=[float(pivot.columns.min()), float(pivot.columns.max()), float(pivot.index.min())/1000, float(pivot.index.max())/1000])
        axes[1,1].set_title("Local spectrogram"); axes[1,1].set_xlabel("Time (ms)"); axes[1,1].set_ylabel("Frequency (kHz)")
        fig.suptitle(f"QDIST {row.stratum} | {row.review_label}\n{row.review_item_id}", fontsize=10)
        output = root / "event_review" / f"{stem}.png"
        legacy = root / "event_review" / f"{stem}.legacy.png"
        if output.exists() and not legacy.exists(): shutil.copy2(output, legacy)
        fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig); count += 1
    return count


def _local_artifact(root: Path, value: object, folder: str, stem: str, extension: str) -> str:
    if value is not None and not (isinstance(value, float) and np.isnan(value)):
        name = Path(str(value).replace("\\", "/")).name
    else:
        name = stem + extension
    return f"{folder}/{name}"


def standardized_figure_index(final_root: Path) -> pd.DataFrame:
    root = Path(final_root)
    source = pd.read_csv(root / "tables" / "qdist_v400_figure_index.csv")
    rows = []
    for row in source.to_dict("records"):
        stem = str(row["stem"])
        local = {"panel": str(row["panel"]), "stem": stem, "bundle_role": row.get("bundle_role", "")}
        for column, ext in [("png", ".png"), ("svg", ".svg"), ("pdf", ".pdf"), ("source_csv", ".source.csv"), ("caption", ".caption.md"), ("provenance", ".provenance.json")]:
            local[column] = _local_artifact(root, row.get(column), "figures", stem, ext)
        if str(row["panel"]) == "G":
            local["audio_wav"] = _local_artifact(root, row.get("audio_wav"), "event_review", stem.replace("G_01_", "qdist_review_").replace("G_02_", "qdist_review_").replace("G_03_", "qdist_review_").replace("G_04_", "qdist_review_").replace("G_05_", "qdist_review_").replace("G_06_", "qdist_review_").replace("G_07_", "qdist_review_").replace("G_08_", "qdist_review_"), ".wav")
        else:
            local["audio_wav"] = ""
        rows.append(local)
    index = pd.DataFrame(rows)
    for row in index.to_dict("records"):
        for column in ["png", "svg", "pdf", "source_csv", "caption", "provenance"]:
            path = root / row[column]
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing final figure artifact: {path}")
    index.to_csv(root / "figures" / "qdist_v400_standardized_figure_index.csv", index=False)
    return index


def feature_passport_text(definition: FinalFeatureDefinition) -> str:
    return f"""# {definition.display_name}\n\n- **Feature:** `{definition.feature}`\n- **Family:** {FAMILY} - {FAMILY_DISPLAY_NAME}\n- **Measurement version:** {FINAL_MEASUREMENT_VERSION}\n- **Final decision:** {definition.final_decision}\n- **Publication role:** {definition.publication_role}\n- **Default manuscript inclusion:** {definition.default_manuscript_inclusion}\n- **Default simultaneous model inclusion:** {definition.default_joint_model_inclusion}\n- **Unit:** {definition.unit}\n- **Orientation:** {definition.orientation}\n- **Interpretation:** {definition.interpretation_class}\n- **Minimum support:** {definition.minimum_support}\n- **Claim boundary:** {definition.claim_limit}\n- **Known qualifications:** {definition.known_qualifications}\n- **Missingness:** {definition.missing_value_behavior}\n- **Standalone reject gate allowed:** No\n- **Family scalar/composite:** Prohibited\n"""


def _validate_event_review(source_root: Path) -> None:
    root = Path(source_root)
    index = pd.read_csv(root / "tables" / "qdist_v400_event_review_index.csv")
    if len(index) != 60 or not index["all_required_views_present"].astype(bool).all() or not index["selection_label_blind"].astype(bool).all():
        raise ValueError("Event-review index contract failed")
    required = {"waveform", "pcm_derivative", "amplitude_distribution", "spectrogram", "audio_excerpt"}
    for row in index.itertuples(index=False):
        stem = f"qdist_review_{row.review_item_id}"
        source = root / "event_review" / f"{stem}.source.csv"; wav = root / "event_review" / f"{stem}.wav"
        if not source.exists() or not wav.exists(): raise FileNotFoundError(f"Review source/audio missing: {stem}")
        data = pd.read_csv(source)
        if not required.issubset(set(data["view"].astype(str))): raise ValueError(f"Five-view contract failed: {stem}")
    adjud = pd.read_csv(root / "validation" / "qdist_v400_event_adjudication_summary.csv").set_index("stratum")
    if float(adjud.loc["accepted_plateau", "hard_clip_positive_fraction"]) < .90: raise ValueError("Accepted-event gate failed")
    if float(adjud.loc["rejected_candidate", "hard_clip_positive_fraction"]) > .20: raise ValueError("Rejected-candidate gate failed")
    if float(adjud.loc["valid_zero", "hard_clip_positive_fraction"]) != 0: raise ValueError("Valid-zero gate failed")


def finalize_candidate(*, source_root: Path, final_root: Path, scientific_review_decision: str, scientific_reviewer: str, scientific_review_rationale: str, provenance_files: Iterable[Path] = ()) -> dict:
    source_root = Path(source_root); final_root = Path(final_root)
    source_manifest_path = source_root / "manifests" / "qdist_v400_cohort_candidate_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    required = {
        "candidate_only": True, "accepted_preflight": True, "preflight_blocking_checks_pass": True,
        "package_tests_passed": True, "cohort_standardization_completed": True, "cohort_evidence_complete": True,
        "recording_count": 519, "participant_count": 224, "available_recording_count": 519,
        "positive_recording_count": 6, "valid_zero_recording_count": 513, "candidate_plateau_count": 861,
        "accepted_plateau_count": 30, "episode_count": 15, "event_review_item_count": 60,
        "event_review_error_count": 0, "event_review_all_five_views": True,
        "main_figure_bundle_count": 15, "gallery_bundle_count": 8, "figure_bundle_count": 23,
        "required_panels_complete": True, "panel_i_status": "APPLICABLE_complete_event_verification",
        "numerical_equivalence_to_qdist_v311": True, "feature_values_recomputed": False,
        "family_scalar_constructed": False, "standalone_gate_allowed": False,
        "complete_nonlinear_distortion_claim_allowed": False, "missing_values_imputed": False,
    }
    for key, expected in required.items():
        if source_manifest.get(key) != expected: raise ValueError(f"Source manifest mismatch for {key}: {source_manifest.get(key)!r}")
    checks = pd.read_csv(source_root / "validation" / "qdist_v400_cohort_checks.csv")
    if not checks["passed"].astype(bool).all(): raise ValueError("Source cohort checks contain failures")
    errors = pd.read_csv(source_root / "audit" / "qdist_v400_event_review_errors.csv")
    if len(errors): raise ValueError("Event-review error table is nonempty")
    _validate_event_review(source_root)
    if scientific_review_decision != ACCEPTANCE_TOKEN: raise ValueError(f"Decision must equal {ACCEPTANCE_TOKEN}")
    if not str(scientific_reviewer).strip(): raise ValueError("Scientific reviewer required")
    if final_root.exists(): shutil.rmtree(final_root)
    shutil.copytree(source_root, final_root)

    source_features = pd.read_csv(source_root / "tables" / "qdist_v400_recording_features.csv")
    final_features = pd.read_csv(final_root / "tables" / "qdist_v400_recording_features.csv")
    if not analysis_values_equal(source_features, final_features): raise ValueError("Initial copied features differ")
    source_hash = sha256_file(source_root / "tables" / "qdist_v400_recording_features.csv")
    final_features["qdist_measurement_version"] = FINAL_MEASUREMENT_VERSION
    final_features.to_csv(final_root / "tables" / "qdist_v400_recording_features.csv", index=False)
    try: final_features.to_parquet(final_root / "tables" / "qdist_v400_recording_features.parquet", index=False)
    except Exception: pass
    ml = pd.read_csv(final_root / "tables" / "qdist_v400_ml_interface.csv"); ml["qdist_measurement_version"] = FINAL_MEASUREMENT_VERSION; ml["qdist_finalization_revision"] = FINALIZATION_REVISION; ml.to_csv(final_root / "tables" / "qdist_v400_ml_interface.csv", index=False)
    try: ml.to_parquet(final_root / "tables" / "qdist_v400_ml_interface.parquet", index=False)
    except Exception: pass

    decisions = final_decisions_frame(); dashboard = ten_domain_dashboard_frame(); gates = final_gate_summary_frame()
    decisions.to_csv(final_root / "tables" / "qdist_v400_feature_registry.csv", index=False)
    decisions[["feature", "display_name", "final_decision", "publication_role", "unit", "orientation"]].to_csv(final_root / "tables" / "qdist_v400_analysis_features.csv", index=False)
    decisions.to_csv(final_root / "validation" / "qdist_v400_g10_feature_decisions.csv", index=False)
    dashboard.to_csv(final_root / "validation" / "qdist_v400_ten_domain_dashboard.csv", index=False)
    gates.to_csv(final_root / "validation" / "qdist_v400_gate_summary_final.csv", index=False)

    provenance_files = [Path(p) for p in provenance_files]
    checklist_path = next((p for p in provenance_files if p.name == "QDIST_Validation_Checklist_v1_0.csv"), None)
    if checklist_path and checklist_path.exists():
        checklist = pd.read_csv(checklist_path)
    else:
        checklist = final_checklist_frame(pd.read_csv(final_root / "validation" / "qdist_v400_cohort_checks.csv"))
    checklist.to_csv(final_root / "validation" / "qdist_v400_checklist_final.csv", index=False)

    regenerated = regenerate_audited_figures(final_root)
    event_png_count = regenerate_event_review_pngs(final_root)
    figure_index = standardized_figure_index(final_root)

    passports = final_root / "feature_passports"; passports.mkdir(exist_ok=True)
    for definition in FINAL_FEATURE_DEFINITIONS: (passports / f"{definition.feature}.md").write_text(feature_passport_text(definition), encoding="utf-8")
    provenance = final_root / "provenance"; provenance.mkdir(exist_ok=True)
    for path in provenance_files:
        if path.exists(): shutil.copy2(path, provenance / path.name)

    final_features_check = pd.read_csv(final_root / "tables" / "qdist_v400_recording_features.csv")
    if not analysis_values_equal(source_features, final_features_check): raise ValueError("Finalization changed analysis values")
    manifest = {
        "measurement_version": FINAL_MEASUREMENT_VERSION,
        "source_measurement_version": SOURCE_MEASUREMENT_VERSION,
        "legacy_measurement_version": source_manifest.get("legacy_measurement_version"),
        "finalization_revision": FINALIZATION_REVISION,
        "figure_package_version": FIGURE_PACKAGE_VERSION,
        "freeze_status": "ready_for_atomic_freeze",
        "freeze_allowed": True,
        "scientific_review_decision": scientific_review_decision,
        "scientific_reviewer": scientific_reviewer,
        "scientific_review_rationale": scientific_review_rationale,
        "recording_count": 519, "participant_count": 224, "available_recording_count": 519,
        "positive_recording_count": 6, "valid_zero_recording_count": 513,
        "candidate_plateau_count": 861, "accepted_plateau_count": 30, "episode_count": 15,
        "event_review_item_count": 60, "event_review_standardized_png_count": int(event_png_count),
        "event_review_adjudication_type": "AI-assisted blinded morphology review",
        "event_review_independent_human_ground_truth": False,
        "event_review_rejected_ambiguous_count": 13,
        "positive_part_repeatability_estimable": False,
        "numerical_equivalence_to_cohort_candidate": True,
        "numerical_equivalence_to_qdist_v311": True,
        "feature_values_recomputed": False,
        "source_recording_feature_sha256": source_hash,
        "analysis_features": list(ANALYSIS_FEATURES),
        "feature_decisions": decisions[["feature", "final_decision", "publication_role", "default_manuscript_inclusion", "default_joint_model_inclusion"]].to_dict("records"),
        "figure_count": int(len(figure_index)), "main_figure_bundle_count": int((figure_index["panel"] != "G").sum()), "gallery_bundle_count": int((figure_index["panel"] == "G").sum()),
        "required_panels_complete": True, "panel_i_status": "APPLICABLE_complete_event_verification",
        "regenerated_figure_stems": regenerated,
        "family_scalar_status": "prohibited_not_constructed", "family_scalar_constructed": False,
        "standalone_gate_allowed": False, "complete_nonlinear_distortion_claim_allowed": False,
        "missing_values_imputed": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutability_policy": "freeze scripts refuse overwrite; any numerical change requires a new semantic measurement version",
    }
    manifest_dir = final_root / "manifests"; manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "qdist_v400_final_candidate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
