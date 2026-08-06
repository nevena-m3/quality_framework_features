"""Standard A–J figure bundles for the QDIST v4.1 candidate cohort.

Panels A–C are accepted preflight bundles. This module creates D–J without
using clinical labels or human-QC outcomes. Gallery items are detector-selected
candidate morphology and remain explicitly unconfirmed until two independent
human reviews and adjudication are complete.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "qdist_hard_clipped_sample_fraction": "Accepted plateau support\n(channel-sample fraction; primary)",
    "qdist_hard_clip_event_rate_per_min": "Merged episode rate\n(events/min; secondary)",
    "qdist_hard_clipped_frame_fraction": "Intersected 30-ms frames\n(fraction; conditional)",
}


def _json_safe(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_bundle(
    output_root: Path,
    panel: str,
    stem: str,
    figure: plt.Figure,
    source: pd.DataFrame,
    caption: str,
    provenance: Mapping[str, Any],
    *,
    bundle_role: str = "main",
) -> dict[str, Any]:
    folder = output_root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": folder / f"{stem}.png",
        "svg": folder / f"{stem}.svg",
        "pdf": folder / f"{stem}.pdf",
        "source_csv": folder / f"{stem}.source.csv",
        "caption": folder / f"{stem}.caption.md",
        "provenance": folder / f"{stem}.provenance.json",
    }
    figure.savefig(paths["png"], dpi=300, bbox_inches="tight")
    figure.savefig(paths["svg"], bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(figure)
    source.to_csv(paths["source_csv"], index=False)
    paths["caption"].write_text(caption.strip() + "\n", encoding="utf-8")
    payload = {
        "panel": panel,
        "stem": stem,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "clinical_labels_used": False,
        "human_qc_labels_used": False,
        **dict(provenance),
    }
    paths["provenance"].write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    row: dict[str, Any] = {"panel": panel, "stem": stem, "bundle_role": bundle_role}
    for field, path in paths.items():
        row[field] = str(path.relative_to(output_root))
        row[f"{field}_sha256"] = _sha256(path)
    return row


def _empty_axis(ax: plt.Axes, text: str) -> None:
    ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center")
    ax.set_axis_off()


def _panel_d1(challenge: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    if challenge.empty:
        _empty_axis(ax, "Known-truth cohort challenge unavailable")
    else:
        for geometry, group in challenge.groupby("geometry", sort=True):
            summary = (
                group.groupby("target_fraction")
                .agg(realized=("realized_fraction", "median"), estimated=("estimated_sample_fraction", "median"))
                .reset_index()
            )
            ax.plot(summary["realized"], summary["estimated"], marker="o", label=geometry.replace("_", " "))
        limits = [
            max(1e-6, challenge["realized_fraction"].min() * .7),
            max(challenge["realized_fraction"].max(), challenge["estimated_sample_fraction"].max()) * 1.3,
        ]
        ax.plot(limits, limits, ls="--", color=".35", label="identity (not expected)")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-6, linscale=.7)
        ax.set_ylim(0, limits[1])
        ax.set(xlabel="Exact altered channel-sample fraction", ylabel="Accepted plateau support fraction")
        ax.legend(frameon=False)
    ax.set_title("D1. Matched-burden response on cohort-derived speech")
    return fig, challenge, (
        "Panel D1. Candidate primary-output response to known hard limits imposed on "
        "label-blind valid-zero cohort speech. The x-axis is the exact fraction of "
        "decoded channel samples numerically changed; the y-axis is accepted plateau "
        "support, not an unbiased estimate of all altered samples. Geometry is matched "
        "by realized burden. These interventions validate decoded-waveform morphology "
        "and do not localize an analog or codec stage."
    )


def _panel_d2(challenge_summary: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    if challenge_summary.empty:
        for ax in axes:
            _empty_axis(ax, "Known-truth summary unavailable")
    else:
        for geometry, group in challenge_summary.groupby("geometry", sort=True):
            axes[0].plot(group["target_fraction"], group["occurrence_sensitivity"], marker="o", label=geometry.replace("_", " "))
            axes[1].plot(group["target_fraction"], group["sample_precision_median"], marker="o", label=f"precision: {geometry.replace('_', ' ')}")
            axes[1].plot(group["target_fraction"], group["sample_recall_median"], marker="s", ls="--", label=f"recall: {geometry.replace('_', ' ')}")
        axes[0].set(xscale="log", ylim=(-.03, 1.03), xlabel="Target altered fraction", ylabel="Occurrence sensitivity")
        axes[1].set(xscale="log", ylim=(-.03, 1.03), xlabel="Target altered fraction", ylabel="Median sample-level metric")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].legend(frameon=False, fontsize=7, ncol=2)
    axes[0].set_title("Any visible plateau detected")
    axes[1].set_title("Accepted support precision and recall")
    fig.suptitle("D2. Recovery operating characteristics")
    return fig, challenge_summary, (
        "Panel D2. Occurrence sensitivity and sample-level precision/recall against "
        "the exact altered-sample mask in cohort-derived speech. Recall quantifies "
        "how much imposed clipping support is represented by accepted plateaus; it is "
        "not required to equal one because the construct is conservative visible plateau support."
    )


def _panel_d3(support_summary: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    if support_summary.empty:
        _empty_axis(ax, "Support calibration unavailable")
    else:
        ax.plot(support_summary["duration_sec"], support_summary["occurrence_sensitivity"], marker="o", label="occurrence sensitivity")
        ax.plot(support_summary["duration_sec"], support_summary["sample_precision_median"], marker="s", label="median precision")
        ax.plot(support_summary["duration_sec"], support_summary["sample_recall_median"], marker="^", label="median recall")
        ax.plot(support_summary["duration_sec"], support_summary["availability"], marker="d", ls="--", label="availability")
        ax.set(xlabel="Analyzed task-span duration (s)", ylabel="Fraction", ylim=(-.03, 1.03))
        ax.legend(frameon=False)
    ax.set_title("D3. Known-truth recovery versus independent exposure")
    return fig, support_summary, (
        "Panel D3. Availability and known-truth recovery for a fixed symmetric "
        "altered-sample target across prespecified task-span durations. This panel "
        "supports exposure-tier interpretation; it does not convert short exposures "
        "into valid zeros when the extractor marks them indeterminate."
    )


def _panel_e1(recordings: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    features = list(FEATURE_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
    rows: list[dict[str, Any]] = []
    for ax, feature in zip(axes, features):
        values = pd.to_numeric(recordings[feature], errors="coerce")
        positive = values.loc[values > 0]
        rows.append({
            "feature": feature, "available_n": int(values.notna().sum()),
            "zero_n": int(values.eq(0).sum()), "positive_n": int(values.gt(0).sum()),
            "positive_median": float(positive.median()) if len(positive) else np.nan,
            "positive_maximum": float(positive.max()) if len(positive) else np.nan,
        })
        if len(positive):
            ax.hist(positive, bins=min(20, max(5, int(np.sqrt(len(positive))))), color="#4472C4", alpha=.85)
            if positive.min() > 0 and positive.max() / positive.min() > 50:
                ax.set_xscale("log")
        else:
            _empty_axis(ax, "No positive recordings")
        ax.set_title(FEATURE_LABELS[feature], fontsize=9)
        ax.set_xlabel("Positive-part value")
        ax.set_ylabel("Recordings")
    fig.suptitle("E1. Candidate cohort positive-part distributions (zeros reported separately)")
    return fig, pd.DataFrame(rows), (
        "Panel E1. Positive-part empirical distributions for the three related QDIST "
        "views. Structural observed zeros and unavailable values are not mixed into "
        "the positive-part histograms and are enumerated in the source table."
    )


def _panel_e2(recordings: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    summary = (
        recordings.groupby("qdist_support_tier", dropna=False, sort=True)
        .agg(recordings=("logical_recording_id", "size"), available=("qdist_available", "sum"), positive=("qdist_positive", "sum"), exposure_median_sec=("qdist_finite_exposure_sec", "median"))
        .reset_index()
    )
    summary["available_fraction"] = summary["available"] / summary["recordings"]
    summary["positive_fraction"] = summary["positive"] / summary["recordings"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
    x = np.arange(len(summary))
    axes[0].bar(x - .18, summary["available_fraction"], width=.36, label="available")
    axes[0].bar(x + .18, summary["positive_fraction"], width=.36, label="positive")
    axes[0].set(xticks=x, xticklabels=summary["qdist_support_tier"].astype(str), ylim=(0, 1.05), ylabel="Fraction")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False)
    status = recordings["qdist_status"].astype(str).value_counts().rename_axis("status").reset_index(name="recordings")
    axes[1].barh(status["status"], status["recordings"], color="#70AD47")
    axes[1].set(xlabel="Recordings", title="Governed status")
    axes[0].set_title("Availability and occurrence by support tier")
    fig.suptitle("E2. Support, availability, and valid absence")
    source = summary.assign(source_section="support_tier")
    status["source_section"] = "status"
    source = pd.concat([source, status], ignore_index=True, sort=False)
    return fig, source, (
        "Panel E2. Availability and observed QDIST occurrence by prespecified support "
        "tier, with governed recording status shown separately. Available-no-event is "
        "a valid zero; insufficient or invalid support remains unavailable."
    )


def _panel_e3(legacy_summary: pd.DataFrame, legacy_long: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    transitions = legacy_long["occurrence_transition"].value_counts().rename_axis("transition").reset_index(name="recordings")
    axes[0].barh(transitions["transition"], transitions["recordings"], color="#A5A5A5")
    axes[0].set(xlabel="Recordings", title="Occurrence transition")
    feature = "qdist_hard_clipped_sample_fraction"
    old = pd.to_numeric(legacy_long[f"legacy_{feature}"], errors="coerce")
    new = pd.to_numeric(legacy_long[feature], errors="coerce")
    axes[1].scatter(old, new, s=16, alpha=.65)
    axes[1].set(xlabel="qdist-v3.1.1 sample fraction", ylabel="qdist-v4.1 candidate sample fraction", title="Primary-view comparison")
    if (old > 0).any() or (new > 0).any():
        axes[1].set_xscale("symlog", linthresh=1e-6)
        axes[1].set_yscale("symlog", linthresh=1e-6)
    fig.suptitle("E3. Governed detector-version comparison")
    source = pd.concat([
        transitions.assign(source_section="occurrence_transitions"),
        legacy_summary.assign(source_section="summary"),
    ], ignore_index=True, sort=False)
    return fig, source, (
        "Panel E3. Recording-level comparison of the frozen qdist-v3.1.1 baseline "
        "with values recomputed by qdist-v4.1.0. Differences are detector-version "
        "effects and are not described as numerical reconstruction error."
    )


def _panel_f(parameter_summary: pd.DataFrame, merge_summary: pd.DataFrame, deletion_summary: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), constrained_layout=True)
    variants = parameter_summary.loc[~parameter_summary["variant"].eq("baseline")].copy() if len(parameter_summary) else pd.DataFrame()
    if len(variants):
        variants = variants.sort_values("occurrence_agreement")
        axes[0].barh(np.arange(len(variants)), variants["occurrence_agreement"], color="#4472C4")
        axes[0].set(yticks=np.arange(len(variants)), yticklabels=variants["variant"].str.replace("__", " = ", regex=False), xlim=(0, 1.01), xlabel="Occurrence agreement")
        axes[0].tick_params(axis="y", labelsize=6)
    else:
        _empty_axis(axes[0], "Parameter reruns unavailable")
    axes[0].set_title("One-factor parameter neighborhood")
    if len(merge_summary):
        axes[1].plot(merge_summary["merge_gap_ms"], merge_summary["occurrence_agreement"], marker="o", label="occurrence agreement")
        axes[1].plot(merge_summary["merge_gap_ms"], 1 - merge_summary["event_count_changed_fraction"], marker="s", label="event-count agreement")
        axes[1].set(xlabel="Episode merge gap (ms)", ylabel="Agreement", ylim=(-.03, 1.03))
        axes[1].legend(frameon=False, fontsize=8)
    else:
        _empty_axis(axes[1], "Merge-gap audit unavailable")
    axes[1].set_title("Episode construction")
    if len(deletion_summary):
        axes[2].bar(deletion_summary["deletion_type"], deletion_summary["maximum_sample_fraction_absolute_change"], label="sample fraction")
        axes[2].set(ylabel="Maximum absolute change", title="Single-item deletion influence")
    else:
        _empty_axis(axes[2], "Deletion audit unavailable")
    source = pd.concat([
        parameter_summary.assign(source_section="parameter_neighborhood"),
        merge_summary.assign(source_section="merge_gap"),
        deletion_summary.assign(source_section="deletion_influence"),
    ], ignore_index=True, sort=False)
    fig.suptitle("F. Boundary, construction, and influence robustness")
    return fig, source, (
        "Panel F. Prespecified one-factor detector variants, episode merge-gap "
        "sensitivity, and single-ledger-item deletion influence. Parameter testing is "
        "performed on a label-blind enriched subset containing all baseline positives, "
        "near-boundary candidates, and deterministic valid zeros."
    )


def _panel_h1(weighting: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    if weighting.empty:
        _empty_axis(ax, "Weighting summary unavailable")
    else:
        x = np.arange(len(weighting))
        y = weighting["positive_fraction"]
        low = y - weighting["wilson95_low"]
        high = weighting["wilson95_high"] - y
        ax.bar(x, y, color=["#4472C4", "#70AD47"][:len(x)])
        ax.errorbar(x, y, yerr=np.vstack([low, high]), fmt="none", color="black", capsize=4)
        ax.set(xticks=x, xticklabels=weighting["analysis_level"], ylabel="Positive fraction", ylim=(0, min(1, max(.15, float(weighting["wilson95_high"].max()) * 1.3))))
        ax.tick_params(axis="x", rotation=15)
    ax.set_title("H1. Recording- and participant-weighted occurrence")
    return fig, weighting, (
        "Panel H1. Descriptive occurrence under recording weighting and participant-"
        "ever-positive weighting with Wilson intervals. Neither analysis is a disease "
        "association test, and no diagnostic or human-QC outcome is used."
    )


def _panel_h2(repeated: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    if repeated.empty:
        _empty_axis(ax, "Repeated-recording evidence unavailable")
    else:
        metrics = repeated.copy()
        ax.bar(np.arange(len(metrics)), metrics["overall_agreement"], color="#ED7D31")
        ax.set(xticks=np.arange(len(metrics)), xticklabels=metrics["metric"].replace(FEATURE_LABELS).astype(str), ylim=(0, 1.05), ylabel="First-pair occurrence agreement")
        ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.set_title("H2. Within-participant persistence is descriptive, not reliability")
    return fig, repeated, (
        "Panel H2. First-pair within-participant occurrence agreement and positive-part "
        "estimability. Acquisition artifacts need not persist; sparse positive-positive "
        "pairs are not overinterpreted as test-retest reliability."
    )


def _panel_h3(redundancy: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    if redundancy.empty:
        _empty_axis(ax, "Related-view audit unavailable")
    else:
        labels = [f"{row.feature_1.split('qdist_')[-1]} vs\n{row.feature_2.split('qdist_')[-1]}" for row in redundancy.itertuples()]
        x = np.arange(len(redundancy))
        ax.bar(x - .18, redundancy["all_recordings_spearman_rho"], width=.36, label="all available")
        ax.bar(x + .18, redundancy["positive_recordings_spearman_rho"], width=.36, label="positive part")
        ax.set(xticks=x, xticklabels=labels, ylabel="Spearman rho", ylim=(-1, 1))
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(frameon=False)
    ax.set_title("H3. Three related views, not independent biomarkers")
    return fig, redundancy, (
        "Panel H3. Pairwise rank association among the three related outputs for all "
        "available recordings and, where estimable, the positive part. This audit "
        "prevents treating the views as independent constructs or averaging them into a scalar."
    )


def _panel_i(challenge_summary: pd.DataFrame, review_index: pd.DataFrame, review_errors: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), constrained_layout=True)
    if len(challenge_summary):
        geometry = challenge_summary.groupby("geometry").agg(sensitivity=("occurrence_sensitivity", "mean"), precision=("sample_precision_median", "median"), recall=("sample_recall_median", "median")).reset_index()
        x = np.arange(len(geometry))
        axes[0].bar(x - .22, geometry["sensitivity"], width=.22, label="occurrence")
        axes[0].bar(x, geometry["precision"], width=.22, label="precision")
        axes[0].bar(x + .22, geometry["recall"], width=.22, label="recall")
        axes[0].set(xticks=x, xticklabels=geometry["geometry"].str.replace("_", " "), ylim=(0, 1.05), title="Known-truth decoded-speech challenge")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].legend(frameon=False, fontsize=7)
    else:
        _empty_axis(axes[0], "Known-truth challenge unavailable")
    axes[1].bar(["generated", "errors"], [len(review_index), len(review_errors)], color=["#70AD47", "#C00000"])
    axes[1].set(title="Blinded evidence package", ylabel="Review items")
    axes[2].text(.5, .64, "TWO HUMAN REVIEWERS\nREQUIRED", ha="center", va="center", fontsize=14, weight="bold", color="#C00000")
    axes[2].text(.5, .36, "Status: PENDING\nNo AI labels\nNo freeze permitted", ha="center", va="center", fontsize=11)
    axes[2].set_axis_off()
    fig.suptitle("I. Verification evidence and unresolved human validation")
    source = pd.concat([
        challenge_summary.assign(source_section="known_truth"),
        pd.DataFrame([{"source_section": "blind_review", "review_items": len(review_index), "generation_errors": len(review_errors), "human_review_status": "PENDING_TWO_INDEPENDENT_REVIEWERS"}]),
    ], ignore_index=True, sort=False)
    return fig, source, (
        "Panel I. Known-truth decoded-speech validation and the status of independent "
        "human morphology review. The review package contains every accepted plateau, "
        "near-threshold rejections, and valid-zero windows. Human review remains pending; "
        "therefore G9, final feature decisions, and freeze do not pass."
    )


def _panel_j(decisions: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame, str]:
    fig, ax = plt.subplots(figsize=(13.2, 6.2), constrained_layout=True)
    ax.set_axis_off()
    if len(decisions):
        columns = ["feature", "candidate_role", "status", "permitted_interpretation"]
        shown = decisions[[column for column in columns if column in decisions]].copy()
        shown["feature"] = shown["feature"].astype(str).str.replace("qdist_", "", regex=False)
        shown["candidate_role"] = shown["candidate_role"].replace({
            "CONDITIONAL_AUDIT_OR_LEGACY_COMPATIBILITY": "CONDITIONAL / AUDIT",
            "COMPANION_STATUS": "COMPANION STATUS",
        })
        shown["status"] = "PENDING HUMAN REVIEW\n+ SCIENTIFIC DECISION"
        shown["permitted_interpretation"] = shown["permitted_interpretation"].astype(str).map(
            lambda value: "\n".join(textwrap.wrap(value, width=50))
        )
        table = ax.table(
            cellText=shown.values,
            colLabels=["Feature", "Candidate role", "Status", "Permitted interpretation"],
            colWidths=[.21, .17, .20, .42],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 2.4)
    else:
        ax.text(.5, .5, "Candidate feature decisions unavailable", ha="center")
    ax.set_title("J. Candidate roles and freeze boundary", pad=12)
    return fig, decisions, (
        "Panel J. Candidate, not final, feature roles. Accepted plateau support is the "
        "primary construct view, merged episode rate is secondary, and the legacy "
        "30-ms frame view is conditional because of frame-origin sensitivity. No "
        "family scalar, standalone recording gate, complete nonlinear-distortion claim, "
        "publication freeze, or diagnostic interpretation is permitted."
    )


def create_figures(
    output_root: str | Path,
    preflight_index: pd.DataFrame,
    evidence: Mapping[str, pd.DataFrame],
    *,
    parameter_hash: str,
    detector_sha256: str,
    gallery_minimum: int = 8,
) -> pd.DataFrame:
    root = Path(output_root)
    rows = preflight_index.to_dict("records")
    provenance = {
        "measurement_version": "qdist-v4.1.0-candidate",
        "parameter_hash": parameter_hash,
        "detector_sha256": detector_sha256,
        "signal_view": "native-rate multichannel decoded first audio stream; no transform",
        "feature_roles": {
            "primary": "qdist_hard_clipped_sample_fraction",
            "secondary": "qdist_hard_clip_event_rate_per_min",
            "conditional": "qdist_hard_clipped_frame_fraction",
        },
    }
    panel_builders = [
        ("D1", "qdist_v410_panel-D1_matched-burden-response", _panel_d1(evidence["challenge_long"])),
        ("D2", "qdist_v410_panel-D2_recovery-characteristics", _panel_d2(evidence["challenge_summary"])),
        ("D3", "qdist_v410_panel-D3_support-calibration", _panel_d3(evidence["support_summary"])),
        ("E1", "qdist_v410_panel-E1_empirical-distributions", _panel_e1(evidence["recordings"])),
        ("E2", "qdist_v410_panel-E2_support-status", _panel_e2(evidence["recordings"])),
        ("E3", "qdist_v410_panel-E3_legacy-comparison", _panel_e3(evidence["legacy_summary"], evidence["legacy_long"])),
        ("F", "qdist_v410_panel-F_robustness", _panel_f(evidence["parameter_summary"], evidence["merge_summary"], evidence["deletion_summary"])),
        ("H1", "qdist_v410_panel-H1_participant-weighting", _panel_h1(evidence["weighting"])),
        ("H2", "qdist_v410_panel-H2_repeated-recordings", _panel_h2(evidence["repeated_summary"])),
        ("H3", "qdist_v410_panel-H3_related-views", _panel_h3(evidence["redundancy"])),
        ("I", "qdist_v410_panel-I_verification-status", _panel_i(evidence["challenge_summary"], evidence["review_index"], evidence["review_errors"])),
        ("J", "qdist_v410_panel-J_candidate-decisions", _panel_j(evidence["decisions"])),
    ]
    for panel, stem, (figure, source, caption) in panel_builders:
        rows.append(_save_bundle(root, panel, stem, figure, source, caption, provenance))

    review_index = evidence["review_index"].head(max(gallery_minimum, 12)).copy()
    for gallery_number, item in enumerate(review_index.to_dict("records"), start=1):
        image_path = root / "blind_review" / str(item["image_path"])
        if not image_path.exists():
            continue
        image = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(f"G{gallery_number}. Detector-selected candidate morphology — human review pending")
        source = pd.DataFrame([{
            "blind_id": item["blind_id"],
            "review_status": item["review_status"],
            "selection_interpretation": "candidate morphology; not confirmed hard clipping",
        }])
        caption = (
            f"Panel G{gallery_number}. Blinded detector-selected candidate morphology "
            "with waveform, sample zoom, derivative, amplitude occupancy, spectrogram, "
            "and empirical CDF. This is an audit example, not a confirmed hard-clipping "
            "label; two independent human reviews remain required."
        )
        rows.append(_save_bundle(
            root, "G", f"qdist_v410_panel-G{gallery_number:02d}_{item['blind_id']}",
            fig, source, caption, provenance, bundle_role="gallery"
        ))
    return pd.DataFrame(rows)


def verify_figure_index(
    output_root: str | Path,
    index: pd.DataFrame,
    required_panels: tuple[str, ...],
    *,
    gallery_minimum: int = 8,
) -> pd.DataFrame:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    present = set(index.loc[index["panel"].ne("G"), "panel"].astype(str))
    rows.append({
        "gate": "G10", "check": "all required A–J panel bundles indexed",
        "status": "PASS" if set(required_panels).issubset(present) else "FAIL",
        "observed": "|".join(sorted(present)), "required": "|".join(required_panels),
    })
    gallery_count = int(index["panel"].eq("G").sum())
    rows.append({
        "gate": "G10", "check": "minimum candidate gallery bundles indexed",
        "status": "PASS" if gallery_count >= gallery_minimum else "FAIL",
        "observed": gallery_count, "required": gallery_minimum,
    })
    fields = ["png", "svg", "pdf", "source_csv", "caption", "provenance"]
    missing: list[str] = []
    for row in index.to_dict("records"):
        for field in fields:
            path = root / str(row.get(field, ""))
            if not path.exists() or path.stat().st_size == 0:
                missing.append(f"{row.get('stem')}::{field}")
    rows.append({
        "gate": "G10", "check": "all six files exist for every figure bundle",
        "status": "PASS" if not missing else "FAIL",
        "observed": "none" if not missing else "|".join(missing), "required": "none missing",
    })
    return pd.DataFrame(rows)
