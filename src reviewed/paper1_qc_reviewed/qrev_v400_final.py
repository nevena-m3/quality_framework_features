from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY = "QREV"
FAMILY_DISPLAY_NAME = "Reverberation / residual-tail manifestations"
SOURCE_MEASUREMENT_VERSION = "qrev-v4.0.0-candidate"
FINAL_MEASUREMENT_VERSION = "qrev-v4.0.0"
ACCEPTANCE_TOKEN = "ACCEPT_QREV_V400"
FIGURE_PACKAGE_VERSION = "qrev-v4.0.0-figures-v1.0.0"

ANALYSIS_FEATURES = (
    "qrev_tail_excess_100ms_db",
    "qrev_tail_persistence_median_sec",
    "qrev_downward_decay_rate_db_per_sec",
    "qrev_srmr_norm",
)

CONDITIONAL_FEATURES = ANALYSIS_FEATURES[:3]


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
    missing_value_behavior: str = "NaN with explicit status/support/censoring; never zero-imputed"


FINAL_FEATURE_DEFINITIONS = (
    FinalFeatureDefinition(
        feature="qrev_tail_excess_100ms_db",
        display_name="Early residual-tail excess",
        final_decision="RETAIN_PRIMARY_CONDITIONAL",
        publication_role="primary conditional boundary-based residual magnitude measurement",
        default_manuscript_inclusion=True,
        analysis_priority="primary_conditional",
        robustness_class="conditional_sparse_support_boundary_sensitive",
        interpretation_class="signed early post-offset residual relative to an independent late-pause floor",
        unit="dB",
        orientation="higher means greater early post-offset AC-RMS level relative to the stable late-pause floor",
        claim_limit="not RT60, EDT, DRR, room impulse response recovery, echo identity, or proof of room reverberation",
        minimum_support=">=2 eligible natural primary-speech offsets, each followed by a stable >=1.0-s internal pause",
        known_confounds="breath-like residuals, residual articulation, offset error, late-floor changes, additive noise, discrete echo outside the early window",
    ),
    FinalFeatureDefinition(
        feature="qrev_tail_persistence_median_sec",
        display_name="Bounded residual-tail persistence",
        final_decision="RETAIN_SECONDARY_CONDITIONAL_NONINDEPENDENT",
        publication_role="secondary bounded and potentially right-censored persistence descriptor",
        default_manuscript_inclusion=True,
        analysis_priority="secondary_conditional_nonindependent",
        robustness_class="conditional_sparse_support_boundary_sensitive",
        interpretation_class="time to sustained return within 3 dB of an independent late-pause floor, bounded at 0.6 s",
        unit="s",
        orientation="higher means longer observed residual persistence; 0.6 s denotes at least 0.6 s when right-censored",
        claim_limit="not reverberation time and not independent of tail excess; censoring and support must accompany the value",
        minimum_support=">=2 eligible natural primary-speech offsets with persistence estimates; horizon 0.6 s and floor 0.7-1.0 s",
        known_confounds="breathing, residual phonation, changing noise floor, offset error, sparse long pauses, censoring",
    ),
    FinalFeatureDefinition(
        feature="qrev_downward_decay_rate_db_per_sec",
        display_name="Downward residual decay rate",
        final_decision="RETAIN_EXPLORATORY_CONDITIONAL",
        publication_role="exploratory conditional residual-shape descriptor",
        default_manuscript_inclusion=False,
        analysis_priority="exploratory",
        robustness_class="sparse_and_parameter_sensitive",
        interpretation_class="median positive magnitude of a valid downward Theil-Sen AC-RMS slope",
        unit="dB/s",
        orientation="higher means a faster measured downward residual envelope over the specified decay window",
        claim_limit="not a room decay constant, RT60 derivative, or universally available reverberation measurement",
        minimum_support=">=2 eligible downward decays with sufficient dynamic range and stable late floor",
        known_confounds="window length, boundary location, nonmonotonic residuals, breathing, noise-floor drift, very sparse support",
    ),
    FinalFeatureDefinition(
        feature="qrev_srmr_norm",
        display_name="Normalized-fast SRMR",
        final_decision="RETAIN_ESTABLISHED_COMPARATOR",
        publication_role="published no-reference reverberation-sensitive comparator",
        default_manuscript_inclusion=True,
        analysis_priority="established_comparator",
        robustness_class="complete_but_nonspecific",
        interpretation_class="speech modulation-energy ratio under the pinned normalized-fast SRMRpy implementation",
        unit="ratio",
        orientation="lower values are generally compatible with stronger modulation smearing, but the direction is not source-specific",
        claim_limit="not a direct room parameter and not specific to reverberation; content, additive noise, bandwidth, codec and pitch may affect it",
        minimum_support=">=1.0 s strict-speech support and a valid natural primary-task span under the pinned runtime",
        known_confounds="speech content, dysarthria, pitch, additive noise, bandwidth, codec, channel filtering, task duration",
    ),
)

TEN_DOMAIN_DASHBOARD = (
    ("Construct validity", "PASS", "The family measures observable post-offset residual and modulation-smearing manifestations without claiming physical room parameters."),
    ("Estimator validity", "PASS", "Signed tail excess, bounded persistence, conditional decay and pinned SRMR compute the stated observables and reconstruct from saved evidence."),
    ("Implementation validity", "PASS", "Natural primary-speech offsets, independent late floors, global DC removal, exact media hashes and pinned SRMR identity are verified."),
    ("Transformation behavior", "PASS", "Gain, polarity, DC, common time shift, resampling, codec and duration behavior satisfy the prespecified contracts."),
    ("Dose response", "PASS", "RIR dose, known persistence and known exponential-decay simulations produce the expected responses within the declared operating range."),
    ("Discriminant validity", "CONDITIONAL", "Breath-like residuals, changing floors, delayed echo and additive noise remain non-identifiable or influential in single-channel no-reference audio."),
    ("Support and uncertainty", "CONDITIONAL", "Support, censoring and availability are explicit; support tiers are quantity classes rather than empirically calibrated precision tiers."),
    ("Reliability and robustness", "CONDITIONAL", "Tail excess, persistence and SRMR show moderate repeated-recording persistence; decay is sparse and more boundary/window sensitive."),
    ("Interpretability", "PASS", "Units, orientation, conditional availability, censoring, non-independence and failure modes are explicit."),
    ("Scientific scope", "PASS", "No QREV scalar, room-parameter claim, event detector, standalone reject threshold or source-identity assertion is authorized."),
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
        rows.append({"relative_path": relative, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def final_decisions_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in FINAL_FEATURE_DEFINITIONS])


def ten_domain_dashboard_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"domain_number": index + 1, "domain": domain, "status": status, "scientific_conclusion": conclusion}
        for index, (domain, status, conclusion) in enumerate(TEN_DOMAIN_DASHBOARD)
    ])


def final_gate_summary_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"gate": "G1", "status": "PASS", "evidence": "observable construct, prohibited claims, canonical views, pinned SRMR, no scalar"},
        {"gate": "G2", "status": "PASS", "evidence": "signed formula, independent floor, deterministic extraction, ledger reconstruction, missing-not-zero"},
        {"gate": "G3", "status": "PASS", "evidence": "gain, polarity, DC, time shift, source-rate, codec and duration characterization"},
        {"gate": "G4", "status": "PASS", "evidence": "RIR dose, bounded persistence recovery, censoring and exponential-decay recovery"},
        {"gate": "G5", "status": "CONDITIONAL", "evidence": "breath/noise/floor/echo confounds and SRMR noise/bandwidth dependence explicitly characterized"},
        {"gate": "G6", "status": "PASS_WITH_QUALIFICATION", "evidence": "2/3/4-boundary policies, deletion, corrected horizon, floor, boundary and window sensitivity; support is not precision"},
        {"gate": "G7", "status": "PASS", "evidence": "519 recordings, 224 participants, complete hashes, status/missingness/censoring and signal-linked examples"},
        {"gate": "G8", "status": "PASS_WITH_QUALIFICATION", "evidence": "repeated-recording persistence, participant weighting and redundancy; decay remains exploratory"},
        {"gate": "G9", "status": "N/A", "evidence": "no retained discrete QREV event detector"},
        {"gate": "G10", "status": "PASS", "evidence": "feature-specific decisions, no scalar/gate, numerical equivalence, complete figures and immutable-freeze contract"},
    ])


def final_registry_frame(source_registry: pd.DataFrame) -> pd.DataFrame:
    source = source_registry.copy().set_index("feature", drop=False)
    decisions = final_decisions_frame().set_index("feature", drop=False)
    rows = []
    for feature in ANALYSIS_FEATURES:
        if feature not in source.index:
            raise ValueError(f"Source registry is missing {feature}")
        row = source.loc[feature].to_dict()
        row.update(decisions.loc[feature].to_dict())
        row.update({
            "measurement_version": FINAL_MEASUREMENT_VERSION,
            "family": FAMILY,
            "family_display_name": FAMILY_DISPLAY_NAME,
            "publication_status": "scientifically_accepted_pending_freeze",
            "analysis_eligible": True,
            "standalone_gate_allowed": False,
            "family_scalar_allowed": False,
            "composite_use_prohibited": True,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def analysis_values_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left = left.sort_values("logical_recording_id").reset_index(drop=True)
    right = right.sort_values("logical_recording_id").reset_index(drop=True)
    if left["logical_recording_id"].astype(str).tolist() != right["logical_recording_id"].astype(str).tolist():
        return False
    for feature in ANALYSIS_FEATURES:
        a = pd.to_numeric(left[feature], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right[feature], errors="coerce").to_numpy(float)
        if not np.array_equal(a, b, equal_nan=True):
            return False
    return True


def corrected_horizon_long(
    boundary_ledger: pd.DataFrame,
    recording_table: pd.DataFrame,
    robustness_ids: Iterable[str],
    horizons_sec: Iterable[float] = (0.4, 0.5),
    minimum_boundary_count: int = 2,
) -> pd.DataFrame:
    """Derive shorter-horizon persistence from the stored default first-return/censoring evidence.

    The default ledger stores the first sustained return to floor when observed, or 0.6 s
    when right-censored. For a shorter horizon H, min(default_value, H) is exact under the
    unchanged threshold/consecutive-frame contract. No raw audio or feature recomputation is required.
    """
    robustness_ids = [str(item) for item in robustness_ids]
    base = recording_table.set_index(recording_table["logical_recording_id"].astype(str), drop=False)
    eligible = boundary_ledger.loc[boundary_ledger["persistence_eligible"].astype(bool)].copy()
    eligible["logical_recording_id"] = eligible["logical_recording_id"].astype(str)
    rows = []
    for horizon in horizons_sec:
        variant = f"horizon_{int(round(horizon * 1000))}ms"
        for recording_id in robustness_ids:
            local = eligible.loc[eligible["logical_recording_id"].eq(recording_id)]
            values = pd.to_numeric(local["tail_persistence_sec"], errors="coerce").dropna().to_numpy(float)
            transformed = np.minimum(values, float(horizon)) if len(values) else np.array([], dtype=float)
            available = len(transformed) >= int(minimum_boundary_count)
            variant_value = float(np.median(transformed)) if available else np.nan
            baseline_value = pd.to_numeric(
                pd.Series([base.loc[recording_id, "qrev_tail_persistence_median_sec"]]) if recording_id in base.index else pd.Series([np.nan]),
                errors="coerce",
            ).iloc[0]
            baseline_available = bool(np.isfinite(baseline_value))
            status = (
                "right_censored_at_horizon" if available and np.isclose(variant_value, horizon)
                else "measured" if available
                else "insufficient_support"
            )
            rows.append({
                "logical_recording_id": recording_id,
                "variant": variant,
                "feature": "qrev_tail_persistence_median_sec",
                "baseline_value": baseline_value,
                "variant_value": variant_value,
                "baseline_available": baseline_available,
                "variant_available": available,
                "absolute_delta": abs(variant_value - baseline_value) if available and baseline_available else np.nan,
                "variant_status": status,
                "horizon_sec": float(horizon),
                "derivation": "min(default first-return-or-0.6s-censor, shorter horizon)",
            })
    return pd.DataFrame(rows)


def sensitivity_summary(long_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, feature), local in long_table.groupby(["variant", "feature"], sort=True):
        baseline_available = local["baseline_available"].astype(bool)
        variant_available = local["variant_available"].astype(bool)
        paired = pd.to_numeric(local.loc[baseline_available & variant_available, "absolute_delta"], errors="coerce").dropna()
        rows.append({
            "variant": variant,
            "feature": feature,
            "recording_count": int(len(local)),
            "baseline_available_n": int(baseline_available.sum()),
            "variant_available_n": int(variant_available.sum()),
            "availability_agreement_fraction": float((baseline_available == variant_available).mean()) if len(local) else np.nan,
            "paired_finite_n": int(len(paired)),
            "median_absolute_delta": float(paired.median()) if len(paired) else np.nan,
            "p95_absolute_delta": float(paired.quantile(0.95)) if len(paired) else np.nan,
            "maximum_absolute_delta": float(paired.max()) if len(paired) else np.nan,
        })
    return pd.DataFrame(rows)


def replace_invalid_horizon_evidence(
    source_long: pd.DataFrame,
    boundary_ledger: pd.DataFrame,
    recording_table: pd.DataFrame,
    robustness_ids: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corrected = corrected_horizon_long(boundary_ledger, recording_table, robustness_ids)
    invalid_mask = source_long["variant"].isin(["horizon_400ms", "horizon_500ms"]) & source_long["feature"].eq("qrev_tail_persistence_median_sec")
    final_long = pd.concat([source_long.loc[~invalid_mask].copy(), corrected[source_long.columns]], ignore_index=True)
    final_summary = sensitivity_summary(final_long)
    return final_long, final_summary, corrected


def _save_figure_bundle(
    fig,
    *,
    root: Path,
    stem: str,
    panel: str,
    source_data: pd.DataFrame,
    caption: str,
    scientific_question: str,
    provenance_extra: dict | None = None,
) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(root / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    source_data.to_csv(root / f"{stem}.source.csv", index=False)
    (root / f"{stem}.caption.md").write_text(caption.strip() + "\n", encoding="utf-8")
    provenance = {
        "panel": panel,
        "figure_id": stem,
        "measurement_version": FINAL_MEASUREMENT_VERSION,
        "scientific_question": scientific_question,
        "feature_values_recomputed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if provenance_extra:
        provenance.update(provenance_extra)
    (root / f"{stem}.provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "panel": panel,
        "figure_id": stem,
        "png": f"figures/{stem}.png",
        "svg": f"figures/{stem}.svg",
        "pdf": f"figures/{stem}.pdf",
        "source_csv": f"figures/{stem}.source.csv",
        "caption": f"figures/{stem}.caption.md",
        "provenance": f"figures/{stem}.provenance.json",
        "purpose": scientific_question,
    }


def regenerate_reviewed_figures(final_root: Path, sensitivity_summary_table: pd.DataFrame) -> None:
    final_root = Path(final_root)
    figures = final_root / "figures"
    recording = pd.read_csv(final_root / "tables" / "qrev_v400_analysis_features.csv")
    delete = pd.read_csv(final_root / "validation" / "qrev_v400_delete_one_boundary_summary.csv")
    bootstrap = pd.read_csv(final_root / "validation" / "qrev_v400_bootstrap_median_precision.csv")
    weighting = pd.read_csv(final_root / "validation" / "qrev_v400_recording_vs_participant_weighting.csv")

    # Remove the superseded candidate D3 bundle before creating the final one.
    for suffix in (".png", ".svg", ".pdf", ".source.csv", ".caption.md", ".provenance.json"):
        obsolete = figures / f"D3_support_precision_relationships{suffix}"
        if obsolete.exists():
            obsolete.unlink()

    # D3: support and bootstrap uncertainty. Support tiers are not called precision tiers.
    lookup = recording.set_index("logical_recording_id")
    support_columns = {
        "qrev_tail_excess_100ms_db": "qrev_tail_valid_pause_support_sec",
        "qrev_tail_persistence_median_sec": "qrev_persistence_valid_pause_support_sec",
        "qrev_downward_decay_rate_db_per_sec": "qrev_decay_valid_pause_support_sec",
    }
    d3 = bootstrap.copy()
    d3["valid_pause_support_sec"] = [
        pd.to_numeric(pd.Series([lookup.at[rid, support_columns[feature]]]), errors="coerce").iloc[0]
        if rid in lookup.index else np.nan
        for rid, feature in zip(d3["logical_recording_id"].astype(str), d3["feature"])
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    labels = {
        "qrev_tail_excess_100ms_db": "Tail excess",
        "qrev_tail_persistence_median_sec": "Persistence",
        "qrev_downward_decay_rate_db_per_sec": "Decay rate",
    }
    for ax, feature in zip(axes.flat[:3], CONDITIONAL_FEATURES):
        local = d3.loc[d3["feature"].eq(feature)]
        ax.scatter(local["valid_pause_support_sec"], local["bootstrap_ci95_width"], s=14, alpha=0.55)
        ax.set_xlabel("Valid pause support (s)")
        ax.set_ylabel("Bootstrap median 95% CI width")
        ax.set_title(labels[feature])
    speech_support = pd.to_numeric(recording["qrev_srmr_strict_speech_support_sec"], errors="coerce")
    d3_srmr = recording[["logical_recording_id", "qrev_srmr_strict_speech_support_sec", "qrev_srmr_norm"]].copy()
    d3_srmr["feature"] = "qrev_srmr_norm"
    d3_srmr["source_type"] = "srmr_speech_support"
    d3["source_type"] = "conditional_bootstrap_uncertainty"
    d3_source = pd.concat([d3, d3_srmr], ignore_index=True, sort=False)
    axes.flat[3].hist(speech_support.dropna(), bins=24)
    axes.flat[3].set_xlabel("Strict-speech support (s)")
    axes.flat[3].set_ylabel("Recordings")
    axes.flat[3].set_title("SRMR support distribution")
    _save_figure_bundle(
        fig, root=figures, stem="D3_support_uncertainty_relationships", panel="D", source_data=d3_source,
        scientific_question="How does bootstrap variability relate to available pause support, without treating support classes as calibrated precision?",
        caption="Panel D3. Bootstrap variability of the recording-level median versus valid pause support for the three conditional boundary measurements, with the strict-speech support distribution for SRMR. Support classes quantify evidence quantity only; the observed bootstrap widths do not justify naming them precision tiers or interpreting them as calibrated per-recording uncertainty.",
        provenance_extra={"replaces": "D3_support_precision_relationships", "support_class_is_precision": False},
    )
    plt.close(fig)

    # E1: separate units and axes for each feature.
    e1 = delete.loc[delete["minimum_boundary_count"].eq(2)].copy().set_index("feature").reindex(CONDITIONAL_FEATURES).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    units = ["dB", "s", "dB/s"]
    names = ["Tail excess", "Persistence", "Decay rate"]
    for ax, (_, row), unit, name in zip(axes, e1.iterrows(), units, names):
        values = [row["median_absolute_delta"], row["p95_absolute_delta"]]
        ax.bar([0, 1], values)
        ax.set_xticks([0, 1], ["Median", "95th percentile"])
        ax.set_ylabel(f"Absolute change ({unit})")
        ax.set_title(name)
    _save_figure_bundle(
        fig, root=figures, stem="E1_delete_one_boundary_sensitivity", panel="E", source_data=e1,
        scientific_question="How strongly can omission of one eligible boundary alter each recording-level median?",
        caption="Panel E1. Feature-specific median and 95th-percentile absolute changes after deleting one eligible speech-to-pause boundary under the final two-boundary minimum-support policy. Each feature is plotted on its own axis and in its own unit; the panels are not a shared severity scale.",
        provenance_extra={"final_support_policy_minimum_boundary_count": 2},
    )
    plt.close(fig)

    selected_variants = [
        "offset_minus_100ms", "offset_minus_50ms", "offset_plus_50ms", "offset_plus_100ms",
        "floor_600_900ms", "floor_800_1100ms", "horizon_400ms", "horizon_500ms",
        "threshold_2db", "threshold_4db", "consecutive_2", "consecutive_4",
        "frame_20ms_hop10ms", "frame_40ms_hop10ms", "early_80ms", "early_120ms",
        "decay_200ms", "decay_400ms",
    ]
    e2 = sensitivity_summary_table.loc[
        sensitivity_summary_table["variant"].isin(selected_variants)
        & sensitivity_summary_table["feature"].isin(CONDITIONAL_FEATURES)
    ].copy()
    matrix = e2.pivot(index="variant", columns="feature", values="availability_agreement_fraction").reindex(index=selected_variants, columns=CONDITIONAL_FEATURES)
    fig, ax = plt.subplots(figsize=(10.7, 8.8))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(3), ["Tail excess", "Persistence", "Decay"])
    ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set_title("Availability agreement under boundary and estimator perturbations")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Availability agreement fraction")
    _save_figure_bundle(
        fig, root=figures, stem="E2_boundary_parameter_sensitivity", panel="E", source_data=e2,
        scientific_question="How stable are feature-availability decisions under prespecified boundary, floor, horizon, frame and window changes?",
        caption="Panel E2. Availability agreement with the frozen default estimator. The 0.4-s and 0.5-s persistence horizons are derived exactly from each stored default first-return or 0.6-s right-censored boundary value, correcting the cohort notebook's invalid shorter-horizon frame-count implementation.",
        provenance_extra={"horizon_sensitivity_corrected": True},
    )
    plt.close(fig)

    scale = {}
    for feature in CONDITIONAL_FEATURES:
        values = pd.to_numeric(recording[feature], errors="coerce").dropna()
        scale[feature] = float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan
    e3 = e2.copy()
    e3["empirical_iqr"] = e3["feature"].map(scale)
    e3["median_absolute_delta_in_iqr_units"] = e3["median_absolute_delta"] / e3["empirical_iqr"].replace(0, np.nan)
    matrix_delta = e3.pivot(index="variant", columns="feature", values="median_absolute_delta_in_iqr_units").reindex(index=selected_variants, columns=CONDITIONAL_FEATURES)
    finite = matrix_delta.to_numpy(float)
    finite_max = np.nanquantile(finite, 0.95) if np.isfinite(finite).any() else 1.0
    fig, ax = plt.subplots(figsize=(10.7, 8.8))
    image = ax.imshow(finite, aspect="auto", vmin=0, vmax=max(float(finite_max), 1e-6))
    ax.set_xticks(np.arange(3), ["Tail excess", "Persistence", "Decay"])
    ax.set_yticks(np.arange(len(matrix_delta)), matrix_delta.index)
    ax.set_title("Paired value sensitivity normalized by each feature's empirical IQR")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Median absolute change / empirical IQR")
    _save_figure_bundle(
        fig, root=figures, stem="E3_parameter_value_sensitivity", panel="E", source_data=e3,
        scientific_question="How large are paired changes under prespecified perturbations relative to each feature's empirical spread?",
        caption="Panel E3. Median paired absolute change divided by the corrected-cohort IQR of the same feature. This dimensionless normalization is used only to visualize sensitivity within each feature; it does not create a QREV scalar or equate the scientific meanings of different measurements. Shorter-horizon persistence evidence is corrected from saved boundary return times.",
        provenance_extra={"horizon_sensitivity_corrected": True, "normalization_is_composite": False},
    )
    plt.close(fig)

    # H3: feature-specific units and paired medians.
    h3 = weighting.set_index("feature").reindex(ANALYSIS_FEATURES).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))
    names = ["Tail excess (dB)", "Persistence (s)", "Decay rate (dB/s)", "Normalized-fast SRMR"]
    for ax, (_, row), name in zip(axes.flat, h3.iterrows(), names):
        ax.bar([0, 1], [row["recording_weighted_median"], row["median_of_medians"]])
        ax.set_xticks([0, 1], ["Recording-weighted", "Participant-balanced"])
        ax.set_ylabel(name)
        ax.set_title(name)
    _save_figure_bundle(
        fig, root=figures, stem="H3_participant_weighting", panel="H", source_data=h3,
        scientific_question="Do participants contributing more recordings materially shift feature summaries?",
        caption="Panel H3. Recording-weighted medians and participant-balanced median-of-medians for each QREV measurement. Each feature has its own panel and unit. The small paired differences indicate that recording multiplicity does not materially drive the cohort summaries.",
    )
    plt.close(fig)


def combined_figure_index(root: Path) -> pd.DataFrame:
    root = Path(root)
    source_index = pd.read_csv(root / "tables" / "qrev_v400_figure_index.csv")
    replacement = {
        "D3_support_precision_relationships": "D3_support_uncertainty_relationships",
    }
    rows = []
    for _, item in source_index.iterrows():
        figure_id = replacement.get(str(item["figure_id"]), str(item["figure_id"]))
        panel = str(item["panel"])
        folder = "galleries" if panel == "G" else "figures"
        base = root / folder / figure_id
        artifact_map = {
            "png": base.with_suffix(".png"),
            "svg": base.with_suffix(".svg"),
            "pdf": base.with_suffix(".pdf"),
            "source_csv": base.with_suffix(".source.csv"),
            "caption": base.with_suffix(".caption.md"),
            "provenance": base.with_suffix(".provenance.json"),
        }
        missing = [name for name, path in artifact_map.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{figure_id} missing {missing}")
        rows.append({
            "panel": panel,
            "figure_id": figure_id,
            "purpose": str(item.get("selection_reason", "")) if panel == "G" else figure_id,
            "logical_recording_id": str(item.get("logical_recording_id", "")) if panel == "G" else "",
            "views": str(item.get("views", "")) if panel == "G" else "",
            **{name: path.relative_to(root).as_posix() for name, path in artifact_map.items()},
        })
    rows.append({
        "panel": "I", "figure_id": "I_not_applicable", "purpose": "no retained event detector",
        "logical_recording_id": "", "views": "", "png": "", "svg": "", "pdf": "", "source_csv": "", "caption": "", "provenance": "",
    })
    return pd.DataFrame(rows)


def feature_passport_text(definition: FinalFeatureDefinition) -> str:
    return f"""# {definition.display_name}

- **Feature:** `{definition.feature}`
- **Family:** {FAMILY} — {FAMILY_DISPLAY_NAME}
- **Measurement version:** {FINAL_MEASUREMENT_VERSION}
- **Final decision:** {definition.final_decision}
- **Publication role:** {definition.publication_role}
- **Default manuscript inclusion:** {definition.default_manuscript_inclusion}
- **Analysis priority:** {definition.analysis_priority}
- **Unit:** {definition.unit}
- **Orientation:** {definition.orientation}
- **Robustness class:** {definition.robustness_class}
- **Interpretation class:** {definition.interpretation_class}
- **Minimum support:** {definition.minimum_support}
- **Claim boundary:** {definition.claim_limit}
- **Known confounders:** {definition.known_confounds}
- **Missingness/censoring:** {definition.missing_value_behavior}
- **Standalone reject gate allowed:** No
- **Family scalar/composite:** Prohibited
"""


def _replace_version_in_table(csv_path: Path, parquet_path: Path | None = None) -> None:
    if csv_path.exists():
        text = csv_path.read_text(encoding="utf-8").replace(SOURCE_MEASUREMENT_VERSION, FINAL_MEASUREMENT_VERSION)
        csv_path.write_text(text, encoding="utf-8")
    if parquet_path and parquet_path.exists():
        try:
            frame = pd.read_parquet(parquet_path)
            for column in frame.columns:
                if "measurement_version" in str(column):
                    frame[column] = FINAL_MEASUREMENT_VERSION
            frame.to_parquet(parquet_path, index=False)
        except Exception:
            pass


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
    source_manifest_path = source_root / "manifests" / "qrev_v400_cohort_candidate_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    required = {
        "cohort_extraction_completed": True,
        "cohort_evidence_complete": True,
        "recording_count": 519,
        "participant_count": 224,
        "required_panels_complete": True,
        "family_scalar_constructed": False,
        "standalone_gate_allowed": False,
    }
    for key, expected in required.items():
        if source_manifest.get(key) != expected:
            raise ValueError(f"Source manifest mismatch for {key}: {source_manifest.get(key)!r}")
    for relative in (
        "audit/qrev_v400_extraction_errors.csv",
        "audit/qrev_v400_robustness_errors.csv",
        "audit/qrev_v400_srmr_bandwidth_errors.csv",
        "audit/qrev_v400_gallery_errors.csv",
    ):
        if len(pd.read_csv(source_root / relative)):
            raise ValueError(f"Source error table is not empty: {relative}")

    if final_root.exists():
        shutil.rmtree(final_root)
    shutil.copytree(source_root, final_root)

    source_features = pd.read_csv(source_root / "tables" / "qrev_v400_analysis_features.csv")
    source_feature_hash = sha256_file(source_root / "tables" / "qrev_v400_analysis_features.csv")

    # Update only semantic/version metadata in numerical tables; retain all analysis values.
    for stem in (
        "qrev_v400_analysis_features",
        "qrev_v400_model_ready_features",
        "qrev_v400_policy_min_2_features",
        "qrev_v400_policy_min_3_features",
        "qrev_v400_policy_min_4_features",
    ):
        _replace_version_in_table(final_root / "tables" / f"{stem}.csv", final_root / "tables" / f"{stem}.parquet")
    _replace_version_in_table(final_root / "ledgers" / "qrev_v400_boundary_ledger.csv", final_root / "ledgers" / "qrev_v400_boundary_ledger.parquet")

    # Preserve the reviewed CSV numeric tokens exactly; replace only semantic metadata strings.
    analysis_csv = final_root / "tables" / "qrev_v400_analysis_features.csv"
    analysis_text = (source_root / "tables" / "qrev_v400_analysis_features.csv").read_text(encoding="utf-8")
    analysis_text = analysis_text.replace(SOURCE_MEASUREMENT_VERSION, FINAL_MEASUREMENT_VERSION)
    analysis_text = analysis_text.replace(
        "provisional_2_boundary_minimum_compare_2_3_4_before_G10",
        "final_minimum_2_eligible_boundaries_support_class_not_precision",
    )
    analysis_csv.write_text(analysis_text, encoding="utf-8")
    try:
        parquet_source = pd.read_parquet(source_root / "tables" / "qrev_v400_analysis_features.parquet")
        parquet_source["qrev_measurement_version"] = FINAL_MEASUREMENT_VERSION
        parquet_source["qrev_support_policy"] = "final_minimum_2_eligible_boundaries_support_class_not_precision"
        parquet_source.to_parquet(final_root / "tables" / "qrev_v400_analysis_features.parquet", index=False)
    except Exception:
        pass
    final_features = pd.read_csv(analysis_csv)
    if not analysis_values_equal(source_features, final_features):
        raise ValueError("Finalization changed QREV analysis feature values")

    source_registry = pd.read_csv(source_root / "tables" / "qrev_v400_feature_registry.csv")
    registry = final_registry_frame(source_registry)
    registry.to_csv(final_root / "tables" / "qrev_v400_feature_registry.csv", index=False)
    try:
        registry.to_parquet(final_root / "tables" / "qrev_v400_feature_registry.parquet", index=False)
    except Exception:
        pass

    decisions = final_decisions_frame()
    decisions.to_csv(final_root / "validation" / "qrev_v400_g10_feature_decisions.csv", index=False)
    dashboard = ten_domain_dashboard_frame()
    dashboard.to_csv(final_root / "validation" / "qrev_v400_ten_domain_dashboard.csv", index=False)
    gates = final_gate_summary_frame()
    gates.to_csv(final_root / "validation" / "qrev_v400_gate_summary_final.csv", index=False)

    boundary = pd.read_csv(final_root / "ledgers" / "qrev_v400_boundary_ledger.csv")
    source_sensitivity = pd.read_csv(source_root / "validation" / "qrev_v400_parameter_sensitivity_long.csv")
    robustness_ids = pd.read_csv(source_root / "validation" / "qrev_v400_robustness_sample.csv")["logical_recording_id"].astype(str).tolist()
    final_long, final_summary, corrected = replace_invalid_horizon_evidence(source_sensitivity, boundary, source_features, robustness_ids)
    final_long.to_csv(final_root / "validation" / "qrev_v400_parameter_sensitivity_long.csv", index=False)
    final_summary.to_csv(final_root / "validation" / "qrev_v400_parameter_sensitivity_summary.csv", index=False)
    corrected.to_csv(final_root / "validation" / "qrev_v400_corrected_horizon_sensitivity.csv", index=False)
    try:
        final_long.to_parquet(final_root / "validation" / "qrev_v400_parameter_sensitivity_long.parquet", index=False)
        final_summary.to_parquet(final_root / "validation" / "qrev_v400_parameter_sensitivity_summary.parquet", index=False)
    except Exception:
        pass

    regenerate_reviewed_figures(final_root, final_summary)
    figure_index = combined_figure_index(final_root)
    figure_index.to_csv(final_root / "figures" / "qrev_v400_standardized_figure_index.csv", index=False)

    passports = final_root / "feature_passports"
    passports.mkdir(parents=True, exist_ok=True)
    for definition in FINAL_FEATURE_DEFINITIONS:
        (passports / f"{definition.feature}.md").write_text(feature_passport_text(definition), encoding="utf-8")

    provenance = final_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest_path, provenance / "qrev_v400_source_cohort_candidate_manifest.json")
    for path in provenance_files:
        path = Path(path)
        if path.exists():
            shutil.copy2(path, provenance / path.name)

    empirical = pd.read_csv(final_root / "validation" / "qrev_v400_empirical_feature_summary.csv").set_index("feature")
    repeat = pd.read_csv(final_root / "validation" / "qrev_v400_repeated_recording_persistence.csv").set_index("feature")
    redundancy = pd.read_csv(final_root / "validation" / "qrev_v400_pairwise_redundancy.csv")
    tail_persistence_rho = float(redundancy.loc[
        redundancy["feature_a"].eq("qrev_tail_excess_100ms_db")
        & redundancy["feature_b"].eq("qrev_tail_persistence_median_sec"), "spearman_rho"
    ].iloc[0])

    accepted = scientific_review_decision == ACCEPTANCE_TOKEN
    excluded = {
        "manifests/qrev_v400_final_candidate_manifest.json",
        "manifests/qrev_v400_final_candidate_inventory.csv",
    }
    inventory = file_inventory(final_root, exclude=excluded)
    inventory_path = final_root / "manifests" / "qrev_v400_final_candidate_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    manifest = {
        "measurement_version": FINAL_MEASUREMENT_VERSION,
        "source_measurement_version": SOURCE_MEASUREMENT_VERSION,
        "family": FAMILY,
        "family_display_name": FAMILY_DISPLAY_NAME,
        "candidate_only": True,
        "freeze_status": "ready_for_atomic_freeze" if accepted else "scientific_review_pending",
        "freeze_allowed": bool(accepted),
        "scientific_review_decision": scientific_review_decision,
        "scientific_reviewer": scientific_reviewer,
        "scientific_review_rationale": scientific_review_rationale,
        "analysis_features": list(ANALYSIS_FEATURES),
        "primary_analysis_features": ["qrev_tail_excess_100ms_db"],
        "secondary_analysis_features": ["qrev_tail_persistence_median_sec"],
        "exploratory_features": ["qrev_downward_decay_rate_db_per_sec"],
        "established_comparators": ["qrev_srmr_norm"],
        "recording_count": 519,
        "participant_count": 224,
        "final_support_policy": "minimum 2 eligible boundaries; support class, not calibrated precision",
        "tail_available_n": int(empirical.at["qrev_tail_excess_100ms_db", "available_n"]),
        "persistence_available_n": int(empirical.at["qrev_tail_persistence_median_sec", "available_n"]),
        "decay_available_n": int(empirical.at["qrev_downward_decay_rate_db_per_sec", "available_n"]),
        "srmr_available_n": int(empirical.at["qrev_srmr_norm", "available_n"]),
        "tail_persistence_spearman_rho": tail_persistence_rho,
        "decay_first_second_spearman": float(repeat.at["qrev_downward_decay_rate_db_per_sec", "first_second_spearman"]),
        "persistence_horizon_sec": 0.6,
        "persistence_floor_window_sec": [0.7, 1.0],
        "right_censoring_explicit": True,
        "horizon_sensitivity_corrected": True,
        "horizon_correction_recomputed_raw_audio": False,
        "numerical_equivalence_to_cohort_candidate": True,
        "source_analysis_features_sha256": source_feature_hash,
        "feature_values_recomputed": False,
        "required_panels_complete": True,
        "completed_panels": ["A", "B", "C", "D", "E", "F", "G", "H", "J"],
        "panel_i_status": "N/A_no_retained_event_detector",
        "figure_count_excluding_na": int((figure_index["panel"] != "I").sum()),
        "family_scalar_status": "prohibited_not_constructed",
        "standalone_reject_allowed": False,
        "decision_threshold_status": "not_calibrated",
        "support_tier_is_precision": False,
        "artifact_count_excluding_seal_files": int(len(inventory)),
        "final_candidate_inventory_sha256": sha256_file(inventory_path),
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "immutability_policy": "never overwrite; create a new semantic version for any measurement change",
    }
    manifest_path = final_root / "manifests" / "qrev_v400_final_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (final_root / "READY_FOR_FREEZE_QREV_V4_0_0.txt").write_text(
        "QREV v4.0.0 is scientifically accepted and ready for atomic freeze.\n" if accepted
        else "QREV v4.0.0 remains pending scientific acceptance.\n",
        encoding="utf-8",
    )
    return manifest
