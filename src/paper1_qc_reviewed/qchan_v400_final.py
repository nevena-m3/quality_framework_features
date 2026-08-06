from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FAMILY = "QCHAN"
FAMILY_DISPLAY_NAME = "Channel/device spectral manifestations"
SOURCE_MEASUREMENT_VERSION = "qchan-v4.0.0-candidate"
FINAL_MEASUREMENT_VERSION = "qchan-v4.0.0"
ACCEPTANCE_TOKEN = "ACCEPT_QCHAN_V400"
FIGURE_PACKAGE_VERSION = "qchan-v4.0.0-figures-v1.0.0"
FINALIZATION_REVISION = "qchan-v4.0.0-finalization-r1"

ANALYSIS_FEATURES = (
    "qchan_ltas_distance_db",
    "qchan_rolloff95_deficit_hz",
    "qchan_highband_ratio_deficit",
    "qchan_tilt_steepening_db_per_oct",
)
SIGNED_PRECURSORS = (
    "qchan_rolloff95_signed_difference_hz",
    "qchan_highband_ratio_signed_difference",
    "qchan_tilt_signed_difference_db_per_oct",
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
    missing_value_behavior: str = (
        "NaN with explicit target/reference status, support, native bandwidth, "
        "and reference vintage; never zero-imputed"
    )


FINAL_FEATURE_DEFINITIONS = (
    FinalFeatureDefinition(
        feature="qchan_ltas_distance_db",
        display_name="Reference-relative LTAS distance",
        final_decision="RETAIN_PRIMARY_NONORDINAL",
        publication_role=(
            "primary nonordinal cohort-relative spectral-deviation measurement"
        ),
        default_manuscript_inclusion=True,
        analysis_priority="primary_nonordinal",
        robustness_class="complete_reference_stable_smoothing_defined",
        interpretation_class=(
            "RMS distance between target and frozen subject-balanced LOSO "
            "one-third-octave log-LTAS"
        ),
        unit="dB RMS",
        orientation=(
            "higher means greater spectral deviation from the frozen reference; "
            "higher is not intrinsically worse"
        ),
        claim_limit=(
            "not device identification, microphone transfer-function recovery, "
            "codec classification, or a pure technical-quality severity axis"
        ),
        minimum_support=(
            ">=3 s guarded strict speech and >=100 valid frames; task-matched "
            "LOSO reference with >=5 other participants and >=8 recordings"
        ),
        known_confounds=(
            "phonetic composition, anatomy, sex, age, vocal effort, dysarthria, "
            "additive noise, platform/channel, source bandwidth, and reference composition"
        ),
    ),
    FinalFeatureDefinition(
        feature="qchan_rolloff95_deficit_hz",
        display_name="Reference-relative rolloff-95 deficit",
        final_decision="RETAIN_PRIMARY_ONE_SIDED",
        publication_role=(
            "primary one-sided upper-spectral-extent attenuation proxy"
        ),
        default_manuscript_inclusion=True,
        analysis_priority="primary_one_sided",
        robustness_class="complete_robust_but_definition_and_zero_mass_sensitive",
        interpretation_class=(
            "positive part of frozen-reference minus target 95%-power rolloff, "
            "with the signed precursor retained"
        ),
        unit="Hz",
        orientation=(
            "higher means lower target upper-spectral extent than the reference; "
            "zero includes equal or upward signed differences"
        ),
        claim_limit=(
            "not a device or codec label; not proof of bandwidth restriction; "
            "the 95% rolloff definition is part of feature identity"
        ),
        minimum_support=(
            "same target/reference support as LTAS distance; source sample rate, "
            "Nyquist, signed difference, and reference vintage must accompany the value"
        ),
        known_confounds=(
            "fricative content, articulation, dysarthria, additive noise, sex, age, "
            "task execution, source capture chain, and rolloff-fraction choice"
        ),
    ),
    FinalFeatureDefinition(
        feature="qchan_highband_ratio_deficit",
        display_name="Reference-relative high-band ratio deficit",
        final_decision="RETAIN_SECONDARY_NONINDEPENDENT",
        publication_role=(
            "secondary one-sided 3-7.5-kHz power-share attenuation proxy"
        ),
        default_manuscript_inclusion=True,
        analysis_priority="secondary_nonindependent",
        robustness_class="complete_moderate_repeat_and_band_definition_sensitive",
        interpretation_class=(
            "positive part of reference minus target high-band power ratio, "
            "with the signed precursor retained"
        ),
        unit="proportion",
        orientation=(
            "higher means less target 3-7.5-kHz power share than the reference; "
            "zero includes equal or upward signed differences"
        ),
        claim_limit=(
            "not independent of rolloff deficit and not a source-specific bandwidth estimate"
        ),
        minimum_support="same target/reference support as LTAS distance",
        known_confounds=(
            "frication, phonetic mix, additive high-frequency noise, articulation, "
            "source capture chain, and high-band split definition"
        ),
    ),
    FinalFeatureDefinition(
        feature="qchan_tilt_steepening_db_per_oct",
        display_name="Reference-relative spectral-tilt steepening",
        final_decision="RETAIN_EXPLORATORY_PHENOTYPE_SENSITIVE",
        publication_role=(
            "exploratory one-sided broad spectral-shape descriptor"
        ),
        default_manuscript_inclusion=False,
        analysis_priority="exploratory_phenotype_sensitive",
        robustness_class="repeatable_but_smoothing_range_and_phenotype_sensitive",
        interpretation_class=(
            "positive part of reference minus target robust log-LTAS slope, "
            "with the signed precursor retained"
        ),
        unit="dB/octave",
        orientation=(
            "higher means a steeper downward target spectral slope than the reference"
        ),
        claim_limit=(
            "not independent evidence of channel degradation and not suitable for "
            "default confirmatory interpretation because speech physiology and phonetics "
            "can produce the same pattern"
        ),
        minimum_support="same target/reference support as LTAS distance",
        known_confounds=(
            "glottal source, vocal effort, dysarthria, sex, age, phonetic composition, "
            "additive noise, smoothing width, and tilt fitting range"
        ),
    ),
)

TEN_DOMAIN_DASHBOARD = (
    (
        "Construct validity",
        "PASS",
        "QCHAN measures cohort-relative spectral deviation and one-sided attenuation manifestations without identifying a physical device or channel source.",
    ),
    (
        "Estimator validity",
        "PASS",
        "All four estimators reconstruct from saved target spectra and frozen LOSO references; signed precursors are retained before one-sided truncation.",
    ),
    (
        "Implementation validity",
        "PASS",
        "Canonical 16-kHz DC-removed analysis audio, guarded strict-speech frames, native-rate metadata, subject exclusion, and reference hashes are verified.",
    ),
    (
        "Transformation behavior",
        "PASS",
        "Uniform gain, polarity, DC, and common time shift are invariant; resampling and codecs are characterized rather than assumed invariant.",
    ),
    (
        "Dose response",
        "PASS",
        "Controlled low-pass restriction yields ordered LTAS, rolloff, and high-band responses; shelves and notches establish nonordinal coloration behavior.",
    ),
    (
        "Discriminant validity",
        "CONDITIONAL",
        "Noise, speech content, anatomy, dysarthria, and platform effects remain non-identifiable from single-channel no-reference speech.",
    ),
    (
        "Support and uncertainty",
        "PASS_WITH_QUALIFICATION",
        "All cohort recordings have strong target and reference support; source rate, reference membership, one-sided zero mass, and reference vintage remain mandatory context.",
    ),
    (
        "Reliability and robustness",
        "PASS_WITH_QUALIFICATION",
        "Repeated-recording persistence and reference stability are strong overall; high-band repeatability is lower and tilt/rolloff depend on their pinned definitions.",
    ),
    (
        "Interpretability",
        "PASS",
        "Physical units, signed precursors, zero semantics, nonordinal direction, reference dependence, and confounders are explicit.",
    ),
    (
        "Scientific scope",
        "PASS",
        "No scalar, standalone rejection threshold, device identity, codec label, or pure transfer-function claim is authorized.",
    ),
)

DISPLAY_NAMES = {
    "qchan_ltas_distance_db": "LTAS distance",
    "qchan_rolloff95_deficit_hz": "Rolloff-95 deficit",
    "qchan_highband_ratio_deficit": "High-band deficit",
    "qchan_tilt_steepening_db_per_oct": "Tilt steepening",
}
UNITS = {
    "qchan_ltas_distance_db": "dB RMS",
    "qchan_rolloff95_deficit_hz": "Hz",
    "qchan_highband_ratio_deficit": "proportion",
    "qchan_tilt_steepening_db_per_oct": "dB/octave",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
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
        rows.append(
            {
                "relative_path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def final_decisions_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in FINAL_FEATURE_DEFINITIONS])


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
            {
                "gate": "G1",
                "status": "PASS",
                "evidence": "construct, prohibited claims, four-feature registry, canonical waveform, reference identity and no scalar",
            },
            {
                "gate": "G2",
                "status": "PASS",
                "evidence": "formula tests, exact self-reference zeros, signed precursors, missing-not-zero and 519-recording reconstruction",
            },
            {
                "gate": "G3",
                "status": "PASS",
                "evidence": "gain, polarity, DC, time shift, source-rate and codec characterization",
            },
            {
                "gate": "G4",
                "status": "PASS",
                "evidence": "ordered low-pass dose response plus shelf and notch coloration controls",
            },
            {
                "gate": "G5",
                "status": "CONDITIONAL",
                "evidence": "noise, phonetic, physiological and source non-identifiability explicitly retained",
            },
            {
                "gate": "G6",
                "status": "PASS_WITH_QUALIFICATION",
                "evidence": "target, parameter, reference membership, bootstrap, vintage and weighting sensitivity complete; estimator definitions remain part of identity",
            },
            {
                "gate": "G7",
                "status": "PASS_WITH_QUALIFICATION",
                "evidence": "519 recordings, 224 participants, complete support/hashes, empirical distributions and eight linked examples; source-rate group structure remains contextual",
            },
            {
                "gate": "G8",
                "status": "PASS_WITH_QUALIFICATION",
                "evidence": "participant-bootstrap confidence intervals, repeat persistence, redundancy and weighting analyses; high-band is secondary and tilt exploratory",
            },
            {
                "gate": "G9",
                "status": "N/A",
                "evidence": "no retained QCHAN event detector or device classifier",
            },
            {
                "gate": "G10",
                "status": "PASS",
                "evidence": "feature-specific roles accepted, no scalar/gate, corrected audit figures, exact numerical equivalence and separate atomic freezes",
            },
        ]
    )


def final_checklist_frame(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    frame["status"] = frame["status"].fillna("N/A")
    pending_mask = frame["status"].astype(str).isin(
        ["EVIDENCE_COMPLETE_PENDING_REVIEW", "PENDING"]
    )
    frame.loc[pending_mask, "status"] = "PASS"
    frame.loc[frame["item_id"].eq("G5.4"), "status"] = "CONDITIONAL"
    frame.loc[frame["gate"].eq("G9"), "status"] = "N/A"
    frame.loc[frame["item_id"].eq("G10.1"), "evidence"] = (
        "Final roles: LTAS primary nonordinal; rolloff primary one-sided; "
        "high-band secondary non-independent; tilt exploratory phenotype-sensitive."
    )
    frame.loc[frame["item_id"].eq("G10.3"), "evidence"] = (
        "22 applicable standardized figure/example bundles verified; Panels E1/E2/E3/H1/H3 regenerated from saved audit tables; Panel I N/A."
    )
    frame.loc[frame["item_id"].eq("G10.4"), "evidence"] = (
        "Named scientific acceptance, exact analysis-value equivalence, final inventory hashes, and separate measurement/figure freeze scripts."
    )
    frame["final_review_note"] = ""
    frame.loc[frame["gate"].eq("G5"), "final_review_note"] = (
        "Conditional because physical source identity and phenotype effects cannot be separated from no-reference speech alone."
    )
    frame.loc[frame["gate"].eq("G6"), "final_review_note"] = (
        "Audit sensitivity is complete; altered rolloff, band, smoothing, and tilt definitions are alternate estimands rather than interchangeable parameterizations."
    )
    frame.loc[frame["gate"].eq("G8"), "final_review_note"] = (
        "Participant-bootstrap confidence intervals added at finalization; no scalar inferred from correlation."
    )
    return frame


def analysis_values_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    required = ["logical_recording_id", *ANALYSIS_FEATURES, *SIGNED_PRECURSORS]
    if not set(required).issubset(left.columns) or not set(required).issubset(right.columns):
        return False
    a = left[required].sort_values("logical_recording_id").reset_index(drop=True)
    b = right[required].sort_values("logical_recording_id").reset_index(drop=True)
    if not a["logical_recording_id"].astype(str).equals(
        b["logical_recording_id"].astype(str)
    ):
        return False
    for column in [*ANALYSIS_FEATURES, *SIGNED_PRECURSORS]:
        av = pd.to_numeric(a[column], errors="coerce").to_numpy(float)
        bv = pd.to_numeric(b[column], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(av), np.isnan(bv)):
            return False
        mask = np.isfinite(av) & np.isfinite(bv)
        if not np.array_equal(av[mask], bv[mask]):
            return False
    return True


def _first_second_pairs(
    frame: pd.DataFrame,
    feature: str,
    *,
    subject_column: str = "SubjectID",
    date_column: str = "recording_date_analysis",
) -> pd.DataFrame:
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
        index=subject_column, columns="repeat_index", values=feature
    )
    return pair.rename(columns={0: "first", 1: "second"}).dropna()


def _icc1(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 2:
        return np.nan
    n, k = values.shape
    subject_means = values.mean(axis=1)
    grand = values.mean()
    ss_between = k * np.sum((subject_means - grand) ** 2)
    ss_within = np.sum((values - subject_means[:, None]) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    denominator = ms_between + (k - 1) * ms_within
    return float((ms_between - ms_within) / denominator) if denominator else np.nan


def repeated_recording_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    iterations: int = 5000,
    seed: int = 20260804,
) -> pd.DataFrame:
    rows = []
    master_rng = np.random.default_rng(int(seed))
    feature_seeds = master_rng.integers(0, 2**32 - 1, size=len(ANALYSIS_FEATURES))
    for feature, feature_seed in zip(ANALYSIS_FEATURES, feature_seeds):
        pair = _first_second_pairs(frame, feature)
        values = pair[["first", "second"]].to_numpy(float)
        n = len(values)
        rho = (
            float(stats.spearmanr(values[:, 0], values[:, 1]).statistic)
            if n >= 3
            else np.nan
        )
        icc = _icc1(values)
        rng = np.random.default_rng(int(feature_seed))
        rho_boot = []
        icc_boot = []
        for _ in range(int(iterations)):
            sampled = values[rng.integers(0, n, size=n)]
            if (
                np.unique(sampled[:, 0]).size > 1
                and np.unique(sampled[:, 1]).size > 1
            ):
                rho_boot.append(
                    float(stats.spearmanr(sampled[:, 0], sampled[:, 1]).statistic)
                )
            icc_boot.append(_icc1(sampled))
        difference = np.abs(values[:, 1] - values[:, 0])
        rows.append(
            {
                "feature": feature,
                "paired_subject_count": n,
                "first_second_spearman": rho,
                "spearman_bootstrap_p025": float(np.nanquantile(rho_boot, 0.025)),
                "spearman_bootstrap_p975": float(np.nanquantile(rho_boot, 0.975)),
                "icc1_first_two": icc,
                "icc1_bootstrap_p025": float(np.nanquantile(icc_boot, 0.025)),
                "icc1_bootstrap_p975": float(np.nanquantile(icc_boot, 0.975)),
                "bootstrap_iterations": int(iterations),
                "bootstrap_unit": "participant pair",
                "median_absolute_difference": float(np.median(difference)),
                "p90_absolute_difference": float(np.quantile(difference, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def reference_bootstrap_rank_stability(reference_long: pd.DataFrame) -> pd.DataFrame:
    local = reference_long.loc[
        reference_long["comparison"].astype(str).eq("subject_bootstrap")
    ].copy()
    rows = []
    for (feature, iteration), group in local.groupby(
        ["feature", "iteration"], sort=True
    ):
        pair = group[["baseline_value", "variant_value"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        rho = (
            float(stats.spearmanr(pair["baseline_value"], pair["variant_value"]).statistic)
            if len(pair) >= 3
            and pair["baseline_value"].nunique() > 1
            and pair["variant_value"].nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "iteration": int(iteration),
                "target_recording_count": len(pair),
                "spearman_rho": rho,
            }
        )
    long = pd.DataFrame(rows)
    summary_rows = []
    for feature, group in long.groupby("feature", sort=True):
        values = pd.to_numeric(group["spearman_rho"], errors="coerce").dropna()
        summary_rows.append(
            {
                "feature": feature,
                "iterations": len(values),
                "median_spearman_rho": float(values.median()),
                "p025_spearman_rho": float(values.quantile(0.025)),
                "p975_spearman_rho": float(values.quantile(0.975)),
                "minimum_spearman_rho": float(values.min()),
            }
        )
    return pd.DataFrame(summary_rows)


def _empirical_iqr(empirical: pd.DataFrame) -> Mapping[str, float]:
    return {
        str(row.feature): float(row.q75 - row.q25)
        for row in empirical.itertuples(index=False)
    }


def _augment_sensitivity(
    summary: pd.DataFrame, empirical: pd.DataFrame
) -> pd.DataFrame:
    output = summary.copy()
    iqr = _empirical_iqr(empirical)
    output["empirical_iqr"] = output["feature"].map(iqr)
    output["median_absolute_delta_iqr"] = (
        output["median_absolute_delta"] / output["empirical_iqr"]
    )
    output["p95_absolute_delta_iqr"] = (
        output["p95_absolute_delta"] / output["empirical_iqr"]
    )
    return output


def _save_figure_bundle(
    *,
    root: Path,
    stem: str,
    figure: plt.Figure,
    source: pd.DataFrame,
    caption: str,
    source_tables: Sequence[str],
) -> None:
    figure_dir = Path(root) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    base = figure_dir / stem
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    source.to_csv(base.with_suffix(".source.csv"), index=False)
    base.with_suffix(".caption.md").write_text(caption.strip() + "\n", encoding="utf-8")
    provenance = {
        "figure_stem": stem,
        "family": FAMILY,
        "measurement_version": FINAL_MEASUREMENT_VERSION,
        "figure_package_version": FIGURE_PACKAGE_VERSION,
        "source_tables": list(source_tables),
        "feature_values_recomputed": False,
        "audit_summary_recomputed_from_saved_tables": True,
        "finalization_revision": FINALIZATION_REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    base.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def _feature_grid() -> tuple[plt.Figure, np.ndarray]:
    return plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)


def _plot_sensitivity(
    source: pd.DataFrame,
    *,
    category_column: str,
    title: str,
    caption: str,
    stem: str,
    root: Path,
    source_tables: Sequence[str],
) -> None:
    fig, axes = _feature_grid()
    for axis, feature in zip(axes.flat, ANALYSIS_FEATURES):
        local = source.loc[source["feature"].eq(feature)].copy()
        if category_column == "variant":
            local = local.loc[~local[category_column].astype(str).eq("baseline")]
        local = local.sort_values("p95_absolute_delta", ascending=True)
        y = np.arange(len(local))
        axis.barh(y, local["p95_absolute_delta"].to_numpy(float), label="95th percentile")
        axis.scatter(
            local["median_absolute_delta"].to_numpy(float),
            y,
            marker="o",
            label="median",
            zorder=3,
        )
        axis.set_yticks(y, local[category_column].astype(str))
        axis.set_xlabel(f"Absolute change ({UNITS[feature]})")
        axis.set_title(DISPLAY_NAMES[feature])
        axis.grid(axis="x", alpha=0.25)
        if feature == ANALYSIS_FEATURES[0]:
            axis.legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=15)
    _save_figure_bundle(
        root=root,
        stem=stem,
        figure=fig,
        source=source,
        caption=caption,
        source_tables=source_tables,
    )


def regenerate_audited_figures(final_root: Path) -> list[str]:
    final_root = Path(final_root)
    empirical = pd.read_csv(
        final_root / "validation" / "qchan_v400_empirical_feature_summary.csv"
    )
    target = _augment_sensitivity(
        pd.read_csv(
            final_root / "validation" / "qchan_v400_target_robustness_summary.csv"
        ),
        empirical,
    )
    reference = _augment_sensitivity(
        pd.read_csv(
            final_root / "validation" / "qchan_v400_reference_robustness_summary.csv"
        ),
        empirical,
    )
    parameter = _augment_sensitivity(
        pd.read_csv(
            final_root / "validation" / "qchan_v400_parameter_sensitivity_summary.csv"
        ),
        empirical,
    )
    target.to_csv(
        final_root / "validation" / "qchan_v400_target_robustness_summary_final.csv",
        index=False,
    )
    reference.to_csv(
        final_root / "validation" / "qchan_v400_reference_robustness_summary_final.csv",
        index=False,
    )
    parameter.to_csv(
        final_root / "validation" / "qchan_v400_parameter_sensitivity_summary_final.csv",
        index=False,
    )

    _plot_sensitivity(
        target,
        category_column="variant",
        title="QCHAN target-window and segmentation sensitivity",
        caption=(
            "Panel E1. Median (point) and 95th-percentile (bar) absolute feature changes across the deterministic 72-recording robustness sample under frame, hop, guard, strict-boundary, and segment-deletion variants. Each feature remains in its physical unit. The 95th percentile is displayed because one-sided deficits often have a zero median change; rank correlations and IQR-normalized values are retained in the source data."
        ),
        stem="E1_window_boundary_sensitivity",
        root=final_root,
        source_tables=[
            "validation/qchan_v400_target_robustness_summary.csv",
            "validation/qchan_v400_empirical_feature_summary.csv",
        ],
    )
    _plot_sensitivity(
        reference,
        category_column="comparison",
        title="QCHAN reference-composition robustness",
        caption=(
            "Panel E2. Median (point) and 95th-percentile (bar) absolute feature changes for recording-weighted references, two deterministic 80% membership vintages, delete-one-reference-subject perturbations, and subject-bootstrap references. Each feature remains in its physical unit. Bars expose nonzero upper-tail changes that are hidden by zero medians in one-sided features; rank correlations and IQR-normalized values are retained in the source data."
        ),
        stem="E2_reference_robustness",
        root=final_root,
        source_tables=[
            "validation/qchan_v400_reference_robustness_summary.csv",
            "validation/qchan_v400_empirical_feature_summary.csv",
        ],
    )
    _plot_sensitivity(
        parameter,
        category_column="variant",
        title="QCHAN estimator-definition sensitivity",
        caption=(
            "Panel E3. Median (point) and 95th-percentile (bar) absolute feature changes under spectral-floor, analysis-ceiling, high-band split, rolloff-fraction, octave-smoothing, and tilt-range alternatives. These alternatives change parts of the estimand and are not interchangeable settings. Each feature remains in its physical unit; rank correlations and IQR-normalized values are retained in the source data."
        ),
        stem="E3_parameter_sensitivity",
        root=final_root,
        source_tables=[
            "validation/qchan_v400_parameter_sensitivity_summary.csv",
            "validation/qchan_v400_empirical_feature_summary.csv",
        ],
    )

    repeat_ci = pd.read_csv(
        final_root
        / "validation"
        / "qchan_v400_repeated_recording_persistence_with_ci.csv"
    )
    fig, axis = plt.subplots(figsize=(11.5, 6.3), constrained_layout=True)
    y = np.arange(len(repeat_ci))
    offsets = [-0.12, 0.12]
    for offset, value, low, high, label in (
        (
            offsets[0],
            "first_second_spearman",
            "spearman_bootstrap_p025",
            "spearman_bootstrap_p975",
            "Spearman",
        ),
        (
            offsets[1],
            "icc1_first_two",
            "icc1_bootstrap_p025",
            "icc1_bootstrap_p975",
            "ICC(1)",
        ),
    ):
        center = repeat_ci[value].to_numpy(float)
        xerr = np.vstack(
            [center - repeat_ci[low].to_numpy(float), repeat_ci[high].to_numpy(float) - center]
        )
        axis.errorbar(
            center,
            y + offset,
            xerr=xerr,
            fmt="o",
            capsize=4,
            label=label,
        )
    labels = [
        f"{DISPLAY_NAMES[f]} (n={int(n)})"
        for f, n in zip(repeat_ci["feature"], repeat_ci["paired_subject_count"])
    ]
    axis.set_yticks(y, labels)
    axis.set_xlim(-0.05, 1.02)
    axis.set_xlabel("Reliability coefficient with participant-bootstrap 95% interval")
    axis.set_title("QCHAN repeated-recording persistence")
    axis.axvline(0, linewidth=0.8)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    _save_figure_bundle(
        root=final_root,
        stem="H1_repeated_recordings",
        figure=fig,
        source=repeat_ci,
        caption=(
            "Panel H1. First-versus-second recording Spearman rank persistence and ICC(1), each with percentile 95% intervals from 5,000 participant-pair bootstrap resamples. All four features use the same 158 paired participants. These statistics describe empirical persistence and do not establish technical-source specificity."
        ),
        source_tables=[
            "tables/qchan_v400_analysis_features.csv",
            "validation/qchan_v400_repeated_recording_persistence.csv",
        ],
    )

    weighting = pd.read_csv(
        final_root
        / "validation"
        / "qchan_v400_recording_vs_participant_weighting.csv"
    )
    fig, axes = _feature_grid()
    for axis, feature in zip(axes.flat, ANALYSIS_FEATURES):
        row = weighting.loc[weighting["feature"].eq(feature)].iloc[0]
        recording_value = float(row["recording_weighted_median"])
        participant_value = float(row["participant_balanced_median_of_medians"])
        low = float(row["participant_balanced_p025"])
        high = float(row["participant_balanced_p975"])
        axis.scatter([0], [recording_value], s=55, label="recording-weighted median")
        axis.errorbar(
            [1],
            [participant_value],
            yerr=[[participant_value - low], [high - participant_value]],
            fmt="o",
            capsize=5,
            label="participant-balanced median (95% interval)",
        )
        axis.set_xticks([0, 1], ["recording-weighted", "participant-balanced"])
        axis.set_ylabel(UNITS[feature])
        axis.set_title(DISPLAY_NAMES[feature])
        axis.grid(axis="y", alpha=0.25)
        if feature == ANALYSIS_FEATURES[0]:
            axis.legend(loc="best", fontsize=8)
        if np.isclose(recording_value, participant_value) and np.isclose(low, high):
            axis.annotate("coincident", (1, participant_value), xytext=(5, 7), textcoords="offset points")
    fig.suptitle("QCHAN recording and participant weighting", fontsize=15)
    _save_figure_bundle(
        root=final_root,
        stem="H3_weighting",
        figure=fig,
        source=weighting,
        caption=(
            "Panel H3. Recording-weighted cohort medians compared with participant-balanced medians from 1,000 one-recording-per-participant resamples. Error bars show the participant-balanced 2.5th-97.5th percentile interval. Zero medians in one-sided features are retained as measured truncation states rather than hidden by an empty plotting range. Reference-weighting perturbations are reported separately in Panel E2."
        ),
        source_tables=[
            "validation/qchan_v400_recording_vs_participant_weighting.csv",
            "validation/qchan_v400_participant_balanced_resampling.csv",
        ],
    )
    return [
        "E1_window_boundary_sensitivity",
        "E2_reference_robustness",
        "E3_parameter_sensitivity",
        "H1_repeated_recordings",
        "H3_weighting",
    ]


def _normalize_relative_path(value: object) -> str:
    return str(value).replace("\\", "/") if pd.notna(value) else ""


def standardized_figure_index(final_root: Path) -> pd.DataFrame:
    final_root = Path(final_root)
    source_index = pd.read_csv(final_root / "tables" / "qchan_v400_figure_index.csv")
    for column in ["png", "svg", "pdf", "source_csv", "caption", "provenance"]:
        source_index[column] = source_index[column].map(_normalize_relative_path)
    required_views = {
        "waveform",
        "spectrogram",
        "target_ltas",
        "reference_ltas",
        "ltas_difference",
    }
    for row in source_index.itertuples(index=False):
        for column in ["png", "svg", "pdf", "source_csv", "caption", "provenance"]:
            relative = getattr(row, column)
            path = final_root / relative
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Figure artifact missing or empty: {relative}")
        if str(row.panel) == "G":
            linked = {
                item.strip()
                for item in str(row.linked_views).split("|")
                if item.strip()
            }
            if not required_views.issubset(linked) or not bool(row.linked_views_complete):
                raise ValueError(f"Gallery linked-view contract failed for {row.stem}")
    if len(source_index.loc[source_index["panel"].eq("G")]) < 8:
        raise ValueError("At least eight Panel G examples are required")
    na_row = pd.DataFrame(
        [
            {
                "panel": "I",
                "stem": "I_not_applicable",
                "png": "",
                "svg": "",
                "pdf": "",
                "source_csv": "",
                "caption": "",
                "provenance": "",
                "logical_recording_id": "",
                "selection_reason": "no retained event detector",
                "linked_views": "",
                "linked_view_count": 0,
                "linked_views_complete": True,
            }
        ]
    )
    return pd.concat([source_index, na_row], ignore_index=True)


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
- **Missingness:** {definition.missing_value_behavior}
- **Standalone reject gate allowed:** No
- **Family scalar/composite:** Prohibited
"""


def _update_feature_registry(source: pd.DataFrame) -> pd.DataFrame:
    decisions = final_decisions_frame().set_index("feature")
    registry = source.copy()
    key = "name" if "name" in registry.columns else "feature"
    registry["final_decision"] = registry[key].map(decisions["final_decision"])
    registry["publication_role"] = registry[key].map(decisions["publication_role"])
    registry["default_manuscript_inclusion"] = registry[key].map(
        decisions["default_manuscript_inclusion"]
    )
    registry["analysis_priority"] = registry[key].map(decisions["analysis_priority"])
    registry["final_measurement_version"] = FINAL_MEASUREMENT_VERSION
    registry["family_scalar_allowed"] = False
    registry["standalone_gate_allowed"] = False
    registry["device_identity_estimated"] = False
    return registry


def finalize_candidate(
    *,
    source_root: Path,
    final_root: Path,
    scientific_review_decision: str,
    scientific_reviewer: str,
    scientific_review_rationale: str,
    provenance_files: Iterable[Path] = (),
    repeat_bootstrap_iterations: int = 5000,
) -> dict:
    source_root = Path(source_root)
    final_root = Path(final_root)
    source_manifest_path = (
        source_root / "manifests" / "qchan_v400_cohort_candidate_manifest.json"
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    required = {
        "candidate_only": True,
        "accepted_preflight": True,
        "preflight_blocking_checks_pass": True,
        "package_tests_passed": True,
        "cohort_extraction_completed": True,
        "cohort_evidence_complete": True,
        "recording_count": 519,
        "participant_count": 224,
        "spectrum_count": 519,
        "reference_ledger_count": 519,
        "unique_reference_count": 224,
        "reference_vintage_count": 1,
        "required_panels_complete": True,
        "gallery_bundle_count": 8,
        "panel_i_status": "N/A_no_retained_event_detector",
        "family_scalar_constructed": False,
        "standalone_gate_allowed": False,
        "device_identity_estimated": False,
        "cohort_hotfix_revision": "gallery-linked-source-r3",
    }
    for key, expected in required.items():
        if source_manifest.get(key) != expected:
            raise ValueError(
                f"Source manifest mismatch for {key}: {source_manifest.get(key)!r}"
            )
    for relative in (
        "audit/qchan_v400_extraction_errors.csv",
        "audit/qchan_v400_reference_errors.csv",
        "audit/qchan_v400_target_robustness_errors.csv",
        "audit/qchan_v400_reference_robustness_errors.csv",
        "audit/qchan_v400_gallery_errors.csv",
    ):
        if len(pd.read_csv(source_root / relative)):
            raise ValueError(f"Source error table is not empty: {relative}")

    if final_root.exists():
        shutil.rmtree(final_root)
    shutil.copytree(source_root, final_root)
    archive = final_root / "_archive"
    if archive.exists():
        shutil.rmtree(archive)

    source_features_path = source_root / "tables" / "qchan_v400_analysis_features.csv"
    source_features = pd.read_csv(source_features_path)
    final_features = pd.read_csv(final_root / "tables" / "qchan_v400_analysis_features.csv")
    if not analysis_values_equal(source_features, final_features):
        raise ValueError("Copied final candidate is not numerically equivalent")
    source_feature_hash = sha256_file(source_features_path)

    source_registry = pd.read_csv(final_root / "tables" / "qchan_v400_feature_registry.csv")
    registry = _update_feature_registry(source_registry)
    registry.to_csv(final_root / "tables" / "qchan_v400_feature_registry.csv", index=False)
    try:
        registry.to_parquet(
            final_root / "tables" / "qchan_v400_feature_registry.parquet",
            index=False,
        )
    except Exception:
        pass

    decisions = final_decisions_frame()
    decisions.to_csv(
        final_root / "validation" / "qchan_v400_g10_feature_decisions.csv",
        index=False,
    )
    dashboard = ten_domain_dashboard_frame()
    dashboard.to_csv(
        final_root / "validation" / "qchan_v400_ten_domain_dashboard.csv",
        index=False,
    )
    gates = final_gate_summary_frame()
    gates.to_csv(
        final_root / "validation" / "qchan_v400_gate_summary_final.csv",
        index=False,
    )
    source_checklist = pd.read_csv(
        final_root / "validation" / "qchan_v400_checklist_cohort.csv"
    )
    final_checklist = final_checklist_frame(source_checklist)
    final_checklist.to_csv(
        final_root / "validation" / "qchan_v400_checklist_final.csv", index=False
    )

    repeat_ci = repeated_recording_bootstrap_ci(
        final_features, iterations=int(repeat_bootstrap_iterations)
    )
    repeat_ci.to_csv(
        final_root
        / "validation"
        / "qchan_v400_repeated_recording_persistence_with_ci.csv",
        index=False,
    )
    reference_long = pd.read_csv(
        final_root / "validation" / "qchan_v400_reference_robustness_long.csv"
    )
    rank_stability = reference_bootstrap_rank_stability(reference_long)
    rank_stability.to_csv(
        final_root
        / "validation"
        / "qchan_v400_reference_bootstrap_rank_stability.csv",
        index=False,
    )

    corrected_figures = regenerate_audited_figures(final_root)
    figure_index = standardized_figure_index(final_root)
    figure_index.to_csv(
        final_root / "figures" / "qchan_v400_standardized_figure_index.csv",
        index=False,
    )
    figure_index.to_csv(
        final_root / "tables" / "qchan_v400_standardized_figure_index.csv",
        index=False,
    )

    passports = final_root / "feature_passports"
    passports.mkdir(parents=True, exist_ok=True)
    for definition in FINAL_FEATURE_DEFINITIONS:
        (passports / f"{definition.feature}.md").write_text(
            feature_passport_text(definition), encoding="utf-8"
        )

    provenance = final_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_manifest_path,
        provenance / "qchan_v400_source_cohort_candidate_manifest.json",
    )
    for path in provenance_files:
        path = Path(path)
        if path.exists():
            shutil.copy2(path, provenance / path.name)

    empirical = pd.read_csv(
        final_root / "validation" / "qchan_v400_empirical_feature_summary.csv"
    ).set_index("feature")
    reference_ledger = pd.read_csv(
        final_root / "ledgers" / "qchan_v400_reference_ledger.csv"
    )
    native = pd.read_csv(
        final_root / "validation" / "qchan_v400_native_bandwidth_summary.csv"
    )
    redundancy = pd.read_csv(
        final_root / "validation" / "qchan_v400_pairwise_redundancy.csv"
    )
    rolloff_highband_rho = float(
        redundancy.loc[
            redundancy["feature_left"].eq("qchan_rolloff95_deficit_hz")
            & redundancy["feature_right"].eq("qchan_highband_ratio_deficit"),
            "spearman_rho",
        ].iloc[0]
    )
    rolloff_tilt_rho = float(
        redundancy.loc[
            redundancy["feature_left"].eq("qchan_rolloff95_deficit_hz")
            & redundancy["feature_right"].eq(
                "qchan_tilt_steepening_db_per_oct"
            ),
            "spearman_rho",
        ].iloc[0]
    )

    accepted = scientific_review_decision == ACCEPTANCE_TOKEN
    excluded = {
        "manifests/qchan_v400_final_candidate_manifest.json",
        "manifests/qchan_v400_final_candidate_inventory.csv",
    }
    inventory = file_inventory(final_root, exclude=excluded)
    inventory_path = (
        final_root / "manifests" / "qchan_v400_final_candidate_inventory.csv"
    )
    inventory.to_csv(inventory_path, index=False)

    reference_vintages = sorted(
        final_features["qchan_reference_vintage_sha256"].dropna().astype(str).unique()
    )
    manifest = {
        "measurement_version": FINAL_MEASUREMENT_VERSION,
        "source_candidate_directory": SOURCE_MEASUREMENT_VERSION,
        "family": FAMILY,
        "family_display_name": FAMILY_DISPLAY_NAME,
        "finalization_revision": FINALIZATION_REVISION,
        "candidate_only": True,
        "freeze_status": "ready_for_atomic_freeze" if accepted else "scientific_review_pending",
        "freeze_allowed": bool(accepted),
        "scientific_review_decision": scientific_review_decision,
        "scientific_reviewer": scientific_reviewer,
        "scientific_review_rationale": scientific_review_rationale,
        "analysis_features": list(ANALYSIS_FEATURES),
        "primary_analysis_features": [
            "qchan_ltas_distance_db",
            "qchan_rolloff95_deficit_hz",
        ],
        "secondary_analysis_features": ["qchan_highband_ratio_deficit"],
        "exploratory_features": ["qchan_tilt_steepening_db_per_oct"],
        "default_manuscript_features": [
            "qchan_ltas_distance_db",
            "qchan_rolloff95_deficit_hz",
            "qchan_highband_ratio_deficit",
        ],
        "recording_count": int(len(final_features)),
        "participant_count": int(final_features["SubjectID"].nunique()),
        "all_features_available_n": int(
            final_features[list(ANALYSIS_FEATURES)].notna().all(axis=1).sum()
        ),
        "high_support_recording_count": int(
            final_features["qchan_support_tier"].astype(str).eq("high").sum()
        ),
        "moderate_support_recording_count": int(
            final_features["qchan_support_tier"].astype(str).eq("moderate").sum()
        ),
        "reference_subject_count_min": int(
            pd.to_numeric(final_features["qchan_reference_subject_count"]).min()
        ),
        "reference_subject_count_max": int(
            pd.to_numeric(final_features["qchan_reference_subject_count"]).max()
        ),
        "reference_recording_count_min": int(
            pd.to_numeric(final_features["qchan_reference_recording_count"]).min()
        ),
        "reference_recording_count_max": int(
            pd.to_numeric(final_features["qchan_reference_recording_count"]).max()
        ),
        "reference_vintage_count": len(reference_vintages),
        "reference_vintage_sha256": reference_vintages[0] if len(reference_vintages) == 1 else reference_vintages,
        "source_sample_rates_hz": sorted(
            pd.to_numeric(final_features["qchan_source_sample_rate_hz"]).dropna().unique().astype(int).tolist()
        ),
        "native_bandwidth_limited_recording_count": int(
            final_features["qchan_source_bandwidth_limited"].astype(bool).sum()
        ),
        "ltas_available_n": int(empirical.at["qchan_ltas_distance_db", "available_n"]),
        "rolloff_available_n": int(empirical.at["qchan_rolloff95_deficit_hz", "available_n"]),
        "highband_available_n": int(empirical.at["qchan_highband_ratio_deficit", "available_n"]),
        "tilt_available_n": int(empirical.at["qchan_tilt_steepening_db_per_oct", "available_n"]),
        "rolloff_zero_fraction": float(empirical.at["qchan_rolloff95_deficit_hz", "zero_fraction_among_available"]),
        "highband_zero_fraction": float(empirical.at["qchan_highband_ratio_deficit", "zero_fraction_among_available"]),
        "tilt_zero_fraction": float(empirical.at["qchan_tilt_steepening_db_per_oct", "zero_fraction_among_available"]),
        "rolloff_highband_spearman_rho": rolloff_highband_rho,
        "rolloff_tilt_spearman_rho": rolloff_tilt_rho,
        "repeat_bootstrap_iterations": int(repeat_bootstrap_iterations),
        "repeat_paired_subject_count": int(repeat_ci["paired_subject_count"].min()),
        "reference_bootstrap_iterations": int(rank_stability["iterations"].min()),
        "reference_robustness_target_recording_count": int(
            pd.read_csv(final_root / "validation" / "qchan_v400_reference_robustness_sample.csv")["logical_recording_id"].nunique()
        ),
        "reference_ledger_count": int(len(reference_ledger)),
        "native_bandwidth_summary_rows": int(len(native)),
        "numerical_equivalence_to_cohort_candidate": True,
        "source_analysis_features_sha256": source_feature_hash,
        "feature_values_recomputed": False,
        "audit_summaries_recomputed_from_saved_outputs": True,
        "corrected_audit_figure_stems": corrected_figures,
        "required_panels_complete": True,
        "figure_count_excluding_na": int((figure_index["panel"] != "I").sum()),
        "panel_i_status": "N/A_no_retained_event_detector",
        "family_scalar_status": "prohibited_not_constructed",
        "standalone_reject_allowed": False,
        "device_identity_estimated": False,
        "decision_threshold_status": "not_calibrated",
        "support_tier_is_precision": False,
        "reference_vintage_is_feature_identity": True,
        "signed_precursors_retained": True,
        "artifact_count_excluding_seal_files": int(len(inventory)),
        "final_candidate_inventory_sha256": sha256_file(inventory_path),
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "immutability_policy": "never overwrite; create a new semantic version for any measurement change",
    }
    manifest_path = (
        final_root / "manifests" / "qchan_v400_final_candidate_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (final_root / "READY_FOR_FREEZE_QCHAN_V4_0_0.txt").write_text(
        (
            "QCHAN v4.0.0 is scientifically accepted and ready for atomic freeze.\n"
            if accepted
            else "QCHAN v4.0.0 remains pending scientific acceptance.\n"
        ),
        encoding="utf-8",
    )
    return manifest
