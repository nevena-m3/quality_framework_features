#!/usr/bin/env python3
"""Close QTEMP as a negative/limited analytical-validation outcome.

This script freezes the implementation, evidence, and final feature decisions.
It does not label G9 as passed and does not place QTEMP values in the validated
primary feature table. Four outputs remain available for descriptive or
exploratory sensitivity work; splice remains dropped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from scipy.io import wavfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "/tmp")) / "qtemp-final-mpl"))
import matplotlib.pyplot as plt


SOURCE_VERSION = "qtemp-v1.0.0-candidate-g9-pending"
FINAL_VERSION = "qtemp-v1.0.0-analytical-final-no-retained"
FEATURES = [
    "qtemp_dropout_duration_fraction",
    "qtemp_dropout_event_rate_per_min",
    "qtemp_frozen_audio_duration_fraction",
    "qtemp_frozen_audio_event_rate_per_min",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> int:
    manifest = root / "manifests" / "qtemp_v100_artifact_sha256.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest}")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    for row in rows:
        path = root / row["relative_path"]
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {path}")
    return len(rows)


def archive_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    archive_root = path.parent / "_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = archive_root / f"{path.name}_{stamp}"
    counter = 1
    while target.exists():
        target = archive_root / f"{path.name}_{stamp}_{counter}"
        counter += 1
    shutil.move(str(path), str(target))
    return target


def feature_decisions() -> pd.DataFrame:
    rows = [
        {
            "feature": "qtemp_dropout_duration_fraction",
            "role": "exploratory descriptive burden",
            "decision": "EXPLORATORY_NOT_PRIMARY_CONSTRUCT_SPECIFICITY_LIMITED",
            "primary_validated_set": False,
            "rationale": "Only 2/519 recordings were positive, and accepted pause-boundary ambiguity prevents promotion to the validated primary set.",
            "permitted_claim": "Algorithmically detected bracketed dropout-like decoded support, reported descriptively with the documented pause-boundary construct-specificity limitation.",
            "prohibited_claim": "Validated packet loss, network failure, missing speech content, physiological marker, or primary inferential biomarker.",
        },
        {
            "feature": "qtemp_dropout_event_rate_per_min",
            "role": "exploratory same-ledger frequency",
            "decision": "EXPLORATORY_NOT_PRIMARY_CONSTRUCT_SPECIFICITY_LIMITED",
            "primary_validated_set": False,
            "rationale": "Secondary frequency view of the same six accepted events; it is not independent of duration burden and is not promoted to the validated primary set.",
            "permitted_claim": "Exploratory frequency of algorithmically accepted bracketed dropout-like events per eligible minute.",
            "prohibited_claim": "Independent dimension, calibrated failure rate, packet-loss rate, or primary inferential biomarker.",
        },
        {
            "feature": "qtemp_frozen_audio_duration_fraction",
            "role": "monitoring output; zero variation",
            "decision": "MONITORING_ONLY_ZERO_VARIATION_POSITIVE_BEHAVIOR_UNVERIFIED",
            "primary_validated_set": False,
            "rationale": "Synthetic and participant-disjoint real-speech injection recovery passed, but 0/519 cohort recordings were positive, leaving empirical positive behavior unverified.",
            "permitted_claim": "No event meeting the registered near-exact consecutive decoded-repetition rule was observed in this cohort.",
            "prohibited_claim": "Validated detection of all freezes, packet-loss concealment, buffering, or a robust real-world positive-event biomarker.",
        },
        {
            "feature": "qtemp_frozen_audio_event_rate_per_min",
            "role": "monitoring same-ledger frequency; zero variation",
            "decision": "MONITORING_ONLY_ZERO_VARIATION_POSITIVE_BEHAVIOR_UNVERIFIED",
            "primary_validated_set": False,
            "rationale": "All cohort values are zero and event grouping is algorithm-dependent; empirical positive behavior is unverified.",
            "permitted_claim": "No accepted near-exact repetition event was observed per eligible minute in this cohort.",
            "prohibited_claim": "Independent evidence, calibrated failure threshold, or primary inferential biomarker.",
        },
        {
            "feature": "qtemp_splice_discontinuity_rate_per_min",
            "role": "dropped",
            "decision": "DROP_FAILED_ANALYTICAL_VALIDATION",
            "primary_validated_set": False,
            "rationale": "Held-out real-speech boundary recovery failed; the earlier detector produced 318/519 positives and was sample-rate dependent.",
            "permitted_claim": "None in the final QTEMP feature set.",
            "prohibited_claim": "Do not export or analyze as a retained feature.",
        },
    ]
    return pd.DataFrame(rows)


def rewrite_gallery_index(final_root: Path) -> None:
    path = final_root / "galleries" / "qtemp_v100_gallery_index.csv"
    frame = pd.read_csv(path)
    for idx, row in frame.iterrows():
        letter = str(row["panel"])
        directories = list((final_root / "figures").glob(f"panel-{letter}_*"))
        if len(directories) != 1:
            raise RuntimeError(f"Expected one directory for Panel {letter}")
        directory = directories[0]
        for column, extension in [
            ("svg_path", ".svg"), ("pdf_path", ".pdf"), ("png_path", ".png"),
            ("source_data_path", ".csv"), ("caption_path", ".md"),
            ("provenance_path", ".json"),
        ]:
            files = [p for p in directory.glob(f"*{extension}") if not p.name.endswith(".tmp.png")]
            if len(files) != 1:
                raise RuntimeError(f"Panel {letter}: expected one {extension} file")
            frame.at[idx, column] = files[0].relative_to(final_root).as_posix()
    frame.loc[frame.panel.eq("I"), "status"] = "N/A_NO_RETAINED_PRIMARY_EVENT_FEATURES"
    frame.loc[frame.panel.eq("J"), "status"] = "FINAL_NO_PRIMARY_FEATURES"
    frame.to_csv(path, index=False)


def remove_superseded_event_workflow(final_root: Path) -> None:
    gallery_root = final_root / "galleries"
    for path in gallery_root.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
    retained_validation_files = {
        "qtemp_v100_feature_decisions.csv",
        "qtemp_v100_gate_summary.csv",
        "qtemp_v100_validation_checklist.csv",
    }
    for path in (final_root / "validation").iterdir():
        if path.is_file() and path.name not in retained_validation_files:
            path.unlink()
    for path in [
        final_root / "audit" / "QTEMP_v100_FREEZE_DECISION.md",
        final_root / "manifests" / "qtemp_v100_artifact_sha256.csv",
        final_root / "manifests" / "qtemp_v100_validation_status.json",
    ]:
        if path.exists():
            path.unlink()
    for path in final_root.rglob("*.tmp.png"):
        path.unlink()


def replace_panel_b(final_root: Path) -> None:
    directory = final_root / "figures" / "panel-B_discriminant-specificity"
    base = directory / "qtemp_v100_panel-B_discriminant-specificity"
    source = pd.read_csv(base.with_suffix(".csv"))
    source = source.rename(columns={"reviewed_units": "evaluated_units"})
    source.loc[source["label"].eq("rejected high-score candidates"), "evidence"] = "held-out hard negatives"
    source.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".md").write_text(
        "**Panel B - Discriminant specificity.** Responses to matched competing synthetic mechanisms, periodic and connected-speech proxies, 12 unmodified participant-disjoint real-speech recordings, and six held-out high-score hard negatives. Bars show unit-level accepted-positive fractions with numerator/denominator labels; event and indeterminate-candidate counts are retained in the source table. No accepted positives occurred in these controls. Pause boundaries, ordinary silence, stop closure, periodic voicing, and low-energy speech remain the principal construct-specificity risks; these limitations prevent promotion to the validated primary feature set. No diagnosis or clinical label was used.\n",
        encoding="utf-8",
    )
    base.with_suffix(".json").write_text(json.dumps({
        "family": "QTEMP",
        "measurement_version": FINAL_VERSION,
        "panel": "B",
        "slug": "discriminant-specificity",
        "status": "GENERATED_CONDITIONAL",
        "created_utc": utc_now(),
        "clinical_labels_used": False,
        "input_files": [],
    }, indent=2), encoding="utf-8")


def _mono_float(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples)
    if values.ndim > 1:
        values = values[:, 0]
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        scale = float(max(abs(info.min), info.max))
        return values.astype(np.float64) / scale
    return values.astype(np.float64)


def replace_panel_g(project_root: Path, final_root: Path) -> None:
    directory = final_root / "figures" / "panel-G_signal-examples"
    base = directory / "qtemp_v100_panel-G_signal-examples"
    original = pd.read_csv(base.with_suffix(".csv"))
    rename = {
        "review_id": "example_id",
        "review_stratum": "selection_stratum",
    }
    keep = [
        "review_id", "gallery_id", "logical_recording_id", "event_type", "event_subtype",
        "review_stratum", "candidate_id", "event_id", "start_sec", "end_sec",
        "selected_channel_index", "native_channel_count", "native_sample_rate_hz", "score",
        "disposition_reason", "candidate_evidence_json", "selection_category",
    ]
    source = original[keep].rename(columns=rename).copy()
    source["selection_stratum"] = source["selection_stratum"].replace({"random_event_free": "candidate_free"})
    source.to_csv(base.with_suffix(".csv"), index=False)

    gallery = (
        project_root / "outputs" / "02_features" / "temporal_discontinuity"
        / "qtemp-v0.3.1-finalization" / "gallery"
    )
    labels = [
        "Algorithmically accepted dropout-like interval",
        "Accepted interval at a pause boundary",
        "Algorithmically rejected high-score candidate",
        "Deterministic candidate-free excerpt",
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 12.5), gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    for row_index, (_, row) in enumerate(source.iterrows()):
        example_id = str(row["example_id"])
        matches = sorted(gallery.glob(f"*{example_id}*_channel.wav"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one deterministic signal excerpt for example {example_id}; found {len(matches)}")
        audio_path = matches[0]
        sample_rate, samples = wavfile.read(audio_path)
        signal = _mono_float(samples)
        time = np.arange(signal.size, dtype=float) / float(sample_rate)
        duration = signal.size / float(sample_rate)
        interval_duration = max(0.0, float(row["end_sec"]) - float(row["start_sec"]))
        display_duration = max(interval_duration, 0.02)
        interval_start = duration / 2.0 - display_duration / 2.0
        interval_end = duration / 2.0 + display_duration / 2.0

        waveform_ax = axes[row_index, 0]
        waveform_ax.plot(time, signal, color="#1A202C", linewidth=0.55)
        waveform_ax.axvspan(interval_start, interval_end, color="#ED8936", alpha=0.28)
        waveform_ax.set_xlim(0, duration)
        waveform_ax.set_ylabel("Decoded amplitude")
        waveform_ax.set_title(f"{chr(97 + row_index)}) {labels[row_index]}", loc="left", fontsize=11, weight="bold")
        waveform_ax.grid(alpha=0.15)

        spectrum_ax = axes[row_index, 1]
        nfft = min(1024, max(128, 2 ** int(np.floor(np.log2(max(128, signal.size // 16))))))
        noverlap = int(nfft * 0.75)
        with np.errstate(divide="ignore", invalid="ignore"):
            spectrum_ax.specgram(signal, NFFT=nfft, Fs=sample_rate, noverlap=noverlap, cmap="magma")
        spectrum_ax.axvspan(interval_start, interval_end, color="#ED8936", alpha=0.22)
        spectrum_ax.set_xlim(0, duration)
        spectrum_ax.set_ylim(0, min(12000, sample_rate / 2))
        spectrum_ax.set_ylabel("Frequency (Hz)")
        spectrum_ax.set_title("Local spectrogram", fontsize=10)
        if row_index == 3:
            waveform_ax.set_xlabel("Excerpt-relative time (s)")
            spectrum_ax.set_xlabel("Excerpt-relative time (s)")
    fig.suptitle("Panel G. Deterministic signal-linked QTEMP examples", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=260)
    plt.close(fig)
    base.with_suffix(".md").write_text(
        "**Panel G - Signal-linked examples.** Four deterministic waveform/spectrogram examples link the tabular evidence to the decoded signal: an algorithmically accepted dropout-like interval, an accepted interval at a pause boundary, a rejected high-score candidate, and a candidate-free excerpt. Orange shading marks the registered interval or target location. The pause-boundary example illustrates the principal construct-specificity limitation and supports the decision not to promote QTEMP event outputs to the validated primary feature set. No diagnosis or clinical label was used.\n",
        encoding="utf-8",
    )
    base.with_suffix(".json").write_text(json.dumps({
        "family": "QTEMP",
        "measurement_version": FINAL_VERSION,
        "panel": "G",
        "slug": "signal-examples",
        "status": "GENERATED_CONDITIONAL",
        "created_utc": utc_now(),
        "clinical_labels_used": False,
        "selection_method": "deterministic fixed evidence categories",
        "input_files": [],
    }, indent=2), encoding="utf-8")


def replace_panel_j(final_root: Path) -> None:
    directory = final_root / "figures" / "panel-J_ml-handoff"
    base = directory / "qtemp_v100_panel-J_ml-handoff"
    rows = [
        ["Dropout duration fraction", "exploratory descriptive", "not primary", "eligible native duration", "accepted + indeterminate counts"],
        ["Dropout event rate", "exploratory same-ledger", "not primary", "eligible native duration", "accepted + indeterminate counts"],
        ["Frozen duration fraction", "zero-variation monitoring", "not primary", "eligible native duration", "accepted + indeterminate counts"],
        ["Frozen event rate", "zero-variation same-ledger", "not primary", "eligible native duration", "accepted + indeterminate counts"],
    ]
    source = pd.DataFrame(rows, columns=["feature", "final_role", "primary_analysis", "support", "event_evidence"])
    source["measurement_version"] = FINAL_VERSION
    source.to_csv(base.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(14.5, 5.2))
    ax.axis("off")
    ax.set_title("Panel J. Final QTEMP analytical handoff", fontsize=17, weight="bold", pad=24)
    display_frame = source[["feature", "final_role", "primary_analysis", "support", "event_evidence"]].rename(columns={
        "feature": "Feature",
        "final_role": "Final role",
        "primary_analysis": "Primary analysis",
        "support": "Support",
        "event_evidence": "Event evidence",
    })
    table = ax.table(cellText=display_frame.values, colLabels=display_frame.columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.2)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2B6CB0")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EBF4FF")
    fig.tight_layout()
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    base.with_suffix(".md").write_text(
        "**Panel J - Final analytical handoff.** The four deterministic QTEMP outputs retain feature-specific status, eligible native exposure, accepted and indeterminate counts, rate uncertainty, and measurement version. The dropout pair is restricted to exploratory descriptive use; the frozen-audio pair is restricted to zero-variation monitoring. No QTEMP output is eligible for the validated primary analysis. Duration fraction and event rate remain same-ledger companions rather than independent dimensions.\n",
        encoding="utf-8",
    )
    base.with_suffix(".json").write_text(json.dumps({
        "family": "QTEMP",
        "measurement_version": FINAL_VERSION,
        "panel": "J",
        "slug": "ml-handoff",
        "status": "FINAL_NO_PRIMARY_FEATURES",
        "created_utc": utc_now(),
        "clinical_labels_used": False,
        "input_files": [],
    }, indent=2), encoding="utf-8")


def normalize_panel_provenance(final_root: Path) -> None:
    for json_path in sorted((final_root / "figures").glob("panel-*/*.json")):
        panel_dir = json_path.parent
        data = json.loads(json_path.read_text(encoding="utf-8"))
        old_version = data.get("measurement_version", SOURCE_VERSION)
        data["measurement_version"] = FINAL_VERSION
        data["source_measurement_version"] = old_version
        data["created_utc"] = utc_now()
        data["finalization_state"] = "FINAL_ANALYTICAL_IMPLEMENTATION_FREEZE_NO_RETAINED_PRIMARY_FEATURES"
        data.pop("code_file", None)
        data.pop("source_table", None)
        inputs = []
        for item in data.get("input_files", []):
            source_name = Path(str(item.get("path", item.get("source_name", "")))).name
            if source_name:
                inputs.append({"source_name": source_name, "sha256": item.get("sha256")})
        data["input_files"] = inputs
        csv_files = list(panel_dir.glob("*.csv"))
        if len(csv_files) == 1:
            data["source_table_relative"] = csv_files[0].relative_to(final_root).as_posix()
            data["source_table_sha256"] = sha256_file(csv_files[0])
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def replace_panel_i(final_root: Path) -> None:
    directory = final_root / "figures" / "panel-I_event-verification"
    base = directory / "qtemp_v100_panel-I_event-verification"
    source = pd.DataFrame([
        ["validated primary QTEMP features", 0, "FINAL"],
        ["exploratory descriptive outputs", 2, "FINAL"],
        ["monitoring zero-variation outputs", 2, "FINAL"],
        ["dropped outputs", 1, "FINAL"],
    ], columns=["final_category", "feature_count", "status"])
    source.to_csv(base.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off")
    ax.text(0.5, 0.84, "Panel I. Final event-feature disposition", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(0.5, 0.66, "N/A for the validated primary set", ha="center", va="center", fontsize=16, color="#2B6CB0", weight="bold")
    lines = [
        "0 QTEMP event features retained in the validated primary feature set",
        "2 dropout-ledger outputs preserved for exploratory descriptive use",
        "2 repetition-ledger outputs preserved for zero-variation monitoring",
        "1 splice output dropped after failed analytical validation",
        "Family closed as a final limited/negative feature-selection result",
    ]
    for index, line in enumerate(lines):
        ax.text(0.5, 0.50 - index * 0.085, line, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=320)
    plt.close(fig)
    base.with_suffix(".md").write_text(
        "**Panel I - Final event-feature disposition.** QTEMP contributes no feature to the validated primary set. Two dropout-ledger outputs remain exploratory descriptive measures; two repetition-ledger outputs remain zero-variation monitoring measures; the splice output is dropped. Gate G9 is not applicable because no QTEMP event detector is retained as a primary validated feature.\n",
        encoding="utf-8",
    )
    base.with_suffix(".json").write_text(json.dumps({
        "family": "QTEMP",
        "measurement_version": FINAL_VERSION,
        "panel": "I",
        "status": "N/A_NO_RETAINED_PRIMARY_EVENT_FEATURES",
        "created_utc": utc_now(),
        "source_table": "figures/panel-I_event-verification/qtemp_v100_panel-I_event-verification.csv",
        "clinical_labels_used": False,
    }, indent=2), encoding="utf-8")


def write_final_records(final_root: Path, source_manifest_entries: int) -> None:
    decisions = feature_decisions()
    decisions.to_csv(final_root / "validation" / "qtemp_v100_feature_decisions.csv", index=False)

    checklist_path = final_root / "validation" / "qtemp_v100_validation_checklist.csv"
    checklist = pd.read_csv(checklist_path)
    checklist["measurement_version"] = FINAL_VERSION
    if "item" in checklist.columns:
        checklist = checklist.drop(columns=["item"])
    checklist.loc[checklist.id.eq("V2"), ["status", "blocking_fail", "evidence_path_notes"]] = [
        "N/A", False,
        "No event detector is retained in the validated primary feature set; Panel I records the final disposition.",
    ]
    checklist.loc[checklist.id.eq("V2"), "checklist_item"] = (
        "External event verification applies only when an event feature is retained in the validated primary set."
    )
    checklist.loc[checklist.id.eq("V3"), ["status", "blocking_fail", "evidence_path_notes"]] = [
        "N/A", False,
        "No primary retained event feature; exploratory/monitoring limitations are recorded in the feature decisions.",
    ]
    checklist.loc[checklist.id.eq("S5"), ["status", "blocking_fail", "evidence_path_notes"]] = [
        "N/A", False,
        "Arbitrary categorical support tiers are not applicable; continuous eligible exposure and exact count intervals are exported.",
    ]
    checklist.loc[checklist.id.eq("R5"), ["status", "blocking_fail", "evidence_path_notes"]] = [
        "N/A", False,
        "No cohort reference model is estimated for QTEMP; subject-balanced model stability is therefore not applicable.",
    ]
    checklist.loc[checklist.id.eq("ML3"), ["status", "blocking_fail", "evidence_path_notes"]] = [
        "CONDITIONAL", False,
        "Provenance-preserving handoff is defined for exploratory/monitoring sensitivity work; no QTEMP output is eligible for primary ML analysis.",
    ]
    checklist.loc[checklist.id.eq("G11"), ["status", "blocking_fail", "evidence_path_notes"]] = [
        "PASS", False,
        "Final analytical implementation archive, figures, tables, decisions, provenance, and SHA-256 manifest sealed.",
    ]
    checklist.loc[checklist.id.eq("G12"), ["status", "blocking_fail", "evidence_path_notes"]] = [
        "CONDITIONAL", False,
        "Manuscript must count zero QTEMP features in the validated primary set; four outputs may be described as exploratory/monitoring only.",
    ]
    checklist["completed_utc"] = utc_now()
    checklist.to_csv(checklist_path, index=False)

    gates = pd.DataFrame([
        ["G1", "Contract and provenance", "PASS", True, "native decoded view and observational claim boundary pinned"],
        ["G2", "Numerical correctness", "PASS", True, "determinism, exact reconstruction, missingness, and edge cases pass"],
        ["G3", "Transformation behavior", "PASS", True, "gain, polarity, time shift, native-view, and sample-rate behavior characterized"],
        ["G4", "Target recovery", "PASS", True, "controlled and participant-disjoint real-speech injection recovery passes within registered scope"],
        ["G5", "Discriminant specificity", "CONDITIONAL", True, "tested controls pass; pause-boundary and periodic-speech limitations require exploratory roles"],
        ["G6", "Support and robustness", "PASS", True, "exposure, uncertainty, merge, sample-rate, and parameter audits complete"],
        ["G7", "Empirical plausibility", "PASS", True, "519 recordings; 2 dropout-positive and 0 frozen-positive recordings"],
        ["G8", "Reliability and redundancy", "CONDITIONAL", True, "exact reconstruction; zero-dominated persistence; same-ledger summaries flagged"],
        ["G9", "Event verification", "N/A_NO_RETAINED_PRIMARY_EVENT_FEATURES", False, "no QTEMP event detector is retained in the validated primary feature set"],
        ["G10", "Final family disposition", "FINALIZED_NO_RETAINED_PRIMARY_FEATURES", True, "implementation and negative/limited feature decisions frozen; no QTEMP feature enters validated primary set"],
    ], columns=["gate", "requirement", "state", "passed", "evidence"])
    gates.to_csv(final_root / "validation" / "qtemp_v100_gate_summary.csv", index=False)

    decision_text = f"""# QTEMP final analytical disposition

Generated: {utc_now()}

## Final outcome

QTEMP is closed as an **analytical implementation freeze with no retained primary validated features**. Gate G9 is not applicable because no QTEMP event detector is retained in the validated primary feature set.

- Four deterministic outputs are preserved for descriptive, exploratory, monitoring, and downstream sensitivity work.
- None of the four belongs to the validated primary feature set for Paper 1.
- The splice-discontinuity feature remains dropped.
- The implementation, evidence, figures, tables, decisions, provenance, and hashes are immutable in this archive.
- This is a final negative/limited feature-selection result, not a pending workflow.

## Manuscript boundary

Permitted: report that 2/519 recordings contained algorithmically accepted bracketed dropout-like decoded support and 0/519 contained an event meeting the registered near-exact repetition rule, explicitly labeling these findings exploratory or monitoring-only and outside the validated primary feature set.

Prohibited: packet-loss, network-failure, buffering, concealment, missing-speech, physiological-biomarker, or fully validated real-world event-detector claims.

## Event-feature disposition

The event-detector outputs remain available only under their frozen exploratory/monitoring roles. Panel I and the feature-decision table record that no QTEMP event feature enters the validated primary analysis set.
"""
    (final_root / "audit" / "QTEMP_v100_FINAL_ANALYTICAL_DECISION.md").write_text(decision_text, encoding="utf-8")

    status = {
        "family": "QTEMP",
        "measurement_version": FINAL_VERSION,
        "created_utc": utc_now(),
        "finalization_state": "FINAL_ANALYTICAL_IMPLEMENTATION_FREEZE_NO_RETAINED_PRIMARY_FEATURES",
        "source_measurement_version": SOURCE_VERSION,
        "source_manifest_entries_verified": source_manifest_entries,
        "recordings": 519,
        "participants": 224,
        "validated_primary_features": [],
        "exploratory_descriptive_features": FEATURES[:2],
        "monitoring_zero_variation_features": FEATURES[2:],
        "dropped_features": ["qtemp_splice_discontinuity_rate_per_min"],
        "g9_status": "N/A_NO_RETAINED_PRIMARY_EVENT_FEATURES",
        "g9_applicable": False,
        "g9_passed": False,
        "implementation_frozen": True,
        "feature_decisions_frozen": True,
        "publication_ready_under_no_primary_qtemp_claim": True,
        "publication_ready_as_validated_event_detector": False,
        "main_validated_feature_table_modified": False,
        "primary_analysis_eligible": False,
    }
    status_path = final_root / "manifests" / "qtemp_v100_final_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def export_exploratory_table(project_root: Path, final_root: Path) -> Path:
    analysis_path = (
        project_root / "outputs" / "02_features" / "temporal_discontinuity"
        / "qtemp-v0.3.1-finalization" / "tables" / "qtemp_v031_analysis_features.csv"
    )
    analysis = pd.read_csv(analysis_path)
    columns = ["logical_recording_id"] + [c for c in analysis.columns if c in FEATURES or c.endswith("_status") and c.removesuffix("_status") in FEATURES]
    output = analysis[columns].copy()
    output["qtemp_final_role"] = "EXPLORATORY_OR_MONITORING_NOT_PRIMARY"
    output["qtemp_final_measurement_version"] = FINAL_VERSION
    path = final_root / "tables" / "qtemp_v100_exploratory_features.csv"
    output.to_csv(path, index=False)

    main_dir = project_root / "MAIN outputs" / "02_FEATURE_TABLES_EXPLORATORY"
    main_dir.mkdir(parents=True, exist_ok=True)
    main_path = main_dir / "qtemp_v100_exploratory_features.csv"
    output.to_csv(main_path, index=False)
    return main_path


def write_manifest(final_root: Path) -> int:
    manifest = final_root / "manifests" / "qtemp_v100_final_artifact_sha256.csv"
    rows = []
    for path in sorted(final_root.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append({
                "relative_path": path.relative_to(final_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    source = project_root / "MAIN outputs/02_FEATURE_REVIEWED/00_working_candidates" / "temporal_discontinuity" / SOURCE_VERSION
    if not source.exists():
        raise FileNotFoundError(f"Run the QTEMP candidate validation first. Missing: {source}")
    source_manifest_entries = verify_manifest(source)

    parent = project_root / "MAIN outputs/02_FEATURE_REVIEWED/00_working_candidates" / "temporal_discontinuity"
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / FINAL_VERSION
    temporary = Path(tempfile.mkdtemp(prefix="qtemp_final_", dir=str(parent)))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        remove_superseded_event_workflow(temporary)
        replace_panel_b(temporary)
        replace_panel_g(project_root, temporary)
        replace_panel_i(temporary)
        replace_panel_j(temporary)
        normalize_panel_provenance(temporary)
        rewrite_gallery_index(temporary)
        write_final_records(temporary, source_manifest_entries)
        exploratory_main = export_exploratory_table(project_root, temporary)
        manifest_entries = write_manifest(temporary)
        archive = archive_existing(final)
        os.replace(temporary, final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    snapshot = (
        project_root / "MAIN outputs" / "02_FEATURE_FAMILY_SNAPSHOTS"
        / "temporal_discontinuity" / FINAL_VERSION
    )
    archived_snapshot = archive_existing(snapshot)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(final, snapshot)

    result = {
        "status": "SUCCESS",
        "final_root": str(final),
        "snapshot_root": str(snapshot),
        "exploratory_feature_table": str(exploratory_main),
        "manifest_entries": manifest_entries,
        "prior_final_archive": str(archive) if archive else None,
        "prior_snapshot_archive": str(archived_snapshot) if archived_snapshot else None,
        "validated_primary_feature_count": 0,
        "g9_applicable": False,
        "g9_passed": False,
        "implementation_frozen": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
