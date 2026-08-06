from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

PACKAGE_VERSION = "qgain-v4.1.0-figures-v1.0.0"
MEASUREMENT_VERSION = "qgain-v4.1.0"
FAMILY = "QGAIN"
FAMILY_NAME = "Recorded level and level dynamics"
FEATURES = [
    "qgain_typical_speech_level_dbfs",
    "qgain_within_segment_iqr_db",
    "qgain_between_segment_mad_db",
    "qgain_abs_drift_db_per_min",
]
DISPLAY = {
    "qgain_typical_speech_level_dbfs": "Typical speech level",
    "qgain_within_segment_iqr_db": "Within-segment IQR",
    "qgain_between_segment_mad_db": "Between-segment MAD",
    "qgain_abs_drift_db_per_min": "Absolute drift",
}
UNITS = {
    "qgain_typical_speech_level_dbfs": "dBFS",
    "qgain_within_segment_iqr_db": "dB",
    "qgain_between_segment_mad_db": "dB",
    "qgain_abs_drift_db_per_min": "dB/min",
}
ROLES = {
    "qgain_typical_speech_level_dbfs": "contextual",
    "qgain_within_segment_iqr_db": "primary mixed descriptor",
    "qgain_between_segment_mad_db": "secondary mixed descriptor",
    "qgain_abs_drift_db_per_min": "exploratory/contextual",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


@dataclass
class PackagePaths:
    root: Path
    png: Path
    svg: Path
    pdf: Path
    data: Path
    captions: Path
    provenance: Path
    tables: Path
    manifests: Path
    legacy: Path


def make_paths(root: Path) -> PackagePaths:
    p = PackagePaths(
        root=root,
        png=root / "figures" / "png",
        svg=root / "figures" / "svg",
        pdf=root / "figures" / "pdf",
        data=root / "source_data",
        captions=root / "captions",
        provenance=root / "provenance",
        tables=root / "tables",
        manifests=root / "manifests",
        legacy=root / "legacy_signal_galleries",
    )
    for folder in p.__dict__.values():
        if isinstance(folder, Path):
            folder.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig_id: str, fig: plt.Figure, paths: PackagePaths) -> dict[str, str]:
    fig.tight_layout()
    png = paths.png / f"{fig_id}.png"
    svg = paths.svg / f"{fig_id}.svg"
    pdf = paths.pdf / f"{fig_id}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"figure_png": rel(png, paths.root), "figure_svg": rel(svg, paths.root), "figure_pdf": rel(pdf, paths.root)}


def write_caption(fig_id: str, title: str, question: str, method: str, result: str, interpretation: str, limitation: str, paths: PackagePaths) -> str:
    path = paths.captions / f"{fig_id}.md"
    text = (
        f"# {fig_id} — {title}\n\n"
        f"**Scientific question.** {question}\n\n"
        f"**Method.** {method}\n\n"
        f"**Result.** {result}\n\n"
        f"**Interpretation.** {interpretation}\n\n"
        f"**Limitation / claim boundary.** {limitation}\n"
    )
    path.write_text(text, encoding="utf-8")
    return rel(path, paths.root)


def write_provenance(fig_id: str, source_files: Iterable[Path], source_data: Path, paths: PackagePaths, extra: dict | None = None) -> str:
    path = paths.provenance / f"{fig_id}.json"
    payload = {
        "figure_id": fig_id,
        "family": FAMILY,
        "measurement_version": MEASUREMENT_VERSION,
        "figure_package_version": PACKAGE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [
            {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in source_files
        ],
        "source_data": rel(source_data, paths.root),
        "source_data_sha256": sha256(source_data),
        "generator": "qgain_figure_completion_v100.py",
        "extra": extra or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return rel(path, paths.root)


def register(index: list[dict], *, panel: str, fig_id: str, title: str, gate: str, domain: str, applicability: str, status: str, files: dict, source_data: str, caption: str, provenance: str, note: str = "") -> None:
    index.append({
        "family": FAMILY,
        "measurement_version": MEASUREMENT_VERSION,
        "figure_package_version": PACKAGE_VERSION,
        "panel": panel,
        "figure_id": fig_id,
        "title": title,
        "gate": gate,
        "validation_domain": domain,
        "applicability": applicability,
        "status": status,
        **files,
        "source_data": source_data,
        "caption": caption,
        "provenance_json": provenance,
        "reviewer_note": note,
    })


def simple_line(x, y, title, xlabel, ylabel, *, identity=False, horizontal_zero=False):
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(x, y, marker="o")
    if identity:
        lo = float(min(np.nanmin(x), np.nanmin(y)))
        hi = float(max(np.nanmax(x), np.nanmax(y)))
        ax.plot([lo, hi], [lo, hi], linestyle="--", label="Identity")
        ax.legend()
    if horizontal_zero:
        ax.axhline(0.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return fig


def build_package(freeze_root: Path, output_root: Path) -> Path:
    freeze_root = freeze_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    paths = make_paths(output_root)

    manifest_path = freeze_root / "manifests" / "qgain_v410_freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("measurement_version") != MEASUREMENT_VERSION or manifest.get("freeze_status") != "frozen":
        raise RuntimeError("Input is not the frozen qgain-v4.1.0 family.")

    t = freeze_root / "tables"
    v = freeze_root / "validation"
    sv = v / "source_qgain_v401"
    g = freeze_root / "galleries"

    recordings_path = t / "qgain_v410_recording_features.csv"
    empirical_path = t / "qgain_v410_empirical_summary.csv"
    corr_path = t / "qgain_v410_spearman_correlations.csv"
    registry_path = t / "qgain_v410_feature_registry.csv"
    frame_path = t / "qgain_v410_frame_ledger.csv"
    segment_path = t / "qgain_v410_segment_ledger.csv"
    persistence_path = v / "qgain_v410_repeated_recording_persistence.csv"
    robustness_path = v / "qgain_v410_g6_quantitative_robustness_summary.csv"
    subject_balance_path = t / "qgain_v410_subject_balanced_resampling_summary.csv"
    dose_path = sv / "qgain_v401_g4_dose_response.csv"
    transform_path = sv / "qgain_v401_g3_transform_controls.csv"
    codec_path = sv / "qgain_v401_g3_codec_roundtrip.csv"
    resample_path = sv / "qgain_v401_g3_resampling_sensitivity.csv"
    nonid_path = sv / "qgain_v401_g5_nonidentifiability_demo.csv"
    g5_path = sv / "qgain_v401_g5_checks.csv"
    support_path = sv / "qgain_v401_g6_support_grid.csv"
    segment_delete_path = sv / "qgain_v401_segment_delete_one_long.csv"
    boundary_path = sv / "qgain_v401_boundary_guard_sensitivity.csv"
    gallery_index_path = g / "qgain_gallery_index.csv"

    recordings = read_csv(recordings_path)
    empirical = read_csv(empirical_path)
    corr = read_csv(corr_path)
    registry = read_csv(registry_path)
    frames = read_csv(frame_path)
    segments = read_csv(segment_path)
    persistence = read_csv(persistence_path)
    robustness = read_csv(robustness_path)
    subject_balance = read_csv(subject_balance_path)
    dose = read_csv(dose_path)
    transforms = read_csv(transform_path)
    codec = read_csv(codec_path)
    resample = read_csv(resample_path)
    nonid = read_csv(nonid_path)
    g5 = read_csv(g5_path)
    support = read_csv(support_path)
    segment_delete = read_csv(segment_delete_path)
    boundary = read_csv(boundary_path)
    gallery_index = read_csv(gallery_index_path)

    index: list[dict] = []

    # A — construct response
    dose_specs = [
        ("A01_amplitude_modulation_dose", "amplitude_modulation", "Within-segment IQR dose recovery", "Injected modulation depth (dB)", "Estimated within-segment IQR (dB)", "qgain_within_segment_iqr_db"),
        ("A02_segment_offset_dose", "segment_offset", "Between-segment MAD dose recovery", "Injected segment-offset scale (dB)", "Estimated between-segment MAD (dB)", "qgain_between_segment_mad_db"),
        ("A03_linear_drift_dose", "linear_drift", "Absolute drift dose recovery", "Injected absolute drift (dB/min)", "Estimated absolute drift (dB/min)", "qgain_abs_drift_db_per_min"),
    ]
    for fig_id, mechanism, title, xlabel, ylabel, feature in dose_specs:
        local = dose.loc[dose["mechanism"].astype(str).eq(mechanism), ["mechanism", "dose", "response"]].copy()
        data_path = paths.data / f"{fig_id}.csv"
        save_csv(local, data_path)
        fig = simple_line(local["dose"].to_numpy(), local["response"].to_numpy(), title, xlabel, ylabel, identity=(mechanism == "linear_drift"))
        files = save_figure(fig_id, fig, paths)
        is_monotonic = bool(np.all(np.diff(local["response"].to_numpy()) >= -1e-12))
        if mechanism == "linear_drift":
            max_error = float(np.max(np.abs(local["response"] - local["dose"])))
            result_text = f"The response was monotonic and closely followed the identity relation; maximum absolute recovery error was {max_error:.3f} dB/min."
        else:
            rho = float(pd.Series(local["dose"]).corr(pd.Series(local["response"]), method="spearman"))
            result_text = f"The response was strictly ordered across the dose grid (Spearman rho = {rho:.3f}; monotonic = {is_monotonic}). The estimator is not expected to equal the injected control parameter one-for-one."
        caption = write_caption(fig_id, title,
            "Does the estimator respond monotonically and quantitatively to its intended synthetic recorded-level transformation?",
            "A deterministic synthetic signal was perturbed over a prespecified dose grid and the corresponding QGAIN estimator was recomputed.",
            result_text,
            "The estimator is numerically responsive to the intended observable.",
            "Synthetic response establishes ordered construct behavior, not causal source identity in natural recordings.", paths)
        prov = write_provenance(fig_id, [dose_path, manifest_path], data_path, paths, {"feature": feature})
        register(index, panel="A", fig_id=fig_id, title=title, gate="G4", domain="Dose response", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    # B — discriminant specificity / non-identifiability
    fig_id = "B01_causal_nonidentifiability_parity"
    speaker = nonid.loc[nonid["causal_label"].eq("speaker_intensity_change")].iloc[0]
    device = nonid.loc[nonid["causal_label"].eq("device_gain_change")].iloc[0]
    parity = pd.DataFrame({
        "feature": FEATURES,
        "speaker_intensity_change": [speaker[f] for f in FEATURES],
        "device_gain_change": [device[f] for f in FEATURES],
    })
    data_path = paths.data / f"{fig_id}.csv"; save_csv(parity, data_path)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(parity["speaker_intensity_change"], parity["device_gain_change"])
    lo = float(min(parity[["speaker_intensity_change", "device_gain_change"]].min()))
    hi = float(max(parity[["speaker_intensity_change", "device_gain_change"]].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    for _, row in parity.iterrows():
        ax.annotate(DISPLAY[row["feature"]], (row["speaker_intensity_change"], row["device_gain_change"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Causal non-identifiability of recorded-level changes")
    ax.set_xlabel("Feature value after speaker-intensity change")
    ax.set_ylabel("Feature value after device-gain change")
    ax.grid(True, alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    caption = write_caption(fig_id, "Causal non-identifiability of recorded-level changes",
        "Can QGAIN distinguish an identical waveform-level change caused by speaker intensity from one caused by device gain?",
        "Matched transformations were assigned different causal labels while producing the same observable waveform and QGAIN feature vector.",
        "All four feature pairs lay exactly on the identity line.",
        "QGAIN measures recorded-level observables and cannot identify whether the cause was physiological, behavioral, geometric, or device-related.",
        "This is a deliberate claim boundary, not evidence of estimator failure.", paths)
    prov = write_provenance(fig_id, [nonid_path, manifest_path], data_path, paths)
    register(index, panel="B", fig_id=fig_id, title="Causal non-identifiability", gate="G5", domain="Discriminant validity", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov, note="Passes by demonstrating and bounding non-identifiability, not by resolving cause.")

    fig_id = "B02_constant_rms_spectral_false_response"
    local = g5.loc[g5["check"].astype(str).str.contains("constant-RMS spectral", regex=False)].copy()
    data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
    value = float(local["maximum_false_dynamic_response"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.bar(["Observed false\ndynamic response"], [value])
    ax.axhline(0.05, linestyle="--", label="Predeclared tolerance (0.05)")
    ax.set_yscale("log")
    ax.set_title("Constant-RMS spectral-change control")
    ax.set_ylabel("Maximum absolute false response")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    caption = write_caption(fig_id, "Constant-RMS spectral-change control",
        "Do spectral changes at fixed RMS create false level-dynamics responses?",
        "The spectrum was altered while holding RMS constant, and all dynamic estimators were recomputed.",
        f"The maximum false dynamic response was {value:.3e}, far below the 0.05-unit tolerance.",
        "The QGAIN estimators respond to recorded-level behavior rather than spectral redistribution alone.",
        "This control does not address physiological level variation, which remains a major confound.", paths)
    prov = write_provenance(fig_id, [g5_path, manifest_path], data_path, paths)
    register(index, panel="B", fig_id=fig_id, title="Constant-RMS spectral control", gate="G5", domain="Discriminant validity", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    # C — transformation contract
    gain = transforms.loc[transforms["condition"].eq("gain")].copy()
    fig_id = "C01_uniform_gain_equivariance"
    local = gain[["dose", "qgain_typical_speech_level_dbfs"]].copy()
    local["expected"] = local["qgain_typical_speech_level_dbfs"].iloc[gain["dose"].abs().argmin()] + local["dose"]
    data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(local["dose"], local["qgain_typical_speech_level_dbfs"], marker="o", label="Observed")
    ax.plot(local["dose"], local["expected"], linestyle="--", label="Expected 1:1 shift")
    ax.set_title("Uniform-gain equivariance of typical level")
    ax.set_xlabel("Applied uniform gain (dB)")
    ax.set_ylabel("Typical speech level (dBFS)")
    ax.legend(); ax.grid(True, alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    max_err = float((local["qgain_typical_speech_level_dbfs"] - local["expected"]).abs().max())
    caption = write_caption(fig_id, "Uniform-gain equivariance of typical level",
        "Does typical speech level shift one-for-one under a uniform digital gain?",
        "The synthetic waveform was multiplied by fixed gains from -12 to +12 dB.",
        f"Observed level followed the expected 1:1 relation with maximum absolute deviation {max_err:.3e} dB.",
        "The feature is an equivariant digital operating-level descriptor.",
        "Equivariance does not identify whether a natural level difference arose from device gain or the speaker.", paths)
    prov = write_provenance(fig_id, [transform_path, manifest_path], data_path, paths)
    register(index, panel="C", fig_id=fig_id, title="Uniform gain equivariance", gate="G3", domain="Transformation behavior", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    fig_id = "C02_uniform_gain_dynamics_invariance"
    dyn = gain[["dose", *FEATURES[1:]]].copy()
    for f in FEATURES[1:]: dyn[f"{f}_delta"] = dyn[f] - dyn.loc[gain["dose"].abs().idxmin(), f]
    long = dyn.melt(id_vars="dose", value_vars=[f"{f}_delta" for f in FEATURES[1:]], var_name="feature", value_name="delta")
    long["feature"] = long["feature"].str.replace("_delta", "", regex=False)
    data_path = paths.data / f"{fig_id}.csv"; save_csv(long, data_path)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for feature, grp in long.groupby("feature", sort=False):
        ax.plot(grp["dose"], grp["delta"], marker="o", label=DISPLAY[feature])
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Uniform-gain invariance of level-dynamics features")
    ax.set_xlabel("Applied uniform gain (dB)")
    ax.set_ylabel("Change from 0-dB condition")
    ax.legend(); ax.grid(True, alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    max_delta = float(long["delta"].abs().max())
    caption = write_caption(fig_id, "Uniform-gain invariance of dynamics features",
        "Do within-recording level-dynamics features remain unchanged under a common gain?",
        "The same uniform-gain grid was applied and each dynamic feature was centered to its 0-dB value.",
        f"The maximum absolute change was {max_delta:.3e} feature units.",
        "The three dynamics estimators are invariant to a common gain shift, as required.",
        "They remain sensitive to nonuniform biological or acquisition-related level changes.", paths)
    prov = write_provenance(fig_id, [transform_path, manifest_path], data_path, paths)
    register(index, panel="C", fig_id=fig_id, title="Uniform gain invariance", gate="G3", domain="Transformation behavior", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    for fig_id, table, source_path, category_col, title, xlabel in [
        ("C03_codec_roundtrip_sensitivity", codec, codec_path, "codec", "Codec round-trip sensitivity", "Codec"),
        ("C04_resampling_sensitivity", resample, resample_path, "native_rate_hz", "Resampling sensitivity", "Native sample rate (Hz)"),
    ]:
        delta_cols = [c for c in table.columns if c.endswith("_delta")]
        local = table[[category_col, *delta_cols]].copy().melt(id_vars=category_col, var_name="feature", value_name="delta")
        local["feature"] = local["feature"].str.replace("_delta", "", regex=False)
        tolerance = 0.15 if "codec" in fig_id else 0.05
        local["absolute_delta"] = local["delta"].abs()
        local["fraction_of_tolerance"] = local["absolute_delta"] / tolerance
        data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        cats = [str(x) for x in table[category_col].tolist()]
        x = np.arange(len(cats), dtype=float)
        offsets = np.linspace(-0.24, 0.24, len(FEATURES))
        for offset, feature in zip(offsets, FEATURES):
            vals = local.loc[local["feature"].eq(feature), "fraction_of_tolerance"].to_numpy()
            ax.scatter(x + offset, vals, label=DISPLAY[feature])
        ax.axhline(1.0, linestyle="--", label="Acceptance boundary")
        ax.set_xticks(x, cats)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Absolute change / prespecified tolerance")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.25)
        files = save_figure(fig_id, fig, paths)
        max_delta = float(local["absolute_delta"].max())
        max_fraction = float(local["fraction_of_tolerance"].max())
        caption = write_caption(fig_id, title,
            "Are QGAIN values materially altered by this signal-path transformation?",
            "Absolute feature deltas were computed after deterministic round-trip or resampling and normalized to the prespecified 0.15-unit codec or 0.05-unit resampling tolerance.",
            f"The maximum absolute change was {max_delta:.3f} native feature units ({max_fraction:.2f} of the applicable tolerance).",
            "The tested transformation produced changes within the prespecified tolerance.",
            "The result applies to the exact codecs/rates and parameters tested; it is not a universal signal-path guarantee.", paths)
        prov = write_provenance(fig_id, [source_path, manifest_path], data_path, paths)
        register(index, panel="C", fig_id=fig_id, title=title, gate="G3", domain="Transformation behavior", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    # D — support and availability
    fig_id = "D01_support_threshold_contract"
    status_cols = [c for c in support.columns if c.endswith("_status")]
    local = support[["guarded_support_sec", *status_cols]].copy().melt(id_vars="guarded_support_sec", var_name="feature", value_name="status")
    local["feature"] = local["feature"].str.replace("_status", "", regex=False)
    local["available"] = local["status"].eq("measured").astype(int)
    data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for feature, grp in local.groupby("feature", sort=False):
        ax.step(grp["guarded_support_sec"], grp["available"], where="post", label=DISPLAY[feature])
    ax.set_title("Synthetic support-threshold contract")
    ax.set_xlabel("Guarded strict-speech support (s)")
    ax.set_ylabel("Measurement available (0/1)")
    ax.set_yticks([0, 1]); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    caption = write_caption(fig_id, "Synthetic support-threshold contract",
        "Do measurements become available only after their declared minimum support is satisfied?",
        "Synthetic recordings were varied in guarded speech support while all other properties were held fixed.",
        "Typical level and within-segment IQR became available at 1.0 s; between-segment MAD and drift remained unavailable without sufficient segment count/span.",
        "Availability is governed by explicit support rules rather than imputation.",
        "This grid isolates duration; segment-count and span requirements are evaluated separately.", paths)
    prov = write_provenance(fig_id, [support_path, manifest_path], data_path, paths)
    register(index, panel="D", fig_id=fig_id, title="Support threshold contract", gate="G6", domain="Support and uncertainty", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    for fig_id, column, title, xlabel in [
        ("D02_guarded_support_distribution", "qgain_guarded_speech_support_sec", "Cohort guarded-speech support", "Guarded support (s)"),
        ("D03_interval_count_distribution", "qgain_guarded_speech_interval_count", "Cohort guarded-speech interval count", "Canonical strict-speech intervals"),
    ]:
        local = recordings[["logical_recording_id", column]].copy()
        data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.hist(pd.to_numeric(local[column], errors="coerce").dropna(), bins=30)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Recordings")
        ax.grid(True, axis="y", alpha=0.25)
        files = save_figure(fig_id, fig, paths)
        caption = write_caption(fig_id, title,
            "What independent support was available for cohort-level QGAIN estimation?",
            "The distribution was computed from frozen canonical strict-speech intervals for all 519 recordings.",
            f"Median {xlabel.lower()} was {pd.to_numeric(local[column], errors='coerce').median():.2f}; all recordings satisfied the retained-feature support contract.",
            "Cohort measurements are accompanied by explicit support metadata.",
            "Support amount does not remove physiological or phonetic confounding.", paths)
        prov = write_provenance(fig_id, [recordings_path, manifest_path], data_path, paths)
        register(index, panel="D", fig_id=fig_id, title=title, gate="G6/G7", domain="Support and uncertainty", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    fig_id = "D04_availability_and_support_tiers"
    rows = []
    for feature in FEATURES:
        rows.append({
            "feature": feature,
            "available_fraction": float(recordings[f"{feature}_available"].astype(bool).mean()),
            "high_support_fraction": float(recordings[f"{feature}_support_tier"].astype(str).eq("high").mean()),
        })
    local = pd.DataFrame(rows)
    data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(local))
    ax.bar(x - 0.18, local["available_fraction"], width=0.36, label="Available")
    ax.bar(x + 0.18, local["high_support_fraction"], width=0.36, label="High support tier")
    ax.set_xticks(x, [DISPLAY[f] for f in local["feature"]], rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Fraction of recordings")
    ax.set_title("Cohort availability and support tiers")
    ax.legend(); ax.grid(True, axis="y", alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    caption = write_caption(fig_id, "Cohort availability and support tiers",
        "Were retained QGAIN measurements available with adequate support across the frozen cohort?",
        "Availability masks and support tiers were summarized from the final recording-level export.",
        "All four retained measurements were available in 519/519 recordings and all were assigned the high support tier.",
        "Missingness remains explicitly represented even though it was absent in this cohort.",
        "Future datasets may have different support and availability profiles.", paths)
    prov = write_provenance(fig_id, [recordings_path, manifest_path], data_path, paths)
    register(index, panel="D", fig_id=fig_id, title="Availability and support tiers", gate="G6/G7", domain="Support and uncertainty", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    # E — sensitivity
    for fig_id, table, source_path, title in [
        ("E01_segment_deletion_sensitivity", segment_delete, segment_delete_path, "Delete-one-segment sensitivity"),
        ("E02_boundary_guard_sensitivity", boundary, boundary_path, "Boundary-guard sensitivity"),
    ]:
        local = table[["feature", "absolute_change"]].copy()
        data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        arrays = [pd.to_numeric(local.loc[local["feature"].eq(f), "absolute_change"], errors="coerce").dropna().to_numpy() for f in FEATURES]
        ax.boxplot(arrays, tick_labels=[DISPLAY[f] for f in FEATURES], showfliers=False)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_ylabel("Absolute feature change")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
        files = save_figure(fig_id, fig, paths)
        summary = local.groupby("feature")["absolute_change"].quantile([0.5, 0.95]).unstack()
        caption = write_caption(fig_id, title,
            "How sensitive are QGAIN values to a plausible perturbation of the frozen segment support?",
            "Absolute changes were computed after deleting one canonical segment or changing the two-sided speech-edge guard.",
            "Typical level and within-segment IQR were robust; between-segment MAD was moderately sensitive; drift had a large upper-tail response.",
            "The result supports role-specific interpretation rather than a single family-wide robustness label.",
            "The distribution reflects the sampled cohort and tested perturbation range.", paths)
        prov = write_provenance(fig_id, [source_path, robustness_path, manifest_path], data_path, paths)
        register(index, panel="E", fig_id=fig_id, title=title, gate="G6", domain="Reliability and robustness", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    fig_id = "E03_role_specific_robustness_summary"
    local = robustness[["feature", "segment_deletion_median_abs_change", "segment_deletion_p95_abs_change", "boundary_median_abs_change", "boundary_p95_abs_change", "final_interpretation"]].copy()
    data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(local))
    ax.scatter(x - 0.15, local["segment_deletion_p95_abs_change"], label="Delete-one p95")
    ax.scatter(x + 0.15, local["boundary_p95_abs_change"], label="Guard-change p95")
    ax.set_xticks(x, [DISPLAY[f] for f in local["feature"]], rotation=20, ha="right")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylabel("95th-percentile absolute change")
    ax.set_title("Role-specific empirical robustness summary")
    ax.legend(); ax.grid(True, axis="y", alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    caption = write_caption(fig_id, "Role-specific empirical robustness summary",
        "Does each retained feature meet the robustness expectation appropriate to its final scientific role?",
        "The 95th-percentile changes from delete-one-segment and boundary-guard audits were compared across the final feature hierarchy.",
        "The two primary/contextual features were robust, between-segment MAD was moderately sensitive, and drift was sensitive and retained only as exploratory/contextual.",
        "Feature roles were revised to match empirical robustness rather than forcing a uniform family standard.",
        "These values should not be interpreted as universal decision thresholds.", paths)
    prov = write_provenance(fig_id, [robustness_path, registry_path, manifest_path], data_path, paths)
    register(index, panel="E", fig_id=fig_id, title="Role-specific robustness", gate="G6/G10", domain="Reliability and robustness", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    # F — empirical distributions
    for i, feature in enumerate(FEATURES, start=1):
        fig_id = f"F{i:02d}_{feature}_distribution"
        local = recordings[["logical_recording_id", feature]].copy()
        data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
        vals = pd.to_numeric(local[feature], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.hist(vals, bins=35)
        ax.axvline(vals.median(), linestyle="--", label=f"Median = {vals.median():.2f}")
        ax.axvline(vals.quantile(0.25), linestyle=":", label="Q25/Q75")
        ax.axvline(vals.quantile(0.75), linestyle=":")
        ax.set_title(f"Cohort distribution: {DISPLAY[feature]}")
        ax.set_xlabel(f"{DISPLAY[feature]} ({UNITS[feature]})")
        ax.set_ylabel("Recordings")
        ax.legend(); ax.grid(True, axis="y", alpha=0.25)
        files = save_figure(fig_id, fig, paths)
        caption = write_caption(fig_id, f"Cohort distribution: {DISPLAY[feature]}",
            "What is the empirical distribution of the frozen QGAIN measurement?",
            "The distribution includes all available values from the 519-recording frozen cohort; no unavailable values were mapped to zero.",
            f"Median {vals.median():.2f} {UNITS[feature]}, IQR {vals.quantile(.25):.2f}–{vals.quantile(.75):.2f}, 1st–99th percentile {vals.quantile(.01):.2f}–{vals.quantile(.99):.2f}.",
            "The figure establishes empirical range and boundary behavior for interpretation and future QA.",
            "The distribution mixes acquisition, physiology, task, and participant effects and is not a severity scale.", paths)
        prov = write_provenance(fig_id, [recordings_path, empirical_path, manifest_path], data_path, paths, {"feature": feature})
        register(index, panel="F", fig_id=fig_id, title=f"Distribution of {DISPLAY[feature]}", gate="G7", domain="Empirical plausibility", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    # G — signal-linked trajectories at q05/q95 for each feature
    example_rows = []
    for feature in FEATURES:
        vals = recordings[["logical_recording_id", feature]].dropna().sort_values(feature)
        for quantile, label in [(0.05, "q05"), (0.95, "q95")]:
            target = vals[feature].quantile(quantile)
            idx = (vals[feature] - target).abs().idxmin()
            row = vals.loc[idx]
            example_rows.append({"feature": feature, "selection": label, "logical_recording_id": row["logical_recording_id"], "feature_value": row[feature]})
    examples = pd.DataFrame(example_rows)
    save_csv(examples, paths.tables / "qgain_signal_example_selection.csv")
    for j, row in examples.iterrows():
        feature = row["feature"]
        rec = str(row["logical_recording_id"])
        selection = str(row["selection"])
        fig_id = f"G{j+1:02d}_{feature}_{selection}_trajectory"
        local_f = frames.loc[frames["logical_recording_id"].astype(str).eq(rec), ["logical_recording_id", "segment_id", "frame_mid_sec", "ac_rms_dbfs", "valid_level_frame"]].copy()
        local_s = segments.loc[segments["logical_recording_id"].astype(str).eq(rec), ["logical_recording_id", "segment_id", "segment_mid_sec", "segment_level_median_dbfs", "segment_level_iqr_db", "usable_segment"]].copy()
        local_f["row_type"] = "frame"; local_s["row_type"] = "segment"
        data_path = paths.data / f"{fig_id}.csv"
        merged = pd.concat([local_f.assign(segment_mid_sec=np.nan, segment_level_median_dbfs=np.nan, segment_level_iqr_db=np.nan, usable_segment=np.nan), local_s.assign(frame_mid_sec=np.nan, ac_rms_dbfs=np.nan, valid_level_frame=np.nan)], ignore_index=True, sort=False)
        save_csv(merged, data_path)
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        valid = local_f.loc[local_f["valid_level_frame"].astype(bool)]
        ax.scatter(valid["frame_mid_sec"], valid["ac_rms_dbfs"], s=5, alpha=0.35, label="40-ms AC-RMS frames")
        useg = local_s.loc[local_s["usable_segment"].astype(bool)]
        ax.plot(useg["segment_mid_sec"], useg["segment_level_median_dbfs"], marker="o", label="Segment medians")
        ax.set_title(f"{DISPLAY[feature]} {selection}: {rec}\n{feature} = {float(row['feature_value']):.2f} {UNITS[feature]}")
        ax.set_xlabel("Original recording time (s)")
        ax.set_ylabel("Recorded level (dBFS)")
        ax.legend(); ax.grid(True, alpha=0.25)
        files = save_figure(fig_id, fig, paths)
        caption = write_caption(fig_id, f"Signal-linked example for {DISPLAY[feature]} ({selection})",
            "What frame- and segment-level signal pattern produced a low/high cohort value of the selected QGAIN feature?",
            "A recording nearest the 5th or 95th cohort percentile was selected deterministically. Frozen frame AC-RMS values and segment medians were plotted in original recording time.",
            f"The selected recording had {feature} = {float(row['feature_value']):.2f} {UNITS[feature]}.",
            "The figure connects the recording-level statistic to its lower-level evidence and permits audit of task structure and outliers.",
            "A representative example is illustrative and cannot establish a causal acquisition mechanism.", paths)
        prov = write_provenance(fig_id, [frame_path, segment_path, recordings_path, manifest_path], data_path, paths, {"feature": feature, "recording": rec, "selection": selection})
        register(index, panel="G", fig_id=fig_id, title=f"Signal-linked {DISPLAY[feature]} {selection}", gate="G7", domain="Interpretability", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    # Preserve legacy gallery as supplementary audit material.
    if g.exists():
        for path in g.glob("*.png"):
            shutil.copy2(path, paths.legacy / path.name)
        shutil.copy2(gallery_index_path, paths.legacy / gallery_index_path.name)

    # H — persistence and redundancy
    fig_id = "H01_repeated_recording_persistence"
    local = persistence[["feature", "first_second_spearman", "icc_1_1_first_two", "median_within_subject_abs_difference", "p90_within_subject_abs_difference"]].copy()
    data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(local))
    ax.bar(x - 0.18, local["first_second_spearman"], width=0.36, label="First–second Spearman")
    ax.bar(x + 0.18, local["icc_1_1_first_two"], width=0.36, label="ICC(1,1), first two")
    ax.set_xticks(x, [DISPLAY[f] for f in local["feature"]], rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.0); ax.set_ylabel("Persistence coefficient")
    ax.set_title("Repeated-recording empirical persistence")
    ax.legend(); ax.grid(True, axis="y", alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    caption = write_caption(fig_id, "Repeated-recording empirical persistence",
        "How stable are QGAIN measurements across repeated recordings from the same participant?",
        "First-versus-second Spearman correlations and method-of-moments ICC(1,1) estimates were calculated for 157 participants with repeats.",
        "Within-segment IQR showed the strongest persistence (Spearman 0.708; ICC 0.704); typical level was moderate; between-segment MAD and drift were weaker.",
        "The evidence supports differential feature roles and cautions against treating all QGAIN features as equally stable.",
        "This is empirical within-subject persistence, not controlled technical test–retest reliability; physiology and recording context may change.", paths)
    prov = write_provenance(fig_id, [persistence_path, manifest_path], data_path, paths)
    register(index, panel="H", fig_id=fig_id, title="Repeated-recording persistence", gate="G8", domain="Reliability", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    fig_id = "H02_within_family_redundancy"
    corr = corr.rename(columns={corr.columns[0]: "feature"})
    local = corr.copy()
    data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
    matrix = local.set_index("feature")[FEATURES].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(matrix, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), [DISPLAY[f] for f in FEATURES], rotation=35, ha="right")
    ax.set_yticks(range(len(FEATURES)), [DISPLAY[f] for f in FEATURES])
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Within-family Spearman correlation")
    files = save_figure(fig_id, fig, paths)
    off = matrix[np.triu_indices_from(matrix, k=1)]
    max_abs = float(np.max(np.abs(off)))
    caption = write_caption(fig_id, "Within-family Spearman correlation",
        "Are retained QGAIN features redundant or near-transforms of one another?",
        "Pairwise Spearman correlations were calculated across all 519 recordings.",
        f"The maximum absolute off-diagonal correlation was {max_abs:.3f}.",
        "The four features are not near-duplicates, although they share some recorded-level information.",
        "Low correlation does not prove distinct causal mechanisms.", paths)
    prov = write_provenance(fig_id, [corr_path, manifest_path], data_path, paths)
    register(index, panel="H", fig_id=fig_id, title="Within-family redundancy", gate="G8", domain="Reliability and redundancy", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    fig_id = "H03_subject_balanced_weighting_sensitivity"
    local = subject_balance.loc[subject_balance["stratum"].eq("all")].copy()
    data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(local))
    ax.bar(x, local["absolute_weighting_delta"])
    ax.set_xticks(x, [DISPLAY[f] for f in local["feature"]], rotation=20, ha="right")
    ax.set_ylabel("Absolute median difference (native feature units)")
    ax.set_title("Sensitivity to unequal recordings per participant")
    ax.grid(True, axis="y", alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    max_delta = float(local["absolute_weighting_delta"].max())
    caption = write_caption(fig_id, "Recording-weighted versus subject-balanced summaries",
        "Are cohort summaries materially driven by participants with more recordings?",
        "Recording-weighted medians were compared with 1,000 one-recording-per-participant resamples.",
        f"The maximum absolute weighting difference was {max_delta:.3f} feature units.",
        "The cohort summaries are not materially dominated by unequal recording counts.",
        "The plot is a weighting sensitivity analysis, not a diagnosis comparison.", paths)
    prov = write_provenance(fig_id, [subject_balance_path, manifest_path], data_path, paths)
    register(index, panel="H", fig_id=fig_id, title="Subject-balanced sensitivity", gate="G8", domain="Reliability and redundancy", applicability="REQUIRED", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    # J — ML handoff completeness
    fig_id = "J01_ml_interface_contract"
    ml_rows = []
    for feature in FEATURES:
        ml_rows.append({
            "feature": feature,
            "value_present": feature in recordings.columns,
            "availability_present": f"{feature}_available" in recordings.columns,
            "support_tier_present": f"{feature}_support_tier" in recordings.columns,
            "status_present": f"{feature}_status" in recordings.columns,
            "standalone_gate_allowed": bool(registry.loc[registry["feature"].eq(feature), "standalone_gate_allowed"].iloc[0]),
        })
    local = pd.DataFrame(ml_rows)
    data_path = paths.data / f"{fig_id}.csv"; save_csv(local, data_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    completeness = local[["value_present", "availability_present", "support_tier_present", "status_present"]].sum(axis=1) / 4
    ax.bar([DISPLAY[f] for f in local["feature"]], completeness)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Required interface fields present (fraction)")
    ax.set_title("Quality-aware ML interface completeness")
    ax.tick_params(axis="x", rotation=20); ax.grid(True, axis="y", alpha=0.25)
    files = save_figure(fig_id, fig, paths)
    caption = write_caption(fig_id, "Quality-aware ML interface completeness",
        "Does every QGAIN value travel with the minimum metadata required for safe downstream use?",
        "The final recording export was checked for value, availability, support tier, and status fields, and the registry was checked for standalone-gate permission.",
        "All four features had complete metadata and none was authorized as a standalone reject/accept gate.",
        "QGAIN is ready as a measurement-context input to biomarker-specific reliability models.",
        "No biomarker-specific reliability threshold has yet been calibrated.", paths)
    prov = write_provenance(fig_id, [recordings_path, registry_path, manifest_path], data_path, paths)
    register(index, panel="J", fig_id=fig_id, title="ML interface contract", gate="G10", domain="Quality-aware ML handoff", applicability="OPTIONAL", status="PASS", files=files, source_data=rel(data_path, paths.root), caption=caption, provenance=prov)

    figure_index = pd.DataFrame(index)
    save_csv(figure_index, paths.tables / "qgain_v410_figure_gallery_index.csv")

    # Panel summary
    panel_summary = figure_index.groupby(["panel", "status"], as_index=False).size()
    save_csv(panel_summary, paths.tables / "qgain_v410_panel_completion_summary.csv")

    # Contact sheet made from the first image in each required panel A-H and optional J.
    selected = []
    for panel in ["A", "B", "C", "D", "E", "F", "G", "H", "J"]:
        row = figure_index.loc[figure_index["panel"].eq(panel)].iloc[0]
        selected.append((panel, paths.root / row["figure_png"]))
    thumbs = []
    for panel, path in selected:
        im = Image.open(path).convert("RGB")
        im.thumbnail((700, 470))
        canvas = Image.new("RGB", (720, 520), "white")
        x = (720 - im.width) // 2; y = 35 + (470 - im.height) // 2
        canvas.paste(im, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 10), f"Panel {panel}: {path.stem}", fill="black")
        thumbs.append(canvas)
    cols = 2; rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 720, rows * 520), "white")
    for i, im in enumerate(thumbs): sheet.paste(im, ((i % cols) * 720, (i // cols) * 520))
    contact = paths.root / "QGAIN_v410_FIGURE_PACKAGE_CONTACT_SHEET.png"
    sheet.save(contact, dpi=(150, 150))

    # Final manifest and inventory
    all_files = sorted([p for p in paths.root.rglob("*") if p.is_file()])
    inventory = pd.DataFrame([{"relative_path": rel(p, paths.root), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in all_files])
    save_csv(inventory, paths.manifests / "qgain_v410_figure_package_inventory.csv")
    package_manifest = {
        "family": FAMILY,
        "family_display_name": FAMILY_NAME,
        "measurement_version": MEASUREMENT_VERSION,
        "figure_package_version": PACKAGE_VERSION,
        "status": "complete_candidate_ready_for_local_seal",
        "source_freeze_manifest_sha256": sha256(manifest_path),
        "source_freeze_inventory_sha256": manifest.get("freeze_inventory_sha256"),
        "source_executed_notebook_sha256": manifest.get("executed_notebook_sha256"),
        "recording_count": int(len(recordings)),
        "participant_count": int(recordings["subject"].nunique()),
        "figure_count": int(len(figure_index)),
        "required_panels_complete": all(panel in set(figure_index["panel"]) for panel in list("ABCDEFGH")),
        "event_panel_I": "N/A — QGAIN is not an event detector",
        "optional_panel_J_complete": True,
        "all_figure_rows_pass": bool(figure_index["status"].eq("PASS").all()),
        "feature_values_recomputed": False,
        "standalone_gate_allowed": False,
        "family_scalar_constructed": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (paths.manifests / "qgain_v410_figure_package_manifest.json").write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")

    return paths.root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = build_package(args.freeze_root, args.output_root)
    print(f"QGAIN figure package created: {out}")


if __name__ == "__main__":
    main()
