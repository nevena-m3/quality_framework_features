"""Generate auditable stage notebooks with figures, tables, and explicit decision gates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


BOOTSTRAP = r"""from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from IPython.display import Image, Markdown, display

def find_project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "paper1_qc").exists():
            return candidate
    raise FileNotFoundError("Run this notebook from inside the paper_1 project.")

ROOT = find_project_root()
CONFIG = ROOT / "config" / "project.yaml"
OUTPUT = ROOT / "outputs"
MAIN_OUTPUTS = ROOT / "MAIN outputs"
MAIN_OUTPUTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def run_cli(*arguments):
    command = [sys.executable, "-m", "paper1_qc.cli", "--config", str(CONFIG), *arguments]
    print("RUN:", " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, check=True)

def read_table(path_without_suffix):
    stem = Path(path_without_suffix)
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        try:
            return pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise FileNotFoundError(f"Missing table: {parquet} or {csv}")

def stage_directories(relative_stage):
    stage = OUTPUT / relative_stage
    figures = stage / "figures"
    tables = stage / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return stage, figures, tables

def save_table(frame, directory, name):
    path = Path(directory) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print("TABLE:", path.relative_to(ROOT), f"({len(frame):,} rows)")
    return path

def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{name}.png"
    svg = directory / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print("FIGURE:", png.relative_to(ROOT))
    return png, svg

def stage_gate(stage_name, can_continue, reasons, next_step):
    status = "PASS — safe to continue" if can_continue else "BLOCKED — decision/action required"
    color = "#1B7F3A" if can_continue else "#B22222"
    details = "\n".join(f"- {reason}" for reason in reasons) if reasons else "- No blocking findings."
    display(Markdown(
        f"### {stage_name}: <span style='color:{color}'>{status}</span>\n\n"
        f"{details}\n\n**Next step:** {next_step}"
    ))
    return can_continue

assert CONFIG.exists(), "Copy config/project.example.yaml to config/project.yaml and review it."
print("Project:", ROOT)
print("Config:", CONFIG)
print("MAIN outputs:", MAIN_OUTPUTS)
"""


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_notebook(relative: str, title: str, purpose: str, cells: list[dict]) -> None:
    payload = {
        "cells": [
            markdown(
                f"# {title}\n\n{purpose}\n\n"
                "This notebook saves its visual summary and audit tables into separate "
                "`figures/` and `tables/` directories. Its final cell states the main output, "
                "any decision required, and whether the next stage is allowed."
            ),
            code(BOOTSTRAP),
            *cells,
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Paper 1 QC",
                "language": "python",
                "name": "paper1-qc",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def cli_switch_cell(variable: str, command: str, message: str) -> dict:
    arguments = ", ".join(repr(part) for part in command.split())
    return code(
        f"{variable} = False  # Change to True only when you intend to run this stage.\n\n"
        f"if {variable}:\n"
        f"    run_cli({arguments})\n"
        "else:\n"
        f"    print({message!r})"
    )


write_notebook(
    "notebooks/00_setup/00_environment_check.ipynb",
    "00 — Environment and configuration check",
    "Checks the active interpreter, configuration, FFmpeg/FFprobe, and package entry point.",
    [
        markdown(
            "## Main output and decision\n\n"
            "Main output: `outputs/00_environment/tables/environment_checks.csv` and a "
            "pass/fail figure. Fix every failed check before reading participant data."
        ),
        code(
            r"""STAGE, FIGURES, TABLES = stage_directories("00_environment")

checks = []
checks.append({"check": "project configuration exists", "pass": CONFIG.exists(), "detail": str(CONFIG)})
checks.append({"check": "Python 3.11", "pass": sys.version_info[:2] == (3, 11), "detail": sys.version.split()[0]})
checks.append({"check": "FFmpeg on PATH", "pass": shutil.which("ffmpeg") is not None, "detail": str(shutil.which("ffmpeg"))})
checks.append({"check": "FFprobe on PATH", "pass": shutil.which("ffprobe") is not None, "detail": str(shutil.which("ffprobe"))})

help_run = subprocess.run(
    [sys.executable, "-m", "paper1_qc.cli", "--help"],
    cwd=ROOT,
    capture_output=True,
    text=True,
)
checks.append({
    "check": "paper1_qc CLI imports",
    "pass": help_run.returncode == 0,
    "detail": help_run.stderr.strip() or "ok",
})

checks = pd.DataFrame(checks)
save_table(checks, TABLES, "environment_checks")
display(checks)

fig, ax = plt.subplots(figsize=(9, 3.8))
colors = checks["pass"].map({True: "#59A14F", False: "#E15759"})
ax.barh(checks["check"], 1, color=colors)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("Paper 1 environment preflight")
for index, row in checks.iterrows():
    ax.text(0.02, index, "PASS" if row["pass"] else "FAIL", va="center", color="white", fontweight="bold")
fig.tight_layout()
save_figure(fig, FIGURES, "environment_preflight")
plt.show()

environment_ready = stage_gate(
    "Environment",
    bool(checks["pass"].all()),
    checks.loc[~checks["pass"], "check"].tolist(),
    "Open the metadata/media audit notebook only after every check passes.",
)"""
        ),
    ],
)


AUDIT_VISUAL = r"""STAGE, FIGURES, TABLES = stage_directories("00_audit")

def audit_table(stem):
    return read_table(STAGE / stem)

bamboo = audit_table("bamboo_canonical_recordings")
rest = audit_table("rest_canonical_recordings")
bamboo_media = audit_table("bamboo_media_rows_audited")
rest_media = audit_table("rest_media_rows_audited")
bamboo_inventory = audit_table("bamboo_media_inventory")
rest_inventory = audit_table("rest_media_inventory")
exact_pairs = audit_table("exact_bamboo_rest_session_pairs")

order = [
    "Reported ALS",
    "Reported control",
    "Control candidate from ID",
    "Unresolved diagnosis",
    "Conflicting diagnosis",
]

def participant_table(frame, task):
    rows = []
    for subject_id, group in frame.groupby("SubjectID", dropna=False):
        reported = sorted(group["diagnosis_reported"].dropna().astype(str).unique())
        inferred = sorted(group["diagnosis_inferred_from_id"].dropna().astype(str).unique())
        if len(reported) > 1:
            status = "Conflicting diagnosis"
        elif reported == ["ALS"]:
            status = "Reported ALS"
        elif reported == ["CONTROLS"]:
            status = "Reported control"
        elif not reported and "CONTROLS" in inferred:
            status = "Control candidate from ID"
        else:
            status = "Unresolved diagnosis"
        rows.append({
            "SubjectID": subject_id,
            "task": task,
            "cohort_status": status,
            "logical_recordings": group["logical_recording_id"].nunique(),
        })
    return pd.DataFrame(rows)

participants = pd.concat(
    [participant_table(bamboo, "Bamboo"), participant_table(rest, "Rest")],
    ignore_index=True,
)
participant_counts = (
    participants.groupby(["task", "cohort_status"], observed=False)
    .size().rename("participants").reset_index()
)
recording_counts = (
    participants.groupby(["task", "cohort_status"], observed=False)["logical_recordings"]
    .sum().rename("logical_recordings").reset_index()
)
bamboo_ids = set(bamboo["SubjectID"])
rest_ids = set(rest["SubjectID"])
availability = pd.DataFrame({
    "task_availability": ["Both Bamboo and Rest", "Bamboo only", "Rest only"],
    "participants": [
        len(bamboo_ids & rest_ids),
        len(bamboo_ids - rest_ids),
        len(rest_ids - bamboo_ids),
    ],
})
overview = pd.DataFrame([
    {
        "dataset": "Bamboo",
        "metadata_media_rows": len(bamboo_media),
        "physical_files_found": len(bamboo_inventory),
        "logical_recordings": bamboo["logical_recording_id"].nunique(),
        "participants": bamboo["SubjectID"].nunique(),
    },
    {
        "dataset": "Rest",
        "metadata_media_rows": len(rest_media),
        "physical_files_found": len(rest_inventory),
        "logical_recordings": rest["logical_recording_id"].nunique(),
        "participants": rest["SubjectID"].nunique(),
    },
])
pair_summary = pd.DataFrame([{
    "exact_session_pairs_before_freeze": len(exact_pairs),
    "participants_with_pair": exact_pairs["SubjectID"].nunique(),
}])

issue_frames = []
for role in ["bamboo", "rest", "combined"]:
    issues = audit_table(f"{role}_audit_issues")
    if not issues.empty:
        issue_frames.append(issues.assign(dataset=role.capitalize()))
issues = pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame()
issue_counts = (
    issues.groupby(["severity", "rule"]).size().rename("issue_rows").reset_index()
    if not issues.empty else pd.DataFrame(columns=["severity", "rule", "issue_rows"])
)

for name, frame in {
    "participant_counts_pre_freeze": participant_counts,
    "logical_recording_counts_pre_freeze": recording_counts,
    "study_overview_pre_freeze": overview,
    "participant_task_availability": availability,
    "exact_session_pair_summary_pre_freeze": pair_summary,
    "metadata_issue_counts": issue_counts,
}.items():
    save_table(frame, TABLES, name)

display(overview)
display(pair_summary)

palette = {
    "Reported ALS": "#C76D6D",
    "Reported control": "#5E81A8",
    "Control candidate from ID": "#E2AE4D",
    "Unresolved diagnosis": "#8C8C8C",
    "Conflicting diagnosis": "#8E5EA2",
}
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
sns.barplot(
    data=participant_counts, x="task", y="participants", hue="cohort_status",
    hue_order=order, palette=palette, ax=axes[0, 0],
)
for container in axes[0, 0].containers:
    axes[0, 0].bar_label(container, fmt="%.0f", padding=3, fontsize=8)
axes[0, 0].set(title="A. Unique participants before diagnosis freeze", xlabel="", ylabel="Participants")

sns.barplot(
    data=recording_counts, x="task", y="logical_recordings", hue="cohort_status",
    hue_order=order, palette=palette, ax=axes[0, 1],
)
for container in axes[0, 1].containers:
    axes[0, 1].bar_label(container, fmt="%.0f", padding=3, fontsize=8)
axes[0, 1].set(title="B. Logical recordings (WAV/WEBM collapsed)", xlabel="", ylabel="Logical recordings")

sns.barplot(
    data=availability, x="participants", y="task_availability",
    hue="task_availability", palette=["#59A14F", "#4C78A8", "#B07AA1"],
    legend=False, ax=axes[1, 0],
)
for container in axes[1, 0].containers:
    axes[1, 0].bar_label(container, fmt="%.0f", padding=4)
axes[1, 0].set(title="C. Participant task availability", xlabel="Participants", ylabel="")

if not issue_counts.empty:
    top = issue_counts.groupby("rule")["issue_rows"].sum().nlargest(10).index
    plot = issue_counts.loc[issue_counts["rule"].isin(top)]
    sns.barplot(
        data=plot, x="issue_rows", y="rule", hue="severity",
        palette={"error": "#D55E5E", "review": "#F2B134"},
        estimator="sum", errorbar=None, ax=axes[1, 1],
    )
    axes[1, 1].set(title="D. Most frequent audit findings", xlabel="Flagged metadata rows", ylabel="")
else:
    axes[1, 1].text(0.5, 0.5, "No audit findings", ha="center", va="center")
fig.suptitle("Paper 1 metadata and cohort audit — pre-freeze", fontsize=17, fontweight="bold")
fig.tight_layout()
save_figure(fig, FIGURES, "metadata_and_cohort_overview_pre_freeze")
plt.show()
"""


write_notebook(
    "notebooks/00_setup/00_metadata_and_media_audit.ipynb",
    "00 — Metadata, media, and cohort freeze",
    "Audits all metadata/media, visualizes the raw cohort, and creates the immutable analysis freeze.",
    [
        markdown(
            "## Main output and decisions\n\n"
            "Main output: the immutable files under "
            "`MAIN outputs/00_DATA_FREEZE/<version>/`. The pre-freeze audit is descriptive "
            "only. Do not use its candidate/unresolved counts as final cohort counts.\n\n"
            "Decision required: every diagnosis not covered by a documented rule must be "
            "`ALS`, `CONTROLS`, or `EXCLUDE`. Media must resolve to one decodable path. WAV "
            "has priority over WEBM."
        ),
        code(
            r"""RUN_METADATA_AUDIT = False
RUN_MEDIA_INVENTORY = False  # Slow: probes, fully decodes, and hashes every file.

if RUN_METADATA_AUDIT:
    run_cli("audit")
else:
    print("Metadata audit not rerun.")

if RUN_MEDIA_INVENTORY:
    run_cli("inventory")
else:
    print("Media inventory not rerun. Reuse it only if the data folders have not changed.")"""
        ),
        code(AUDIT_VISUAL),
        markdown(
            "## Diagnosis adjudication\n\n"
            "In local `config/project.yaml`, list investigator-confirmed exceptional controls "
            "under `data_freeze.confirmed_control_subject_ids` and document the evidence. "
            "Then run the template cell. Open `config/metadata_adjudication.csv`; complete any "
            "remaining row and save it. The freeze cell must remain off until this is done."
        ),
        code(
            r"""cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
freeze_cfg = cfg.get("data_freeze", {})
display(pd.DataFrame([{
    "freeze_version": freeze_cfg.get("version"),
    "exact_control_patterns_confirmed": freeze_cfg.get("confirm_configured_control_id_patterns"),
    "exceptional_controls_confirmed": len(freeze_cfg.get("confirmed_control_subject_ids", [])),
    "exceptional_control_evidence_present": bool(freeze_cfg.get("confirmed_control_subject_evidence")),
}]))

run_cli("freeze-template")
adjudication_path = ROOT / freeze_cfg.get(
    "diagnosis_adjudication", "config/metadata_adjudication.csv"
)
adjudication = pd.read_csv(adjudication_path, keep_default_na=False)
display(adjudication)
blank = adjudication["diagnosis_analysis"].astype(str).str.strip().eq("")
diagnosis_ready = stage_gate(
    "Diagnosis adjudication",
    not blank.any(),
    [
        f"{row.SubjectID}: choose ALS, CONTROLS, or EXCLUDE and provide evidence"
        for row in adjudication.loc[blank].itertuples()
    ],
    "Save the completed CSV, rerun this cell, then enable RUN_DATA_FREEZE.",
)"""
        ),
        cli_switch_cell(
            "RUN_DATA_FREEZE",
            "freeze",
            "Freeze not run. Set RUN_DATA_FREEZE=True only after the diagnosis gate passes.",
        ),
        code(
            r"""cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
version = str(cfg.get("data_freeze", {}).get("version", "v1"))
FREEZE = MAIN_OUTPUTS / "00_DATA_FREEZE" / version

if (FREEZE / "data_freeze_manifest.json").exists():
    freeze_summary = pd.read_csv(FREEZE / "freeze_summary.csv")
    provenance = pd.read_csv(FREEZE / "diagnosis_provenance.csv")
    bamboo_frozen = pd.read_csv(FREEZE / "frozen_bamboo_recordings.csv")
    rest_frozen = pd.read_csv(FREEZE / "frozen_rest_recordings.csv")
    pairs_frozen = pd.read_csv(FREEZE / "frozen_exact_bamboo_rest_pairs.csv")

    save_table(freeze_summary, TABLES, "frozen_cohort_summary")
    encoding = (
        pd.concat([
            bamboo_frozen.assign(task="Bamboo"),
            rest_frozen.assign(task="Rest"),
        ])
        .groupby(["task", "extension_parsed"]).size()
        .rename("logical_recordings").reset_index()
    )
    save_table(encoding, TABLES, "selected_encoding_counts")

    plot = freeze_summary.loc[
        freeze_summary["diagnosis_analysis"].isin(["ALS", "CONTROLS"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(data=plot, x="dataset_role", y="participants", hue="diagnosis_analysis", ax=axes[0])
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.0f", padding=3)
    axes[0].set(title="Frozen participants", xlabel="", ylabel="Participants")
    sns.barplot(data=encoding, x="task", y="logical_recordings", hue="extension_parsed", ax=axes[1])
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.0f", padding=3)
    axes[1].set(title="Selected encoding (WAV priority)", xlabel="", ylabel="Logical recordings")
    fig.suptitle("Frozen Paper 1 cohort and selected media")
    fig.tight_layout()
    save_figure(fig, FIGURES, "frozen_cohort_and_encoding_summary")
    plt.show()

    display(freeze_summary)
    freeze_ready = stage_gate(
        "Metadata/media freeze",
        True,
        [],
        "Open the Silero segmentation notebook. All downstream stages now read the frozen tables.",
    )
else:
    freeze_ready = stage_gate(
        "Metadata/media freeze",
        False,
        ["No immutable data_freeze_manifest.json exists for the configured version."],
        "Complete diagnosis adjudication and run the freeze cell.",
    )"""
        ),
    ],
)


write_notebook(
    "notebooks/01_segmentation/01_segmentation_silero_full_dataset.ipynb",
    "01 — Version-pinned Silero segmentation and visual QC",
    "Runs Silero, creates the mandatory review queue, supports audio/manual boundary review, and freezes adjudicated segments.",
    [
        markdown(
            "## Main output and decisions\n\n"
            "Authoritative outputs: versioned `frozen_segmentation_decisions.csv` and "
            "`frozen_segmentation_intervals.csv` under "
            "`MAIN outputs/01_SEGMENTATION_FREEZE/<version>/`. Audit outputs include "
            "`bamboo_segmentation_intervals.csv`, one original-format `_segments.csv` and "
            "`_frames.csv` per recording under "
            "`outputs/01_segmentation/segmentation/silero/`, and the "
            "original four-panel figures under "
            "`outputs/01_segmentation/figures/segmentation/silero/"
            "{accepted,flagged,excluded}`. A separate boundary-audit CSV and PNG compare "
            "sample-index analysis edges with the 30-ms display representation.\n\n"
            "After review, `segment-adjudicate` creates a separate immutable publication tree "
            "at `outputs/01_segmentation_after_review/`, with per-recording figures, frames, "
            "segments, and boundary audits organized as accepted, flagged, or excluded. "
            "Accepted and flagged recordings proceed; excluded recordings do not.\n\n"
            "The segment roles are exactly `leading_nonspeech`, `internal_nonspeech`, "
            "`trailing_nonspeech`, and `speech`; the figure rows are labelled "
            "`non-speech` and `speech`. The primary boundaries use zero speech padding "
            "and no second post-VAD bridge/filter pass; the 30-ms mask is visualization, "
            "not the frozen timing source.\n\n"
            "Decision required: inspect every flagged/excluded recording and every accepted "
            "recording selected by the prespecified segmentation-only outlier rules. Listen to "
            "the audio when needed. Use `KEEP + AUTO`, `KEEP + MANUAL`, or `EXCLUDE + NONE`. "
            "Feature extraction is blocked until eligibility and the final primary speech "
            "boundaries are frozen."
        ),
        cli_switch_cell(
            "RUN_SEGMENTATION",
            "segment",
            "Segmentation not run. Set RUN_SEGMENTATION=True only after the data freeze passes.",
        ),
        code(
            r"""STAGE, FIGURES, TABLES = stage_directories("01_segmentation")
summary = read_table(STAGE / "bamboo_segmentation_summary")
status_counts = read_table(STAGE / "segmentation_qc_status_counts")
flag_counts = read_table(STAGE / "segmentation_qc_flag_counts")
display(status_counts)
display(flag_counts.head(20))

LEGACY = STAGE / "segmentation" / "silero"
LEGACY_FIGURES = STAGE / "figures" / "segmentation" / "silero"
legacy_summary = pd.read_csv(LEGACY / "summary" / "silero_all_summary.csv")

artifact_audit = pd.DataFrame([{
    "frozen_recordings": len(summary),
    "summary_rows": len(legacy_summary),
    "segment_csv_files": len(list((LEGACY / "segments").glob("*_segments.csv"))),
    "frame_csv_files": len(list((LEGACY / "frames").glob("*_frames.csv"))),
    "accepted_png_files": len(list((LEGACY_FIGURES / "accepted").glob("*_silero.png"))),
    "flagged_png_files": len(list((LEGACY_FIGURES / "flagged").glob("*_silero.png"))),
    "excluded_png_files": len(list((LEGACY_FIGURES / "excluded").glob("*_silero.png"))),
    "boundary_audit_csv_files": len(list((LEGACY / "boundary_audit").glob("*_boundary_audit.csv"))),
    "boundary_audit_png_files": len(list((LEGACY_FIGURES / "boundary_audit").glob("*_boundary_audit.png"))),
}])
artifact_audit["total_png_files"] = artifact_audit[
    ["accepted_png_files", "flagged_png_files", "excluded_png_files"]
].sum(axis=1)
display(artifact_audit)
save_table(artifact_audit, TABLES, "notebook_silero_artifact_audit")

expected = int(artifact_audit.iloc[0]["frozen_recordings"])
assert int(artifact_audit.iloc[0]["summary_rows"]) == expected
assert int(artifact_audit.iloc[0]["segment_csv_files"]) == expected
assert int(artifact_audit.iloc[0]["frame_csv_files"]) == expected
assert int(artifact_audit.iloc[0]["total_png_files"]) == expected
assert int(artifact_audit.iloc[0]["boundary_audit_csv_files"]) == expected
assert int(artifact_audit.iloc[0]["boundary_audit_png_files"]) == expected

print("Representative original-pipeline diagnostic figures:")
for status in ["accepted", "flagged", "excluded"]:
    subset = summary.loc[summary["qc_status"].eq(status)]
    if not subset.empty and Path(subset.iloc[0]["plot_path"]).exists():
        display(Markdown(f"**{status.upper()} example**"))
        display(Image(filename=subset.iloc[0]["plot_path"], width=1000))"""
        ),
        code(
            r"""# Boundary science audit: exact timestamps are not the 30-ms display bins.
resolved_parameters = json.loads(
    (STAGE / "logs" / "silero_segmentation_config.json").read_text(encoding="utf-8")
)
display(pd.DataFrame([resolved_parameters]).T.rename(columns={0: "resolved_value"}))

boundary_summary = summary[[
    "file_name", "qc_status", "boundary_edges",
    "boundary_low_contrast_edges", "boundary_low_contrast_fraction",
    "boundary_min_contrast_db", "boundary_audit_path", "boundary_plot_path",
]].copy()
display(boundary_summary.sort_values(
    ["boundary_low_contrast_fraction", "boundary_min_contrast_db"],
    ascending=[False, True],
).head(30))
save_table(boundary_summary, TABLES, "notebook_boundary_alignment_summary")

EXAMPLE_BOUNDARY_FILE = None  # set exact filename, or use the most ambiguous edge record
if EXAMPLE_BOUNDARY_FILE:
    boundary_row = boundary_summary.loc[
        boundary_summary["file_name"].eq(EXAMPLE_BOUNDARY_FILE)
    ].iloc[0]
else:
    boundary_row = boundary_summary.sort_values(
        ["boundary_low_contrast_fraction", "boundary_min_contrast_db"],
        ascending=[False, True],
    ).iloc[0]
display(pd.read_csv(boundary_row["boundary_audit_path"]))
display(Image(filename=str(boundary_row["boundary_plot_path"]), width=1100))"""
        ),
        markdown(
            "A low boundary-contrast flag is not an automatic error and never moves an edge. "
            "Breathy, weak, or slowly decaying ALS speech may have low contrast. It prompts "
            "listening/manual inspection. Parameter optimality cannot be claimed from plots "
            "alone; if timing accuracy is a reported result, use a diagnosis-blind manual "
            "boundary reference subset and report onset/offset error and overlap by group."
        ),
        code(
            r"""run_cli("segment-template")
segmentation_path = ROOT / yaml.safe_load(CONFIG.read_text(encoding="utf-8")).get(
    "data_freeze", {}
).get("segmentation_adjudication", "config/segmentation_adjudication.csv")
manual_path = ROOT / yaml.safe_load(CONFIG.read_text(encoding="utf-8")).get(
    "data_freeze", {}
).get("manual_segmentation_overrides", "config/manual_segmentation_overrides.csv")
segmentation_review = pd.read_csv(segmentation_path, keep_default_na=False)
from paper1_qc.segmentation import segmentation_pending_reviews

required_review = segmentation_review.loc[
    segmentation_review["review_required"].astype(str).str.lower().isin(["true", "1", "yes"])
].copy()
automatic_task_exclusions = segmentation_review.loc[
    segmentation_review["automatic_task_exclusion"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
].copy()
pending_review = segmentation_pending_reviews(segmentation_review)
display(required_review[[
    "file_name", "automatic_qc_status", "task_completed_as_instructed",
    "accepted_outlier", "review_reasons",
    "decision", "boundary_source", "reviewer", "review_date", "notes"
]])
display(Markdown(
    f"**Locked automatic task exclusions:** {len(automatic_task_exclusions)} "
    "(Task Completed as Instructed = NO)"
))
display(automatic_task_exclusions[[
    "file_name", "task_completed_as_instructed", "automatic_exclusion_reason",
    "decision", "boundary_source", "notes"
]])
segmentation_review_ready = stage_gate(
    "Segmentation review queue",
    pending_review.empty,
    [f"{len(pending_review)} required/incomplete reviews remain."],
    "Use the review widget below; then rerun this cell to update the gate.",
)"""
        ),
        markdown(
            "## Interactive recording review\n\n"
            "The scrollable widget starts with **all recordings** and supports filtering and "
            "filename/ID search. It displays the original Silero plot, boundary audit, and an "
            "audio player. **Keep Silero + next** retains the automatic boundaries and advances. "
            "`KEEP + MANUAL` activates the optional boundary editor; enter "
            "one speech interval per line as `start_sec,end_sec`, preview it, and document why "
            "the correction was necessary. `EXCLUDE + NONE` removes the recording and requires "
            "a reason. Rows with `Task Completed as Instructed = NO` are shown but locked to "
            "exclusion. Manual editing changes speech boundaries only—it must not remove noise."
        ),
        code(
            r"""from paper1_qc.config import load_config, resolve_executable
from paper1_qc.segmentation_review import launch_segmentation_review_widget

cfg = load_config(CONFIG)
automatic_intervals = read_table(STAGE / "bamboo_segmentation_intervals")
DEFAULT_REVIEWER = ""  # enter your name once, e.g. "Nevena Musikic"

review_widget = launch_segmentation_review_widget(
    summary=summary,
    automatic_intervals=automatic_intervals,
    review_path=segmentation_path,
    overrides_path=manual_path,
    default_reviewer=DEFAULT_REVIEWER,
    ffmpeg=resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg"),
    ffprobe=resolve_executable(cfg["software"]["ffprobe"], "ffprobe"),
)
display(review_widget)"""
        ),
        code(
            r"""# Run after using the widget. This reload is required because the widget saves to disk.
segmentation_review = pd.read_csv(segmentation_path, keep_default_na=False)
pending_review = segmentation_pending_reviews(segmentation_review)
review_progress = pd.DataFrame([{
    "total_recordings": len(segmentation_review),
    "mandatory_review_recordings": int(
        segmentation_review["review_required"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
    ),
    "pending_or_incomplete_reviews": len(pending_review),
    "automatic_task_exclusions": int(
        segmentation_review["automatic_task_exclusion"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        ).sum()
    ),
    "keep_auto": int((
        segmentation_review["decision"].eq("KEEP")
        & segmentation_review["boundary_source"].eq("AUTO")
    ).sum()),
    "keep_manual": int((
        segmentation_review["decision"].eq("KEEP")
        & segmentation_review["boundary_source"].eq("MANUAL")
    ).sum()),
    "exclude": int(segmentation_review["decision"].eq("EXCLUDE").sum()),
}])
display(review_progress)
display(pending_review[[
    "file_name", "automatic_qc_status", "task_completed_as_instructed",
    "review_reasons", "decision", "boundary_source", "reviewer",
    "review_date", "notes"
]])
save_table(review_progress, TABLES, "notebook_segmentation_review_progress")"""
        ),
        code(
            r"""# Complete, auditable decision ledger before freezing.
review_audit = segmentation_review.copy()
as_bool = lambda value: (
    value if isinstance(value, bool)
    else str(value).strip().lower() in {"true", "1", "yes", "y"}
)
review_audit["review_required_bool"] = review_audit["review_required"].map(as_bool)
review_audit["automatic_task_exclusion_bool"] = (
    review_audit["automatic_task_exclusion"].map(as_bool)
)
review_audit["is_pending"] = review_audit["logical_recording_id"].astype(str).isin(
    set(pending_review["logical_recording_id"].astype(str))
)
decision_upper = review_audit["decision"].astype(str).str.strip().str.upper()
source_upper = review_audit["boundary_source"].astype(str).str.strip().str.upper()
review_audit["final_decision_category"] = np.select(
    [
        review_audit["automatic_task_exclusion_bool"]
        & decision_upper.eq("EXCLUDE") & source_upper.eq("NONE"),
        decision_upper.eq("KEEP") & source_upper.eq("AUTO")
        & ~review_audit["review_required_bool"] & ~review_audit["is_pending"],
        decision_upper.eq("KEEP") & source_upper.eq("AUTO")
        & review_audit["review_required_bool"] & ~review_audit["is_pending"],
        decision_upper.eq("KEEP") & source_upper.eq("MANUAL")
        & ~review_audit["is_pending"],
        decision_upper.eq("EXCLUDE") & source_upper.eq("NONE")
        & ~review_audit["is_pending"],
    ],
    [
        "SYSTEM_EXCLUDE_TASK_NOT_COMPLETED",
        "KEEP_AUTO_DEFAULT",
        "KEEP_AUTO_REVIEWED",
        "KEEP_MANUAL",
        "EXCLUDE_REVIEWED",
    ],
    default="UNRESOLVED",
)
review_audit["will_be_analysis_eligible"] = review_audit[
    "final_decision_category"
].isin(["KEEP_AUTO_DEFAULT", "KEEP_AUTO_REVIEWED", "KEEP_MANUAL"])
review_audit["boundary_provenance_after_freeze"] = np.select(
    [
        review_audit["final_decision_category"].isin(
            ["KEEP_AUTO_DEFAULT", "KEEP_AUTO_REVIEWED"]
        ),
        review_audit["final_decision_category"].eq("KEEP_MANUAL"),
        review_audit["final_decision_category"].str.startswith(
            ("EXCLUDE", "SYSTEM_EXCLUDE")
        ),
    ],
    [
        "automatic_silero",
        "manual_override",
        "not_used_excluded_recording",
    ],
    default="not_frozen_unresolved",
)
decision_audit_summary = (
    review_audit.groupby(
        [
            "final_decision_category",
            "will_be_analysis_eligible",
            "boundary_provenance_after_freeze",
        ],
        dropna=False,
    ).size().rename("logical_recordings").reset_index()
)
display(decision_audit_summary)
display(review_audit.loc[
    review_audit["final_decision_category"].eq("UNRESOLVED"),
    [
        "file_name", "automatic_qc_status", "task_completed_as_instructed",
        "review_reasons", "decision", "boundary_source", "reviewer",
        "review_date", "notes",
    ],
])
save_table(
    decision_audit_summary,
    TABLES,
    "notebook_segmentation_decision_audit_summary",
)
save_table(
    review_audit,
    TABLES,
    "notebook_segmentation_decision_audit_ledger",
)"""
        ),
        code(
            r"""# Freeze only after the audit above contains zero UNRESOLVED rows.
# This cell captures full stdout/stderr instead of hiding the CLI failure cause.
RUN_SEGMENTATION_ADJUDICATION = False

if not RUN_SEGMENTATION_ADJUDICATION:
    print(
        "Segments not frozen. Set RUN_SEGMENTATION_ADJUDICATION=True only "
        "after pending_or_incomplete_reviews is zero."
    )
elif not pending_review.empty:
    print(
        f"BLOCKED: {len(pending_review)} pending/incomplete reviews remain. "
        "Return to the widget; do not freeze yet."
    )
else:
    command = [
        sys.executable, "-m", "paper1_qc.cli", "--config", str(CONFIG),
        "segment-adjudicate",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    print("RETURN CODE:", result.returncode)
    print("\n========== STDOUT ==========")
    print(result.stdout or "(empty)")
    print("\n========== STDERR ==========")
    print(result.stderr or "(empty)")
    if result.returncode == 0:
        display(Markdown("**Segmentation freeze: PASS**"))
    else:
        display(Markdown(
            "**Segmentation freeze: BLOCKED.** Read STDERR above; do not continue."
        ))"""
        ),
        code(
            r"""project_cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
segmentation_freeze_version = project_cfg.get("segmentation_freeze", {}).get(
    "version",
    project_cfg.get("data_freeze", {}).get("version", "v1"),
)
SEGMENTATION_FREEZE = (
    MAIN_OUTPUTS / "01_SEGMENTATION_FREEZE" / str(segmentation_freeze_version)
)
REVIEWED_OUTPUT = OUTPUT / "01_segmentation_after_review"
decision_path = SEGMENTATION_FREEZE / "frozen_segmentation_decisions.csv"
interval_path = SEGMENTATION_FREEZE / "frozen_segmentation_intervals.csv"
reviewed_summary_path = (
    REVIEWED_OUTPUT / "segmentation" / "silero" / "summary"
    / "silero_after_review_summary.csv"
)
if decision_path.exists() and interval_path.exists() and reviewed_summary_path.exists():
    decisions = pd.read_csv(decision_path)
    frozen_intervals = pd.read_csv(interval_path)
    reviewed_summary = pd.read_csv(reviewed_summary_path)
    decision_summary = (
        decisions.groupby(
            ["automatic_qc_status", "review_required", "decision", "boundary_source"]
        ).size()
        .rename("logical_recordings").reset_index()
    )
    display(decision_summary)
    save_table(decision_summary, TABLES, "notebook_segmentation_decision_summary")
    reviewed_status = (
        reviewed_summary.groupby(
            ["final_review_status", "analysis_included"], dropna=False
        ).size().rename("logical_recordings").reset_index()
    )
    display(Markdown(
        "**Post-review contract:** accepted and flagged proceed; excluded do not."
    ))
    display(reviewed_status)
    save_table(
        reviewed_status,
        TABLES,
        "notebook_post_review_segmentation_status",
    )
    eligible_mask = decisions["segmentation_analysis_eligible"].map(
        lambda value: value if isinstance(value, bool)
        else str(value).strip().lower() in {"true", "1", "yes", "y"}
    )
    kept_ids = set(
        decisions.loc[eligible_mask, "logical_recording_id"].astype(str)
    )
    frozen_primary_ids = set(
        frozen_intervals.loc[
            frozen_intervals["profile"].eq("primary")
            & frozen_intervals["view"].eq("primary_speech")
            & frozen_intervals["duration_sec"].gt(0),
            "logical_recording_id",
        ].astype(str)
    )
    segmentation_ready = stage_gate(
        "Segmentation stage",
        decisions["decision"].isin(["KEEP", "EXCLUDE"]).all()
        and len(reviewed_summary) == len(decisions)
        and kept_ids.issubset(frozen_primary_ids),
        [
            f"{len(kept_ids - frozen_primary_ids)} KEEP recordings lack frozen primary speech."
        ] if not kept_ids.issubset(frozen_primary_ids) else [],
        "Open 02a Additive interference. Feature extraction reads only the versioned MAIN outputs freeze.",
    )
else:
    segmentation_ready = stage_gate(
        "Segmentation stage",
        False,
        [
            "Frozen decisions, frozen intervals, and/or "
            "outputs/01_segmentation_after_review do not exist."
        ],
        "Complete the widget review and run segment-adjudicate.",
    )"""
        ),
    ],
)


FAMILY_SPECS = [
    ("02a_additive_interference.ipynb", "02a — Additive interference", "additive_interference", "qadd"),
    ("02b_gain_dynamics.ipynb", "02b — Gain and amplitude dynamics", "gain_dynamics", "qgain"),
    ("02c_reverberation_tail.ipynb", "02c — Reverberation-tail proxies", "reverberation_tail", "qrev"),
    ("02d_channel_device.ipynb", "02d — Channel and device descriptors", "channel_device", "qchan"),
    ("02e_nonlinear_distortion.ipynb", "02e — Nonlinear distortion", "nonlinear_distortion", "qdist"),
    ("02f_temporal_discontinuity.ipynb", "02f — Temporal discontinuity", "temporal_discontinuity", "qtemp"),
]


def feature_cells(family: str, prefix: str, run_extractor: bool) -> list[dict]:
    cells = [
        markdown(
            f"## Main output and decisions\n\n"
            f"Main outputs: `outputs/02_features/{family}/tables/metric_summary.csv` "
            f"and `figures/metric_distributions.*`.\n\n"
            "Decision required only if extraction errors occur, a metric is entirely missing, "
            "or support/status distributions are scientifically implausible. Do not tune "
            "thresholds after examining clinical or human-label associations."
        )
    ]
    if run_extractor:
        cells.append(
            cli_switch_cell(
                "RUN_PRIMARY_EXTRACTION",
                "extract --profile primary",
                "Primary extraction not run. Enable only after segmentation decisions are frozen.",
            )
        )
    cells.append(
        code(
            f"""from paper1_qc.registry import metric_registry_frame

FAMILY = {family!r}
PREFIX = {prefix!r}
STAGE, FIGURES, TABLES = stage_directories(Path("02_features") / FAMILY)
registry = metric_registry_frame()
family_registry = registry.loc[registry["family"].eq(FAMILY)].copy()
display(family_registry)
save_table(family_registry, TABLES, "metric_registry")

metrics = read_table(OUTPUT / "02_features" / "bamboo_q_metrics")
errors = read_table(OUTPUT / "02_features" / "feature_extraction_errors")
features = [feature for feature in family_registry["feature"] if feature in metrics.columns]

summary_rows = []
for feature in features:
    values = pd.to_numeric(metrics[feature], errors="coerce")
    summary_rows.append({{
        "feature": feature,
        "recordings": len(values),
        "nonmissing": int(values.notna().sum()),
        "missing_fraction": float(values.isna().mean()),
        "zero_fraction_nonmissing": float(values.dropna().eq(0).mean()) if values.notna().any() else np.nan,
        "median": values.median(),
        "q25": values.quantile(0.25),
        "q75": values.quantile(0.75),
        "minimum": values.min(),
        "maximum": values.max(),
    }})
metric_summary = pd.DataFrame(summary_rows)
status_columns = [column for column in metrics if column.startswith(PREFIX) and column.endswith("_status")]
status_summary = (
    metrics[status_columns].melt(var_name="status_field", value_name="status")
    .groupby(["status_field", "status"], dropna=False).size()
    .rename("recordings").reset_index()
    if status_columns else pd.DataFrame()
)
save_table(metric_summary, TABLES, "metric_summary")
save_table(status_summary, TABLES, "status_summary")
save_table(errors, TABLES, "extraction_errors")
display(metric_summary)
display(status_summary)

ncols = 3
nrows = max(1, int(np.ceil(len(features) / ncols)))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows))
axes = np.atleast_1d(axes).ravel()
for ax, feature in zip(axes, features):
    sns.histplot(pd.to_numeric(metrics[feature], errors="coerce"), bins=30, ax=ax, color="#4C78A8")
    ax.set_title(feature, fontsize=9)
    ax.set_xlabel("")
for ax in axes[len(features):]:
    ax.set_axis_off()
fig.suptitle(f"{{FAMILY.replace('_', ' ').title()}} — observed metric distributions")
fig.tight_layout()
save_figure(fig, FIGURES, "metric_distributions")
plt.show()

blocking = []
if not errors.empty:
    blocking.append(f"{{len(errors)}} extraction errors require investigation.")
if metric_summary.empty:
    blocking.append("No registered metrics were found.")
else:
    all_missing = metric_summary.loc[metric_summary["nonmissing"].eq(0), "feature"].tolist()
    if all_missing:
        blocking.append("Entirely missing metrics: " + ", ".join(all_missing))

feature_stage_ready = stage_gate(
    FAMILY.replace("_", " ").title(),
    not blocking,
    blocking,
    "Proceed to the next family notebook if PASS; otherwise inspect saved errors/support tables and rerun.",
)"""
        )
    )
    return cells


def additive_feature_cells() -> list[dict]:
    """Generate the fully audited first feature-family notebook."""
    return [
        markdown(
            "## Run contract\n\n"
            "This is the **only family notebook that launches primary extraction**. The CLI "
            "decodes each included Bamboo recording once and computes all six families into "
            "`outputs/02_features/bamboo_q_metrics`; notebooks 02b–02f audit their registered "
            "columns without re-extracting audio.\n\n"
            "Set `RUN_PRIMARY_EXTRACTION=True` only after the segmentation notebook reports "
            "PASS. A rerun replaces the mutable feature-extraction workspace, so rerun all "
            "family audits after any extraction-code change.\n\n"
            "**This notebook does not tune thresholds using diagnosis, clinical scores, or "
            "human QC labels.**"
        ),
        cli_switch_cell(
            "RUN_PRIMARY_EXTRACTION",
            "extract --profile primary",
            "Primary extraction not run. Enable only after segmentation decisions are frozen.",
        ),
        markdown(
            r"""## 1. Frozen measurement specification

The current implementation calculates six additive-interference measures from the
16-kHz mono analysis signal without peak normalization. Frame-level measures use 30-ms
frames with a 10-ms hop.

\[
L_t=20\log_{10}\left(\sqrt{\frac{1}{N}\sum_{n=1}^{N}x_t[n]^2}\right)\;\mathrm{dBFS}
\]

- Nonspeech level is the median guarded-pause \(L_t\).
- The SNR proxy is median strict-speech \(L_t\) minus median guarded-pause \(L_t\).
- Variability is \(Q_{0.75}(L_t)-Q_{0.25}(L_t)\).
- Transients are **segment-aware runs** above
  \(\mathrm{median}(L_t)+\max(12,\;6\times1.4826\,\mathrm{MAD}(L_t))\), divided by
  guarded-nonspeech minutes.
- Hum prominence compares ±1-Hz bands at harmonics 1–4 of 50/60 Hz against local
  sidebands 4–12 Hz away. It requires at least one continuous one-second pause.
- Spectral flatness is geometric-mean PSD / arithmetic-mean PSD over 20–1000 Hz.
  It is a noise-type descriptor, **not an ordinal severity score**.

`qadd_snr_proxy_db` is not a calibrated physical SNR because both regions contain
speech-, speaker-, room-, microphone-, and segmentation-dependent energy."""
        ),
        code(
            r"""from paper1_qc.registry import metric_registry_frame

FAMILY = "additive_interference"
PREFIX = "qadd"
STAGE, FIGURES, TABLES = stage_directories(Path("02_features") / FAMILY)

registry = metric_registry_frame()
family_registry = registry.loc[registry["family"].eq(FAMILY)].copy()
definition_columns = [
    "feature", "display_name", "role", "unit", "worse", "formula",
    "signal_region", "minimum_support", "mathematical_range",
    "status_field", "support_fields", "interpretation", "confounding",
    "expected_control_response", "metric_version",
]
missing_definition_columns = [
    column for column in definition_columns if column not in family_registry
]
assert not missing_definition_columns, (
    "Additive registry is incomplete: " + ", ".join(missing_definition_columns)
)
feature_definitions = family_registry[definition_columns].copy()
save_table(feature_definitions, TABLES, "feature_definitions")
display(feature_definitions)

direction_legend = pd.DataFrame([
    {"registry_value": "higher", "meaning": "larger numeric values indicate greater artifact burden"},
    {"registry_value": "lower", "meaning": "smaller numeric values indicate greater artifact burden"},
    {"registry_value": "contextual", "meaning": "descriptor only; no universal better/worse direction"},
])
save_table(direction_legend, TABLES, "direction_legend")
display(direction_legend)"""
        ),
        markdown(
            "## 2. Extraction and frozen-cohort contract\n\n"
            "This check proves that the feature table contains exactly one row for every "
            "segmentation-eligible logical recording and no other recordings. Alternate "
            "encodings remain technical replicates outside the primary table."
        ),
        code(
            r"""project_cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
segmentation_version = str(
    project_cfg.get("segmentation_freeze", {}).get(
        "version",
        project_cfg.get("data_freeze", {}).get("version", "v1"),
    )
)
data_version = str(project_cfg.get("data_freeze", {}).get("version", "v1"))
SEGMENTATION_FREEZE = (
    MAIN_OUTPUTS / "01_SEGMENTATION_FREEZE" / segmentation_version
)
DATA_FREEZE = MAIN_OUTPUTS / "00_DATA_FREEZE" / data_version

metrics = read_table(OUTPUT / "02_features" / "bamboo_q_metrics")
errors = read_table(OUTPUT / "02_features" / "feature_extraction_errors")
frozen_decisions = read_table(
    SEGMENTATION_FREEZE / "frozen_segmentation_decisions"
)
frozen_recordings = read_table(DATA_FREEZE / "frozen_bamboo_recordings")

def as_bool(series):
    return series.map(
        lambda value: value if isinstance(value, bool)
        else str(value).strip().lower() in {"true", "1", "yes", "y"}
    )

eligible = as_bool(frozen_decisions["segmentation_analysis_eligible"])
expected_ids = set(
    frozen_decisions.loc[eligible, "logical_recording_id"].astype(str)
)
observed_ids = (
    metrics["logical_recording_id"].astype(str)
    if "logical_recording_id" in metrics
    else pd.Series(dtype=str)
)
observed_id_set = set(observed_ids)
duplicate_ids = sorted(
    observed_ids.loc[observed_ids.duplicated(keep=False)].unique()
)
missing_ids = sorted(expected_ids - observed_id_set)
unexpected_ids = sorted(observed_id_set - expected_ids)

extraction_contract = pd.DataFrame([
    {"check": "segmentation_freeze_version", "observed": segmentation_version, "expected": segmentation_version},
    {"check": "eligible_frozen_recordings", "observed": len(expected_ids), "expected": len(expected_ids)},
    {"check": "extracted_rows", "observed": len(metrics), "expected": len(expected_ids)},
    {"check": "unique_extracted_logical_ids", "observed": len(observed_id_set), "expected": len(expected_ids)},
    {"check": "duplicate_logical_ids", "observed": len(duplicate_ids), "expected": 0},
    {"check": "missing_frozen_ids", "observed": len(missing_ids), "expected": 0},
    {"check": "unexpected_ids", "observed": len(unexpected_ids), "expected": 0},
    {"check": "extraction_errors", "observed": len(errors), "expected": 0},
])
save_table(extraction_contract, TABLES, "extraction_contract")
save_table(pd.DataFrame({"logical_recording_id": missing_ids}), TABLES, "missing_frozen_recordings")
save_table(pd.DataFrame({"logical_recording_id": unexpected_ids}), TABLES, "unexpected_recordings")
save_table(pd.DataFrame({"logical_recording_id": duplicate_ids}), TABLES, "duplicate_recordings")
save_table(errors, TABLES, "extraction_errors")
display(extraction_contract)
if not errors.empty:
    display(errors.head(50))"""
        ),
        markdown(
            "## 3. Metric-specific support, missingness, and status\n\n"
            "Support is evaluated separately for every feature. `status='ok'` must imply a "
            "finite value; any non-OK status must imply a missing value. The family-level "
            "`qadd_status` is `ok` only when all five primary metrics are available, "
            "`partial_support` when at least one is available, and `insufficient_support` "
            "when none is available."
        ),
        code(
            r"""features = family_registry["feature"].tolist()
summary_rows = []
status_rows = []
status_value_mismatch_rows = []

for spec in family_registry.itertuples(index=False):
    feature = spec.feature
    status_field = spec.status_field
    if feature not in metrics or status_field not in metrics:
        summary_rows.append({
            "feature": feature,
            "display_name": spec.display_name,
            "role": spec.role,
            "unit": spec.unit,
            "worse": spec.worse,
            "recordings": len(metrics),
            "participants_nonmissing": 0,
            "nonmissing": 0,
            "missing_fraction": 1.0,
            "status_ok": 0,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
        })
        continue

    values = pd.to_numeric(metrics[feature], errors="coerce")
    statuses = metrics[status_field].fillna("MISSING_STATUS").astype(str)
    finite = pd.Series(np.isfinite(values), index=metrics.index)
    nonmissing = values.notna()
    ok = statuses.eq("ok")
    participants = (
        metrics.loc[finite, "SubjectID"].nunique()
        if "SubjectID" in metrics else np.nan
    )
    summary_rows.append({
        "feature": feature,
        "display_name": spec.display_name,
        "role": spec.role,
        "unit": spec.unit,
        "worse": spec.worse,
        "recordings": len(values),
        "participants_nonmissing": participants,
        "nonmissing": int(finite.sum()),
        "missing_fraction": float((~finite).mean()),
        "status_ok": int(ok.sum()),
        "zero_fraction_nonmissing": (
            float(values.loc[finite].eq(0).mean()) if finite.any() else np.nan
        ),
        "median": values.loc[finite].median(),
        "q25": values.loc[finite].quantile(0.25),
        "q75": values.loc[finite].quantile(0.75),
        "minimum": values.loc[finite].min(),
        "maximum": values.loc[finite].max(),
    })
    for status, count in statuses.value_counts(dropna=False).items():
        status_rows.append({
            "feature": feature,
            "display_name": spec.display_name,
            "status_field": status_field,
            "status": status,
            "recordings": int(count),
        })

    mismatch = ok.ne(finite)
    for index in metrics.index[mismatch]:
        status_value_mismatch_rows.append({
            "file_name": metrics.at[index, "file_name"],
            "logical_recording_id": metrics.at[index, "logical_recording_id"],
            "feature": feature,
            "status_field": status_field,
            "status": statuses.at[index],
            "value": values.at[index],
            "reason": (
                "status_ok_but_value_missing_or_nonfinite"
                if ok.at[index]
                else "non_ok_status_but_value_present"
            ),
        })

metric_summary = pd.DataFrame(summary_rows)
status_summary = pd.DataFrame(status_rows)
status_value_mismatches = pd.DataFrame(status_value_mismatch_rows)

support_columns = [
    "qadd_speech_support_sec",
    "qadd_nonspeech_support_sec",
    "qadd_speech_frame_count",
    "qadd_nonspeech_frame_count",
    "qadd_nonspeech_interval_count",
    "qadd_max_nonspeech_interval_sec",
    "qadd_flatness_spectral_support_sec",
    "qadd_hum_spectral_support_sec",
]
available_support = [column for column in support_columns if column in metrics]
support_summary = (
    metrics[available_support].apply(pd.to_numeric, errors="coerce")
    .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    .T.reset_index().rename(columns={"index": "support_field"})
    if available_support else pd.DataFrame()
)

family_status_summary = (
    metrics["qadd_status"].fillna("MISSING_STATUS").value_counts()
    .rename_axis("qadd_status").rename("recordings").reset_index()
    if "qadd_status" in metrics else pd.DataFrame()
)

save_table(metric_summary, TABLES, "metric_summary")
save_table(status_summary, TABLES, "metric_status_summary")
save_table(family_status_summary, TABLES, "family_status_summary")
save_table(support_summary, TABLES, "support_summary")
save_table(status_value_mismatches, TABLES, "status_value_mismatches")
display(metric_summary)
display(family_status_summary)
display(support_summary)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
plot_summary = metric_summary.sort_values("missing_fraction")
axes[0].barh(
    plot_summary["display_name"],
    100 * (1 - plot_summary["missing_fraction"]),
    color="#4C78A8",
)
axes[0].set_xlim(0, 100)
axes[0].set_xlabel("Recordings with valid value (%)")
axes[0].set_title("A. Metric-specific usable support")
for y, value in enumerate(100 * (1 - plot_summary["missing_fraction"])):
    axes[0].text(min(value + 1, 98), y, f"{value:.1f}%", va="center", fontsize=9)

if not status_summary.empty:
    status_matrix = status_summary.pivot(
        index="display_name", columns="status", values="recordings"
    ).fillna(0)
    status_matrix = status_matrix.div(status_matrix.sum(axis=1), axis=0) * 100
    sns.heatmap(
        status_matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% of recordings"},
        ax=axes[1],
    )
    axes[1].set_xlabel("Metric-specific status")
    axes[1].set_ylabel("")
axes[1].set_title("B. Why values are available or missing")
fig.suptitle("Additive-interference support and status audit", fontweight="bold")
fig.tight_layout()
save_figure(fig, FIGURES, "support_and_status_audit")
plt.show()"""
        ),
        markdown(
            "## 4. Hard integrity and mathematical-range checks\n\n"
            "These are blocking checks. Empirically unusual but mathematically valid values "
            "are not deleted: genuine high artifact burden is part of the study target."
        ),
        code(
            r"""range_violation_rows = []
missing_feature_columns = [
    feature for feature in features if feature not in metrics
]
missing_status_columns = [
    field for field in family_registry["status_field"] if field not in metrics
]

for spec in family_registry.itertuples(index=False):
    if spec.feature not in metrics:
        continue
    values = pd.to_numeric(metrics[spec.feature], errors="coerce")
    finite_values = values[np.isfinite(values)]
    invalid = pd.Series(False, index=metrics.index)
    if pd.notna(spec.hard_min):
        invalid |= values.lt(float(spec.hard_min)).fillna(False)
    if pd.notna(spec.hard_max):
        invalid |= values.gt(float(spec.hard_max)).fillna(False)
    invalid |= values.notna() & ~np.isfinite(values)
    for index in metrics.index[invalid]:
        range_violation_rows.append({
            "file_name": metrics.at[index, "file_name"],
            "logical_recording_id": metrics.at[index, "logical_recording_id"],
            "feature": spec.feature,
            "value": values.at[index],
            "hard_min": spec.hard_min,
            "hard_max": spec.hard_max,
            "mathematical_range": spec.mathematical_range,
        })
range_violations = pd.DataFrame(range_violation_rows)

family_status_mismatches = pd.DataFrame()
if not missing_status_columns and "qadd_status" in metrics:
    primary_status_fields = family_registry.loc[
        family_registry["role"].eq("primary"), "status_field"
    ].tolist()
    primary_ok_count = metrics[primary_status_fields].eq("ok").sum(axis=1)
    expected_family_status = np.select(
        [
            primary_ok_count.eq(len(primary_status_fields)),
            primary_ok_count.gt(0),
        ],
        ["ok", "partial_support"],
        default="insufficient_support",
    )
    mismatch = metrics["qadd_status"].astype(str).ne(expected_family_status)
    family_status_mismatches = metrics.loc[
        mismatch, ["file_name", "logical_recording_id", "qadd_status"]
    ].copy()
    family_status_mismatches["expected_qadd_status"] = expected_family_status[mismatch]

primary_under_supported = metric_summary.loc[
    metric_summary["role"].eq("primary")
    & (
        metric_summary["nonmissing"].lt(20)
        | metric_summary["participants_nonmissing"].lt(20)
    ),
    ["feature", "nonmissing", "participants_nonmissing"],
]

integrity_checks = pd.DataFrame([
    {
        "check": "No extraction errors",
        "pass": errors.empty,
        "observed": len(errors),
        "required": 0,
        "blocking": True,
    },
    {
        "check": "Exactly one row per frozen eligible recording",
        "pass": (
            len(metrics) == len(expected_ids)
            and not duplicate_ids and not missing_ids and not unexpected_ids
        ),
        "observed": len(metrics),
        "required": len(expected_ids),
        "blocking": True,
    },
    {
        "check": "All registered feature columns exist",
        "pass": not missing_feature_columns,
        "observed": len(missing_feature_columns),
        "required": 0,
        "blocking": True,
    },
    {
        "check": "All metric-specific status columns exist",
        "pass": not missing_status_columns,
        "observed": len(missing_status_columns),
        "required": 0,
        "blocking": True,
    },
    {
        "check": "Status and value presence agree",
        "pass": status_value_mismatches.empty,
        "observed": len(status_value_mismatches),
        "required": 0,
        "blocking": True,
    },
    {
        "check": "Family status agrees with primary metric statuses",
        "pass": family_status_mismatches.empty,
        "observed": len(family_status_mismatches),
        "required": 0,
        "blocking": True,
    },
    {
        "check": "All nonmissing values satisfy hard mathematical ranges",
        "pass": range_violations.empty,
        "observed": len(range_violations),
        "required": 0,
        "blocking": True,
    },
    {
        "check": "Every primary metric has >=20 recordings and participants",
        "pass": primary_under_supported.empty,
        "observed": len(primary_under_supported),
        "required": 0,
        "blocking": True,
    },
])
save_table(integrity_checks, TABLES, "integrity_checks")
save_table(range_violations, TABLES, "range_violations")
save_table(family_status_mismatches, TABLES, "family_status_mismatches")
save_table(primary_under_supported, TABLES, "primary_under_supported")
display(integrity_checks)
if not range_violations.empty:
    display(range_violations.head(50))
if not status_value_mismatches.empty:
    display(status_value_mismatches.head(50))
if not family_status_mismatches.empty:
    display(family_status_mismatches.head(50))
if not primary_under_supported.empty:
    display(primary_under_supported)"""
        ),
        markdown(
            "## 5. Controlled analytical verification\n\n"
            "These deterministic controls use the same production function as the data. They "
            "check expected direction, dose response, event separation, noise-type behavior, "
            "and global-gain invariance. They verify implementation behavior; they do not "
            "replace validation on independently corrupted speech or the perceptual analysis."
        ),
        code(
            r"""from scipy import stats
from paper1_qc.metrics import additive_interference_metrics
from paper1_qc.segmentation import Interval

CONTROL_SR = 16000
CONTROL_DURATION_SEC = 12.0
control_time = np.arange(int(CONTROL_DURATION_SEC * CONTROL_SR)) / CONTROL_SR
control_speech = [
    Interval(0.5, 1.5), Interval(2.0, 3.0), Interval(3.5, 4.5),
    Interval(5.0, 6.0), Interval(6.5, 7.5), Interval(8.0, 9.0),
    Interval(10.0, 11.0),
]
control_pauses = [
    Interval(1.5, 2.0), Interval(3.0, 3.5), Interval(4.5, 5.0),
    Interval(6.0, 6.5), Interval(7.5, 8.0), Interval(9.0, 10.0),
]
speech_mask = np.zeros(len(control_time), dtype=bool)
for interval in control_speech:
    speech_mask[
        int(interval.start_sec * CONTROL_SR):
        int(interval.end_sec * CONTROL_SR)
    ] = True

clean_source = np.zeros(len(control_time), dtype=float)
clean_source[speech_mask] = (
    0.05 * np.sin(2 * np.pi * 180 * control_time[speech_mask])
    + 0.02 * np.sin(2 * np.pi * 300 * control_time[speech_mask])
)
control_rng = np.random.default_rng(20260713)
unit_noise = control_rng.normal(0, 1, len(control_time))

def qadd_control(waveform):
    return additive_interference_metrics(
        waveform,
        CONTROL_SR,
        strict_speech=control_speech,
        strict_internal_nonspeech=control_pauses,
    )

noise_rows = []
for noise_sd in [0.0005, 0.001, 0.002, 0.005, 0.010]:
    result = qadd_control(clean_source + noise_sd * unit_noise)
    noise_rows.append({
        "noise_sd": noise_sd,
        "qadd_nonspeech_level_dbfs": result["qadd_nonspeech_level_dbfs"],
        "qadd_snr_proxy_db": result["qadd_snr_proxy_db"],
    })
noise_controls = pd.DataFrame(noise_rows)

base_control = clean_source + 0.001 * unit_noise
hum_rows = []
for hum_amplitude in [0.0, 0.001, 0.002, 0.005, 0.010]:
    waveform = (
        base_control
        + hum_amplitude * np.sin(2 * np.pi * 60 * control_time)
    )
    result = qadd_control(waveform)
    hum_rows.append({
        "hum_amplitude": hum_amplitude,
        "qadd_hum_prominence_db": result["qadd_hum_prominence_db"],
    })
hum_controls = pd.DataFrame(hum_rows)

transient_rows = []
for injected_events in [0, 1, 2, 4]:
    waveform = base_control.copy()
    for interval in control_pauses[:injected_events]:
        center = (interval.start_sec + interval.end_sec) / 2
        start = int((center - 0.03) * CONTROL_SR)
        end = int((center + 0.03) * CONTROL_SR)
        waveform[start:end] += 0.03 * unit_noise[start:end]
    result = qadd_control(waveform)
    transient_rows.append({
        "injected_events": injected_events,
        "qadd_transient_rate_per_min": result["qadd_transient_rate_per_min"],
    })
transient_controls = pd.DataFrame(transient_rows)

white_result = qadd_control(clean_source + 0.003 * unit_noise)
tonal_waveform = clean_source + 0.003 * sum(
    np.sin(2 * np.pi * frequency * control_time)
    for frequency in [60, 120, 180, 240]
)
tonal_result = qadd_control(tonal_waveform)
flatness_controls = pd.DataFrame([
    {"control": "broadband_white", "qadd_spectral_flatness": white_result["qadd_spectral_flatness"]},
    {"control": "harmonic_tonal", "qadd_spectral_flatness": tonal_result["qadd_spectral_flatness"]},
])

gain_rows = []
for gain in [0.5, 1.0, 2.0]:
    result = qadd_control(base_control * gain)
    gain_rows.append({"gain": gain, **{
        feature: result[feature] for feature in features
    }})
gain_controls = pd.DataFrame(gain_rows)

noise_level_rho = stats.spearmanr(
    noise_controls["noise_sd"],
    noise_controls["qadd_nonspeech_level_dbfs"],
).statistic
noise_snr_rho = stats.spearmanr(
    noise_controls["noise_sd"],
    noise_controls["qadd_snr_proxy_db"],
).statistic
hum_rho = stats.spearmanr(
    hum_controls["hum_amplitude"],
    hum_controls["qadd_hum_prominence_db"],
).statistic
transient_rho = stats.spearmanr(
    transient_controls["injected_events"],
    transient_controls["qadd_transient_rate_per_min"],
).statistic

observed_level_shift = (
    gain_controls["qadd_nonspeech_level_dbfs"]
    - gain_controls.loc[gain_controls["gain"].eq(1.0), "qadd_nonspeech_level_dbfs"].iloc[0]
)
expected_level_shift = 20 * np.log10(gain_controls["gain"])
gain_level_max_error = float(
    np.max(np.abs(observed_level_shift - expected_level_shift))
)
relative_gain_ranges = {
    feature: float(gain_controls[feature].max() - gain_controls[feature].min())
    for feature in [
        "qadd_snr_proxy_db",
        "qadd_nonspeech_variability_db",
        "qadd_hum_prominence_db",
        "qadd_transient_rate_per_min",
        "qadd_spectral_flatness",
    ]
}

synthetic_checks = pd.DataFrame([
    {
        "check": "Nonspeech level increases monotonically with injected broadband noise",
        "observed": noise_level_rho,
        "criterion": "Spearman rho >= 0.95",
        "pass": bool(noise_level_rho >= 0.95),
    },
    {
        "check": "SNR proxy decreases monotonically with injected broadband noise",
        "observed": noise_snr_rho,
        "criterion": "Spearman rho <= -0.95",
        "pass": bool(noise_snr_rho <= -0.95),
    },
    {
        "check": "Hum prominence increases with injected 60-Hz hum",
        "observed": hum_rho,
        "criterion": "Spearman rho >= 0.90",
        "pass": bool(hum_rho >= 0.90),
    },
    {
        "check": "Transient rate increases with separated injected events",
        "observed": transient_rho,
        "criterion": "Spearman rho >= 0.90",
        "pass": bool(transient_rho >= 0.90),
    },
    {
        "check": "Broadband control is flatter than harmonic tonal control",
        "observed": (
            white_result["qadd_spectral_flatness"]
            - tonal_result["qadd_spectral_flatness"]
        ),
        "criterion": "difference > 0.25",
        "pass": bool(
            white_result["qadd_spectral_flatness"]
            - tonal_result["qadd_spectral_flatness"] > 0.25
        ),
    },
    {
        "check": "Nonspeech level follows the theoretical global-gain shift",
        "observed": gain_level_max_error,
        "criterion": "maximum absolute error <= 0.05 dB",
        "pass": bool(gain_level_max_error <= 0.05),
    },
    {
        "check": "SNR, variability, hum, and transient metrics are gain invariant",
        "observed": max(
            relative_gain_ranges[feature]
            for feature in relative_gain_ranges
            if feature != "qadd_spectral_flatness"
        ),
        "criterion": "maximum range <= 0.05 in native units",
        "pass": bool(
            max(
                relative_gain_ranges[feature]
                for feature in relative_gain_ranges
                if feature != "qadd_spectral_flatness"
            ) <= 0.05
        ),
    },
    {
        "check": "Spectral flatness is gain invariant",
        "observed": relative_gain_ranges["qadd_spectral_flatness"],
        "criterion": "maximum range <= 0.005",
        "pass": bool(
            relative_gain_ranges["qadd_spectral_flatness"] <= 0.005
        ),
    },
])

save_table(noise_controls, TABLES, "synthetic_noise_dose_response")
save_table(hum_controls, TABLES, "synthetic_hum_dose_response")
save_table(transient_controls, TABLES, "synthetic_transient_dose_response")
save_table(flatness_controls, TABLES, "synthetic_flatness_controls")
save_table(gain_controls, TABLES, "synthetic_global_gain_controls")
save_table(synthetic_checks, TABLES, "synthetic_validation_checks")
display(synthetic_checks)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes[0, 0].plot(
    noise_controls["noise_sd"],
    noise_controls["qadd_nonspeech_level_dbfs"],
    marker="o",
)
axes[0, 0].set_xscale("log")
axes[0, 0].set_title("A. Broadband noise → nonspeech level")
axes[0, 0].set_xlabel("Injected noise SD")
axes[0, 0].set_ylabel("dBFS")

axes[0, 1].plot(
    noise_controls["noise_sd"],
    noise_controls["qadd_snr_proxy_db"],
    marker="o",
)
axes[0, 1].set_xscale("log")
axes[0, 1].set_title("B. Broadband noise → SNR proxy")
axes[0, 1].set_xlabel("Injected noise SD")
axes[0, 1].set_ylabel("dB")

axes[0, 2].plot(
    hum_controls["hum_amplitude"],
    hum_controls["qadd_hum_prominence_db"],
    marker="o",
)
axes[0, 2].set_title("C. 60-Hz dose → hum prominence")
axes[0, 2].set_xlabel("Injected sinusoid amplitude")
axes[0, 2].set_ylabel("dB")

axes[1, 0].plot(
    transient_controls["injected_events"],
    transient_controls["qadd_transient_rate_per_min"],
    marker="o",
)
axes[1, 0].set_title("D. Separated bursts → transient rate")
axes[1, 0].set_xlabel("Injected events")
axes[1, 0].set_ylabel("events/min")

sns.barplot(
    data=flatness_controls,
    x="control",
    y="qadd_spectral_flatness",
    ax=axes[1, 1],
    color="#4C78A8",
)
axes[1, 1].set_title("E. Broadband versus tonal flatness")
axes[1, 1].set_xlabel("")
axes[1, 1].set_ylabel("ratio")

axes[1, 2].plot(
    gain_controls["gain"],
    observed_level_shift,
    marker="o",
    label="Observed",
)
axes[1, 2].plot(
    gain_controls["gain"],
    expected_level_shift,
    marker="s",
    linestyle="--",
    label="20 log10(gain)",
)
axes[1, 2].set_xscale("log", base=2)
axes[1, 2].set_title("F. Global-gain verification")
axes[1, 2].set_xlabel("Global amplitude gain")
axes[1, 2].set_ylabel("Nonspeech-level shift (dB)")
axes[1, 2].legend()

fig.suptitle("Additive-interference deterministic analytical controls", fontweight="bold")
fig.tight_layout()
save_figure(fig, FIGURES, "synthetic_validation_controls")
plt.show()"""
        ),
        markdown(
            "## 6. Empirical distributions and support dependence\n\n"
            "The first figure shows the actual feature ranges and missingness. The second asks "
            "whether a metric is strongly associated with the amount of available signal. "
            "Support dependence is diagnostic, not automatically a reason to delete values."
        ),
        code(
            r"""display_names = family_registry.set_index("feature")["display_name"].to_dict()
units = family_registry.set_index("feature")["unit"].to_dict()
directions = family_registry.set_index("feature")["worse"].to_dict()

ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9))
axes = axes.ravel()
for ax, feature in zip(axes, features):
    values = pd.to_numeric(metrics[feature], errors="coerce")
    finite = values[np.isfinite(values)]
    sns.histplot(finite, bins=30, ax=ax, color="#4C78A8")
    ax.set_title(display_names[feature], fontsize=10)
    ax.set_xlabel(f"{units[feature]} | worse: {directions[feature]}")
    ax.set_ylabel("Recordings")
    ax.text(
        0.98,
        0.95,
        f"n={len(finite):,}\nmissing={values.isna().mean():.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
fig.suptitle("Additive-interference empirical distributions", fontweight="bold")
fig.tight_layout()
save_figure(fig, FIGURES, "metric_distributions")
plt.show()

primary_support_field = {
    "qadd_nonspeech_level_dbfs": "qadd_nonspeech_support_sec",
    "qadd_snr_proxy_db": "qadd_speech_support_sec",
    "qadd_nonspeech_variability_db": "qadd_nonspeech_support_sec",
    "qadd_hum_prominence_db": "qadd_hum_spectral_support_sec",
    "qadd_transient_rate_per_min": "qadd_nonspeech_support_sec",
    "qadd_spectral_flatness": "qadd_flatness_spectral_support_sec",
}
support_association_rows = []
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()
for ax, feature in zip(axes, features):
    support_field = primary_support_field[feature]
    x = pd.to_numeric(metrics[support_field], errors="coerce")
    y = pd.to_numeric(metrics[feature], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y)
    rho = (
        stats.spearmanr(x[valid], y[valid]).statistic
        if valid.sum() >= 5 and x[valid].nunique() >= 2 and y[valid].nunique() >= 2
        else np.nan
    )
    support_association_rows.append({
        "feature": feature,
        "support_field": support_field,
        "recordings": int(valid.sum()),
        "spearman_rho": rho,
        "interpretation": (
            "review strong dependence; do not infer bias from rho alone"
            if np.isfinite(rho) and abs(rho) >= 0.30 else "no automatic action"
        ),
    })
    ax.scatter(x[valid], y[valid], s=16, alpha=0.45, color="#4C78A8")
    ax.set_title(f"{display_names[feature]}\nSpearman rho={rho:.2f}" if np.isfinite(rho)
                 else f"{display_names[feature]}\nSpearman rho=NA")
    ax.set_xlabel(support_field.replace("qadd_", "").replace("_", " "))
    ax.set_ylabel(units[feature])
fig.suptitle("Metric value versus primary support quantity", fontweight="bold")
fig.tight_layout()
save_figure(fig, FIGURES, "metric_support_dependence")
plt.show()

support_associations = pd.DataFrame(support_association_rows)
save_table(support_associations, TABLES, "support_associations")
display(support_associations)"""
        ),
        markdown(
            "## 7. Acquisition-stratified audit and empirical examples\n\n"
            "These summaries expose codec, native sample-rate, and channel composition. They "
            "are descriptive controls, not grounds for post-hoc threshold tuning.\n\n"
            "The empirical review queue contains low, median-nearest, high, and robust-extreme "
            "examples. A genuine extreme artifact stays in the dataset. Exclude or reprocess "
            "only a documented decode, segmentation, or extraction failure."
        ),
        code(
            r"""technical_columns = [
    column for column in ["native_codec", "native_sample_rate_hz", "native_channels"]
    if column in metrics
]
stratified_rows = []
for technical_column in technical_columns:
    for stratum, group in metrics.groupby(technical_column, dropna=False):
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            stratified_rows.append({
                "technical_variable": technical_column,
                "stratum": str(stratum),
                "feature": feature,
                "recordings_total": len(group),
                "recordings_nonmissing": len(finite),
                "median": finite.median(),
                "q25": finite.quantile(0.25),
                "q75": finite.quantile(0.75),
            })
acquisition_summary = pd.DataFrame(stratified_rows)
save_table(acquisition_summary, TABLES, "acquisition_stratified_summary")
display(acquisition_summary.head(30))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
if "native_codec" in metrics:
    codec_order = metrics["native_codec"].fillna("missing").value_counts().index[:8]
    codec_counts = (
        metrics.assign(native_codec=metrics["native_codec"].fillna("missing"))
        ["native_codec"].value_counts().reindex(codec_order)
    )
    axes[0, 0].bar(codec_counts.index.astype(str), codec_counts.values, color="#4C78A8")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].set_title("A. Recordings by native codec")
    axes[0, 0].set_ylabel("Recordings")
    codec_plot = metrics.loc[metrics["native_codec"].fillna("missing").isin(codec_order)].copy()
    codec_plot["native_codec"] = codec_plot["native_codec"].fillna("missing")
    sns.boxplot(
        data=codec_plot,
        x="native_codec",
        y="qadd_nonspeech_level_dbfs",
        order=codec_order,
        showfliers=False,
        ax=axes[0, 1],
        color="#A0CBE8",
    )
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].set_title("B. Nonspeech level by codec")
else:
    axes[0, 0].set_axis_off()
    axes[0, 1].set_axis_off()

if "native_sample_rate_hz" in metrics:
    sample_rate_plot = metrics.copy()
    sample_rate_plot["native_sample_rate_hz"] = (
        pd.to_numeric(sample_rate_plot["native_sample_rate_hz"], errors="coerce")
        .astype("Int64").astype(str)
    )
    sns.boxplot(
        data=sample_rate_plot,
        x="native_sample_rate_hz",
        y="qadd_snr_proxy_db",
        showfliers=False,
        ax=axes[1, 0],
        color="#FFBE7D",
    )
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].set_title("C. SNR proxy by native sample rate")
else:
    axes[1, 0].set_axis_off()

if "native_channels" in metrics:
    channel_status = (
        metrics.groupby(["native_channels", "qadd_status"], dropna=False)
        .size().rename("recordings").reset_index()
    )
    sns.barplot(
        data=channel_status,
        x="native_channels",
        y="recordings",
        hue="qadd_status",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("D. Family support status by native channels")
else:
    axes[1, 1].set_axis_off()
fig.suptitle("Additive-interference acquisition audit", fontweight="bold")
fig.tight_layout()
save_figure(fig, FIGURES, "acquisition_stratified_audit")
plt.show()

review_rows = []
for spec in family_registry.itertuples(index=False):
    values = pd.to_numeric(metrics[spec.feature], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        continue
    median = float(finite.median())
    mad = float(np.median(np.abs(finite - median)))
    robust_z = pd.Series(np.nan, index=metrics.index, dtype=float)
    if mad > 0:
        robust_z.loc[finite.index] = 0.67448975 * (finite - median) / mad
    positions = {
        "low": finite.idxmin(),
        "median_nearest": (finite - median).abs().idxmin(),
        "high": finite.idxmax(),
    }
    for position, index in positions.items():
        review_rows.append({
            "feature": spec.feature,
            "display_name": spec.display_name,
            "position": position,
            "file_name": metrics.at[index, "file_name"],
            "logical_recording_id": metrics.at[index, "logical_recording_id"],
            "value": values.at[index],
            "unit": spec.unit,
            "robust_z": robust_z.at[index],
            "review_instruction": "listen/check provenance; retain if genuine",
        })
    for index in robust_z.index[robust_z.abs().ge(4.5).fillna(False)]:
        review_rows.append({
            "feature": spec.feature,
            "display_name": spec.display_name,
            "position": "robust_extreme",
            "file_name": metrics.at[index, "file_name"],
            "logical_recording_id": metrics.at[index, "logical_recording_id"],
            "value": values.at[index],
            "unit": spec.unit,
            "robust_z": robust_z.at[index],
            "review_instruction": "check extraction/decode/segmentation; do not remove genuine artifact",
        })

empirical_review_queue = (
    pd.DataFrame(review_rows)
    .drop_duplicates(["feature", "logical_recording_id", "position"])
    .sort_values(["feature", "position", "value"])
)
media_lookup = frozen_recordings[
    ["logical_recording_id", "media_path"]
].drop_duplicates("logical_recording_id")
empirical_review_queue = empirical_review_queue.merge(
    media_lookup, on="logical_recording_id", how="left", validate="many_to_one"
)
save_table(empirical_review_queue, TABLES, "empirical_example_review_queue")
display(empirical_review_queue)

# Optional playback: change FEATURE_TO_AUDIT and rerun this small block.
from IPython.display import Audio
FEATURE_TO_AUDIT = "qadd_nonspeech_level_dbfs"
examples = empirical_review_queue.loc[
    empirical_review_queue["feature"].eq(FEATURE_TO_AUDIT)
]
for row in examples.itertuples(index=False):
    display(Markdown(
        f"**{row.position}: {row.file_name} — {row.value:.3f} {row.unit}**"
    ))
    if isinstance(row.media_path, str) and Path(row.media_path).exists():
        display(Audio(filename=row.media_path))
    else:
        print("Audio path not available:", row.media_path)"""
        ),
        markdown(
            "## 8. Decision gate and outputs\n\n"
            "A PASS means the implementation contract, mathematical checks, support/value "
            "logic, and deterministic controls passed. It does **not** yet establish "
            "perceptual validity; that is tested later against the separate 4RA and 2RA "
            "systems after all families are frozen."
        ),
        code(
            r"""technical_failures = integrity_checks.loc[
    integrity_checks["blocking"] & ~integrity_checks["pass"], "check"
].tolist()
synthetic_failures = synthetic_checks.loc[
    ~synthetic_checks["pass"], "check"
].tolist()
blocking = [
    *[f"Integrity: {item}" for item in technical_failures],
    *[f"Synthetic control: {item}" for item in synthetic_failures],
]

review_notes = []
robust_extreme_count = int(
    empirical_review_queue["position"].eq("robust_extreme").sum()
)
if robust_extreme_count:
    review_notes.append(
        f"{robust_extreme_count} feature-specific robust extremes are queued for "
        "audio/provenance inspection; genuine artifacts remain included."
    )
strong_support = support_associations.loc[
    support_associations["spearman_rho"].abs().ge(0.30).fillna(False),
    "feature",
].tolist()
if strong_support:
    review_notes.append(
        "Strong descriptive support association (|rho|>=0.30): "
        + ", ".join(strong_support)
        + ". Review the saved scatterplots; do not tune thresholds post hoc."
    )

decision_summary = pd.DataFrame([
    {"item": "frozen eligible recordings expected", "value": len(expected_ids)},
    {"item": "feature rows extracted", "value": len(metrics)},
    {"item": "primary metrics registered", "value": int(family_registry["role"].eq("primary").sum())},
    {"item": "secondary metrics registered", "value": int(family_registry["role"].eq("secondary").sum())},
    {"item": "blocking integrity failures", "value": len(technical_failures)},
    {"item": "failed deterministic controls", "value": len(synthetic_failures)},
    {"item": "robust empirical examples queued", "value": robust_extreme_count},
])
save_table(decision_summary, TABLES, "decision_summary")
display(decision_summary)

additive_stage_ready = stage_gate(
    "Additive interference",
    not blocking,
    blocking,
    (
        "If PASS, inspect the empirical example queue and saved figures, document any "
        "decode/extraction anomaly, then open 02b Gain and amplitude dynamics. "
        "Do not exclude a recording merely because a valid Q metric is extreme."
    ),
)
if review_notes:
    display(Markdown("### Non-blocking scientific review notes\n\n" + "\n".join(
        f"- {note}" for note in review_notes
    )))

display(Markdown(
    "### Main outputs\n\n"
    "- `tables/feature_definitions.csv`: formulas, units, directions, support, ranges, and confounding.\n"
    "- `tables/integrity_checks.csv`: blocking implementation checks.\n"
    "- `tables/metric_summary.csv` and `metric_status_summary.csv`: empirical support and values.\n"
    "- `tables/synthetic_validation_checks.csv`: deterministic control results.\n"
    "- `tables/empirical_example_review_queue.csv`: low/middle/high/extreme recordings for inspection.\n"
    "- `figures/`: support, control, distribution, support-dependence, and acquisition figures."
))"""
        ),
    ]


for index, (filename, title, family, prefix) in enumerate(FAMILY_SPECS):
    write_notebook(
        f"notebooks/02_feature_extraction/{filename}",
        title,
        (
            "Defines, verifies, and audits additive-interference metrics before any "
            "clinical or perceptual association analysis."
            if index == 0
            else "Reviews registered Q metrics, extraction support, observed ranges, and saved distributions."
        ),
        (
            additive_feature_cells()
            if index == 0
            else feature_cells(family, prefix, run_extractor=False)
        ),
    )


write_notebook(
    "notebooks/03_dataset_assembly/03a_assemble_analysis_dataset.ipynb",
    "03a — Assemble the audited analysis dataset",
    "Performs validated one-to-one merging and creates explicit measurement, diagnosis, segmentation, and clinical gates.",
    [
        markdown(
            "## Main output and decisions\n\n"
            "Main output: `outputs/03_dataset_assembly/paper1_analysis_dataset.csv`. "
            "Investigate any merge/extraction failure or unexpected task-completion missingness "
            "before analysis."
        ),
        cli_switch_cell(
            "RUN_ASSEMBLY",
            "assemble",
            "Assembly not run. Enable only after all six feature-family gates pass.",
        ),
        code(
            r"""STAGE, FIGURES, TABLES = stage_directories("03_dataset_assembly")
flow = read_table(STAGE / "eligibility_flow_counts")
data = read_table(STAGE / "paper1_analysis_dataset")
save_table(flow, TABLES, "eligibility_flow_counts")
display(flow)

plot = flow.sort_values("n_true")
fig, ax = plt.subplots(figsize=(11, 5.5))
sns.barplot(data=plot, x="n_true", y="criterion", color="#4C78A8", ax=ax)
for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)
ax.set(title="Dataset assembly eligibility and exclusion gates", xlabel="Logical recordings", ylabel="")
fig.tight_layout()
save_figure(fig, FIGURES, "eligibility_flow")
plt.show()

blocking_columns = ["feature_extraction_missing"]
blocking = [
    f"{column}: {int(data[column].fillna(False).sum())} recordings"
    for column in blocking_columns
    if column in data and data[column].fillna(False).any()
]
assembly_ready = stage_gate(
    "Dataset assembly",
    not blocking,
    blocking,
    "Open the dataset statistics notebook if PASS; otherwise reconcile the saved rows.",
)"""
        ),
    ],
)


write_notebook(
    "notebooks/03_dataset_assembly/03b_dataset_statistics.ipynb",
    "03b — Frozen cohort and missingness accounting",
    "Produces all participant/recording denominators from the assembled dataset.",
    [
        markdown(
            "## Main output and decisions\n\n"
            "Main outputs: diagnosis-stratified cohort counts and metric missingness. No "
            "manuscript denominator should be typed manually."
        ),
        code(
            r"""STAGE, FIGURES, TABLES = stage_directories(Path("03_dataset_assembly") / "statistics")
data = read_table(OUTPUT / "03_dataset_assembly" / "paper1_analysis_dataset")
cohort = (
    data.groupby("diagnosis_analysis", dropna=False)
    .agg(
        participants=("SubjectID", "nunique"),
        logical_recordings=("logical_recording_id", "nunique"),
        primary_eligible=("primary_measurement_eligible", "sum"),
    )
    .reset_index()
)
status_columns = [column for column in data if column.endswith("_status")]
missingness = (
    data.select_dtypes(include=[np.number]).isna().mean()
    .sort_values(ascending=False).rename("missing_fraction").reset_index(names="variable")
)
save_table(cohort, TABLES, "cohort_counts")
save_table(missingness, TABLES, "numeric_missingness")
display(cohort)
display(missingness.head(30))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cohort_long = cohort.melt(
    id_vars="diagnosis_analysis",
    value_vars=["participants", "logical_recordings"],
    var_name="unit", value_name="count",
)
sns.barplot(data=cohort_long, x="diagnosis_analysis", y="count", hue="unit", ax=axes[0])
for container in axes[0].containers:
    axes[0].bar_label(container, fmt="%.0f", padding=3)
axes[0].set(title="Frozen cohort denominators", xlabel="", ylabel="Count")
top_missing = missingness.head(20).sort_values("missing_fraction")
sns.barplot(data=top_missing, x="missing_fraction", y="variable", color="#E2AE4D", ax=axes[1])
axes[1].set(title="Highest numeric missingness", xlabel="Missing fraction", ylabel="")
fig.tight_layout()
save_figure(fig, FIGURES, "cohort_and_missingness_summary")
plt.show()

statistics_ready = stage_gate(
    "Cohort statistics",
    data["diagnosis_analysis"].isin(["ALS", "CONTROLS"]).all(),
    ["Non-target or missing frozen diagnoses remain in the assembled table."]
    if not data["diagnosis_analysis"].isin(["ALS", "CONTROLS"]).all() else [],
    "Run Goal 1 descriptive analysis if PASS.",
)"""
        ),
    ],
)


write_notebook(
    "notebooks/04_analysis/05_study_goal_1_acquisition_variability.ipynb",
    "05 — Goal 1: occurrence and acquisition variability",
    "Runs participant-clustered descriptives and participant-level ALS/control contrasts.",
    [
        markdown(
            "## Main output and decisions\n\n"
            "Main output: metric support/descriptives and exploratory participant-level effect "
            "sizes. Diagnosis-associated Q differences are acquisition/confounding patterns, "
            "not diagnostic biomarker performance."
        ),
        cli_switch_cell("RUN_GOAL1", "describe", "Goal 1 analysis not run."),
        code(
            r"""STAGE, FIGURES, TABLES = stage_directories(Path("04_analysis") / "goal1")
descriptive = read_table(OUTPUT / "04_analysis" / "descriptive" / "metric_descriptive_statistics")
contrasts = read_table(OUTPUT / "04_analysis" / "descriptive" / "exploratory_participant_level_diagnosis_contrasts")
save_table(descriptive, TABLES, "metric_descriptive_statistics")
save_table(contrasts, TABLES, "participant_level_diagnosis_contrasts")
display(descriptive)
display(contrasts)

plot = contrasts.loc[contrasts["status"].eq("ok")].sort_values("cliffs_delta_a_vs_b")
fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(plot))))
if not plot.empty:
    y = np.arange(len(plot))
    ax.errorbar(
        plot["cliffs_delta_a_vs_b"], y,
        xerr=[
            plot["cliffs_delta_a_vs_b"] - plot["cliffs_delta_ci_low"],
            plot["cliffs_delta_ci_high"] - plot["cliffs_delta_a_vs_b"],
        ],
        fmt="o", color="#4C78A8", ecolor="#9ECAE1", capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot["feature"], fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set(title="Participant-level ALS vs control Q contrasts", xlabel="Cliff's delta (ALS vs controls)")
fig.tight_layout()
save_figure(fig, FIGURES, "diagnosis_effect_size_forest")
plt.show()

blocked = contrasts["status"].ne("ok").sum() if not contrasts.empty else 0
goal1_ready = stage_gate(
    "Goal 1",
    blocked == 0,
    [f"{blocked} contrasts were under-supported or failed; retain them as explicit blocked rows."]
    if blocked else [],
    "Proceed to Goal 2; do not delete under-supported estimands.",
)"""
        ),
    ],
)


write_notebook(
    "notebooks/04_analysis/06_study_goal_2_participant_persistence.ipynb",
    "06 — Goal 2: participant persistence",
    "Visualizes participant-level variance persistence without calling it test–retest reliability.",
    [
        markdown(
            "## Main output and decisions\n\n"
            "Main output: participant persistence ICC estimates and model-status table. Review "
            "sparse, under-supported, or nonconverged features; never relabel this as reliability."
        ),
        code(
            r"""STAGE, FIGURES, TABLES = stage_directories(Path("04_analysis") / "goal2")
persistence = read_table(OUTPUT / "04_analysis" / "descriptive" / "participant_persistence_not_reliability")
save_table(persistence, TABLES, "participant_persistence")
display(persistence)

plot = persistence.loc[persistence["status"].eq("ok")].sort_values("persistence_icc")
fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(plot))))
sns.barplot(data=plot, x="persistence_icc", y="feature", color="#59A14F", ax=ax)
ax.set(xlim=(0, 1), title="Participant rank persistence (not test–retest reliability)", xlabel="Variance-partition ICC", ylabel="")
fig.tight_layout()
save_figure(fig, FIGURES, "participant_persistence")
plt.show()

status = persistence["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="features")
save_table(status, TABLES, "persistence_model_status")
goal2_ready = stage_gate(
    "Goal 2",
    status.loc[status["status"].astype(str).str.startswith("model_failed"), "features"].sum() == 0,
    status.loc[~status["status"].eq("ok")].astype(str).agg(": ".join, axis=1).tolist(),
    "Proceed to Goal 3 after documenting every skipped/under-supported feature.",
)"""
        ),
    ],
)


write_notebook(
    "notebooks/04_analysis/07_study_goal_3_internal_structure_and_robustness.ipynb",
    "07 — Goal 3: multidimensional structure and robustness",
    "Visualizes participant-clustered feature structure and preserves pairwise support.",
    [
        markdown(
            "## Main output and decisions\n\n"
            "Main output: correlation and support matrices. Interpret only estimable pairs; "
            "do not impute or erase sparse/failed metrics to make the structure look cleaner."
        ),
        code(
            r"""STAGE, FIGURES, TABLES = stage_directories(Path("04_analysis") / "goal3")
correlations = read_table(OUTPUT / "04_analysis" / "descriptive" / "pairwise_clustered_spearman")
save_table(correlations, TABLES, "pairwise_clustered_spearman")
ok = correlations.loc[correlations["status"].eq("ok")]
features = sorted(set(ok["feature_left"]) | set(ok["feature_right"]))
matrix = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
for row in ok.itertuples():
    matrix.loc[row.feature_left, row.feature_right] = row.rho
    matrix.loc[row.feature_right, row.feature_left] = row.rho
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Participant-clustered Spearman structure")
fig.tight_layout()
save_figure(fig, FIGURES, "clustered_spearman_heatmap")
plt.show()

blocked = int(correlations["status"].ne("ok").sum())
goal3_ready = stage_gate(
    "Goal 3",
    True,
    [f"{blocked} pairwise estimands are explicitly under-supported."] if blocked else [],
    "Proceed to Goal 4; use pair-specific denominators in all interpretation.",
)"""
        ),
    ],
)


write_notebook(
    "notebooks/04_analysis/08_study_goal_4_perceptual_family_alignment.ipynb",
    "08 — Goal 4: perceptual family alignment and reliability",
    "Separates distributed main ratings, crossed four-RA reliability, and the broad two-RA labels.",
    [
        markdown(
            "## Main output and decisions\n\n"
            "Main outputs: four-RA agreement, rater-stratified distributed alignment, crossed "
            "consensus alignment, and paired comparison with broad two-RA labels. Confirm scale "
            "direction and rater-folder design before running."
        ),
        cli_switch_cell(
            "RUN_GOAL4",
            "human-qc --schema config/human_qc_schema.yaml",
            "Goal 4 not run. Confirm the schema, four RA names, crossed Reliability folders, and label direction first.",
        ),
        code(
            r"""STAGE, FIGURES, TABLES = stage_directories(Path("04_analysis") / "goal4")
source = OUTPUT / "04_analysis" / "human_qc"
agreement = read_table(source / "reliability_interrater_agreement_complete")
alignment = read_table(source / "main_distributed_rater_stratified_family_alignment")
direction = read_table(source / "two_ra_broad_direction_and_scale_audit")
save_table(agreement, TABLES, "four_ra_interrater_agreement")
save_table(alignment, TABLES, "main_rater_stratified_alignment")
save_table(direction, TABLES, "two_ra_direction_and_scale_audit")
display(direction)
display(agreement)
display(alignment)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
if not agreement.empty:
    sns.barplot(data=agreement, x="category", y="gwet_ac1_nominal", color="#59A14F", ax=axes[0])
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set(title="Four-RA crossed reliability", xlabel="", ylabel="Gwet AC1")
matched = alignment.loc[alignment["matched_family"].fillna(False)]
if not matched.empty:
    sns.barplot(data=matched, x="human_family", y="effect", color="#4C78A8", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set(title="Distributed main ratings: matched-family alignment", xlabel="", ylabel="Alignment effect")
fig.tight_layout()
save_figure(fig, FIGURES, "reliability_and_family_alignment")
plt.show()

direction_ok = (
    not direction.empty
    and direction["direction"].astype(str).eq("higher_is_worse").all()
)
goal4_ready = stage_gate(
    "Goal 4",
    direction_ok and not agreement.empty,
    ([] if direction_ok else ["Broad two-RA scale direction is not confirmed as higher-is-worse."])
    + ([] if not agreement.empty else ["Four-RA reliability output is missing or blocked."]),
    "Run sensitivity summary only after reliability and scale-direction gates pass.",
)"""
        ),
    ],
)


write_notebook(
    "notebooks/04_analysis/09_sensitivity_summary.ipynb",
    "09 — Sensitivity and robustness summary",
    "Compares segmentation profiles, encodings, one-recording-per-participant estimates, and exact-session Rest context.",
    [
        markdown(
            "## Main output and decisions\n\n"
            "Main output: a visual robustness dashboard and the underlying sensitivity tables. "
            "Any conclusion that changes sign or loses adequate support must be reported as "
            "sensitive rather than hidden."
        ),
        cli_switch_cell("RUN_CONSERVATIVE", "extract --profile conservative", "Conservative profile not run."),
        cli_switch_cell("RUN_PERMISSIVE", "extract --profile permissive", "Permissive profile not run."),
        cli_switch_cell("RUN_SENSITIVITY", "sensitivity", "Sensitivity aggregation not run."),
        cli_switch_cell("RUN_ENCODING", "encoding-sensitivity", "Encoding sensitivity not run."),
        cli_switch_cell("RUN_REST", "rest-reference", "Rest-reference sensitivity not run."),
        code(
            r"""STAGE, FIGURES, TABLES = stage_directories(Path("04_analysis") / "sensitivity_summary")
profile = read_table(OUTPUT / "04_analysis" / "sensitivity" / "segmentation_profile_robustness")
encoding = read_table(OUTPUT / "04_analysis" / "encoding_sensitivity" / "paired_encoding_robustness")
rest = read_table(OUTPUT / "04_analysis" / "rest_reference" / "rest_reference_summary")
save_table(profile, TABLES, "segmentation_profile_robustness")
save_table(encoding, TABLES, "encoding_robustness")
save_table(rest, TABLES, "rest_reference_summary")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
if not profile.empty:
    sns.barplot(data=profile, x="profile", y="spearman_rho", hue="feature", ax=axes[0])
    axes[0].legend([], [], frameon=False)
    axes[0].set(title="Segmentation-profile robustness", ylabel="Spearman rho vs primary")
if not encoding.empty:
    plot = encoding.sort_values("spearman_rho").tail(20)
    sns.barplot(data=plot, x="spearman_rho", y="feature", color="#4C78A8", ax=axes[1])
    axes[1].set(title="WAV/WEBM metric agreement", xlabel="Spearman rho", ylabel="")
fig.tight_layout()
save_figure(fig, FIGURES, "sensitivity_dashboard")
plt.show()

missing_outputs = [
    name for name, frame in {
        "segmentation profiles": profile,
        "encoding comparison": encoding,
        "Rest reference": rest,
    }.items() if frame.empty
]
sensitivity_ready = stage_gate(
    "Sensitivity summary",
    not missing_outputs,
    ["Missing output: " + name for name in missing_outputs],
    "Proceed to manuscript tables/figures only after all prespecified sensitivities exist.",
)"""
        ),
    ],
)


print("Generated auditable notebooks under", ROOT / "notebooks")
