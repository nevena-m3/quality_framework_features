"""Generate the long-form, auditable visualization notebooks.

Unlike the thin orchestration notebooks under ``notebooks/``, these notebooks keep
plotting, denominator checks, table construction, and figure export code visible so a
reviewer can audit every paper-facing result.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


BOOTSTRAP = r"""from pathlib import Path
import hashlib
import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, Markdown, display

def find_project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "paper1_qc").exists():
            return candidate
    raise FileNotFoundError("Open Jupyter from inside the paper_1 project.")

ROOT = find_project_root()
CONFIG = ROOT / "config" / "project.yaml"
OUTPUT = ROOT / "outputs"
MAIN_OUTPUTS = ROOT / "MAIN outputs"
VIZ_ROOT = OUTPUT / "visualization"
sys.path.insert(0, str(ROOT / "src"))

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def read_stage(relative_without_suffix):
    stem = OUTPUT / relative_without_suffix
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        try:
            return pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise FileNotFoundError(f"Missing required stage table: {parquet} or {csv}")

def read_optional_stage(relative_without_suffix):
    stem = OUTPUT / relative_without_suffix
    if stem.with_suffix(".parquet").exists() or stem.with_suffix(".csv").exists():
        return read_stage(relative_without_suffix)
    print("OPTIONAL TABLE NOT AVAILABLE:", relative_without_suffix)
    return pd.DataFrame()

def run_cli(*arguments):
    command = [sys.executable, "-m", "paper1_qc.cli", "--config", str(CONFIG), *arguments]
    print("RUN:", " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, check=True)

def save_table(frame, folder, name):
    target = VIZ_ROOT / folder / "tables"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.csv"
    frame.to_csv(path, index=False)
    print("TABLE:", path.relative_to(ROOT), f"({len(frame):,} rows)")
    return path

def save_figure(fig, folder, name):
    target = VIZ_ROOT / folder / "figures"
    target.mkdir(parents=True, exist_ok=True)
    png = target / f"{name}.png"
    svg = target / f"{name}.svg"
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
print("Visualization outputs:", VIZ_ROOT)
"""


def notebook(title: str, purpose: str, cells: list[dict]) -> dict:
    return {
        "cells": [
            md(
                f"# {title}\n\n{purpose}\n\n"
                "Every displayed denominator and paper-facing visual is also saved under "
                "separate `figures/` and `tables/` folders within `outputs/visualization/`. "
                "Empty or under-supported analyses remain visible as audit rows; they are "
                "never silently removed. The final cell explains whether the next stage is allowed."
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


def write(name: str, title: str, purpose: str, cells: list[dict]) -> None:
    path = ROOT / "visualization" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(title, purpose, cells), indent=1), encoding="utf-8")


write(
    "00_preflight_and_run_order.ipynb",
    "00 — Preflight, data gates, and execution order",
    "Validates the local environment and runs or reviews the non-negotiable metadata/media gates.",
    [
        md(
            "Set `RUN_PIPELINE_STAGES=True` only after `config/project.yaml` points to the "
            "updated data root. Audit and inventory run before the immutable data freeze. "
            "The template step never overwrites an existing adjudication file."
        ),
        code(
            r"""RUN_PIPELINE_STAGES = False

if RUN_PIPELINE_STAGES:
    run_cli("audit")
    run_cli("inventory")
    run_cli("freeze-template")
else:
    print("Dry review only. Set RUN_PIPELINE_STAGES=True to run audit, inventory, and freeze-template.")
"""
        ),
        md(
            "Open `config/metadata_adjudication.csv`; enter `ALS`, `CONTROLS`, or "
            "`EXCLUDE` and an evidence source for every row. Then set `RUN_DATA_FREEZE=True` "
            "once. A completed version cannot be overwritten."
        ),
        code(
            r"""RUN_DATA_FREEZE = False

if RUN_DATA_FREEZE:
    run_cli("freeze")
else:
    print("Freeze not requested in this run.")
"""
        ),
        code(
            r"""audit_summary = read_stage("00_audit/bamboo_audit_summary")
cross_workbook = read_stage("00_audit/cross_workbook_summary")
metadata_issues = read_stage("00_audit/bamboo_audit_issues")
inventory = read_stage("00_audit/bamboo_media_inventory")

display(audit_summary)
display(cross_workbook)

issue_counts = (
    metadata_issues.groupby(["severity", "rule"], dropna=False)
    .size().rename("n").reset_index().sort_values(["severity", "n"], ascending=[True, False])
)
save_table(issue_counts, "00_preflight", "metadata_issue_counts")
display(issue_counts)

media_summary = pd.DataFrame({
    "recordings_on_disk": [inventory["file_name"].nunique()],
    "physical_files": [len(inventory)],
    "extensions": [", ".join(sorted(inventory["file_name"].map(lambda value: Path(str(value)).suffix.lower()).unique()))],
    "probe_failures": [int((~inventory["probe_ok"].fillna(False)).sum()) if "probe_ok" in inventory else np.nan],
})
save_table(media_summary, "00_preflight", "media_inventory_summary")
display(media_summary)
"""
        ),
        code(
            r"""# Human-QC folder design can be checked before expensive signal processing.
import yaml

project_cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
schema_path = ROOT / "config" / "human_qc_schema.yaml"
assert schema_path.exists(), "Copy config/human_qc_schema.example.yaml to config/human_qc_schema.yaml."
human_schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
data_root = Path(project_cfg["paths"]["data_root"])
human_root = data_root / project_cfg["paths"]["detailed_human_qc"]
ra_names = human_schema["rater_directory_names"]
reliability_root = human_root / human_schema.get("reliability_subdirectory", "Reliability")

main_sets = {
    ra: {path.name for path in (human_root / ra).rglob("*.csv")}
    if (human_root / ra).exists() else set()
    for ra in ra_names
}
reliability_sets = {
    ra: {path.name for path in (reliability_root / ra).rglob("*.csv")}
    if (reliability_root / ra).exists() else set()
    for ra in ra_names
}
folder_counts = pd.DataFrame([
    {
        "rater_id": ra,
        "main_csv_files": len(main_sets[ra]),
        "reliability_csv_files": len(reliability_sets[ra]),
        "main_directory_exists": (human_root / ra).exists(),
        "reliability_directory_exists": (reliability_root / ra).exists(),
    }
    for ra in ra_names
])
save_table(folder_counts, "00_preflight", "human_qc_folder_counts")
display(folder_counts)

main_owners = {}
for ra, names in main_sets.items():
    for name in names:
        main_owners.setdefault(name, []).append(ra)
main_overlap = pd.DataFrame([
    {"export_file": name, "n_main_raters": len(owners), "main_raters": "|".join(owners)}
    for name, owners in sorted(main_owners.items()) if len(owners) > 1
], columns=["export_file", "n_main_raters", "main_raters"])
save_table(main_overlap, "00_preflight", "unexpected_main_assignment_overlap")

reliability_union = set().union(*reliability_sets.values()) if reliability_sets else set()
reliability_intersection = set.intersection(*reliability_sets.values()) if reliability_sets else set()
reliability_gaps = pd.DataFrame([
    {
        "export_file": name,
        "n_raters_present": sum(name in reliability_sets[ra] for ra in ra_names),
        "missing_raters": "|".join(ra for ra in ra_names if name not in reliability_sets[ra]),
    }
    for name in sorted(reliability_union)
    if any(name not in reliability_sets[ra] for ra in ra_names)
], columns=["export_file", "n_raters_present", "missing_raters"])
save_table(reliability_gaps, "00_preflight", "reliability_filename_coverage_gaps")

design_check = pd.DataFrame([{
    "main_files_have_one_assignment_by_export_name": len(main_overlap) == 0,
    "reliability_union_files": len(reliability_union),
    "reliability_files_common_to_all_four_raters": len(reliability_intersection),
    "reliability_files_with_filename_coverage_gaps": len(reliability_gaps),
    "primary_agreement_gate": (
        "provisional_pass"
        if len(reliability_intersection) > 0 and len(reliability_gaps) == 0
        else "review_required"
    ),
}])
save_table(design_check, "00_preflight", "human_qc_folder_design_check")
display(design_check)
if len(main_overlap):
    display(main_overlap.head(50))
if len(reliability_gaps):
    display(reliability_gaps.head(50))
"""
        ),
        code(
            r"""# Hard-stop ledger: resolve every error before interpreting Q.
blocking = metadata_issues.loc[metadata_issues["severity"].astype(str).str.lower().eq("error")].copy()
save_table(blocking, "00_preflight", "blocking_metadata_issues")
print(f"Blocking metadata rows: {len(blocking):,}")
if len(blocking):
    display(blocking.head(50))
"""
        ),
        code(
            r"""# Stage decision: the immutable freeze, not the raw audit count, controls continuation.
freeze_version = str(project_cfg.get("data_freeze", {}).get("version", "v1"))
freeze_root = MAIN_OUTPUTS / "00_DATA_FREEZE" / freeze_version
freeze_manifest = freeze_root / "data_freeze_manifest.json"
confirmed_exceptional = project_cfg.get("data_freeze", {}).get(
    "confirmed_control_subject_ids", []
)
exceptional_evidence = project_cfg.get("data_freeze", {}).get(
    "confirmed_control_subject_evidence", ""
)
preflight_reasons = []
if confirmed_exceptional and not str(exceptional_evidence).strip():
    preflight_reasons.append("Exceptional control IDs are listed without evidence.")
if not freeze_manifest.exists():
    preflight_reasons.append(
        f"Immutable freeze is missing: {freeze_manifest.relative_to(ROOT)}"
    )
preflight_ready = stage_gate(
    "Preflight and data freeze",
    not preflight_reasons,
    preflight_reasons,
    "Open 01_segmentation_visual_audit.ipynb only after this gate passes.",
)"""
        ),
    ],
)


write(
    "01_segmentation_visual_audit.ipynb",
    "01 — Segmentation visual audit",
    "Runs Silero segmentation and audits the exact original per-recording frame, segment, and four-panel figure artifacts.",
    [
        md(
            "The visible Silero artifacts in this notebook reproduce the original pipeline: "
            "one 30-ms frame CSV, one segment CSV, and one four-panel PNG per recording. "
            "All of them now live inside `outputs/01_segmentation`. The aggregate "
            "raw/primary/strict-speech/guarded-nonspeech interval table retains unpadded "
            "sample-index boundaries; the 30-ms artifact layer is visualization only. "
            "A separate boundary-audit CSV/PNG quantifies display binning and local edge evidence. "
            "Unusual ALS speech is not invalid merely because it is fragmented by VAD."
        ),
        code(
            r"""RUN_SEGMENTATION = False
if RUN_SEGMENTATION:
    run_cli("segment")
else:
    print("Using existing segmentation outputs.")
"""
        ),
        code(
            r"""segments = read_stage("01_segmentation/bamboo_segmentation_intervals")
errors = read_stage("01_segmentation/segmentation_errors")

required = {"file_name", "profile", "view", "start_sec", "end_sec", "duration_sec"}
missing = required - set(segments.columns)
assert not missing, f"Segmentation table is missing columns: {sorted(missing)}"
assert (segments["end_sec"] >= segments["start_sec"]).all(), "Negative interval detected"
assert np.allclose(
    segments["duration_sec"],
    segments["end_sec"] - segments["start_sec"],
    rtol=0,
    atol=1e-6,
), "Saved duration does not equal end-start"

summary = (
    segments.groupby(["profile", "view"], as_index=False)
    .agg(
        recordings=("file_name", "nunique"),
        intervals=("duration_sec", "size"),
        total_duration_sec=("duration_sec", "sum"),
        median_interval_sec=("duration_sec", "median"),
        q05_interval_sec=("duration_sec", lambda x: x.quantile(.05)),
        q95_interval_sec=("duration_sec", lambda x: x.quantile(.95)),
    )
)
save_table(summary, "01_segmentation", "interval_summary_by_profile_and_view")
display(summary)
display(errors.head(50))
"""
        ),
        code(
            r"""# One row per recording/profile: denominators, interval counts, and speech fractions.
recording_duration = (
    segments.groupby(["file_name", "profile"], as_index=False)["end_sec"]
    .max().rename(columns={"end_sec": "recording_duration_sec"})
)
duration_wide = (
    segments.pivot_table(
        index=["file_name", "profile"],
        columns="view",
        values="duration_sec",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
)
count_wide = (
    segments.pivot_table(
        index=["file_name", "profile"],
        columns="view",
        values="duration_sec",
        aggfunc="size",
        fill_value=0,
    ).add_prefix("n_intervals__").reset_index()
)
recording = recording_duration.merge(duration_wide, on=["file_name", "profile"]).merge(
    count_wide, on=["file_name", "profile"]
)
for view in ["raw_speech", "primary_speech", "strict_speech", "strict_internal_nonspeech"]:
    if view not in recording:
        recording[view] = 0.0
    recording[f"fraction__{view}"] = recording[view] / recording["recording_duration_sec"]
save_table(recording, "01_segmentation", "recording_profile_support")
display(recording.describe(include="all").T)
"""
        ),
        code(
            r"""LEGACY = OUTPUT / "01_segmentation" / "segmentation" / "silero"
LEGACY_FIGURES = OUTPUT / "01_segmentation" / "figures" / "segmentation" / "silero"
legacy_summary = pd.read_csv(LEGACY / "summary" / "silero_all_summary.csv")

artifact_audit = pd.DataFrame([{
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
artifact_audit["one_segment_file_per_summary_row"] = (
    artifact_audit["segment_csv_files"] == artifact_audit["summary_rows"]
)
artifact_audit["one_frame_file_per_summary_row"] = (
    artifact_audit["frame_csv_files"] == artifact_audit["summary_rows"]
)
artifact_audit["one_figure_per_summary_row"] = (
    artifact_audit["total_png_files"] == artifact_audit["summary_rows"]
)
artifact_audit["one_boundary_audit_csv_per_summary_row"] = (
    artifact_audit["boundary_audit_csv_files"] == artifact_audit["summary_rows"]
)
artifact_audit["one_boundary_audit_png_per_summary_row"] = (
    artifact_audit["boundary_audit_png_files"] == artifact_audit["summary_rows"]
)
save_table(artifact_audit, "01_segmentation", "legacy_artifact_completeness")
display(artifact_audit)
assert artifact_audit[[
    "one_segment_file_per_summary_row",
    "one_frame_file_per_summary_row",
    "one_figure_per_summary_row",
    "one_boundary_audit_csv_per_summary_row",
    "one_boundary_audit_png_per_summary_row",
]].all(axis=None), "Silero per-recording artifact contract is incomplete"
"""
        ),
        code(
            r"""# Inspect the exact original-format per-recording CSVs and PNG.
EXAMPLE_FILE = None  # replace with an exact filename, or leave None for the first row
candidate = EXAMPLE_FILE or legacy_summary["file_name"].dropna().iloc[0]
candidate_row = legacy_summary.loc[legacy_summary["file_name"].eq(candidate)].iloc[0]
candidate_stem = Path(candidate).stem
candidate_frames = pd.read_csv(LEGACY / "frames" / f"{candidate_stem}_frames.csv")
candidate_segments = pd.read_csv(LEGACY / "segments" / f"{candidate_stem}_segments.csv")

expected_frame_columns = [
    "frame_idx", "mid_sec", "rms", "rms_db", "speech_vad_raw",
    "speech_vad_smooth", "speech_mask_strict", "nonspeech_mask_strict",
    "threshold", "frame_ms",
]
expected_segment_columns = [
    "segment_type", "start_sec", "end_sec", "duration_sec",
    "run_start_frame", "run_end_frame", "segment_role",
]
assert candidate_frames.columns.tolist() == expected_frame_columns
assert candidate_segments.columns.tolist() == expected_segment_columns
assert set(candidate_segments["segment_role"]).issubset({
    "speech", "leading_nonspeech", "internal_nonspeech", "trailing_nonspeech"
})

display(Markdown(f"### Original-format artifact audit: `{candidate}`"))
display(candidate_segments)
display(candidate_frames.head(20))
display(Image(filename=str(candidate_row["plot_path"]), width=1100))
"""
        ),
        code(
            r"""# Exact-boundary audit: review evidence only; no energy-based auto-snapping.
resolved_parameters = json.loads(
    (OUTPUT / "01_segmentation" / "logs" / "silero_segmentation_config.json")
    .read_text(encoding="utf-8")
)
display(pd.DataFrame([resolved_parameters]).T.rename(columns={0: "resolved_value"}))

boundary_summary = read_stage("01_segmentation/bamboo_segmentation_summary")[[
    "file_name", "qc_status", "boundary_edges", "boundary_low_contrast_edges",
    "boundary_low_contrast_fraction", "boundary_min_contrast_db",
    "boundary_audit_path", "boundary_plot_path",
]]
save_table(boundary_summary, "01_segmentation", "boundary_alignment_summary")
display(boundary_summary.sort_values(
    ["boundary_low_contrast_fraction", "boundary_min_contrast_db"],
    ascending=[False, True],
).head(40))

BOUNDARY_EXAMPLE_FILE = None
if BOUNDARY_EXAMPLE_FILE:
    boundary_row = boundary_summary.loc[
        boundary_summary["file_name"].eq(BOUNDARY_EXAMPLE_FILE)
    ].iloc[0]
else:
    boundary_row = boundary_summary.sort_values(
        ["boundary_low_contrast_fraction", "boundary_min_contrast_db"],
        ascending=[False, True],
    ).iloc[0]
display(pd.read_csv(boundary_row["boundary_audit_path"]))
display(Image(filename=str(boundary_row["boundary_plot_path"]), width=1100))
"""
        ),
        code(
            r"""# Preserve the original pipeline's accepted/flagged/excluded visual audit.
qc_summary = read_stage("01_segmentation/bamboo_segmentation_summary")
qc_counts = (
    qc_summary["qc_status"].value_counts(dropna=False)
    .rename_axis("qc_status").reset_index(name="logical_recordings")
)
qc_counts["percent"] = 100 * qc_counts["logical_recordings"] / max(1, len(qc_summary))
save_table(qc_summary, "01_segmentation", "recording_level_silero_qc")
save_table(qc_counts, "01_segmentation", "accepted_flagged_excluded_counts")
display(qc_counts)

for status in ["accepted", "flagged", "excluded"]:
    subset = qc_summary.loc[qc_summary["qc_status"].eq(status)]
    if subset.empty:
        print(f"No {status} recording exists.")
        continue
    example_path = Path(str(subset.iloc[0]["plot_path"]))
    display(Markdown(f"**{status.upper()} example:** `{subset.iloc[0]['file_name']}`"))
    if example_path.exists():
        display(Image(filename=str(example_path), width=1000))
    else:
        print("Diagnostic path is missing:", example_path)
"""
        ),
        md(
            "## Required segmentation decision\n\n"
            "Non-outlying accepted recordings default to `KEEP + AUTO`. Every flagged/excluded "
            "recording and every accepted segmentation-only outlier requires review. The widget "
            "shows all recordings in a scrollable/searchable browser and supports audio playback, "
            "the original four-panel plot, one-click `KEEP + AUTO`, "
            "`KEEP + MANUAL`, and `EXCLUDE + NONE`. Do not exclude unusual ALS speech merely "
            "because VAD fragmented it, and do not edit boundaries to remove acoustic noise. "
            "`Task Completed as Instructed = NO` is a locked automatic exclusion."
        ),
        code(
            r"""run_cli("segment-template")
import yaml

project_cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
adjudication_path = ROOT / project_cfg.get("data_freeze", {}).get(
    "segmentation_adjudication", "config/segmentation_adjudication.csv"
)
manual_override_path = ROOT / project_cfg.get("data_freeze", {}).get(
    "manual_segmentation_overrides", "config/manual_segmentation_overrides.csv"
)
review = pd.read_csv(adjudication_path, keep_default_na=False)
from paper1_qc.segmentation import segmentation_pending_reviews

pending_review = segmentation_pending_reviews(review)
save_table(pending_review, "01_segmentation", "pending_segmentation_decisions")
display(pending_review[[
    "file_name", "automatic_qc_status", "task_completed_as_instructed",
    "automatic_task_exclusion", "accepted_outlier", "review_reasons",
    "decision", "boundary_source", "reviewer", "review_date", "notes"
]])

selection_summary = (
    review.groupby(
        [
            "automatic_qc_status", "automatic_task_exclusion",
            "accepted_outlier", "review_required",
        ],
        dropna=False,
    ).size().rename("logical_recordings").reset_index()
)
save_table(selection_summary, "01_segmentation", "review_selection_summary")
display(selection_summary)
"""
        ),
        code(
            r"""from paper1_qc.config import load_config, resolve_executable
from paper1_qc.segmentation_review import launch_segmentation_review_widget

cfg = load_config(CONFIG)
DEFAULT_REVIEWER = ""  # enter your name once
review_widget = launch_segmentation_review_widget(
    summary=qc_summary,
    automatic_intervals=segments,
    review_path=adjudication_path,
    overrides_path=manual_override_path,
    default_reviewer=DEFAULT_REVIEWER,
    ffmpeg=resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg"),
    ffprobe=resolve_executable(cfg["software"]["ffprobe"], "ffprobe"),
)
display(review_widget)
"""
        ),
        code(
            r"""# Run after the interactive review; the widget writes decisions to disk.
review = pd.read_csv(adjudication_path, keep_default_na=False)
pending_review = segmentation_pending_reviews(review)
display(pending_review[[
    "file_name", "automatic_qc_status", "task_completed_as_instructed",
    "review_reasons", "decision", "boundary_source", "reviewer",
    "review_date", "notes"
]])

RUN_SEGMENTATION_ADJUDICATION = False
if RUN_SEGMENTATION_ADJUDICATION:
    assert pending_review.empty, "Complete every required review before freezing."
    run_cli("segment-adjudicate")
else:
    print("Set RUN_SEGMENTATION_ADJUDICATION=True only after pending_review is empty.")
"""
        ),
        code(
            r"""segmentation_freeze_version = project_cfg.get("segmentation_freeze", {}).get(
    "version",
    project_cfg.get("data_freeze", {}).get("version", "v1"),
)
SEGMENTATION_FREEZE = (
    MAIN_OUTPUTS / "01_SEGMENTATION_FREEZE" / str(segmentation_freeze_version)
)
frozen_decision_path = SEGMENTATION_FREEZE / "frozen_segmentation_decisions.csv"
frozen_interval_path = SEGMENTATION_FREEZE / "frozen_segmentation_intervals.csv"
reviewed_summary_path = (
    OUTPUT / "01_segmentation_after_review" / "segmentation" / "silero"
    / "summary" / "silero_after_review_summary.csv"
)
decision_exists = frozen_decision_path.exists()
interval_exists = frozen_interval_path.exists()
reviewed_exists = reviewed_summary_path.exists()
segmentation_reasons = []
if not pending_review.empty:
    segmentation_reasons.append(
        f"{len(pending_review)} required/incomplete segmentation reviews remain."
    )
if not decision_exists:
    segmentation_reasons.append("Frozen segmentation decision table does not exist.")
if not interval_exists:
    segmentation_reasons.append("Frozen segmentation interval table does not exist.")
if not reviewed_exists:
    segmentation_reasons.append("Post-review segmentation summary does not exist.")
segmentation_ready = stage_gate(
    "Silero segmentation",
    not segmentation_reasons,
    segmentation_reasons,
    "Open 02_goal1_occurrence_and_acquisition_variability.ipynb only after this gate passes.",
)"""
        ),
    ],
)


write(
    "02_goal1_occurrence_and_acquisition_variability.ipynb",
    "02 — Goal 1: occurrence and acquisition variability",
    "Audits eligibility, metric support, sparse-event prevalence, distributions, and exploratory acquisition/cohort contrasts.",
    [
        code(
            r"""RUN_GOAL_1 = False
if RUN_GOAL_1:
    run_cli("extract", "--profile", "primary")
    run_cli("assemble")
    run_cli("describe")
"""
        ),
        code(
            r"""from paper1_qc.registry import metric_registry_frame

data = read_stage("03_dataset_assembly/paper1_analysis_dataset")
flow = read_stage("03_dataset_assembly/eligibility_flow_counts")
descriptive = read_stage("04_analysis/descriptive/metric_descriptive_statistics")
contrasts = read_stage("04_analysis/descriptive/exploratory_participant_level_diagnosis_contrasts")
registry = metric_registry_frame()

display(flow)
save_table(flow, "02_goal1", "eligibility_flow_counts")

eligible = data.loc[data["primary_measurement_eligible"].fillna(False)].copy()
cohort = (
    eligible.groupby("diagnosis_analysis", dropna=False)
    .agg(recordings=("logical_recording_id", "nunique"), participants=("SubjectID", "nunique"))
    .reset_index()
)
cohort["recordings_per_participant"] = cohort["recordings"] / cohort["participants"]
save_table(cohort, "02_goal1", "cohort_imbalance")
display(cohort)
"""
        ),
        code(
            r"""# Metric support is an outcome, not a nuisance to hide.
support = descriptive.merge(
    registry[["feature", "family", "unit", "role", "worse", "minimum_support"]],
    on=["feature", "family", "role"],
    how="left",
    validate="one_to_one",
)
support["supported_fraction"] = support["recordings_nonmissing"] / len(eligible)
support = support.sort_values(["family", "supported_fraction", "feature"])
save_table(support, "02_goal1", "metric_support_and_descriptives")
display(support)

fig, ax = plt.subplots(figsize=(11, max(6, 0.27 * len(support))))
sns.scatterplot(
    data=support,
    x="supported_fraction",
    y="feature",
    hue="family",
    style="role",
    s=70,
    ax=ax,
)
ax.axvline(.8, color="0.4", linestyle="--", linewidth=1)
ax.set(title="Metric-specific support denominators", xlabel="Fraction of eligible recordings", ylabel="")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
save_figure(fig, "02_goal1", "metric_support_fraction")
plt.show()
"""
        ),
        code(
            r"""# Family-faceted raw-unit distributions. No cross-unit composite is plotted.
metric_columns = [feature for feature in registry["feature"] if feature in eligible]
long = eligible[["file_name", "SubjectID", "diagnosis_analysis", *metric_columns]].melt(
    id_vars=["file_name", "SubjectID", "diagnosis_analysis"],
    var_name="feature",
    value_name="value",
)
long = long.merge(registry[["feature", "family", "unit", "role"]], on="feature", how="left")

for family, family_frame in long.groupby("family", sort=True):
    selected = family_frame["feature"].drop_duplicates().tolist()
    ncols = 2
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.3 * nrows), squeeze=False)
    for ax, feature in zip(axes.flat, selected):
        plot_data = family_frame.loc[family_frame["feature"].eq(feature)]
        sns.histplot(
            data=plot_data,
            x="value",
            hue="diagnosis_analysis",
            element="step",
            stat="density",
            common_norm=False,
            bins=25,
            ax=ax,
        )
        unit = registry.set_index("feature").loc[feature, "unit"]
        ax.set(title=feature, xlabel=unit, ylabel="Density")
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.suptitle(f"Raw metric distributions — {family}", y=1.01)
    fig.tight_layout()
    save_figure(fig, "02_goal1", f"raw_distributions__{family}")
    plt.show()
"""
        ),
        code(
            r"""# Sparse-event table: absence is zero only when the metric extraction status had support.
sparse = support.loc[support["zero_fraction_nonmissing"].ge(.5)].copy()
sparse["recording_event_prevalence"] = 1 - sparse["zero_fraction_nonmissing"]
sparse["interpretation_gate"] = np.where(
    sparse["supported_fraction"].ge(.8),
    "report prevalence and positive magnitude",
    "support-limited; report missingness first",
)
save_table(sparse, "02_goal1", "sparse_event_prevalence")
display(sparse[[
    "feature", "family", "recordings_nonmissing", "supported_fraction",
    "recording_event_prevalence", "interpretation_gate"
]])

fig, ax = plt.subplots(figsize=(10, max(4, .35 * len(sparse))))
sns.barplot(data=sparse, x="recording_event_prevalence", y="feature", hue="family", ax=ax)
ax.set(title="Observed event prevalence among supported recordings", xlabel="Prevalence", ylabel="")
save_figure(fig, "02_goal1", "sparse_event_prevalence")
plt.show()
"""
        ),
        code(
            r"""# Exploratory participant-level ALS-control effects; counts and CI status stay attached.
estimable = contrasts.loc[contrasts["status"].eq("ok")].copy()
save_table(contrasts, "02_goal1", "participant_level_diagnosis_contrasts")
fig, ax = plt.subplots(figsize=(11, max(6, .3 * len(estimable))))
ax.errorbar(
    estimable["cliffs_delta_a_vs_b"],
    np.arange(len(estimable)),
    xerr=np.vstack([
        estimable["cliffs_delta_a_vs_b"] - estimable["cliffs_delta_ci_low"],
        estimable["cliffs_delta_ci_high"] - estimable["cliffs_delta_a_vs_b"],
    ]),
    fmt="o",
    color="0.2",
    ecolor="0.55",
    capsize=2,
)
ax.axvline(0, color="0.3", linestyle="--")
ax.set_yticks(np.arange(len(estimable)), estimable["feature"])
ax.set(title="Exploratory participant-level ALS vs control contrasts", xlabel="Cliff's delta", ylabel="")
save_figure(fig, "02_goal1", "diagnosis_cliffs_delta_forest")
plt.show()
"""
        ),
        code(
            r"""feature_errors = read_optional_stage("02_features/feature_extraction_errors")
goal1_reasons = []
if eligible.empty:
    goal1_reasons.append("No recordings satisfy primary measurement eligibility.")
if support.empty:
    goal1_reasons.append("Metric support table is empty.")
if not feature_errors.empty:
    goal1_reasons.append(
        f"{len(feature_errors)} feature-extraction errors require correction or documented exclusion."
    )
goal1_ready = stage_gate(
    "Goal 1 occurrence/acquisition variability",
    not goal1_reasons,
    goal1_reasons,
    "Open 03_goal2_participant_persistence.ipynb only after this gate passes.",
)"""
        ),
    ],
)


write(
    "03_goal2_participant_persistence.ipynb",
    "03 — Goal 2: participant persistence",
    "Quantifies repeated-recording variance structure while explicitly avoiding a test–retest reliability claim.",
    [
        code(
            r"""RUN_GOAL_2 = False
if RUN_GOAL_2:
    run_cli("describe")

from paper1_qc.registry import metric_registry_frame
data = read_stage("03_dataset_assembly/paper1_analysis_dataset")
data = data.loc[data["primary_measurement_eligible"].fillna(False)].copy()
persistence = read_stage("04_analysis/descriptive/participant_persistence_not_reliability")
registry = metric_registry_frame()
persistence = persistence.merge(
    registry[["feature", "family", "unit", "role"]],
    on="feature", how="left", validate="one_to_one"
)
"""
        ),
        code(
            r"""repeat_counts = (
    data.groupby("SubjectID").size().rename("recordings").reset_index()
)
repeat_summary = repeat_counts["recordings"].value_counts().sort_index().rename_axis(
    "recordings_per_participant"
).rename("participants").reset_index()
repeat_summary["participant_fraction"] = repeat_summary["participants"] / len(repeat_counts)
save_table(repeat_summary, "03_goal2", "repeated_recording_design")
display(repeat_summary)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=repeat_summary, x="recordings_per_participant", y="participants", ax=ax)
ax.set(title="Repeated-recording support", xlabel="Eligible recordings per participant", ylabel="Participants")
save_figure(fig, "03_goal2", "recordings_per_participant")
plt.show()
"""
        ),
        code(
            r"""status_summary = (
    persistence.groupby(["family", "status"], dropna=False)
    .size().rename("metrics").reset_index()
)
save_table(status_summary, "03_goal2", "persistence_model_status")
display(status_summary)

estimable = persistence.loc[persistence["status"].isin(["ok", "not_converged"])].copy()
estimable = estimable.sort_values(["family", "persistence_icc"])
fig, ax = plt.subplots(figsize=(11, max(5, .35 * len(estimable))))
sns.scatterplot(
    data=estimable,
    x="persistence_icc",
    y="feature",
    hue="family",
    s=70,
    ax=ax,
)
ax.axvline(.5, color="0.4", linestyle="--", linewidth=1)
ax.set(
    title="Participant rank persistence (not test–retest reliability)",
    xlabel="Between-participant variance / total variance",
    ylabel="",
    xlim=(-.05, 1.05),
)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
save_figure(fig, "03_goal2", "participant_rank_persistence")
plt.show()
"""
        ),
        code(
            r"""# Auditable spaghetti plots for selected high-, middle-, and low-persistence metrics.
if len(estimable):
    ordered = estimable.sort_values("persistence_icc")
    selected_features = list(dict.fromkeys([
        ordered.iloc[0]["feature"],
        ordered.iloc[len(ordered)//2]["feature"],
        ordered.iloc[-1]["feature"],
    ]))
else:
    selected_features = []

date_column = "Recording date"
for feature in selected_features:
    work = data[["SubjectID", date_column, feature]].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna().sort_values(["SubjectID", date_column])
    work["occasion"] = work.groupby("SubjectID").cumcount() + 1
    repeated_ids = work.groupby("SubjectID").size().loc[lambda x: x >= 2].index
    work = work.loc[work["SubjectID"].isin(repeated_ids)]
    # Deterministic display subset only; the model used all eligible repeated participants.
    display_ids = sorted(repeated_ids.astype(str))[:40]
    plot_data = work.loc[work["SubjectID"].astype(str).isin(display_ids)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, subject in plot_data.groupby("SubjectID"):
        ax.plot(subject["occasion"], subject[feature], color="0.35", alpha=.35, linewidth=.8)
    ax.set(title=f"Repeated-recording trajectories: {feature}", xlabel="Observed occasion", ylabel=feature)
    save_figure(fig, "03_goal2", f"spaghetti__{feature}")
    plt.show()
"""
        ),
        code(
            r"""# Between- vs within-participant variance table preserves the actual estimand.
variance_table = persistence[[
    "feature", "family", "n_recordings", "n_participants", "n_repeated_participants",
    "between_participant_variance", "within_participant_variance", "persistence_icc",
    "zero_fraction", "status",
]].copy()
save_table(variance_table, "03_goal2", "participant_persistence_full")
display(variance_table)
"""
        ),
        code(
            r"""goal2_reasons = []
if repeat_counts["recordings"].ge(2).sum() == 0:
    goal2_reasons.append("No participant has at least two eligible recordings.")
if estimable.empty:
    goal2_reasons.append("No persistence metric is estimable.")
goal2_ready = stage_gate(
    "Goal 2 participant persistence",
    not goal2_reasons,
    goal2_reasons,
    "Open 04_goal3_multidimensional_structure_and_robustness.ipynb after this gate passes.",
)"""
        ),
    ],
)


write(
    "04_goal3_multidimensional_structure_and_robustness.ipynb",
    "04 — Goal 3: multidimensional structure and reference robustness",
    "Evaluates family organization, pairwise support, segmentation/encoding sensitivity, and the exact-session Rest reference.",
    [
        code(
            r"""RUN_GOAL_3_SENSITIVITY = False
if RUN_GOAL_3_SENSITIVITY:
    run_cli("extract", "--profile", "conservative")
    run_cli("extract", "--profile", "permissive")
    run_cli("sensitivity")
    run_cli("encoding-sensitivity")
    run_cli("rest-reference")

from paper1_qc.registry import metric_registry_frame
registry = metric_registry_frame()
pairwise = read_stage("04_analysis/descriptive/pairwise_clustered_spearman")
"""
        ),
        code(
            r"""features = registry.loc[
    registry["feature"].isin(set(pairwise["feature_left"]) | set(pairwise["feature_right"])),
    "feature",
].tolist()
rho = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
n_participants = pd.DataFrame(np.nan, index=features, columns=features)
for row in pairwise.itertuples():
    rho.loc[row.feature_left, row.feature_right] = row.rho
    rho.loc[row.feature_right, row.feature_left] = row.rho
    n_participants.loc[row.feature_left, row.feature_right] = row.n_participants
    n_participants.loc[row.feature_right, row.feature_left] = row.n_participants

family_order = (
    registry.set_index("feature").loc[features, "family"].sort_values(kind="stable").index.tolist()
)
rho = rho.loc[family_order, family_order]
n_participants = n_participants.loc[family_order, family_order]
save_table(rho.reset_index(names="feature"), "04_goal3", "spearman_matrix")
save_table(n_participants.reset_index(names="feature"), "04_goal3", "pairwise_participant_support")

fig, axes = plt.subplots(1, 2, figsize=(19, 8))
sns.heatmap(rho, vmin=-1, vmax=1, center=0, cmap="vlag", ax=axes[0], square=True)
axes[0].set_title("Pairwise Spearman structure")
sns.heatmap(n_participants, cmap="viridis", ax=axes[1], square=True)
axes[1].set_title("Pair-specific participant denominator")
for ax in axes:
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
fig.tight_layout()
save_figure(fig, "04_goal3", "correlation_and_support_matrices")
plt.show()
"""
        ),
        code(
            r"""# Family coherence statistic with a fixed-label permutation null.
family_lookup = registry.set_index("feature")["family"].to_dict()
observed_pairs = pairwise.loc[pairwise["rho"].notna()].copy()
observed_pairs["same_family"] = (
    observed_pairs["feature_left"].map(family_lookup)
    == observed_pairs["feature_right"].map(family_lookup)
)
observed_stat = (
    observed_pairs.loc[observed_pairs["same_family"], "rho"].abs().mean()
    - observed_pairs.loc[~observed_pairs["same_family"], "rho"].abs().mean()
)

rng = np.random.default_rng(20260713)
feature_labels = pd.Series(family_lookup)
null = []
for _ in range(5000):
    shuffled = pd.Series(rng.permutation(feature_labels.values), index=feature_labels.index)
    same = (
        observed_pairs["feature_left"].map(shuffled)
        == observed_pairs["feature_right"].map(shuffled)
    )
    if same.any() and (~same).any():
        null.append(
            observed_pairs.loc[same, "rho"].abs().mean()
            - observed_pairs.loc[~same, "rho"].abs().mean()
        )
null = np.asarray(null)
p_value = (1 + np.sum(null >= observed_stat)) / (1 + len(null))
coherence = pd.DataFrame([{
    "observed_within_minus_between_abs_rho": observed_stat,
    "permutations": len(null),
    "one_sided_permutation_p": p_value,
    "seed": 20260713,
}])
save_table(coherence, "04_goal3", "family_coherence_permutation")
display(coherence)

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(null, bins=50, ax=ax, color="0.55")
ax.axvline(observed_stat, color="#C44E52", linewidth=2, label="Observed")
ax.set(title="Permutation null for family coherence", xlabel="Within-family |rho| − between-family |rho|")
ax.legend(frameon=False)
save_figure(fig, "04_goal3", "family_coherence_permutation")
plt.show()
"""
        ),
        code(
            r"""# Segmentation and native-encoding robustness are sensitivities, not new observations.
segmentation_robustness = read_stage("04_analysis/sensitivity/segmentation_profile_robustness")
save_table(segmentation_robustness, "04_goal3", "segmentation_profile_robustness")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=segmentation_robustness, x="profile", y="spearman_rho", ax=ax)
sns.stripplot(data=segmentation_robustness, x="profile", y="spearman_rho", color="0.25", alpha=.6, ax=ax)
ax.set(title="Q-metric rank robustness to segmentation profiles", ylabel="Paired Spearman rho")
save_figure(fig, "04_goal3", "segmentation_profile_metric_robustness")
plt.show()

encoding = read_stage("04_analysis/encoding_sensitivity/paired_encoding_robustness")
save_table(encoding, "04_goal3", "paired_wav_webm_robustness")
display(encoding.sort_values("spearman_rho").head(20))
"""
        ),
        code(
            r"""# Rest is a matched acquisition-context sensitivity, never a speech-VAD input.
rest = read_stage("04_analysis/rest_reference/exact_session_bamboo_rest_comparison")
rest_summary = pd.DataFrame([{
    "exact_session_pairs": rest["logical_recording_id_bamboo"].nunique(),
    "participants": rest["SubjectID"].nunique(),
    "complete_level_pairs": rest[[
        "qadd_nonspeech_level_dbfs", "restref_level_dbfs"
    ]].dropna().shape[0],
}])
save_table(rest_summary, "04_goal3", "rest_exact_session_support")
display(rest_summary)

complete = rest[["SubjectID", "qadd_nonspeech_level_dbfs", "restref_level_dbfs"]].dropna().copy()
if len(complete):
    rho_rest = complete[["qadd_nonspeech_level_dbfs", "restref_level_dbfs"]].corr(method="spearman").iloc[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=complete, x="restref_level_dbfs", y="qadd_nonspeech_level_dbfs", ax=axes[0])
    axes[0].set(title=f"Exact-session Rest vs Bamboo pause level (rho={rho_rest:.2f})")
    mean_level = complete[["qadd_nonspeech_level_dbfs", "restref_level_dbfs"]].mean(axis=1)
    difference = complete["qadd_nonspeech_level_dbfs"] - complete["restref_level_dbfs"]
    axes[1].scatter(mean_level, difference, alpha=.7)
    axes[1].axhline(difference.mean(), color="0.2")
    axes[1].axhline(difference.mean() + 1.96*difference.std(ddof=1), color="0.5", linestyle="--")
    axes[1].axhline(difference.mean() - 1.96*difference.std(ddof=1), color="0.5", linestyle="--")
    axes[1].set(title="Contextual agreement view", xlabel="Mean level (dBFS)", ylabel="Bamboo pause − Rest (dB)")
    fig.tight_layout()
    save_figure(fig, "04_goal3", "rest_reference_level_comparison")
    plt.show()
"""
        ),
        code(
            r"""goal3_reasons = []
if pairwise.empty:
    goal3_reasons.append("Pairwise metric-structure table is empty.")
if segmentation_robustness.empty:
    goal3_reasons.append("Segmentation-profile sensitivity is missing.")
if encoding.empty:
    goal3_reasons.append("Paired WAV/WEBM sensitivity is missing.")
if rest_summary["exact_session_pairs"].iloc[0] == 0:
    goal3_reasons.append("No exact-session Bamboo–Rest pair is available.")
goal3_ready = stage_gate(
    "Goal 3 structure and robustness",
    not goal3_reasons,
    goal3_reasons,
    "Open 05_goal4_perceptual_family_alignment.ipynb after this gate passes.",
)"""
        ),
    ],
)


write(
    "05_goal4_perceptual_family_alignment.ipynb",
    "05 — Goal 4: perceptual family alignment",
    "Separates the distributed four-RA main annotations from the crossed 70-file reliability subset, evaluates family alignment, and compares both with merged broad 2RA labels.",
    [
        md(
            "Primary alignment excludes competing speech and non-task content because the "
            "estimand is family perceptual alignment, not source recognition. The broad "
            "metadata direction gate must be confirmed from the RA codebook before the "
            "2RA comparison runs. The main annotation set has one independent RA per "
            "recording; agreement and four-RA consensus are estimated only in "
            "`Reliability/<RA name>/`, where the same files were rated by all four RAs."
        ),
        code(
            r"""RUN_GOAL_4 = False
if RUN_GOAL_4:
    run_cli("human-qc", "--schema", "config/human_qc_schema.yaml")
else:
    print("Using existing Goal 4 outputs.")
"""
        ),
        code(
            r"""main_coverage = read_stage("04_analysis/human_qc/main_distributed_item_coverage")
main_design = read_stage("04_analysis/human_qc/main_distributed_design_summary")
main_ratings = read_stage("04_analysis/human_qc/main_distributed_ratings_long")
main_workload = read_stage("04_analysis/human_qc/main_rater_workload_and_prevalence")
reliability_status = read_stage("04_analysis/human_qc/reliability_analysis_status")
reliability_coverage = read_optional_stage("04_analysis/human_qc/reliability_item_coverage")
reliability_ratings = read_optional_stage("04_analysis/human_qc/reliability_ratings_long")
agreement = read_optional_stage("04_analysis/human_qc/reliability_interrater_agreement_complete")
consensus = read_optional_stage("04_analysis/human_qc/reliability_four_ra_consensus_primary")
direction_audit = read_stage("04_analysis/human_qc/two_ra_broad_direction_and_scale_audit")

display(direction_audit)
save_table(direction_audit, "05_goal4", "direction_and_scale_audit")
save_table(main_design, "05_goal4", "main_distributed_design_summary")
save_table(main_workload, "05_goal4", "main_rater_workload_and_prevalence")
save_table(reliability_status, "05_goal4", "reliability_analysis_status")
display(main_design)
display(main_workload)
display(reliability_status)
"""
        ),
        code(
            r"""# Main-set coverage: every recording-family should have exactly one RA.
main_matrix = (
    main_ratings.assign(rated=1)
    .pivot_table(index=["file_name", "category"], columns="rater_id", values="rated", aggfunc="max", fill_value=0)
)
save_table(main_matrix.reset_index(), "05_goal4", "main_distributed_coverage_matrix")
fig, ax = plt.subplots(figsize=(10, min(18, max(5, .08 * len(main_matrix)))))
sns.heatmap(main_matrix, cmap=["#f2f2f2", "#4C78A8"], cbar=False, ax=ax)
ax.set(title="Main distributed coverage: one RA per item", xlabel="Rater", ylabel="Recording × perceptual family")
ax.tick_params(axis="y", labelleft=False)
save_figure(fig, "05_goal4", "main_distributed_rating_coverage")
plt.show()

# Reliability coverage: every recording-family should have all four RAs.
if not reliability_ratings.empty:
    reliability_matrix = (
        reliability_ratings.assign(rated=1)
        .pivot_table(index=["file_name", "category"], columns="rater_id", values="rated", aggfunc="max", fill_value=0)
    )
    save_table(reliability_matrix.reset_index(), "05_goal4", "reliability_four_ra_coverage_matrix")
    fig, ax = plt.subplots(figsize=(10, min(18, max(5, .08 * len(reliability_matrix)))))
    sns.heatmap(reliability_matrix, cmap=["#f2f2f2", "#59A14F"], cbar=False, ax=ax)
    ax.set(title="Crossed reliability coverage: four RAs per item", xlabel="Rater", ylabel="Recording × perceptual family")
    ax.tick_params(axis="y", labelleft=False)
    save_figure(fig, "05_goal4", "reliability_four_ra_rating_coverage")
    plt.show()
"""
        ),
        code(
            r"""# Reliability prevalence is shown beside agreement because imbalance can depress kappa.
if not consensus.empty and not agreement.empty:
    prevalence = (
        consensus.groupby("category")["consensus_rating"]
    .agg(
        n_consensus="count",
        positive=lambda x: int(pd.to_numeric(x, errors="coerce").sum()),
        prevalence=lambda x: float(pd.to_numeric(x, errors="coerce").mean()),
    ).reset_index()
    )
    agreement_view = agreement.merge(prevalence, on="category", how="left")
    save_table(agreement_view, "05_goal4", "reliability_agreement_and_prevalence")
    display(agreement_view)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=prevalence, x="prevalence", y="category", ax=axes[0], color="#59A14F")
    axes[0].set(title="Reliability-set 4RA consensus prevalence", xlabel="Positive fraction", ylabel="")
    axes[1].errorbar(
        agreement_view["gwet_ac1_nominal"],
        np.arange(len(agreement_view)),
        xerr=np.vstack([
            agreement_view["gwet_ac1_nominal"] - agreement_view["gwet_ac1_ci_low"],
            agreement_view["gwet_ac1_ci_high"] - agreement_view["gwet_ac1_nominal"],
        ]),
        fmt="o", color="0.2", ecolor="0.55", capsize=3,
    )
    axes[1].set_yticks(np.arange(len(agreement_view)), agreement_view["category"])
    axes[1].set(title="Complete-item Gwet AC1, bootstrap 95% CI", xlabel="Agreement", xlim=(-.1, 1.05))
    fig.tight_layout()
    save_figure(fig, "05_goal4", "reliability_prevalence_and_agreement")
    plt.show()
else:
    print("Reliability agreement is not estimable yet; inspect reliability_analysis_status.csv.")
"""
        ),
        code(
            r"""# Primary broad-coverage estimand: weighted within-rater effects in the distributed set.
main_alignment = read_stage("04_analysis/human_qc/main_distributed_rater_stratified_family_alignment")
main_effect = main_alignment.pivot(index="human_family", columns="objective_family", values="effect")
main_n = main_alignment.pivot(index="human_family", columns="objective_family", values="n_recordings")
save_table(main_alignment, "05_goal4", "main_rater_stratified_family_alignment")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(main_effect, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", ax=axes[0])
axes[0].set(title="Main-set rater-stratified effect", xlabel="Objective Q family", ylabel="Perceptual family")
sns.heatmap(main_n, cmap="viridis", annot=True, fmt=".0f", ax=axes[1])
axes[1].set(title="Pair-specific recording denominator", xlabel="Objective Q family", ylabel="")
fig.tight_layout()
save_figure(fig, "05_goal4", "main_rater_stratified_alignment_and_denominators")
plt.show()

matched_summary = (
    main_alignment.loc[main_alignment["estimable"]]
    .groupby("matched_family")["effect"]
    .agg(["count", "mean", "median"]).reset_index()
)
save_table(matched_summary, "05_goal4", "main_matched_vs_mismatched_descriptive")
display(matched_summary)

# Higher-confidence sensitivity: crossed-set consensus, if class support permits.
reliability_alignment = read_optional_stage("04_analysis/human_qc/reliability_four_ra_consensus_family_alignment")
if not reliability_alignment.empty:
    reliability_effect = reliability_alignment.pivot(index="human_family", columns="objective_family", values="effect")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(reliability_effect, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", ax=ax)
    ax.set(title="Reliability-subset 4RA consensus alignment", xlabel="Objective Q family", ylabel="Perceptual family")
    save_figure(fig, "05_goal4", "reliability_consensus_alignment")
    plt.show()
"""
        ),
        code(
            r"""# The merged 2RA workflow is comparable only for explicit shared families.
comparison = read_stage("04_analysis/human_qc/main_distributed_vs_two_ra_paired_alignment")
reliability_comparison = read_optional_stage("04_analysis/human_qc/reliability_four_ra_consensus_vs_two_ra_paired_alignment")
save_table(comparison, "05_goal4", "main_distributed_vs_two_ra_paired_alignment")
display(comparison)
if not reliability_comparison.empty:
    save_table(reliability_comparison, "05_goal4", "reliability_consensus_vs_two_ra_paired_alignment")
    display(reliability_comparison)

if "delta_auc_a_minus_b" in comparison and comparison["delta_auc_a_minus_b"].notna().any():
    plot = comparison.loc[comparison["status"].eq("ok")].copy()
    fig, ax = plt.subplots(figsize=(9, max(4, .8 * len(plot))))
    ax.errorbar(
        plot["delta_auc_a_minus_b"],
        np.arange(len(plot)),
        xerr=np.vstack([
            plot["delta_auc_a_minus_b"] - plot["delta_ci_low"],
            plot["delta_ci_high"] - plot["delta_auc_a_minus_b"],
        ]),
        fmt="o", capsize=3, color="0.2", ecolor="0.55",
    )
    ax.axvline(0, color="0.35", linestyle="--")
    ax.set_yticks(np.arange(len(plot)), plot["family"])
    ax.set(
        title="Paired shared-recording comparison: main distributed vs 2RA",
        xlabel="ΔAUC: distributed detailed − merged 2RA broad",
        ylabel="",
    )
    save_figure(fig, "05_goal4", "main_distributed_minus_two_ra_delta_auc")
    plt.show()
"""
        ),
        code(
            r"""# Secondary duration/fraction analysis preserves the richer interval annotations.
extent = read_stage("04_analysis/human_qc/main_distributed_extent_labels_secondary")
context = read_stage("04_analysis/human_qc/main_context_annotations_not_family_alignment")
reliability_extent = read_optional_stage("04_analysis/human_qc/reliability_four_ra_extent_consensus_secondary")
save_table(extent, "05_goal4", "main_distributed_extent_labels_secondary")

extent_summary = (
    extent.groupby("category")["annotated_fraction"]
    .agg(n="count", median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
    .reset_index()
)
save_table(extent_summary, "05_goal4", "main_extent_summary")
display(extent_summary)

if not reliability_extent.empty:
    reliability_extent_summary = (
        reliability_extent.groupby("category")["consensus_annotated_fraction"]
        .agg(n="count", median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
        .reset_index()
    )
    save_table(reliability_extent_summary, "05_goal4", "reliability_extent_consensus_summary")
    display(reliability_extent_summary)

context_summary = (
    context.groupby("category")["rating"]
    .agg(rater_recordings="size", positive="sum", prevalence="mean")
    .reset_index()
)
context_summary["primary_family_alignment"] = False
save_table(context_summary, "05_goal4", "context_annotations_excluded_from_family_alignment")
display(context_summary)
"""
        ),
        code(
            r"""goal4_reasons = []
if direction_audit.empty or not direction_audit["direction"].eq("higher_is_worse").all():
    goal4_reasons.append(
        "The 2RA scale direction is not confirmed as higher-is-worse; verify the codebook."
    )
if main_alignment.empty:
    goal4_reasons.append("Distributed detailed-rating family alignment is missing.")
if agreement.empty:
    goal4_reasons.append(
        "Complete four-RA crossed reliability agreement is missing or not estimable."
    )
goal4_ready = stage_gate(
    "Goal 4 perceptual family alignment",
    not goal4_reasons,
    goal4_reasons,
    "Open 06_results_registry_and_manuscript_tables.ipynb only after this gate passes.",
)"""
        ),
    ],
)


write(
    "06_results_registry_and_manuscript_tables.ipynb",
    "06 — Results registry and manuscript table inventory",
    "Indexes every saved visualization/table with a hash so manuscript figures can be traced to an exact output.",
    [
        code(
            r"""files = sorted(path for path in VIZ_ROOT.rglob("*") if path.is_file())
rows = []
for path in files:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append({
        "relative_path": path.relative_to(ROOT).as_posix(),
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    })
registry = pd.DataFrame(rows)
save_table(registry, "06_registry", "visualization_output_registry")
display(registry)
"""
        ),
        code(
            r"""figure_candidates = pd.DataFrame([
    {"manuscript_role": "Segmentation supplement", "source": "outputs/01_segmentation/figures/segmentation/silero/{accepted,flagged,excluded}/<recording>_silero.png", "status": "candidate"},
    {"manuscript_role": "Goal 1 support panel", "source": "outputs/visualization/02_goal1/figures/metric_support_fraction.png", "status": "candidate"},
    {"manuscript_role": "Goal 1 acquisition distributions", "source": "outputs/visualization/02_goal1/figures/raw_distributions__<family>.png", "status": "family facets"},
    {"manuscript_role": "Goal 2 persistence", "source": "outputs/visualization/03_goal2/figures/participant_rank_persistence.png", "status": "candidate"},
    {"manuscript_role": "Goal 3 structure", "source": "outputs/visualization/04_goal3/figures/correlation_and_support_matrices.png", "status": "candidate"},
    {"manuscript_role": "Goal 3 Rest sensitivity", "source": "outputs/visualization/04_goal3/figures/rest_reference_level_comparison.png", "status": "supplement candidate"},
    {"manuscript_role": "Goal 4 distributed family validity", "source": "outputs/visualization/05_goal4/figures/main_rater_stratified_alignment_and_denominators.png", "status": "candidate"},
    {"manuscript_role": "Goal 4 reliability-subset agreement", "source": "outputs/visualization/05_goal4/figures/reliability_prevalence_and_agreement.png", "status": "candidate if estimable"},
    {"manuscript_role": "Goal 4 label-system comparison", "source": "outputs/visualization/05_goal4/figures/main_distributed_minus_two_ra_delta_auc.png", "status": "candidate if estimable"},
])
save_table(figure_candidates, "06_registry", "manuscript_figure_candidates")
display(figure_candidates)
"""
        ),
        code(
            r"""required_candidates = figure_candidates.loc[
    ~figure_candidates["source"].str.contains("<family>", regex=False)
]
missing_candidates = required_candidates.loc[
    ~required_candidates["source"].map(lambda value: (ROOT / value).exists())
]
registry_ready = stage_gate(
    "Results registry",
    not registry.empty and missing_candidates.empty,
    [
        f"Missing candidate output: {row.source}"
        for row in missing_candidates.itertuples()
    ],
    "Freeze the manuscript-facing shortlist only after every required output is present.",
)"""
        ),
    ],
)

print("Generated visualization notebooks under", ROOT / "visualization")
