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
from IPython.display import display

def find_project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "paper1_qc").exists():
            return candidate
    raise FileNotFoundError("Open Jupyter from inside paper1_pipeline_rebuilt.")

ROOT = find_project_root()
CONFIG = ROOT / "config" / "project.yaml"
OUTPUT = ROOT / "outputs"
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

def run_cli(*arguments):
    command = [sys.executable, "-m", "paper1_qc.cli", "--config", str(CONFIG), *arguments]
    print("RUN:", " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, check=True)

def save_table(frame, folder, name):
    target = VIZ_ROOT / folder
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.csv"
    frame.to_csv(path, index=False)
    print("TABLE:", path.relative_to(ROOT), f"({len(frame):,} rows)")
    return path

def save_figure(fig, folder, name):
    target = VIZ_ROOT / folder
    target.mkdir(parents=True, exist_ok=True)
    png = target / f"{name}.png"
    svg = target / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print("FIGURE:", png.relative_to(ROOT))
    return png, svg

assert CONFIG.exists(), "Copy config/project.example.yaml to config/project.yaml and review it."
print("Project:", ROOT)
print("Config:", CONFIG)
print("Visualization outputs:", VIZ_ROOT)
"""


def notebook(title: str, purpose: str, cells: list[dict]) -> dict:
    return {
        "cells": [
            md(
                f"# {title}\n\n{purpose}\n\n"
                "Every displayed denominator and paper-facing visual is also saved under "
                "`outputs/visualization/`. Empty or under-supported analyses remain visible "
                "as audit rows; they are never silently removed."
            ),
            code(BOOTSTRAP),
            *cells,
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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
            "updated data root. The audit is intentionally run before any signal processing."
        ),
        code(
            r"""RUN_PIPELINE_STAGES = False

if RUN_PIPELINE_STAGES:
    run_cli("audit")
    run_cli("inventory")
else:
    print("Dry review only. Set RUN_PIPELINE_STAGES=True to run audit and inventory.")
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
    metadata_issues.groupby(["severity", "issue"], dropna=False)
    .size().rename("n").reset_index().sort_values(["severity", "n"], ascending=[True, False])
)
save_table(issue_counts, "00_preflight", "metadata_issue_counts")
display(issue_counts)

media_summary = pd.DataFrame({
    "recordings_on_disk": [inventory["file_name"].nunique()],
    "physical_files": [len(inventory)],
    "extensions": [", ".join(sorted(inventory["extension"].dropna().astype(str).unique())) if "extension" in inventory else "not available"],
    "probe_failures": [int(inventory.get("probe_status", pd.Series(dtype=str)).astype(str).ne("ok").sum()) if "probe_status" in inventory else np.nan],
})
save_table(media_summary, "00_preflight", "media_inventory_summary")
display(media_summary)
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
    ],
)


write(
    "01_segmentation_visual_audit.ipynb",
    "01 — Segmentation visual audit",
    "Runs Silero segmentation, quantifies profile sensitivity, and overlays saved intervals on decoded waveforms.",
    [
        md(
            "This notebook does not decide that unusual ALS speech is invalid. It visualizes "
            "raw, primary, strict-speech, and guarded-nonspeech views and keeps flags separate "
            "from hard exclusions."
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
            r"""fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, view in zip(axes, ["raw_speech", "primary_speech", "strict_speech"]):
    sns.ecdfplot(
        data=recording,
        x=f"fraction__{view}",
        hue="profile",
        ax=ax,
    )
    ax.set(title=view.replace("_", " ").title(), xlabel="Fraction of recording", ylabel="ECDF")
fig.suptitle("Segmentation support across pre-specified profiles", y=1.04)
save_figure(fig, "01_segmentation", "speech_fraction_ecdf")
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
primary_counts = recording.loc[recording["profile"].eq("primary")].copy()
count_column = "n_intervals__primary_speech"
sns.histplot(primary_counts[count_column], bins=30, ax=ax)
ax.set(title="Primary speech fragmentation audit", xlabel="Number of primary speech intervals")
save_figure(fig, "01_segmentation", "primary_speech_fragmentation")
plt.show()
"""
        ),
        code(
            r"""# Waveform overlay for a deliberately editable example.
from paper1_qc.config import load_config, resolve_executable
from paper1_qc.media import decode_audio_views

cfg = load_config(CONFIG)
inventory = read_stage("00_audit/bamboo_media_inventory")
EXAMPLE_FILE = None  # replace with an exact filename, or leave None for the first resolvable file
candidate = EXAMPLE_FILE or segments["file_name"].dropna().iloc[0]
paths = inventory.loc[inventory["file_name"].eq(candidate), "file_path"].tolist()
assert len(paths) == 1, f"Expected one disk path for {candidate}; found {len(paths)}"
audio = decode_audio_views(
    paths[0],
    ffmpeg=resolve_executable(cfg["software"]["ffmpeg"], "ffmpeg"),
    ffprobe=resolve_executable(cfg["software"]["ffprobe"], "ffprobe"),
)
wave = audio.analysis_16k
time = np.arange(len(wave)) / 16000

example_intervals = segments.loc[
    segments["file_name"].eq(candidate) & segments["profile"].eq("primary")
].copy()
palette = {
    "raw_speech": "#4C78A8",
    "primary_speech": "#59A14F",
    "strict_speech": "#F28E2B",
    "strict_internal_nonspeech": "#E15759",
}
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(time, wave, color="0.25", linewidth=0.45, alpha=0.8)
for view, rows in example_intervals.groupby("view"):
    for first, interval in enumerate(rows.itertuples()):
        ax.axvspan(
            interval.start_sec,
            interval.end_sec,
            color=palette.get(view, "0.7"),
            alpha=0.14,
            label=view if first == 0 else None,
        )
ax.set(title=f"Waveform and segmentation views: {candidate}", xlabel="Time (s)", ylabel="Amplitude")
ax.legend(ncol=2, frameon=False)
save_figure(fig, "01_segmentation", "example_waveform_interval_overlay")
plt.show()
"""
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
    eligible.groupby("diagnosis_reported", dropna=False)
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
long = eligible[["file_name", "SubjectID", "diagnosis_reported", *metric_columns]].melt(
    id_vars=["file_name", "SubjectID", "diagnosis_reported"],
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
            hue="diagnosis_reported",
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
    ],
)


write(
    "05_goal4_perceptual_family_alignment.ipynb",
    "05 — Goal 4: perceptual family alignment",
    "Audits four independent RA annotations, evaluates matched family alignment, and compares the 4RA and merged 2RA systems on shared recordings.",
    [
        md(
            "Primary alignment excludes competing speech and non-task content because the "
            "estimand is family perceptual alignment, not source recognition. The broad "
            "metadata direction gate must be confirmed from the RA codebook before the "
            "4RA-versus-2RA comparison runs."
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
            r"""rating_coverage = read_stage("04_analysis/human_qc/rating_design_item_coverage")
design_summary = read_stage("04_analysis/human_qc/rating_design_summary")
ratings = read_stage("04_analysis/human_qc/ratings_long")
agreement = read_stage("04_analysis/human_qc/interrater_agreement")
consensus = read_stage("04_analysis/human_qc/four_ra_consensus_primary")
direction_audit = read_stage("04_analysis/human_qc/two_ra_broad_direction_and_scale_audit")

display(direction_audit)
assert direction_audit["direction"].eq("higher_is_worse").all()
save_table(direction_audit, "05_goal4", "direction_and_scale_audit")
save_table(design_summary, "05_goal4", "four_ra_design_summary")
display(design_summary)
"""
        ),
        code(
            r"""# Coverage heatmap makes missing/rotating rater designs visible.
coverage_matrix = (
    ratings.assign(rated=1)
    .pivot_table(index=["file_name", "category"], columns="rater_id", values="rated", aggfunc="max", fill_value=0)
)
save_table(coverage_matrix.reset_index(), "05_goal4", "four_ra_coverage_matrix")
fig, ax = plt.subplots(figsize=(10, min(18, max(5, .08 * len(coverage_matrix)))))
sns.heatmap(coverage_matrix, cmap=["#f2f2f2", "#4C78A8"], cbar=False, ax=ax)
ax.set(title="Independent detailed-rating coverage", xlabel="Rater", ylabel="Recording × perceptual family")
ax.tick_params(axis="y", labelleft=False)
save_figure(fig, "05_goal4", "four_ra_rating_coverage")
plt.show()
"""
        ),
        code(
            r"""# Prevalence is shown beside agreement because sparse categories can distort kappa.
prevalence = (
    consensus.groupby("category")["consensus_rating"]
    .agg(
        n_consensus="count",
        positive=lambda x: int(pd.to_numeric(x, errors="coerce").sum()),
        prevalence=lambda x: float(pd.to_numeric(x, errors="coerce").mean()),
    ).reset_index()
)
agreement_view = agreement.merge(prevalence, on="category", how="left")
save_table(agreement_view, "05_goal4", "agreement_and_prevalence")
display(agreement_view)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=prevalence, x="prevalence", y="category", ax=axes[0], color="#4C78A8")
axes[0].set(title="4RA consensus artifact prevalence", xlabel="Positive fraction", ylabel="")
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
axes[1].set(title="Gwet AC1 with item-bootstrap 95% CI", xlabel="Agreement", xlim=(-.1, 1.05))
fig.tight_layout()
save_figure(fig, "05_goal4", "prevalence_and_agreement")
plt.show()
"""
        ),
        code(
            r"""# Primary cross-family matrix. Diagonal cells are matched perceptual families.
four = read_stage("04_analysis/human_qc/four_ra_family_alignment_matrix")
four_effect = four.pivot(index="human_family", columns="objective_family", values="effect")
four_n = four.pivot(index="human_family", columns="objective_family", values="n_recordings")
save_table(four, "05_goal4", "four_ra_family_alignment_matrix")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(four_effect, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", ax=axes[0])
axes[0].set(title="4RA family alignment effect", xlabel="Objective Q family", ylabel="Perceptual family")
sns.heatmap(four_n, cmap="viridis", annot=True, fmt=".0f", ax=axes[1])
axes[1].set(title="Pair-specific recording denominator", xlabel="Objective Q family", ylabel="")
fig.tight_layout()
save_figure(fig, "05_goal4", "four_ra_alignment_and_denominators")
plt.show()

matched_summary = (
    four.loc[four["estimable"]]
    .groupby("matched_family")["effect"]
    .agg(["count", "mean", "median"]).reset_index()
)
save_table(matched_summary, "05_goal4", "matched_vs_mismatched_descriptive")
display(matched_summary)

specificity = read_stage("04_analysis/human_qc/four_ra_matched_family_specificity")
save_table(specificity, "05_goal4", "four_ra_matched_family_specificity")
display(specificity)
"""
        ),
        code(
            r"""# The merged 2RA workflow is comparable only for families with explicit overlap.
comparison = read_stage("04_analysis/human_qc/four_ra_vs_two_ra_paired_alignment")
save_table(comparison, "05_goal4", "four_ra_vs_two_ra_paired_alignment")
display(comparison)

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
        title="Paired shared-recording comparison",
        xlabel="ΔAUC: 4RA detailed − merged 2RA broad",
        ylabel="",
    )
    save_figure(fig, "05_goal4", "four_ra_minus_two_ra_delta_auc")
    plt.show()
"""
        ),
        code(
            r"""# Secondary duration/fraction analysis preserves the richer interval annotations.
extent = read_stage("04_analysis/human_qc/four_ra_extent_consensus_secondary")
context = read_stage("04_analysis/human_qc/context_annotations_not_family_alignment")
save_table(extent, "05_goal4", "four_ra_extent_consensus_secondary")

extent_summary = (
    extent.groupby("category")["consensus_annotated_fraction"]
    .agg(n="count", median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
    .reset_index()
)
save_table(extent_summary, "05_goal4", "extent_consensus_summary")
display(extent_summary)

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
    {"manuscript_role": "Segmentation supplement", "source": "outputs/visualization/01_segmentation/speech_fraction_ecdf.png", "status": "candidate"},
    {"manuscript_role": "Goal 1 support panel", "source": "outputs/visualization/02_goal1/metric_support_fraction.png", "status": "candidate"},
    {"manuscript_role": "Goal 1 acquisition distributions", "source": "outputs/visualization/02_goal1/raw_distributions__<family>.png", "status": "family facets"},
    {"manuscript_role": "Goal 2 persistence", "source": "outputs/visualization/03_goal2/participant_rank_persistence.png", "status": "candidate"},
    {"manuscript_role": "Goal 3 structure", "source": "outputs/visualization/04_goal3/correlation_and_support_matrices.png", "status": "candidate"},
    {"manuscript_role": "Goal 3 Rest sensitivity", "source": "outputs/visualization/04_goal3/rest_reference_level_comparison.png", "status": "supplement candidate"},
    {"manuscript_role": "Goal 4 family validity", "source": "outputs/visualization/05_goal4/four_ra_alignment_and_denominators.png", "status": "candidate"},
    {"manuscript_role": "Goal 4 label-system comparison", "source": "outputs/visualization/05_goal4/four_ra_minus_two_ra_delta_auc.png", "status": "candidate if estimable"},
])
save_table(figure_candidates, "06_registry", "manuscript_figure_candidates")
display(figure_candidates)
"""
        ),
    ],
)

print("Generated visualization notebooks under", ROOT / "visualization")
