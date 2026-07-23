"""Generate thin, reviewable notebooks that call the tested package implementation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


BOOTSTRAP = """from pathlib import Path
import subprocess, sys
import pandas as pd

def find_project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / 'pyproject.toml').exists():
            return candidate
    raise FileNotFoundError('Run this notebook from inside the paper_1 project.')

ROOT = find_project_root()
CONFIG = ROOT / 'config' / 'project.yaml'
OUTPUT = ROOT / 'outputs'
print('Project:', ROOT)
print('Config:', CONFIG)
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
            markdown(f"# {title}\n\n{purpose}\n\nCore algorithms live in `src/paper1_qc`; this notebook is an auditable orchestration/reporting layer."),
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
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def cli_cell(command: str) -> dict:
    return code(
        f"cmd = [sys.executable, '-m', 'paper1_qc.cli', '--config', str(CONFIG), {', '.join(repr(p) for p in command.split())}]\n"
        "print(' '.join(cmd))\nsubprocess.run(cmd, cwd=ROOT, check=True)"
    )


def feature_review_cell(family: str) -> dict:
    return code(
        "from paper1_qc.registry import metric_registry_frame\n"
        "registry = metric_registry_frame()\n"
        f"display(registry.loc[registry['family'] == {family!r}].reset_index(drop=True))"
    )


write_notebook(
    "notebooks/00_setup/00_environment_check.ipynb",
    "00 — Environment and configuration check",
    "Checks the local configuration and executable entry point before reading participant data.",
    [
        code("assert CONFIG.exists(), 'Copy config/project.example.yaml to config/project.yaml and review it.'\nsubprocess.run([sys.executable, '-m', 'paper1_qc.cli', '--help'], cwd=ROOT, check=True)"),
    ],
)
write_notebook(
    "notebooks/00_setup/00_metadata_and_media_audit.ipynb",
    "00 — Metadata and media integrity audit",
    "Runs metadata reconciliation first, then native-stream inventory/hashing. Modeling is blocked until errors are reviewed.",
    [cli_cell("audit"), cli_cell("inventory"), code("display(pd.read_csv(OUTPUT / '00_audit' / 'cross_workbook_summary.csv'))")],
)
write_notebook(
    "notebooks/01_segmentation/01_segmentation_silero_full_dataset.ipynb",
    "01 — Version-pinned speech segmentation",
    "Runs installed Silero 6.2.1 and saves raw, primary, strict-speech, and guarded internal-nonspeech intervals for three profiles.",
    [cli_cell("segment"), code("segments = pd.read_csv(OUTPUT / '01_segmentation' / 'bamboo_segmentation_intervals.csv')\ndisplay(segments.groupby(['profile', 'view']).duration_sec.agg(['count','sum','median']))")],
)

feature_notebooks = [
    ("02a_additive_interference.ipynb", "02a — Additive interference", "additive_interference"),
    ("02b_gain_dynamics.ipynb", "02b — Gain and amplitude dynamics", "gain_dynamics"),
    ("02c_reverberation_tail.ipynb", "02c — Reverberation-tail proxies", "reverberation_tail"),
    ("02d_channel_device.ipynb", "02d — Channel and device descriptors", "channel_device"),
    ("02e_nonlinear_distortion.ipynb", "02e — Nonlinear distortion", "nonlinear_distortion"),
    ("02f_temporal_discontinuity.ipynb", "02f — Temporal discontinuity", "temporal_discontinuity"),
]
for index, (filename, title, family) in enumerate(feature_notebooks):
    cells = [feature_review_cell(family)]
    if index == 0:
        cells.append(markdown("Run the centralized extractor once. All six families are produced together so row sets and provenance cannot drift between notebooks."))
        cells.append(cli_cell("extract --profile primary"))
    cells.append(
        code(
            "metrics = pd.read_csv(OUTPUT / '02_features' / 'bamboo_q_metrics.csv')\n"
            f"columns = ['file_name'] + [c for c in metrics if c.startswith({family.split('_')[0].replace('additive','qadd').replace('gain','qgain').replace('reverberation','qrev').replace('channel','qchan').replace('nonlinear','qdist').replace('temporal','qtemp')!r})]\n"
            "display(metrics[columns].head())"
        )
    )
    write_notebook(
        f"notebooks/02_feature_extraction/{filename}",
        title,
        "Reviews the frozen registry, support requirements, interpretation, and observed outputs for this family.",
        cells,
    )

write_notebook(
    "notebooks/03_dataset_assembly/03a_assemble_analysis_dataset.ipynb",
    "03a — Assemble audited analysis dataset",
    "Performs a validated one-to-one merge and creates explicit measurement, diagnosis, and clinical eligibility gates.",
    [cli_cell("assemble"), code("display(pd.read_csv(OUTPUT / '03_dataset_assembly' / 'eligibility_flow_counts.csv'))")],
)
write_notebook(
    "notebooks/03_dataset_assembly/03b_dataset_statistics.ipynb",
    "03b — Cohort and missingness accounting",
    "Produces cohort counts from the assembled table; no counts are copied manually into the manuscript.",
    [code("data = pd.read_csv(OUTPUT / '03_dataset_assembly' / 'paper1_analysis_dataset.csv')\ndisplay(data.groupby('diagnosis_reported', dropna=False).agg(recordings=('logical_recording_id','nunique'), participants=('SubjectID','nunique')))\ndisplay(data.filter(regex='_status$').apply(pd.Series.value_counts))")],
)
write_notebook(
    "notebooks/04_analysis/05_study_goal_1_acquisition_variability.ipynb",
    "05 — Goal 1: distributions and acquisition variability",
    "Runs participant-clustered descriptive inference and participant-level exploratory group contrasts without SMOTE or record-level pseudoreplication.",
    [cli_cell("describe"), code("display(pd.read_csv(OUTPUT / '04_analysis' / 'descriptive' / 'metric_descriptive_statistics.csv'))")],
)
write_notebook(
    "notebooks/04_analysis/06_study_goal_2_participant_persistence.ipynb",
    "06 — Goal 2: participant persistence",
    "Reviews the separately labeled participant-persistence variance partition. Persistence is not called test–retest reliability.",
    [code("display(pd.read_csv(OUTPUT / '04_analysis' / 'descriptive' / 'participant_persistence_not_reliability.csv'))")],
)
write_notebook(
    "notebooks/04_analysis/07_study_goal_3_internal_structure_and_robustness.ipynb",
    "07 — Goal 3: multidimensional structure and robustness",
    "Reviews pairwise structure and points to the separate sensitivity stages. The long-form visual audit is in visualization/04_goal3_multidimensional_structure_and_robustness.ipynb.",
    [
        code("display(pd.read_csv(OUTPUT / '04_analysis' / 'descriptive' / 'pairwise_clustered_spearman.csv').head(40))"),
    ],
)
write_notebook(
    "notebooks/04_analysis/08_study_goal_4_perceptual_family_alignment.ipynb",
    "08 — Goal 4: perceptual family alignment and reliability",
    "Audits the distributed one-RA-per-recording main labels separately from the crossed four-RA Reliability subset, then compares matched Q-family alignment with the merged two-RA metadata workflow.",
    [
        markdown("Review and copy the human-QC schema. The four declared RA folders identify main assignments; Reliability/<RA name>/ must contain the shared crossed subset. Confirm broad-label direction from the RA codebook."),
        cli_cell("human-qc --schema config/human_qc_schema.yaml"),
    ],
)
write_notebook(
    "notebooks/04_analysis/09_sensitivity_summary.ipynb",
    "09 — Sensitivity and robustness summary",
    "Compares conservative and permissive segmentation profiles, paired encodings, and exact-session Rest context against the pre-specified primary analysis.",
    [
        cli_cell("extract --profile conservative"),
        cli_cell("extract --profile permissive"),
        cli_cell("sensitivity"),
        cli_cell("rest-reference"),
        cli_cell("encoding-sensitivity"),
        code("display(pd.read_csv(OUTPUT / '04_analysis' / 'sensitivity' / 'segmentation_profile_robustness.csv'))"),
    ],
)

print("Generated notebooks under", ROOT / "notebooks")
