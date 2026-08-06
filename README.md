# Quality Framework Features Pipeline

Minimal runnable repository for the Paper 1 remote-speech quality framework. The active tree contains source code, current source notebooks, required notebook inputs, current freeze helpers, and notebook-invoked tests. Historical versions and completed runs are available through Git history.

## Setup

Use Python 3.11 and install FFmpeg/FFprobe on `PATH`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,reverb]"
Copy-Item config\project.example.yaml config\project.yaml
Copy-Item config\human_qc_schema.example.yaml config\human_qc_schema.yaml
$env:PYTHONPATH = "$PWD\src;$PWD\src reviewed"
```

Review `config/project.yaml` before running cohort notebooks. Local configuration, participant media, and generated outputs are intentionally excluded from Git.

## Run order

Run the core preparation notebooks first:

1. `notebooks/00_setup/00_environment_check.ipynb`
2. `notebooks/00_setup/00_metadata_and_media_audit.ipynb`
3. `notebooks/01_segmentation/01_segmentation_silero_full_dataset.ipynb`

Then run each current reviewed feature family in folder order:

1. `notebooks reviewed/01_QGAIN/` — v4.0.1 extraction, v4.1.0 figure completion, then finalization.
2. `notebooks reviewed/02_QADD/` — preflight, cohort, then finalization.
3. `notebooks reviewed/03_QREV/` — preflight, cohort, then finalization.
4. `notebooks reviewed/04_QCHAN/` — preflight, cohort, then finalization.
5. `notebooks reviewed/05_QDIST/` — remediation preflight, candidate cohort, computational verification, optional human adjudication, then automated finalization.
6. `notebooks reviewed/06_QTEMP/` — reviewed source, then final analytical disposition. `RUN_QTEMP_FINALIZE.cmd` automates the final step.

Only `*_SOURCE.ipynb` notebooks are version-controlled. Executed notebooks and generated tables/figures are written to ignored output directories.

After feature extraction, run:

1. `notebooks/03_dataset_assembly/03a_assemble_analysis_dataset.ipynb`
2. `notebooks/03_dataset_assembly/03b_dataset_statistics.ipynb`
3. Notebooks in `notebooks/04_analysis/` in numeric order
4. `notebooks/05_human_QC_reliability/Bamboo_RA_Interrater_Reliability_FULL.ipynb` when human-QC reliability analysis is required

## Repository contents

```text
config/              Configuration templates
notebooks/           Preparation, assembly, and analysis notebooks
notebooks reviewed/  Current feature-family source notebooks and required inputs
scripts reviewed/    Current finalization and freeze helpers
src/                  Shared core package required by reviewed code
src reviewed/         Current reviewed feature implementations
tests reviewed/       Tests invoked by current feature notebooks
```

## Verification

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\src reviewed"
python -m pytest "tests reviewed"
```

Do not commit local configuration, participant data, virtual environments, generated outputs, executed notebooks, or manual backup copies.
