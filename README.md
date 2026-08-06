# Quality Framework Features Pipeline

Minimal runnable repository for the Paper 1 remote-speech quality framework. Reviewed work is identified by the `paper1_qc_reviewed` package, `REVIEWED` notebook names, and explicit version numbers—not by duplicate top-level folders.

## Setup

Use Python 3.11 and install FFmpeg/FFprobe on `PATH`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,reverb]"
Copy-Item config\project.example.yaml config\project.yaml
Copy-Item config\human_qc_schema.example.yaml config\human_qc_schema.yaml
$env:PYTHONPATH = "$PWD\src"
```

Review `config/project.yaml` before running cohort notebooks. Local configuration, participant media, and generated outputs are excluded from Git.

## Run order

Run the preparation notebooks first:

1. `notebooks/00_setup/00_environment_check.ipynb`
2. `notebooks/00_setup/00_metadata_and_media_audit.ipynb`
3. `notebooks/01_segmentation/01_segmentation_silero_full_dataset.ipynb`

Then run each reviewed feature family in folder order:

1. `notebooks/02_feature_extraction/01_QGAIN/` — v4.0.1 extraction, v4.1.0 figure completion, then finalization.
2. `notebooks/02_feature_extraction/02_QADD/` — preflight, cohort, then finalization.
3. `notebooks/02_feature_extraction/03_QREV/` — preflight, cohort, then finalization.
4. `notebooks/02_feature_extraction/04_QCHAN/` — preflight, cohort, then finalization.
5. `notebooks/02_feature_extraction/05_QDIST/` — remediation preflight, candidate cohort, computational verification, optional human adjudication, then automated finalization.
6. `notebooks/02_feature_extraction/06_QTEMP/` — reviewed source, then final analytical disposition. `RUN_QTEMP_FINALIZE.cmd` automates the final step.

Only `*_SOURCE.ipynb` notebooks are version-controlled. Executed notebooks and generated tables/figures are written beneath `outputs/`; reviewed results use `outputs/reviewed/`.

After feature extraction, run:

1. `notebooks/03_dataset_assembly/03a_assemble_analysis_dataset.ipynb`
2. `notebooks/03_dataset_assembly/03b_dataset_statistics.ipynb`
3. Notebooks in `notebooks/04_analysis/` in numeric order
4. `notebooks/05_human_QC_reliability/Bamboo_RA_Interrater_Reliability_FULL.ipynb` when human-QC reliability analysis is required

## Repository contents

```text
config/                          Configuration templates
notebooks/                       All pipeline notebooks
notebooks/02_feature_extraction/ Reviewed feature-family notebooks and inputs
scripts/                         Reviewed finalization and freeze helpers
src/paper1_qc/                   Shared core package
src/paper1_qc_reviewed/          Reviewed feature implementations
tests/                           Tests invoked by reviewed notebooks
```

## Verification

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest
```

Do not commit local configuration, participant data, virtual environments, generated outputs, executed notebooks, or manual backup copies.
