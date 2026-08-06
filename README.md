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

1. `01_QGAIN`: `01_extract.ipynb`, `02_figures.ipynb`, `03_finalize.ipynb`
2. `02_QADD`: `01_preflight.ipynb`, `02_extract_cohort.ipynb`, `03_finalize.ipynb`
3. `03_QREV`: `01_preflight.ipynb`, `02_extract_cohort.ipynb`, `03_finalize.ipynb`
4. `04_QCHAN`: `01_preflight.ipynb`, `02_extract_cohort.ipynb`, `03_finalize.ipynb`
5. `05_QDIST`: `01_remediation_preflight.ipynb`, `02_extract_candidate_cohort.ipynb`, `03_verify_computational.ipynb`, optional `04_human_review_optional.ipynb`, then `05_finalize.ipynb`
6. `06_QTEMP`: `01_extract_reviewed.ipynb`, then `02_finalize_disposition.ipynb`. `RUN_QTEMP_FINALIZE.cmd` automates the final step.

Run notebooks within each folder in numeric order. Supporting review contracts, checklists, and audit records are kept in that family's `support/` directory. Executed notebooks and generated tables/figures are written beneath `outputs/`; reviewed results use `outputs/reviewed/`.

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
