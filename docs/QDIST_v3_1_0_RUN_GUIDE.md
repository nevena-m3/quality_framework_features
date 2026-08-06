# QDIST v3.1.0 — Windows/PowerShell run guide

Project root:

```text
C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1
```

## 1. Place the candidate files

Extract the delivered patch or updated pipeline ZIP into the project root, preserving the relative folders. Confirm these files exist:

```text
src\paper1_qc\qdist.py
tests\test_qdist_v31.py
tests\test_qdist_notebook_v310.py
scripts\generate_qdist_v31_notebook.py
notebooks\02_feature_extraction\02e_nonlinear_distortion_QDIST_v3_1_0.ipynb
```

## 2. Open PowerShell and activate the project

```powershell
cd "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1"
```

Activate the same Python environment used for QADD/QGAIN/QREV. Then install the project in editable mode if needed:

```powershell
python -m pip install -e .
```

## 3. Regenerate the notebook from the governed generator

```powershell
python .\scripts\generate_qdist_v31_notebook.py
```

The generator should print:

```text
...\notebooks\02_feature_extraction\02e_nonlinear_distortion_QDIST_v3_1_0.ipynb
```

## 4. Run focused package tests

```powershell
python -m pytest .\tests\test_qdist_v31.py .\tests\test_qdist_notebook_v310.py -q
```

Expected candidate-package result:

```text
24 passed
```

Do not run the cohort notebook if these tests fail.

## 5. Open Jupyter

```powershell
jupyter lab
```

Open:

```text
notebooks\02_feature_extraction\02e_nonlinear_distortion_QDIST_v3_1_0.ipynb
```

Run the notebook from top to bottom. The first run must remain candidate-only:

```python
PUBLISH_AND_FREEZE_QDIST_V31 = False
QDIST_REVIEW_DECISION = "PENDING"
```

## 6. Review required outputs

Candidate outputs are written under:

```text
outputs\02_features\nonlinear_distortion\qdist-v3.1.0\
```

Review, at minimum:

- `tables\qdist_v31_gate_summary.csv`
- `tables\qdist_v31_analysis_features.csv`
- `tables\qdist_v31_candidate_plateau_ledger.csv`
- `tables\qdist_v31_accepted_plateau_ledger.csv`
- `tables\qdist_v31_episode_ledger.csv`
- `tables\qdist_v31_cohort_parameter_robustness_summary.csv`
- `gallery\qdist_v31_gallery_review.csv`
- `audit\qdist_v31_candidate_manifest.json`

Fill `gallery\qdist_v31_gallery_review.csv` using only these labels:

```text
DEFINITE_HARD_CLIP
PROBABLE_HARD_CLIP
AMBIGUOUS
NOT_HARD_CLIP
CANNOT_DETERMINE
```

Rerun the gallery and final gate cells after review. Do not enable freeze until the cohort results and event morphology have been interpreted scientifically.
