# QDIST v3.1.1 — Windows PowerShell run guide

Project root:

```text
C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1
```

## Apply the patch

Assuming the downloaded ZIP is in `Downloads`:

```powershell
$Project = "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1"
$Zip = "$env:USERPROFILE\Downloads\QDIST_v3_1_1_corrected_patch.zip"
$Temp = "$env:TEMP\QDIST_v3_1_1_patch"

Remove-Item $Temp -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $Zip -DestinationPath $Temp -Force
Copy-Item "$Temp\QDIST_v3_1_1_corrected_patch\*" $Project -Recurse -Force
Set-Location $Project
```

## Install and test

```powershell
python -m pip install -e .
python -m pytest `
  .\tests\test_qdist_v31.py `
  .\tests\test_qdist_notebook_v310.py `
  .\tests\test_qdist_v311.py `
  .\tests\test_qdist_notebook_v311.py `
  .\tests\test_media.py -q
```

The corrected delivered set should report all focused tests passing. Do not proceed if any QDIST or media test fails.

## Regenerate and open the notebook

```powershell
python .\scripts\generate_qdist_v311_notebook.py
python -m jupyter lab
```

Open:

```text
notebooks\02_feature_extraction\02e_nonlinear_distortion_QDIST_v3_1_1.ipynb
```

Run from top to bottom. Leave these controls unchanged for the first run:

```python
PUBLISH_AND_FREEZE_QDIST_V311 = False
QDIST_REVIEW_DECISION = "PENDING"
```

## Runtime behavior

The first full run decodes every recording and writes one compressed checkpoint per recording. Later reruns reuse valid checkpoints. Progress prints every 25 recordings with an estimated remaining time.

Do not copy or rename old v3.1.0 outputs into the new directory. Corrected outputs are isolated under:

```text
outputs\02_features\nonlinear_distortion\qdist-v3.1.1\
```

## Required review before freeze

Review at minimum:

```text
tables\qdist_v311_gate_summary.csv
tables\qdist_v311_feature_decisions.csv
tables\qdist_v311_analysis_features.csv
tables\qdist_v311_accepted_plateau_ledger.csv
tables\qdist_v311_episode_ledger.csv
tables\qdist_v311_cohort_parameter_robustness_summary.csv
gallery\qdist_v311_gallery_review.csv
audit\qdist_v311_candidate_manifest.json
```

The gallery review includes every accepted plateau. Allowed labels are:

```text
DEFINITE_HARD_CLIP
PROBABLE_HARD_CLIP
AMBIGUOUS
NOT_HARD_CLIP
CANNOT_DETERMINE
```

Complete the reviewer, label, and rationale fields, then rerun the gallery and final gate cells. Enable freeze only after all blocking gates pass and the feature-decision table shows `PASS_RETAIN` for every retained feature.
