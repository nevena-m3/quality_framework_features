# QDIST v3.1.0 Windows/Jupyter run guide

## 1. Install the patch

Close JupyterLab and stop its PowerShell server with `Ctrl+C` twice. Then run:

```powershell
$repo = "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1"
$downloads = Join-Path $env:USERPROFILE "Downloads"
$zip = (Get-ChildItem "$downloads\QDIST_v3_1_0_patch*.zip" |
    Sort-Object LastWriteTime -Descending)[0]
$tmp = Join-Path $env:TEMP "QDIST_v310"

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive $zip.FullName $tmp -Force
Copy-Item "$tmp\QDIST_v3_1_0_patch\*" $repo -Recurse -Force
Set-Location $repo
```

Do not remove the old `qdist-v3.0.0` output if you want to preserve the failed candidate audit. Version 3.1.0 writes to a separate folder.

## 2. Regenerate the canonical clean notebook

```powershell
& "$repo\.venv\Scripts\python.exe" `
    "scripts\generate_qdist_v3_notebook.py"
```

The generated notebook is:

```text
notebooks\02_feature_extraction\02e_nonlinear_distortion_QDIST_v3_1_0.ipynb
```

## 3. Run package tests

```powershell
& "$repo\.venv\Scripts\python.exe" -m pytest `
    tests\test_qdist_v30.py `
    tests\test_qdist_notebook_v300.py `
    -q
```

Expected result:

```text
67 passed
```

Do not continue if any test fails.

## 4. Launch Jupyter from the repository root

```powershell
& powershell.exe -ExecutionPolicy Bypass `
    -File "$repo\scripts\run_qdist_jupyter.ps1" `
    -Repo "$repo"
```

Do not pass the notebook path directly to `jupyter lab`; the launcher fixes the server root and prevents duplicated save paths.

## 5. First candidate-run settings

Keep:

```python
PACKAGE_TESTS_CONFIRMED = True

RUN_COHORT_EXTRACTION = True
RESUME_FROM_CHECKPOINTS = True
RUN_CODEC_CHARACTERIZATION = True
BUILD_GALLERY = True

PUBLISH_AND_FREEZE_QDIST_V310 = False
QDIST_REVIEW_DECISION = "PENDING"
```

Keep all three feature decisions and rationales pending/blank. Do not enable freezing on the first run.

## 6. Execute cleanly

In JupyterLab:

```text
Kernel → Restart Kernel and Clear All Outputs
Run → Run All Cells
```

Progress is checkpointed after every recording. The live status file is:

```text
outputs\02_features\nonlinear_distortion\qdist-v3.1.0\audit\qdist_v310_live_progress.json
```

The detector prints total, identity, decode, detector, and checkpoint-write times. A stopped run can resume from valid v3.1.0 checkpoints.

## 7. Candidate outputs to upload for review

After the complete run, upload:

1. the executed notebook;
2. a ZIP of `outputs\02_features\nonlinear_distortion\qdist-v3.1.0`.

Do not freeze yet. The empirical event morphology and blinded review gallery must be examined before setting final feature decisions.

## 8. Freeze only after review

After scientific acceptance, complete:

```python
QDIST_REVIEW_DECISION = "ACCEPT_QDIST_V310"
QDIST_REVIEWER = "Nevena Musikic"
QDIST_REVIEW_RATIONALE = "..."

QDIST_FINAL_FEATURE_DECISIONS = {
    "qdist_hard_clipped_frame_fraction": "PASS_PRIMARY",  # or REVISE/DROP
    "qdist_hard_clip_event_rate_per_min": "PASS_PRIMARY_EVENT",  # or REVISE/DROP
    "qdist_hard_clipped_sample_fraction": "PASS_SECONDARY",  # or AUDIT_ONLY/DROP
}
QDIST_FEATURE_DECISION_RATIONALES = {
    "qdist_hard_clipped_frame_fraction": "...",
    "qdist_hard_clip_event_rate_per_min": "...",
    "qdist_hard_clipped_sample_fraction": "...",
}
PUBLISH_AND_FREEZE_QDIST_V310 = True
```

Then restart, clear outputs, and run all cells once. The notebook refuses to freeze when any blocking gate fails or a final feature decision is incomplete.
