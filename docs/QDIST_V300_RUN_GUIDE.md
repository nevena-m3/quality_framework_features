# QDIST v3.0.0 Windows/Jupyter run guide

## Patch contents

```text
src/paper1_qc/qdist.py
scripts/generate_qdist_v3_notebook.py
notebooks/02_feature_extraction/02e_nonlinear_distortion_QDIST_v3_0_0.ipynb
tests/test_qdist_v30.py
tests/test_qdist_notebook_v300.py
scripts/run_qdist_jupyter.ps1
docs/QDIST_V300_SCIENTIFIC_DECISION.md
docs/QDIST_V300_RUN_GUIDE.md
docs/QDIST_V300_RELEASE_NOTES.md
```

The legacy `02e_nonlinear_distortion.ipynb` is not the authoritative QDIST
notebook and should not be run for the manuscript feature release.

## Install the patch

From PowerShell:

```powershell
$repo = "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1"
$downloads = Join-Path $env:USERPROFILE "Downloads"
$zip = (Get-ChildItem "$downloads\QDIST_v3_0_0_R4_patch*.zip" |
    Sort-Object LastWriteTime -Descending)[0]
$tmp = Join-Path $env:TEMP "QDIST_v300"

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
# R4 changes detector parameters and implementation identity. Remove the
# incomplete R3 candidate output so no stale ledgers or tables remain.
Remove-Item "$repo\outputs\02_features\nonlinear_distortion\qdist-v3.0.0" `
    -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive $zip.FullName $tmp -Force
Copy-Item "$tmp\QDIST_v3_0_0_patch\*" $repo -Recurse -Force
Set-Location $repo
```

## Verify the clean source package

```powershell
& "$repo\.venv\Scripts\python.exe" -m pytest `
  tests\test_qdist_v30.py `
  tests\test_qdist_notebook_v300.py `
  -q
```

Expected result for this release:

```text
77 passed
```

Do not run the cohort notebook if a QDIST test fails.

## Regenerate the clean notebook

```powershell
& "$repo\.venv\Scripts\python.exe" `
  scripts\generate_qdist_v3_notebook.py
```

This writes the clean canonical notebook to:

```text
notebooks\02_feature_extraction\02e_nonlinear_distortion_QDIST_v3_0_0.ipynb
```

The generator is authoritative. Editing only the generated notebook will cause
the generator-identity test to fail and the edit can be lost at the next
regeneration.

## Launch in JupyterLab

Do not pass the notebook path as a positional argument to `jupyter lab`. On
Windows/JupyterLab 4 that can reset the server root to the notebook folder and
produce duplicated paths such as
`...\notebooks\02_feature_extraction\Desktop\...`, preventing autosave and
leaving the browser connected to a stale kernel.

Use the included launcher:

```powershell
& powershell.exe -ExecutionPolicy Bypass `
  -File "$repo\scripts\run_qdist_jupyter.ps1" `
  -Repo "$repo"
```

Or launch directly with an explicit repository root:

```powershell
$env:PYTHONPATH = "$repo\src"
& "$repo\.venv\Scripts\python.exe" -m jupyter lab `
  --ServerApp.root_dir="$repo" `
  --ServerApp.default_url="/lab/tree/notebooks/02_feature_extraction/02e_nonlinear_distortion_QDIST_v3_0_0.ipynb"
```

If a previous broken Jupyter session is still open, stop it with `Ctrl+C` twice
and close the old browser tab before launching again.

For the first candidate run, set:

```python
RUN_COHORT_EXTRACTION = True
RESUME_FROM_CHECKPOINTS = True
RUN_CODEC_CHARACTERIZATION = True
BUILD_GALLERY = True
PACKAGE_TESTS_CONFIRMED = True

PUBLISH_AND_FREEZE_QDIST_V300 = False
QDIST_REVIEW_DECISION = "PENDING"
```

Keep every final feature decision as `PENDING` during the first run.

Before running, ensure `config/project.yaml` explicitly pins the input vintages:

```yaml
data_freeze:
  version: v1

segmentation_freeze:
  version: v1
```

Preserve all existing keys under `data_freeze`; add `version: v1` rather than
replacing the section.

Use **Kernel → Restart Kernel and Clear All Outputs**, followed by **Run → Run
All Cells**. Do not rerun individual cells during the candidate execution.

Cell 7 prints one line per recording, including total time and separate source-identity/hash, decode, detector, and checkpoint-write times, plus candidate/plateau/episode counts, checkpoint action, and ETA. A slow label now distinguishes detector work from storage/decoding delay. It also updates:

```text
outputs\02_features\nonlinear_distortion\qdist-v3.0.0\audit\qdist_v300_live_progress.json
```

The extraction is sequential by design to limit memory and preserve deterministic
per-recording checkpoints. The R4 detector removes the candidate-count-dependent histogram bottleneck. A
first 519-recording pass should normally be governed by file hashing and native
decoding rather than tens of seconds of detector work per recording. Exact time
still depends on local storage and media format, but the run must show continuous
progress rather than multi-hour detector ETAs. Restarting the
cell reuses only complete source/code/parameter-matched checkpoints.

## Candidate outputs

The candidate run writes to:

```text
outputs\02_features\nonlinear_distortion\qdist-v3.0.0\
```

Review at minimum:

```text
tables\qdist_v300_gate_summary.csv
tables\qdist_v300_recording_features_full.csv
tables\qdist_v300_occurrence_summary.csv
tables\qdist_v300_empirical_distributions.csv
tables\qdist_v300_sparsity_analysis_suitability.csv
tables\qdist_v300_event_merge_rank_stability.csv
audit\qdist_v300_extraction_errors.csv
audit\qdist_v300_candidate_ledger.csv
audit\qdist_v300_accepted_plateau_ledger.csv
audit\qdist_v300_episode_ledger.csv
gallery\qdist_v300_blinded_review_template.csv
```

A detector-available no-event recording must contain zeros. A row with
`qdist_status = unavailable_coarse_quantization` must contain missing retained
features, not zeros.

## Blinded event review

Complete the generated review template without opening the private truth index.
For each item, listen to the 500-ms audio snippet and inspect the waveform and
local amplitude histogram. Record whether hard-clipping-like plateau evidence
is present, confidence, and notes.

The package always contains synthetic hard-clipping positive controls and clean
high-amplitude negative controls. When real accepted or rejected candidates
exist, those are included as well. The final review must document disagreement
or ambiguous cases; do not tune detector thresholds to the reviewed cohort.

## Final feature decisions

After reviewing all validation tables and the gallery, set each feature
independently. Typical pass labels are:

```python
QDIST_FINAL_FEATURE_DECISIONS = {
    "qdist_hard_clipped_frame_fraction": "PASS_PRIMARY",
    "qdist_hard_clip_event_rate_per_min": "PASS_PRIMARY_EVENT",
    "qdist_hard_clipped_sample_fraction": "PASS_SECONDARY",
}
```

A feature can instead be `DROP`. Enter a nonempty rationale for every feature.
The event-rate feature should be dropped if episode-merging robustness or event
adjudication is inadequate even when the two burden features pass.

Then set:

```python
QDIST_REVIEW_DECISION = "ACCEPT_QDIST_V300"
QDIST_REVIEWER = "Nevena Musikic"
QDIST_REVIEW_RATIONALE = "...specific review rationale..."
PUBLISH_AND_FREEZE_QDIST_V300 = True
```

Restart, clear all outputs, and run every cell once. Any failed blocking gate
stops publication. Do not bypass or rewrite a failed gate.

## Immutable outputs

A successful final run creates:

```text
MAIN outputs\02_FEATURE_FREEZE\nonlinear_distortion\qdist-v3.0.0\
MAIN outputs\02_FEATURE_TABLES\qdist_v300_analysis_features.csv
MAIN outputs\02_FEATURE_TABLES\qdist_v300_analysis_features.parquet
```

The central table contains only features that received a final PASS decision,
plus support, status, provenance, event-count, uncertainty, and quantization
availability fields. Candidate ledgers and gallery files remain inside the
family freeze.
