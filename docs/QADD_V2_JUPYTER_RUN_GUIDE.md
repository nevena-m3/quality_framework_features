# Qadd v2 notebook — Windows/Jupyter run guide

## 1. Commit the current working state first

Open PowerShell in the repository:

```powershell
Set-Location "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1"
git status
git add .
git commit -m "Checkpoint before Qadd v2 measurement update"
git push
git switch -c qadd-v2-measurement
```

If `git status` shows files you do not want committed, stop and review them
before running `git add .`.

## 2. Install the notebook

The ZIP is arranged relative to the repository root. Expand it while PowerShell
is inside `paper_1`:

```powershell
Expand-Archive `
  -Path "$env:USERPROFILE\Downloads\paper1_qadd_v2_notebook_patch.zip" `
  -DestinationPath (Get-Location).Path `
  -Force
```

This replaces:

```text
notebooks\02_feature_extraction\02a_additive_interference.ipynb
```

and adds the scientific specification and this guide under `docs\`.

## 3. Activate the correct Python environment

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -e ".[dev,reverb]"
python -m pytest
```

Use Python 3.11 or 3.12. The notebook requires `pyarrow` because the complete
frame- and window-level audit ledgers are Parquet files.

## 4. Start Jupyter

Use the module form so Windows does not call a stale launcher:

```powershell
python -m jupyter lab
```

Open:

```text
notebooks/02_feature_extraction/02a_additive_interference.ipynb
```

Select the `Paper 1 QC` kernel.

## 5. First complete run

In the run-control cell set:

```python
RUN_CENTRAL_EXTRACTION = False
RUN_QADD_V2_EXTRACTION = True
RUN_BOUNDARY_SENSITIVITY = True
RUN_REAL_AUDIO_GAIN_CHECK = True
RUN_REST_CONTEXT = False

PUBLISH_AND_FREEZE_QADD_V2 = False
QADD_REVIEW_DECISION = "UNDECIDED"
```

Keep `RUN_CENTRAL_EXTRACTION=False` when
`outputs/02_features/bamboo_q_metrics.csv` already exists. The new Qadd code
will use the frozen post-review segmentation directly.

Choose **Run → Run All Cells**. The full extraction is the long step. The
progress bar advances once per recording.

## 6. Review before freezing

The final technical gate must show PASS. Inspect:

```text
outputs\02_features\additive_interference\figures
outputs\02_features\additive_interference\tables\empirical_example_review_queue.csv
outputs\02_features\additive_interference\tables\qadd_v2_technical_gate.csv
```

Use the recording-audit cell to play and plot low, central, high, and
robust-extreme examples. A genuine extreme is retained. Return to segmentation
only for a documented boundary error.

If a gate fails, do not open the human-QC outcome tables and do not continue to
perceptual alignment. The failed gate names the exact saved table to inspect.

## 7. Freeze and publish

After reviewing every required output, edit the run-control cell:

```python
QADD_REVIEW_DECISION = "ACCEPT_QADD_V2"
QADD_REVIEWER = "your name"
QADD_REVIEW_RATIONALE = "Reviewed all technical gates, figures, and queued examples."
PUBLISH_AND_FREEZE_QADD_V2 = True
```

Rerun only the final cell. It will:

- preserve old Qadd columns with a `legacy_` prefix;
- publish Qadd v2 into the central all-family feature table;
- create the necessary frozen artifacts under:

```text
MAIN outputs\02_QADD_FREEZE\qadd-v2.0.0
```

Figures and large audit ledgers remain under `outputs`; they are intentionally
not duplicated into `MAIN outputs`.

## 8. Commit the accepted implementation

Do not commit generated `outputs` unless your repository policy explicitly
tracks them.

```powershell
git add notebooks\02_feature_extraction\02a_additive_interference.ipynb `
        docs\Qadd_Scientific_Measurement_Specification_v2.docx `
        docs\QADD_V2_JUPYTER_RUN_GUIDE.md
git commit -m "Implement auditable Qadd v2 measurement notebook"
git push -u origin qadd-v2-measurement
```
