# Git and Windows run guide

## 1. Put the project in a safe local location

Keep the code outside the data directory. One suitable layout is:

```text
C:\Users\musikicn\Desktop\Nevena_project\
  paper_1\
  Data_13072026\
```

Use your cloned `paper_1` repository for the code. Do not place audio, metadata workbooks,
RA exports, generated outputs, or `config/project.yaml` in Git. The included `.gitignore`
blocks the common forms, but `git status` must still be reviewed before every commit.

## 2. Install the prerequisites

- Git for Windows
- Python 3.11 (64 bit)
- FFmpeg and FFprobe on `PATH`

In PowerShell:

```powershell
git --version
py -3.11 --version
ffmpeg -version
ffprobe -version
```

If PowerShell blocks virtual-environment activation, use a process-only policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3. Create the Python environment

```powershell
cd "C:\Users\musikicn\Desktop\Nevena_project\paper_1"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,reverb]"
```

The `reverb` extra is optional only if the optional SRMR metric will not be used. The
registered core reverberation-tail metrics do not depend on SRMR.

## 4. Create and review local configuration

```powershell
Copy-Item config\project.example.yaml config\project.yaml
Copy-Item config\human_qc_schema.example.yaml config\human_qc_schema.yaml
notepad config\project.yaml
notepad config\human_qc_schema.yaml
```

The example data root is already:

```text
C:\Users\musikicn\Desktop\Nevena_project\Data_13072026
```

Check that directory names and workbook filenames exactly match the local data. The
human-QC schema is preconfigured for:

```text
Bamboo_passage_HumanQC\
  Abbas\
  Liya\
  Samaana\
  Samara\
  Reliability\
    Abbas\
    Liya\
    Samaana\
    Samara\
```

Do not create a manifest for this layout. The four top-level RA folders contain different
main files (one independent RA per recording). The four `Reliability` folders contain the
same approximately 70 recordings rated independently by all four RAs.

In the local `config\project.yaml` only, record the investigator-confirmed exceptional
controls and their evidence. These identifiers are deliberately absent from the tracked
example configuration:

```yaml
data_freeze:
  confirmed_control_subject_ids: ['C05-1', 'CNEC024-1']
  confirmed_control_subject_evidence: 'Investigator confirmation on 2026-07-23, consistent with Ivan data-manager message'
```

Do not commit `config\project.yaml`; it contains local paths and participant identifiers.

Before the 2RA comparison, verify the broad-QC codebook. If and only if `Yes` means that
the artifact is present and `No` means it is absent, change:

```yaml
direction_confirmed: true
```

in `config/human_qc_schema.yaml`.

## 5. Run the tested pipeline in order

Run these from the project root with the virtual environment active:

```powershell
paper1-qc --config config/project.yaml audit
paper1-qc --config config/project.yaml inventory
paper1-qc --config config/project.yaml freeze-template
```

Open `config\metadata_adjudication.csv`. Every generated row must contain
`ALS`, `CONTROLS`, or `EXCLUDE` in `diagnosis_analysis`, plus a nonblank
`evidence_source`. Save it, then continue:

```powershell
paper1-qc --config config/project.yaml freeze
paper1-qc --config config/project.yaml segment
paper1-qc --config config/project.yaml segment-template
```

Open `notebooks\01_segmentation\01_segmentation_silero_full_dataset.ipynb` in Jupyter.
The notebook creates a diagnosis/outcome-independent mandatory queue containing every flagged/excluded
recording and accepted segmentation-only outliers. Non-outlying accepted recordings are
pre-filled as `KEEP + AUTO`. The scrollable/searchable review widget shows every
recording, the original PNG, boundary audit, audio player, review reasons, and an
optional manual speech-boundary editor. Use **Keep Silero + next** for the ordinary case.
Rows whose frozen metadata says `Task Completed as Instructed = NO` are automatically
set to `EXCLUDE + NONE`, visibly documented, and locked.

```text
outputs\01_segmentation\figures\segmentation\silero\flagged\
outputs\01_segmentation\figures\segmentation\silero\excluded\
```

The Silero command keeps all segmentation-stage artifacts under
`outputs\01_segmentation`, with separate `figures` and `segmentation` branches:

```text
outputs\01_segmentation\
  logs\silero_segmentation_config.json
  segmentation\silero\
    segments\<recording>_segments.csv
    frames\<recording>_frames.csv
    boundary_audit\<recording>_boundary_audit.csv
    summary\silero_all_summary.csv
  figures\segmentation\silero\
    accepted\<recording>_silero.png
    flagged\<recording>_silero.png
    excluded\<recording>_silero.png
    boundary_audit\<recording>_boundary_audit.png
```

There must be exactly one segment CSV, one frame CSV, one original-style PNG across
the three status folders, one boundary-audit CSV, and one boundary-audit PNG for every
frozen Bamboo recording. Primary boundaries are unpadded Silero sample indices and
receive no second gap-bridge/filter pass. The 30-ms layer is retained only for the
familiar visualization/CSV contract. Low local energy contrast prompts review but
never moves a boundary automatically, because weak/breathy ALS speech can legitimately
have low contrast.

For each mandatory row choose:

- `KEEP + AUTO` when Silero boundaries are usable;
- `KEEP + MANUAL` when boundaries must be corrected, entering one
  `start_sec,end_sec` speech interval per line and a reason;
- `EXCLUDE + NONE` when the recording is unusable/non-task, with a reason.

The reviewer and review date are required. Manual intervals must be previewed and must
not be edited to remove noise or other artifacts. After the notebook reports zero
pending/incomplete reviews, set `RUN_SEGMENTATION_ADJUDICATION=True`. This runs:

```powershell
paper1-qc --config config/project.yaml segment-adjudicate
```

Confirm both freeze tables exist before continuing:

```text
MAIN outputs\01_SEGMENTATION_FREEZE\<version>\frozen_segmentation_decisions.csv
MAIN outputs\01_SEGMENTATION_FREEZE\<version>\frozen_segmentation_intervals.csv
```

Also confirm the separate post-review artifact tree exists:

```text
outputs\01_segmentation_after_review\
  segmentation\silero\
    segments\accepted|flagged|excluded\
    frames\accepted|flagged|excluded\
    boundary_audit\accepted|flagged|excluded\
    summary\silero_after_review_summary.csv
  figures\segmentation\silero\
    accepted|flagged|excluded\
    boundary_audit\accepted|flagged|excluded\
  tables\reviewed_segmentation_recordings.csv
  tables\reviewed_segmentation_intervals.csv
  tables\reviewed_segmentation_status_counts.csv
```

`accepted` and `flagged` are both retained for feature extraction. `flagged` means the
recording remains visibly auditable after required review or manual boundary editing;
it does not mean exclusion. Only `excluded` recordings are removed downstream.

This directory is immutable. If a justified correction is required after freezing,
increment `segmentation_freeze.version` in local `config\project.yaml`; never overwrite
the prior version. Archive/rename the matching `outputs\01_segmentation_after_review`
directory before deliberately creating a revised freeze.

Then run:

```powershell
paper1-qc --config config/project.yaml extract --profile primary
paper1-qc --config config/project.yaml assemble
paper1-qc --config config/project.yaml describe
```

Stop and review the corresponding `tables\` and `figures\` directories after every
notebook stage. Do not proceed past an incomplete diagnosis adjudication, unresolved
metadata errors, a logical recording without one selected decodable encoding, failed
decoding, incomplete Silero adjudication, or unexplained support failures. When valid WAV
and WEBM copies have the same logical recording name, the freeze selects WAV; the paired
encoding command later treats them as technical replicates.

Run the required sensitivities:

```powershell
paper1-qc --config config/project.yaml extract --profile conservative
paper1-qc --config config/project.yaml extract --profile permissive
paper1-qc --config config/project.yaml sensitivity
paper1-qc --config config/project.yaml encoding-sensitivity
paper1-qc --config config/project.yaml rest-reference
```

Run Goal 4 last:

```powershell
paper1-qc --config config/project.yaml human-qc --schema config/human_qc_schema.yaml
```

This command verifies the top-level distributed design separately from the crossed
Reliability design. It estimates main-set family alignment within rater; only complete
four-rater Reliability items enter primary agreement and consensus. Deviations are saved
as audit tables rather than silently pooled.

## 6. Run the visualization notebooks

From a new PowerShell window:

```powershell
cd "C:\Users\musikicn\Desktop\Nevena_project\paper_1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m ipykernel install --user --name paper1-qc --display-name "Paper 1 QC"
jupyter lab
```

In JupyterLab, double-click the `visualization` folder, open
`00_preflight_and_run_order.ipynb`, select **Kernel → Change Kernel → Paper 1 QC**, change
`RUN_PIPELINE_STAGES = False` to `True`, and choose **Run → Run All Cells**. Review every
red/error audit row before opening the next notebook. This first notebook also counts the
main files per RA and verifies that every Reliability export filename is present under
all four RA folders before any agreement statistic is attempted.

Open and run:

```text
00_preflight_and_run_order.ipynb
01_segmentation_visual_audit.ipynb
02_goal1_occurrence_and_acquisition_variability.ipynb
03_goal2_participant_persistence.ipynb
04_goal3_multidimensional_structure_and_robustness.ipynb
05_goal4_perceptual_family_alignment.ipynb
06_results_registry_and_manuscript_tables.ipynb
```

On a first analysis run, set each notebook's `RUN_...` switch to `True`. Run one notebook
at a time in the listed order; segmentation and extraction can be long-running. For a
frozen reporting rerun, leave the switches `False` and use the saved stage outputs.
Figures and tables are written to `outputs\visualization`.

## 7. Run automated checks

```powershell
python -m pytest
python -m ruff check src tests scripts
```

## 8. Create a private GitHub repository

Create an empty **private** repository on GitHub. Do not initialize it with a README,
license, or `.gitignore`, because those files already exist locally. Then:

```powershell
cd "C:\Users\musikicn\Desktop\Nevena_project\paper_1"
git init
git branch -M main
git add .
git status
git diff --cached --stat
git commit -m "Rebuild Paper 1 acoustic QC pipeline"
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-PRIVATE-REPOSITORY.git
git push -u origin main
```

Before committing, `git status` should show code, documentation, configuration examples,
tests, and notebooks only. It must not show participant audio, workbooks, raw RA exports,
local paths/configuration, or `outputs`.

For later work:

```powershell
git status
git add src tests docs visualization notebooks config README.md pyproject.toml
git diff --cached
git commit -m "Describe the scientific change"
git push
```

Never use `git add -f` to override the data exclusions. If a sensitive file is ever staged,
remove it from the Git index before committing:

```powershell
git restore --staged "path\to\sensitive-file"
```
