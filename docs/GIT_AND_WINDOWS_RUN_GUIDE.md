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
paper1-qc --config config/project.yaml segment
paper1-qc --config config/project.yaml extract --profile primary
paper1-qc --config config/project.yaml assemble
paper1-qc --config config/project.yaml describe
```

Stop and review the corresponding error/issue tables after every command. Do not proceed
past unresolved metadata errors, ambiguous file paths, failed decoding, or unexplained
support failures.

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
