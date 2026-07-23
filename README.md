# Paper 1: quality control for remote speech biomarkers

This is a clean rebuild of the original Paper 1 pipeline. It keeps the familiar notebook sequence while moving every reusable algorithm into a tested Python package. The pipeline is a **measurement and validation study**, not an ALS diagnostic model.

The principal design decisions are:

- audit every workbook and media stream before cohort selection;
- keep native audio for clipping, dropout, codec, bandwidth, and channel evidence;
- create a separate 16-kHz mono view only for VAD and comparable frame analyses;
- treat each Q metric as an observed proxy with its own support/status, not as a reflective latent scale;
- never average heterogeneous metrics into an overall Q score without separate construct-validation evidence;
- retain one logical recording when WAV/WEBM rows describe the same recording;
- cluster inference by participant and prevent repeated recordings from becoming pseudoreplicates;
- estimate main perceptual alignment within RA and quantify four-RA agreement only in
  the crossed reliability subset before forming consensus;
- use no SMOTE, synthetic minority examples, or record-level train/test splitting.

## What is currently verifiable

The uploaded metadata workbooks were audited and the results are summarized in `reports/reference_audit/REFERENCE_METADATA_AUDIT.md`. The repository also passes the included synthetic checks for segmentation guards, sentinel handling, additive-noise behavior, hard clipping, dropouts, reverberation tails, and rater consensus.

The full empirical run remains intentionally unavailable in this environment because the
audio directories and complete annotations are local to the study computer. The code is
configured for the stated Windows root and can be run locally now. It recognizes rater
identity from the four declared RA folders, excludes `Reliability` from the distributed
main import, and separately validates the crossed 70-file reliability set.

## Windows setup

Use Python 3.11 and install FFmpeg/FFprobe on `PATH`.

```powershell
cd "C:\Users\musikicn\Desktop\Nevena_project\paper_1"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,reverb]"
Copy-Item config\project.example.yaml config\project.yaml
Copy-Item config\human_qc_schema.example.yaml config\human_qc_schema.yaml
ffmpeg -version
ffprobe -version
```

Review `config/project.yaml`. The example already points to:

```text
C:\Users\musikicn\Desktop\Nevena_project\Data_13072026
```

Do not change thresholds after inspecting associations with clinical outcomes or human labels. Any threshold change becomes a new analysis version and must be recorded.

## Execution order

```powershell
paper1-qc --config config/project.yaml audit
paper1-qc --config config/project.yaml inventory
paper1-qc --config config/project.yaml segment
paper1-qc --config config/project.yaml extract --profile primary
paper1-qc --config config/project.yaml assemble
paper1-qc --config config/project.yaml describe
```

After checking the four RA folder names, the nested `Reliability` layout, and the broad
label direction:

```powershell
paper1-qc --config config/project.yaml human-qc --schema config/human_qc_schema.yaml
```

Sensitivity run:

```powershell
paper1-qc --config config/project.yaml extract --profile conservative
paper1-qc --config config/project.yaml extract --profile permissive
paper1-qc --config config/project.yaml sensitivity
paper1-qc --config config/project.yaml rest-reference
paper1-qc --config config/project.yaml encoding-sensitivity
```

`Rest` is not pooled into the primary Bamboo cohort. The pipeline audits `Rest.xlsx` and
the Rest media inventory, then uses only exact participant/date/protocol/iteration
Bamboo–Rest pairs as a contextual acquisition sensitivity. Rest is summarized as a
whole-recording reference without speech VAD; unmatched or nearest-date Rest recordings
are excluded from this comparison.

The notebooks under `notebooks/` mirror the computational order and remain thin
orchestration layers. The separate `visualization/` folder contains long-form audit code,
intermediate figures, denominator tables, and paper-figure candidates for segmentation
plus four study goals. Change reusable algorithms in the package, registry, and tests
together; use the visualization notebooks to audit and present those saved results.

See `docs/GIT_AND_WINDOWS_RUN_GUIDE.md` for exact PowerShell, Git, and notebook instructions.

## Output contracts

| Stage | Required output | Gate before continuing |
|---|---|---|
| `00_audit` | audited rows, canonical recordings, issue ledgers, column profiles, media hashes/probes | resolve or disposition every error; freeze cohort provenance |
| `01_segmentation` | interval table with raw/primary/strict views and profiles; error ledger | inspect support distributions and a stratified visual sample |
| `02_features` | one metric row per logical Bamboo recording; family statuses; metric registry | zero silent failures; investigate impossible ranges and support failures |
| `03_dataset_assembly` | validated one-to-one merge; explicit eligibility columns | reproduce the participant/recording flow diagram from saved counts |
| `04_analysis` | clustered descriptive estimates, structure, persistence, rater-stratified main alignment, crossed-set four-RA agreement/consensus, paired detailed/2RA comparisons, sensitivity | verify denominators, confidence intervals, class support, direction/scale mapping, and blocked analyses |

Every command saves CSV (and Parquet when available), an error ledger where relevant, and a run manifest containing configuration/input hashes and package versions.

## Hard stops

Stop rather than improvise if any of the following occurs:

- the same filename resolves to zero or multiple disk paths;
- filename subject/date/task disagrees with metadata;
- diagnosis is inferred from an ID but has not been reviewed;
- a clinical sentinel or impossible score/date remains in an analysis field;
- independent rater identity is absent, the folder design is inconsistent, or crossed
  reliability exports appear adjudicated rather than independent;
- the class/category count is too small for the pre-specified estimand;
- VAD or feature extraction fails without a saved error row;
- a result requires treating WAV and WEBM copies or repeated visits as independent observations.

See `docs/SCIENTIFIC_MEASUREMENT_PROTOCOL.md`, `docs/STATISTICAL_ANALYSIS_PLAN.md`, and `docs/ORIGINAL_PIPELINE_AUDIT_AND_CHANGES.md` before modifying the analysis.
