# Quality Framework Features Pipeline

Auditable quality-control measurement pipeline for remote ALS speech recordings. The repository takes raw metadata, media, segmentation decisions, and optional human quality-control annotations through governed feature extraction, reviewed feature freezes, dataset assembly, and study analyses.

The pipeline is designed around traceability:

- Raw participant data and local configuration remain outside Git.
- Intermediate results are separated from authoritative freezes.
- Every reviewed feature family has an explicit preflight, extraction, verification, and/or finalization sequence.
- Missing, unavailable, and measured-zero values remain distinct.
- Review gates must be completed before downstream publication artifacts are treated as final.
- Feature measurements describe observable recording behavior; they are not standalone clinical biomarkers or automatic recording accept/reject rules.

## Pipeline at a glance

```text
Local metadata + audio
        |
        v
00 Environment, metadata, and media audit
        |
        v
01 Silero segmentation + mandatory adjudication
        |
        v
02 Reviewed feature families
   QGAIN -> QADD -> QREV -> QCHAN -> QDIST -> QTEMP
        |
        v
03 Audited analysis-dataset assembly
        |
        v
04 Study Goals 1-4 + sensitivity analysis
        |
        v
05 Optional standalone human-QC reliability report
```

## Repository layout

```text
config/                          Portable configuration templates
notebooks/                       Ordered pipeline notebooks
  00_setup/                      Environment and data audit
  01_segmentation/               Segmentation, review, and freeze
  02_feature_extraction/         One numbered folder per feature family
  03_dataset_assembly/           Analysis-dataset construction and accounting
  04_analysis/                   Study-goal and sensitivity analyses
  05_human_QC_reliability/       Standalone four-rater reliability analysis
scripts/                         Execution, finalization, and freeze helpers
src/paper1_qc/                   Shared pipeline implementation
src/paper1_qc_reviewed/          Reviewed feature-family implementations
tests/                           Automated feature and freeze-contract tests
outputs/                         Mutable intermediate and analytical outputs
MAIN outputs/                    Authoritative freezes and reviewed features
```

Generated outputs, participant data, local configuration, executed notebooks, archives, and media are excluded from Git. The GitHub repository contains the code and portable templates needed to reproduce the workflow, not protected study data or generated result bundles.

## Requirements

- Windows PowerShell is recommended because the freeze helpers are `.ps1` scripts.
- Python 3.11 is the primary supported interpreter; Python 3.12 is also permitted by the package metadata.
- FFmpeg and FFprobe must be installed and available on `PATH` unless explicit executable paths are supplied in the local configuration.
- Sufficient disk space for decoded media, segmentation artifacts, feature candidates, figures, and immutable freezes.
- Access to the study metadata, audio, and—when Goal 4 is required—human-QC annotation exports.

The Python dependencies include NumPy, pandas, SciPy, statsmodels, SoundFile, PyArrow, Matplotlib, seaborn, Silero VAD, ONNX Runtime, JupyterLab, and pytest. The optional `reverb` dependency installs `gammatone` for reverberation-related processing.

## Installation

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,reverb]"
```

Confirm the installation:

```powershell
python -c "import paper1_qc; print('paper1_qc import OK')"
ffmpeg -version
ffprobe -version
python -m pytest
```

The full repository test suite should complete before running study data.

## Local configuration

Create local configuration files from the tracked templates:

```powershell
Copy-Item config\project.example.yaml config\project.yaml
Copy-Item config\human_qc_schema.example.yaml config\human_qc_schema.yaml
Copy-Item config\human_qc_manifest.example.csv config\human_qc_manifest.csv
```

Only `config/project.yaml` is required for the main metadata, segmentation, feature, and analysis pipeline. The human-QC files are required for Goal 4 and the standalone reliability workflow.

### Configure the study paths

Edit `config/project.yaml` and set `paths.data_root` to the local study-data directory. Relative entries such as `Bamboo_passage_only` are resolved beneath `data_root`.

Expected data layout:

```text
<data_root>/
  Bamboo_passage_only/          Bamboo-passage media
  Rest_only/                    Rest-task media
  Rest_Bamboo/                  Combined or paired media
  Bamboo_passage_HumanQC/       Detailed annotation exports
  Bamboo.xlsx                   Bamboo metadata
  Rest.xlsx                     Rest metadata
  Rest_Bamboo.xlsx              Combined metadata
```

The example uses Windows paths enclosed in single quotes so YAML treats backslashes literally. Do not commit the local configuration: it may contain participant paths, identifiers, and adjudication decisions.

### Review scientific settings

Before the first run, review these configuration sections:

- `data_freeze`: cohort version and diagnosis-adjudication evidence.
- `vad`: Silero version, sampling rate, boundary rules, and sensitivity profiles.
- `segmentation_review`: mandatory review-queue and guardrail settings.
- `segmentation_freeze`: immutable segmentation version.
- `metrics`: frame, hop, support, pause, and reverberation settings.
- `analysis`: bootstrap count, alpha, FDR method, and minimum support.
- `clinical_alignment`: assessment-window and sentinel-value handling.

Version values identify immutable outputs. Do not reuse an existing version for changed scientific inputs or decisions.

### Human-QC configuration

Review every mapping in `config/human_qc_schema.yaml` before Goal 4. The expected annotation design separates:

- distributed main ratings, where raters may annotate different recordings; and
- the crossed `Reliability/` subset, where the same recordings are independently rated by all expected raters.

These designs must not be pooled as if every recording had four ratings. Confirm rater directory names, family mappings, the interval time base, and the direction of broad metadata labels.

## How to run notebooks

Start JupyterLab from the repository root so relative project discovery is consistent:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
jupyter lab
```

Run each notebook from top to bottom. Read its opening scientific contract and its final decision cell. A notebook completing without a Python exception does not automatically mean a review or publication gate passed.

Recommended operating rules:

1. Keep `config/project.yaml` under local version control or a secure run log, but never commit it to this repository.
2. Do not edit frozen files in place.
3. Save executed notebooks locally for provenance; they are Git-ignored.
4. Inspect validation tables, figures, manifests, and gate summaries before proceeding.
5. Use a new declared version when scientific inputs, parameters, or adjudications change.
6. Run `python -m pytest` after code or path changes.

## End-to-end notebook guide

### Stage 00 — Environment and data freeze

| Order | Notebook | Purpose | Principal output / next decision |
|---|---|---|---|
| 1 | `notebooks/00_setup/00_environment_check.ipynb` | Checks Python, configuration, package imports, FFmpeg, and FFprobe. | `outputs/00_environment/tables/environment_checks.csv` plus a pass/fail figure. Fix all failed checks before participant-data processing. |
| 2 | `notebooks/00_setup/00_metadata_and_media_audit.ipynb` | Reconciles metadata and media, profiles the cohort, generates adjudication templates, and creates the immutable data freeze. | Audit artifacts under `outputs/00_audit/`; authoritative cohort files under `MAIN outputs/00_DATA_FREEZE/<version>/`. Resolve every diagnosis and media-path issue before freezing. |

Metadata adjudication is a manual gate. Investigator-confirmed exceptional control IDs belong in local configuration with evidence. Remaining cases are completed in `config/metadata_adjudication.csv`. Media must resolve to one decodable preferred path.

### Stage 01 — Segmentation and boundary review

| Order | Notebook | Purpose | Principal output / next decision |
|---|---|---|---|
| 3 | `notebooks/01_segmentation/01_segmentation_silero_full_dataset.ipynb` | Runs version-pinned Silero VAD, builds distinct segmentation views, creates the mandatory review queue, applies adjudication/manual overrides, and freezes accepted intervals. | Working artifacts under `outputs/01_segmentation/`; review publication under `outputs/01_segmentation_after_review/`; authoritative decisions and intervals under `MAIN outputs/01_SEGMENTATION_FREEZE/<version>/`. |

Important segmentation outputs include:

- `frozen_segmentation_decisions.csv`: recording-level final decisions.
- `frozen_segmentation_intervals.csv`: authoritative sample-index interval ledger.
- `outputs/01_segmentation/segmentation/silero/`: per-recording segment/frame tables.
- `outputs/01_segmentation/figures/segmentation/silero/{accepted,flagged,excluded}/`: visual QC.
- local segmentation adjudication and manual-boundary templates under `config/`.

A low boundary-contrast flag prompts inspection; it is not automatically an error and does not itself move a boundary.

### Stage 02 — Reviewed feature extraction

Run feature families in folder order. Within a family, run notebooks in numeric order. Each family writes mutable review-stage work beneath:

```text
MAIN outputs/02_FEATURE_REVIEWED/00_working_candidates/<feature_family>/
```

Approved artifacts are published to numbered sibling directories such as `06_family_freezes`, `07_figure_packages`, and `08_validation_workbooks`. The family’s `support/` folder contains its reviewed scientific contract, checklist, decision record, audit, or workbook.

#### 01_QGAIN — Recorded level and gain dynamics

| Notebook | What it does | Where to look |
|---|---|---|
| `01_extract.ipynb` | Extracts and validates the reviewed recorded-level and level-dynamics candidate while preserving missingness and support. | `.../00_working_candidates/gain_dynamics/` |
| `02_figures.ipynb` | Builds the standardized scientific figure supplement from the immutable QGAIN freeze; it does not recompute feature values. | Candidate under `gain_dynamics/`; sealed package under `.../07_figure_packages/gain_dynamics/`. |
| `03_finalize.ipynb` | Verifies source artifacts, assigns final roles, adds sensitivity/uncertainty evidence, and creates a freeze-ready candidate without changing the numerical estimators. | Candidate under `gain_dynamics/`; final freeze via `scripts/freeze_qgain_v410.ps1`. |

After successful review, use `scripts/freeze_qgain_v410.ps1` and `scripts/freeze_qgain_figure_package_v100.ps1` as directed by the notebooks.

#### 02_QADD — Additive interference

| Notebook | What it does | Where to look |
|---|---|---|
| `01_preflight.ipynb` | Runs controlled G1-G5 analytical validation of pause-region additive-interference measurements and support logic. | `.../00_working_candidates/additive_interference/` |
| `02_extract_cohort.ipynb` | Runs the corrected cohort extraction, null calibration, support/robustness audits, empirical summaries, and non-imputed ML export. | `.../additive_interference/qadd-v4.2.0-candidate/` |
| `03_finalize.ipynb` | Verifies the candidate, preserves numerical columns, applies final roles, completes validation records/passports, and prepares the freeze. | Final candidate under `additive_interference/`; publish with `scripts/freeze_qadd_v420.ps1`. |

Use `scripts/freeze_qadd_figure_package_v100.ps1` for the approved figure package when the notebook’s review conditions are satisfied.

#### 03_QREV — Reverberation and residual tails

| Notebook | What it does | Where to look |
|---|---|---|
| `01_preflight.ipynb` | Validates the residual-tail construct, natural-boundary contract, and SRMR support usage before cohort extraction. | `.../00_working_candidates/reverberation/` |
| `02_extract_cohort.ipynb` | Runs corrected cohort extraction and empirical validation with standardized G1-G10 evidence and Panels A-J. | `.../reverberation/qrev-v4.0.0-candidate/` |
| `03_finalize.ipynb` | Verifies the candidate, applies accepted roles, rebuilds audited evidence from saved ledgers, and prepares the immutable freeze. | Final candidate under `reverberation/`; publish with `scripts/freeze_qrev_v400.ps1`. |

Use `scripts/freeze_qrev_figure_package_v100.ps1` for the approved figure package.

#### 04_QCHAN — Channel/device spectral effects

| Notebook | What it does | Where to look |
|---|---|---|
| `01_preflight.ipynb` | Runs controlled validation of reference-relative spectral deviation and attenuation proxies. Cohort extraction and freezing remain disabled here. | `.../00_working_candidates/channel_device/` |
| `02_extract_cohort.ipynb` | Runs cohort extraction, reference construction, support/uncertainty audits, standardized panels, and review packaging. | `.../channel_device/qchan-v4.0.0-candidate/` |
| `03_finalize.ipynb` | Verifies the candidate and saved spectra, applies accepted roles, adds uncertainty, regenerates audited figures, and prepares the freeze. | Final candidate under `channel_device/`; publish with `scripts/freeze_qchan_v400.ps1`. |

Use `scripts/freeze_qchan_figure_package_v100.ps1` for the approved figure package.

#### 05_QDIST — Visible hard-plateau morphology

| Notebook | What it does | Where to look |
|---|---|---|
| `01_remediation_preflight.ipynb` | Tests the corrected hard-plateau detector against synthetic and algebraic checks. Passing is not permission to freeze. | `.../00_working_candidates/nonlinear_distortion/qdist_v410_remediation_preflight/` |
| `02_extract_candidate_cohort.ipynb` | Long-running native-audio cohort recomputation with checkpoints, parameter sensitivity, reconstruction, signal challenges, and blinded-review package generation. | `.../nonlinear_distortion/qdist_v410_candidate_cohort/` |
| `03_verify_computational.ipynb` | Verifies hashes and known-truth/signal-evidence records, assigns deterministic decisions, and builds corrected validation panels without inventing human labels. | `.../qdist_v410_candidate_cohort/computational_verification_v1/` |
| `04_human_review_optional.ipynb` | Optional adjudication after two independent reviewers complete every blinded item; reports agreement and disagreements but does not freeze. | The candidate cohort’s `blind_review/` and adjudication outputs. |
| `05_finalize.ipynb` | Performs automated reviewer-free reference validation, applies feature-role gates, and prepares the immutable measurement freeze. | Publish with `scripts/freeze_qdist_v410_r1.ps1` after all required gates pass. |

The QDIST candidate-cohort run is restartable when its notebook’s `RESUME=True` contract is followed. Do not describe operational detector positives as human-confirmed clipping.

#### 06_QTEMP — Temporal discontinuity manifestations

| Notebook | What it does | Where to look |
|---|---|---|
| `01_extract_reviewed.ipynb` | Creates the reviewed candidate evidence for dropout/frozen-audio manifestations. QTEMP remains candidate-only where G9/G10 are not satisfied. | `.../00_working_candidates/temporal_discontinuity/` |
| `02_finalize_disposition.ipynb` | Creates or verifies the final analytical disposition. No QTEMP feature enters the validated primary feature set; permitted exploratory outputs remain documented. | `.../temporal_discontinuity/qtemp-v1.0.0-analytical-final-no-retained/` |

On Windows, `RUN_QTEMP_FINALIZE.cmd` executes the final notebook and checks the expected output. Additional historical QTEMP evidence is preserved beneath `temporal_discontinuity/_archive/`.

### Stage 03 — Dataset assembly

| Order | Notebook | Purpose | Principal output |
|---|---|---|---|
| 4 | `notebooks/03_dataset_assembly/03a_assemble_analysis_dataset.ipynb` | Performs validated one-to-one merges and creates explicit measurement, diagnosis, segmentation, clinical, and eligibility gates. | `outputs/03_dataset_assembly/paper1_analysis_dataset.csv`, plus audit tables and figures. |
| 5 | `notebooks/03_dataset_assembly/03b_dataset_statistics.ipynb` | Produces frozen cohort denominators and feature missingness accounting from the assembled dataset. | `outputs/03_dataset_assembly/statistics/tables/` and `figures/`. |

Investigate merge failures, duplicate keys, extraction failures, and unexpected task-completion missingness before analysis. Manuscript denominators should be generated from these outputs, never typed manually.

### Stage 04 — Study analyses

| Order | Notebook | Purpose | Output folder |
|---|---|---|---|
| 6 | `05_study_goal_1_acquisition_variability.ipynb` | Participant-clustered descriptives and exploratory participant-level ALS/control contrasts. These are acquisition/confounding patterns, not diagnostic performance. | `outputs/04_analysis/goal1/{tables,figures}/` |
| 7 | `06_study_goal_2_participant_persistence.ipynb` | Participant-level variance persistence and model-status accounting. Do not label it test-retest reliability. | `outputs/04_analysis/goal2/{tables,figures}/` |
| 8 | `07_study_goal_3_internal_structure_and_robustness.ipynb` | Participant-clustered feature structure with pairwise support preserved. | `outputs/04_analysis/goal3/{tables,figures}/` |
| 9 | `08_study_goal_4_perceptual_family_alignment.ipynb` | Separates distributed ratings, crossed four-rater reliability, consensus alignment, and broad two-rater comparisons. | `outputs/04_analysis/goal4/{tables,figures}/` |
| 10 | `09_sensitivity_summary.ipynb` | Summarizes segmentation-profile, encoding, participant-sampling, and Rest-context robustness. | `outputs/04_analysis/sensitivity_summary/{tables,figures}/` |

Some analysis notebooks consume intermediate tables produced by the package CLI, including descriptive, human-QC, segmentation-sensitivity, encoding-sensitivity, and Rest-reference outputs. If a required input is missing, run the corresponding upstream notebook or CLI stage rather than fabricating a placeholder.

### Stage 05 — Standalone human-QC reliability

`notebooks/05_human_QC_reliability/Bamboo_RA_Interrater_Reliability_FULL.ipynb` evaluates:

1. family presence;
2. specific source/item presence; and
3. one-to-one temporal interval alignment.

It validates annotation files, reconstructs continuous events, reports coverage, computes overall/family/item/pairwise reliability, audits rater detection rates, lists temporal matches and disagreement priorities, generates figures, and writes manuscript-ready text.

Its output directory is:

```text
notebooks/05_human_QC_reliability/_reliability_analysis_outputs/
```

Key files include `00_run_manifest.json`, inventories and coverage tables, event ledgers, `07_overall_presence_reliability.csv` through `15_disagreement_priority.csv`, `16_manuscript_text.txt`, and `figures/`. Duplicate exports for the same rater/recording are blocking by default.

## Understanding output locations

### `outputs/` — mutable intermediate work

This tree contains diagnostics, audit tables, per-recording processing artifacts, assembled data, and analytical results. It can be regenerated and is not committed.

Common locations:

```text
outputs/00_environment/             Environment report
outputs/00_audit/                   Metadata/media reconciliation
outputs/01_segmentation/            Working segmentation artifacts and QC
outputs/01_segmentation_after_review/ Reviewed segmentation publication tree
outputs/02_features/                General feature-extraction intermediates
outputs/03_dataset_assembly/        Analysis dataset and accounting
outputs/04_analysis/                Study analyses and sensitivity results
```

Most stages use `tables/` and `figures/` subdirectories. Manifests, logs, provenance, validation, and per-recording artifacts may appear where required by the stage contract.

### `MAIN outputs/` — authoritative and reviewed artifacts

```text
MAIN outputs/
  00_DATA_FREEZE/                   Immutable audited cohort
  01_SEGMENTATION_FREEZE/           Immutable accepted segmentation
  02_FEATURE_FAMILY_SNAPSHOTS/     Feature-family snapshots
  02_FEATURE_FREEZE/                Approved feature freeze
  02_FEATURE_TABLES/                Main feature tables
  02_FEATURE_TABLES_EXPLORATORY/   Explicitly exploratory feature tables
  02_FEATURE_REVIEWED/              Reviewed feature workflow and products
```

The reviewed feature area contains:

```text
02_FEATURE_REVIEWED/
  00_working_candidates/           Mutable candidates grouped by family
  00_feature_registry/             Reviewed registry
  01_analysis_features/            Reviewed analysis-feature tables
  02_support_and_availability/     Availability/support artifacts
  03_event_summaries/              Event summaries
  04_model_ready_features/         Non-imputed model-facing features
  05_feature_passports/            Feature-level scientific passports
  06_family_freezes/               Approved immutable family freezes
  07_figure_packages/              Approved figure packages
  08_validation_workbooks/         Reviewed validation workbooks
```

Only `00_working_candidates/` is intended as mutable review-stage workspace. Treat approved freezes and packages as immutable. See `MAIN outputs/README.md` for the concise on-disk index.

## Command-line interface

The installed `paper1-qc` command exposes the reusable pipeline stages:

```powershell
paper1-qc --help
paper1-qc --config config\project.yaml <command> --help
```

Available commands:

| Command | Role |
|---|---|
| `audit` | Audit and reconcile metadata workbooks. |
| `inventory` | Probe and reconcile media files. |
| `freeze-template` | Create the diagnosis-adjudication template without overwriting it. |
| `freeze` | Create the versioned cohort freeze after audit and adjudication. |
| `segment` | Run version-pinned Silero and create segmentation views. |
| `segment-template` | Create or refresh segmentation review and manual-boundary tables. |
| `segment-adjudicate` | Validate decisions/overrides and freeze final intervals. |
| `extract` | Extract registered quality metrics. |
| `assemble` | Merge audited metadata and metrics with explicit eligibility gates. |
| `rest-reference` | Extract exact-session Rest context. |
| `encoding-sensitivity` | Re-extract paired WAV/WEBM technical replicates. |
| `describe` | Run participant-clustered descriptive inference. |
| `sensitivity` | Compare segmentation sensitivity profiles. |
| `broad-qc` | Run the legacy merged broad-metadata QC analysis. |
| `human-qc` | Audit detailed annotations and run family-level alignment. |

Use each command’s `--help` output for its exact arguments. The notebooks remain the recommended guided interface because they include scientific interpretation, visual checks, and explicit decisions around the underlying commands.

## Validation and reproducibility

Run automated tests from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
python -m pytest
```

Optional static checking:

```powershell
python -m ruff check src tests scripts
```

For a defensible run, retain locally:

- the exact Git commit;
- `config/project.yaml` and adjudication files in secure storage;
- executed notebooks;
- run and freeze manifests;
- package/environment versions;
- input and output hashes;
- review checklists and final decision records; and
- the immutable data, segmentation, and feature freeze versions.

Do not manually edit manifest-listed files after sealing. A changed input, implementation, parameter, or adjudication requires a new versioned run.

## Troubleshooting

### Project root cannot be found

Start JupyterLab in the repository root and ensure `config/project.yaml` exists. For QTEMP, the `QTEMP_PROJECT_ROOT` environment variable can provide an explicit root when required.

### FFmpeg or FFprobe is missing

Add the executables to `PATH`, or set explicit values under `software.ffmpeg` and `software.ffprobe` in `config/project.yaml`. Re-run the environment notebook.

### A notebook reports a missing upstream table or freeze

Do not create an empty substitute. Confirm that the upstream stage completed, its version matches configuration, and the expected manifest/freeze directory exists.

### A freeze script refuses to publish

This is intentional when tests, hashes, required files, gate states, or destination-safety checks fail. Review the candidate’s validation output and the family support contract. Do not bypass the check or copy candidate files manually into a freeze directory.

### A candidate directory already exists

Follow the notebook’s resume/overwrite contract. QDIST provides a content-addressed resume workflow. Other families may require a new version or deliberate removal/archival of an incomplete local candidate; never overwrite an approved freeze.

### Human-QC coverage is lower than expected

Verify rater folder names, the `Reliability/` layout, filename-to-recording resolution, duplicate exports, and the schema’s interval time base. Do not interpret distributed main ratings as a fully crossed design.

### Results contain missing values

Inspect the associated status, availability, support, and provenance columns. The pipeline deliberately does not impute unsupported feature measurements merely to obtain a complete matrix.

## Data protection and Git policy

Never commit:

- participant audio or metadata workbooks;
- human-QC exports;
- local paths or participant identifiers;
- local configuration and adjudication files;
- generated `outputs/` or `MAIN outputs/` data;
- executed notebooks;
- virtual environments, caches, backups, or archives.

The `.gitignore` enforces these exclusions. `MAIN outputs/README.md` is the only tracked file inside `MAIN outputs/`.

## Scientific interpretation boundaries

- Quality features measure observable properties of stored recordings and declared analysis views.
- They do not uniquely identify a device, codec, acquisition stage, network failure, physiological mechanism, or clinical condition unless a separate validated study establishes that claim.
- Diagnosis-associated differences are not diagnostic classifier performance.
- Participant persistence is not automatically test-retest reliability.
- Operational detector positives are not human-confirmed events.
- A passed computational or synthetic preflight is not equivalent to cohort validity or publication approval.
- Sparse and unsupported estimates must remain visible; they should not be hidden through imputation or selective reporting.

These boundaries are part of the pipeline contract and should be preserved in downstream reports and manuscripts.
