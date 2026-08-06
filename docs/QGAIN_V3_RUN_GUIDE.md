# QGAIN v3.0.0 integration and run guide

## What this patch adds

- `src/paper1_qc/qgain.py`: authoritative five-feature estimator
- `tests/test_qgain_v3.py`: deterministic analytical tests
- `scripts/generate_qgain_v3_notebook.py`: reproducible notebook generator
- `notebooks/02_feature_extraction/02b_gain_dynamics_QGAIN_v3_0_0.ipynb`

QGAIN v3 is an incompatible scientific redesign of the legacy v2 notebook. It
does not modify or overwrite a legacy freeze.

## Install and verify in Windows PowerShell

From the repository root:

```powershell
Expand-Archive `
  "$env:USERPROFILE\Downloads\QGAIN_v3_0_0_repository_patch.zip" `
  -DestinationPath . `
  -Force

.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe -m pytest tests\test_qgain_v3.py -q

.\.venv\Scripts\python.exe scripts\generate_qgain_v3_notebook.py

.\.venv\Scripts\python.exe -m jupyter lab `
  notebooks\02_feature_extraction\02b_gain_dynamics_QGAIN_v3_0_0.ipynb
```

The package tests must pass before cohort extraction.

## First notebook run

Keep these controls:

```python
RUN_COHORT_EXTRACTION = True
RUN_CODEC_ROUNDTRIP = True
RUN_PACKAGE_TESTS = True
BUILD_GALLERY = True

PUBLISH_AND_FREEZE_QGAIN_V3 = False
STEP_EVENT_VALIDATION_APPROVED = False
QGAIN_REVIEW_DECISION = "UNDECIDED"
```

Restart the kernel and run all cells. The expected state is **CANDIDATE ONLY**.
This is deliberate. Upload the executed notebook, the complete
`outputs/02_features/gain_dynamics/qgain-v3.0.0` folder, and the fixed event
review table:

```text
tables/qgain_v3_fixed_step_event_review_sample.csv
```

Do not approve the event detector merely because the notebook ran. Candidate
events require waveform/gallery review for specificity.

## Five analysis features

```text
qgain_typical_speech_level_dbfs
qgain_within_segment_iqr_db
qgain_between_segment_mad_db
qgain_abs_drift_db_per_min
qgain_sustained_step_rate_per_min
```

The signed drift, counts, exposure, confidence intervals, floor fractions,
support tiers, and raw estimates are audit companions. No scalar QGAIN score is
created.

## Event-review completion

After reviewing the fixed sample, populate its `adjudication` and
`reviewer_notes` columns. Approval requires a documented scientific decision;
it is not automatic. If specificity is inadequate, revise the detector and
create a new measurement version.

Only after every blocking gate passes should these controls be populated:

```python
STEP_EVENT_VALIDATION_APPROVED = True
STEP_EVENT_REVIEWER = "..."
STEP_EVENT_REVIEW_RATIONALE = "..."

QGAIN_REVIEW_DECISION = "ACCEPT_QGAIN_V3"
QGAIN_REVIEWER = "..."
QGAIN_REVIEW_RATIONALE = "..."

PUBLISH_AND_FREEZE_QGAIN_V3 = True
```

Rerun the entire notebook from a clean kernel. The freeze cell refuses failed
gates and refuses to overwrite an existing version.

## Authoritative output

The recording-level family table is:

```text
qgain_v3_recording_features.csv
qgain_v3_recording_features.parquet
```

After a valid freeze, the last cell copies this serialization pair to:

```text
MAIN outputs/02_FEATURE_TABLES/
```

Ledgers, validation tables, and galleries remain in the versioned family freeze
and are not copied into the central feature-table folder.

