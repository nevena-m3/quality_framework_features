# QGAIN v3.1.0 integration and run guide

## What this patch adds

- `src/paper1_qc/qgain.py`: authoritative four-feature estimator
- `tests/test_qgain_v31.py`: deterministic analytical tests
- `scripts/generate_qgain_v31_notebook.py`: reproducible notebook generator
- `notebooks/02_feature_extraction/02b_gain_dynamics_QGAIN_v3_1_0.ipynb`

QGAIN v3.1 is the finalized correction of v3.0. It preserves the four validated
continuous estimators and removes the invalid local-transition rate from the
analysis profile. It does not overwrite a legacy freeze.

## Install and verify in Windows PowerShell

From the repository root:

```powershell
Expand-Archive `
  "$env:USERPROFILE\Downloads\QGAIN_v3_1_0_repository_patch.zip" `
  -DestinationPath . `
  -Force

.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe -m pytest tests\test_qgain_v31.py -q

.\.venv\Scripts\python.exe scripts\generate_qgain_v31_notebook.py

.\.venv\Scripts\python.exe -m jupyter lab `
  notebooks\02_feature_extraction\02b_gain_dynamics_QGAIN_v3_1_0.ipynb
```

The package tests must pass before cohort extraction.

## Final notebook run

The notebook is configured for the approved final run:

```python
RUN_COHORT_EXTRACTION = True
RUN_CODEC_ROUNDTRIP = True
RUN_PACKAGE_TESTS = True
BUILD_GALLERY = True

PUBLISH_AND_FREEZE_QGAIN_V31 = True
QGAIN_REVIEW_DECISION = "ACCEPT_QGAIN_V31"
```

Restart the kernel and run all cells. The expected terminal notebook status is
**FROZEN - QGAIN v3.1 passed every blocking layer**. Any failed blocking gate
stops the freeze. Do not bypass a failed gate.

## Four analysis features

```text
qgain_typical_speech_level_dbfs
qgain_within_segment_iqr_db
qgain_between_segment_mad_db
qgain_abs_drift_db_per_min
```

The signed drift and its confidence interval, floor fractions, support tiers,
and raw estimates are audit companions. No scalar QGAIN score is created.

## Explicitly rejected metric

The v3.0 frame-local transition rate is not an analysis feature. It produced
pervasive candidates in real speech and visually followed ordinary phonetic and
prosodic transitions. QGAIN v3.1 retains only an exploratory ledger and a
negative-result audit table:

```text
qgain_v31_excluded_local_transition_metric.csv
qgain_v31_exploratory_local_transition_ledger.parquet
```

These diagnostics are excluded from the feature registry, empirical
correlations, central table, and downstream analysis.

## Authoritative output

The manuscript-facing recording table is:

```text
qgain_v31_analysis_features.csv
qgain_v31_analysis_features.parquet
```

After a valid freeze, the last cell copies this serialization pair to:

```text
MAIN outputs/02_FEATURE_TABLES/
```

Ledgers, validation tables, and galleries remain in the versioned family freeze
and are not copied into the central feature-table folder.

After the run, save the notebook so every code cell retains its execution count
and output. Zip the complete `qgain-v3.1.0` output directory for final archival
review.
