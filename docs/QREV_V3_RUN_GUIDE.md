# QREV v3.0 integration and run guide

## Patch contents

- `src/paper1_qc/qrev.py` - authoritative four-feature estimator
- `src/paper1_qc/_vendor/srmrpy/` - pinned MIT-licensed SRMRpy implementation
- `tests/test_qrev_v3.py` and `tests/fixtures/srmrpy/`
- `scripts/generate_qrev_v3_notebook.py`
- `notebooks/02_feature_extraction/02c_reverberation_QREV_v3_0_0.ipynb`
- `pyproject.toml` - lightweight pinned `reverb` extra

## Windows PowerShell

From the repository root:

```powershell
Expand-Archive `
  "$env:USERPROFILE\Downloads\QREV_v3_0_0_repository_patch.zip" `
  -DestinationPath . `
  -Force

.\.venv\Scripts\python.exe -m pip install -e ".[dev,reverb]"

.\.venv\Scripts\python.exe -m pytest tests\test_qrev_v3.py -q

.\.venv\Scripts\python.exe scripts\generate_qrev_v3_notebook.py

.\.venv\Scripts\python.exe -m jupyter lab `
  notebooks\02_feature_extraction\02c_reverberation_QREV_v3_0_0.ipynb
```

Restart the notebook kernel and run all cells.

## First run

The notebook intentionally starts as a candidate:

```python
PUBLISH_AND_FREEZE_QREV_V30 = False
QREV_REVIEW_DECISION = "PENDING"
```

This runs every analytical and cohort validation without freezing. Save the
executed notebook and zip:

```text
outputs/02_features/reverberation/qrev-v3.0.0/
```

Review those artifacts before changing the review decision.

## Freeze run

Only after the candidate output and gallery are accepted:

```python
PUBLISH_AND_FREEZE_QREV_V30 = True
QREV_REVIEW_DECISION = "ACCEPT_QREV_V30"
QREV_REVIEWER = "Nevena Musikic"
QREV_REVIEW_RATIONALE = "..."
```

Restart and run all cells. Any failed blocking gate stops the freeze. Do not
edit gate results or create exceptions.

The manuscript-facing table bundle is:

```text
qrev_v30_analysis_features.csv
qrev_v30_analysis_features.parquet
```

After a valid freeze, it is copied with hash verification to:

```text
MAIN outputs/02_FEATURE_TABLES/
```

The output freeze retains boundary ledgers, support/censoring fields,
validation tables, figures, galleries, parameters, implementation hashes, and
the exact SRMR implementation record.
