# QADD v4.1 Windows run and freeze guide

Run these commands in PowerShell from the repository root:

```powershell
cd "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1"

.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests\test_qadd_v4.py -q
.\.venv\Scripts\python.exe scripts\generate_qadd_v4_1_notebook.py
.\.venv\Scripts\python.exe -m jupyter lab notebooks\02_feature_extraction\02a_additive_interference_QADD_v4_1_0.ipynb
```

Do not use bare `py`; on this computer it previously selected Python 3.14,
which is outside the repository's supported Python 3.11–3.12 range.

## First full run

In the notebook control cell set:

```python
RUN_COHORT_EXTRACTION = True
RUN_CODEC_ROUNDTRIP = True
RUN_PACKAGE_TESTS = False
PACKAGE_TESTS_CONFIRMED = True  # only after pytest passed in this environment
BUILD_GALLERY = True

PUBLISH_AND_FREEZE_QADD_V4_1 = False
PACKAGE_INTEGRATION_APPROVED = False
QADD_REVIEW_DECISION = "UNDECIDED"
QADD_REVIEWER = ""
QADD_REVIEW_RATIONALE = ""
```

Restart the kernel and run all cells. Inspect the final gate table and gallery.
The first run must remain a candidate run.

## Freeze run

Only if every blocking gate passes and you accept the gallery, change:

```python
PUBLISH_AND_FREEZE_QADD_V4_1 = True
QADD_REVIEW_DECISION = "ACCEPT_QADD_V4_1"
QADD_REVIEWER = "Nevena Musikic"
QADD_REVIEW_RATIONALE = (
    "Reviewed the label-blind QADD gallery, feature distributions, "
    "whole-pause deletion summaries, boundary-erosion summaries, and all "
    "blocking validation tables; accepted for the QADD measurement freeze."
)
```

Rerun from the control cell through the final cell in the same live kernel.
The final cell will create:

```text
MAIN outputs/02_FEATURE_FREEZE/additive_interference/qadd-v4.1.0/
```

The operation refuses to overwrite an existing freeze. Do not delete or replace
that directory casually; it is the immutable input for downstream analyses.

`PACKAGE_INTEGRATION_APPROVED` remains `False` until the frozen table is wired
into the central registry/assembly notebook. Integration is intentionally not a
prerequisite for freezing the validated measurement output.
