# QADD v3 candidate run guide

## What changed

QADD v3 is a candidate measurement implementation for **extrinsic additive
acoustic interference**. The notebook imports the authoritative estimator from
`src/paper1_qc/qadd.py`; it does not contain a second algorithm copy.

The analysis vector is:

- primary: `qadd_pause_ac_level_dbfs_median`;
- secondary: `qadd_pause_level_iqr_db`;
- mixed secondary: `qadd_speech_pause_level_contrast_db`;
- non-ordinal descriptor: `qadd_pause_spectral_flatness`;
- targeted descriptor: `qadd_mains_hum_comb_score_db`.

The v2 upper-tail and 30-ms transient-rate measures are not part of the v3
confirmatory vector.

## Before opening the notebook

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_qadd_v3.py -q
```

For the full regression suite:

```bash
python -m pytest -q
```

The tests must pass before setting `PACKAGE_TESTS_CONFIRMED = True` in the
notebook.

## Required cohort inputs

The notebook expects:

- `config/project.yaml`;
- the configured frozen data folder under `MAIN outputs/00_DATA_FREEZE/`;
- the configured frozen segmentation folder under
  `MAIN outputs/01_SEGMENTATION_FREEZE/`;
- one selected media path per eligible recording;
- primary speech, strict speech, and strict internal-nonspeech interval views.

If `config/project.yaml` is absent, cohort extraction remains off while all
deterministic, synthetic, censoring, and codec controls still run.

## Run

Open:

`notebooks/02_feature_extraction/02a_additive_interference_QADD_v3_0_0.ipynb`

Then:

1. confirm the run controls and human-QC path in section 0;
2. run all cells;
3. resolve every failed blocking check instead of changing its threshold;
4. inspect the recording gallery without diagnosis or outcome labels;
5. add the reviewer name, decision, and rationale only after review;
6. keep `PUBLISH_AND_FREEZE_QADD_V3 = False` until every gate passes.

## Outputs

The notebook writes to:

`outputs/02_features/additive_interference/qadd-v3.0.0/`

Subfolders contain:

- matched CSV/Parquet tables;
- frame, pause-interval, and spectral-window ledgers;
- SVG, PDF, and 600-dpi PNG publication figures with caption/alt-text sidecars;
- a label-blind audit gallery and selection index;
- candidate or frozen provenance manifest.

## Important current boundary

The v3 module and notebook are intentionally not yet wired into the legacy
central metric registry/CLI. Integration is a final gate after the full cohort,
support/boundary, external-validity, and gallery-review checks pass. Until then,
the notebook manifest must remain `candidate_only`.
