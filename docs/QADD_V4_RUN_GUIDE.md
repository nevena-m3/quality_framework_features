# QADD v4 candidate run guide

> Superseded by `QADD_V4_1_WINDOWS_RUN_AND_FREEZE.md`. Retained only as the
> historical v4.0 run record; do not use it for a new freeze.

## What changed

QADD v4 is a candidate measurement implementation for **extrinsic additive
acoustic interference**. The notebook imports the authoritative estimator from
`src/paper1_qc/qadd.py`; it does not contain a second algorithm copy.

The analysis vector is:

- primary: `qadd_pause_ac_level_dbfs_median`;
- secondary: `qadd_pause_level_iqr_db`;
- mixed secondary: `qadd_speech_pause_level_contrast_db`;
- non-ordinal descriptor: `qadd_pause_spectral_flatness`;
- targeted descriptor: `qadd_mains_hum_comb_score_db`.

The v2 upper-tail and 30-ms transient-rate measures are not part of the v4
confirmatory vector.

QADD v4 also:

- uses 80–7000 Hz flatness, because the 7000–7500 Hz edge created a large
  MP3-dependent shift in the prespecified codec control;
- uses a ±2 Hz hum tone band, consistent with the 2 Hz DFT-bin spacing of the
  500 ms windows;
- requires score-above-null plus at least three supported low-order harmonics
  for positive hum evidence, while retaining the raw comb score as the feature;
- applies digital-floor censoring to level estimands, while spectral
  descriptors use their own valid non-floor-window support;
- contains no human-QC association or gate; that belongs to Goal 3.

## Before opening the notebook

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_qadd_v4.py -q
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

Cohort extraction defaults to `False` even when `config/project.yaml` exists.
This prevents accidental partial runs when frozen outputs are not mounted.
Deterministic, synthetic, censoring, and codec controls still run.

## Run

Open:

`notebooks/02_feature_extraction/02a_additive_interference_QADD_v4_0_0.ipynb`

Then:

1. confirm the run controls and set `RUN_COHORT_EXTRACTION = True` only when the frozen inputs are mounted;
2. run all cells;
3. resolve every failed blocking check instead of changing its threshold;
4. inspect the recording gallery without diagnosis or outcome labels;
5. add the reviewer name, decision, and rationale only after review;
6. keep `PUBLISH_AND_FREEZE_QADD_V4 = False` until every gate passes.

## Outputs

The notebook writes to:

`outputs/02_features/additive_interference/qadd-v4.0.0/`

Subfolders contain:

- matched CSV/Parquet tables;
- frame, pause-interval, and spectral-window ledgers;
- SVG, PDF, and 600-dpi PNG publication figures with caption/alt-text sidecars;
- a label-blind audit gallery and selection index;
- candidate or frozen provenance manifest.

## Important current boundary

The v4 module and notebook are intentionally not yet wired into the legacy
central metric registry/CLI. Integration is a final gate after the full cohort,
support/boundary, empirical, and gallery-review checks pass. Human-QC
correspondence is evaluated later under Goal 3 and is not a feature-freeze
gate. Until then, the notebook manifest must remain `candidate_only`.
