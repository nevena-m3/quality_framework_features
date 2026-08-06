# QTEMP v0.2.0 Windows/Jupyter Run Guide

## Release status

This is a full measurement-development run. It is deliberately blocked from publication freeze and from copying features into `MAIN outputs/02_FEATURE_TABLES`.

## Required project inputs

The notebook expects the existing project configuration and frozen inputs used by the mature family notebooks:

- `config/project.yaml`
- `MAIN outputs/00_DATA_FREEZE`
- `MAIN outputs/01_SEGMENTATION_FREEZE`
- native source media resolvable from the frozen recording table
- optional frozen QDIST event ledger for clipping-edge arbitration

The input adapter searches versioned CSV/Parquet tables and fails explicitly when a required logical identifier, source path, native provenance field, or frozen task interval cannot be resolved. It does not silently substitute resampled analysis audio for native audio.

## Full-run controls

The notebook defaults to:

```python
RUN_PACKAGE_TESTS = True
RUN_SYNTHETIC_VALIDATION = True
RUN_REAL_SPEECH_INJECTION = True
RUN_SIGNAL_CHAIN_CHARACTERIZATION = True
RUN_COHORT_EXTRACTION = True
RUN_PARAMETER_SENSITIVITY = True
BUILD_GALLERY = True
PUBLISH_AND_FREEZE_QTEMP_V02 = False
```

Do not turn off scientific stages for a final run. Reduced flags are only for debugging.

## Recommended execution

1. Deploy the package into the project and run the QTEMP tests.
2. Open the generated notebook from the project root using the project `.venv` kernel.
3. Restart the kernel and run all cells from top to bottom.
4. Save the executed notebook without running the generator again afterward; the generator intentionally creates a clean, unexecuted canonical copy.
5. Zip the complete folder:
   `outputs/02_features/temporal_discontinuity/qtemp-v0.2.0-measurement-development`
6. Upload the executed notebook and output ZIP for detector-level interpretation.

## Expected output structure

```text
outputs/02_features/temporal_discontinuity/qtemp-v0.2.0-measurement-development/
├── tables/
│   ├── feature registry and parameters
│   ├── synthetic and real-speech validation tables
│   ├── signal-chain characterization
│   ├── recording features
│   ├── candidate/disposition/accepted-event/exposure ledgers
│   ├── reconstruction and sensitivity audits
│   ├── empirical prevalence, recurrence, redundancy, and suitability
│   └── computed gates and feature decisions
├── figures/
├── gallery/
│   ├── native-channel and review-channel audio clips
│   ├── waveform/spectrogram panels
│   ├── unblinded detector-evidence index
│   └── detector-label-blind adjudication sheet
├── audit/
│   ├── errors and provenance
│   ├── parameters and candidate manifest
│   └── SHA-256 output inventory
└── checkpoints/
```

## Manual adjudication

The reviewer should use `qtemp_v02_blinded_adjudication_sheet.csv`. This sheet intentionally omits diagnosis, human QC, detector type, disposition, score, and recording identifier. The unblinded index is used only after review to estimate event credibility by detector and confidence stratum.

Do not tune thresholds on the same adjudication sample. Any detector or threshold change creates a new measurement version and requires an independent review sample.

## Interpreting outcomes

- A detector can pass analytical tests but remain pending because G9 is incomplete.
- A detector with poor real-speech specificity should be revised, narrowed, made audit-only, or dropped.
- A detector with zero cohort events may remain technically valid but unsuitable for downstream continuous analysis.
- Full repository tests may contain stale tests for prior family versions; the authoritative deployment tests are `test_qtemp_v02.py` and `test_qtemp_notebook_v020.py`.
