# QCHAN v3.0.1 release notes

## Release decision

QCHAN v3.0.1 is a patch release of the validated four-feature QCHAN design. The retained estimands and numerical estimator parameters are unchanged from v3.0.0. The version changes because the executable measurement contract, output provenance, and validation artifacts changed.

## Required repairs included

1. **Task schema corrected.** `protocol` is treated as a numeric protocol identifier, not as an alias for the speech-task label. Genuine task aliases (`task_stratum`, `task_name`, `task`, `protocol_task`) are still cross-checked for agreement.
2. **Frozen vintages pinned.** `DATA_FREEZE_VERSION = "v1"` and `SEGMENTATION_FREEZE_VERSION = "v1"` are explicit notebook constants. A conflicting value in `project.yaml` now fails early.
3. **Input contract strengthened.** The schema audit records `protocol` separately, and a blocking gate confirms it was not used as `task_stratum`.
4. **Empty audit tables remain machine-readable.** `qchan_v301_robustness_errors.csv/.parquet` has a fixed schema even when there are zero errors.
5. **Floor claim corrected.** Synthetic calibration now states that ranking may remain stable while the absolute LTAS-distance scale changes when bands approach the logarithmic floor.
6. **Real-cohort floor sensitivity added.** Cached cohort spectra are recomputed at -60, -80, and -100 dB floors without re-decoding audio. Rank stability is blocking; absolute scale changes are reported rather than hidden.
7. **Semicontinuous gallery corrected.** LTAS uses q05/q50/q95. Each one-sided deficit feature uses a zero case plus q10/q50/q90 of its positive distribution.
8. **Version isolation.** The notebook writes to `outputs/02_features/channel_device/qchan-v3.0.1` and uses `qchan_v301_*` artifact names. Existing v3.0.0 outputs are not overwritten.
9. **Generator and notebook synchronized.** The clean notebook is generated from `scripts/generate_qchan_v3_notebook.py`; deterministic cell IDs and exact generator reproduction are tested.

## Validation performed

- QCHAN package and notebook tests: **40 passed**.
- Every code cell compiles.
- The generated notebook is clean and unexecuted.
- Replay against the uploaded v3.0.0 cohort checkpoints: **519 recordings**, **224 subjects**, **519 measured LOSO references**, each with **223 reference subjects**.
- Real-output floor replay minimum pairwise Spearman correlations:
  - LTAS distance: **0.999995**
  - rolloff95 deficit: **1.000000**
  - high-band deficit: **1.000000**
  - tilt steepening: **1.000000**
- The revised label-blind gallery selected **13 unique recordings**, including zero and positive-distribution cases for every one-sided feature.

The full cohort notebook must still be executed once in the user's project environment because the current container does not include a Parquet engine. The user's prior successful QCHAN execution confirms that their project environment already has the required engine.

## Scientific interpretation

The release does not expand the permitted claims. QCHAN remains a task-matched, cohort-relative spectral profile. It is not a device classifier, microphone transfer-function estimate, or phenotype-independent acquisition measure. Values are comparable only within the same frozen reference vintage.
