# QCHAN v4.0.0 — Immutable freeze contract

## Measurement identity

- Family: QCHAN — channel/device spectral manifestations
- Measurement version: `qchan-v4.0.0`
- Figure-package version: `qchan-v4.0.0-figures-v1.0.0`
- Source candidate: `qchan-v4.0.0-candidate`
- Finalization revision: `qchan-v4.0.0-finalization-r1`
- Acceptance token: `ACCEPT_QCHAN_V400`

## Frozen analysis features

1. `qchan_ltas_distance_db`
2. `qchan_rolloff95_deficit_hz`
3. `qchan_highband_ratio_deficit`
4. `qchan_tilt_steepening_db_per_oct`

The three signed precursors are also frozen and must remain available:

- `qchan_rolloff95_signed_difference_hz`
- `qchan_highband_ratio_signed_difference`
- `qchan_tilt_signed_difference_db_per_oct`

## Frozen reference contract

The reference is task-matched, subject-balanced, and leave-one-subject-out. No global or cross-task fallback is allowed. Reference participant membership, recording membership, task stratum, estimator parameters, spectral grid, and reference hashes are part of feature identity.

The accepted cohort uses one reference vintage:

`74c0b334f69b5b7aa5d5fd8919810bbf8e40c7991f4485b754fcfabf0acd622e`

## Cohort seal requirements

The atomic freeze is authorized only when the final manifest confirms:

- scientific decision `ACCEPT_QCHAN_V400`;
- 519 recordings and 224 participants;
- exact numerical equivalence to the completed cohort candidate;
- all four analysis features available in 519 recordings;
- 519 reference-ledger rows and one reference vintage;
- no extraction, reference, robustness, or gallery errors;
- 22 applicable figure/example bundles plus Panel I explicit N/A;
- feature values were not recomputed during finalization;
- no family scalar, standalone rejection threshold, or device identity;
- the executed finalization notebook contains no unexecuted code cells or saved errors.

## Versioned interpretation

- LTAS distance is primary and nonordinal.
- Rolloff-95 deficit is primary and one-sided.
- High-band ratio deficit is secondary and non-independent.
- Tilt steepening is exploratory and phenotype-sensitive.
- One-sided zero values include equal or upward signed differences and do not prove absence of channel effects.
- Source rate and native bandwidth remain context variables even when no native-bandwidth limitation is present.

## Immutability

The freeze scripts refuse to overwrite an existing `qchan-v4.0.0` measurement freeze or `qchan-v4.0.0-figures-v1.0.0` figure package. Any subsequent change to estimators, support, reference construction, reference vintage, feature values, final roles, figures, captions, or provenance requires a new semantic version or a separately versioned figure package.
