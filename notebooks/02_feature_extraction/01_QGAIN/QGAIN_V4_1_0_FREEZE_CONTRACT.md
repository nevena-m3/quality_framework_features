# QGAIN v4.1.0 Freeze Contract

## Scientific scope

QGAIN v4.1.0 measures **recorded level and level dynamics** in frozen `strict_speech / primary` regions. The features are observable waveform descriptors. They do not uniquely identify automatic gain control, device gain, microphone movement, vocal effort, respiratory change, dysarthria, or fatigue.

## Final retained features

| Feature | Final role | Default manuscript inclusion | Standalone gate |
|---|---|---:|---:|
| `qgain_typical_speech_level_dbfs` | Contextual operating-level measurement | Yes | Prohibited |
| `qgain_within_segment_iqr_db` | Primary mixed acquisition/physiology descriptor | Yes | Prohibited |
| `qgain_between_segment_mad_db` | Secondary mixed descriptor | No | Prohibited |
| `qgain_abs_drift_db_per_min` | Exploratory/contextual trend descriptor | No | Prohibited |

The sustained level-step detector remains dropped.

## Numerical continuity

The v4.1.0 finalization does not change the four estimators or cohort values from validated v4.0.1. Numerical equivalence must be bitwise exact with missing values preserved.

## Final governance rules

- Canonical segmentation is exactly `strict_speech / primary`.
- No QGAIN family scalar is constructed.
- No extraction-time or standalone accept/reject threshold is approved.
- P.56 remains an optional comparator and is not a blocking validation requirement.
- Every ML-facing value must retain availability, status, support tier, version, and signal-view metadata.
- Drift must retain its signed slope, confidence interval, and evidence status and remain exploratory.
- The final freeze must contain the saved executed notebook, implementation, tests, feature passports, source hashes, and a SHA-256 inventory.
- Any future change requires a new semantic version. The v4.1.0 freeze must never be overwritten.

## Acceptance token

`ACCEPT_QGAIN_V410`
