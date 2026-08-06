# QREV v4.0.0 — Immutable freeze contract

## Accepted measurement

The numerical measurement version is `qrev-v4.0.0`. The accepted feature set is:

- `qrev_tail_excess_100ms_db` — primary conditional;
- `qrev_tail_persistence_median_sec` — secondary conditional and non-independent;
- `qrev_downward_decay_rate_db_per_sec` — exploratory conditional, excluded from default confirmatory analyses;
- `qrev_srmr_norm` — pinned established comparator.

## Input and estimator contract

- Deterministic 16-kHz mono analysis waveform with global DC removal and no amplitude normalization.
- Natural speech offsets: exactly `primary_speech / primary`.
- SRMR support: exactly `strict_speech / primary`; natural primary task span defines the analyzed segment.
- Early tail: 0–100 ms after the natural speech offset.
- Stable late floor: 700–1,000 ms.
- Persistence horizon: 600 ms; values at the horizon are explicitly right-censored.
- Final support policy: minimum two eligible boundaries. Support classes describe quantity, not calibrated precision.
- SRMR identity and dependency versions are pinned in the manifest and feature passport.

## Prohibited interpretations and uses

QREV does not estimate RT60, EDT, DRR, C50/C80, D50, STI, an RIR, or discrete echo identity. It does not identify breathing, room reverberation, noise or echo as a physical source. No QREV family scalar, standalone reject threshold or operational pass/fail rule is authorized. Missing and censored values must remain explicit and must not be replaced with zero.

## Immutability

The measurement freeze and figure package are never overwritten. Any numerical, support-policy or semantic measurement change requires a new measurement version. Any presentation-only figure change requires a new figure-package version and must retain the measurement-freeze hashes.
