# QTEMP v1.0.0 final analytical disposition

This package closes QTEMP as an immutable analytical implementation and final feature-selection result.

## Final scientific outcome

- Zero QTEMP features enter the validated primary feature set.
- `qtemp_dropout_duration_fraction` and `qtemp_dropout_event_rate_per_min` remain available for exploratory descriptive use only.
- `qtemp_frozen_audio_duration_fraction` and `qtemp_frozen_audio_event_rate_per_min` remain available for zero-variation monitoring only.
- `qtemp_splice_discontinuity_rate_per_min` remains dropped after failed analytical validation.
- G9 is recorded as `N/A_NO_RETAINED_PRIMARY_EVENT_FEATURES`; it is not represented as passed.
- G10 is recorded as `FINALIZED_NO_RETAINED_PRIMARY_FEATURES`.

The final archive includes the standardized A-J gallery, validation checklist, G1-G10 gate summary, feature-level decisions, empirical tables, provenance, and SHA-256 manifest. It does not alter the validated primary feature table.

## Automated execution

`RUN_QTEMP_FINALIZE.cmd` executes the final notebook, verifies the immutable artifact manifest, writes the final snapshot and exploratory table, and opens the executed notebook automatically.

Final archive:

`outputs reviewed\06_QTEMP\qtemp-v1.0.0-analytical-final-no-retained`

Final snapshot:

`MAIN outputs\02_FEATURE_FAMILY_SNAPSHOTS\temporal_discontinuity\qtemp-v1.0.0-analytical-final-no-retained`

Exploratory/monitoring table:

`MAIN outputs\02_FEATURE_TABLES_EXPLORATORY\qtemp_v100_exploratory_features.csv`

## Manuscript boundary

Permitted: describe that 2/519 recordings contained algorithmically accepted bracketed dropout-like decoded support and 0/519 contained an event meeting the registered near-exact repetition rule, while explicitly labeling these outputs exploratory or monitoring-only and outside the validated primary feature set.

Not permitted: describe QTEMP as a validated measure of packet loss, network failure, buffering, concealment, missing speech, physiology, or a primary inferential biomarker family.
