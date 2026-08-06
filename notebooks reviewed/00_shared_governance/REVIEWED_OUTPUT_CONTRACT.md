# Reviewed output contract

Every reviewed family must export three logically separate layers.

## Scientific measurement layer

One row per recording, containing feature values, exact measurement version,
signal-view provenance, support, status, missingness, censoring, and audit
companions. Missing measurements remain `NaN`; zero is reserved for validly
observed zero-valued measurements.

## Long-form measurement layer

One row per recording-feature pair with standardized fields:

- `logical_recording_id`
- `family`
- `feature`
- `value`
- `unit`
- `available`
- `measurement_status`
- `support_tier`
- `measurement_version`
- `signal_view`
- `phenotype_confounding_risk`
- `standalone_gate_allowed`
- `ml_role`

## ML interface layer

A wide table containing each feature value together with availability masks,
status, and support tier. The export performs no imputation and defines no
"good/bad" threshold. Operational thresholds must be calibrated later against
biomarker-specific extraction error, model error, or abstention utility.

## Freeze rules

A family freeze requires all blocking gates to pass, a feature-specific G10
decision, an immutable parameter and code hash, exact output reconstruction,
and explicit confirmation that no unreviewed scalar/composite entered the
analysis or ML interface.
