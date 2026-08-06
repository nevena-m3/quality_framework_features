# Shared governance

This folder contains contracts shared across reviewed family notebooks. Common
code belongs in `src reviewed/paper1_qc_reviewed`; notebooks orchestrate and
visualize but do not duplicate authoritative estimators.

The reviewed family sequence is QGAIN → QREV → QCHAN → QADD → QDIST → QTEMP.
Each family must explicitly implement G1–G10, including `not applicable` where
a gate does not apply.
