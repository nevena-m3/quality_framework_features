# QGAIN v4.0.1 input-contract correction

This patch supersedes the qgain-v4.0.0 cohort outputs. The feature estimators are unchanged.

Blocking correction:
- Production extraction now selects exactly `view == strict_speech` and `profile == primary` from the frozen interval table.
- It never unions `primary_speech` with `strict_speech` or pools conservative/primary/permissive profiles.
- Canonical interval identity, coverage, duplicates, and overlap are blocking G1 checks.
- Segment-deletion robustness now acts on unique canonical intervals.

Governance corrections:
- ITU-T P.56 remains an optional comparability analysis, as specified in the Master Feature Design; it is not a blocking freeze gate.
- Repeated-recording output is labelled empirical within-subject persistence, not technical repeatability. It uses first/second-session correlations and a balanced ICC(1,1) method-of-moments estimate.
- QGAIN remains candidate-only. G10 must remain pending until the corrected cohort outputs are reviewed.
