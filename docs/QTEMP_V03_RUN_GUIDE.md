# QTEMP v0.3 run guide

Run `02f_temporal_discontinuity_QTEMP_v0_3_0_MEASUREMENT_DEVELOPMENT_EXECUTED_REVIEW.ipynb`, not the clean source notebook.

The default full run enables package tests, synthetic validation, participant-disjoint real-speech injection, signal-chain characterization, cohort extraction, parameter sensitivity, and the reviewer gallery. Cohort extraction uses four workers and a recording-level cache. Re-running after an interruption reuses valid cache entries; changing the implementation, parameters, task intervals, QDIST intervals, media size, or media modification time invalidates the affected cache automatically.

The output root is:

`outputs/02_features/temporal_discontinuity/qtemp-v0.3.0-measurement-development`

Key outputs include the feature registry, recording features, candidate ledger, candidate-disposition ledger, accepted-event ledger, exposure ledger, native inventory, real-speech validation tables, parameter sensitivity, empirical prevalence, participant recurrence, reviewer gallery, blinded adjudication sheet, G1–G10 summary, feature decisions, and candidate manifest.

## Blinded review

The notebook creates an unblinded internal key and a separate blinded response sheet. The response sheet hides recording identity, detector type, event disposition, score, diagnosis, and human-QC labels. Complete every row in the blinded sheet, then rerun the gallery and gate cells after setting the reviewer name, rationale, and explicit review decision. Threshold changes require a new measurement version and a new review sample.

## Runtime controls

- `QTEMP_WORKERS`: PowerShell environment variable, 1–4; default 4.
- `REUSE_COHORT_CACHE = True`: reuse matching recording cache entries.
- `FORCE_REEXTRACT = False`: leave false unless intentionally invalidating every cache entry.
- `MAX_SENSITIVITY_RECORDINGS = 18`: stratified sensitivity subset.
- `MAX_GALLERY_ACCEPTED_EVENTS = 100`: all accepted events are reviewed when manageable.

The first full run is expected to be much slower than a cached rerun because every native recording must be decoded once. Synthetic validation is deliberately replicated but bounded; it should no longer dominate runtime.
