# Release notes — v0.9.0

## All-recording Silero curation

- Replaced the required-only dropdown with a scrollable browser that starts with all
  recordings and supports queue filters plus filename/ID/QC search.
- Added one-click **Keep Silero + next**, **Exclude + next**, and
  **Save manual + next** actions.
- Kept manual interval editing optional and hidden unless `MANUAL` is selected.
- Displayed the original four-panel Silero figure, exact-edge boundary audit, audio
  player, quantitative diagnostics, and decision provenance together.
- Added explicit all-recording, mandatory-review, and locked-exclusion progress counts.

## Frozen human-QC task gate

- Propagated `Task Completed as Instructed` from frozen Bamboo metadata into the
  segmentation summary and decision ledger.
- Normalized only exact `NO` values to an automatic, locked `EXCLUDE + NONE` decision.
- Preserved missing values without silently interpreting them as either yes or no.
- Prevented prior review CSVs or widget edits from overriding the frozen metadata gate.

## Post-review outputs

- Added `outputs/01_segmentation_after_review` beside the untouched automatic
  `outputs/01_segmentation` stage.
- Materialized one final frames CSV, segments CSV, four-panel figure, and boundary
  audit per recording, grouped into accepted, flagged, and excluded directories.
- Preserved excluded artifacts for audit. Accepted and flagged recordings are
  analysis-eligible; excluded recordings are not.
- Added reviewed recording, interval, status-count, optional-artifact, and manifest
  tables.
- Manual primary boundaries now receive their own exact-edge boundary audit before
  post-review materialization.

The automatic Silero run is never rewritten by adjudication. The separate reviewed
tree and immutable versioned `MAIN outputs/01_SEGMENTATION_FREEZE/<version>` preserve
the complete provenance from automatic result to final downstream eligibility.
