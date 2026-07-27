# Release notes — v0.6.0

## Cohort and media freeze

- Added investigator-confirmed exceptional control IDs with required evidence provenance.
- Kept subject-level confirmations in ignored local configuration rather than the portable
  example.
- Reduced the current diagnosis-adjudication template to only genuinely unresolved IDs.
- Selected one uniquely decodable encoding per logical recording using the prespecified
  order WAV, WEBM, then MP4.
- Kept alternate encodings available for technical-replicate sensitivity analysis.

## Silero visual QC and adjudication

- Preserved one four-panel diagnostic plot per Bamboo recording.
- Saved plots under `outputs/01_segmentation/figures/accepted`,
  `outputs/01_segmentation/figures/flagged`, and
  `outputs/01_segmentation/figures/excluded`.
- Preserved the original pipeline's hard and soft automatic triage rules.
- Added `segment-template` and `segment-adjudicate`.
- Accepted recordings default to KEEP; flagged/excluded recordings require a documented
  KEEP or EXCLUDE decision and reviewer.
- Blocked feature extraction until a complete frozen segmentation decision table exists.

## Notebook output contract

- Every thin execution notebook and long-form visualization notebook now:
  - saves at least one CSV table;
  - saves at least one PNG and SVG figure;
  - uses separate `tables/` and `figures/` folders for its stage;
  - ends with an explicit PASS/BLOCKED gate and next-step instruction.
- Corrected Goal 4 output names and Gwet AC1 plotting.
- Added exact-session Rest reference summaries.

## Verification

- All 23 generated notebooks passed JSON and Python-cell syntax validation.
- All 28 unit tests passed in the available manual test harness.
- The CLI imports and exposes the two new segmentation-adjudication commands.
