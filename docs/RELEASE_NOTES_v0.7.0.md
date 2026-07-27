# Release notes — v0.7.0

## Segmentation review and freeze

- Added diagnosis/outcome-independent mandatory review of every flagged/excluded recording.
- Added accepted-recording review prompts using prespecified segmentation-only robust
  outlier and near-threshold rules.
- Added a Jupyter reviewer with original-plot display, audio playback, decision controls,
  manual speech-interval editing, and waveform preview.
- Separated recording eligibility (`KEEP`/`EXCLUDE`) from boundary provenance
  (`AUTO`/`MANUAL`/`NONE`).
- Added strict validation of manual intervals: numeric, finite, ordered, non-overlapping,
  inside recording duration, and complete reviewer/date/reason provenance.
- Added manual frame CSV, segment CSV, and diagnostic PNG artifacts.
- Added an atomic, versioned immutable freeze under
  `MAIN outputs/01_SEGMENTATION_FREEZE/<version>`.
- Added `frozen_segmentation_intervals`; feature extraction now reads only the
  authoritative MAIN outputs freeze.
- Manual correction replaces primary boundaries only. Conservative/permissive
  sensitivity profiles remain automatic.
- Added automatic migration/backup of the pre-v0.7.0 segmentation review sheet.

## Verification

- Added tests for accepted-outlier selection, mandatory review completion, atomic notebook
  saves, manual interval validation, and primary-only manual replacement.
- Added an integration test proving that the versioned MAIN outputs freeze is created
  atomically and cannot be overwritten.
- All generated notebook code cells pass Python syntax validation.
