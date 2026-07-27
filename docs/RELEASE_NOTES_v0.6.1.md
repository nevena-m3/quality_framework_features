# Release notes — v0.6.1

## Exact original Silero artifacts restored

- Restored the original output hierarchy under `outputs/segmentation/silero`,
  `outputs/figures/segmentation/silero`, and `outputs/logs`.
- Restored one `<recording>_segments.csv`, one `<recording>_frames.csv`, and one
  `<recording>_silero.png` per frozen Bamboo recording.
- Restored the original 30-ms non-overlapping diagnostic frames and original columns.
- Restored segment roles `leading_nonspeech`, `internal_nonspeech`,
  `trailing_nonspeech`, and `speech`.
- Restored the original four-panel plot layout, labels, legends, colors, boundaries,
  and accepted/flagged/excluded folders.
- Restored the original Silero speech padding of 50 ms.
- Added a clean stage regeneration so a recording cannot retain a stale PNG in more
  than one status folder.
- Added completeness assertions in both Silero notebooks.

## Scientific separation retained

The original frame CSV contains identically defined raw/smooth/strict compatibility
columns. Those columns are preserved exactly for reproducibility, but they are not used
as distinct sensitivity definitions. The rebuilt aggregate interval table continues to
hold genuinely distinct raw, primary, strict-speech, and guarded-nonspeech views for
Q-metric extraction and sensitivity analysis.

## Verification

- Added unit tests for exact frame columns, segment columns, segment roles, frame timing,
  plot axis labels, legend labels, and output creation.
