# QREV v4.0.0 cohort AudioViews contract hotfix R2

## Failure diagnosis

The cohort extraction decoded every media file successfully, but the notebook
requested `views.analysis`. The repository's authoritative
`paper1_qc.media.AudioViews` dataclass exposes the standardized mono,
DC-removed, resampled analysis signal as `analysis_16k`.

Because the incorrect attribute was referenced in the common extraction path,
every recording failed at the same line before QREV feature estimation began.
No QREV measurement was computed and no scientific conclusion was affected.

## Correction

R2 adds a single canonical adapter,
`analysis_waveform_from_audio_views(views)`, to the reviewed cohort module. The
adapter:

- requires `AudioViews.analysis_16k`;
- returns a one-dimensional float64 analysis waveform;
- rejects empty, non-finite, or noncanonical views with explicit errors.

All four notebook paths now use the adapter:

1. primary cohort extraction;
2. robustness extraction;
3. SRMR bandwidth characterization;
4. Panel G signal-linked galleries.

Two regression tests verify that the canonical `analysis_16k` field is used and
that the obsolete `analysis` alias is rejected.

## Governance impact

No feature equation, parameter, support rule, censoring rule, SRMR variant,
checklist item, figure requirement, or scientific claim changed. The
measurement remains `qrev-v4.0.0-candidate`; cohort extraction and freezing
remain candidate-only and blocked pending post-cohort review.

## Restart state

The installer archives the failed checkpoint directory, creates a fresh local
cohort notebook, and leaves `REBUILD_CHECKPOINTS=False`. The next run therefore
starts cleanly while preserving restart-safe behavior for any later
interruption.
