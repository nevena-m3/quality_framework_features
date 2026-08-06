# QADD v4 implementation and validation note

> Historical v4.0 note. See `QADD_V4_1_ROBUSTNESS_REPAIR.md` for the corrected
> clustered-sensitivity, boundary-erosion, support-label, and freeze contract.

## Scope

QADD v4 operationalizes extrinsic additive acoustic interference as a
five-feature recording-level vector. It does not create a scalar family score,
does not revalidate segmentation, and does not use human-QC labels. Human-QC
correspondence is reserved for the separate Goal 3 analysis after the estimator
is locked.

## Confirmatory feature vector

1. `qadd_pause_ac_level_dbfs_median` — primary guarded-pause AC level.
2. `qadd_pause_level_iqr_db` — secondary pause-level heterogeneity.
3. `qadd_speech_pause_level_contrast_db` — mixed secondary within-recording
   speech-pause separation.
4. `qadd_pause_spectral_flatness` — non-ordinal spectral-type descriptor.
5. `qadd_mains_hum_comb_score_db` — targeted 50/60 Hz harmonic descriptor.

All primary analysis values are status-gated. Raw estimates, support, exact
zero/floor fractions, window counts, and reconstructable ledgers are retained
for audit.

## Decisions changed by validation

### Flatness upper frequency

The initial 80–7500 Hz band was rejected. In the prespecified representative
round-trip control, MP3 at 128 kb/s changed spectral flatness by approximately
0.163 at 7500 Hz, but by approximately 0.001 with upper cutoffs from 6000 to
7000 Hz. QADD v4 therefore freezes 80–7000 Hz to avoid measuring codec
low-pass behavior near the 8 kHz analysis Nyquist edge.

### Hum evidence

The raw robust comb score remains the analysis feature. Positive interpretive
hum evidence requires both:

- a score above a colored-noise null P95 calibrated by valid-window support;
- a median of at least three supported low-order harmonics.

This joint rule reduced independent colored-noise and 53 Hz off-grid false
positive rates to 0 in the current synthetic evaluation while retaining 100%
sensitivity for the prespecified −32 dBFS 50/60 Hz injections. These are
engineering operating-characteristic results, not perceptual thresholds.

### Digital floor

The 2% level-dependent floor-censoring ceiling applies to the three level
estimands. Flatness and hum descriptors never substitute floor-only windows
into their calculations; their availability is governed by valid non-floor
window support, with floor-window fractions stored as audit fields.

## Completed source-only checks

- 13 estimator tests passed.
- Analytical sine AC-RMS error was below numerical tolerance.
- Global-gain equivariance/invariance checks passed.
- Every raw feature reconstructed from saved ledgers.
- Additive-noise dose recovery had Spearman rho = 1.0.
- Speech-pause contrast dose response had Spearman rho = −1.0.
- All 10 mechanism/discriminant checks passed.
- All three floor-censoring checks passed.
- PCM, MP3, and Opus round trips completed; all four codec gates passed after
  excluding the unstable 7000–7500 Hz edge from flatness.
- The complete notebook executed in source-only mode and produced all
  deterministic/synthetic tables and publication figures.

## Still required before freeze

The current artifact is deliberately `candidate_only`. The following require
the user's frozen cohort and cannot be truthfully completed in a source-only
checkout:

- frozen-input and eligible-ID coverage checks;
- full-cohort extraction and error reconciliation;
- empirical distributions and availability;
- interval-cluster precision and boundary robustness;
- label-blind recording gallery review;
- named scientific acceptance and final pipeline integration.

Human-QC correspondence is not on this list because it is a downstream Goal 3
analysis, not a QADD feature-freeze criterion.
