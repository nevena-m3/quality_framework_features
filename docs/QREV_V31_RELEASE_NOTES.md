# QREV v3.1.0 final revision

## Scientific changes

- Classifies tail excess and persistence as primary conditional
  natural-boundary features, downward decay rate as a secondary conditional
  feature, and normalized SRMR as a broadly available established comparator.
- States explicitly that QREV does not estimate standard room-acoustic
  parameters, identify echo, or implement a discrete-delay echo detector.
- Treats feature availability as a measurement result and prespecifies its
  association with ALS severity and cohort for downstream analysis.
- Adds positive post-offset breath and delayed-echo controls, an additive-noise
  sensitivity grid for SRMR, and retains codec/resampling validation.
- Adds 0.8-s and 1.2-s persistence-horizon sensitivity around the prespecified
  1.0-s estimator.

## Robustness changes

- Replaces the first-sorted-recording subset with deterministic stratified
  sampling across support tier, tail magnitude, and baseline availability.
- Requires at least 30 paired observations for every blocking offset/window
  comparison.
- Reports baseline availability, perturbed availability, paired sample size,
  availability retention, rank stability, and absolute change.
- Prevents missing or zero-pair comparisons from passing through dropped
  values.
- Includes exactly four-boundary recordings in delete-one-boundary precision
  analysis and summarizes results by support tier.
- Retains the former 3-dB offset result as a descriptive diagnostic; the
  blocking magnitude check uses the empirical feature IQR because no validated
  universal 3-dB threshold exists for this study-specific relative estimator.

## Reporting changes

- Adds Wilson 95% intervals for feature availability.
- Adds availability by support tier and pairwise sample sizes for correlations.
- Adds a downstream missingness-analysis specification table.
- Uses concise feature labels, units, sample sizes, percentages, and thresholds
  in publication figures.
- Expands the label-blind gallery to include unavailable and minimum-support
  recordings.

## Verification completed before release

- 18/18 QREV package tests passed.
- All 14 generated notebook code cells compiled.
- All 14 notebook code cells executed in a full assembled repository with
  cohort-dependent operations disabled.
- All 10 synthetic construct/discriminant checks passed.
- Robustness and empirical cells executed against a 160-record synthetic
  contract fixture, including 800 offset rows and 1,120 window rows.
- The prior 519-record empirical feature table passed the revised empirical
  checks; the conditional-feature availability estimates were retained rather
  than reinterpreted as failures.

The actual cohort robustness grids, gallery, review decision, and immutable
freeze must be produced by running the notebook against the repository's
frozen recordings and segmentation outputs. No cohort result is fabricated or
pre-approved by this patch.
