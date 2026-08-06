# QADD v4.1 robustness repair and freeze decision

## Why v4.0 was not frozen

The first complete v4.0 cohort run successfully processed all 519 eligible
recordings and passed the deterministic formula, reconstruction, mechanism,
floor-censoring, codec, and empirical-accounting checks. Three checks in the
original support/boundary section failed:

- the row-weighted 90th percentile of leave-one-pause level change was
  1.075 dB for recordings with at least 3 s of non-floor pause support;
- the pooled 90th percentile of the nominal 100/300-ms guard perturbations was
  2.237 dB;
- the pooled availability-transition fraction was 11.67%.

These failures must not be “fixed” by loosening thresholds. Inspection of the
saved review tables identified three design errors in the validation itself.

1. Each omitted pause was treated as an independent population observation.
   Recordings with more pauses therefore received more weight.
2. A support class defined by duration and interval count was called “robust,”
   even though duration cannot establish stability when pause intervals are
   heterogeneous.
3. The 100-ms guard condition was exactly identical to the 200-ms condition in
   every audited recording after intersection with the frozen strict-pause
   view. The 300-ms condition was a one-sided erosion that changed the sampled
   construct and removed support; it was not a repeatability experiment.

The v4.0 gallery was also reviewed. Signal regions and feature behavior were
coherent, but unavailable spectral descriptors were drawn as empty axes with
meaningless tick labels.

## v4.1 scientific correction

The five feature formulas, analysis view, 200-ms pause guard, digital-floor
rule, flatness band, hum-comb definition, and null calibration are unchanged.
The version increment records a change to support terminology, validation
design, output presentation, and the freeze contract.

### Support terminology

Support classes are now `minimum`, `moderate`, `high`, and `unavailable`.
They describe usable signal quantity and interval diversity only. They do not
claim empirical robustness. Feature availability continues to be determined by
the prespecified feature-specific minimum support and floor rules.

### Whole-pause deletion

Every raw QADD estimand is recomputed after deleting one complete pause
interval:

- pause AC-level median;
- pause-level IQR;
- speech–pause level contrast;
- pause spectral flatness;
- mains-hum comb score.

Perturbations are summarized within each recording first (median, P90, and
maximum absolute change). Population summaries then contain one row per
recording and feature. Absolute-unit values are retained. A cohort-IQR-scaled
value is used only to place features with different units on a common figure.

Whole-pause deletion is explicitly called *construct sensitivity*, not
repeatability. A large change may indicate genuine nonstationary or rare
interference. It is preserved for later sensitivity analysis and is not used
to erase the feature value.

### Boundary sensitivity

The fixed 200-ms guard remains the operational definition. The alternative
condition starts from the exact reference pauses and erodes each boundary by an
additional 100 ms. This creates a genuine, traceable change in support.
Numerical changes and availability transitions are reported for all five
features. No post-hoc 1-dB or 5% threshold is imposed.

The blocking scientific checks test whether the audit is complete, correctly
clustered, reconstructable, and explicit about missingness. The observed
sensitivity is a result to report, not a code-correctness criterion.

## Gallery and figures

Gallery plots now state `insufficient_support` when flatness or hum is
unavailable and suppress meaningless secondary axes. The common robustness
figure shows:

1. recording-level whole-pause deletion sensitivity for every feature;
2. additional-boundary-erosion sensitivity for every feature;
3. reference and eroded feature availability.

Absolute-unit tables accompany the normalized display.

## Freeze contract

The notebook separates measurement freeze from downstream package integration.
Integration is recorded but is nonblocking at this stage. A freeze requires:

- package and deterministic tests;
- frozen-input and extraction contracts;
- mechanism, floor, and codec controls;
- reconstructability and complete clustered/boundary sensitivity accounting;
- empirical distribution/availability accounting;
- an explicit reviewer decision with reviewer name and rationale.

When requested, the notebook refuses to overwrite an existing freeze and
atomically copies the complete staged output to:

`MAIN outputs/02_FEATURE_FREEZE/additive_interference/qadd-v4.1.0/`

The frozen manifest records the implementation hash, gate-table hash,
parameters, review decision, and hashes of staged files.

## Interpretation for the manuscript

QADD v4.1 is suitable for feature extraction after the full cohort notebook
passes its blocking gates and the gallery is explicitly accepted. The
heterogeneity revealed by pause deletion is not evidence that the formulas are
incorrect. It is evidence that additive interference may vary within a
recording, which is scientifically compatible with the family definition.
Sensitivity summaries and availability must remain available for later
analysis and reviewer audit.
