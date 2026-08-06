# QGAIN v3.1 scientific decision record

## Final analysis profile

QGAIN v3.1 operationalizes recorded speech level as a four-dimensional profile:

1. `qgain_typical_speech_level_dbfs` — median framewise AC-RMS level during
   guarded strict speech. This is a contextual digital operating-level measure,
   not calibrated sound-pressure level and not ITU-T P.56 active speech level.
2. `qgain_within_segment_iqr_db` — pooled interquartile range of frame levels
   after subtracting each segment median. It measures short-term level
   dispersion while removing segment offsets.
3. `qgain_between_segment_mad_db` — 1.4826 times the median absolute deviation
   of usable segment-median levels. It measures robust segment-to-segment level
   variability.
4. `qgain_abs_drift_db_per_min` — absolute Theil-Sen slope of segment-median
   level against original recording time, expressed in dB/min. The signed slope
   and 95% slope interval are retained for audit.

These estimators describe recorded-level behavior. They are compatible with
acquisition gain changes, platform processing, or microphone-distance changes,
but they are not source-identifying. Vocal intensity, prosody, respiration,
dysarthria, posture, task content, and segmentation remain recognized
confounds.

## Removal of the v3.0 local-transition rate

The v3.0 candidate `qgain_sustained_step_rate_per_min` is not a QGAIN v3.1
analysis feature. In the 519-recording validation cohort it generated 8,412
candidates, occurred in every recording, and had a median rate of 43.70
candidates/min (interquartile range 37.03–51.75/min). Gallery inspection showed
that many candidates followed ordinary phonetic and prosodic level transitions.
This behavior was incompatible with the intended interpretation as occasional
sustained acquisition-gain steps.

The detector is retained only as an explicitly rejected exploratory audit so
the negative result remains reproducible. Its outputs are excluded from:

- `ANALYSIS_FEATURES` and the immutable feature registry;
- manuscript-facing empirical summaries and correlations;
- blocking scientific claims;
- `qgain_v31_analysis_features.csv/.parquet`;
- the central `MAIN outputs/02_FEATURE_TABLES` export.

No threshold was retuned after inspecting cohort results. Reintroducing an
abrupt-change feature would require a new estimator, new measurement version,
and independent real-speech specificity validation.

## Validation status

The four retained estimators preserve the v3.0 implementation without numerical
changes. The v3.0 cohort audit found complete extraction for 519/519 recordings,
no extraction errors, exact gain-transform behavior, expected synthetic
dose-response behavior, successful digital-floor censoring, small codec
round-trip effects, explicit support accounting, and completed boundary and
segment-deletion sensitivity analyses.

QGAIN v3.1 adds automated gates that prevent any step/transition diagnostic
from entering the analysis table or central export.

