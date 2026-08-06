# QGAIN v4.0.1 — Final Scientific Audit

## Executive decision

The corrected v4.0.1 cohort extraction is technically valid and uses the intended frozen `strict_speech / primary` interval contract. The prior v4.0.0 pooled-interval defect is resolved. No further change to the four numerical estimators is required before downstream use.

QGAIN should **not yet be frozen exactly as currently coded**, because the final notebook contains a governance typo in the G10 acceptance token and the feature roles need to be revised to reflect the empirical evidence. The required next change is a small v4.1.0 semantic/governance finalization, not another estimator redesign.

## Cohort integrity

- Recordings: 519
- Participants: 224
- ALS recordings: 418
- Control recordings: 101
- Canonical intervals: 7,738
- Canonical view/profile: `strict_speech / primary`
- Duplicate canonical interval identities: 0
- Overlapping canonical intervals: 0
- Media hashes verified: all recordings
- Extraction errors: 0
- Robustness errors: 0
- Gallery errors: 0
- Floor-contaminated recordings: 0
- All four measurements available: 519/519

The uploaded review package contains the generated outputs, but the included cohort notebook is an unexecuted template with zero execution counts. The outputs show that the run occurred, but the executed notebook must be saved and included in the eventual immutable freeze.

## Empirical feature distributions

| Feature | Median | IQR | 1st–99th percentile |
|---|---:|---:|---:|
| Typical speech level | -27.38 dBFS | 5.26 dB | -46.14 to -20.62 |
| Within-segment IQR | 9.42 dB | 3.14 dB | 3.66 to 15.11 |
| Between-segment MAD | 1.96 dB | 1.28 dB | 0.41 to 5.13 |
| Absolute drift | 3.04 dB/min | 3.96 dB/min | 0.03 to 14.87 |

No exact or near-exact duplicate feature was found. The maximum absolute Spearman correlation was 0.288, between between-segment MAD and absolute drift.

## Repeated-recording persistence

These are empirical within-subject persistence estimates, not technical test–retest reliability.

| Feature | First–second Spearman | ICC(1,1), first two | Median within-subject absolute difference |
|---|---:|---:|---:|
| Typical speech level | 0.531 | 0.398 | 1.90 dB |
| Within-segment IQR | 0.708 | 0.704 | 1.05 dB |
| Between-segment MAD | 0.355 | 0.345 | 0.75 dB |
| Absolute drift | 0.330 | 0.369 | 2.15 dB/min |

Within-segment IQR is the strongest and most persistent dynamics measurement. Between-segment MAD and drift are substantially less persistent.

## Segment-deletion robustness

The table reports change after deleting one canonical speech segment.

| Feature | Median absolute change | 95th percentile | Maximum |
|---|---:|---:|---:|
| Typical speech level | 0.074 dB | 0.328 dB | 1.078 dB |
| Within-segment IQR | 0.104 dB | 0.468 dB | 2.379 dB |
| Between-segment MAD | 0.170 dB | 0.788 dB | 1.689 dB |
| Absolute drift | 0.323 dB/min | 3.139 dB/min | 9.178 dB/min |

Typical level and within-segment IQR are robust. Between-segment MAD is moderately sensitive. Drift is strongly sensitive in the upper tail, particularly when fewer segments are available.

## Boundary-guard sensitivity

Reference guard: 200 ms. Alternatives: 100 and 300 ms.

| Feature | Median change, 100 ms | 95th percentile, 100 ms | Median change, 300 ms | 95th percentile, 300 ms |
|---|---:|---:|---:|---:|
| Typical speech level | 0.096 dB | 0.320 dB | 0.098 dB | 0.307 dB |
| Within-segment IQR | 0.261 dB | 0.659 dB | 0.175 dB | 0.537 dB |
| Between-segment MAD | 0.270 dB | 0.963 dB | 0.380 dB | 1.336 dB |
| Absolute drift | 0.716 dB/min | 3.831 dB/min | 0.926 dB/min | 4.414 dB/min |

The results support a fixed 200-ms contract. They do not support claiming that between-segment MAD or drift is insensitive to reasonable boundary variation.

## Segment-weighting sensitivity

The duration-weighted primary aggregation was compared with segment-balanced audit companions.

- Typical-level absolute difference: median 0.31 dB; 95th percentile 0.98 dB.
- Within-IQR absolute difference: median 0.44 dB; 95th percentile 1.40 dB.

These differences are acceptable as audit evidence, but they confirm that the estimands are frame-weighted, task-dependent summaries rather than segment-invariant properties.

## Drift-specific evidence

- Signed Theil–Sen 95% interval excluded zero in 87/519 recordings (16.8%).
- In the 50-recording time-order permutation audit, 9/50 recordings had `p < 0.05`.
- Median permutation p-value: 0.361.
- Drift had the weakest boundary and segment-deletion robustness.

The estimator correctly measures the absolute slope of segment medians. The concern is not numerical correctness; it is weak identification of meaningful gradual drift in a short, phonetic-content-varying reading task.

## Migration and continuity

Relative to the prior v3.1 extraction, v4.0.1 values were essentially unchanged:

- Typical level median absolute delta: 0.000 dB
- Within-segment IQR median absolute delta: 0.000 dB
- Between-segment MAD median absolute delta: 0.000003 dB
- Drift median absolute delta: 0.000 dB/min

This indicates that the reviewed implementation preserves the established estimator lineage while correcting provenance, support, validation, and governance.

Relative to invalid v4.0.0 pooled-interval outputs, the impact was materially larger for between-segment MAD and drift, confirming that the v4.0.1 rerun was necessary.

## Final feature recommendations

1. `qgain_typical_speech_level_dbfs` — **RETAIN_CONTEXTUAL**
2. `qgain_within_segment_iqr_db` — **RETAIN_PRIMARY_MIXED**
3. `qgain_between_segment_mad_db` — **RETAIN_SECONDARY_MIXED**
4. `qgain_abs_drift_db_per_min` — **RETAIN_EXPLORATORY_CONTEXTUAL**
5. Sustained level-step detector — **DROP**

All retained values remain nonordinal, cannot be used as standalone accept/reject gates, and must be accompanied by support/status metadata in ML pipelines.

## Required v4.1.0 finalization

The next patch should not change the estimators. It should:

1. Correct the G10 acceptance token from `ACCEPT_QGAIN_V400` to a v4.1.0-specific token.
2. Update the registry roles and publication status according to the decisions above.
3. Add explicit quantitative G6 summaries rather than treating mere audit coverage as robustness success.
4. Add subject-balanced one-recording-per-participant resampling summaries.
5. Add drift evidence fields and clearly mark drift as exploratory/contextual.
6. Preserve all four measurements in the ML interface, with availability, tier, status, and uncertainty companions.
7. Save the executed notebook in the final freeze.
8. Freeze no same-family scalar and define no standalone quality threshold.

## Publication wording

Recommended family label:

> Recorded level and level dynamics

Recommended claim:

> QGAIN features quantify contextual operating level and within-recording level variation in strict-speech regions. They are mathematically compatible with changes in recording gain, microphone distance, or automatic level control, but are not source-identifying and can also reflect vocal intensity, prosody, respiratory control, dysarthria, fatigue, and task structure.

## Overall conclusion

The corrected QGAIN extraction has scientific value. It is suitable as a measurement-context layer and as input to future biomarker-specific reliability models. It is not suitable as a generic “bad recording” score, and QGAIN group differences must not be interpreted as proof of poorer acquisition in ALS.

The numerical extraction is ready. One semantic/governance finalization remains before family freeze.
