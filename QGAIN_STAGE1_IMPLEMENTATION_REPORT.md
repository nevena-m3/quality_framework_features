# Stage 1 implementation report — QGAIN reviewed candidate

## Scope completed

Stage 1 creates a parallel reviewed repository structure and a candidate redesign of QGAIN without modifying the original `notebooks`, `src`, `tests`, `outputs`, or `MAIN outputs` folders.

The reviewed QGAIN family is framed as **recorded level and level dynamics**, not as a source-identifying estimate of device gain or automatic gain control. Four public feature names are retained for downstream compatibility:

1. `qgain_typical_speech_level_dbfs`
2. `qgain_within_segment_iqr_db`
3. `qgain_between_segment_mad_db`
4. `qgain_abs_drift_db_per_min`

The rejected sustained/local level-step detector is absent from the reviewed production module, feature registry, scientific export, and ML interface.

## Blocking implementation corrections

- Corrected the signal-view contract to mono, globally DC-removed, deterministically resampled 16-kHz audio with framewise mean-removed AC-RMS.
- Corrected computational-floor logic so sub-floor nonzero frames are marked as floor affected rather than only numerically clamped.
- Restricted the pooled within-segment IQR to frames from usable segments.
- Made the normal-consistency MAD factor explicit (`1.482602218505602`) and exported the unscaled MAD as an audit companion.
- Added segment-balanced companions to expose recording-duration and segment-weighting effects.
- Added exact support, status, availability, provenance, full-scale, and missingness fields.
- Added a long-form measurement export and a non-imputed ML-facing interface containing values, availability masks, status, and support tiers.
- Explicitly prohibited standalone QGAIN reject/accept decisions and extraction-time family composites.

## Validation implemented

The candidate notebook contains the common G1–G10 structure. Synthetic preflight validation covers:

- gain equivariance and dynamics invariance;
- polarity, DC-offset, and common time-shift invariance;
- 44.1/48-kHz resampling sensitivity;
- FLAC, Opus, and AAC round-trip sensitivity;
- amplitude-modulation, segment-offset, and linear-drift dose response;
- source non-identifiability between speaker-level and device-gain envelopes;
- constant-RMS spectral-change specificity;
- guarded-support, segment-count, span, floor-censoring, and boundary-guard behavior;
- deterministic package behavior and missing-stays-missing exports.

Package test result: **16 passed**.

## Cohort validation designed but not executed here

The notebook includes local-cohort code for:

- frozen input and media-hash verification;
- reviewed extraction from frozen primary/strict-speech intervals;
- v3.1-to-v4.0 migration analysis;
- empirical distributions and support availability;
- segment delete-one and boundary-guard robustness;
- time-order permutation auditing of drift;
- repeated-recording persistence using within-subject differences and a random-intercept ICC;
- feature redundancy analysis;
- label-blind, signal-linked galleries;
- scientific, long-form, and ML-interface exports.

These components were not executed because the uploaded package did not include the raw cohort audio. They remain blocking G6–G8 items.

## Deliberately open pre-freeze item

The Master Feature Design requires comparison with, but not relabeling as, ITU-T P.56 active speech level. Stage 1 does not reproduce that standard from an unverified reimplementation. The official G.191 Speech Voltmeter implementation must be pinned, built, reference-tested, and integrated before the QGAIN publication freeze. This is recorded as a blocking G8 item.

## Current status

- Measurement version: `qgain-v4.0.0-candidate`
- Scientific feature count: 4
- Synthetic/package validation: passed
- Cohort extraction: pending local run
- Official P.56 comparator: pending
- Scientific G10 decision: pending
- Publication freeze: **not created**
- Operational ML thresholds: **not calibrated**

## Local run sequence

1. Extract the Stage 1 ZIP at the existing project root.
2. Open `notebooks reviewed/01_QGAIN/02b_gain_dynamics_QGAIN_v4_0_0_REVIEWED_SOURCE.ipynb`.
3. Confirm `MEDIA_ROOT_OVERRIDE` or `MEDIA_PATH_MAP` only if frozen paths do not resolve.
4. Run the default synthetic preflight unchanged.
5. Set `RUN_COHORT_EXTRACTION = True` and rerun with the raw audio available.
6. Review extraction errors, migration deltas, support robustness, persistence, correlations, and galleries.
7. Integrate the pinned official P.56 comparator.
8. Make feature-specific G10 decisions.
9. Only then set `SCIENTIFIC_REVIEW_DECISION = "ACCEPT_QGAIN_V400"` and consider `PUBLISH_AND_FREEZE = True`.
