# QADD Stage 0 Integrated Audit

## Executive decision

The legacy QADD v4.1.0 implementation is scientifically sophisticated and provides strong evidence for retaining a five-feature, no-scalar QADD vector. It should **not** be adopted unchanged as the reviewed family freeze. Two implementation-level corrections and several standardized validation gaps must be resolved in a new reviewed candidate.

The recommended next version is **`qadd-v4.2.0-candidate`**. Most numerical primitives are retained; the cohort must be rerun because the speech-pause contrast currently applies an unintended second 50-ms erosion to already-frozen strict-speech intervals.

## Legacy evidence retained

- 519 eligible recordings processed with zero extraction errors.
- 16 current QADD unit tests pass.
- Analytic AC-RMS/dBFS checks, gain equivariance/invariance, raw-ledger reconstruction, synthetic noise-dose controls, floor-censoring calibration, codec characterization, whole-pause deletion, boundary erosion, empirical distributions, and label-blind galleries are present.
- No family scalar or standalone rejection threshold is constructed.

## Current availability

- `qadd_pause_ac_level_dbfs_median`: 462/519 (89.0%), median -65.505, IQR 15.385.
- `qadd_pause_level_iqr_db`: 440/519 (84.8%), median 5.957, IQR 8.184.
- `qadd_speech_pause_level_contrast_db`: 462/519 (89.0%), median 38.341, IQR 14.269.
- `qadd_pause_spectral_flatness`: 371/519 (71.5%), median 0.080, IQR 0.113.
- `qadd_mains_hum_comb_score_db`: 213/519 (41.0%), median 2.821, IQR 2.897.

Hum joint evidence exceeded the count-matched colored-noise null in 11 recordings; the raw score was available in 213 recordings. These are descriptive observations, not prevalence of proven electrical hum.

## Blocking corrections

### 1. Duplicate strict-speech erosion

The frozen segmentation already defines `strict_speech` by eroding primary speech by 50 ms per edge. The current QADD extractor then erodes that strict view by another 50 ms. Across the 519 recordings this removes approximately 1.49 s of otherwise canonical strict-speech support on average (median 1.20 s; maximum 5.70 s). The pause estimators are not affected by this exact issue, but the speech-pause contrast must be recomputed.

### 2. Hum winner-support bookkeeping

The raw 50/60-Hz comb score is correctly formed from recording-level medians. However, the stored supported-harmonic count is the per-window winner's count, and the subsequent recording-level winner companion can therefore mix 50- and 60-Hz evidence. Separate support-count columns are required for each comb.

### 3. Canonical interval provenance

The current frame and spectral ledgers retain a local interval index but not the frozen source view/profile/interval identity. The reviewed ledgers must make each frame/window traceable to the canonical segmentation freeze.

## Scientific gaps that block reviewed freeze

- Hum specificity does not yet include low-F0/voiced leakage, musical tones, fan-like near-grid periodicity, or frequency-drift controls required by the Master design.
- Contrast has not been tested in an independent speech-level x pause-noise factorial design.
- Current support tiers describe quantity, not empirically calibrated precision. Whole-pause deletion and boundary results show material sensitivity for IQR, hum, and some low-support recordings.
- Repeated-recording persistence and participant-balanced summaries are absent.
- Parameter sensitivity is incomplete for the guard grid, level-frame size, flatness band/window, and hum window/sidebands.
- The legacy figures do not satisfy the standardized A-H/J bundle contract.
- A reviewed non-imputed ML interface and immutable executed-notebook freeze are absent.

## Provisional feature dispositions

- Guarded-pause AC level: retain as the primary contextual measurement with explicit gain-noise entanglement.
- Pause-level IQR: retain conditionally as a secondary nonstationarity descriptor.
- Speech-pause contrast: correct and retain as a secondary mixed descriptor, never SNR.
- Spectral flatness: retain as a nonordinal descriptor; freeze 80-7000 Hz and update the Master registry.
- Hum-comb score: correct audit companions and expand specificity controls, then retain as a targeted descriptor.
- Competing speech: explicitly unmeasured as a distinct mechanism in the confirmatory QADD vector.

## Existing robustness must be interpreted carefully

The legacy gate marks sensitivity as passed when the audit completed, not when changes were small. Whole-pause deletion population-P90 changes relative to the cohort IQR were approximately 0.26 for pause level, 0.65 for IQR, 0.28 for contrast, 0.34 for flatness, and 0.42 for hum. Additional 100-ms erosion changed availability in 23-28% of sampled recordings for the general/flatness features and 10% for hum. These results do not invalidate the estimators, but they require explicit support and uncertainty interpretation.

## Reviewed implementation sequence

1. Freeze this scientific contract and machine-readable checklist.
2. Build `qadd_v420.py` with the two numerical/provenance corrections.
3. Run expanded G2-G5 synthetic and discriminant preflight.
4. Run corrected cohort extraction against canonical primary-profile intervals.
5. Complete G6-G8 support, persistence, participant weighting, and redundancy analyses.
6. Produce standardized panels A-H and J; record I as N/A.
7. Make feature-specific G10 decisions and create an immutable reviewed freeze.
