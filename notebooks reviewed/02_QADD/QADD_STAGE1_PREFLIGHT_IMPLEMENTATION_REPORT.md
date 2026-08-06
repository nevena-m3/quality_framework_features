# QADD v4.2.0 reviewed preflight — implementation report

## Decision

The legacy qadd-v4.1.0 extraction is not accepted as the final reviewed family. Most estimator primitives are retained, but two implementation defects require a reviewed cohort rerun:

1. `strict_speech` was already eroded in the frozen segmentation and was eroded a second time inside QADD. This affects speech–pause level contrast.
2. The recording-level hum winner could be paired with harmonic-support counts mixed across 50- and 60-Hz window winners.

The reviewed candidate is `qadd-v4.2.0-candidate`. It is preflight-only and cannot be frozen yet.

## Retained candidate measurements

- `qadd_pause_ac_level_dbfs_median`: primary contextual recorded pause level.
- `qadd_pause_level_iqr_db`: secondary pause-level nonstationarity.
- `qadd_speech_pause_level_contrast_db`: secondary mixed descriptor; not SNR.
- `qadd_pause_spectral_flatness`: nonordinal spectral-type descriptor, frozen over 80–7000 Hz.
- `qadd_mains_hum_comb_score_db`: targeted 50/60-Hz hum-like structure descriptor.

No QADD scalar or standalone rejection threshold is permitted.

## Code changes

- Added `speech_intervals_are_guarded` and `pause_intervals_are_guarded` input contracts.
- Canonical `strict_speech / primary` support can now be used without a second guard.
- Added separate 50- and 60-Hz supported/evaluated harmonic counts to the spectral ledger.
- Recording-level winner support now comes from the winning 50- or 60-Hz comb.
- Added signal-source and applied-guard provenance fields.
- Preserved reconstructable frame, interval, and spectral ledgers.

## Validation completed

- 18 package tests passed.
- 9/9 notebook code cells executed with no saved errors.
- G1: reviewed contract and feature registry — PASS.
- G2: formula, reconstruction, and corrected guarding — PASS.
- G3: gain, polarity, DC, common time shift, resampling, and FLAC/Opus/AAC/MP3 round trips — PASS.
- G4: pause-noise dose, speech × pause-noise factorial, amplitude modulation, spectral type, and hum dose — PASS.
- G5: colored noise, off-grid combs, single tones, breath-like energy, competing-speech-like energy, and true 50/60-Hz controls — CONDITIONAL PASS. Generic additive sources are measured without resolving source identity; exact low-F0 60-Hz periodic structure remains causally non-identifiable.
- G6 synthetic support/floor handling — PREFLIGHT PASS.
- G7–G8 cohort evidence — PENDING.
- G9 event adjudication — N/A.
- G10 final decisions/freeze — PENDING.

## Preflight figures

Panels A–C are generated as auditable bundles with PNG, SVG, PDF, source CSV, caption, and provenance JSON:

- A: controlled construct response.
- B: hum discriminant specificity and periodic confounds.
- C: transformation contract, resampling, and codec characterization.

Panels D–H/J require the corrected cohort run. Panel I is not applicable.

## Next authorized action

Run the reviewed preflight locally and return the executed notebook and `outputs reviewed/additive_interference/qadd-v4.2.0-candidate`. After independent confirmation, build and run the corrected cohort extraction. Do not publish or freeze at the preflight stage.
