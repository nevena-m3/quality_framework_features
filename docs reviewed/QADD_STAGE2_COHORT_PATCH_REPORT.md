# QADD v4.2.0 — Preflight Audit and Cohort-Stage Decision

## Decision

The uploaded local preflight run is accepted for progression to corrected cohort extraction.

It is not ready for scientific freeze. G7, G8, the cohort portion of G6, Panels D–H/J, and feature-specific G10 decisions remain unresolved by design.

## Verified preflight execution

- Notebook code cells executed: 9/9
- Saved Python errors: 0
- Package tests in the uploaded run: 18 passed
- Independent reviewed test suite after the cohort-stage additions: 27 passed
- Preflight blocking checks: all passed
- Cohort extraction completed: no
- Publication/freeze enabled: no
- Family scalar constructed: no
- Standalone quality gate allowed: no

## Gate status after preflight

| Gate | Status | Interpretation |
|---|---|---|
| G1 | PASS | Contract, feature registry, and input-view rules are defined. |
| G2 | PASS | Numerical formulas, reconstruction, and no-double-guard behavior pass. |
| G3 | PASS | Gain, polarity, DC, shift, resampling, and codec behavior pass. |
| G4 | PASS | Controlled noise, contrast, amplitude-modulation, spectral-type, and hum dose responses pass. |
| G5 | CONDITIONAL | Specificity is adequate within the stated claim boundary; exact low-F0 periodic sources remain an irreducible hum-like confound. |
| G6 | PREFLIGHT PASS | Floor and minimum-support logic pass; real-cohort support and boundary evidence remain pending. |
| G7 | PENDING | Real-cohort availability, distributions, and signal-linked examples are not yet available. |
| G8 | PENDING | Persistence, participant weighting, and redundancy are not yet available. |
| G9 | N/A | No QADD event detector is retained. |
| G10 | PENDING | Final retain/demote/revise/drop decisions require the cohort audit. |

## Scientific findings from preflight

1. Pause level increases monotonically and approximately one-for-one with injected pause-region noise level.
2. Speech–pause contrast decreases with increasing pause noise and increases with independent speech gain, confirming that it is a mixed within-recording contrast rather than physical SNR.
3. Pause-level IQR increases monotonically under controlled amplitude modulation.
4. Spectral flatness separates broadband-like, colored, and tonal pause spectra and remains explicitly nonordinal.
5. The hum-comb detector detects injected 50/60-Hz harmonic structure and rejects colored noise, off-grid combs, isolated tones, and generic breath/competing-speech controls.
6. A periodic source whose fundamental aligns with 50/60 Hz can remain indistinguishable from mains-like structure. Therefore, the hum feature is a targeted structural descriptor, not a source-identity detector.

## Cohort-stage corrections incorporated

The cohort notebook uses exactly:

- `primary_speech / primary`
- `strict_speech / primary`
- `strict_internal_nonspeech / primary`

The strict speech and strict pause views are already guarded. No second guard is applied.

The cohort stage also corrects two legacy issues:

- speech–pause contrast is recomputed from canonical strict speech without duplicate erosion;
- the hum winner and its supported-harmonic companion are kept frequency-consistent.

The hum null is now calibrated against every observed eligible valid-window count, not a fixed eight-window reference or coarse support bins. Monte-Carlo P95 thresholds are made non-increasing with support using a conservative reverse cumulative maximum, which never lowers a simulated threshold.

## What the cohort notebook will produce

- corrected five-feature recording table for 519 frozen recordings;
- frame, interval, and spectral ledgers with frozen interval identities;
- media-hash and reconstruction audit;
- exact-support-count hum null calibration;
- migration comparison with legacy QADD v4.1;
- whole-pause deletion and boundary sensitivity;
- support, floor-mixture, availability, and precision summaries;
- empirical distributions and hum-like evidence prevalence;
- repeated-recording persistence;
- participant-balanced resampling;
- within-family redundancy;
- non-imputed ML-facing export;
- Panels D, E, F, H, and J;
- deterministic signal-linked galleries for Panel G;
- a candidate manifest that explicitly blocks freeze pending G10 review.

## Required next sequence

1. Install the cohort-stage patch.
2. Run and save the local cohort notebook.
3. Package the executed notebook and the complete `qadd-v4.2.0-candidate` output directory.
4. Perform the post-cohort scientific audit.
5. Complete the family workbook and ten-domain checklist with empirical evidence.
6. Make final feature-specific G10 decisions.
7. Issue a semantic/governance finalization patch if needed.
8. Freeze the numerical family.
9. Seal the standardized figure supplement.

No feature should be accepted, rejected, or assigned an operational threshold before the post-cohort audit.
