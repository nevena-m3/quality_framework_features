# QCHAN Stage 0 integrated audit — v1

## Decision

The legacy `qchan-v3.0.1` implementation is a technically useful estimator foundation, but it is **not accepted as the final family under the standardized QGAIN/QADD/QREV evaluation framework**. QCHAN will proceed as a reviewed `qchan-v4.0.0` candidate with the same four estimands, subject to a fresh G1–G10 review, standardized A–J figures, and separate immutable measurement and figure freezes.

No legacy numerical freeze is overwritten. No cohort values are accepted or centrally published by this preflight.

## Construct and claim boundary

QCHAN measures reference-relative manifestations of speech-spectrum coloration and upper-band attenuation. It does not identify a microphone, device, browser, codec, platform, or transfer function. The observed spectrum also depends on anatomy, sex/age, phonetics, articulation, dysarthria, breathiness, noise and task execution.

The required reference is task-matched and leave-one-subject-out. Recording spectra are aggregated within each reference subject before the across-subject median. There is no cross-task or global fallback. Reference membership, parameters, support and SHA-256 vintage are part of feature identity.

## Four reviewed candidates

1. `qchan_ltas_distance_db`: primary nonordinal magnitude of gain-anchored LTAS deviation over 100–7500 Hz.
2. `qchan_rolloff95_deficit_hz`: primary one-sided reduction in 95%-power rolloff relative to reference.
3. `qchan_highband_ratio_deficit`: secondary one-sided reduction in 3–7.5-kHz power share.
4. `qchan_tilt_steepening_db_per_oct`: secondary one-sided spectral-tilt steepening; explicitly phenotype-sensitive.

## Stage 0 findings

The legacy design already contains several strong elements: deterministic 16-kHz analysis, strict-speech support, task-matched subject-balanced LOSO references, signed precursor values, source-bandwidth metadata, no scalar, and reference robustness analyses. However, the legacy notebook and figures were produced before the current 50-item checklist and standardized artifact-bundle contract. Previous notebook-generation/execution-state test failures also show that a fresh clean source/executed separation is required.

The principal scientific risks are:

- reference dependence and vintage drift;
- absolute LTAS-distance dependence on the logarithmic spectral floor;
- one-sided zero masses that are measured truncation, not absence of spectral deviation;
- native source-bandwidth limitation masquerading as channel attenuation;
- high-frequency noise masking rolloff and high-band deficits;
- phenotype/phonetic confounding that cannot be removed from single-channel no-reference speech;
- task/reference support missingness that must not trigger a fallback.

## Required validation sequence

The reviewed candidate follows the same sequence used for the completed families:

- G1 contract/provenance;
- G2 numerical correctness and reconstruction;
- G3 transformations, source-rate and codec behavior;
- G4 controlled low-pass/shelf/notch response;
- G5 competing mechanisms and non-identifiability;
- G6 target support, reference support and sensitivity;
- G7 empirical cohort plausibility;
- G8 repeated recordings, redundancy and reference robustness;
- G9 explicit N/A because no event detector is retained;
- G10 feature decisions, review, final figures and immutable freezes.

## Preflight result

The reviewed module passed 16/16 tests. The independently executed preflight notebook completed all seven code cells without saved errors. Every blocking G1–G6 preflight check passed. Panels A–C were generated with complete PNG, SVG, PDF, source-data, caption and provenance bundles.

G5 is intentionally conditional: source identity remains non-identifiable. G7, G8 and G10 remain pending until the corrected 519-recording cohort is run and audited.
