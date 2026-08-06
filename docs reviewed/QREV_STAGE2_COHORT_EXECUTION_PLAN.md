# QREV v4.0.0 — Stage 2 Cohort Extraction and Empirical-Validation Plan

## Authorization decision

The corrected `dc-contract-hotfix-1` analytical preflight is accepted. The executed notebook contains 9/9 executed code cells, no saved errors, all preflight checks pass, the pinned SRMR runtime is available, global DC removal is enforced, and Panels A-C have complete PNG/SVG/PDF/source/caption/provenance bundles.

QREV may proceed to corrected cohort extraction. It is not eligible for freeze. G6 cohort sensitivity, G7 empirical plausibility, G8 reliability/redundancy, final feature decisions, and immutable freezes remain pending.

## Common evaluation architecture

QREV follows the same ten-domain and G1-G10 structure used for QGAIN and QADD. Family-specific evidence is adapted to reverberation/residual-tail measurements:

- natural `primary_speech / primary` offsets for all boundary estimators;
- `strict_speech / primary` only for SRMR speech-support duration;
- explicit bounded persistence and right-censoring;
- support-policy comparison at 2, 3, and 4 valid boundaries;
- no imputation, no family scalar, and no standalone reject threshold;
- no RT60, EDT, DRR, RIR, room-identity, or confirmed-echo claim.

## Cohort notebook outputs

The reviewed cohort notebook will process all 519 eligible recordings and 224 participants with verified frozen media hashes. It will generate:

1. Corrected recording-level values for the four QREV measurements.
2. Boundary ledger with frozen primary interval identities, exclusion reasons, floor evidence, eligibility, and censoring.
3. Exact reconstruction audit for all three boundary-conditioned recording estimates.
4. Raw and analysis values under 2-, 3-, and 4-boundary support policies.
5. Whole-boundary deletion and deterministic bootstrap precision evidence.
6. Boundary-offset perturbations at -100, -50, +50, and +100 ms.
7. Independent floor-window, persistence-horizon, threshold, consecutive-frame, frame-length, early-window, and decay-window sensitivity.
8. SRMR bandwidth characterization, supplementing preflight codec characterization.
9. Empirical availability, statuses, missingness reasons, mathematical ranges, and boundary/recording censoring.
10. Participant-aware first-two-recording Spearman, ICC(1), absolute differences, and an uncensored persistence subset.
11. One-recording-per-participant resampling and recording-weighted comparison.
12. Pairwise redundancy/convergent-evidence analysis with pairwise support.
13. Non-imputed ML interface carrying value, availability, status, missing reason, support tier, censoring, version, and provenance.

## Standardized figures

The package preserves completed Panels A-C and generates:

- D1 support-policy availability;
- D2 status, missingness, and censoring;
- D3 support-precision relationships: bootstrap CI width versus eligible pause support and SRMR availability versus strict-speech duration;
- E1 delete-one-boundary sensitivity;
- E2 boundary/parameter availability sensitivity;
- E3 paired value sensitivity normalized by feature-specific empirical IQR for visualization only;
- F empirical distributions with feature-specific axes and explicit censoring interpretation;
- G at least eight deterministic label-blind signal-linked examples, each with aligned waveform, spectrogram, and AC-RMS envelope source data and provenance, selected without causal source labels;
- H1 repeated-recording persistence;
- H2 redundancy and SRMR convergence with pairwise n;
- H3 participant-versus-recording weighting;
- J quality-aware ML interface completeness;
- I explicit N/A because no discrete event detector is retained.

Each figure or gallery example has PNG, SVG, PDF, source CSV, scientific caption, and provenance JSON.

## Gate status entering cohort extraction

| Gate | Status |
|---|---|
| G1 | PASS |
| G2 | PASS |
| G3 | PASS |
| G4 | PASS |
| G5 | CONDITIONAL PASS; cohort bandwidth characterization pending |
| G6 | PREFLIGHT PASS; cohort robustness and support-policy evidence pending |
| G7 | PENDING COHORT |
| G8 | PENDING COHORT |
| G9 | N/A |
| G10 | PENDING POST-COHORT SCIENTIFIC REVIEW |

## Freeze rule

The cohort notebook is structurally unable to freeze QREV. It writes `freeze_allowed: false` and leaves every G10 feature decision pending. After the run is independently audited, a separate finalization patch may retain, revise, demote, or drop each feature and only then authorize atomic measurement and figure freezes.
