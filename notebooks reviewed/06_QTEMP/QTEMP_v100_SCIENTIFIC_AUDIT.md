# QTEMP v1.0.0 candidate — scientific and structural audit

## Bottom line

QTEMP is scientifically valuable **only as a narrow, event-ledger description of observable decoded-stream manifestations**. It is not an established clinical acoustic biomarker, a packet-loss estimator, or a causal transport-failure classifier. The defensible retained candidate has four features from two detectors:

1. bracketed dropout-like duration fraction;
2. bracketed dropout-like event rate per minute;
3. near-exact consecutive decoded-repetition duration fraction;
4. near-exact consecutive decoded-repetition event rate per minute.

The two rates are secondary same-ledger views, not independent evidence. The splice-discontinuity rate is dropped after failed held-out recovery, implausible cohort positivity (318/519), and sample-rate sensitivity.

QTEMP is **not publication-frozen**. The prior executed review reported only 3/6 accepted dropout excerpts as clearly observable (0.50 versus a prespecified 0.80 threshold). G9 therefore fails/remains unresolved and G10 is blocked. The later appended “G9-deferred internal freeze” is preserved as historical output but is not accepted as a scientific or publication freeze.

## Scientific maturity and references

All four retained quantities are study-specific event-detector summaries. Diener et al. (INTERSPEECH 2022 Audio Deep Packet Loss Concealment Challenge, doi:10.21437/Interspeech.2022-616) supports the distinction between packet-loss gaps and concealment mechanisms; it does not validate these detectors or thresholds. Goldsack et al. (npj Digital Medicine 2020;3:55, doi:10.1038/s41746-020-0260-4) supports the verification/analytical-validation/clinical-validation separation; it does not validate the QTEMP estimator. The implementation must therefore stand on controlled perturbation, specificity, robustness, and blinded event verification evidence.

## Construct and claim boundary

The family label “temporal discontinuity” is acceptable if manuscript wording remains observational:

- permitted: “bracketed dropout-like decoded support” and “near-exact consecutive decoded-waveform repetition”;
- prohibited: “packet-loss fraction,” “network failure,” “buffering event,” or “all freezes” without transport metadata and broader detector validation;
- required input: native decoded channels before resampling, mono conversion, normalization, filtering, interpolation, denoising, or codec re-encoding;
- required denominator: eligible frozen task-stream duration after symmetric edge guards;
- required missingness: unavailable remains `NaN`; zero means a valid analyzed stream with no accepted event.

## Defects found and remediated in the reviewed candidate

### 1. Export-contract mismatch

The v0.3 core still exports five features and permits splice detection, while finalization retains four. The reviewed candidate exposes exactly four features and refuses `splice` as an enabled event type.

### 2. Final frozen-audio scope was post-hoc

The v0.3 core detects repetition from 18 ms and the notebook later filters to 40 ms. The reviewed candidate enforces the final scope during extraction. This prevents downstream consumption of the development contract by mistake.

### 3. Frame/hop boundary bias at 40 ms

A parameter named `40 ms` did not reliably recover a 40 ms injected event. With 4 ms frames and 2 ms hops, non-grid alignment can produce about 38 ms of evidenced support; at 44.1 kHz, native-sample rounding produced 37.914 ms. The candidate therefore defines:

- scientific truth scope: injected/repeated target at least 40 ms;
- inclusion tolerance: 2.5 ms for frame/hop/native-sample boundary localization;
- minimum directly evidenced support: 37.5 ms;
- reported ledger duration: the actual evidenced support, never padded to 40 ms.

Regression tests now cover 8, 16, 24, 44.1, and 48 kHz at grid and non-grid start times.

### 4. Exact minimum-exposure floating-point failure

A 1.2 s recording with two 100 ms guards should yield exactly 1.0 s exposure, but binary floating-point produced a value just below 1.0 and incorrectly marked it unavailable. The candidate uses a 1 ns numerical comparison tolerance without changing the scientific 1.0 s rule.

### 5. Freeze-state contradiction

The executed-review gate table states G9 PENDING and G10 BLOCKED, but a later appended cell creates a canonical internal freeze with a G9 override. The reviewed notebook prohibits publication freeze and records the override only as historical provenance.

## Evidence interpretation

- Synthetic dropout recovery is strong for exact-zero and constant-low-information runs under the tested active-context conditions.
- Attenuated active audio is correctly not relabeled as missing support in the tested control.
- The narrow repetition detector rejects tested tones, harmonic vowel proxies, and connected-speech proxies, but real ALS dysarthric speech and sustained-phonation hard negatives remain necessary because the detector is high-risk on periodic signals.
- Prior cohort evidence reports 2/519 dropout-positive recordings and 0/519 frozen-audio-positive recordings. Zero variation is scientifically meaningful as measured absence, but it prevents continuous multivariable use and leaves positive predictive behavior of the frozen detector unverified in this cohort.
- The dropout fraction and rate describe different aspects of the same ledger (burden versus frequency). They must not be treated as independent biological dimensions.

## Current gate decision

| Gate | State | Interpretation |
|---|---|---|
| G1 | PASS | Four-feature observational contract and native ordering are explicit. |
| G2 | PASS | Relevant legacy/current/candidate suite passes; exact reconstruction and missing-versus-zero pass. |
| G3 | CONDITIONAL | Candidate gain/polarity/sample-rate preflight passes; full prior signal-chain artifacts were not uploaded. |
| G4 | PASS | Controlled target recovery is supported within the narrow registered scopes. |
| G5 | CONDITIONAL | Synthetic controls pass; real phenotype/periodic hard negatives are incomplete here. |
| G6 | PASS | Exposure, boundary tolerance, and parameter sensitivity are explicit. |
| G7 | CONDITIONAL | Embedded cohort summary exists; raw MAIN outputs were not uploaded. |
| G8 | CONDITIONAL | Ledger reconstruction passes; participant repeat/persistence needs raw outputs. |
| G9 | FAIL / unresolved | Prior accepted-event observable-yes fraction is 0.50, below 0.80. |
| G10 | BLOCKED | No publication freeze until a genuine held-out blinded G9 passes. |

## Required next execution

1. Supply the frozen MAIN output tables, event ledgers, native review audio, gallery images, and their hashes.
2. Re-run the reviewed four-feature candidate on all 519 recordings and reconcile every difference with the v0.3.1 reconstructed table.
3. Diagnose the three uncertain accepted dropout events without changing labels post hoc.
4. If the detector is revised, separate development examples from a new held-out blinded adjudication set.
5. Include all accepted retained events, fixed candidate-free excerpts, and hard negatives; keep reviewers blind to identity, detector disposition, and scores.
6. Regenerate full Panels F–I, including signal-linked examples and participant-aware recurrence/persistence.
7. Freeze only after G1–G9 pass and manuscript feature census/wording matches the four-feature registry.
