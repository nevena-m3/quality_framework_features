# QDIST v3.1 — Scientific Measurement Specification

## 1. Scope

QDIST measures **native-waveform evidence compatible with hard clipping or saturation**. It does not measure nonlinear distortion comprehensively and does not estimate total harmonic distortion, intermodulation distortion, soft clipping, dynamic-range compression, limiting, automatic gain control, codec distortion, or perceptual distortion.

The operational family subtitle is:

> Hard-clipping and saturation evidence on the native decoded waveform.

The detector is a study-specific, literature-informed engineering estimator. The three retained recording-level features are reconstructable views of one accepted clipping-event system, not independent measurements of separate nonlinear mechanisms.

## 2. Scientific measurement question

What proportion of the native task-span waveform contains intervals whose amplitude distribution and local time-domain morphology are jointly compatible with hard clipping or saturation?

High amplitude alone is not clipping. Repeated values alone are not clipping. Edge-histogram concentration alone is not clipping. Acceptance requires joint evidence.

## 3. Authoritative signal view

The canonical input is the first native decoded waveform before:

- resampling;
- mono averaging or channel reduction;
- amplitude normalization;
- filtering or DC removal;
- denoising;
- interpolation;
- codec re-encoding.

The detector preserves native sample rate and all channels. Frozen segmentation defines the continuous natural task span from first retained speech onset to last retained speech offset. Internal pauses are preserved. Speech intervals are not concatenated.

The extraction must save source path, source hash, decoded-waveform hash, container, codec, native sample rate, channel count, decoded sample representation, decoder version, finite sample count, task-span duration, and parameter hash.

## 4. Retained analysis features

### 4.1 `qdist_hard_clipped_frame_fraction`

Fraction of fixed-duration native task-span frames that intersect at least one accepted clipping plateau.

\[
Q_{frame} = \frac{N_{frames\ intersecting\ accepted\ clipping}}{N_{eligible\ frames}}.
\]

- Role: primary prevalence view
- Unit: fraction [0, 1]
- Orientation: higher means more time-localized clipping evidence
- Default frame: 30 ms, non-overlapping, anchored at task-span start
- Source: accepted plateau ledger

### 4.2 `qdist_hard_clip_event_rate_per_min`

Number of merged clipping episodes per minute of finite native task-span exposure.

\[
Q_{rate} = \frac{N_{episodes}}{T_{finite}/60}.
\]

- Role: primary event-frequency view
- Unit: events/min
- Orientation: higher means more distinct clipping episodes
- Source: episode ledger
- Must be accompanied by event count, analyzed minutes, merge rule, and exact Poisson interval

### 4.3 `qdist_hard_clipped_sample_fraction`

Fraction of finite eligible channel-samples covered by accepted clipping plateaus.

\[
Q_{sample} = \frac{N_{accepted\ clipped\ channel\mbox{-}samples}}{N_{finite\ eligible\ channel\mbox{-}samples}}.
\]

- Role: secondary burden view
- Unit: fraction [0, 1]
- Orientation: higher means greater clipping support
- Source: accepted plateau ledger

### 4.4 Diagnostic-only outputs

The following are retained only for audit and detector characterization:

- near-full-scale sample fraction;
- polarity-specific edge occupancy;
- edge-to-interior occupancy ratio;
- candidate limiting level;
- samples beyond the proposed limiting level;
- candidate plateau count;
- rejected-candidate counts and reasons;
- any-channel affected time fraction;
- affected-channel count.

They are not manuscript analysis features and are not exported into the canonical analysis feature table.

## 5. Detector architecture

### Stage A — native-view and representation contract

1. Decode the first audio stream preserving sample rate and channels.
2. Verify finite exposure and task-span mapping.
3. Determine whether exact integer PCM geometry is known or whether the stream is floating/lossy decoded.
4. Preserve code-step information where available.
5. Refuse canonical extraction when the native-view contract is unverifiable.

Statuses:

- `available_no_events`
- `available_events`
- `unavailable_decode_failure`
- `unavailable_no_finite_exposure`
- `unavailable_native_view_not_verified`
- `unavailable_preprocessed_source`
- `indeterminate_insufficient_support`

An available zero is never conflated with unavailable.

### Stage B — polarity-specific limiting-edge proposals

Estimate positive and negative candidate limiting levels independently. Candidate edge proposals combine:

- extreme-value support;
- repeated or near-repeated level support;
- edge-shell occupancy;
- neighboring interior-shell occupancy;
- support beyond the proposed limiting level;
- recording-relative and local-context amplitude.

No fixed ±0.98 or ±0.99 full-scale threshold defines clipping. Clipping may occur below digital full scale, and legitimate peaks may approach full scale without clipping.

### Stage C — plateau candidate generation

Generate contiguous polarity-specific candidate runs near each proposed limiting level.

For exact integer PCM, use quantization-aware code equality or narrow code neighborhoods. For floating/lossy decoded streams, use an adaptive tolerance tied to numerical precision, local scale, and candidate edge magnitude.

Candidate generation is permissive; acceptance is strict.

### Stage D — local morphology qualification

For every candidate, measure:

- start/end sample and time;
- duration and sample count;
- polarity;
- limiting level;
- unique-level count;
- within-plateau range;
- median absolute first difference;
- robust plateau slope;
- entry and exit derivatives;
- local pre/post amplitude;
- local peak amplitude;
- candidate-to-context magnitude ratio;
- neighboring extrema behavior.

### Stage E — edge-distribution and terminal-edge qualification

Edge evidence is evaluated within temporally contiguous components of same-level candidates rather than across the entire recording. This preserves sensitivity to transient clipping under changing gain while still requiring local terminality. For each candidate limiting level, quantify:

- edge-zone occupancy;
- adjacent interior-shell occupancy;
- edge-to-interior ratio;
- edge excess;
- support beyond the proposed limit;
- polarity-specific terminality.

A recurring amplitude level is not a clipping edge when meaningful waveform support continues beyond it.

### Stage F — conjunctive acceptance

A plateau is accepted only when all necessary conditions pass:

\[
A_i = E_{plateau,i}\land E_{magnitude,i}\land E_{edge,i}\land E_{terminal,i}\land E_{context,i}\land \neg E_{confound,i}.
\]

A weighted score cannot compensate for failure of a necessary condition. Every component pass/fail and rejection reason is stored.

### Stage G — episode construction

Accepted plateaus are merged into episodes using a frozen maximum gap and compatibility rule. The plateau ledger and episode ledger remain separate.

- Sample and frame burden reconstruct from accepted plateau intervals.
- Event rate reconstructs from merged episodes.

## 6. Mandatory confound protections

### 6.1 Quantization

Primary false-positive risk. Validate 8-, 12-equivalent, 16-, and 24-bit PCM, floating audio, requantization, and post-quantization gain. Low-amplitude repeated codes must not be accepted as clipping.

### 6.2 Natural speech extrema

Validate clean connected speech, sustained vowels, loud speech, breathy speech, dysarthric speech where available, plosives, and varied pitch. Specificity is prioritized over sensitivity to weak ambiguous clipping.

### 6.3 Tonal and square-like signals

Sinusoids and harmonic tones should remain negative. Ideal square waves are out-of-domain morphology controls and must not be interpreted automatically as acquisition clipping.

### 6.4 Normalization and post-clipping attenuation

Post-clipping attenuation can preserve morphology below full scale; therefore digital-headroom thresholds are invalid. Known preprocessing before inspection invalidates the canonical provenance contract or must be explicitly marked provenance-limited.

### 6.5 Lossy codecs and resampling

Codec encoding and resampling may smear, ring, shorten, or fragment plateaus. Native decoded audio remains authoritative. Transformed variants are characterization-only. Absence of detection does not prove absence of upstream clipping.

### 6.6 Soft clipping and compression

Soft clipping, limiting, and compression are sensitivity/scope controls, not required positives. Weak or absent response is acceptable and documents the family boundary.

## 7. Minimum support and exposure

QDIST is a rare-event detector. Minimum support is required primarily for interpretable exposure-normalized rate and reliable edge-distribution estimation, not because zero events require positive support beyond the task span.

Proposed initial support contract, subject to validation:

- finite task-span duration ≥ 3 s;
- finite sample fraction = 1.000 for canonical extraction; any non-finite decoded sample makes the recording indeterminate;
- at least 100 complete 30-ms frames for frame fraction;
- adequate polarity-specific edge support before a candidate edge can be accepted;
- explicit low/standard/high exposure tier based on task-span duration, not labeled “robust.”

## 8. Ledgers

### 8.1 Candidate plateau ledger

One row per candidate containing identity, native indices, channel, polarity, limiting level, morphology components, edge-distribution components, context components, all acceptance flags, accepted state, rejection reason, estimator version, and parameter hash.

### 8.2 Accepted plateau ledger

One row per accepted plateau retaining all candidate evidence.

### 8.3 Episode ledger

One row per merged episode containing constituent plateau IDs, interval, duration, channels, polarity composition, merge gaps, affected samples, affected frames, and merge-rule version.

### 8.4 Recording table

One row per eligible recording containing the three analysis features, exact event count, plateau count, exposure, finite support, frame count, channel count, availability, status, support tier, confidence interval, decode provenance, and parameter hash.

## 9. Analytical validation program

### G1 — contract and provenance

- immutable feature registry;
- native-view verification;
- source/decode hashes;
- no hidden transformation;
- estimator and parameter versions;
- claim boundaries.

### G2 — deterministic formula and reconstruction

Hand-computable arrays test positive, negative, asymmetric, separated and merged events, frame crossings, multichannel cases, finite/nonfinite exposure, exact zeros, empty support, and partial final frames. All recording features reconstruct exactly from saved ledgers.

### G3 — transformation behavior

Predeclare and test polarity inversion, time shift, channel permutation, channel duplication, pre/post clipping gain, DC offset, normalization, resampling, filtering, and codec round-trip. Invariance is required only where theoretically justified.

### G4 — synthetic construct recovery

Use realistic speech-like carriers and vary clipping threshold, severity, polarity, asymmetry, duration, count, spacing, sample rate, bit depth, channel count, local signal level, pre/post gain, and codec placement.

Metrics:

- plateau precision/recall;
- episode precision/recall;
- sample-level precision/recall;
- onset/offset error;
- sample-fraction error;
- frame-fraction error;
- event-rate error;
- monotonic dose ordering.

### G5 — discriminant validity

Controls include clean connected speech, sustained vowels, music, tones, harmonic complexes, square-like signals, plosives, impulses, digital silence, coarse quantization, low-amplitude repeated codes, soft clipping, compression, limiting, clean codec-processed audio, and clean resampled audio.

### G6 — support and parameter robustness

Perturb minimum plateau samples, singleton rule, edge-zone width, edge support, edge/interior ratio, support-beyond-edge allowance, local context, magnitude threshold, merge gap, frame duration, sample rate, and bit depth. Retained defaults must lie in a stable operating region.

### G7 — empirical plausibility

Report availability, valid-zero prevalence, positive prevalence, event counts, durations, limiting levels, polarity, affected channels, format/codec/sample-rate strata, rejection reasons, and event concentration. Accepted limiting levels near zero are a blocking warning.

### G8 — redundancy and reconstructability

Frame fraction, event rate, and sample fraction are one detector’s views. Quantify mathematical and empirical relationships, merge-rule dependence, and incremental information. A feature may be dropped even if the detector passes.

### G9 — blinded adjudication

Review a fixed stratified sample of accepted events, rejected candidates, no-event controls, quantized cases, codec-smeared cases, polarity types, durations, sample rates, and burden levels. Labels: definite, probable, ambiguous, not clipping, cannot determine.

### G10 — freeze and integration

Freeze only after blocking gates pass, tests pass, event reconstruction is exact, review is complete, hashes are written, notebook and package use one implementation, registry/CLI/dataset assembly are updated, and only approved features reach the canonical export.

## 10. Rare-event reporting

Low prevalence is acceptable. Thresholds must not be loosened to create a continuous distribution.

Always report:

- available recordings;
- valid-zero recordings;
- positive recordings;
- positive participants;
- exact event counts and exposure;
- nonzero burden distribution;
- concentration of events;
- whether reliability/modeling is estimable.

Do not force ICC or ordinary Gaussian summaries when variation is insufficient.

## 11. Notebook structure

0. Environment, versions, run controls, output contract  
1. Immutable feature registry, equations, claim limits, legacy crosswalk  
2. Frozen-input and native-waveform provenance contract  
3. Hand-computable formula, interval, transform, and reconstruction tests  
4. Synthetic hard-clipping dose and sample-accurate recovery  
5. Quantization, speech, tone, music, impulse, soft-clip, and compression controls  
6. Native-versus-resampled/codec/normalized signal-chain characterization  
7. Parameter calibration and stable-region selection  
8. Frozen cohort loading and source-media audit  
9. Full cohort extraction with candidate, plateau, episode, status, and error ledgers  
10. Exact independent reconstruction of all three features  
11. Exposure, availability, valid-zero, and sparse-event characterization  
12. Parameter, format, sample-rate, channel, and merge-rule robustness  
13. Within-family redundancy and cross-family arbitration diagnostics  
14. Label-blind event gallery and adjudication export  
15. G1–G10 validation matrix, feature-specific disposition, and freeze decision  
16. Immutable central export and negative-result/scope audit

## 12. Legacy disposition

| Legacy item | Decision | Reason |
|---|---|---|
| `_hard_clip_mask` based on 0.995 × observed max and local flatness | Replace | A recording maximum is not necessarily a clipping limit; lacks edge terminality, quantization controls, local morphology ledger, and provenance contract |
| strict-speech interval concatenation | Replace | Destroys continuous native context and creates artificial interval joins |
| maximum across channels | Replace | Conceals multichannel burden and exposure; use explicit channel-sample and any-channel summaries |
| `qdist_hard_clip_sample_fraction` | Redesign/rename | Retain construct with event-ledger reconstruction and channel-aware denominator |
| `qdist_clip_event_rate_per_min` | Redesign/rename | Retain only after episode merge-rule robustness |
| `qdist_clipped_frame_fraction` | Redesign/rename | Retain with task-span anchored native frames |
| `qdist_near_fullscale_fraction` | Diagnostic only | High headroom use is not clipping |
| `qdist_edge_histogram_spike` | Diagnostic component only | Histogram concentration alone is insufficient evidence |
| one clean-vs-clipped sine test | Replace with full validation suite | Demonstrates only an easy positive control and no false-positive or recovery performance |

## 13. Freeze rule

Each candidate feature receives `PASS_RETAIN`, `REVISE`, `DROP`, or `DIAGNOSTIC_ONLY`. The family can freeze with fewer than three features. Detector validity is not inferred from prevalence, diagnosis association, human-QC association, or downstream prediction.
