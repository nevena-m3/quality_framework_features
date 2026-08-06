# QDIST v3.1.1 — scientific measurement specification

## Construct and claim boundary

**Construct:** observable native-waveform evidence compatible with hard clipping or saturation.

**Not measured:** total nonlinear distortion, THD, intermodulation distortion, reliable soft-clipping severity, compressor/limiter activity, AGC, general codec distortion, or perceptual quality.

**Authoritative input:** first native-rate decoded audio stream, all channels preserved, before resampling, mono reduction, normalization, filtering, denoising, interpolation, DC removal, or codec re-encoding. Frozen segmentation defines only the continuous outer task span; internal pauses remain intact.

## Analysis features

### `qdist_hard_clipped_frame_fraction`

Fraction of complete non-overlapping native task-span frames intersecting at least one accepted clipping plateau on any channel.

### `qdist_hard_clip_event_rate_per_min`

Number of merged accepted clipping episodes per minute of finite task-span exposure. Event count, exposure, confidence interval, and merge rule accompany the rate.

### `qdist_hard_clipped_sample_fraction`

Fraction of finite eligible native channel-samples covered by accepted clipping plateaus.

The three features are related aggregations of one ledger system, not independent evidence for separate nonlinear mechanisms.

## Default candidate parameters

Key defaults include:

- 30-ms frame length;
- minimum 3-s task-span support and 100 complete frames;
- minimum plateau support of 4 samples;
- maximum plateau duration of 10 ms;
- recording-relative edge floor of 0.45 × channel robust peak;
- local edge ratio of at least 0.90 × local context peak;
- minimum edge-zone and level-cluster support of 8 samples;
- edge-to-interior ratio at least 2.0;
- maximum 20-ms gap for merging plateaus into an episode.

All values are engineering parameters and remain identified by the complete parameter hash. The notebook tests reasonable neighborhoods rather than treating defaults as universal constants.

## Ledgers

The extractor returns:

1. candidate plateau ledger, including every materialized candidate, component evidence, acceptance flags, and rejection reason;
2. accepted plateau ledger;
3. merged episode ledger;
4. polarity/channel edge evidence ledger;
5. recording-level support, status, diagnostic, provenance, and feature fields.

All three analysis outputs must reconstruct exactly from accepted plateau and episode ledgers.

## Missingness and sparse-event contract

The detector distinguishes:

- `available_no_events`;
- `available_events`;
- `indeterminate_insufficient_support`;
- `indeterminate_nonfinite_support`;
- `unavailable_native_view_not_verified`;
- `unavailable_preprocessed_source`;
- decode or input failure states.

Available recordings with no accepted event receive valid zeros. Indeterminate or unavailable measurements remain missing.

## Blocking validation requirements

- **G1:** registry, version, parameter hash, source/decode identity, native-view contract.
- **G2:** hand-computable formulas and exact ledger reconstruction.
- **G3:** polarity and post-clipping-gain invariance; transformed-view loss characterized rather than hidden.
- **G4:** nonzero known truth in every hard-clip dose cell, detection in every cell, sample precision ≥0.99, recall ≥0.85, F1 ≥0.90, monotonic burden, and Spearman burden recovery ≥0.85.
- **G5:** clean speech-like signals, tones, impulse, quantization depths, soft clipping, and square-like ambiguity do not create unacceptable positive events.
- **G6:** synthetic and empirical parameter-neighborhood stability; sparse sample-burden robustness evaluated in clipped milliseconds per analyzed minute.
- **G7:** complete cohort extraction, explicit statuses, valid-zero/missingness integrity, sparse-event morphology, and runtime audit.
- **G8:** exact reconstruction, explicit near-transform relationship, and merge-gap stability isolated from detector-threshold sensitivity.
- **G9:** label-blind review of every accepted plateau plus fixed rejected-candidate and valid-zero controls, with accepted-event precision-like evidence ≥0.90.
- **G10:** package and notebook-governance tests and immutable freeze manifest.

Freeze is prohibited until every retained feature has an explicit pass/revise/drop decision and every blocking gate passes.


## Final two-tier amplitude rule

Candidate generation uses a permissive recording-relative floor of 0.25. Final
acceptance requires either (a) a strong recording-edge ratio of at least 0.45,
or (b) a lower-level repeated-saturation path with at least four clustered
plateau candidates, 24 plateau samples, and 24 edge-zone samples. Both paths
also require local prominence and all morphology, context, transition,
terminality, quantization, and square-wave guards. This rule preserves
sensitivity to severe clipping in a lower gain state while rejecting isolated
quantized natural extrema.
